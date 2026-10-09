"""R1 verification-layer and calibrated-gate comparison.

Covers reviewer items R3.2b/R3.3a (Platt + isotonic scalar gates) and R4.3
(off-the-shelf NLI verifier + self-consistency guardrail).  Every arm uses
the same train/validation/test split, backbone and independent rule judge.
Calibrated gates additionally share identical test generations with the FIS.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from config import RESULTS_DIR
from data import load_splits
from framework.pipeline import CLLLMOpsPipeline
from framework.verification.calibrated_gate import (
    CalibratedProbabilityGate,
)
from framework.verification.baseline_verifiers import (
    NLIEntailmentVerifier, SelfConsistencyGuardrailVerifier,
)
from parallel_eval import get_workers, iter_job_pool, reseed_and_build


class _LockedVerifier:
    """Serialize calls into a shared HF-backed verifier.

    The NLI verifier wraps a single transformers pipeline (loaded once,
    as in the sequential runner) whose __call__ is not thread-safe;
    inference itself is stateless, so a lock preserves exact results
    while episodes of independent jobs proceed concurrently.
    """

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def score(self, *args, **kwargs):
        with self._lock:
            return self._inner.score(*args, **kwargs)


CORE_METRICS = (
    "accuracy", "f1", "hallucination_rate", "action_accuracy",
    "false_escalation_rate", "proceed_rate", "defer_rate",
    "escalation_rate",
)


def aggregate(per_seed):
    summary = {}
    for method in per_seed[0]["metrics"]:
        summary[method] = {}
        for metric in CORE_METRICS:
            values = [row["metrics"][method].get(metric)
                      for row in per_seed]
            values = [v for v in values if isinstance(v, (int, float))]
            if values:
                summary[method][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "raw": values,
                }
    return summary


def _seed_job(dataset, splits, seed, calibration_max, nli):
    """One (dataset, seed) arm: calibration + paired gates + baselines.

    Every pipeline is a fresh instance built under reseed_and_build, so
    the job is independent of other (dataset, seed) jobs and produces
    exactly the sequential loop's outputs.
    """
    full = reseed_and_build(
        seed, lambda: CLLLMOpsPipeline(dataset_name=dataset, seed=seed))
    rows = full.collect_gate_calibration(
        splits["X_val"], splits["y_val"], seed=seed,
        max_episodes=calibration_max)
    scores = [row["v_score"] for row in rows]
    correct = [row["correct"] for row in rows]
    decisions = [row["reference_decision"] for row in rows]
    platt = CalibratedProbabilityGate("platt").fit(
        scores, correct, decisions)
    isotonic = CalibratedProbabilityGate("isotonic").fit(
        scores, correct, decisions)

    metrics = full.evaluate_gate_family(
        splits["X_test"], splits["y_test"], seed=seed,
        gates={"cl_llmops": None,
               "calibrated_platt": platt,
               "calibrated_isotonic": isotonic})

    guardrail = reseed_and_build(
        seed, lambda: CLLLMOpsPipeline(
            dataset_name=dataset, seed=seed,
            verifier_override=SelfConsistencyGuardrailVerifier()))
    metrics["guardrail"] = guardrail.evaluate(
        splits["X_test"], splits["y_test"], seed=seed)

    if nli is not None:
        nli_pipeline = reseed_and_build(
            seed, lambda: CLLLMOpsPipeline(
                dataset_name=dataset, seed=seed,
                verifier_override=nli))
        metrics["nli_verifier"] = nli_pipeline.evaluate(
            splits["X_test"], splits["y_test"], seed=seed)

    return {
        "seed": seed,
        "calibration": {
            "n": len(rows),
            "platt": platt.to_dict(),
            "isotonic": isotonic.to_dict(),
        },
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 1, 2])
    parser.add_argument("--calibration-max", type=int, default=400)
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent (dataset x seed) jobs (default env "
                             "ASOC_EVAL_WORKERS or 16). Each job builds "
                             "freshly-seeded pipeline instances, so results "
                             "do not depend on scheduling.")
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "verification_baselines.json")
    args = parser.parse_args()

    nli = None if args.skip_nli else _LockedVerifier(
        NLIEntailmentVerifier(model_name=args.nli_model))
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_items": ["R3.2b", "R3.3a", "R4.3"],
            "datasets": args.datasets,
            "seeds": args.seeds,
            "calibration_split": "validation only",
            "calibrated_gate_operating_point": (
                "validation proceed/reject coverage matched to native FIS"),
            "evaluation_judge": "independent deterministic rule judge",
        },
        "datasets": {},
    }

    # Splits load in the main thread; the (dataset x seed) jobs are
    # independent and run on a thread pool.
    splits_by_dataset = {ds: load_splits(ds) for ds in args.datasets}
    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            jobs.append(((dataset, seed),
                         lambda ds=dataset, s=seed: _seed_job(
                             ds, splits_by_dataset[ds], s,
                             args.calibration_max, nli)))

    print(f"{len(jobs)} verification jobs (workers={args.workers})")
    rows_by_key = {}
    for done, (key, row) in enumerate(
            iter_job_pool(jobs, args.workers), start=1):
        dataset, seed = key
        rows_by_key[key] = row
        print(f"[{done}/{len(jobs)}] {dataset} seed={seed} done",
              flush=True)

    for dataset in args.datasets:
        per_seed = [rows_by_key[(dataset, seed)] for seed in args.seeds]
        payload["datasets"][dataset] = {
            "per_seed": per_seed,
            "summary": aggregate(per_seed),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
