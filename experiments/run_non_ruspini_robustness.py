"""Non-Ruspini membership robustness across PQ16, CWRU and CARE (R3.3c)."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import RESULTS_DIR
from data import load_splits
from framework.pipeline import CLLLMOpsPipeline
from framework.verification.fuzzy_gating import MamdaniFIS
from parallel_eval import get_workers, iter_job_pool, reseed_and_build


METRICS = (
    "accuracy", "f1", "hallucination_rate", "action_accuracy",
    "false_escalation_rate", "proceed_rate", "defer_rate",
    "escalation_rate",
)


def partition_diagnostics(gate):
    grid = np.linspace(0.0, 1.0, 1001)
    sums = np.vstack([
        gate.score_mf.low(grid), gate.score_mf.medium(grid),
        gate.score_mf.high(grid),
    ]).sum(axis=0)
    return {
        "mean_abs_sum_minus_one": float(np.mean(np.abs(sums - 1.0))),
        "max_abs_sum_minus_one": float(np.max(np.abs(sums - 1.0))),
        "ambiguity_fraction_sum_gt_one": float(np.mean(sums > 1.0 + 1e-9)),
        "gap_fraction_sum_lt_one": float(np.mean(sums < 1.0 - 1e-9)),
    }


def summarize(per_seed, shifts):
    summary = {}
    for shift in shifts:
        name = f"shift_{shift:+.2f}"
        summary[name] = {}
        for metric in METRICS:
            vals = [row["metrics"][name].get(metric) for row in per_seed]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                summary[name][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "raw": vals,
                }
    base = summary["shift_+0.00"]
    max_delta = {}
    for metric in METRICS:
        if metric not in base:
            continue
        ref = base[metric]["mean"]
        max_delta[metric] = float(max(
            abs(values[metric]["mean"] - ref)
            for values in summary.values() if metric in values))
    return summary, max_delta


def _seed_job(dataset, splits, seed, shifts):
    """One (dataset, seed): shared generations re-gated under all shifts.

    The pipeline and the per-shift FIS gates are job-local, freshly
    seeded instances, so the job is independent of other
    (dataset, seed) jobs and reproduces the sequential outputs.
    """
    pipeline = reseed_and_build(
        seed, lambda: CLLLMOpsPipeline(dataset_name=dataset, seed=seed))
    op = pipeline._load_operating_params()
    gates = {
        f"shift_{shift:+.2f}": MamdaniFIS(
            alpha_p=op.get("alpha_p", 0.7),
            alpha_d=op.get("alpha_d", 0.3),
            delta=op.get("delta", 0.1),
            lambda_r=op.get("lambda_r", 0.4),
            breakpoint_shift=shift)
        for shift in shifts
    }
    metrics = pipeline.evaluate_gate_family(
        splits["X_test"], splits["y_test"], gates=gates, seed=seed)
    return {"seed": seed, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--shifts", nargs="+", type=float,
                        default=[-0.10, -0.05, 0.0, 0.05, 0.10])
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent (dataset x seed) jobs (default env "
                             "ASOC_EVAL_WORKERS or 16). Each job builds a "
                             "freshly-seeded pipeline instance, so results "
                             "do not depend on scheduling.")
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "non_ruspini_robustness.json")
    args = parser.parse_args()

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_item": "R3.3c",
            "protocol": "shared test generations; simultaneous shifts of "
                        "the six score/coverage breakpoints",
            "datasets": args.datasets,
            "seeds": args.seeds,
            "shifts": args.shifts,
        },
        "partition_diagnostics": {},
        "datasets": {},
    }
    for shift in args.shifts:
        gate = MamdaniFIS(breakpoint_shift=shift)
        payload["partition_diagnostics"][f"shift_{shift:+.2f}"] = \
            partition_diagnostics(gate)

    # Independent (dataset x seed) jobs on a thread pool; per-dataset
    # rows are re-assembled in seed order, identical to the sequential
    # loop.
    splits_by_dataset = {ds: load_splits(ds) for ds in args.datasets}
    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            jobs.append(((dataset, seed),
                         lambda ds=dataset, s=seed: _seed_job(
                             ds, splits_by_dataset[ds], s, args.shifts)))

    print(f"{len(jobs)} robustness jobs (workers={args.workers})")
    rows_by_key = {}
    for done, (key, row) in enumerate(
            iter_job_pool(jobs, args.workers), start=1):
        rows_by_key[key] = row
        print(f"[{done}/{len(jobs)}] {key[0]} seed={key[1]} "
              f"shifts={args.shifts}", flush=True)

    for dataset in args.datasets:
        per_seed = [rows_by_key[(dataset, seed)] for seed in args.seeds]
        summary, max_delta = summarize(per_seed, args.shifts)
        payload["datasets"][dataset] = {
            "per_seed": per_seed,
            "summary": summary,
            "max_absolute_delta_from_unshifted": max_delta,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
