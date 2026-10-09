"""Independent hallucination measurement and human-audit packet (R3.2a).

Runs the deterministic evidence judge and an off-the-shelf NLI judge over
identical pipeline episodes.  It never substitutes automated agreement for
human agreement: the output records automated judge agreement separately and
marks human Cohen's kappa pending until a completed two-annotator JSONL file
is supplied with ``--annotations``.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from config import RESULTS_DIR
from data import load_splits
from evaluation.metrics import NLIJudge, RuleBasedJudge
from framework.pipeline import CLLLMOpsPipeline
from parallel_eval import get_workers, iter_job_pool, reseed_and_build


class _LockedJudge:
    """Serialize calls into the shared NLI judge.

    One transformers cross-encoder is loaded once (as in the sequential
    runner) and shared across jobs; its HF pipeline __call__ is not
    thread-safe, while judging itself is stateless, so a lock preserves
    exact results with episodes otherwise running concurrently.
    """

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def judge(self, *args, **kwargs):
        with self._lock:
            return self._inner.judge(*args, **kwargs)


def agreement(rows):
    if not rows:
        return {"n": 0, "status": "pending", "reason": "no audit rows"}
    a = np.array([r["judge_outputs"]["rule"]["hallucinated"]
                  for r in rows], dtype=int)
    b = np.array([r["judge_outputs"]["nli"]["hallucinated"]
                  for r in rows], dtype=int)
    kappa = cohen_kappa_score(a, b)
    return {
        "n": int(len(rows)),
        "raw_agreement": float(np.mean(a == b)),
        "automated_judge_kappa": (float(kappa)
                                    if np.isfinite(kappa) else None),
        "rule_positive_rate": float(a.mean()),
        "nli_positive_rate": float(b.mean()),
        "confusion_rule_rows_nli_cols": [
            [int(np.sum((a == i) & (b == j))) for j in (0, 1)]
            for i in (0, 1)
        ],
        "note": "Automated-judge agreement; not human inter-annotator "
                "agreement.",
    }


def choose_packet(rows, size, seed=20260721):
    rng = np.random.default_rng(seed)
    disagreements = [r for r in rows if
                     r["judge_outputs"]["rule"]["hallucinated"] !=
                     r["judge_outputs"]["nli"]["hallucinated"]]
    agreed = [r for r in rows if r not in disagreements]
    rng.shuffle(disagreements)
    rng.shuffle(agreed)
    # At most half disagreement cases prevents an adversarially enriched
    # packet from masquerading as a prevalence sample.
    chosen_disagreements = disagreements[:size // 2]
    chosen = list(chosen_disagreements)
    chosen += agreed[:max(0, size - len(chosen))]
    if len(chosen) < size:
        start = len(chosen_disagreements)
        chosen += disagreements[start:start + size - len(chosen)]
    packet = []
    for i, row in enumerate(chosen[:size]):
        packet.append({
            "audit_id": f"HJ-{i:04d}",
            "dataset": row["dataset"],
            "hypothesis_text": row["hypothesis_text"],
            "retrieved_evidence": row["evidence_docs"],
            "released_to_operator": row["decision"] == "proceed",
            "annotation_schema": {
                "hallucinated": "yes|no",
                "contradiction": "yes|no",
                "notes": "free text",
            },
            "annotator_1": None,
            "annotator_2": None,
        })
    return packet


def human_kappa(path):
    if path is None:
        return {"status": "pending", "reason": "no annotations supplied"}
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    a, b = [], []
    for row in rows:
        one, two = row.get("annotator_1"), row.get("annotator_2")
        if not isinstance(one, dict) or not isinstance(two, dict):
            continue
        if one.get("hallucinated") not in {"yes", "no"} or \
                two.get("hallucinated") not in {"yes", "no"}:
            continue
        a.append(one["hallucinated"] == "yes")
        b.append(two["hallucinated"] == "yes")
    if len(a) < 2:
        return {"status": "pending", "reason": "fewer than two "
                "double-annotated rows", "n": len(a)}
    kappa = cohen_kappa_score(a, b)
    return {"status": "complete", "n": len(a),
            "cohen_kappa": float(kappa) if np.isfinite(kappa) else None,
            "raw_agreement": float(np.mean(np.asarray(a) == np.asarray(b)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--nli-model",
                        default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli-device", type=int, default=0)
    parser.add_argument("--packet-size", type=int, default=120)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent (dataset x seed) jobs (default env "
                             "ASOC_EVAL_WORKERS or 16). Each job builds a "
                             "freshly-seeded pipeline instance, so results "
                             "do not depend on scheduling.")
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "independent_judge.json")
    parser.add_argument("--packet-out", type=Path,
                        default=RESULTS_DIR / "human_audit_packet.jsonl")
    args = parser.parse_args()

    nli = _LockedJudge(NLIJudge(model_name=args.nli_model,
                                device=args.nli_device))

    # Independent (dataset x seed) jobs; audit rows are re-assembled in
    # the sequential loop's dataset-then-seed order so downstream packet
    # selection (seeded shuffle) is bit-identical to the sequential run.
    splits_by_dataset = {ds: load_splits(ds) for ds in args.datasets}
    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            jobs.append(((dataset, seed),
                         lambda ds=dataset, s=seed: reseed_and_build(
                             s, lambda: CLLLMOpsPipeline(
                                 dataset_name=ds, seed=s))
                         .evaluate_judge_family(
                             splits_by_dataset[ds]["X_test"],
                             splits_by_dataset[ds]["y_test"], seed=s,
                             judges={"rule": RuleBasedJudge(),
                                     "nli": nli})))

    print(f"{len(jobs)} judge jobs (workers={args.workers})")
    outputs = {}
    for done, (key, output) in enumerate(
            iter_job_pool(jobs, args.workers), start=1):
        outputs[key] = output
        print(f"[{done}/{len(jobs)}] {key[0]} seed={key[1]} done",
              flush=True)

    all_rows = []
    datasets = {}
    for dataset in args.datasets:
        per_seed = []
        for seed in args.seeds:
            output = outputs[(dataset, seed)]
            all_rows.extend(output["audit_rows"])
            per_seed.append({"seed": seed, "metrics": output["metrics"]})
        ds_rows = [r for r in all_rows if r["dataset"] == dataset]
        datasets[dataset] = {
            "per_seed": per_seed,
            "agreement": agreement(ds_rows),
        }

    packet = choose_packet(all_rows, min(args.packet_size, len(all_rows)))
    args.packet_out.parent.mkdir(parents=True, exist_ok=True)
    args.packet_out.write_text("".join(json.dumps(row) + "\n"
                                       for row in packet))
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_item": "R3.2a",
            "nli_model": args.nli_model,
            "evaluation_independence": "Neither judge calls CAEVN/FIS.",
        },
        "datasets": datasets,
        "overall_automated_agreement": agreement(all_rows),
        "human_audit": {
            "packet": str(args.packet_out),
            "packet_size": len(packet),
            "selection": "up to 50% judge disagreements, remainder random",
            "inter_annotator": human_kappa(args.annotations),
        },
        "audit_rows": all_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {args.out} and {args.packet_out}")


if __name__ == "__main__":
    main()
