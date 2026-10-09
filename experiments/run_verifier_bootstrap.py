"""Verifier-training resampling for the headline comparison (R3-1).

Reviewer 3 asked whether the reported gains survive data-level resampling.
The statistical protocol resamples the evaluation set (10 stratified bootstrap
partitions) and the decoding stream (3 seeds), but every learned artifact stays
fixed, so it says nothing about how much the effect depends on which data the
verifier was trained on. Re-fitting the whole pipeline per fold would require
re-decoding the corpus each time; re-fitting the verifier does not, because the
CAEVN training pool is drawn independently of the evaluation episodes.

This runner evaluates the paired comparison (CL-LLMOps vs Agent-noV, shared
generations) once per verifier checkpoint. Each checkpoint comes from an
independent draw of the verifier's training pool: train_caevn.py --tag boot<b>
--seed <s> re-samples which training episodes are used, re-decodes them, and
retrains CAEVN from scratch. The deployed operating point is held fixed, so the
spread across folds isolates the verifier-training draw.

    python experiments/run_verifier_bootstrap.py --tags boot1 boot2 boot3 \
        --datasets pq16 cwru care --seeds 0 1 2 --workers 8

Writes results/verifier_bootstrap.json: per fold, per dataset, per seed metrics
for both arms, plus the across-fold mean/sd of the released-hallucination
reduction that the paper reports.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from config import MODEL_DIR, RESULTS_DIR
from data import load_splits
from framework.pipeline import CLLLMOpsPipeline
from parallel_eval import get_workers, iter_job_pool, reseed_and_build

METRICS = ("accuracy", "f1", "hallucination_rate", "action_accuracy",
           "false_escalation_rate", "proceed_rate", "defer_rate",
           "escalation_rate")


def _job(dataset_name, X_test, y_test, seed, checkpoint):
    """Paired arms on shared generations under one verifier checkpoint."""
    pipeline = reseed_and_build(
        seed, lambda: CLLLMOpsPipeline(dataset_name=dataset_name, seed=seed,
                                       caevn_checkpoint=str(checkpoint)))
    paired = pipeline.evaluate_paired_agent_nov(X_test, y_test, seed=seed)
    return {"cl_llmops": paired["cl_llmops"], "agent_nov": paired["agent_nov"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", required=True,
                    help="checkpoint tags produced by train_caevn.py --tag")
    ap.add_argument("--datasets", nargs="+",
                    default=["pq16", "cwru", "care"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS_DIR / "verifier_bootstrap.json"))
    args = ap.parse_args()
    workers = args.workers if args.workers > 0 else get_workers()

    missing = [f"caevn_{ds}_{tag}.pt" for tag in args.tags
               for ds in args.datasets
               if not (MODEL_DIR / f"caevn_{ds}_{tag}.pt").exists()]
    if missing:
        raise SystemExit("missing verifier checkpoints, train them first:\n  "
                         + "\n  ".join(missing))

    jobs, splits_cache = [], {}
    for ds in args.datasets:
        splits_cache[ds] = load_splits(ds)
    for tag in args.tags:
        for ds in args.datasets:
            X_test, y_test = splits_cache[ds]["X_test"], splits_cache[ds]["y_test"]
            ckpt = MODEL_DIR / f"caevn_{ds}_{tag}.pt"
            for seed in args.seeds:
                jobs.append(((tag, ds, seed),
                             lambda d=ds, x=X_test, y=y_test, s=seed, c=ckpt:
                             _job(d, x, y, s, c)))

    print(f"verifier bootstrap: {len(args.tags)} folds x {len(args.datasets)} "
          f"datasets x {len(args.seeds)} seeds = {len(jobs)} paired jobs "
          f"(workers={workers})", flush=True)
    per_fold = {}
    t0 = time.time()
    for done, ((tag, ds, seed), result) in enumerate(
            iter_job_pool(jobs, workers), start=1):
        per_fold.setdefault(tag, {}).setdefault(ds, {})[str(seed)] = {
            arm: {m: result[arm].get(m) for m in METRICS} for arm in result}
        print(f"  [{done}/{len(jobs)}] {tag} {ds} seed={seed} "
              f"ours_halluc={result['cl_llmops']['hallucination_rate']:.3f} "
              f"nov_halluc={result['agent_nov']['hallucination_rate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for ds in args.datasets:
        reductions, ours, nov = [], [], []
        for tag in args.tags:
            seeds = per_fold.get(tag, {}).get(ds, {})
            if not seeds:
                continue
            o = statistics.mean(v["cl_llmops"]["hallucination_rate"]
                                for v in seeds.values())
            n = statistics.mean(v["agent_nov"]["hallucination_rate"]
                                for v in seeds.values())
            ours.append(o)
            nov.append(n)
            reductions.append(100 * (n - o))
        if not reductions:
            continue
        sd = statistics.stdev(reductions) if len(reductions) > 1 else 0.0
        summary[ds] = {
            "n_folds": len(reductions),
            "reduction_pp": {"mean": round(statistics.mean(reductions), 2),
                             "std": round(sd, 2),
                             "min": round(min(reductions), 2),
                             "max": round(max(reductions), 2),
                             "per_fold": [round(r, 2) for r in reductions]},
            "cl_llmops_hallucination": {"mean": round(statistics.mean(ours), 4),
                                        "per_fold": [round(o, 4) for o in ours]},
            "agent_nov_hallucination": {"mean": round(statistics.mean(nov), 4),
                                        "per_fold": [round(n, 4) for n in nov]},
        }

    doc = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_item": "R3-1 (verifier-training resampling)",
            "design": "each fold retrains CAEVN from an independent draw of the "
                      "verifier training pool (train_caevn.py --tag/--seed); the "
                      "deployed operating point, the perception layer and the "
                      "evaluation episodes are held fixed",
            "not_resampled": ["perception classifier", "LLM generations per fold "
                              "are redrawn only through the decoding seeds",
                              "MOPSO operating point"],
            "tags": args.tags,
            "datasets": args.datasets,
            "seeds": args.seeds,
        },
        "summary": summary,
        "per_fold": per_fold,
    }
    out = Path(args.out)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")

    from config import RESULTS_JSON
    if RESULTS_JSON.exists():
        mirror = json.loads(RESULTS_JSON.read_text())
        mirror["verifier_bootstrap"] = {
            "provenance": f"{out.name} (independent draws of the CAEVN training pool; "
                          "each fold re-decoded and retrained from scratch, evaluated on "
                          "the paired comparison with the deployed operating point fixed)",
            "paper_section": "sec:main_results",
            "values": summary,
        }
        RESULTS_JSON.write_text(json.dumps(mirror, ensure_ascii=False, indent=2))
        print(f"mirrored into {RESULTS_JSON.name}")
    for ds, s in summary.items():
        r = s["reduction_pp"]
        print(f"  {ds:6} reduction {r['mean']:.2f} +/- {r['std']:.2f} pp "
              f"(min {r['min']}, max {r['max']}, folds={s['n_folds']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
