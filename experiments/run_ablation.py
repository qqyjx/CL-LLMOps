"""Ablation study runner: Table 5 (component ablation).

Tests CL-LLMOps variants with components removed:
  - w/o CAEVN: replace cross-attention verifier with MLP
  - w/o FIS: replace fuzzy gating with hard threshold
  - w/o Self-consistency: M=1 instead of M=5
  - w/o Regression: skip regression testing stage
  - w/o Gating: remove verification entirely (= Agent-noV)
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from config import RESULTS_JSON, RANDOM_SEED
from data import load_splits
from framework.pipeline import CLLLMOpsPipeline
from parallel_eval import get_workers, iter_job_pool, reseed_and_build


ABLATION_VARIANTS = [
    "full",
    "no_verifier",
    "no_gating",
    "no_self_consistency",
    "no_regression",
    "rule_only_verifier",
    "model_only_verifier",
    "mlp_verifier",
    "hard_gating",
    "grid_search",
]


METRICS = (
    "accuracy", "f1", "hallucination_rate", "action_accuracy",
    "false_escalation_rate", "proceed_rate", "defer_rate",
    "escalation_rate",
)


def _paired_job(dataset_name, X_test, y_test, seed):
    """full vs no_verifier on shared generations (single pipeline)."""
    pipeline = reseed_and_build(
        seed, lambda: CLLLMOpsPipeline(dataset_name=dataset_name, seed=seed))
    paired = pipeline.evaluate_paired_agent_nov(X_test, y_test, seed=seed)
    return {"full": paired["cl_llmops"], "no_verifier": paired["agent_nov"]}


def _variant_job(dataset_name, X_test, y_test, seed, variant):
    """One ablation variant under one seed (fresh pipeline instance)."""
    pipeline = reseed_and_build(
        seed, lambda: CLLLMOpsPipeline(
            dataset_name=dataset_name,
            ablation_variant=variant,
            seed=seed,
        ))
    return pipeline.evaluate(X_test, y_test, seed=seed)


def run_ablation(dataset_name, seeds, variants=None, workers=1):
    """Run ablation study on given dataset across all variants.

    Independent (seed x variant) jobs run on a thread pool; every job
    builds a freshly-seeded pipeline instance via reseed_and_build, and
    each instance's internal episode loop stays sequential, so results
    are identical to the sequential loop regardless of completion order.
    """
    variants = variants or ABLATION_VARIANTS
    splits = load_splits(dataset_name)
    X_test, y_test = splits["X_test"], splits["y_test"]

    paired = "full" in variants and "no_verifier" in variants
    singles = [v for v in variants
               if not (paired and v in ("full", "no_verifier"))]
    if paired:
        print("\n  Paired variants: full / no_verifier")

    jobs = []
    for seed in seeds:
        if paired:
            jobs.append(((seed, "paired"),
                         lambda s=seed: _paired_job(
                             dataset_name, X_test, y_test, s)))
        for variant in singles:
            jobs.append(((seed, variant),
                         lambda s=seed, v=variant: _variant_job(
                             dataset_name, X_test, y_test, s, v)))

    per_seed = {str(seed): {} for seed in seeds}
    n_jobs = len(jobs)
    print(f"  {n_jobs} ablation jobs (workers={workers})")
    for done, (key, result) in enumerate(iter_job_pool(jobs, workers),
                                         start=1):
        seed, variant = key
        if variant == "paired":
            per_seed[str(seed)]["full"] = result["full"]
            per_seed[str(seed)]["no_verifier"] = result["no_verifier"]
            shown = result["full"]
        else:
            per_seed[str(seed)][variant] = result
            shown = result
        print(f"  [{done}/{n_jobs}] seed={seed} variant={variant} "
              f"acc={shown.get('accuracy', 0):.3f}  "
              f"halluc={shown.get('hallucination_rate', 0):.3f}  "
              f"action_acc={shown.get('action_accuracy', 0):.2f}",
              flush=True)

    # Restore the sequential loop's canonical variant order per seed.
    order = (["full", "no_verifier"] if paired else []) + singles
    for seed in seeds:
        per_seed[str(seed)] = {v: per_seed[str(seed)][v] for v in order}
    return per_seed


def aggregate(per_seed):
    variants = list(next(iter(per_seed.values())).keys())
    output = {}
    for variant in variants:
        output[variant] = {}
        for metric in METRICS:
            vals = [rows[variant].get(metric) for rows in per_seed.values()]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                output[variant][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "raw": vals,
                }
    return output


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"],
                        help="Datasets for ablation")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Ablation variants to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent (seed x variant) jobs (default env "
                             "ASOC_EVAL_WORKERS or 16). Each job builds a "
                             "fresh pipeline instance from its own seed, "
                             "so results do not depend on scheduling.")
    args = parser.parse_args()

    all_results = {}
    for ds_name in args.datasets:
        print(f"\n{'='*70}\nAblation Study on {ds_name}\n{'='*70}")
        per_seed = run_ablation(
            ds_name, args.seeds, args.variants, workers=args.workers)
        all_results[ds_name] = {
            "per_seed": per_seed,
            "summary": aggregate(per_seed),
        }

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": "ablation",
            "datasets": args.datasets,
            "seeds": args.seeds,
        },
        "results": all_results,
    }

    output_path = RESULTS_JSON.parent / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
