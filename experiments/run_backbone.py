"""Multi-backbone evaluation: validate CL-LLMOps is model-agnostic.

Runs the full CL-LLMOps pipeline with an alternative LLM backbone
(default: Mistral-7B-Instruct-v0.3) across all 3 datasets x 10 seeds.
All other components (CAEVN, FIS, MOPSO, KB) remain unchanged.

Outputs: results/backbone_results.json
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATASETS, RESULTS_DIR, LLM_BACKBONES, DEVICE,
)
from data import load_splits
from framework.pipeline import CLLLMOpsPipeline
from framework.llm_client import VLLMClient
from parallel_eval import get_workers, iter_job_pool, reseed_and_build


DATASET_KEY_MAP = {"pq16": "PQ_Classification", "pq": "PQ_legacy",
                   "cwru": "CWRU", "care": "CARE_Wind"}


def _seed_job(ds, model_id, splits, seed):
    """One (dataset, seed) full-pipeline evaluation (one job).

    The client and pipeline are fresh instances built under
    reseed_and_build (np.random.seed + config.RANDOM_SEED set inside
    the build lock, then explicit seed kwargs), matching the
    sequential loop's seeding exactly.
    """
    def build():
        client = VLLMClient(model=model_id, seed=seed)
        return CLLLMOpsPipeline(
            dataset_name=ds, llm_client=client, seed=seed,
        )

    pipeline = reseed_and_build(seed, build)
    metrics = pipeline.evaluate(splits["X_test"], splits["y_test"],
                                seed=seed)
    return {"seed": seed, **metrics}


def run_backbone(model_key, datasets, seeds, workers=1):
    """Run full pipeline with the specified backbone across seeds.

    Independent (dataset x seed) jobs run on a thread pool; per-seed
    rows are stored by seed, so results are identical to the
    sequential loop regardless of completion order.
    """
    model_id = LLM_BACKBONES[model_key]
    print(f"Loading vLLM with model: {model_id}")

    results = {}
    for ds in datasets:
        ds_key = DATASET_KEY_MAP[ds]
        print(f"\n{'='*50}")
        print(f"Dataset: {DATASETS[ds]['name']}  Backbone: {model_key}")
        print(f"{'='*50}")

        splits = load_splits(ds)

        seed_pos = {seed: i for i, seed in enumerate(seeds)}
        jobs = [(seed, lambda s=seed: _seed_job(ds, model_id, splits, s))
                for seed in seeds]
        seed_results = [None] * len(seeds)
        print(f"  {len(jobs)} evaluation jobs (workers={workers})")
        for done, (seed, row) in enumerate(
                iter_job_pool(jobs, workers), start=1):
            seed_results[seed_pos[seed]] = row
            print(f"  [{done}/{len(jobs)}] Seed {seed}: "
                  f"acc={row.get('accuracy', 0):.3f}", flush=True)

        accs = [r["accuracy"] for r in seed_results]
        metric_keys = sorted(set.intersection(*[
            {k for k, v in row.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
            for row in seed_results
        ]))
        results[ds_key] = {
            "summary": {
                key: {"mean": float(np.mean([r[key] for r in seed_results])),
                      "std": float(np.std([r[key] for r in seed_results]))}
                for key in metric_keys
            },
            "per_seed": seed_results,
            "model_id": model_id,
        }
        print(f"  Mean acc: {results[ds_key]['summary']['accuracy']['mean']:.3f} "
              f"+/- {results[ds_key]['summary']['accuracy']['std']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CL-LLMOps Multi-Backbone Evaluation")
    parser.add_argument("--model", choices=list(LLM_BACKBONES.keys()),
                        default="mistral7b",
                        help="LLM backbone to evaluate")
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"],
                        help="Datasets to evaluate")
    parser.add_argument("--seeds", type=str, default="0-2",
                        help="Seed range (e.g., '0-9')")
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent (dataset x seed) jobs (default env "
                             "ASOC_EVAL_WORKERS or 16). Each job builds a "
                             "fresh client+pipeline from its own seed, so "
                             "results do not depend on scheduling.")
    args = parser.parse_args()

    if "-" in args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Multi-backbone evaluation: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {seeds}\n")

    results = run_backbone(args.model, args.datasets, seeds,
                           workers=args.workers)

    output = {
        "metadata": {
            "experiment": "multi_backbone_evaluation",
            "backbone": args.model,
            "model_id": LLM_BACKBONES[args.model],
            "seeds": seeds,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": results,
    }

    output_path = RESULTS_DIR / "backbone_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
