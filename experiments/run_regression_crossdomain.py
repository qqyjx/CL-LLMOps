"""Cross-domain regression interference after a single-domain update (R4.2).

For each source domain, MOPSO selects a source-specific gate update using only
that source's validation cache.  The frozen update is then replayed on five
decode-seed regression cycles for every target domain.  A target cycle passes
only when F1, hallucination rate and false-escalation rate all remain within
pre-registered tolerances relative to the global PQ16 operating point.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import RESULTS_DIR
from framework.pipeline import CLLLMOpsPipeline
from optimization.mopso import MOPSO
from optimization.objectives import PipelineObjective
from parallel_eval import get_workers, iter_job_pool, reseed_and_build


PARAM_KEYS = ("alpha_p", "alpha_d", "delta", "lambda_r", "M", "beta")


def vector_to_params(vector):
    return {
        "alpha_p": float(vector[0]), "alpha_d": float(vector[1]),
        "delta": float(vector[2]), "lambda_r": float(vector[3]),
        "M": int(round(vector[4])), "beta": float(vector[5]),
    }


def evaluate_params(pipeline, params):
    return pipeline.evaluate_with_params(**{
        key: params[key] for key in PARAM_KEYS
    })


def _optimize_source(source, opt_seed, swarm, iterations):
    """MOPSO source-only update for one source domain (one job).

    The pipeline/objective/optimizer are all job-local and seeded from
    opt_seed exactly as in the sequential loop; the LLM-backed
    validation cache builds inside this job.
    """
    pipeline = reseed_and_build(
        opt_seed,
        lambda: CLLLMOpsPipeline(dataset_name=source, seed=opt_seed))
    objective = PipelineObjective(pipeline=pipeline, dataset=source)
    optimizer = MOPSO(
        objective, n_particles=swarm,
        n_iterations=iterations, seed=opt_seed)
    _, _, _ = optimizer.optimize(verbose=False)
    best_pos, best_obj = optimizer.get_balanced_solution()
    if best_pos is None:
        raise RuntimeError(f"no feasible MOPSO update for {source}")
    return {
        "params": vector_to_params(best_pos),
        "source_validation_objectives": {
            "f1": float(-best_obj[0]),
            "hallucination_rate": float(best_obj[1]),
            "false_escalation_rate": float(best_obj[2]),
        },
    }


def _replay_job(target, seed, global_op, updates, tolerances):
    """One (seed, target) regression suite replay (one job).

    Baseline and every frozen source update are re-gated on this job's
    own freshly-seeded pipeline (its cached generations are built once
    and shared across the updates, as in the sequential loop).  Rows
    are returned in updates order; the caller re-assembles all rows in
    the sequential seed -> target order.
    """
    f1_tol, halluc_tol, esc_tol = tolerances
    pipeline = reseed_and_build(
        seed, lambda: CLLLMOpsPipeline(dataset_name=target, seed=seed))
    baseline = evaluate_params(pipeline, global_op)
    rows = []
    for source, update in updates.items():
        changed = evaluate_params(pipeline, update["params"])
        deltas = {
            "f1": float(changed["f1"] - baseline["f1"]),
            "hallucination_rate": float(
                changed["hallucination_rate"]
                - baseline["hallucination_rate"]),
            "false_escalation_rate": float(
                changed["false_escalation_rate"]
                - baseline["false_escalation_rate"]),
        }
        passed = (
            deltas["f1"] >= -f1_tol
            and deltas["hallucination_rate"] <= halluc_tol
            and deltas["false_escalation_rate"] <= esc_tol
        )
        rows.append({
            "seed": seed, "source_update": source,
            "target_suite": target,
            "in_domain": source == target,
            "baseline": baseline, "updated": changed,
            "deltas": deltas, "passed": bool(passed),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"])
    parser.add_argument("--eval-seeds", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--optimization-seed", type=int, default=42)
    parser.add_argument("--swarm", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--f1-drop-tol", type=float, default=0.01)
    parser.add_argument("--hallucination-increase-tol", type=float,
                        default=0.01)
    parser.add_argument("--false-escalation-increase-tol", type=float,
                        default=0.02)
    parser.add_argument("--allow-default-operating-point",
                        action="store_true")
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent independent jobs (default env "
                             "ASOC_EVAL_WORKERS or 16): one per source "
                             "domain in the MOPSO phase, one per "
                             "(seed x target) in the replay phase. Each "
                             "job builds freshly-seeded pipelines, so "
                             "results do not depend on scheduling.")
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "regression_crossdomain.json")
    args = parser.parse_args()

    probe = CLLLMOpsPipeline(dataset_name=args.datasets[0])
    global_op = probe._load_operating_params()
    if not global_op:
        if not args.allow_default_operating_point:
            raise RuntimeError(
                "results/operating_point.json is required; run MOPSO on "
                "PQ16 validation first")
        global_op = {
            "alpha_p": 0.7, "alpha_d": 0.3, "delta": 0.1,
            "lambda_r": 0.4, "M": 5, "beta": 0.01,
        }

    # Phase 1: one independent MOPSO job per source domain.
    opt_jobs = [(source,
                 lambda s=source: _optimize_source(
                     s, args.optimization_seed, args.swarm,
                     args.iterations))
                for source in args.datasets]
    print(f"{len(opt_jobs)} source-update MOPSO jobs "
          f"(workers={args.workers})", flush=True)
    updates = {}
    for source, update in iter_job_pool(opt_jobs, args.workers):
        updates[source] = update
        print(f"optimized source-only update: {source}", flush=True)
    # Freeze the updates in the sequential loop's dataset order.
    updates = {source: updates[source] for source in args.datasets}

    # Phase 2: one independent replay job per (seed, target); rows are
    # re-assembled in the sequential seed -> target order.
    tolerances = (args.f1_drop_tol, args.hallucination_increase_tol,
                  args.false_escalation_increase_tol)
    replay_jobs = [((seed, target),
                    lambda sd=seed, tg=target: _replay_job(
                        tg, sd, global_op, updates, tolerances))
                   for seed in args.eval_seeds
                   for target in args.datasets]
    print(f"{len(replay_jobs)} regression replay jobs "
          f"(workers={args.workers})", flush=True)
    rows_by_key = {}
    for key, job_rows in iter_job_pool(replay_jobs, args.workers):
        rows_by_key[key] = job_rows
        print(f"regression target={key[1]} seed={key[0]} done",
              flush=True)
    rows = []
    for seed in args.eval_seeds:
        for target in args.datasets:
            rows.extend(rows_by_key[(seed, target)])

    matrix = {}
    for source in args.datasets:
        matrix[source] = {}
        for target in args.datasets:
            cells = [r for r in rows if r["source_update"] == source
                     and r["target_suite"] == target]
            matrix[source][target] = {
                "pass_rate": float(np.mean([r["passed"] for r in cells])),
                "n_cycles": len(cells),
                "mean_deltas": {
                    metric: float(np.mean([r["deltas"][metric]
                                           for r in cells]))
                    for metric in ("f1", "hallucination_rate",
                                   "false_escalation_rate")
                },
            }

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_item": "R4.2",
            "protocol": "source-validation MOPSO update -> frozen replay on "
                        "all target regression suites",
            "eval_seeds": args.eval_seeds,
            "tolerances": {
                "max_f1_drop": args.f1_drop_tol,
                "max_hallucination_increase":
                    args.hallucination_increase_tol,
                "max_false_escalation_increase":
                    args.false_escalation_increase_tol,
            },
        },
        "global_operating_point": global_op,
        "source_updates": updates,
        "pass_rate_matrix": matrix,
        "per_cycle": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
