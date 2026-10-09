"""MOPSO optimization runner: Table 3 + Figure 6 (Pareto front).

Runs multi-objective optimization to find optimal hyperparameters.
Outputs Pareto front data for plotting.
"""

import argparse
import json
import time
from pathlib import Path

from config import (
    RESULTS_JSON, OPERATING_POINT_JSON, MOPSO_SWARM_SIZE, MOPSO_ITERATIONS,
)
from optimization.objectives import PipelineObjective, N_OBJECTIVES
from optimization.mopso import MOPSO


def main():
    parser = argparse.ArgumentParser(description="MOPSO Optimization")
    parser.add_argument("--swarm", type=int, default=MOPSO_SWARM_SIZE)
    parser.add_argument("--iterations", type=int, default=MOPSO_ITERATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="pq16",
                        choices=["pq16", "pq", "cwru", "care"])
    parser.add_argument("--out", type=Path,
                        default=RESULTS_JSON.parent / "mopso_results.json")
    parser.add_argument("--operating-point", type=Path,
                        default=OPERATING_POINT_JSON)
    args = parser.parse_args()

    print(f"MOPSO Optimization")
    print(f"  Swarm size: {args.swarm}")
    print(f"  Iterations: {args.iterations}")
    print("=" * 60)

    objective = PipelineObjective(dataset=args.dataset)

    optimizer = MOPSO(
        objective_fn=objective,
        n_particles=args.swarm,
        n_iterations=args.iterations,
        seed=args.seed,
    )

    pareto_pos, pareto_obj, history = optimizer.optimize(verbose=True)

    best_pos, best_obj = optimizer.get_balanced_solution()

    print(f"\n{'='*60}")
    print(f"Pareto front size: {len(pareto_pos)}")
    if best_pos is not None:
        print(f"Balanced solution:")
        print(f"  alpha_p = {best_pos[0]:.3f}")
        print(f"  alpha_d = {best_pos[1]:.3f}")
        print(f"  delta   = {best_pos[2]:.3f}")
        print(f"  lambda_r= {best_pos[3]:.3f}")
        print(f"  M       = {int(best_pos[4])}")
        print(f"  beta    = {best_pos[5]:.4f}")
        print(f"  -> F1={-best_obj[0]:.4f}, halluc={best_obj[1]:.4f}, "
              f"handoff={best_obj[2]:.4f}")

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "swarm_size": args.swarm,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "pareto_front": [
            {
                "params": {
                    "alpha_p": float(pos[0]),
                    "alpha_d": float(pos[1]),
                    "delta": float(pos[2]),
                    "lambda_r": float(pos[3]),
                    "M": int(pos[4]),
                    "beta": float(pos[5]),
                },
                "objectives": {
                    "neg_f1": float(obj[0]),
                    "hallucination": float(obj[1]),
                    "handoff_rate": float(obj[2]),
                },
            }
            for pos, obj in zip(pareto_pos, pareto_obj)
        ],
        "balanced_solution": {
            "params": {
                "alpha_p": float(best_pos[0]),
                "alpha_d": float(best_pos[1]),
                "delta": float(best_pos[2]),
                "lambda_r": float(best_pos[3]),
                "M": int(best_pos[4]),
                "beta": float(best_pos[5]),
            },
            "objectives": {
                "f1": float(-best_obj[0]),
                "hallucination": float(best_obj[1]),
                "handoff_rate": float(best_obj[2]),
            },
        } if best_pos is not None else None,
        "convergence": {
            "archive_size_per_iter": history["n_archive"],
        },
    }

    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    if best_pos is not None:
        _, resolved_beta = objective.pipeline._resolve_beta_checkpoint(
            args.dataset, float(best_pos[5]))
        op = dict(output["balanced_solution"]["params"])
        op["beta_requested"] = op["beta"]
        op["beta"] = float(resolved_beta)
        args.operating_point.parent.mkdir(parents=True, exist_ok=True)
        args.operating_point.write_text(json.dumps({
            "source": "validation-only MOPSO balanced Pareto solution",
            "dataset": args.dataset,
            "seed": args.seed,
            "params": op,
            "objectives": output["balanced_solution"]["objectives"],
            "timestamp": output["metadata"]["timestamp"],
        }, indent=2))
        print(f"Operating point saved to {args.operating_point}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
