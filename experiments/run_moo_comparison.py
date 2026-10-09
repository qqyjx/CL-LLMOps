"""MOPSO vs NSGA-II comparison with multi-run significance tests.

Reviewer request: the MOPSO-vs-NSGA-II advantage is marginal on a
single hypervolume indicator and should be substantiated with
multi-run significance tests and complementary indicators (IGD,
spread).

Protocol:
  - Both algorithms run N times (default 10, seeds 0..N-1) with
    identical evaluation budgets: P * (G + 1) objective evaluations
    (P particles/individuals, G iterations/generations, + init).
  - A combined reference front is built as the merged non-dominated
    set across all runs of both algorithms; objectives are normalized
    to [0, 1] by the reference-front ideal/nadir points.
  - Per run: hypervolume (ref point 1.1^m), IGD, generalized spread.
  - Paired Wilcoxon signed-rank test per indicator (scipy.stats if
    available, exact sign-flip permutation otherwise), with Holm
    step-down correction across the three indicators.

Outputs results/moo_comparison.json by default (override with --out).
Use --smoke to swap in a cheap DTLZ2-like synthetic objective on the
same 6-dim box, so the harness can be verified without the LLM
pipeline / GPU.
"""

import argparse
import json
import threading
import time
from itertools import product
from pathlib import Path

import numpy as np

from config import RESULTS_DIR, MOPSO_SWARM_SIZE, MOPSO_ITERATIONS
from optimization.objectives import (
    BOUNDS_LOWER, BOUNDS_UPPER, N_OBJECTIVES,
    discretize_particle, constraint_check,
)
from optimization.mopso import MOPSO
from optimization.nsga2 import NSGA2
from optimization.moo_metrics import (
    nondominated_front, normalize_front, hypervolume, igd,
    generalized_spread,
)
from parallel_eval import get_workers, iter_job_pool

INDICATORS = ("hypervolume", "igd", "spread")
# Direction of improvement per indicator (for reporting which
# algorithm is ahead): +1 = higher is better, -1 = lower is better.
INDICATOR_SIGN = {"hypervolume": +1, "igd": -1, "spread": -1}
HV_REF_COORD = 1.1  # reference point coordinate in normalized space


class _LockedObjective:
    """Serialize fitness calls into the shared pipeline objective.

    All runs of both algorithms share ONE PipelineObjective so the
    expensive LLM/verifier artifact cache (``_build_moo_cache``) is
    built exactly once, as in the sequential runner.  The cache and
    per-beta verification-score tables are warmed in the main thread
    before pooling; the lock then only guards the residual mutations
    (first-time-per-beta ``self.caevn`` swap, cache dict fills) while
    every call returns the same deterministic value it would return
    sequentially.
    """

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def __call__(self, x):
        with self._lock:
            return self._inner(x)


class SmokeObjective:
    """DTLZ2-like synthetic 3-objective problem on the same 6-dim box.

    Keeps the discretization and constraint-penalty behavior of the
    real PipelineObjective so the whole harness (including the
    infeasible-solution path) is exercised without the LLM pipeline.
    """

    def __init__(self):
        self._call_count = 0

    def __call__(self, x):
        x = discretize_particle(x)
        if not constraint_check(x):
            return np.full(N_OBJECTIVES, 2.0)

        self._call_count += 1
        z = (x - BOUNDS_LOWER) / (BOUNDS_UPPER - BOUNDS_LOWER)
        g = float(((z[2:] - 0.5) ** 2).sum())
        t0, t1 = z[0] * np.pi / 2.0, z[1] * np.pi / 2.0
        return np.array([
            (1.0 + g) * np.cos(t0) * np.cos(t1),
            (1.0 + g) * np.cos(t0) * np.sin(t1),
            (1.0 + g) * np.sin(t0),
        ])


def wilcoxon_signed_rank(x, y):
    """Two-sided paired Wilcoxon signed-rank test.

    Uses scipy.stats.wilcoxon when available (scipy is a project
    dependency); otherwise falls back to an exact sign-flip
    permutation test over the 2^n sign assignments (fine for n <= 20).

    Returns:
        (statistic, p_value); p = 1.0 if all paired differences are 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    nz = diffs[diffs != 0.0]
    if len(nz) == 0:
        return 0.0, 1.0

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(x, y)
        return float(stat), float(p)
    except ImportError:
        pass

    # Exact permutation fallback: average ranks of |d|, W+ statistic.
    abs_d = np.abs(nz)
    order = np.argsort(abs_d)
    ranks = np.empty(len(nz))
    ranks[order] = np.arange(1, len(nz) + 1)
    for v in np.unique(abs_d):
        tie = abs_d == v
        ranks[tie] = ranks[tie].mean()

    w_obs = ranks[nz > 0].sum()
    n = len(nz)
    mu = n * (n + 1) / 4.0
    count = 0
    for signs in product((0.0, 1.0), repeat=n):
        w = (ranks * np.array(signs)).sum()
        if abs(w - mu) >= abs(w_obs - mu) - 1e-12:
            count += 1
    w_minus = n * (n + 1) / 2.0 - w_obs
    return float(min(w_obs, w_minus)), count / 2.0 ** n


def holm_correction(pvals):
    """Holm step-down adjusted p-values (family-wise error control)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for k, idx in enumerate(order):
        running_max = max(running_max, (m - k) * pvals[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def run_algorithm(name, objective_fn, pop, iters, seed):
    """Run one algorithm once, return its final front objectives (n, m)."""
    if name == "mopso":
        opt = MOPSO(objective_fn, n_particles=pop, n_iterations=iters,
                    seed=seed)
    else:
        opt = NSGA2(objective_fn, pop_size=pop, n_generations=iters,
                    seed=seed)
    _, front_obj, _ = opt.optimize(verbose=False)
    return np.array(front_obj)


def main():
    parser = argparse.ArgumentParser(
        description="MOPSO vs NSGA-II multi-run comparison")
    parser.add_argument("--dataset", type=str, default="pq16",
                        choices=["pq16", "pq", "cwru", "care"])
    parser.add_argument("--runs", type=int, default=10,
                        help="independent runs per algorithm")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="seeds are seed_base .. seed_base+runs-1")
    parser.add_argument("--swarm", type=int, default=MOPSO_SWARM_SIZE,
                        help="population/swarm size for both algorithms")
    parser.add_argument("--iterations", type=int, default=MOPSO_ITERATIONS,
                        help="iterations/generations for both algorithms")
    parser.add_argument("--smoke", action="store_true",
                        help="use a cheap synthetic objective (no LLM/GPU)")
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent independent runs (default env "
                             "ASOC_EVAL_WORKERS or 16). Runs share one "
                             "warmed objective through a lock, so fronts "
                             "are identical to the sequential loop.")
    parser.add_argument("--out", type=str,
                        default=str(RESULTS_DIR / "moo_comparison.json"))
    args = parser.parse_args()

    seeds = list(range(args.seed_base, args.seed_base + args.runs))
    eval_budget = args.swarm * (args.iterations + 1)

    print("MOPSO vs NSGA-II comparison")
    print(f"  Mode:        {'SMOKE (synthetic)' if args.smoke else 'pipeline'}")
    print(f"  Dataset:     {args.dataset if not args.smoke else '-'}")
    print(f"  Runs:        {args.runs} (seeds {seeds[0]}..{seeds[-1]})")
    print(f"  Budget/run:  {args.swarm} x ({args.iterations}+1) "
          f"= {eval_budget} evaluations")
    print("=" * 60)

    if args.smoke:
        objective = SmokeObjective()
    else:
        from optimization.objectives import PipelineObjective
        objective = PipelineObjective(dataset=args.dataset)
        # Warm the LLM/verifier artifact cache once in the main thread
        # (identical to what the first sequential run would build), so
        # pooled runs only re-gate cached outputs.
        objective.pipeline._ensure_built()
        objective.pipeline._build_moo_cache()
    objective = _LockedObjective(objective)

    # ── Run both algorithms N times each ──
    # Independent (seed x algorithm) jobs; fronts are stored by seed so
    # downstream pairing is identical to the sequential loop.
    seed_pos = {seed: i for i, seed in enumerate(seeds)}
    jobs = []
    for seed in seeds:
        for algo in ("mopso", "nsga2"):
            jobs.append(((seed, algo),
                         lambda s=seed, a=algo: run_algorithm(
                             a, objective, args.swarm, args.iterations, s)))

    fronts = {"mopso": [None] * len(seeds), "nsga2": [None] * len(seeds)}
    t0 = time.time()
    n_jobs = len(jobs)
    print(f"  {n_jobs} runs (workers={args.workers})")
    for done, (key, front) in enumerate(
            iter_job_pool(jobs, args.workers), start=1):
        seed, algo = key
        fronts[algo][seed_pos[seed]] = front
        print(f"  [{done}/{n_jobs}] seed={seed}  {algo:6s}  "
              f"front_size={len(front)}  ({time.time()-t0:.1f}s elapsed)",
              flush=True)

    # ── Combined reference front + normalization ──
    merged = np.vstack([f for algo in fronts for f in fronts[algo]])
    ref_front = nondominated_front(merged)
    ideal = ref_front.min(axis=0)
    nadir = ref_front.max(axis=0)
    ref_front_n = normalize_front(ref_front, ideal, nadir)
    ref_point = np.full(N_OBJECTIVES, HV_REF_COORD)
    print(f"\nReference front: {len(ref_front)} points  "
          f"ideal={np.round(ideal, 4).tolist()}  "
          f"nadir={np.round(nadir, 4).tolist()}")

    # ── Per-run indicators ──
    per_run = {algo: [] for algo in fronts}
    for algo in fronts:
        for seed, front in zip(seeds, fronts[algo]):
            front_n = normalize_front(front, ideal, nadir)
            per_run[algo].append({
                "seed": seed,
                "front_size": int(len(front)),
                "hypervolume": hypervolume(front_n, ref_point),
                "igd": igd(front_n, ref_front_n),
                "spread": generalized_spread(front_n, ref_front_n),
            })

    summary = {
        algo: {
            ind: {
                "mean": float(np.mean([r[ind] for r in per_run[algo]])),
                "std": float(np.std([r[ind] for r in per_run[algo]])),
            }
            for ind in INDICATORS
        }
        for algo in per_run
    }

    # ── Significance tests (paired by seed) + Holm correction ──
    raw_p, stats = {}, {}
    for ind in INDICATORS:
        x = [r[ind] for r in per_run["mopso"]]
        y = [r[ind] for r in per_run["nsga2"]]
        stat, p = wilcoxon_signed_rank(x, y)
        stats[ind], raw_p[ind] = stat, p
    adjusted = holm_correction([raw_p[ind] for ind in INDICATORS])

    tests = {}
    for ind, p_adj in zip(INDICATORS, adjusted):
        mean_diff = (summary["mopso"][ind]["mean"]
                     - summary["nsga2"][ind]["mean"])
        signed = mean_diff * INDICATOR_SIGN[ind]
        better = "mopso" if signed > 0 else ("nsga2" if signed < 0 else "tie")
        tests[ind] = {
            "statistic": stats[ind],
            "p_raw": raw_p[ind],
            "p_holm": float(p_adj),
            "better_mean": better,
        }

    # ── Report ──
    print(f"\n{'Indicator':<12} {'MOPSO (mean±std)':<22} "
          f"{'NSGA-II (mean±std)':<22} {'p_raw':<10} {'p_holm':<10} better")
    for ind in INDICATORS:
        s_m = summary["mopso"][ind]
        s_n = summary["nsga2"][ind]
        t = tests[ind]
        print(f"{ind:<12} {s_m['mean']:.4f} ± {s_m['std']:.4f}      "
              f"{s_n['mean']:.4f} ± {s_n['std']:.4f}      "
              f"{t['p_raw']:<10.4f} {t['p_holm']:<10.4f} {t['better_mean']}")

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": "smoke" if args.smoke else "pipeline",
            "dataset": None if args.smoke else args.dataset,
            "n_runs": args.runs,
            "seeds": seeds,
            "population_size": args.swarm,
            "iterations": args.iterations,
            "eval_budget_per_run": eval_budget,
            "hv_reference_point": [HV_REF_COORD] * N_OBJECTIVES,
            "normalization": {
                "ideal": ideal.tolist(),
                "nadir": nadir.tolist(),
            },
            "test": "Wilcoxon signed-rank (two-sided, paired by seed), "
                    "Holm correction across 3 indicators",
        },
        "reference_front": {
            "size": int(len(ref_front)),
            "points": [row.tolist() for row in ref_front],
        },
        "per_run": per_run,
        "summary": summary,
        "significance": tests,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
