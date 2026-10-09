"""Statistical-rigor protocol runner (reviewer request R3-1).

Reviewer request: the main evaluation uses a single fixed data split with
ten fixed decoding seeds, so the reported variance captures only LLM
decoding noise, not data-partition uncertainty. This script implements a
more rigorous protocol:

  1. Keep every train-side artifact (alarm model, knowledge base, and
     CAEVN checkpoint) fixed and draw N class-stratified bootstrap
     evaluation partitions from the untouched original test set.  This
     captures test-composition uncertainty without ever moving a sample
     used to train CAEVN into evaluation.
  2. For each split, evaluate each method under M decoding seeds
     (default M=3, seeds 0..2). Decoding seeds are plumbed the same way
     as runbooks/rerun_r1_aggregate.py: `config.RANDOM_SEED = seed` is
     set before the method is constructed (and a `seed` kwarg is passed
     through to `evaluate()` when its signature accepts one). This layer
     captures LLM decoding stochasticity.
  3. Statistics: decoding seeds are averaged within a split to give one
     paired value per split; a paired two-sided Wilcoxon signed-rank
     test is run on the N split-level pairs (CL-LLMOps minus each
     baseline), with Holm step-down correction across all
     dataset x metric x baseline comparisons, plus a seeded percentile
     bootstrap 95% CI (10,000 resamples) on the mean paired difference.
     With N=5 pairs the exact two-sided Wilcoxon p-value cannot go
     below 2/2^5 = 0.0625; this floor is recorded in the JSON metadata.

Outputs results/statistical_protocol.json by default (override with
--out). Use --smoke to swap the LLM methods for two cheap local
stand-ins (SVM vs XGBoost on hand-crafted features, on a stratified
subsample) so the whole harness -- resampling, evaluation, pairing,
Wilcoxon+Holm, bootstrap CIs, JSON -- is verified without GPU/vLLM.

Usage:
    python experiments/run_statistical_protocol.py                 # real (A800)
    python experiments/run_statistical_protocol.py --methods cl_llmops agent_nov ragllm
    python experiments/run_statistical_protocol.py --smoke         # local check
"""

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
from config import DATASETS, RESULTS_DIR, SPLIT_RATIOS
from data import load_splits
from parallel_eval import get_workers, iter_job_pool, reseed_and_build
# Reuse the significance-test helpers introduced for the MOPSO/NSGA-II
# comparison (scipy-backed Wilcoxon with exact permutation fallback).
from run_moo_comparison import holm_correction, wilcoxon_signed_rank

OURS = "cl_llmops"
METHOD_CHOICES = ("cl_llmops", "agent_nov", "ragllm", "fdllm")
INFERENCE_METRICS = (
    "accuracy", "f1", "hallucination_rate", "action_accuracy",
    "false_escalation_rate",
)
SPLIT_KEYS = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")

# Datasets without their own DATASETS entry reuse another config for
# feature extraction (pq16 shares the PQ signal layout: 6 x 640).
FEATURE_CONFIG_ALIAS = {"pq16": "pq"}

# ---------------------------------------------------------------------------
# Split resampling
# ---------------------------------------------------------------------------

def resample_splits(splits_dict, seed):
    """Draw a class-stratified bootstrap of the untouched test set.

    Train and validation arrays remain fixed.  This is essential because
    the alarm model and CAEVN checkpoint were trained from those arrays;
    pooling all partitions and re-splitting would silently leak their
    training samples into the evaluation set.

    Args:
        splits_dict: dict with keys X_train, y_train, X_val, y_val,
            X_test, y_test (numpy arrays; empty partitions allowed).
        seed: random_state for both train_test_split stages.

    Returns:
        dict with the same six keys.  Train/validation are unchanged and
        test has the same per-class counts as the original, sampled with
        replacement.
    """
    X_test = splits_dict["X_test"]
    y_test = np.asarray(splits_dict["y_test"])
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(y_test):
        idx = np.flatnonzero(y_test == cls)
        chosen.append(rng.choice(idx, size=len(idx), replace=True))
    chosen = rng.permutation(np.concatenate(chosen))
    out = dict(splits_dict)
    out["X_test"] = X_test[chosen]
    out["y_test"] = y_test[chosen]
    return out


# ---------------------------------------------------------------------------
# Method construction / evaluation (real mode)
# ---------------------------------------------------------------------------

def build_method(name, dataset_name, n_classes):
    """Construct a method following the run_main.py calling convention."""
    if name == "cl_llmops":
        from framework.pipeline import CLLLMOpsPipeline
        return CLLLMOpsPipeline(dataset_name=dataset_name)
    if name == "agent_nov":
        from baselines.llm_baselines import AgentNoVerifyBaseline
        return AgentNoVerifyBaseline(dataset_name=dataset_name)
    if name == "ragllm":
        from baselines.llm_baselines import RAGLLMBaseline
        return RAGLLMBaseline(dataset_name=dataset_name,
                              n_classes=n_classes)
    if name == "fdllm":
        from baselines.llm_baselines import FDLLMBaseline
        return FDLLMBaseline(dataset_name=dataset_name,
                             n_classes=n_classes)
    raise ValueError(f"Unknown method: {name}")


def call_evaluate(method_obj, X_test, y_test, seed):
    """Call evaluate(), passing the decoding seed when accepted.

    None of the current baselines expose a `seed` kwarg on evaluate();
    the effective plumbing is `config.RANDOM_SEED = seed` before
    construction (the runbooks/rerun_r1_aggregate.py pattern). The
    signature check makes the pass-through automatic if a seed kwarg is
    added later.
    """
    try:
        params = inspect.signature(method_obj.evaluate).parameters
        if "seed" in params:
            return method_obj.evaluate(X_test, y_test, seed=seed)
    except (TypeError, ValueError):
        pass
    return method_obj.evaluate(X_test, y_test)


def evaluate_real(method_name, dataset_name, resampled, decode_seed):
    """One real evaluation: reseed, rebuild method, score test partition."""
    method_obj = reseed_and_build(
        decode_seed,
        lambda: build_method(
            method_name, dataset_name,
            _dataset_n_classes(dataset_name, resampled["y_train"])))
    return call_evaluate(method_obj, resampled["X_test"],
                         resampled["y_test"], decode_seed)


def evaluate_real_paired(dataset_name, resampled, decode_seed):
    """Full vs Agent-noV on shared LLM generations (paired protocol)."""
    from framework.pipeline import CLLLMOpsPipeline
    pipeline = reseed_and_build(
        decode_seed,
        lambda: CLLLMOpsPipeline(dataset_name=dataset_name,
                                 seed=decode_seed))
    return pipeline.evaluate_paired_agent_nov(
        resampled["X_test"], resampled["y_test"], seed=decode_seed)


def _dataset_n_classes(dataset_name, y):
    if dataset_name in DATASETS and "n_classes" in DATASETS[dataset_name]:
        return DATASETS[dataset_name]["n_classes"]
    return int(len(np.unique(y)))


# ---------------------------------------------------------------------------
# Smoke mode: cheap local stand-ins exercising the identical harness path
# ---------------------------------------------------------------------------

class SmokeStandIn:
    """Cheap local stand-in for an LLM method (smoke mode only).

    Wraps SVM or XGBoost from baselines/classical.py on hand-crafted
    features. The training set is re-drawn as a seed-dependent
    stratified bootstrap so the decoding-seed layer produces nonzero
    within-split variance, exercising the seed-averaging path.
    """

    def __init__(self, flavor, seed):
        from baselines.classical import SVMBaseline, XGBoostBaseline
        if flavor == "xgb":
            self.base = XGBoostBaseline(n_estimators=60, max_depth=4)
        else:
            self.base = SVMBaseline(C=1.0)
        self.seed = seed

    def train(self, X_train, y_train):
        rng = np.random.default_rng(self.seed)
        idx = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            idx.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
        idx = np.concatenate(idx)
        self.base.train(X_train[idx], y_train[idx])

    def evaluate(self, X_test, y_test, seed=None):
        return self.base.evaluate(X_test, y_test)


def smoke_flavor(method_name):
    """Map a method name to a stand-in flavor (ours -> xgb, rest -> svm)."""
    return "xgb" if method_name == OURS else "svm"


def subsample_splits(splits_dict, max_total, seed=0):
    """Stratified subsample of each partition, preserving proportions."""
    total = sum(len(splits_dict[f"y_{p}"]) for p in ("train", "val", "test"))
    if total <= max_total:
        return splits_dict
    frac = max_total / total
    rng = np.random.default_rng(seed)
    out = {}
    for part in ("train", "val", "test"):
        X, y = splits_dict[f"X_{part}"], splits_dict[f"y_{part}"]
        keep = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n_keep = max(2, int(round(len(cls_idx) * frac)))
            keep.append(rng.choice(cls_idx, size=min(n_keep, len(cls_idx)),
                                   replace=False))
        keep = np.concatenate(keep)
        out[f"X_{part}"], out[f"y_{part}"] = X[keep], y[keep]
    return out


def extract_feature_splits(dataset_name, splits_dict):
    """Replace raw signals with hand-crafted feature vectors (smoke)."""
    from features import FeatureExtractor
    feat_name = FEATURE_CONFIG_ALIAS.get(dataset_name, dataset_name)
    extractor = FeatureExtractor(feat_name)
    out = {}
    for part in ("train", "val", "test"):
        out[f"X_{part}"] = extractor.extract_batch(splits_dict[f"X_{part}"])
        out[f"y_{part}"] = splits_dict[f"y_{part}"]
    return out


def evaluate_smoke(method_name, resampled, decode_seed):
    """One smoke evaluation: bootstrap-train a stand-in, score test."""
    standin = SmokeStandIn(smoke_flavor(method_name), decode_seed)
    standin.train(resampled["X_train"], resampled["y_train"])
    return call_evaluate(standin, resampled["X_test"], resampled["y_test"],
                         decode_seed)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def numeric_metric_keys(per_seed_results):
    """Metric keys with numeric, non-None values in every result dict."""
    keys = None
    for res in per_seed_results:
        cur = {k for k, v in res.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
        keys = cur if keys is None else keys & cur
    return sorted(keys or [])


def split_level_means(raw_method, split_seeds, metric):
    """Average a metric over decoding seeds within each split."""
    means = []
    for s in split_seeds:
        vals = [raw_method[str(s)][d][metric] for d in raw_method[str(s)]]
        means.append(float(np.mean(vals)))
    return means


def bootstrap_ci_mean(diffs, n_boot, seed, alpha=0.05):
    """Seeded percentile bootstrap CI on the mean of paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compute_comparisons(raw, split_seeds, baselines, n_boot, boot_seed):
    """Paired stats (ours - baseline) per dataset x baseline x metric."""
    comparisons = {}
    flat = []  # (dataset, baseline, metric) in Holm family order
    raw_pvals = []

    for ds in raw:
        if OURS not in raw[ds]:
            continue
        comparisons[ds] = {}
        ours_flat = [raw[ds][OURS][str(s)][d]
                     for s in split_seeds for d in raw[ds][OURS][str(s)]]
        for baseline in baselines:
            if baseline not in raw[ds]:
                continue
            base_flat = [raw[ds][baseline][str(s)][d]
                         for s in split_seeds
                         for d in raw[ds][baseline][str(s)]]
            available = (set(numeric_metric_keys(ours_flat))
                         & set(numeric_metric_keys(base_flat)))
            metrics = [m for m in INFERENCE_METRICS if m in available]
            comparisons[ds][baseline] = {}
            for metric in metrics:
                x = split_level_means(raw[ds][OURS], split_seeds, metric)
                y = split_level_means(raw[ds][baseline], split_seeds, metric)
                diffs = (np.array(x) - np.array(y)).tolist()
                stat, p_raw = wilcoxon_signed_rank(x, y)
                ci_lo, ci_hi = bootstrap_ci_mean(diffs, n_boot, boot_seed)
                comparisons[ds][baseline][metric] = {
                    "split_level_ours": [round(v, 6) for v in x],
                    "split_level_baseline": [round(v, 6) for v in y],
                    "paired_diffs": [round(v, 6) for v in diffs],
                    "mean_diff": float(np.mean(diffs)),
                    "ci95_mean_diff": [ci_lo, ci_hi],
                    "wilcoxon_statistic": stat,
                    "p_raw": p_raw,
                }
                flat.append((ds, baseline, metric))
                raw_pvals.append(p_raw)

    if raw_pvals:
        adjusted = holm_correction(raw_pvals)
        for (ds, baseline, metric), p_adj in zip(flat, adjusted):
            comparisons[ds][baseline][metric]["p_holm"] = float(p_adj)
    return comparisons, len(raw_pvals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="R3-1 statistical protocol: stratified test bootstrap "
                    "x decoding seeds, Wilcoxon+Holm, bootstrap CIs")
    parser.add_argument("--datasets", nargs="+",
                        default=["pq", "cwru", "care"],
                        help="Datasets to evaluate")
    parser.add_argument("--methods", nargs="+",
                        default=["agent_nov", "cl_llmops"],
                        choices=list(METHOD_CHOICES),
                        help="Methods to evaluate (must include cl_llmops)")
    parser.add_argument("--n-splits", type=int, default=10,
                        help="N stratified bootstrap evaluation partitions")
    parser.add_argument("--split-seed-base", type=int, default=100,
                        help="Split seeds are base .. base+N-1")
    parser.add_argument("--decode-seeds", type=int, default=3,
                        help="M decoding seeds per split")
    parser.add_argument("--decode-seed-base", type=int, default=0,
                        help="Decoding seeds are base .. base+M-1")
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="Bootstrap resamples for the 95%% CI")
    parser.add_argument("--bootstrap-seed", type=int, default=12345,
                        help="Seed for the bootstrap RNG")
    parser.add_argument("--workers", type=int, default=get_workers(),
                        help="Concurrent evaluation jobs (default env "
                             "ASOC_EVAL_WORKERS or 16). Each job builds a "
                             "fresh method instance from its own decode "
                             "seed, so results do not depend on scheduling.")
    parser.add_argument("--smoke", action="store_true",
                        help="Use cheap local stand-ins (no GPU/vLLM)")
    parser.add_argument("--smoke-subsample", type=int, default=400,
                        help="Max pooled samples per dataset in smoke mode")
    parser.add_argument("--out", type=str,
                        default=str(RESULTS_DIR / "statistical_protocol.json"))
    args = parser.parse_args()

    if OURS not in args.methods:
        parser.error(f"--methods must include {OURS} (the compared method)")
    baselines = [m for m in args.methods if m != OURS]

    split_seeds = list(range(args.split_seed_base,
                             args.split_seed_base + args.n_splits))
    decode_seeds = list(range(args.decode_seed_base,
                              args.decode_seed_base + args.decode_seeds))
    n_evals = (len(args.datasets) * len(args.methods)
               * len(split_seeds) * len(decode_seeds))

    print("R3-1 statistical protocol")
    print(f"  Mode:          {'SMOKE (local stand-ins)' if args.smoke else 'real (LLM pipeline)'}")
    print(f"  Datasets:      {args.datasets}")
    print(f"  Methods:       {args.methods}")
    print(f"  Test bootstrap:N={args.n_splits} "
          f"(seeds {split_seeds[0]}..{split_seeds[-1]}; "
          "train/val fixed)")
    print(f"  Decode seeds:  M={args.decode_seeds} (seeds {decode_seeds[0]}..{decode_seeds[-1]})")
    print(f"  Evaluations:   up to {n_evals}")
    print("=" * 60)

    if args.smoke:
        print("SMOKE: cl_llmops -> XGBoost stand-in, other methods -> SVM "
              "stand-in (features from baselines/classical.py path)")

    raw = {}
    skipped = []
    t0 = time.time()

    for ds_name in args.datasets:
        print(f"\n{'=' * 60}\nDataset: {ds_name}\n{'=' * 60}")
        try:
            base_splits = load_splits(ds_name)
        except (FileNotFoundError, KeyError, OSError) as exc:
            print(f"  SKIP {ds_name}: data unavailable ({exc})")
            skipped.append({"dataset": ds_name, "reason": str(exc)})
            continue

        if args.smoke:
            base_splits = subsample_splits(base_splits, args.smoke_subsample)
            print(f"  Extracting features "
                  f"({sum(len(base_splits[f'y_{p}']) for p in ('train', 'val', 'test'))} "
                  f"pooled samples)...")
            base_splits = extract_feature_splits(ds_name, base_splits)

        raw[ds_name] = {m: {} for m in args.methods}
        paired_path = (set(args.methods) == {"agent_nov", "cl_llmops"})
        for split_seed in split_seeds:
            for method in args.methods:
                raw[ds_name][method][str(split_seed)] = {}

        # Independent (split x seed) jobs run concurrently; each builds a
        # freshly-seeded method instance, so results are identical to the
        # sequential loop regardless of completion order.  The test-set
        # bootstrap is drawn inside the worker to avoid holding all
        # resampled test partitions in memory at once.
        jobs = []
        for split_seed in split_seeds:
            for decode_seed in decode_seeds:
                if not args.smoke and paired_path:
                    jobs.append(((split_seed, decode_seed),
                                 lambda ss=split_seed, ds=decode_seed:
                                 evaluate_real_paired(
                                     ds_name,
                                     resample_splits(base_splits, ss), ds)))
                    continue
                for method in args.methods:
                    if args.smoke:
                        jobs.append(((method, split_seed, decode_seed),
                                     lambda m=method, ss=split_seed,
                                     ds=decode_seed:
                                     evaluate_smoke(
                                         m, resample_splits(base_splits,
                                                            ss), ds)))
                    else:
                        jobs.append(((method, split_seed, decode_seed),
                                     lambda m=method, ss=split_seed,
                                     ds=decode_seed:
                                     evaluate_real(
                                         m, ds_name,
                                         resample_splits(base_splits, ss),
                                         ds)))

        n_jobs = len(jobs)
        print(f"  {n_jobs} evaluation jobs (workers={args.workers})")
        for done, (key, result) in enumerate(
                iter_job_pool(jobs, args.workers), start=1):
            if not args.smoke and paired_path:
                split_seed, decode_seed = key
                for method, metrics in result.items():
                    raw[ds_name][method][str(split_seed)][
                        str(decode_seed)] = metrics
                shown = {
                    m: {k: round(v, 4) for k, v in metrics.items()
                        if isinstance(v, float)}
                    for m, metrics in result.items()
                }
            else:
                method, split_seed, decode_seed = key
                raw[ds_name][method][str(split_seed)][str(decode_seed)] \
                    = result
                shown = {k: round(v, 4) for k, v in result.items()
                         if isinstance(v, float)}
            print(f"  [{done}/{n_jobs}] split={split_seed} "
                  f"seed={decode_seed} {shown}  ({time.time() - t0:.1f}s)",
                  flush=True)
    # ── Paired statistics ──
    comparisons, n_comparisons = compute_comparisons(
        raw, split_seeds, baselines, args.bootstrap, args.bootstrap_seed)

    print(f"\n{'Dataset':<8} {'Baseline':<10} {'Metric':<20} "
          f"{'mean diff':<12} {'95% CI':<22} {'p_raw':<9} p_holm")
    for ds in comparisons:
        for baseline in comparisons[ds]:
            for metric, c in comparisons[ds][baseline].items():
                ci = c["ci95_mean_diff"]
                print(f"{ds:<8} {baseline:<10} {metric:<20} "
                      f"{c['mean_diff']:< 12.4f} "
                      f"[{ci[0]: .4f}, {ci[1]: .4f}]   "
                      f"{c['p_raw']:<9.4f} {c.get('p_holm', float('nan')):.4f}")

    p_floor = 2.0 ** (1 - args.n_splits)
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": "smoke" if args.smoke else "real",
            "reviewer_item": "R3-1 (stratified test-set bootstrap, paired "
                             "significance tests, multiple-comparison "
                             "correction, CIs on differences)",
            "datasets": list(raw.keys()),
            "datasets_skipped": skipped,
            "methods": args.methods,
            "compared_method": OURS,
            "n_splits": args.n_splits,
            "split_seeds": split_seeds,
            "original_split_ratios": SPLIT_RATIOS,
            "n_decode_seeds": args.decode_seeds,
            "decode_seeds": decode_seeds,
            "uncertainty_sources": {
                "split_layer": "test-composition uncertainty: N "
                               "class-stratified bootstrap evaluation "
                               "partitions from the untouched original "
                               "test set; train/validation and all "
                               "learned artifacts remain fixed",
                "decode_seed_layer": "LLM decoding stochasticity: "
                                     "config.RANDOM_SEED reseeded per "
                                     "evaluation (rerun_r1_aggregate "
                                     "pattern); averaged out within "
                                     "each split before pairing",
            },
            "seed_plumbing_note": "Methods do not currently expose a "
                                  "seed kwarg on evaluate(); seeds are "
                                  "plumbed via config.RANDOM_SEED + "
                                  "np.random.seed before construction. "
                                  "A seed kwarg is auto-detected and "
                                  "passed through if added later.",
            "paired_generation_note": "When methods are exactly "
                                      "cl_llmops and agent_nov, both "
                                      "arms share each alarm, sampled "
                                      "hypothesis set, and retrieved "
                                      "evidence. Only verification/gating "
                                      "and the resulting action path differ.",
            "statistics": {
                "test": "paired two-sided Wilcoxon signed-rank on "
                        "split-level means (N pairs), CL-LLMOps minus "
                        "baseline",
                "correction": "Holm step-down across all "
                              "dataset x metric x baseline comparisons",
                "preregistered_metric_family": list(INFERENCE_METRICS),
                "n_comparisons_in_family": n_comparisons,
                "ci": f"percentile bootstrap 95% CI on the mean paired "
                      f"difference, {args.bootstrap} resamples, "
                      f"seed {args.bootstrap_seed}",
                "p_value_floor_note": f"With N={args.n_splits} pairs and "
                                      f"no ties, the exact two-sided "
                                      f"Wilcoxon p-value cannot fall "
                                      f"below 2/2^N = {p_floor:.4f}; "
                                      f"Holm-adjusted values inherit "
                                      f"this floor.",
                "p_value_floor": p_floor,
            },
        },
        "raw_metrics": raw,
        "comparisons": comparisons,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
