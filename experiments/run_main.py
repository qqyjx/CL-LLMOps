"""Main experiment runner: Table 2 (main results across 3 datasets).

Runs all 8 methods on PQ/CWRU/CARE datasets and outputs results.json.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from config import (
    DATASETS, RANDOM_SEED, RESULTS_JSON, MODEL_DIR, DEVICE,
)
from data import load_splits
from features import FeatureExtractor
from baselines.classical import SVMBaseline, XGBoostBaseline
from baselines.classical import ANFISBaseline


FEATURE_ALIAS = {"pq16": "pq"}


def load_dataset(name):
    """Load and split dataset."""
    return load_splits(name)


def run_classical_baselines(dataset_name, X_train, y_train, X_test, y_test):
    """Run SVM and XGBoost baselines on extracted features."""
    extractor = FeatureExtractor(FEATURE_ALIAS.get(dataset_name,
                                                    dataset_name))
    print(f"  Extracting features for {dataset_name}...")

    X_train_feat = extractor.extract_batch(X_train)
    X_test_feat = extractor.extract_batch(X_test)

    results = {}

    print("  Running SVM...")
    svm = SVMBaseline()
    svm.train(X_train_feat, y_train)
    results["svm"] = svm.evaluate(X_test_feat, y_test)
    svm.save(MODEL_DIR / f"svm_{dataset_name}.pkl")

    print("  Running XGBoost...")
    xgb = XGBoostBaseline()
    xgb.train(X_train_feat, y_train)
    results["xgboost"] = xgb.evaluate(X_test_feat, y_test)
    xgb.save(MODEL_DIR / f"xgboost_{dataset_name}.pkl")

    print("  Running ANFIS...")
    anfis = ANFISBaseline()
    anfis.train(X_train_feat, y_train)
    results["anfis"] = anfis.evaluate(X_test_feat, y_test)

    return results


def run_deep_baselines(dataset_name, X_train, y_train, X_val, y_val,
                       X_test, y_test):
    """Run CNN1D and Transformer baselines on raw signals."""
    from baselines.deep_learning import (
        CNN1D, TransformerClassifier, DeepLearningTrainer,
    )

    n_channels = X_train.shape[1]
    n_classes = DATASETS[dataset_name]["n_classes"]
    seq_len = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]

    results = {}
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print("  Running CNN1D...")
    cnn = CNN1D(in_channels=n_channels, n_classes=n_classes, seq_len=seq_len)
    trainer = DeepLearningTrainer(
        cnn, model_name=f"cnn_{dataset_name}", device=DEVICE)
    trainer.train(X_train, y_train, X_val, y_val)
    results["cnn1d"] = trainer.evaluate(X_test, y_test)

    print("  Running Transformer...")
    tfm = TransformerClassifier(
        in_channels=n_channels, n_classes=n_classes, seq_len=seq_len)
    trainer = DeepLearningTrainer(
        tfm, model_name=f"transformer_{dataset_name}", device=DEVICE)
    trainer.train(X_train, y_train, X_val, y_val)
    results["transformer"] = trainer.evaluate(X_test, y_test)

    return results


def run_llm_baselines(dataset_name, X_test, y_test):
    """Run LLM baselines (FD-LLM, RAG-LLM, Agent-noV, CL-LLMOps) via vLLM."""
    from baselines.llm_baselines import (
        FDLLMBaseline, RAGLLMBaseline, AgentNoVerifyBaseline,
    )
    from framework.pipeline import CLLLMOpsPipeline

    results = {}

    print("  Running FD-LLM...")
    fdllm = FDLLMBaseline(dataset_name=dataset_name)
    results["fdllm"] = fdllm.evaluate(X_test, y_test)

    print("  Running RAG-LLM...")
    ragllm = RAGLLMBaseline(dataset_name=dataset_name)
    results["ragllm"] = ragllm.evaluate(X_test, y_test)

    print("  Running Agent (no verification)...")
    agent_nov = AgentNoVerifyBaseline(dataset_name=dataset_name)
    results["agent_nov"] = agent_nov.evaluate(X_test, y_test)

    print("  Running CL-LLMOps (ours)...")
    pipeline = CLLLMOpsPipeline(dataset_name=dataset_name)
    results["cl_llmops"] = pipeline.evaluate(X_test, y_test)

    return results


def main():
    parser = argparse.ArgumentParser(description="CL-LLMOps Main Experiments")
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"],
                        help="Datasets to evaluate")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip deep learning baselines (CPU mode)")
    parser.add_argument("--include-legacy-llm", action="store_true",
                        help="Run the historical direct/RAG LLM harness; "
                             "the R1 SOTA runner is preferred")
    args = parser.parse_args()

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {DATASETS[ds_name]['name']}")
        print(f"{'='*60}")

        splits = load_dataset(ds_name)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        ds_results = {}

        classical = run_classical_baselines(
            ds_name, X_train, y_train, X_test, y_test)
        ds_results.update(classical)

        if not args.skip_dl:
            deep = run_deep_baselines(
                ds_name, X_train, y_train, X_val, y_val, X_test, y_test)
            ds_results.update(deep)

        if args.include_legacy_llm:
            llm = run_llm_baselines(ds_name, X_test, y_test)
            ds_results.update(llm)

        all_results[ds_name] = ds_results
        print(f"\nResults for {ds_name}:")
        for method, metrics in ds_results.items():
            if isinstance(metrics, dict):
                acc = metrics.get("accuracy", "N/A")
                f1 = metrics.get("f1", "N/A")
                print(f"  {method:15s}  acc={acc}  f1={f1}")

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "datasets": args.datasets,
            "random_seed": RANDOM_SEED,
        },
        "results": all_results,
    }

    output_path = RESULTS_JSON.parent / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
