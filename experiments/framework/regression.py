"""Stage 5: Regression Testing.

Validates that the proposed action does not degrade performance on
a curated test set of known fault-action pairs. This is the "closed-loop"
mechanism that distinguishes CL-LLMOps from open-loop LLM agents.

Regression pass criterion: accuracy on test set >= threshold (default 95%).
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    REGRESSION_PASS_THRESHOLD, REGRESSION_CYCLES, INITIAL_TEST_SET_SIZE,
)


class RegressionTester:
    """Regression testing module for closed-loop validation."""

    def __init__(self, test_cases=None, threshold=REGRESSION_PASS_THRESHOLD):
        """
        Args:
            test_cases: list of (input, expected_output) tuples
            threshold: minimum accuracy to pass regression
        """
        self.test_cases = test_cases or []
        self.threshold = threshold
        self.history = []  # track pass/fail over time

    def add_test_case(self, input_data, expected_output):
        """Add a new test case to the regression suite."""
        self.test_cases.append((input_data, expected_output))

    def build_initial_suite(self, dataset, labels, n=INITIAL_TEST_SET_SIZE):
        """Build initial regression suite from stratified sampling."""
        rng = np.random.RandomState(42)
        classes = np.unique(labels)
        per_class = max(1, n // len(classes))

        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            selected = rng.choice(cls_indices,
                                  size=min(per_class, len(cls_indices)),
                                  replace=False)
            for idx in selected:
                self.test_cases.append((dataset[idx], int(labels[idx])))

    def run(self, pipeline_fn):
        """Run regression test suite.

        Args:
            pipeline_fn: callable that takes input and returns prediction

        Returns:
            dict with: passed, accuracy, n_tests, failures
        """
        if not self.test_cases:
            return {"passed": True, "accuracy": 1.0, "n_tests": 0, "failures": []}

        correct = 0
        failures = []

        for i, (input_data, expected) in enumerate(self.test_cases):
            try:
                result = pipeline_fn(input_data)
                predicted = result.get("fault_class", result) if isinstance(result, dict) else result
                if predicted == expected:
                    correct += 1
                else:
                    failures.append({
                        "index": i,
                        "expected": expected,
                        "predicted": predicted,
                    })
            except Exception as e:
                failures.append({"index": i, "expected": expected, "error": str(e)})

        accuracy = correct / len(self.test_cases)
        passed = accuracy >= self.threshold

        result = {
            "passed": passed,
            "accuracy": accuracy,
            "n_tests": len(self.test_cases),
            "failures": failures[:10],  # limit failure details
        }
        self.history.append(result)
        return result

    def run_cycles(self, pipeline_fn, n_cycles=REGRESSION_CYCLES):
        """Run multiple regression cycles to check stability.

        Returns:
            dict with: all_passed, cycle_results, stability_score
        """
        cycle_results = []
        for _ in range(n_cycles):
            result = self.run(pipeline_fn)
            cycle_results.append(result)

        accuracies = [r["accuracy"] for r in cycle_results]
        all_passed = all(r["passed"] for r in cycle_results)

        return {
            "all_passed": all_passed,
            "cycle_results": cycle_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "stability_score": 1.0 - float(np.std(accuracies)),
        }
