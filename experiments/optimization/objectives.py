"""MOPSO objective functions for CL-LLMOps hyperparameter optimization.

Three objectives (all to be MINIMIZED):
  f1: -F1_macro (negative because we maximize F1)
  f2: Hallucination rate (minimize)
  f3: Human handoff rate 1 - proceed_rate (minimize)

f3 changed from false_escalation_rate on 2026-07-22: false escalation
only penalizes escalating gold-A0/A1 episodes, so "escalate everything"
was Pareto-optimal (f1 falls back to the alarm prediction on non-proceed
episodes and hallucination is trivially 0 when nothing proceeds) — the
07-22 step1 run collapsed to alpha_p=0.9 with proceed_rate=0 and a
24-copy degenerate front. Handoff rate makes the safety-vs-autonomy
trade-off real; false_escalation_rate remains a reported metric.

Search space (6 dimensions from paper Table 3):
  x[0]: alpha_p (FIS proceed threshold) in [0.5, 0.9]
  x[1]: alpha_d (FIS defer threshold) in [0.1, 0.5]
  x[2]: delta (high-risk margin) in [0.0, 0.2]
  x[3]: lambda_r (neural rule weight) in [0.1, 0.7]
  x[4]: M (self-consistency samples) in {3, 5, 7, 9}  (discretized)
  x[5]: beta (entropy regularization) in [0.001, 0.1]
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED

BOUNDS_LOWER = np.array([0.5, 0.1, 0.0, 0.1, 3.0, 0.001])
BOUNDS_UPPER = np.array([0.9, 0.5, 0.2, 0.7, 9.0, 0.1])
N_OBJECTIVES = 3
N_DIMS = 6
DISCRETE_DIMS = {4: [3, 5, 7, 9]}


def discretize_particle(x):
    """Snap discrete dimensions to valid values."""
    x_out = x.copy()
    for dim, valid_vals in DISCRETE_DIMS.items():
        idx = np.argmin(np.abs(np.array(valid_vals) - x[dim]))
        x_out[dim] = valid_vals[idx]
    return x_out


def constraint_check(x):
    """Feasibility: proceed threshold above defer band, and the
    high-risk-shifted threshold alpha_p + delta must stay below 1 so
    high-risk episodes remain gateable at all (alpha is in [0, 1])."""
    return x[0] > x[1] + x[2] and x[0] + x[2] <= 0.95


class PipelineObjective:
    """Wraps the CL-LLMOps pipeline as a MOPSO objective function."""

    def __init__(self, pipeline=None, dataset=None):
        if pipeline is None:
            from framework.pipeline import CLLLMOpsPipeline
            pipeline = CLLLMOpsPipeline(dataset_name=dataset or "pq")
        self.pipeline = pipeline
        self.dataset = dataset
        self.rng = np.random.RandomState(RANDOM_SEED)
        self._call_count = 0

    def __call__(self, x):
        """Evaluate objectives for a single particle.

        Args:
            x: (N_DIMS,) parameter vector
        Returns:
            (N_OBJECTIVES,) objective values [neg_f1, halluc, false_esc]
        """
        x = discretize_particle(x)

        if not constraint_check(x):
            return np.array([1.0, 1.0, 1.0])

        self._call_count += 1
        alpha_p, alpha_d, delta, lambda_r, M, beta = x

        metrics = self.pipeline.evaluate_with_params(
            alpha_p=float(alpha_p),
            alpha_d=float(alpha_d),
            delta=float(delta),
            lambda_r=float(lambda_r),
            M=int(M),
            beta=float(beta),
        )
        return np.array([
            -metrics["f1"],
            metrics["hallucination_rate"],
            1.0 - metrics["proceed_rate"],
        ])
