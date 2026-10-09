"""Evidence-driven Mamdani fuzzy gate (paper Section 4.4).

Inputs:
  * verification score ``v``;
  * retrieved-evidence coverage ``coverage``;
  * contradiction severity ``contradiction``.

The eleven wildcard rules are the executable counterpart of Table 4.  Low
coverage means ``defer`` (collect evidence), severe contradiction means
``reject``/escalate to a human, and the fuzzy middle confidence band is the
explicit escalation region.  ``lambda_r`` is retained on the object because
the pipeline uses it to blend the Layer-1 rule score with CAEVN support.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    FIS_PROCEED_THRESHOLD, FIS_DEFER_THRESHOLD,
    FIS_HIGH_RISK_MARGIN, RULE_NEURAL_WEIGHT,
    FIS_EVIDENCE_COVERAGE_THRESHOLD,
)


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return np.maximum(0, np.minimum((x - a) / max(b - a, 1e-8),
                                     (c - x) / max(c - b, 1e-8)))


def trapmf(x, a, b, c, d):
    """Trapezoid with correct degenerate shoulder behavior."""
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / max(b - a, 1e-8),
                    (x >= a).astype(float))
    right = np.where(d > c, (d - x) / max(d - c, 1e-8),
                     (x <= d).astype(float))
    return np.maximum(0, np.minimum(np.minimum(left, 1.0), right))


class ScoreCoverageMembership:
    """Paper breakpoints shared by score and evidence coverage."""

    def __init__(self, shift=0.0):
        self.shift = float(shift)

    def _p(self, value):
        return float(np.clip(value + self.shift, 0.0, 1.0))

    def low(self, x):
        return trapmf(x, 0.0, 0.0, self._p(0.3), self._p(0.5))

    def medium(self, x):
        return trimf(x, self._p(0.3), self._p(0.55), self._p(0.8))

    def high(self, x):
        return trapmf(x, self._p(0.6), self._p(0.8), 1.0, 1.0)


class ContradictionMembership:
    """None/mild/severe breakpoints from paper Section 4.4."""

    def none(self, x):
        return trapmf(x, 0.0, 0.0, 0.05, 0.15)

    def mild(self, x):
        return trimf(x, 0.05, 0.2, 0.4)

    def severe(self, x):
        return trapmf(x, 0.25, 0.4, 1.0, 1.0)


class OutputMembership:
    def __init__(self, n_points=201):
        self.universe = np.linspace(0, 1, n_points)

    def low(self):
        return trapmf(self.universe, 0.0, 0.0, 0.2, 0.4)

    def medium(self):
        return trimf(self.universe, 0.25, 0.5, 0.75)

    def high(self):
        return trapmf(self.universe, 0.6, 0.8, 1.0, 1.0)


class MamdaniFIS:
    """Eleven-rule, three-input Mamdani gate.

    Wildcard antecedents are represented as ``None``.  Decision strings keep
    the existing pipeline convention: ``reject`` means human escalation,
    while ``defer`` means gather more evidence and monitor.
    """

    def __init__(self, alpha_p=FIS_PROCEED_THRESHOLD,
                 alpha_d=FIS_DEFER_THRESHOLD,
                 delta=FIS_HIGH_RISK_MARGIN,
                 lambda_r=RULE_NEURAL_WEIGHT,
                 coverage_defer_threshold=FIS_EVIDENCE_COVERAGE_THRESHOLD,
                 breakpoint_shift=0.0):
        self.score_source = "integrated"
        self.alpha_p = alpha_p
        self.alpha_d = alpha_d
        self.delta = delta
        self.lambda_r = lambda_r
        self.coverage_defer_threshold = coverage_defer_threshold
        self.breakpoint_shift = float(breakpoint_shift)
        self.score_mf = ScoreCoverageMembership(self.breakpoint_shift)
        self.coverage_mf = ScoreCoverageMembership(self.breakpoint_shift)
        self.contradiction_mf = ContradictionMembership()
        self.output_mf = OutputMembership()

        # (score, coverage, contradiction, confidence); None = wildcard.
        self.rules = [
            ("high", "high", "none", "high"),       # R1
            ("high", "high", "mild", "medium"),    # R2
            ("high", "medium", "none", "high"),    # R3
            ("high", "low", None, "low"),           # R4
            ("medium", "high", "none", "medium"),  # R5
            ("medium", "medium", "none", "medium"),# R6
            ("medium", None, "mild", "low"),        # R7
            ("medium", "low", "none", "low"),      # R8
            ("high", "medium", "mild", "medium"),  # R9
            ("low", None, None, "low"),              # R10
            (None, None, "severe", "low"),           # R11
        ]

    @staticmethod
    def _membership(mf_set, value, level):
        return 1.0 if level is None else float(getattr(mf_set, level)(value))

    def infer(self, v, coverage=None, contradiction=None, c=None, r=None,
              high_risk=False):
        """Infer a gate decision.

        ``c``/``r`` are accepted as compatibility aliases for older callers;
        they now mean coverage/contradiction, not self-consistency/reliability.
        """
        if coverage is None:
            coverage = c
        if contradiction is None:
            contradiction = r
        if coverage is None or contradiction is None:
            raise ValueError("coverage and contradiction are required")
        v = float(np.clip(v, 0.0, 1.0))
        coverage = float(np.clip(coverage, 0.0, 1.0))
        contradiction = float(np.clip(contradiction, 0.0, 1.0))

        output_mfs = {
            "low": self.output_mf.low(),
            "medium": self.output_mf.medium(),
            "high": self.output_mf.high(),
        }
        aggregated = np.zeros_like(self.output_mf.universe)
        activations = []
        for s_level, rho_level, con_level, consequent in self.rules:
            mu_s = self._membership(self.score_mf, v, s_level)
            mu_rho = self._membership(
                self.coverage_mf, coverage, rho_level)
            mu_con = self._membership(
                self.contradiction_mf, contradiction, con_level)
            firing = min(mu_s, mu_rho, mu_con)
            if firing > 0:
                aggregated = np.maximum(
                    aggregated, np.minimum(firing,
                                           output_mfs[consequent]))
            activations.append({
                "antecedent": {"score": s_level,
                               "coverage": rho_level,
                               "contradiction": con_level},
                "consequent": consequent,
                "firing": firing,
            })

        if aggregated.sum() < 1e-10:
            alpha = 0.5
        else:
            alpha = float(np.sum(self.output_mf.universe * aggregated)
                          / np.sum(aggregated))

        proceed_threshold = self.alpha_p + (self.delta if high_risk else 0.0)
        severe_mu = float(self.contradiction_mf.severe(contradiction))
        if coverage < self.coverage_defer_threshold:
            decision, reason = "defer", "insufficient_evidence"
        elif severe_mu >= 0.5:
            decision, reason = "reject", "severe_contradiction"
        elif alpha >= proceed_threshold:
            decision, reason = "proceed", "high_confidence"
        elif alpha < self.alpha_d:
            decision, reason = "defer", "low_confidence"
        else:
            decision, reason = "reject", "ambiguity_escalation"

        return decision, alpha, {
            "rule_activations": activations,
            "aggregated_output": aggregated,
            "raw_alpha": alpha,
            "score": v,
            "coverage": coverage,
            "contradiction": contradiction,
            "high_risk": bool(high_risk),
            "proceed_threshold": proceed_threshold,
            "decision_reason": reason,
        }

    def batch_infer(self, vs, coverages, contradictions, high_risks=None):
        if high_risks is None:
            high_risks = [False] * len(vs)
        decisions, alphas = [], []
        for v, coverage, contradiction, high_risk in zip(
                vs, coverages, contradictions, high_risks):
            decision, alpha, _ = self.infer(
                v=v, coverage=coverage, contradiction=contradiction,
                high_risk=high_risk)
            decisions.append(decision)
            alphas.append(alpha)
        return decisions, np.asarray(alphas)
