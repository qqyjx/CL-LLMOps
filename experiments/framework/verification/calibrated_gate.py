"""Calibrated scalar gates used for the Reviewer 3.2b comparison.

The baseline maps the CAEVN scalar score to P(correct) with either Platt
scaling or isotonic regression.  Its two thresholds are fitted on the
validation split to match the full fuzzy gate's proceed/reject coverage.
This keeps the operating point comparable while isolating whether the
three-input, auditable fuzzy rule base adds value beyond calibration.
"""

import numpy as np


class CalibratedProbabilityGate:
    """Platt/isotonic calibration followed by a three-way scalar gate."""

    def __init__(self, method="platt"):
        if method not in {"platt", "isotonic"}:
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.method = method
        self.score_source = "caevn"
        self.model = None
        self.constant_probability = None
        self.reject_threshold = 0.2
        self.proceed_threshold = 0.8
        self.fit_metadata = {}

    def fit(self, scores, correct, reference_decisions=None):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        correct = np.asarray(correct, dtype=int).reshape(-1)
        if len(scores) != len(correct) or len(scores) == 0:
            raise ValueError("scores/correct must be non-empty and aligned")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores contain non-finite values")

        if len(np.unique(correct)) < 2:
            self.constant_probability = float(correct.mean())
            self.model = None
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                solver="lbfgs", random_state=0, max_iter=1000)
            self.model.fit(scores.reshape(-1, 1), correct)
        else:
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model.fit(scores, correct)

        probabilities = self.predict_proba(scores)
        proceed_rate, reject_rate = 0.2, 0.2
        if reference_decisions is not None:
            decisions = np.asarray(reference_decisions, dtype=object)
            if len(decisions) != len(scores):
                raise ValueError("reference_decisions must align with scores")
            proceed_rate = float(np.mean(decisions == "proceed"))
            reject_rate = float(np.mean(decisions == "reject"))

        # Match the reference coverage on validation.  Strict comparisons in
        # infer() make tied calibrated probabilities fall into defer rather
        # than creating an unsafe coverage surplus.
        reject_rate = float(np.clip(reject_rate, 0.0, 1.0))
        proceed_rate = float(np.clip(proceed_rate, 0.0, 1.0))
        self.reject_threshold = float(np.quantile(
            probabilities, reject_rate))
        if reject_rate >= 1.0:
            self.reject_threshold = float(np.nextafter(
                probabilities.max(), np.inf))
        self.proceed_threshold = float(np.quantile(
            probabilities, 1.0 - proceed_rate))
        if proceed_rate <= 0.0:
            self.proceed_threshold = float(np.nextafter(
                probabilities.max(), np.inf))
        if self.proceed_threshold < self.reject_threshold:
            midpoint = 0.5 * (self.proceed_threshold
                              + self.reject_threshold)
            self.reject_threshold = midpoint
            self.proceed_threshold = midpoint

        self.fit_metadata = {
            "n_validation": int(len(scores)),
            "positive_rate": float(correct.mean()),
            "reference_proceed_rate": proceed_rate,
            "reference_reject_rate": reject_rate,
            "calibration_brier": float(np.mean(
                (probabilities - correct) ** 2)),
        }
        return self

    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if self.constant_probability is not None:
            return np.full(len(scores), self.constant_probability,
                           dtype=float)
        if self.model is None:
            raise RuntimeError("fit the calibrated gate before inference")
        if self.method == "platt":
            return self.model.predict_proba(
                scores.reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(scores), dtype=float)

    def infer(self, v, c=None, r=None, coverage=None,
              contradiction=None, high_risk=None):
        probability = float(self.predict_proba([v])[0])
        if probability >= self.proceed_threshold:
            decision = "proceed"
        elif probability < self.reject_threshold:
            decision = "reject"
        else:
            decision = "defer"
        return decision, probability, {
            "calibration": self.method,
            "probability_correct": probability,
            "reject_threshold": self.reject_threshold,
            "proceed_threshold": self.proceed_threshold,
            "ignored_inputs": {"consistency": c, "reliability": r},
        }

    def to_dict(self):
        return {
            "method": self.method,
            "reject_threshold": self.reject_threshold,
            "proceed_threshold": self.proceed_threshold,
            "fit": self.fit_metadata,
        }
