"""Stage 1: Alarm Classification.

Classifies incoming sensor data into fault categories using
either classical ML features or deep learning on raw signals.
Outputs: fault_class, confidence, feature_vector.
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASETS


class AlarmClassifier:
    """Alarm stage: detect and classify faults from sensor data."""

    def __init__(self, classifier, feature_extractor=None, dataset_name="pq"):
        """
        Args:
            classifier: trained model with predict() and predict_proba()
            feature_extractor: FeatureExtractor instance (None for DL models)
            dataset_name: one of "pq", "cwru", "care"
        """
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.dataset_config = DATASETS[dataset_name]
        self.class_names = self._get_class_names(dataset_name)

    def _get_class_names(self, dataset_name):
        if dataset_name == "pq":
            return [
                "Normal", "Sag", "Swell", "Interruption", "Harmonics",
                "Flicker", "Notch", "Spike", "Sag+Harmonics", "Swell+Harmonics",
                "Sag+Flicker", "Swell+Flicker", "Harmonics+Flicker",
                "Interruption+Harmonics", "Sag+Notch", "Swell+Notch",
            ]
        elif dataset_name == "cwru":
            return [
                "Normal", "IR007", "IR014", "IR021",
                "OR007", "OR014", "OR021",
                "Ball007", "Ball014", "Ball021",
            ]
        else:  # care
            return ["Normal", "Pre-fault"]

    def classify(self, signal):
        """Classify a single signal window.

        Args:
            signal: raw signal array (channels x samples) or feature vector

        Returns:
            dict with keys: fault_class, class_name, confidence, top3, features
        """
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract(signal)
            X = features.reshape(1, -1)
        else:
            X = signal.reshape(1, *signal.shape) if signal.ndim < 3 else signal

        pred = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]

        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [(int(idx), self.class_names[idx], float(proba[idx]))
                for idx in top3_idx]

        return {
            "fault_class": int(pred),
            "class_name": self.class_names[pred],
            "confidence": float(proba[pred]),
            "top3": top3,
            "features": features if self.feature_extractor else None,
        }

    def batch_classify(self, signals):
        """Classify a batch of signals."""
        return [self.classify(s) for s in signals]
