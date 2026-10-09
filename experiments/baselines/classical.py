"""Classical ML baselines: B1 (SVM) and B2 (XGBoost).

These baselines operate on hand-crafted features extracted by FeatureExtractor.
They produce only a classification label (no explanations or actions).
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from itertools import product
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, RANDOM_SEED, SVM_MAX_TRAIN


def _stratified_cap(X, y, cap, seed=RANDOM_SEED):
    """Stratified subsample to at most `cap` rows (SVM kernel cost control)."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    keep = []
    for cls, cnt in zip(classes, counts):
        idx = np.flatnonzero(y == cls)
        quota = max(1, int(round(cap * cnt / len(y))))
        if len(idx) > quota:
            idx = rng.choice(idx, size=quota, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


class SVMBaseline:
    """B1: SVM with RBF kernel on hand-crafted features.

    Training rows are capped at config.SVM_MAX_TRAIN via stratified
    subsampling; the RBF kernel is quadratic in n, so the full pq16
    training split (168K windows) would be infeasible. Evaluation always
    uses the full test split. The cap must be disclosed in the paper's
    implementation details.
    """

    def __init__(self, C=10.0, gamma="scale"):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, kernel="rbf", gamma=gamma,
                        random_state=RANDOM_SEED, probability=True)),
        ])

    def train(self, X_train, y_train):
        if len(X_train) > SVM_MAX_TRAIN:
            X_train, y_train = _stratified_cap(
                X_train, y_train, SVM_MAX_TRAIN)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        proba = self.predict_proba(X_test)
        # Top-3 accuracy
        top3 = _topk_accuracy(proba, y_test, k=3)
        return {"accuracy": acc, "f1": f1, "top3_accuracy": top3}

    def save(self, path=None):
        path = path or MODEL_DIR / "svm_baseline.pkl"
        joblib.dump(self.model, path)

    def load(self, path=None):
        path = path or MODEL_DIR / "svm_baseline.pkl"
        self.model = joblib.load(path)


class XGBoostBaseline:
    """B2: XGBoost on hand-crafted features."""

    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=RANDOM_SEED,
                use_label_encoder=False,
                eval_metric="mlogloss",
                tree_method="hist",
            )),
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        proba = self.predict_proba(X_test)
        top3 = _topk_accuracy(proba, y_test, k=3)
        return {"accuracy": acc, "f1": f1, "top3_accuracy": top3}

    def save(self, path=None):
        path = path or MODEL_DIR / "xgboost_baseline.pkl"
        joblib.dump(self.model, path)

    def load(self, path=None):
        path = path or MODEL_DIR / "xgboost_baseline.pkl"
        self.model = joblib.load(path)


class ANFISBaseline:
    """Compact first-order Takagi--Sugeno neuro-fuzzy classifier.

    Four standardized PCA inputs and three Gaussian memberships per input
    yield 81 fuzzy rules.  Antecedent centers/spreads are initialized from
    train-only quantiles; normalized firing strengths times affine inputs form
    the consequent design matrix, whose multiclass consequents are fitted by
    L2-regularized least squares (RidgeClassifier).
    """

    def __init__(self, n_components=4, n_memberships=3, alpha=1.0):
        self.n_components = n_components
        self.n_memberships = n_memberships
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        self.classifier = RidgeClassifier(alpha=alpha)
        self.centers = None
        self.sigmas = None
        self.rules = None

    def _transform_inputs(self, X, fit=False):
        if fit:
            z = self.scaler.fit_transform(X)
            n_components = min(self.n_components, z.shape[1], len(z) - 1)
            self.pca = PCA(n_components=n_components,
                           random_state=RANDOM_SEED)
            return self.pca.fit_transform(z)
        return self.pca.transform(self.scaler.transform(X))

    def _initialize_antecedents(self, z):
        quantiles = np.linspace(0.1, 0.9, self.n_memberships)
        self.centers = np.stack([
            np.quantile(z[:, j], quantiles) for j in range(z.shape[1])
        ])
        self.sigmas = np.empty_like(self.centers)
        for j in range(z.shape[1]):
            spacing = np.diff(self.centers[j])
            fallback = max(float(np.std(z[:, j])), 1e-3)
            for m in range(self.n_memberships):
                adjacent = []
                if m > 0:
                    adjacent.append(spacing[m - 1])
                if m < self.n_memberships - 1:
                    adjacent.append(spacing[m])
                self.sigmas[j, m] = max(
                    float(np.mean(adjacent)) if adjacent else fallback,
                    0.1 * fallback, 1e-3)
        self.rules = np.asarray(list(product(
            range(self.n_memberships), repeat=z.shape[1])), dtype=int)

    def _consequent_features(self, z):
        # Log-domain firing avoids underflow in the product t-norm.
        log_firing = np.zeros((len(z), len(self.rules)), dtype=np.float64)
        for dim in range(z.shape[1]):
            centers = self.centers[dim, self.rules[:, dim]][None, :]
            sigmas = self.sigmas[dim, self.rules[:, dim]][None, :]
            log_firing += -0.5 * ((z[:, dim, None] - centers) / sigmas) ** 2
        log_firing -= log_firing.max(axis=1, keepdims=True)
        firing = np.exp(log_firing)
        firing /= firing.sum(axis=1, keepdims=True) + 1e-12
        affine = np.concatenate([np.ones((len(z), 1)), z], axis=1)
        return (firing[:, :, None] * affine[:, None, :]).reshape(len(z), -1)

    def train(self, X_train, y_train):
        z = self._transform_inputs(X_train, fit=True)
        self._initialize_antecedents(z)
        self.classifier.fit(self._consequent_features(z), y_train)

    def predict(self, X):
        z = self._transform_inputs(X)
        return self.classifier.predict(self._consequent_features(z))

    def predict_proba(self, X):
        z = self._transform_inputs(X)
        scores = self.classifier.decision_function(
            self._consequent_features(z))
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds, average="macro")),
            "top3_accuracy": _topk_accuracy(
                self.predict_proba(X_test), y_test, k=3),
        }


def _topk_accuracy(proba, y_true, k=3):
    """Compute Top-K accuracy from probability matrix."""
    k = min(int(k), proba.shape[1])
    topk_preds = np.argsort(proba, axis=1)[:, -k:]
    correct = sum(1 for i, y in enumerate(y_true) if y in topk_preds[i])
    return correct / len(y_true)
