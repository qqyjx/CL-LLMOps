"""CL-LLMOps Pipeline: 5-stage closed-loop fault diagnosis.

Stage 1: Alarm    -> classify fault from sensor data
Stage 2: Hypothesis -> LLM generates diagnostic hypothesis (M samples)
Stage 3: Verification -> CAEVN + FIS gate the hypothesis
Stage 4: Action   -> LLM generates maintenance action
Stage 5: Regression -> validate action doesn't break existing cases

Two construction modes:
  * component mode (legacy): pass instantiated stage components
    positionally, then call run()/batch_run() per signal;
  * dataset mode: CLLLMOpsPipeline(dataset_name=..., ablation_variant=...,
    llm_client=...) builds all components lazily on first use (alarm
    classifier trained on the train split, KB from
    framework.knowledge_base, CAEVN loaded from
    checkpoints/caevn_{dataset}.pt) and additionally exposes
    evaluate(X, y) and evaluate_with_params(...) for the experiment
    runners (run_main / run_ablation / run_statistical_protocol /
    runbooks.rerun_r1_aggregate / optimization.objectives).

Honesty contract: every metric returned by evaluate*() is computed by
experiments/evaluation/ from actual per-episode stage outputs; the
LLM client is the ONLY difference between local (mock) and server
(vLLM) runs.
"""

import os
import re
import json
import time
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.alarm import AlarmClassifier
from framework.hypothesis import HypothesisGenerator
from framework.action import ActionGenerator
from framework.explanation import ExplanationGenerator
from framework.regression import RegressionTester
from framework.verification.caevn import CAEVNTrainer, MLPVerifier
from framework.verification.fuzzy_gating import MamdaniFIS
from config import (
    SELF_CONSISTENCY_M, SENTENCE_MODEL, MODEL_DIR, EVIDENCE_TOP_K,
    ALARM_TRAIN_CAP, PIPELINE_EVAL_MAX_EPISODES, MOO_EVAL_SAMPLES,
    MOO_MAX_M, REGRESSION_STRIDE, REGRESSION_SUITE_SIZE,
    CAEVN_ENTROPY_BETA,
    OPERATING_POINT_JSON, EXPLAIN_AUDIT,
)

# Feature/DATASETS config alias (pq16 shares the PQ signal layout);
# mirrors run_statistical_protocol.FEATURE_CONFIG_ALIAS.
DATASET_ALIAS = {"pq16": "pq"}

ABLATION_VARIANTS = (
    "full", "no_verifier", "no_gating", "no_self_consistency",
    "no_regression", "rule_only_verifier", "model_only_verifier",
    "mlp_verifier", "hard_gating", "grid_search",
)

# Coarse-grid fallback operating point for the grid_search ablation
# (paper Table: MOPSO vs grid search). The grid itself is executed by
# run_mopso.py --grid on the server; this pins the resulting corner.
GRID_SEARCH_FIS_PARAMS = {
    "alpha_p": 0.75, "alpha_d": 0.35, "delta": 0.05, "lambda_r": 0.3,
}

_ENCODER_SINGLETON = {}


def get_sentence_encoder():
    """Shared frozen all-MiniLM-L6-v2 encoder (loaded once)."""
    if "enc" not in _ENCODER_SINGLETON:
        from sentence_transformers import SentenceTransformer
        _ENCODER_SINGLETON["enc"] = SentenceTransformer(SENTENCE_MODEL)
    return _ENCODER_SINGLETON["enc"]


class _VectorExtractor:
    """Adapter: FeatureExtractor.extract returns (dict, vector); the
    AlarmClassifier expects a bare vector."""

    def __init__(self, extractor):
        self.extractor = extractor

    def extract(self, signal):
        _, vec = self.extractor.extract(signal)
        return vec

    def extract_batch(self, signals):
        return self.extractor.extract_batch(signals)


def build_alarm_classifier(dataset_name, splits, seed):
    """Train the Stage-1 alarm classifier on the train split.

    XGBoost on hand-crafted features (baselines/classical.py), with a
    stratified cap of ALARM_TRAIN_CAP samples to bound feature
    extraction cost on the largest dataset. Deterministic given seed.
    """
    from features import FeatureExtractor
    from baselines.classical import XGBoostBaseline
    from evaluation.action_rules import get_class_names

    alias = DATASET_ALIAS.get(dataset_name, dataset_name)
    extractor = _VectorExtractor(FeatureExtractor(alias))

    X_train, y_train = splits["X_train"], splits["y_train"]
    rng = np.random.RandomState(seed)
    if len(y_train) > ALARM_TRAIN_CAP:
        keep = []
        classes = np.unique(y_train)
        per_class = max(2, ALARM_TRAIN_CAP // len(classes))
        for cls in classes:
            idx = np.where(y_train == cls)[0]
            keep.append(rng.choice(idx, size=min(per_class, len(idx)),
                                   replace=False))
        keep = np.concatenate(keep)
        X_train, y_train = X_train[keep], y_train[keep]

    X_feat = extractor.extract_batch(X_train)
    clf = XGBoostBaseline()
    clf.train(X_feat, y_train)

    alarm = AlarmClassifier(clf, feature_extractor=extractor,
                            dataset_name=alias)
    # pq16 shares the pq config but has its own class naming
    alarm.class_names = get_class_names(dataset_name)
    return alarm


class CLLLMOpsPipeline:
    """Complete CL-LLMOps pipeline with 5 closed-loop stages."""

    def __init__(self, alarm_classifier=None, hypothesis_generator=None,
                 caevn_trainer=None, fis=None, action_generator=None,
                 regression_tester=None, knowledge_base=None,
                 dataset_name=None, ablation_variant=None,
                 llm_client=None, encoder=None, seed=None,
                 caevn_checkpoint=None, verifier_override=None,
                 gate_override=None):
        """
        Component mode (legacy):
            alarm_classifier: AlarmClassifier instance
            hypothesis_generator: HypothesisGenerator instance
            caevn_trainer: CAEVNTrainer instance (with trained model)
            fis: MamdaniFIS instance
            action_generator: ActionGenerator instance
            regression_tester: RegressionTester instance
            knowledge_base: FAISS index or KnowledgeBase for evidence

        Dataset mode (used by the experiment runners):
            dataset_name: "pq16" | "pq" | "cwru" | "care"
            ablation_variant: one of ABLATION_VARIANTS (None == "full")
            llm_client: BaseLLMClient; None -> framework.llm_client
                .get_default_client (vLLM on server, mock when
                ASOC_LLM_MODE=mock)
            encoder: sentence encoder override (default shared MiniLM)
            seed: RandomState seed; None -> config.RANDOM_SEED read at
                build time (rerun_r1_aggregate reseeding pattern)
            caevn_checkpoint: explicit CAEVN .pt path override
            verifier_override: alternative verification layer exposing
                score(hypothesis_text, evidence_docs, ...)
            gate_override: alternative gate exposing infer(v, c, r)
        """
        if ablation_variant is not None and \
                ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant: "
                             f"{ablation_variant}")

        self.alarm = alarm_classifier
        self.hypothesis = hypothesis_generator
        self.caevn = caevn_trainer
        self.fis = fis
        self.action = action_generator
        self.explain = None
        self.regression = regression_tester
        self.kb = knowledge_base

        self.dataset_name = dataset_name
        self.ablation_variant = ablation_variant \
            if ablation_variant != "full" else None
        self._llm_client = llm_client
        self._encoder = encoder
        self._seed = seed
        self._caevn_checkpoint = caevn_checkpoint
        self._verifier_override = verifier_override
        self._gate_override = gate_override
        self._built = alarm_classifier is not None
        self._moo_cache = None
        self._moo_vscores = {}
        self._moo_beta_resolution = {}
        self._operating_params = None

        if not self._built and dataset_name is None:
            raise ValueError(
                "CLLLMOpsPipeline requires either stage components or "
                "dataset_name=...")

        # Track historical reliability per fault class
        self.reliability = {}
        self.alpha_ema = 0.9  # exponential moving average decay

    # ------------------------------------------------------------------
    # Dataset-mode lazy construction
    # ------------------------------------------------------------------

    def _ensure_built(self):
        """Build all stage components for dataset mode (idempotent)."""
        if self._built:
            return
        import config as config_module
        from data import load_splits
        from framework.llm_client import get_default_client
        from framework.knowledge_base import build_knowledge_base

        seed = self._seed if self._seed is not None \
            else config_module.RANDOM_SEED
        self._seed = seed
        variant = self.ablation_variant
        if self._operating_params is None:
            self._operating_params = self._load_operating_params()
        op = self._operating_params

        splits = load_splits(self.dataset_name)
        self._splits = splits

        if self.alarm is None:
            self.alarm = build_alarm_classifier(
                self.dataset_name, splits, seed)

        encoder = self._encoder or get_sentence_encoder()
        self._encoder = encoder

        llm = self._llm_client or get_default_client(
            dataset_name=self.dataset_name, seed=seed)
        self._llm_client = llm

        if self.hypothesis is None:
            M = 1 if variant == "no_self_consistency" \
                else int(op.get("M", SELF_CONSISTENCY_M))
            self.hypothesis = HypothesisGenerator(
                llm_client=llm, encoder=encoder, M=M)
        if self.action is None:
            self.action = ActionGenerator(llm_client=llm)
        if self.explain is None and EXPLAIN_AUDIT:
            self.explain = ExplanationGenerator(llm_client=llm)

        if self.kb is None:
            self.kb = build_knowledge_base(
                self.dataset_name, encoder, splits=splits, seed=seed)

        if (self.caevn is None and self._verifier_override is None
                and variant not in (
                "no_verifier", "rule_only_verifier")):
            self.caevn = self._load_verifier(variant)

        if self.fis is None:
            fis_kwargs = {
                key: op[key] for key in (
                    "alpha_p", "alpha_d", "delta", "lambda_r")
                if key in op
            }
            if variant == "grid_search":
                fis_kwargs = dict(GRID_SEARCH_FIS_PARAMS)
            if variant == "model_only_verifier":
                fis_kwargs["lambda_r"] = 0.0
            self.fis = MamdaniFIS(**fis_kwargs)

        if self.regression is None:
            self.regression = RegressionTester()
            n_val = len(splits["y_val"])
            if n_val > 0:
                self.regression.build_initial_suite(
                    splits["X_val"], splits["y_val"],
                    n=min(REGRESSION_SUITE_SIZE, n_val))

        self._built = True

    @staticmethod
    def _load_operating_params():
        """Read the validation-selected point, falling back to config."""
        if not OPERATING_POINT_JSON.exists():
            return {}
        try:
            payload = json.loads(OPERATING_POINT_JSON.read_text())
            params = payload.get("params", payload)
            required = {"alpha_p", "alpha_d", "delta", "lambda_r",
                        "M", "beta"}
            if not required <= set(params):
                raise ValueError("missing operating-point parameters")
            return params
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid operating point {OPERATING_POINT_JSON}: {exc}") \
                from exc

    @staticmethod
    def _resolve_beta_checkpoint(dataset_name, requested_beta):
        """Nearest log-scale trained beta checkpoint for a dataset."""
        prefix = f"caevn_{dataset_name}_beta"
        candidates = []
        for path in MODEL_DIR.glob(f"{prefix}*.pt"):
            match = re.fullmatch(
                rf"{re.escape(prefix)}([0-9.eE+-]+)\.pt", path.name)
            if match:
                try:
                    candidates.append((float(match.group(1)), path))
                except ValueError:
                    pass
        if not candidates:
            return MODEL_DIR / f"caevn_{dataset_name}.pt", \
                float(CAEVN_ENTROPY_BETA)
        selected_beta, selected_path = min(
            candidates,
            key=lambda item: abs(np.log10(max(item[0], 1e-12))
                                 - np.log10(max(requested_beta, 1e-12))))
        return selected_path, selected_beta

    def _load_verifier(self, variant):
        """Load the trained neural verifier from its checkpoint."""
        if variant == "mlp_verifier":
            path = MODEL_DIR / f"caevn_{self.dataset_name}_mlp.pt"
            trainer = CAEVNTrainer(model=MLPVerifier(),
                                   device=self._verifier_device())
        else:
            if self._caevn_checkpoint:
                path = Path(self._caevn_checkpoint)
            elif "beta" in (self._operating_params or {}):
                path, _ = self._resolve_beta_checkpoint(
                    self.dataset_name, self._operating_params["beta"])
            else:
                path = MODEL_DIR / f"caevn_{self.dataset_name}.pt"
            trainer = CAEVNTrainer(device=self._verifier_device())
        if not path.exists():
            raise RuntimeError(
                f"Verifier checkpoint not found: {path}. Train it "
                f"first:\n    python experiments/train_caevn.py "
                f"--datasets {self.dataset_name}"
                + (" --mock-llm" if os.environ.get("ASOC_LLM_MODE")
                   == "mock" else ""))
        trainer.load(path)
        return trainer

    @staticmethod
    def _verifier_device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Per-signal execution (Stage 1-5)
    # ------------------------------------------------------------------

    def run(self, signal, run_regression=True):
        """Execute full pipeline on a single signal.

        Args:
            signal: raw sensor data array
            run_regression: whether to run regression testing

        Returns:
            dict with stage outputs and timing
        """
        self._ensure_built()
        result = {"timings": {}}
        t0 = time.time()

        # Stage 1: Alarm Classification
        t1 = time.time()
        alarm_result = self.alarm.classify(signal)
        result["alarm"] = alarm_result
        result["timings"]["alarm_ms"] = (time.time() - t1) * 1000

        # Stage 2: Hypothesis Generation
        t2 = time.time()
        hyp_result = self.hypothesis.generate(alarm_result)
        result["hypothesis"] = {
            "selected": hyp_result["selected"],
            "consistency_score": hyp_result["consistency_score"],
            "n_samples": len(hyp_result["hypotheses"]),
        }
        result["timings"]["hypothesis_ms"] = (time.time() - t2) * 1000

        # Stage 3: Verification (CAEVN + FIS); the no_verifier ablation
        # (== Agent-noV baseline) passes every hypothesis through.
        t3 = time.time()
        if self.ablation_variant == "no_verifier":
            verification_result = {
                "score": 1.0,
                "consistency": hyp_result["consistency_score"],
                "reliability": 0.5,
                "decision": "proceed",
                "alpha": 1.0,
                "attn_weights": None,
                "evidence_indices": [],
            }
        else:
            verification_result = self._verify(
                hyp_result, alarm_result["fault_class"])
        result["verification"] = verification_result
        result["timings"]["verification_ms"] = (time.time() - t3) * 1000

        # Layer 3: LLM explanation for the audit log. Post-gating and
        # score-inert by construction — the decision above is already
        # fixed before this call (methodology Sec. 4, Layer 3).
        if self.explain is not None \
                and self.ablation_variant != "no_verifier":
            t3b = time.time()
            verification_result["explanation"] = self.explain.generate(
                hyp_result["selected"], verification_result,
                self._evidence_docs(
                    verification_result.get("evidence_indices", [])))
            result["timings"]["explanation_ms"] = \
                (time.time() - t3b) * 1000

        # Stage 4: Action Generation
        t4 = time.time()
        action_result = self.action.generate(
            alarm_result, hyp_result, verification_result)
        result["action"] = action_result
        result["timings"]["action_ms"] = (time.time() - t4) * 1000

        # Stage 5: Regression Testing (optional)
        if self.ablation_variant == "no_regression":
            run_regression = False
        if run_regression and verification_result["decision"] == "proceed":
            t5 = time.time()
            reg_result = self.regression.run(
                lambda x: self.alarm.classify(x))
            result["regression"] = reg_result
            result["timings"]["regression_ms"] = (time.time() - t5) * 1000

            # Update reliability based on regression result
            self._update_reliability(
                alarm_result["fault_class"], reg_result["passed"])
        else:
            result["regression"] = {"passed": None, "skipped": True}

        result["timings"]["total_ms"] = (time.time() - t0) * 1000
        result["final_decision"] = verification_result["decision"]
        return result

    def _score_hypothesis(self, hypothesis_emb, evidence_embs):
        """Verification score v for one hypothesis (variant-aware)."""
        if self.ablation_variant == "rule_only_verifier":
            h = hypothesis_emb / (np.linalg.norm(hypothesis_emb) + 1e-8)
            e = evidence_embs / (
                np.linalg.norm(evidence_embs, axis=1, keepdims=True)
                + 1e-8)
            sim = float((e @ h).mean())
            return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0)), None
        return self.caevn.verify(hypothesis_emb, evidence_embs)

    def _gate(self, v_score, coverage, contradiction, high_risk=False):
        """Gating decision (variant-aware)."""
        if self._gate_override is not None:
            return self._gate_override.infer(
                v=v_score, c=coverage, r=contradiction)
        if self.ablation_variant == "no_gating":
            decision = "proceed" if v_score >= 0.5 else "reject"
            return decision, float(v_score), {"rule_activations": []}
        if self.ablation_variant == "hard_gating":
            s = float(np.clip(
                0.5 * (v_score + coverage) * (1.0 - contradiction),
                0.0, 1.0))
            threshold = self.fis.alpha_p + (
                self.fis.delta if high_risk else 0.0)
            if contradiction >= 0.4:
                decision = "reject"
            elif coverage < self.fis.coverage_defer_threshold:
                decision = "defer"
            elif s >= threshold:
                decision = "proceed"
            elif s < self.fis.alpha_d:
                decision = "defer"
            else:
                decision = "reject"
            return decision, float(s), {"rule_activations": []}
        return self.fis.infer(
            v=v_score, coverage=coverage, contradiction=contradiction,
            high_risk=high_risk)

    @staticmethod
    def _rule_support(hypothesis_text, evidence_docs, class_names):
        """Layer-1 symbolic support from diagnosis/evidence tag agreement."""
        from evaluation.metrics import parse_structured, find_class_mentions
        parsed = parse_structured(hypothesis_text) or {}
        mentions = find_class_mentions(
            str(parsed.get("diagnosis", "")), class_names)
        if not mentions:
            # Real LLMs vary their JSON keys/wording; before declaring
            # "no diagnosis", fall back to the full hypothesis text.
            mentions = find_class_mentions(hypothesis_text, class_names)
        if not mentions or not evidence_docs:
            return 0.0
        diagnosis = mentions[0].lower()
        return float(np.mean([
            str(doc.get("tag", "")).lower() == diagnosis
            for doc in evidence_docs
        ]))

    @staticmethod
    def _evidence_coverage(hypothesis_emb, evidence_embs):
        """Mean normalized cosine relevance of the retrieved evidence."""
        if len(evidence_embs) == 0:
            return 0.0
        h = hypothesis_emb / (np.linalg.norm(hypothesis_emb) + 1e-8)
        e = evidence_embs / (
            np.linalg.norm(evidence_embs, axis=1, keepdims=True) + 1e-8)
        relevance = np.clip((e @ h + 1.0) / 2.0, 0.0, 1.0)
        return float(np.mean(relevance))

    def _verify(self, hyp_result, fault_class):
        """Run CAEVN verification + FIS gating."""
        hypothesis_emb = hyp_result["selected_embedding"]

        if self.kb is None:
            raise RuntimeError(
                "CLLLMOpsPipeline requires a knowledge base (kb); "
                "none was supplied."
            )
        evidence_embs, evidence_idx = self._retrieve_evidence(
            hypothesis_emb)

        evidence_docs = self._evidence_docs(evidence_idx)
        if self._verifier_override is not None:
            neural_score, verifier_details = self._verifier_override.score(
                hypothesis_text=hyp_result["selected"],
                evidence_docs=evidence_docs,
                hypothesis_embedding=hypothesis_emb,
                evidence_embeddings=evidence_embs,
                consistency=hyp_result["consistency_score"],
                class_names=self.alarm.class_names,
            )
            attn_weights = None
            item_scores = verifier_details.get("evidence_scores") \
                if isinstance(verifier_details, dict) else None
        else:
            neural_score, attn_weights = self._score_hypothesis(
                hypothesis_emb, evidence_embs)
            verifier_details = None
            item_scores = (None if self.ablation_variant
                           == "rule_only_verifier"
                           else self.caevn.verify_each(
                               hypothesis_emb, evidence_embs))

        coverage = self._evidence_coverage(hypothesis_emb, evidence_embs)
        if item_scores is None or len(item_scores) == 0:
            contradiction = float(neural_score < 0.5)
        else:
            contradiction = float(np.mean(np.asarray(item_scores) < 0.5))
        rule_score = self._rule_support(
            hyp_result["selected"], evidence_docs, self.alarm.class_names)
        if self.ablation_variant == "rule_only_verifier":
            v_score = rule_score
        else:
            lambda_r = float(getattr(self.fis, "lambda_r", 0.4))
            v_score = float(np.clip(
                lambda_r * rule_score
                + (1.0 - lambda_r) * float(neural_score), 0.0, 1.0))

        # Historical reliability (fixed prior for model_only ablation)
        if self.ablation_variant == "model_only_verifier":
            r = 0.5
        else:
            r = self.reliability.get(fault_class, 0.5)

        from evaluation.metrics import parse_structured
        parsed = parse_structured(hyp_result["selected"]) or {}
        high_risk = str(parsed.get("severity", "")).lower() in {
            "high", "critical"}
        decision, alpha, fis_details = self._gate(
            v_score, coverage, contradiction, high_risk=high_risk)

        return {
            "score": v_score,
            "caevn_score": float(neural_score),
            "rule_score": rule_score,
            "coverage": coverage,
            "contradiction_severity": contradiction,
            "consistency": hyp_result["consistency_score"],
            "reliability": r,
            "decision": decision,
            "alpha": alpha,
            "attn_weights": attn_weights.tolist()
                if hasattr(attn_weights, "tolist") else attn_weights,
            "evidence_indices": evidence_idx,
            "verifier_details": verifier_details,
            "high_risk": high_risk,
        }

    def _retrieve_evidence(self, query_emb, k=EVIDENCE_TOP_K):
        """Retrieve top-k evidence from knowledge base.

        Returns:
            (embeddings (k, d), indices list) of retrieved entries.
        """
        query = query_emb.reshape(1, -1).astype(np.float32)
        _, indices = self.kb.search(query, k)
        idx_list = [int(i) for i in indices[0] if int(i) >= 0]
        embs = np.array([self.kb.reconstruct(i) for i in idx_list])
        return embs, idx_list

    def _update_reliability(self, fault_class, passed):
        """Update EMA reliability for a fault class."""
        current = self.reliability.get(fault_class, 0.5)
        new_val = self.alpha_ema * current + (1 - self.alpha_ema) * (1.0 if passed else 0.0)
        self.reliability[fault_class] = new_val

    def batch_run(self, signals, run_regression=False):
        """Run pipeline on a batch of signals.

        Returns:
            list of result dicts
            summary dict with aggregate metrics
        """
        results = [self.run(s, run_regression=run_regression) for s in signals]

        decisions = [r["final_decision"] for r in results]
        proceed_rate = decisions.count("proceed") / len(decisions)
        defer_rate = decisions.count("defer") / len(decisions)
        reject_rate = decisions.count("reject") / len(decisions)

        avg_time = np.mean([r["timings"]["total_ms"] for r in results])

        summary = {
            "n_samples": len(results),
            "proceed_rate": proceed_rate,
            "defer_rate": defer_rate,
            "reject_rate": reject_rate,
            "avg_time_ms": avg_time,
        }

        return results, summary

    # ------------------------------------------------------------------
    # Evaluation (dataset mode)
    # ------------------------------------------------------------------

    def _evidence_docs(self, indices):
        """Structured docs for retrieved evidence (judge input)."""
        if not hasattr(self.kb, "get_doc"):
            return []
        return [self.kb.get_doc(i) for i in indices]

    def _episode_record(self, result, judge):
        """Distill one run() result into a judged episode record."""
        from evaluation.metrics import (
            parse_structured, diagnosis_to_class_id,
        )
        alarm = result["alarm"]
        verif = result["verification"]
        hyp_text = result["hypothesis"]["selected"]
        class_names = self.alarm.class_names

        parsed = parse_structured(hyp_text)
        diag_id = diagnosis_to_class_id(hyp_text, class_names)
        severity = parsed.get("severity") if parsed else None

        features = alarm.get("features")
        feat_mean = float(np.mean(features)) \
            if features is not None else None

        evidence_docs = self._evidence_docs(
            verif.get("evidence_indices", []))
        verdict = judge.judge(hyp_text, evidence_docs,
                              features_mean=feat_mean,
                              class_names=class_names)

        decision = verif["decision"]
        final_pred = diag_id if (decision == "proceed"
                                 and diag_id is not None) \
            else alarm["fault_class"]

        return {
            "alarm_pred": alarm["fault_class"],
            "alarm_conf": alarm["confidence"],
            "diagnosis_id": diag_id,
            "severity": severity,
            "decision": decision,
            "alpha": verif.get("alpha", 1.0),
            "v_score": verif.get("score"),
            "consistency": verif.get("consistency"),
            "action_text": result["action"].get("action_text"),
            "final_pred": int(final_pred),
            "halluc": verdict["hallucinated"],
            "contradiction": verdict["contradiction"],
            "judge_reasons": verdict.get("reasons", []),
            "total_ms": result["timings"]["total_ms"],
        }

    def evaluate(self, X, y, seed=None, max_episodes=None, judge=None):
        """Evaluate the full pipeline on labeled test data.

        Runs every stage per episode and aggregates the paper's metric
        set via evaluation.metrics.compute_metrics (hallucination
        judged by an evaluation-time judge independent of CAEVN/FIS).

        Args:
            X, y: test signals and labels
            seed: decoding seed (reseeds the mock client and the
                episode subsampling; auto-passed by
                run_statistical_protocol when present)
            max_episodes: stratified cap on evaluated episodes
                (default config.PIPELINE_EVAL_MAX_EPISODES)
            judge: HallucinationJudge override (default rule-based;
                set ASOC_JUDGE=nli on the server for the DeBERTa-MNLI
                judge)
        Returns:
            dict with accuracy, f1, hallucination_rate,
            action_accuracy, false_escalation_rate, and auxiliary
            rates -- all computed from the episode records.
        """
        self._ensure_built()
        from evaluation.metrics import compute_metrics, get_judge

        if seed is None:
            seed = self._seed
        else:
            if hasattr(self._llm_client, "reseed"):
                self._llm_client.reseed(seed)
        rng = np.random.RandomState(seed)

        y = np.asarray(y)
        max_episodes = max_episodes or PIPELINE_EVAL_MAX_EPISODES
        if len(y) > max_episodes:
            keep = []
            classes = np.unique(y)
            per_class = max(1, max_episodes // len(classes))
            for cls in classes:
                idx = np.where(y == cls)[0]
                keep.append(rng.choice(idx, size=min(per_class, len(idx)),
                                       replace=False))
            sel = np.concatenate(keep)
        else:
            sel = np.arange(len(y))

        judge = judge or get_judge(os.environ.get("ASOC_JUDGE", "rule"))

        episodes = []
        for i, idx in enumerate(sel):
            run_reg = (i % REGRESSION_STRIDE == 0)
            result = self.run(X[idx], run_regression=run_reg)
            episodes.append(self._episode_record(result, judge))

        metrics = compute_metrics(episodes, y[sel], self.dataset_name)
        metrics["avg_time_ms"] = float(np.mean(
            [ep["total_ms"] for ep in episodes]))
        return metrics

    def evaluate_paired_agent_nov(self, X, y, seed=None,
                                  max_episodes=None, judge=None):
        """Evaluate the full gate and Agent-noV on identical generations.

        The statistical protocol compares these two methods repeatedly.
        Running two independent pipelines would double inference and, more
        importantly, confound the gate effect with different sampled
        hypotheses.  This paired path generates the alarm, hypotheses and
        CAEVN evidence once per episode, then applies either the learned gate
        or an unconditional proceed decision.  An action is generated for the
        no-verifier arm only when the full arm did not already proceed.

        Returns a mapping keyed by the names used in
        ``run_statistical_protocol.py``.
        """
        self._ensure_built()
        from evaluation.metrics import compute_metrics, get_judge

        if self.ablation_variant is not None:
            raise ValueError("paired evaluation requires the full pipeline")
        if seed is None:
            seed = self._seed
        elif hasattr(self._llm_client, "reseed"):
            self._llm_client.reseed(seed)
        rng = np.random.RandomState(seed)

        y = np.asarray(y)
        max_episodes = max_episodes or PIPELINE_EVAL_MAX_EPISODES
        if len(y) > max_episodes:
            keep = []
            classes = np.unique(y)
            per_class = max(1, max_episodes // len(classes))
            for cls in classes:
                idx = np.where(y == cls)[0]
                keep.append(rng.choice(idx, size=min(per_class, len(idx)),
                                       replace=False))
            sel = np.concatenate(keep)
        else:
            sel = np.arange(len(y))

        judge = judge or get_judge(os.environ.get("ASOC_JUDGE", "rule"))
        episodes = {"cl_llmops": [], "agent_nov": []}
        for i, idx in enumerate(sel):
            result = self.run(X[idx], run_regression=(
                i % REGRESSION_STRIDE == 0))
            episodes["cl_llmops"].append(
                self._episode_record(result, judge))

            full_verif = result["verification"]
            nov_verif = {
                "score": 1.0,
                "consistency": full_verif.get("consistency"),
                "reliability": full_verif.get("reliability", 0.5),
                "decision": "proceed",
                "alpha": 1.0,
                # Preserve the actual retrieved documents for the
                # independent evaluation judge; they are not used to gate.
                "evidence_indices": full_verif.get(
                    "evidence_indices", []),
            }
            if full_verif["decision"] == "proceed":
                nov_action = result["action"]
            else:
                nov_action = self.action.generate(
                    result["alarm"], result["hypothesis"], nov_verif)
            nov_result = {
                "alarm": result["alarm"],
                "hypothesis": result["hypothesis"],
                "verification": nov_verif,
                "action": nov_action,
                "timings": result["timings"],
            }
            episodes["agent_nov"].append(
                self._episode_record(nov_result, judge))

        output = {}
        for name, rows in episodes.items():
            metrics = compute_metrics(rows, y[sel], self.dataset_name)
            metrics["avg_time_ms"] = float(np.mean(
                [ep["total_ms"] for ep in rows]))
            metrics["paired_shared_generations"] = True
            output[name] = metrics
        return output

    def collect_gate_calibration(self, X, y, seed=None,
                                 max_episodes=400, judge=None):
        """Collect validation-only CAEVN/gate artifacts for calibration.

        ``correct`` means that the selected diagnosis matches the label and
        its evidence claims pass the independent evaluation judge.  No action
        is generated, so these rows are materially cheaper than a full
        evaluation and cannot leak test labels into the fitted calibrator.
        """
        self._ensure_built()
        from evaluation.metrics import (
            diagnosis_to_class_id, get_judge,
        )

        if seed is None:
            seed = self._seed
        elif hasattr(self._llm_client, "reseed"):
            self._llm_client.reseed(seed)
        rng = np.random.RandomState(seed)
        y = np.asarray(y)
        if len(y) > max_episodes:
            keep = []
            classes = np.unique(y)
            per_class = max(1, max_episodes // len(classes))
            for cls in classes:
                idx = np.where(y == cls)[0]
                keep.append(rng.choice(idx, size=min(per_class, len(idx)),
                                       replace=False))
            sel = np.concatenate(keep)
        else:
            sel = np.arange(len(y))

        judge = judge or get_judge(os.environ.get("ASOC_JUDGE", "rule"))
        rows = []
        for idx in sel:
            alarm = self.alarm.classify(X[idx])
            hyp = self.hypothesis.generate(alarm)
            verif = self._verify(hyp, alarm["fault_class"])
            features = alarm.get("features")
            verdict = judge.judge(
                hyp["selected"],
                self._evidence_docs(verif.get("evidence_indices", [])),
                features_mean=(float(np.mean(features))
                               if features is not None else None),
                class_names=self.alarm.class_names)
            diagnosis_id = diagnosis_to_class_id(
                hyp["selected"], self.alarm.class_names)
            rows.append({
                "v_score": float(verif.get("caevn_score", verif["score"])),
                "integrated_score": float(verif["score"]),
                "rule_score": float(verif.get("rule_score", 0.0)),
                "coverage": float(verif.get("coverage", 0.0)),
                "contradiction_severity": float(
                    verif.get("contradiction_severity", 0.0)),
                "consistency": float(verif["consistency"]),
                "reliability": float(verif["reliability"]),
                "reference_decision": verif["decision"],
                "diagnosis_correct": diagnosis_id == int(y[idx]),
                "claim_supported": not verdict["hallucinated"],
                "correct": (diagnosis_id == int(y[idx])
                            and not verdict["hallucinated"]),
            })
        return rows

    def evaluate_gate_family(self, X, y, gates, seed=None,
                             max_episodes=None, judge=None):
        """Compare scalar gates on identical test-time artifacts.

        ``gates`` maps a result name to either ``None`` (the native FIS) or
        an object exposing ``infer(v, c, r)``.  The full pipeline is executed
        once; each alternative only re-gates its CAEVN output.  This is the
        paired, operating-point-controlled protocol requested in R3.2b.
        """
        self._ensure_built()
        from evaluation.metrics import compute_metrics, get_judge

        if seed is None:
            seed = self._seed
        elif hasattr(self._llm_client, "reseed"):
            self._llm_client.reseed(seed)
        rng = np.random.RandomState(seed)
        y = np.asarray(y)
        max_episodes = max_episodes or PIPELINE_EVAL_MAX_EPISODES
        if len(y) > max_episodes:
            keep = []
            classes = np.unique(y)
            per_class = max(1, max_episodes // len(classes))
            for cls in classes:
                idx = np.where(y == cls)[0]
                keep.append(rng.choice(idx, size=min(per_class, len(idx)),
                                       replace=False))
            sel = np.concatenate(keep)
        else:
            sel = np.arange(len(y))

        judge = judge or get_judge(os.environ.get("ASOC_JUDGE", "rule"))
        episode_sets = {name: [] for name in gates}
        for i, idx in enumerate(sel):
            result = self.run(X[idx], run_regression=(
                i % REGRESSION_STRIDE == 0))
            native = result["verification"]
            for name, gate in gates.items():
                if gate is None:
                    arm = result
                else:
                    score_source = getattr(
                        gate, "score_source", "integrated")
                    gate_score = (native.get("caevn_score", native["score"])
                                  if score_source == "caevn"
                                  else native["score"])
                    decision, alpha, details = gate.infer(
                        v=gate_score, coverage=native["coverage"],
                        contradiction=native["contradiction_severity"],
                        high_risk=native.get("high_risk", False))
                    verif = dict(native)
                    verif.update({"decision": decision, "alpha": alpha,
                                  "gate_details": details})
                    if decision == native["decision"]:
                        action = result["action"]
                    else:
                        action = self.action.generate(
                            result["alarm"], result["hypothesis"], verif)
                    arm = {
                        "alarm": result["alarm"],
                        "hypothesis": result["hypothesis"],
                        "verification": verif,
                        "action": action,
                        "timings": result["timings"],
                    }
                episode_sets[name].append(self._episode_record(arm, judge))

        output = {}
        for name, rows in episode_sets.items():
            metrics = compute_metrics(rows, y[sel], self.dataset_name)
            metrics["avg_time_ms"] = float(np.mean(
                [ep["total_ms"] for ep in rows]))
            metrics["paired_shared_generations"] = True
            output[name] = metrics
        return output

    def evaluate_judge_family(self, X, y, judges, seed=None,
                              max_episodes=None):
        """Score identical released hypotheses with independent judges."""
        self._ensure_built()
        from evaluation.metrics import compute_metrics

        if seed is None:
            seed = self._seed
        elif hasattr(self._llm_client, "reseed"):
            self._llm_client.reseed(seed)
        rng = np.random.RandomState(seed)
        y = np.asarray(y)
        max_episodes = max_episodes or PIPELINE_EVAL_MAX_EPISODES
        if len(y) > max_episodes:
            keep = []
            classes = np.unique(y)
            per_class = max(1, max_episodes // len(classes))
            for cls in classes:
                idx = np.where(y == cls)[0]
                keep.append(rng.choice(idx, size=min(per_class, len(idx)),
                                       replace=False))
            sel = np.concatenate(keep)
        else:
            sel = np.arange(len(y))

        episode_sets = {name: [] for name in judges}
        audit_rows = []
        for i, idx in enumerate(sel):
            result = self.run(X[idx], run_regression=(
                i % REGRESSION_STRIDE == 0))
            per_judge = {}
            for name, judge in judges.items():
                episode = self._episode_record(result, judge)
                episode_sets[name].append(episode)
                per_judge[name] = {
                    "hallucinated": bool(episode["halluc"]),
                    "contradiction": bool(episode["contradiction"]),
                    "reasons": episode["judge_reasons"],
                }
            audit_rows.append({
                "dataset": self.dataset_name,
                "seed": int(seed),
                "sample_index": int(idx),
                "y_true": int(y[idx]),
                "hypothesis_text": result["hypothesis"]["selected"],
                "evidence_docs": self._evidence_docs(
                    result["verification"].get("evidence_indices", [])),
                "decision": result["verification"]["decision"],
                "explanation": result["verification"].get("explanation"),
                "judge_outputs": per_judge,
            })

        metrics = {
            name: compute_metrics(rows, y[sel], self.dataset_name)
            for name, rows in episode_sets.items()
        }
        return {"metrics": metrics, "audit_rows": audit_rows}

    # ------------------------------------------------------------------
    # MOPSO objective evaluation (optimization/objectives.py)
    # ------------------------------------------------------------------

    def _build_moo_cache(self):
        """Cache LLM/verifier artifacts for evaluate_with_params.

        For a fixed stratified val subset, generates MOO_MAX_M
        hypothesis samples per episode ONCE (actual LLM calls) plus
        per-hypothesis evidence retrieval and judge verdicts. Every
        subsequent evaluate_with_params call reuses the first M cached
        samples, so varying M/alpha_p/alpha_d/delta/lambda_r never
        fabricates outputs -- it re-scores actual ones.
        """
        if self._moo_cache is not None:
            return self._moo_cache
        from evaluation.metrics import (
            get_judge, parse_structured, diagnosis_to_class_id,
        )

        splits = self._splits
        X_val, y_val = splits["X_val"], splits["y_val"]
        if len(y_val) == 0:
            X_val, y_val = splits["X_test"], splits["y_test"]
        rng = np.random.RandomState(self._seed)
        keep = []
        classes = np.unique(y_val)
        per_class = max(1, MOO_EVAL_SAMPLES // len(classes))
        for cls in classes:
            idx = np.where(y_val == cls)[0]
            keep.append(rng.choice(idx, size=min(per_class, len(idx)),
                                   replace=False))
        sel = np.concatenate(keep)

        judge = get_judge(os.environ.get("ASOC_JUDGE", "rule"))
        class_names = self.alarm.class_names
        old_M = self.hypothesis.M
        self.hypothesis.M = MOO_MAX_M
        cache = []
        try:
            for idx in sel:
                alarm_result = self.alarm.classify(X_val[idx])
                hyp = self.hypothesis.generate(alarm_result)
                features = alarm_result.get("features")
                feat_mean = float(np.mean(features)) \
                    if features is not None else None

                per_hyp = []
                for j, (text, emb) in enumerate(
                        zip(hyp["hypotheses"], hyp["embeddings"])):
                    ev_embs, ev_idx = self._retrieve_evidence(emb)
                    evidence_docs = self._evidence_docs(ev_idx)
                    verdict = judge.judge(
                        text, evidence_docs,
                        features_mean=feat_mean,
                        class_names=class_names)
                    parsed = parse_structured(text)
                    per_hyp.append({
                        "emb": np.asarray(emb, dtype=np.float32),
                        "ev_embs": ev_embs.astype(np.float32),
                        "coverage": self._evidence_coverage(emb, ev_embs),
                        "rule_score": self._rule_support(
                            text, evidence_docs, class_names),
                        "diag_id": diagnosis_to_class_id(text,
                                                         class_names),
                        "severity": parsed.get("severity")
                            if parsed else None,
                        "halluc": verdict["hallucinated"],
                        "contradiction": verdict["contradiction"],
                    })
                cache.append({
                    "y_true": int(y_val[idx]),
                    "alarm_pred": alarm_result["fault_class"],
                    "alarm_conf": alarm_result["confidence"],
                    "hyps": per_hyp,
                })
        finally:
            self.hypothesis.M = old_M
        self._moo_cache = cache
        return cache

    def _moo_v_scores(self, beta):
        """Verification scores for all cached hypotheses.

        beta is a TRAINING-time hyperparameter (attention-entropy
        regularization); at evaluation time it selects a matching
        pre-trained checkpoint caevn_{ds}_beta{beta:g}.pt when one
        exists, falling back to the default checkpoint otherwise
        (documented in paper Sec. 5.4). Scores are cached per
        checkpoint.
        """
        requested_beta = float(beta)
        beta_path, selected_beta = self._resolve_beta_checkpoint(
            self.dataset_name, requested_beta)
        key = (str(beta_path) if beta_path.name !=
               f"caevn_{self.dataset_name}.pt" else "default")
        self._moo_beta_resolution[requested_beta] = {
            "selected_beta": selected_beta,
            "checkpoint": key,
        }
        if key in self._moo_vscores:
            return self._moo_vscores[key]

        if key != "default":
            trainer = CAEVNTrainer(device=self._verifier_device())
            trainer.load(beta_path)
            old_caevn, self.caevn = self.caevn, trainer
        try:
            scores = []
            for sample in self._moo_cache:
                sample_scores = []
                for h in sample["hyps"]:
                    global_score = self._score_hypothesis(
                        h["emb"], h["ev_embs"])[0]
                    item_scores = self.caevn.verify_each(
                        h["emb"], h["ev_embs"])
                    sample_scores.append({
                        "global": float(global_score),
                        "per_evidence": item_scores.tolist(),
                    })
                scores.append(sample_scores)
        finally:
            if key != "default":
                self.caevn = old_caevn
        self._moo_vscores[key] = scores
        return scores

    @staticmethod
    def _consistency(embeddings):
        """Average pairwise cosine similarity (HypothesisGenerator)."""
        M = len(embeddings)
        if M <= 1:
            return 1.0
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        sim = normalized @ normalized.T
        mask = np.triu(np.ones((M, M), dtype=bool), k=1)
        return float(sim[mask].mean())

    def evaluate_with_params(self, alpha_p, alpha_d, delta, lambda_r,
                             M, beta):
        """Evaluate under a MOPSO parameter vector (paper Table 3).

        Re-gates the cached actual LLM/verifier outputs under the
        requested FIS thresholds and self-consistency budget M, then
        recomputes all metrics honestly via compute_metrics.

        Returns:
            metrics dict including f1, hallucination_rate and
            false_escalation_rate (the three MOPSO objectives).
        """
        self._ensure_built()
        from evaluation.metrics import compute_metrics

        cache = self._build_moo_cache()
        v_scores = self._moo_v_scores(float(beta))
        fis = MamdaniFIS(alpha_p=float(alpha_p), alpha_d=float(alpha_d),
                         delta=float(delta), lambda_r=float(lambda_r))
        M = int(np.clip(int(round(M)), 1, MOO_MAX_M))

        episodes, y_true = [], []
        for sample, vs in zip(cache, v_scores):
            hyps = sample["hyps"][:M]
            embs = np.stack([h["emb"] for h in hyps])
            centroid = embs.mean(axis=0)
            sims = embs @ centroid / (
                np.linalg.norm(embs, axis=1)
                * (np.linalg.norm(centroid) + 1e-8) + 1e-8)
            b = int(np.argmax(sims))
            c = self._consistency(embs)
            neural_score = vs[b]["global"]
            v = float(np.clip(
                float(lambda_r) * hyps[b]["rule_score"]
                + (1.0 - float(lambda_r)) * neural_score, 0.0, 1.0))
            per_evidence = np.asarray(vs[b]["per_evidence"])
            contradiction_severity = float(
                np.mean(per_evidence < 0.5)) if len(per_evidence) else 1.0
            high_risk = str(hyps[b].get("severity", "")).lower() in {
                "high", "critical"}
            decision, alpha, _ = fis.infer(
                v=v, coverage=hyps[b]["coverage"],
                contradiction=contradiction_severity,
                high_risk=high_risk)

            diag_id = hyps[b]["diag_id"]
            final_pred = diag_id if (decision == "proceed"
                                     and diag_id is not None) \
                else sample["alarm_pred"]
            episodes.append({
                "alarm_pred": sample["alarm_pred"],
                "alarm_conf": sample["alarm_conf"],
                "diagnosis_id": diag_id,
                "severity": hyps[b]["severity"],
                "decision": decision,
                "alpha": alpha,
                "v_score": v,
                "caevn_score": neural_score,
                "coverage": hyps[b]["coverage"],
                "contradiction_severity": contradiction_severity,
                "consistency": c,
                "action_text": None,
                "final_pred": int(final_pred),
                "halluc": hyps[b]["halluc"],
                "contradiction": hyps[b]["contradiction"],
            })
            y_true.append(sample["y_true"])

        metrics = compute_metrics(episodes, y_true, self.dataset_name)
        metrics["params"] = {
            "alpha_p": float(alpha_p), "alpha_d": float(alpha_d),
            "delta": float(delta), "lambda_r": float(lambda_r),
            "M": M, "beta": float(beta),
            "beta_resolution": self._moo_beta_resolution.get(float(beta)),
        }
        return metrics
