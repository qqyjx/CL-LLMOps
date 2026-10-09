"""Stage 2: Hypothesis Generation via LLM.

Given an alarm classification, generates diagnostic hypotheses using
the LLM with structured prompts. Supports self-consistency (M samples).
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LLM_TEMPERATURE_HYPOTHESIS, SELF_CONSISTENCY_M,
    LLM_HYPOTHESIS_MAX_TOKENS, SENTENCE_MODEL, EMBEDDING_DIM,
)


# Prompt template for hypothesis generation
HYPOTHESIS_PROMPT = """You are an expert industrial fault diagnosis engineer.

Given the following alarm information:
- Fault type: {fault_class}
- Confidence: {confidence:.1%}
- Top-3 candidates: {top3}
- Signal features: {features}

Generate a diagnostic hypothesis covering root cause analysis, severity
assessment, recommended inspection points, and potential cascading
effects if unaddressed.

Respond with ONE JSON object (no prose outside the JSON) with exactly
these keys:
- "diagnosis": the single most likely fault class name (use the
  candidate class names where applicable)
- "root_cause": physical mechanism that caused this fault (1-2 sentences)
- "severity": one of "none", "low", "medium", "high", "critical"
- "evidence_cited": list of evidence items supporting the diagnosis,
  each citing a retrieved knowledge-base item or a specific signal
  feature value from the alarm (do NOT invent numbers)
- "inspection_points": list of recommended checks
- "cascading_effects": likely consequences if unaddressed (1 sentence)
"""


class HypothesisGenerator:
    """Generate and rank diagnostic hypotheses using LLM + self-consistency."""

    def __init__(self, llm_client=None, encoder=None, M=SELF_CONSISTENCY_M):
        """
        Args:
            llm_client: LLM inference client (vLLM or API wrapper)
            encoder: sentence-transformers model for embedding hypotheses
            M: number of self-consistency samples
        """
        self.llm = llm_client
        self.encoder = encoder
        self.M = M

    def generate(self, alarm_result):
        """Generate M hypothesis samples and compute consistency.

        Args:
            alarm_result: dict from AlarmClassifier.classify()

        Returns:
            dict with keys:
                - hypotheses: list of M hypothesis strings
                - embeddings: (M, d) array of hypothesis embeddings
                - selected: best hypothesis (by consistency)
                - selected_embedding: (d,) embedding of selected hypothesis
                - consistency_score: float in [0, 1]
        """
        prompt = HYPOTHESIS_PROMPT.format(
            fault_class=alarm_result["class_name"],
            confidence=alarm_result["confidence"],
            top3=alarm_result["top3"],
            features=self._summarize_features(alarm_result.get("features")),
        )

        if self.llm is None:
            raise RuntimeError(
                "HypothesisGenerator requires an LLM client; "
                "none was supplied."
            )
        if self.encoder is None:
            raise RuntimeError(
                "HypothesisGenerator requires a sentence encoder; "
                "none was supplied."
            )

        hypotheses = self._generate_with_llm(prompt)
        embeddings = self.encoder.encode(hypotheses)

        # Self-consistency: select hypothesis most similar to centroid
        centroid = embeddings.mean(axis=0)
        similarities = np.array([
            np.dot(e, centroid) / (np.linalg.norm(e) * np.linalg.norm(centroid) + 1e-8)
            for e in embeddings
        ])
        best_idx = np.argmax(similarities)

        # Consistency score: average pairwise cosine similarity
        consistency = self._compute_consistency(embeddings)

        return {
            "hypotheses": hypotheses,
            "embeddings": embeddings,
            "selected": hypotheses[best_idx],
            "selected_embedding": embeddings[best_idx],
            "consistency_score": consistency,
        }

    def _generate_with_llm(self, prompt):
        """Generate M samples, using the backend's native batch path."""
        return self.llm.generate_n(
            prompt, n=self.M,
            temperature=LLM_TEMPERATURE_HYPOTHESIS,
            max_tokens=LLM_HYPOTHESIS_MAX_TOKENS,
        )

    def _compute_consistency(self, embeddings):
        """Average pairwise cosine similarity among M embeddings."""
        M = len(embeddings)
        if M <= 1:
            return 1.0
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        sim_matrix = normalized @ normalized.T
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones((M, M), dtype=bool), k=1)
        return float(sim_matrix[mask].mean())

    def _summarize_features(self, features):
        """Create a text summary of numerical features."""
        if features is None:
            return "N/A"
        if isinstance(features, np.ndarray):
            return f"[{len(features)} features, mean={features.mean():.4f}]"
        return str(features)[:200]
