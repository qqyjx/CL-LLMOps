"""Reviewer-requested substitute verification layers.

These baselines replace CAEVN while leaving the alarm, hypothesis, action,
data split, and evaluation judge unchanged.
"""

import os
import numpy as np

from evaluation.metrics import parse_structured, find_class_mentions


class SelfConsistencyGuardrailVerifier:
    """SelfCheck-style structural and consistency guardrail.

    The score combines independently observable properties: agreement among
    sampled hypotheses, structured-output completeness, and whether cited
    fault names are present in retrieved evidence.  It neither calls CAEVN
    nor reuses the evaluation judge's fabrication thresholds.
    """

    REQUIRED_FIELDS = (
        "diagnosis", "root_cause", "severity", "evidence_cited",
    )

    def score(self, hypothesis_text, evidence_docs, consistency,
              class_names=None, **kwargs):
        parsed = parse_structured(hypothesis_text) or {}
        completeness = float(np.mean([
            bool(parsed.get(field)) for field in self.REQUIRED_FIELDS
        ]))
        diagnosis_mentions = find_class_mentions(
            str(parsed.get("diagnosis", "")), class_names or [])
        cited = parsed.get("evidence_cited", [])
        if isinstance(cited, str):
            cited = [cited]
        cited_mentions = find_class_mentions(
            " ".join(map(str, cited)), class_names or [])
        retrieved_tags = {str(doc.get("tag", "")).lower()
                          for doc in evidence_docs}
        mentioned = set(diagnosis_mentions + cited_mentions)
        support = (float(np.mean([
            name.lower() in retrieved_tags for name in mentioned
        ])) if mentioned else 0.0)
        value = float(np.clip(
            0.55 * float(consistency)
            + 0.25 * completeness + 0.20 * support, 0.0, 1.0))
        return value, {
            "kind": "self_consistency_guardrail",
            "consistency": float(consistency),
            "structured_completeness": completeness,
            "retrieval_support": support,
        }


class NLIEntailmentVerifier:
    """Off-the-shelf NLI verifier over retrieved evidence and claims."""

    DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def __init__(self, model_name=None, device=None, backend=None):
        self.model_name = model_name or os.environ.get(
            "ASOC_NLI_VERIFIER_MODEL", self.DEFAULT_MODEL)
        if backend is not None:
            self.nli = backend
            self.id2label = getattr(backend, "id2label", {})
            return
        from transformers import pipeline as hf_pipeline
        if device is None:
            device = int(os.environ.get("ASOC_NLI_DEVICE", "0"))
        self.nli = hf_pipeline(
            "text-classification", model=self.model_name,
            device=device, top_k=None)
        self.id2label = getattr(self.nli.model.config, "id2label", {})

    @staticmethod
    def _flatten_scores(output):
        while (isinstance(output, list) and len(output) == 1
               and isinstance(output[0], list)):
            output = output[0]
        return output if isinstance(output, list) else [output]

    def _entailment_probability(self, premise, claim):
        output = self.nli({"text": premise, "text_pair": claim})
        entries = self._flatten_scores(output)
        scores = {str(row["label"]).lower(): float(row["score"])
                  for row in entries}
        entail = [v for k, v in scores.items() if "entail" in k]
        if entail:
            return max(entail)
        # Common MNLI id convention: label 2 = entailment.  Respect the
        # loaded config when labels are generic LABEL_0/1/2.
        for idx, label in self.id2label.items():
            if "entail" in str(label).lower():
                return scores.get(str(label).lower(), 0.0)
        return scores.get("label_2", 0.0)

    def score(self, hypothesis_text, evidence_docs, **kwargs):
        parsed = parse_structured(hypothesis_text) or {}
        cited = parsed.get("evidence_cited", [])
        if isinstance(cited, str):
            cited = [cited]
        claims = ([str(parsed.get("diagnosis"))]
                  if parsed.get("diagnosis") else []) + list(map(str, cited))
        premise = " ".join(str(doc.get("text", ""))
                           for doc in evidence_docs)
        if not premise or not claims:
            return 0.0, {"kind": "nli", "claim_scores": []}
        claim_scores = [self._entailment_probability(premise, claim)
                        for claim in claims]
        return float(min(claim_scores)), {
            "kind": "nli", "model": self.model_name,
            "claim_scores": claim_scores,
        }
