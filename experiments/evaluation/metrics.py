"""Honest metric computation for the CL-LLMOps evaluation pipeline.

All metrics are computed from actual per-episode pipeline outputs; no
metric value is synthesized or hardcoded anywhere in this module.

Hallucination judging (paper Section 5, review response 3.2a): the
evaluation-time judge is SEPARATE from the deployment-time verifier
(CAEVN + FIS). The default `RuleBasedJudge` checks the LLM's stated
evidence against (a) the evidence actually retrieved from the
knowledge base and (b) the actual signal features, using deterministic
rules. An `NLIJudge` hook allows plugging an off-the-shelf
DeBERTa-MNLI entailment model on the server (never used inside the
pipeline itself), per the reviewer-requested independent-metric
protocol.

Metric definitions (paper Section 5):
  * hallucination_rate: fraction of episodes where a hypothesis whose
    stated evidence contradicts the actual retrieved data (fabricated
    numeric claims, citations of fault signatures unrelated to the
    diagnosis, or citations absent from the retrieved evidence) is
    RELEASED downstream (gating decision = proceed). Deferred and
    rejected hypotheses never reach the operator, so verification can
    legitimately reduce this rate; the ungated per-hypothesis rate is
    additionally reported as hypothesis_hallucination_rate.
  * contradiction_rate: released episodes citing mutually inconsistent
    evidence pieces (signatures of unrelated fault classes).
  * action_accuracy: recommended action level matches the ground-truth
    level from evaluation/action_rules.py. Escalated episodes
    (gating decision = reject -> human review) count as correct iff
    the ground-truth level is A3/A4 (escalation appropriate).
  * false_escalation_rate: episodes escalated to human review although
    the ground-truth action level was low (A0/A1), i.e. automation
    would have sufficed; denominator = all episodes.
  * overconfident_error_rate: episodes with gating confidence
    alpha > 0.8 but an incorrect final diagnosis.
"""

import json
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.action_rules import (
    get_class_names, action_level_for, ground_truth_action,
)

# Relative deviation above which a numeric evidence claim is judged
# fabricated (honest claims echo the actual value; mock fabrications
# multiply it by >= 2.5).
NUMERIC_FABRICATION_TOL = 0.75


# ---------------------------------------------------------------------------
# Structured-output parsing helpers
# ---------------------------------------------------------------------------

def parse_structured(text):
    """Extract a balanced JSON object from LLM output text.

    Prefers the first NON-DEGENERATE object (at least one truthy value):
    raw-completion generations were observed to emit an all-empty
    "example response" skeleton before the real JSON, and taking the
    first object blindly silently blanked severity/evidence fields
    (2026-07-23 incident). Falls back to the first parseable object,
    and returns None when nothing parses.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    first = None
    start = text.find("{")
    while start != -1:
        depth = 0
        obj = None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        obj = None
                    break
        if isinstance(obj, dict):
            if any(bool(v) for v in obj.values()):
                return obj
            if first is None:
                first = obj
            # skip past this object and keep scanning
            start = text.find("{", i + 1)
            continue
        start = text.find("{", start + 1)
    return first


def _norm_tokens(name):
    """Normalize a class name / text to a lowercase token tuple."""
    return tuple(re.split(r"[\s_+\-]+", str(name).strip().lower()))


def _related(name_a, name_b):
    """Whether two class names are compatible (equal or one is a
    component of the other's compound, e.g. 'sag' vs 'sag_harmonic')."""
    ta, tb = set(_norm_tokens(name_a)), set(_norm_tokens(name_b))
    return ta <= tb or tb <= ta


def find_class_mentions(text, class_names):
    """Find class names mentioned in `text` (longest-match wins).

    Returns a list of distinct class names; a compound match (e.g.
    'sag harmonic') suppresses its component matches over the same
    tokens.
    """
    tokens = list(_norm_tokens(text))
    consumed = [False] * len(tokens)
    mentions = []
    ordered = sorted(class_names, key=lambda n: -len(_norm_tokens(n)))
    for name in ordered:
        pat = _norm_tokens(name)
        L = len(pat)
        for i in range(len(tokens) - L + 1):
            if any(consumed[i:i + L]):
                continue
            if tuple(tokens[i:i + L]) == pat:
                for j in range(i, i + L):
                    consumed[j] = True
                if name not in mentions:
                    mentions.append(name)
    return mentions


def diagnosis_to_class_id(hypothesis, class_names):
    """Map an LLM hypothesis (text or parsed dict) to a class id.

    Tries the structured "diagnosis" field first, then falls back to
    scanning the raw text for class-name mentions. Returns None when
    no class can be identified.
    """
    parsed = parse_structured(hypothesis)
    candidates = []
    if parsed is not None and "diagnosis" in parsed:
        candidates = find_class_mentions(str(parsed["diagnosis"]),
                                         class_names)
    if not candidates and isinstance(hypothesis, str):
        candidates = find_class_mentions(hypothesis, class_names)
    if not candidates:
        return None
    return class_names.index(candidates[0])


# ---------------------------------------------------------------------------
# Hallucination judges (evaluation-time; independent of CAEVN/FIS)
# ---------------------------------------------------------------------------

class HallucinationJudge:
    """Interface for evaluation-time evidence-vs-claim judging.

    Implementations must be independent of the deployment verifier:
    they may not call CAEVN or the FIS, and may not reuse their
    parameters.
    """

    def judge(self, hypothesis_text, evidence_docs, features_mean=None,
              class_names=None):
        """Judge one episode.

        Args:
            hypothesis_text: raw LLM hypothesis output (str)
            evidence_docs: list of dicts with keys "text" and "tag"
                (the evidence actually retrieved from the KB)
            features_mean: mean of the actual alarm feature vector
            class_names: dataset class-name list
        Returns:
            dict with keys: hallucinated (bool), contradiction (bool),
            reasons (list of str)
        """
        raise NotImplementedError


class RuleBasedJudge(HallucinationJudge):
    """Deterministic evidence-vs-claim consistency judge (default).

    Rules (all computed from actual episode artifacts):
      1. numeric fabrication: a cited "mean=<x>" claim deviates from
         the actual feature mean by more than NUMERIC_FABRICATION_TOL
         (relative).
      2. claim mismatch: a cited fault-signature class is unrelated to
         the diagnosed class (broken claim-evidence chain).
      3. unsupported citation: when no diagnosis can be parsed, a
         cited class unrelated to every tag among the actually
         retrieved evidence documents. (When a diagnosis exists,
         rule 2 already covers fabricated citations; conditioning
         rule 3 on it avoids penalizing honest citations that dense
         retrieval happened to miss.)
    hallucinated = (1) or (2) or (3);
    contradiction = two or more mutually unrelated cited classes.
    """

    def judge(self, hypothesis_text, evidence_docs, features_mean=None,
              class_names=None):
        class_names = class_names or []
        parsed = parse_structured(hypothesis_text)
        reasons = []

        cited = parsed.get("evidence_cited", []) if parsed else []
        if isinstance(cited, str):
            cited = [cited]
        cited_text = " ; ".join(str(c) for c in cited)
        if not cited_text and isinstance(hypothesis_text, str):
            cited_text = hypothesis_text

        diagnosis = str(parsed.get("diagnosis", "")) if parsed else ""
        cited_classes = find_class_mentions(cited_text, class_names)
        retrieved_tags = [d.get("tag", "") for d in evidence_docs]

        # Rule 1: numeric fabrication
        if features_mean is not None:
            claims = re.findall(
                r"mean[^-\d]{0,12}(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
                cited_text)
            for claim in claims:
                actual = float(features_mean)
                dev = abs(float(claim) - actual) / max(abs(actual), 1e-6)
                if dev > NUMERIC_FABRICATION_TOL:
                    reasons.append(
                        f"numeric fabrication: claimed mean {claim} vs "
                        f"actual {actual:.4f}")
                    break

        # Rules 2 + 3: citation consistency
        diag_classes = find_class_mentions(diagnosis, class_names)
        diag_name = diag_classes[0] if diag_classes else None
        for c in cited_classes:
            if diag_name is not None:
                if not _related(c, diag_name):
                    reasons.append(
                        f"claim mismatch: cites '{c}' signature but "
                        f"diagnoses '{diag_name}'")
            elif retrieved_tags and not any(_related(c, t)
                                            for t in retrieved_tags):
                reasons.append(
                    f"unsupported citation: '{c}' not among retrieved "
                    f"evidence tags")

        contradiction = any(
            not _related(a, b)
            for i, a in enumerate(cited_classes)
            for b in cited_classes[i + 1:])

        return {
            "hallucinated": len(reasons) > 0,
            "contradiction": contradiction,
            "reasons": reasons,
        }


class NLIJudge(HallucinationJudge):
    """NLI-based judge hook (server-only; requires transformers).

    Uses an off-the-shelf DeBERTa-MNLI cross-encoder that is never
    used inside the deployment pipeline: each cited evidence claim is
    checked for contradiction against the concatenated retrieved
    evidence texts. Falls back with an informative error when the
    model is unavailable locally.
    """

    def __init__(self, model_name="microsoft/deberta-v2-xlarge-mnli",
                 device=None, threshold=0.5):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "NLIJudge requires the transformers package (server "
                "environment). Use RuleBasedJudge locally.") from exc
        self.nli = hf_pipeline("text-classification", model=model_name,
                               device=device, top_k=None)
        self.threshold = threshold
        self._rule = RuleBasedJudge()

    def judge(self, hypothesis_text, evidence_docs, features_mean=None,
              class_names=None):
        parsed = parse_structured(hypothesis_text)
        cited = parsed.get("evidence_cited", []) if parsed else []
        if isinstance(cited, str):
            cited = [cited]
        if not cited and isinstance(hypothesis_text, str):
            cited = [hypothesis_text]
        premise = " ".join(d.get("text", "") for d in evidence_docs)

        reasons = []
        for claim in cited:
            scores = self.nli({"text": premise, "text_pair": str(claim)})
            for entry in (scores if isinstance(scores, list) else [scores]):
                if isinstance(entry, list):
                    entry = {e["label"]: e["score"] for e in entry}
                    contr = max((v for k, v in entry.items()
                                 if "contradiction" in k.lower()),
                                default=0.0)
                else:
                    contr = (entry["score"]
                             if "contradiction" in entry["label"].lower()
                             else 0.0)
                if contr > self.threshold:
                    reasons.append(
                        f"NLI contradiction (p={contr:.2f}): {claim}")
                    break

        # Contradiction flag reuses the deterministic mutual-consistency
        # rule (NLI judges claim-vs-evidence, not claim-vs-claim).
        rule_out = self._rule.judge(hypothesis_text, evidence_docs,
                                    features_mean, class_names)
        return {
            "hallucinated": len(reasons) > 0,
            "contradiction": rule_out["contradiction"],
            "reasons": reasons,
        }


def get_judge(name="rule", **kwargs):
    """Factory: 'rule' (default, local) or 'nli' (server hook)."""
    if name == "rule":
        return RuleBasedJudge()
    if name == "nli":
        return NLIJudge(**kwargs)
    raise ValueError(f"Unknown judge: {name}")


# ---------------------------------------------------------------------------
# Episode-level action prediction + metric aggregation
# ---------------------------------------------------------------------------

def predicted_action_level(episode, dataset_name):
    """Recommended action level for one episode record.

    proceed -> the LLM action plan's stated level (fallback: the
        deterministic mapping applied to the diagnosed class and the
        LLM-stated severity, per paper Section 4);
    defer   -> A0 (keep monitoring while gathering evidence);
    reject  -> "ESC" (escalate to human review).
    """
    decision = episode.get("decision", "proceed")
    if decision == "reject":
        return "ESC"
    if decision == "defer":
        return "A0"

    parsed = parse_structured(episode.get("action_text", ""))
    if parsed is not None:
        level = str(parsed.get("action_level", "")).strip().upper()
        if level in {"A0", "A1", "A2", "A3", "A4"}:
            return level

    cls = episode.get("diagnosis_id")
    if cls is None:
        cls = episode.get("alarm_pred", 0)
    return action_level_for(dataset_name, int(cls),
                            severity=episode.get("severity"))


def compute_metrics(episodes, y_true, dataset_name):
    """Aggregate per-episode records into the paper's metric set.

    Args:
        episodes: list of episode dicts (see pipeline.evaluate); each
            must carry final_pred, decision, halluc, contradiction,
            and enough fields for predicted_action_level().
        y_true: iterable of ground-truth labels, aligned with episodes.
        dataset_name: "pq16" | "pq" | "cwru" | "care"
    Returns:
        dict of metrics; every value computed from the episodes.
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray([int(v) for v in y_true])
    y_pred = np.asarray([int(ep["final_pred"]) for ep in episodes])
    n = len(episodes)

    accuracy = float(np.mean(y_pred == y_true))
    f1 = float(f1_score(y_true, y_pred, average="macro",
                        zero_division=0))

    decisions = [ep.get("decision", "proceed") for ep in episodes]
    released = [d == "proceed" for d in decisions]
    halluc = float(np.mean([
        bool(ep.get("halluc")) and rel
        for ep, rel in zip(episodes, released)]))
    halluc_raw = float(np.mean([bool(ep.get("halluc"))
                                for ep in episodes]))
    contra = float(np.mean([
        bool(ep.get("contradiction")) and rel
        for ep, rel in zip(episodes, released)]))

    gt_levels = [ground_truth_action(dataset_name, y) for y in y_true]
    pred_levels = [predicted_action_level(ep, dataset_name)
                   for ep in episodes]
    action_hits = [
        (p == g) or (p == "ESC" and g in {"A3", "A4"})
        for p, g in zip(pred_levels, gt_levels)
    ]
    action_accuracy = float(np.mean(action_hits))

    false_esc = float(np.mean([
        d == "reject" and g in {"A0", "A1"}
        for d, g in zip(decisions, gt_levels)
    ]))
    alphas = np.asarray([float(ep.get("alpha", 1.0)) for ep in episodes])
    overconf = float(np.mean((alphas > 0.8) & (y_pred != y_true)))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "hallucination_rate": halluc,
        "hypothesis_hallucination_rate": halluc_raw,
        "contradiction_rate": contra,
        "action_accuracy": action_accuracy,
        "false_escalation_rate": false_esc,
        "overconfident_error_rate": overconf,
        "escalation_rate": float(np.mean(
            [d == "reject" for d in decisions])),
        "proceed_rate": float(np.mean(
            [d == "proceed" for d in decisions])),
        "defer_rate": float(np.mean([d == "defer" for d in decisions])),
        "n_episodes": n,
    }
