"""Stage V Layer 3: LLM-assisted explanation for the audit log.

The explanation is generated AFTER the FIS gate has fixed its decision
and is recorded in the episode audit trail only; it never enters the
gating score (methodology Sec. 4, Layer 3), so hallucinated reasoning
cannot override the quantitative evidence assessment.
"""

from config import (
    LLM_TEMPERATURE_EXPLANATION,
    LLM_EXPLANATION_MAX_TOKENS,
)

EXPLANATION_PROMPT = """You are a technical reviewer writing an audit log entry.
Explain the verification result of an industrial fault-diagnosis
hypothesis. Your explanation is recorded for human auditors; it does
NOT change the decision.

Hypothesis:
{hypothesis}

Retrieved evidence:
{evidence}

Quantitative verification result:
- Rule-based support: {rule_score:.2f}
- Neural verifier score: {caevn_score:.2f}
- Combined score: {v_score:.2f}
- Evidence coverage: {coverage:.2f}
- Contradiction severity: {contradiction:.2f}
- Gating decision: {decision}

In 2-4 sentences, explain WHY the evidence supports or contradicts the
hypothesis, referencing specific evidence items. Do NOT propose a
different decision and do NOT invent evidence not listed above.
"""


def _format_evidence(evidence_docs):
    if not evidence_docs:
        return "(none retrieved)"
    lines = []
    for i, doc in enumerate(evidence_docs):
        tag = doc.get("tag", "?")
        text = str(doc.get("text", doc))[:200]
        lines.append(f"[{i}] (tag: {tag}) {text}")
    return "\n".join(lines)


class ExplanationGenerator:
    """Layer-3 audit-log explanation (post-gating, score-inert)."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def generate(self, hypothesis_text, verification_result, evidence_docs):
        if self.llm is None:
            raise RuntimeError(
                "ExplanationGenerator requires an LLM client; "
                "none was supplied."
            )
        prompt = EXPLANATION_PROMPT.format(
            hypothesis=hypothesis_text,
            evidence=_format_evidence(evidence_docs),
            rule_score=float(verification_result.get("rule_score", 0.0)),
            caevn_score=float(verification_result.get("caevn_score", 0.0)),
            v_score=float(verification_result.get("score", 0.0)),
            coverage=float(verification_result.get("coverage", 0.0)),
            contradiction=float(
                verification_result.get("contradiction_severity", 0.0)),
            decision=verification_result.get("decision", "?"),
        )
        return self.llm.generate(
            prompt,
            temperature=LLM_TEMPERATURE_EXPLANATION,
            max_tokens=LLM_EXPLANATION_MAX_TOKENS,
        )
