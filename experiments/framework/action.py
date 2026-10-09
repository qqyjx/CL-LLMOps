"""Stage 4: Action Generation via LLM.

Given a verified hypothesis, generates maintenance/remediation actions.
Uses lower temperature (0.1) than hypothesis stage for more deterministic output.
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_TEMPERATURE_ACTION, LLM_ACTION_MAX_TOKENS


ACTION_PROMPT = """You are an expert industrial maintenance engineer.

Based on the verified diagnosis:
- Fault type: {fault_class}
- Root cause hypothesis: {hypothesis}
- Verification score: {verification_score:.2f}
- Gating decision: {gating_decision}

Choose the action level and generate a maintenance action plan.

Action levels:
- A0: monitor (continue observation, no intervention)
- A1: recalibrate (adjust settings or parameters)
- A2: local repair (targeted component maintenance)
- A3: planned shutdown (scheduled corrective action)
- A4: emergency stop (immediate shutdown and isolate)

Respond with ONE JSON object (no prose outside the JSON) with exactly
these keys:
- "action_level": one of "A0", "A1", "A2", "A3", "A4" (match the fault
  severity; never recommend A0 for faults with safety risk)
- "immediate_actions": list, within 1 hour
- "short_term_repairs": list, within 24 hours
- "preventive_measures": list, within 1 week
- "required_parts": list of parts/tools
- "safety_precautions": list
"""


class ActionGenerator:
    """Generate maintenance actions from verified hypotheses."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def generate(self, alarm_result, hypothesis_result, verification_result):
        """Generate action plan based on verified hypothesis.

        Args:
            alarm_result: dict from AlarmClassifier
            hypothesis_result: dict from HypothesisGenerator
            verification_result: dict with score, decision, alpha

        Returns:
            dict with: action_text, action_embedding, action_class
        """
        decision = verification_result.get("decision", "proceed")

        if decision == "reject":
            return {
                "action_text": "REJECTED: Hypothesis did not pass verification. "
                               "Requesting human expert review.",
                "action_class": "escalate",
                "requires_human": True,
            }

        if decision == "defer":
            return {
                "action_text": "DEFERRED: Hypothesis has moderate confidence. "
                               "Gathering additional evidence before action.",
                "action_class": "monitor",
                "requires_human": False,
            }

        # Proceed: generate action
        prompt = ACTION_PROMPT.format(
            fault_class=alarm_result["class_name"],
            hypothesis=hypothesis_result["selected"],
            verification_score=verification_result["score"],
            gating_decision=decision,
        )

        if self.llm is None:
            raise RuntimeError(
                "ActionGenerator requires an LLM client; none was supplied."
            )
        action_text = self.llm.generate(
            prompt,
            temperature=LLM_TEMPERATURE_ACTION,
            max_tokens=LLM_ACTION_MAX_TOKENS,
        )

        return {
            "action_text": action_text,
            "action_class": self._classify_action(alarm_result),
            "requires_human": False,
        }

    def _classify_action(self, alarm_result):
        """Map fault severity to action urgency class."""
        confidence = alarm_result["confidence"]
        if confidence > 0.9:
            return "immediate"
        elif confidence > 0.7:
            return "scheduled"
        else:
            return "monitor"
