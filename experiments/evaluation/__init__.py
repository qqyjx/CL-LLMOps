"""Evaluation package: metrics + action ground-truth rules.

metrics.py       -- honest metric computation from pipeline episode records,
                    with an evaluation-time hallucination judge that is
                    SEPARATE from the deployment verifier (CAEVN/FIS).
action_rules.py  -- deterministic (fault_class, severity) -> A0-A4 rule
                    tables derived from IEEE 1159 / ISO 10816 / IEC 61400,
                    with a LaTeX export for the paper appendix.
"""

from .action_rules import (
    get_class_names, severity_of, ground_truth_action,
    action_level_for, SEVERITY_TO_ACTION, export_latex,
)
from .metrics import (
    HallucinationJudge, RuleBasedJudge, NLIJudge, get_judge,
    parse_structured, diagnosis_to_class_id, compute_metrics,
)

__all__ = [
    "get_class_names", "severity_of", "ground_truth_action",
    "action_level_for", "SEVERITY_TO_ACTION", "export_latex",
    "HallucinationJudge", "RuleBasedJudge", "NLIJudge", "get_judge",
    "parse_structured", "diagnosis_to_class_id", "compute_metrics",
]
