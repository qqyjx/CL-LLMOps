"""Action ground-truth rule tables: (fault_class, severity) -> A0-A4.

Implements the deterministic mapping described in paper Section 5
("Action quality"): each fault type is assigned a severity category
taken from the applicable published standard, and each severity
category maps to one action level of the discrete action space
(paper Eq. action_space):

    A0: monitor    A1: recalibrate    A2: repair
    A3: shutdown   A4: emergency

Severity assignments ENCODE THE PUBLISHED STANDARDS' SEVERITY
CATEGORIES; they are not tuned on any experimental data:

  * pq16 / pq (16 classes): IEEE 1159-2019 severity guidance for
    electromagnetic phenomena. Single disturbances within ride-through
    limits (flicker, notching) are low severity; sustained single
    disturbances (sag, swell, harmonics, transients) are medium;
    compound disturbances involving waveform distortion are high; any
    supply interruption or multi-mode complex disturbance is critical.
  * cwru (10 classes): ISO 10816 vibration severity zones. Zone A
    (healthy) -> none; incipient defects (0.007") -> Zone B, low;
    developed defects (0.014") -> Zone C, medium; severe defects
    (0.021") -> Zone D, high (vibration severity sufficient to cause
    damage; shutdown recommended).
  * care (binary): IEC 61400 wind-turbine alarm procedure. Normal
    operation -> none; a predicted pre-fault state raises a
    maintenance alarm -> medium (scheduled repair intervention).

The severity -> action mapping is monotone by construction:
    none -> A0, low -> A1, medium -> A2, high -> A3, critical -> A4.

`export_latex()` dumps the full table as a LaTeX appendix table
(default target docs/paper_tables/action_rules_appendix.tex; the
paper appendix \\input{}s it after human review).
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Class name registries (must stay in sync with data loaders /
#    framework.alarm.AlarmClassifier) ──

PQ16_CLASS_NAMES = [
    "normal", "sag", "swell", "interruption",
    "harmonic", "flicker", "transient", "notch",
    "sag_harmonic", "swell_harmonic", "sag_transient",
    "swell_transient", "harmonic_flicker", "sag_flicker",
    "interruption_harmonic", "complex_disturbance",
]

PQ_CLASS_NAMES = [
    "Normal", "Sag", "Swell", "Interruption", "Harmonics",
    "Flicker", "Notch", "Spike", "Sag+Harmonics", "Swell+Harmonics",
    "Sag+Flicker", "Swell+Flicker", "Harmonics+Flicker",
    "Interruption+Harmonics", "Sag+Notch", "Swell+Notch",
]

CWRU_CLASS_NAMES = [
    "Normal", "IR007", "IR014", "IR021",
    "OR007", "OR014", "OR021",
    "Ball007", "Ball014", "Ball021",
]

CARE_CLASS_NAMES = ["Normal", "Pre-fault"]

_CLASS_NAMES = {
    "pq16": PQ16_CLASS_NAMES,
    "pq": PQ_CLASS_NAMES,
    "cwru": CWRU_CLASS_NAMES,
    "care": CARE_CLASS_NAMES,
}

# ── Severity assignment per class (encodes the standards' categories,
#    see module docstring; NOT fitted to data) ──

SEVERITY_TO_ACTION = {
    "none": "A0",       # monitor
    "low": "A1",        # recalibrate
    "medium": "A2",     # repair
    "high": "A3",       # shutdown
    "critical": "A4",   # emergency
}

_PQ16_SEVERITY = {
    # IEEE 1159-2019 severity guidance
    "normal": "none",
    "flicker": "low",
    "notch": "low",
    "sag": "medium",
    "swell": "medium",
    "harmonic": "medium",
    "transient": "medium",
    "sag_flicker": "medium",
    "harmonic_flicker": "medium",
    "sag_harmonic": "high",
    "swell_harmonic": "high",
    "sag_transient": "high",
    "swell_transient": "high",
    "interruption": "critical",
    "interruption_harmonic": "critical",
    "complex_disturbance": "critical",
}

_PQ_SEVERITY = {
    # Legacy 16-class PQ naming (same IEEE 1159 guidance)
    "Normal": "none",
    "Flicker": "low",
    "Notch": "low",
    "Sag": "medium",
    "Swell": "medium",
    "Harmonics": "medium",
    "Spike": "medium",
    "Sag+Flicker": "medium",
    "Swell+Flicker": "medium",
    "Harmonics+Flicker": "medium",
    "Sag+Harmonics": "high",
    "Swell+Harmonics": "high",
    "Sag+Notch": "high",
    "Swell+Notch": "high",
    "Interruption": "critical",
    "Interruption+Harmonics": "critical",
}

_CWRU_SEVERITY = {
    # ISO 10816 vibration severity zones A-D
    "Normal": "none",
    "IR007": "low", "OR007": "low", "Ball007": "low",
    "IR014": "medium", "OR014": "medium", "Ball014": "medium",
    "IR021": "high", "OR021": "high", "Ball021": "high",
}

_CARE_SEVERITY = {
    # IEC 61400 alarm procedure
    "Normal": "none",
    "Pre-fault": "medium",
}

_SEVERITY = {
    "pq16": _PQ16_SEVERITY,
    "pq": _PQ_SEVERITY,
    "cwru": _CWRU_SEVERITY,
    "care": _CARE_SEVERITY,
}

_STANDARD = {
    "pq16": "IEEE 1159-2019",
    "pq": "IEEE 1159-2019",
    "cwru": "ISO 10816",
    "care": "IEC 61400",
}


def get_class_names(dataset_name):
    """Class-name list for a dataset (index = integer label)."""
    return _CLASS_NAMES[dataset_name]


def severity_of(dataset_name, fault_class):
    """Severity category for a fault class (id or name).

    Args:
        dataset_name: one of "pq16", "pq", "cwru", "care"
        fault_class: integer label or class-name string
    Returns:
        severity string in {none, low, medium, high, critical}
    """
    if isinstance(fault_class, (int,)) or hasattr(fault_class, "item"):
        fault_class = _CLASS_NAMES[dataset_name][int(fault_class)]
    return _SEVERITY[dataset_name][fault_class]


def action_level_for(dataset_name, fault_class, severity=None):
    """Deterministic action level for (fault_class, severity).

    When `severity` is None, the class's standard-derived severity is
    used; when given (e.g. an LLM-stated severity), the mapping is
    applied to that severity instead, matching paper Section 4
    ("the action level is set by mapping fault type and severity to a
    candidate level").
    """
    if severity is None or severity not in SEVERITY_TO_ACTION:
        severity = severity_of(dataset_name, fault_class)
    return SEVERITY_TO_ACTION[severity]


def ground_truth_action(dataset_name, y):
    """Ground-truth action level A0-A4 for a true label `y`."""
    return action_level_for(dataset_name, int(y))


def full_table(datasets=("pq16", "cwru", "care")):
    """Full rule table as a list of row dicts (for export/inspection)."""
    rows = []
    for ds in datasets:
        for cls_id, name in enumerate(_CLASS_NAMES[ds]):
            sev = _SEVERITY[ds][name]
            rows.append({
                "dataset": ds,
                "class_id": cls_id,
                "class_name": name,
                "standard": _STANDARD[ds],
                "severity": sev,
                "action": SEVERITY_TO_ACTION[sev],
            })
    return rows


_ACTION_DESC = {
    "A0": "monitor", "A1": "recalibrate", "A2": "repair",
    "A3": "shutdown", "A4": "emergency",
}


def export_latex(out_path=None, datasets=("pq16", "cwru", "care")):
    """Render the full rule table as a LaTeX appendix table.

    Args:
        out_path: target .tex path; None returns the string without
            writing. Default (when called via CLI) is
            docs/paper_tables/action_rules_appendix.tex.
    Returns:
        LaTeX source string.
    """
    ds_label = {"pq16": "PQ (16-class)", "pq": "PQ (legacy)",
                "cwru": "CWRU", "care": "CARE"}
    lines = [
        "% Auto-generated by experiments/evaluation/action_rules.py",
        "% (fault_class, severity) -> action rule table; severities",
        "% encode IEEE 1159 / ISO 10816 / IEC 61400 categories.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Deterministic action ground-truth rule table mapping "
        "each fault class to a severity category from the applicable "
        "standard and to an action level $\\mathcal{U}=\\{$A0: monitor, "
        "A1: recalibrate, A2: repair, A3: shutdown, A4: emergency$\\}$.}",
        "\\label{tab:action_rules}",
        "\\begin{tabular}{llllc}",
        "\\toprule",
        "Dataset & Fault class & Standard & Severity & Action \\\\",
        "\\midrule",
    ]
    prev_ds = None
    for row in full_table(datasets):
        ds_cell = ds_label[row["dataset"]] if row["dataset"] != prev_ds else ""
        if prev_ds is not None and ds_cell:
            lines.append("\\midrule")
        prev_ds = row["dataset"]
        name = row["class_name"].replace("_", "\\_")
        lines.append(
            f"{ds_cell} & {name} & {row['standard']} & "
            f"{row['severity']} & {row['action']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    tex = "\n".join(lines)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex)
    return tex


def main():
    import argparse
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Export the action ground-truth rule table as LaTeX")
    parser.add_argument(
        "--out", type=str,
        default=str(project_root / "docs" / "paper_tables"
                    / "action_rules_appendix.tex"))
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"])
    args = parser.parse_args()
    export_latex(args.out, tuple(args.datasets))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
