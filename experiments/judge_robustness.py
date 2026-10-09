#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the verification effect survive swapping the hallucination judge? (R3-2a)

The headline released-hallucination reduction is measured by the deterministic
rule judge. That judge agrees with the independent NLI judge on only 54.8% of
verdicts, so a reviewer can fairly ask whether the effect is a property of the
system or of the judge. This script re-derives both arms' released-hallucination
rates under each judge from the same episodes and the same gate decisions:

    Agent-noV releases every hypothesis, so its rate is the fraction of episodes
    the judge flags; CL-LLMOps releases only 'proceed' episodes, so its rate is
    the fraction of episodes that are both flagged and proceeded.

Reads  results/independent_judge.json (audit_rows: 3 datasets x 3 decode seeds
x 160 episodes, each with the gate decision and both judges' verdicts).
Writes results/judge_robustness.json and mirrors the block into
results/results.json so verify_data.py can check the paper against it.

    python experiments/judge_robustness.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGE = ROOT / "results" / "independent_judge.json"
OUT = ROOT / "results" / "judge_robustness.json"
MIRROR = ROOT / "results" / "results.json"
DATASETS = ("pq16", "cwru", "care")
JUDGES = ("rule", "nli")


def rates(rows, judge):
    """Per-seed (noV, ours) released-hallucination rates, then mean/sd."""
    nov, ours = [], []
    for seed in sorted({r["seed"] for r in rows}):
        sub = [r for r in rows if r["seed"] == seed]
        n = len(sub)
        flagged = [r for r in sub if r["judge_outputs"][judge]["hallucinated"]]
        nov.append(len(flagged) / n)
        ours.append(sum(1 for r in flagged if r["decision"] == "proceed") / n)
    agg = lambda v: {"mean": round(statistics.mean(v), 4),
                     "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                     "per_seed": [round(x, 4) for x in v]}
    return agg(nov), agg(ours)


def main():
    doc = json.loads(JUDGE.read_text())
    rows = doc["audit_rows"]
    out = {
        "provenance": "results/independent_judge.json audit_rows "
                      "(3 decode seeds x 160 episodes per dataset, gate decisions and "
                      "both judges' verdicts from the same run)",
        "paper_section": "sec:reliability",
        "definition": "released-hallucination rate = fraction of all episodes in which a "
                      "judge-flagged hypothesis is gated to proceed; Agent-noV proceeds on all "
                      "episodes by construction",
        "n_episodes_per_dataset": len(rows) // len(DATASETS),
        "values": {},
    }
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        out["values"][ds] = {}
        for judge in JUDGES:
            nov, ours = rates(sub, judge)
            out["values"][ds][judge] = {
                "agent_nov": nov,
                "cl_llmops": ours,
                "reduction_pp": round(100 * (nov["mean"] - ours["mean"]), 2),
            }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")

    mirror = json.loads(MIRROR.read_text())
    mirror["judge_robustness"] = out
    MIRROR.write_text(json.dumps(mirror, ensure_ascii=False, indent=2))
    print(f"mirrored into {MIRROR.relative_to(ROOT)}")

    print(f"\n{'dataset':7}{'judge':6}{'Agent-noV':>11}{'CL-LLMOps':>11}{'reduction':>11}")
    for ds in DATASETS:
        for judge in JUDGES:
            v = out["values"][ds][judge]
            print(f"{ds:7}{judge:6}{v['agent_nov']['mean']:11.4f}"
                  f"{v['cl_llmops']['mean']:11.4f}{v['reduction_pp']:10.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
