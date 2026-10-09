#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does re-tuning the gate on a domain's own validation split restore autonomy?

Section 6.1 reports that the single MOPSO operating point, tuned on PQ and
transferred unchanged, is over-conservative outside its tuning domain (proceed
rate 3.6% on CWRU). The natural remedy is per-domain re-calibration. The
cross-domain regression experiment already ran exactly that: for each domain it
derives a MOPSO update on that domain's validation split and replays it against
every domain's regression suite. The in-domain cells of that matrix are the
re-calibration test, so no new compute is needed.

Reads  results/regression_crossdomain.json per_cycle (in_domain rows: the global
operating point as `baseline`, the domain's own re-tuned point as `updated`,
5 update cycles per domain on that domain's regression suite).
Writes results/per_domain_recalibration.json and mirrors it into results.json.

Caveat carried into the artifact: the regression suite oversamples escalated and
disagreement episodes, so absolute rates are not comparable to Table 3; the
baseline-versus-updated contrast within the suite is paired and is what this
analysis reports.

    python experiments/per_domain_recalibration.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "regression_crossdomain.json"
OUT = ROOT / "results" / "per_domain_recalibration.json"
MIRROR = ROOT / "results" / "results.json"
DATASETS = ("pq16", "cwru", "care")
METRICS = ("proceed_rate", "hallucination_rate", "action_accuracy",
           "false_escalation_rate", "escalation_rate", "defer_rate", "accuracy", "f1")


def main():
    doc = json.loads(SRC.read_text())
    cycles = doc["per_cycle"]
    out = {
        "provenance": "results/regression_crossdomain.json per_cycle, in_domain rows "
                      "(baseline = global PQ-tuned operating point, updated = that domain's own "
                      "MOPSO re-tuning on its validation split)",
        "paper_section": "sec:main_results",
        "caveat": "evaluated on each domain's regression suite, which oversamples escalated and "
                  "disagreement episodes; absolute rates are not comparable to tab:main, the "
                  "paired baseline-vs-updated contrast is",
        "values": {},
    }
    for ds in DATASETS:
        sub = [c for c in cycles if c["in_domain"] and c["target_suite"] == ds]
        entry = {"n_cycles": len(sub), "n_episodes": sub[0]["baseline"]["n_episodes"],
                 "retuned_params": doc["source_updates"][ds]["params"]}
        for m in METRICS:
            b = statistics.mean(c["baseline"][m] for c in sub)
            u = statistics.mean(c["updated"][m] for c in sub)
            entry[m] = {"global_operating_point": round(b, 4),
                        "re_tuned": round(u, 4),
                        "delta": round(u - b, 4)}
        out["values"][ds] = entry
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    mirror = json.loads(MIRROR.read_text())
    mirror["per_domain_recalibration"] = out
    MIRROR.write_text(json.dumps(mirror, ensure_ascii=False, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}; mirrored into {MIRROR.relative_to(ROOT)}\n")

    print(f"{'domain':7}{'proceed':>18}{'halluc':>18}{'ActAcc':>18}")
    for ds in DATASETS:
        v = out["values"][ds]
        fmt = lambda m, s=1: (f"{v[m]['global_operating_point']*s:.3f}->"
                              f"{v[m]['re_tuned']*s:.3f}")
        print(f"{ds:7}{fmt('proceed_rate'):>18}{fmt('hallucination_rate'):>18}"
              f"{fmt('action_accuracy',100):>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
