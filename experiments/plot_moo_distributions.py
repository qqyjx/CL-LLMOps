#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-run distributions of the MOPSO vs NSGA-II indicators (R3-3b).

Table tab:optimizer_compare reports mean and s.d. over the 10 paired runs.
Reviewer 3 asked for multi-run evidence rather than a single hypervolume
number, so this draws the actual per-run distributions behind that table:
hypervolume, inverted generational distance, and generalized spread, with
every run shown as a point next to its box.

Reads  results/moo_comparison.json (per_run.mopso / per_run.nsga2, 10 runs each)
Writes paper/figures/moo_distributions.pdf

    python experiments/plot_moo_distributions.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "moo_comparison.json"
OUT = ROOT / "paper" / "figures" / "moo_distributions.pdf"

PANELS = [("hypervolume", "hypervolume $\\uparrow$"),
          ("igd", "IGD $\\downarrow$"),
          ("spread", "spread $\\downarrow$")]
COLORS = {"mopso": "#1f4e79", "nsga2": "#b54848"}
LABELS = {"mopso": "MOPSO", "nsga2": "NSGA-II"}


def main():
    doc = json.loads(SRC.read_text())
    per_run = doc["per_run"]

    # The document is elsarticle 3p (single column, ~6.5in of text width), so
    # draw at the placed width and let the fonts match the body text.
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(6.1, 2.3))

    for ax, (key, title) in zip(axes, PANELS):
        data = [[r[key] for r in per_run[opt]] for opt in ("mopso", "nsga2")]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True,
                        medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker="", ms=0))
        for patch, opt in zip(bp["boxes"], ("mopso", "nsga2")):
            patch.set_facecolor(COLORS[opt])
            patch.set_alpha(0.25)
            patch.set_edgecolor(COLORS[opt])
        rng = np.random.RandomState(0)
        for i, (vals, opt) in enumerate(zip(data, ("mopso", "nsga2")), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.plot(i + jitter, vals, "o", ms=3.2, color=COLORS[opt],
                    alpha=0.85, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([LABELS["mopso"], LABELS["nsga2"]])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25, ls="--", lw=0.5)
        # p-values live in tab:optimizer_compare; an in-panel annotation
        # collides with whichever optimizer sits low in that panel.

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    n = len(per_run["mopso"])
    print(f"wrote {OUT.relative_to(ROOT)} | {n} runs per optimizer")
    for key, _ in PANELS:
        m = [r[key] for r in per_run["mopso"]]
        g = [r[key] for r in per_run["nsga2"]]
        print(f"  {key:12} MOPSO {np.mean(m):.3f}+/-{np.std(m, ddof=1):.3f}   "
              f"NSGA-II {np.mean(g):.3f}+/-{np.std(g, ddof=1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
