"""Pareto-front figure for the MOPSO section (campaign v3).

Reads results/mopso_results.json (unique objective points) and renders
handoff rate vs hallucination rate with the selected knee annotated.
F1 is 0.935 at every archive point (gate-independent alarm F1), so the
trade-off is 2-D; stated in the caption.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/mopso_results.json"))
op = json.load(open("results/operating_point.json"))

pts = {}
for p in d["pareto_front"]:
    o = p["objectives"]
    pts[(round(o["handoff_rate"], 4), round(o["hallucination"], 4))] = p["params"]
xy = sorted(pts)
xs = [a for a, _ in xy]
ys = [b for _, b in xy]
kx = round(op["objectives"]["handoff_rate"], 4)
ky = round(op["objectives"]["hallucination"], 4)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# elsarticle 3p is single column (~6.5in text width). Draw at the width
# the figure is actually placed at, so label sizes match the body text.
fig, ax = plt.subplots(figsize=(6.1, 3.0))
ax.plot(xs, ys, "-o", color="#1f4e79", lw=2, ms=5, zorder=3)
ax.scatter([kx], [ky], s=110, facecolors="none", edgecolors="#b54848",
           linewidths=2, zorder=4)
ax.annotate("selected knee\n(handoff .59, halluc .11)",
            xy=(kx, ky), xytext=(kx + 0.06, ky + 0.05),
            fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.set_xlabel("human handoff rate (1 $-$ proceed)")
ax.set_ylabel("hallucination rate")
ax.grid(alpha=0.25, ls="--", lw=0.5)
ax.set_xlim(0.5, 1.02)
fig.tight_layout(pad=0.4)
out = Path("paper/figures/pareto_front_v3.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "| points:", len(xs))
