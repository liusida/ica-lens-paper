"""Plot the RAVEL mask-budget sweep (Cause / Isolation vs feature budget).

Usage: python experiments/ravel/trial/plot_budget.py [results_dir]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "pilot-runs/pod-budget"

# Validated categorical order (dataviz reference palette, light mode).
METHODS = [
    ("ica", "ICA", "#2a78d6", "o"),
    ("sae", "Gemma Scope SAE", "#eb6834", "s"),
    ("neurons", "Neurons", "#1baf7a", "^"),
    ("pca", "PCA", "#eda100", "D"),
    ("random", "Random basis", "#e87ba4", "v"),
]
BUDGETS = [1, 10, 100, 1000, None]
X_ALL = 10**4.4  # plotting position for "unlimited"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a65", "#e4e3df", "#fcfcfb"

rows = {}
for f in glob.glob(str(RESULTS / "*_layer12*.json")):
    d = json.load(open(f))
    m = d["metrics"]
    rows[(d["method"], d.get("budget"))] = (m["cause_score"], m["isolation_score"])

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
xs = [b if b else X_ALL for b in BUDGETS]

for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

for key, label, color, marker in METHODS:
    pts = [rows.get((key, b)) for b in BUDGETS]
    cause = [p[0] for p in pts]
    iso = [p[1] for p in pts]
    style = dict(color=color, marker=marker, markersize=7, linewidth=2,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    axes[0].plot(xs, cause, **style)
    axes[1].plot(xs, iso, **style)
    axes[2].plot(cause, iso, **style)
    # direct label at the unlimited end of the trade-off curve
    dy = {"pca": 5, "random": -5, "sae": -6}.get(key, 0)
    axes[2].annotate(label, (cause[-1], iso[-1]), xytext=(8, dy), textcoords="offset points",
                     va="center", fontsize=9, color=INK)

for ax, title, ylabel in ((axes[0], "Cause vs mask budget", "Cause"),
                          (axes[1], "Isolation vs mask budget", "Isolation")):
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["1", "10", "100", "1000", "all"])
    ax.set_xlabel("Max features the mask may swap (k)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, loc="left", fontsize=11)

ax = axes[2]
ax.set_xlim(-0.03, 1.08)
ax.set_ylim(-0.03, 1.06)
ax.set_xlabel("Cause")
ax.set_ylabel("Isolation")
ax.set_title("Trade-off (each point = one budget)", loc="left", fontsize=11)
ax.scatter([1], [1], marker="*", s=180, color=MUTED, zorder=3)
ax.annotate("ideal", (1, 1), xytext=(-8, -12), textcoords="offset points", ha="right",
            fontsize=9, color=MUTED)

axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("RAVEL (SAEBench), Gemma 2 2B layer 12, city: Country + Continent, "
             "single seed, ~70 test pairs per attribute", x=0.01, ha="left", fontsize=11,
             color=MUTED)
fig.tight_layout()
out = RESULTS / "ravel_budget_sweep.png"
fig.savefig(out, dpi=160, facecolor=SURFACE)
print(out)
