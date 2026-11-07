"""Paper-style figures for the default SAEBench RAVEL run (ICA vs SAE vs PCA vs Random).

Story: with SAEBench's default RAVEL protocol, ICA isolates entity attributes best.
PCA and a random orthogonal basis (same sign-split, exactly invertible format)
tie each other, so ICA's lead comes from its directions, not from exact
reconstruction; the SAE trails, mostly on city attributes.

Two figures (each png + pdf + caption txt):
- ravel-gemma2-layer12-overall: one Cause vs Isolation panel averaged over the
  six attribute test sets (main text, single column).
- ravel-gemma2-layer12-attributes: one Cause vs Isolation panel per attribute
  test set, rows = entity type (appendix detail).

Follows src/icalens/experiments/figure.py conventions (serif 8 pt, method
colors/markers, 300 dpi).

Usage: python experiments/ravel/trial/plot_default.py [results_dir] [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "official/results"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "official/figures"

# Same as src/icalens/experiments/figure.py (PCA as squares, as in the paper's Figure 7)
STYLES = {
    "ica": ("ICA", "#3D5F99", "^"),
    "sae": ("SAE", "#B45F4D", "o"),
    "pca": ("PCA", "#5B8C6A", "s"),
    "random": ("Random", "#777777", "D"),
}
LEGEND_ORDER = ("ica", "sae", "pca", "random")
DRAW_ORDER = ("random", "pca", "sae", "ica")
MUTED, GRID = "#6b7280", "#e4e7eb"
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
CITY, NOBEL = "city_results", "nobel_prize_winner_results"
ATTRS = [
    (CITY, "city_Country", "Country"),
    (CITY, "city_Continent", "Continent"),
    (CITY, "city_Language", "Language"),
    (NOBEL, "nobel_prize_winner_Country of Birth", "Birth country"),
    (NOBEL, "nobel_prize_winner_Field", "Field"),
    (NOBEL, "nobel_prize_winner_Gender", "Gender"),
]
# RESULTS holds one folder per seed (seed42/, seed<N>/...), each with <method>_layer12.json.
# Points are seed means; with >1 seed, whiskers show the min-max range across seeds.
SEED_DIRS = sorted(p for p in RESULTS.glob("seed*") if p.is_dir()) or [RESULTS]
runs = {m: [json.loads((d / f"{m}_layer12.json").read_text()) for d in SEED_DIRS]
        for m in LEGEND_ORDER}
N_SEEDS = len(SEED_DIRS)

SEEDS_TEXT = "one seed" if N_SEEDS == 1 else f"mean of {N_SEEDS} seeds; whiskers show the range"
SETUP = ("SAEBench RAVEL on Gemma 2 2B (residual stream, layer 12), default configuration "
         f"(top 500 entities, 5000 pairs per attribute, 2 epochs, {SEEDS_TEXT}). Cause: fraction "
         "of interventions that switch the queried attribute to the source entity's value. "
         "Isolation: fraction of interventions that leave the entity's other attributes "
         "unchanged. Grey lines mark equal Disentangle = (Cause + Isolation) / 2. ICA, PCA "
         "and Random use sign-split coordinates of an exactly invertible basis; SAE is "
         "Gemma Scope 16k (average_l0_41).")


def _stats(values):
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def attr_point(method: str, cls: str, key: str):
    s = [r["per_attribute"][cls][key] for r in runs[method]]
    return _stats([x["cause_score"] for x in s]), _stats([x["isolation_score"] for x in s])


def overall_point(method: str):
    m = [r["metrics"] for r in runs[method]]
    return _stats([x["cause_score"] for x in m]), _stats([x["isolation_score"] for x in m])


def tradeoff_axes(ax, points, lo, hi, guides, labels=None, size=40, guide_labels=True):
    """Cause (x) vs Isolation (y) scatter with equal-Disentangle guide lines."""
    for d in guides:
        xs = np.linspace(lo, hi, 50)
        ys = 2 * d - xs
        keep = (ys >= lo) & (ys <= hi)
        if not keep.any():
            continue
        ax.plot(xs[keep], ys[keep], color=GRID, linewidth=0.7, zorder=0)
        if guide_labels:
            x_end = min(hi, 2 * d - lo)
            y_end = 2 * d - x_end
            ax.text(x_end - 0.004 * (hi - lo) / 0.2, y_end + 0.004 * (hi - lo) / 0.2,
                    f"D = {d:.2f}", ha="right", va="bottom", fontsize=5.6, color=MUTED,
                    rotation=-45, rotation_mode="anchor")
    for z, method in enumerate(DRAW_ORDER, start=2):
        label, color, marker = STYLES[method]
        (c, c_lo, c_hi), (i, i_lo, i_hi) = points[method]
        if N_SEEDS > 1:
            ax.errorbar([c], [i], xerr=[[c - c_lo], [c_hi - c]], yerr=[[i - i_lo], [i_hi - i]],
                        fmt="none", ecolor=color, elinewidth=0.8, capsize=0, zorder=z + 5)
        ax.scatter([c], [i], color=color, marker=marker, s=size, zorder=z + 10,
                   edgecolors="white", linewidths=0.7)
        if labels and method in labels:
            dx, dy, ha = labels[method]
            ax.annotate(label, (c, i), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va="center", fontsize=6.8, color=color)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend(fig, y=1.0):
    handles = [plt.Line2D([], [], color=STYLES[m][1], marker=STYLES[m][2], linestyle="",
                          markersize=4.5, label=STYLES[m][0]) for m in LEGEND_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, y), handletextpad=0.2, columnspacing=1.2)


def save(fig, stem, caption):
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    (OUTPUT / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
    print(OUTPUT / f"{stem}.png")


OUTPUT.mkdir(parents=True, exist_ok=True)
with plt.rc_context(PAPER_STYLE):
    # ---------------- main text: overall ----------------
    fig, ax = plt.subplots(figsize=(3.3, 3.05))
    pts = {m: overall_point(m) for m in LEGEND_ORDER}
    tradeoff_axes(ax, pts, 0.60, 0.82, (0.65, 0.70, 0.75, 0.80),
                  labels={"ica": (6, 2, "left"), "sae": (6, -2, "left"),
                          "pca": (6, -7, "left"), "random": (-6, 5, "right")})
    ax.set_xticks([0.60, 0.65, 0.70, 0.75, 0.80])
    ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80])
    ax.set_xlabel("Cause (target attribute changes)")
    ax.set_ylabel("Isolation (other attributes kept)")
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.14, top=0.97)
    save(fig, "ravel-gemma2-layer12-overall",
         "Non-Gaussian directions isolate entity attributes. Cause vs. Isolation averaged "
         "over six attribute test sets (city: country, continent, language; Nobel laureate: "
         "birth country, field, gender). " + SETUP)

    # ---------------- appendix: one panel per attribute test set ----------------
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 5.0), sharex=True, sharey=True)
    lo, hi = 0.35, 1.0
    guides = (0.5, 0.6, 0.7, 0.8, 0.9)
    entity = {CITY: "City", NOBEL: "Nobel"}
    for n, ((cls, key, name), ax) in enumerate(zip(ATTRS, axes.flat)):
        pts = {m: attr_point(m, cls, key) for m in LEGEND_ORDER}
        tradeoff_axes(ax, pts, lo, hi, guides, size=34, guide_labels=(n == 2))
        ax.set_title(f"{'ABCDEF'[n]} · {entity[cls]}: {name}", loc="left",
                     fontweight="bold", pad=3)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("Cause")
    for ax in axes[:, 0]:
        ax.set_ylabel("Isolation")
    legend(fig, y=1.0)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.9, wspace=0.12,
                        hspace=0.22)
    save(fig, "ravel-gemma2-layer12-attributes",
         "RAVEL Cause vs. Isolation for each attribute test set. Each panel trains one mask "
         "for the named attribute; Cause pairs query that attribute and Isolation pairs query "
         "the other two attributes of the same entity type. " + SETUP)
