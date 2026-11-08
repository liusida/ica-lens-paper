#!/usr/bin/env python3
"""Render GPT-2 Small and Gemma 2 2B autointerpretability panels from twin data.

Same size and style as the Qwen panel (SAE vs ICA stronger-tail sign, bootstrap
95% intervals). Writes figures/autointerpretability-<model>.{png,pdf}.

Usage (from the repo root):
  uv run python experiments/autointerpretability/twin-data/plot_twin_data.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
STYLES = {"sae": ("#B45F4D", "o"), "ica": ("#3D5F99", "^")}
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
      "font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.0,
      "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.5,
      "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42}


def render(npz: Path) -> Path:
    d = np.load(npz, allow_pickle=False)
    layers = d["layers"]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.75, 2.55))
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.24, top=0.72)
        ax.set_title(str(d["panel_title"]), loc="left", fontweight="bold", pad=4)
        for m, (method, label) in enumerate(zip(d["method_ids"], d["method_labels"])):
            color, marker = STYLES[str(method)]
            mean = d["mean_scores"][m]
            yerr = (mean - d["ci95_low"][m], d["ci95_high"][m] - mean)
            ax.errorbar(layers, mean, yerr=yerr, label=str(label), color=color, marker=marker,
                        linewidth=1.5, markersize=3.5, elinewidth=0.8, capsize=2.0)
        ax.set_xticks(layers)
        ax.set_ylim(0.0, 0.8)
        ax.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Interpretability score")
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.56, 0.98),
                   columnspacing=1.5, handlelength=2.0)
        stem = OUT / npz.stem
        for ext in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
        plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for npz in sorted(HERE.glob("autointerpretability-*.npz")):
        print(render(npz))


if __name__ == "__main__":
    main()
