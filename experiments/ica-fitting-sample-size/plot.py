#!/usr/bin/env python3
"""Render the paper-style FastICA excess-kurtosis convergence comparison."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ORDER = ("1k", "5k", "10k", "100k", "1m")
COLORS = {"1k": "#777777", "5k": "#5B8C6A", "10k": "#3D5F99", "100k": "#B45F4D", "1m": "#7A5195"}
LABELS = {"1k": "1K", "5k": "5K", "10k": "10K", "100k": "100K", "1m": "1M"}
MARKERS = {"1k": "o", "5k": "s", "10k": "D", "100k": "^", "1m": "v"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    if not args.force and any((args.output / f"ica-fitting-sample-size.{suffix}").exists() for suffix in ("pdf", "png", "txt")):
        raise SystemExit("figure exists; pass --force to replace it")
    args.output.mkdir(parents=True, exist_ok=True)
    data = {}
    for name in ORDER:
        path = args.input / name / "trajectory.npz"
        if path.exists():
            with np.load(path) as archive:
                data[name] = {key: archive[key].copy() for key in archive.files}
    if not data:
        raise SystemExit(f"No trajectories found in {args.input}")
    configuration = json.loads((args.input / next(iter(data)) / "run.json").read_text())
    title = args.title or configuration["model"]["repo_id"]
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ica-lens-mpl"))
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
             "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "legend.fontsize": 7.5, "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        for name in ORDER:
            if name not in data:
                continue
            d = data[name]
            x, y = d["steps"], d["median_excess_kurtosis"]
            ax.plot(x, y, color=COLORS[name], marker=MARKERS[name], markersize=3.0, markevery=5,
                    linewidth=1.45, label=LABELS[name], zorder=2+ORDER.index(name))
            lower = d["q25_excess_kurtosis"]
            upper = d["q75_excess_kurtosis"]
            ax.fill_between(x, lower, upper, color=COLORS[name], alpha=.10, linewidth=0)
        ax.axhline(0, color="#888888", linestyle="--", linewidth=.75, zorder=0)
        ax.set_xlabel("FastICA iteration")
        ax.set_ylabel("Median component excess kurtosis")
        ax.set_yscale("symlog", linthresh=1.0)
        low = min(0.0, *(float(np.min(d["q25_excess_kurtosis"])) for d in data.values()))
        high = max(float(np.max(d["q75_excess_kurtosis"])) for d in data.values())
        ax.set_ylim(low - .08, high * 1.2)
        ax.set_xlim(0, max(float(np.max(d["steps"])) for d in data.values()))
        ax.grid(axis="y", color="#e4e7eb", linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Fitting tokens", frameon=False, ncol=min(5, len(data)), loc="lower center",
                  bbox_to_anchor=(.5, 1.01), title_fontsize=7.5)
        fig.subplots_adjust(left=.13, right=.975, bottom=.17, top=.80)
        for suffix in ("pdf", "png"):
            fig.savefig(args.output / f"ica-fitting-sample-size.{suffix}", dpi=300)
        plt.close(fig)
    np.savez_compressed(args.output / "ica-fitting-sample-size.npz",
                        **{f"{name}_{key}": value for name, d in data.items() for key, value in d.items()})
    (args.output / "ica-fitting-sample-size.txt").write_text(
        f"{title}, residual stream after block {configuration['layer']}, "
        f"{configuration['components']} components, {configuration['steps']} fixed FastICA steps. "
        "Median across ICA components of excess kurtosis (m4/m2^2 - 3) at each FastICA step. "
        "Shading shows the interquartile range across components. Each budget uses nested seeded random subsets of N "
        "token-position activation rows for fitting and for every per-step diagnostic. "
        "The y axis uses a symmetric logarithmic scale with a linear region from -1 to 1. "
        "Dashed line: Gaussian value (0). Shading describes variation across components, not uncertainty across seeds.\n")


if __name__ == "__main__":
    main()
