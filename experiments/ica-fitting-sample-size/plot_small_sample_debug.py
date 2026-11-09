#!/usr/bin/env python3
"""Render the dimension and Gaussian controls from the completed debug report."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/icalens-debug-mpl")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = HERE / "figures/ica-small-sample-debug"
    if output.with_suffix(".pdf").exists() and not args.force:
        raise SystemExit("pass --force to replace an existing figure")
    report = json.loads((HERE / "work/debug-small-sample/report.json").read_text())
    sweep = report["gpt2_component_sweep"]
    components = np.array([int(key) for key in sweep])
    seeds = np.array([int(key) for key in report["gaussian_controls"]])
    data = {"components": components, "gaussian_seeds": seeds}
    for key in ("training", "unseen_tokens"):
        data[f"gpt2_{key}"] = np.array([v[key]["median_excess_kurtosis"] for v in sweep.values()])
        data[f"gaussian_{key}"] = np.array([v[key]["median_excess_kurtosis"] for v in report["gaussian_controls"].values()])
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
             "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 7.5,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
             "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
        for key, label, color, marker in (("training", "Training tokens", "#3D5F99", "o"),
                                          ("unseen_tokens", "Unseen tokens", "#B45F4D", "s")):
            axes[0].plot(components, data[f"gpt2_{key}"], color=color, marker=marker,
                         markersize=3, linewidth=1.4, label=label)
            axes[1].scatter(seeds, data[f"gaussian_{key}"], color=color, marker=marker, s=16, zorder=3)
        axes[0].set_title("A · GPT-2, 1K fitting tokens", loc="left", fontweight="bold")
        axes[1].set_title("B · Gaussian, 1K × 768", loc="left", fontweight="bold")
        axes[0].set_xlabel("ICA components")
        axes[1].set_xlabel("Independent Gaussian draw")
        axes[0].set_xticks(components)
        axes[1].set_xticks(seeds)
        axes[1].set_xlim(-.3, 2.3)
        axes[0].set_ylabel("Median component excess kurtosis")
        for ax in axes:
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(-.08, 850)
            ax.axhline(0, color=".6", linewidth=.7, linestyle="--", zorder=0)
            ax.grid(axis="y", color=".89", linewidth=.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(.53, 1.01))
        fig.subplots_adjust(left=.13, right=.98, bottom=.22, top=.80, wspace=.16)
        fig.savefig(output.with_suffix(".pdf"))
        fig.savefig(output.with_suffix(".png"), dpi=300)
        plt.close(fig)
    np.savez_compressed(output.with_suffix(".npz"), **data)
    output.with_suffix(".txt").write_text(
        "Finite-sample FastICA controls, all fitted for 50 fixed updates. A: GPT-2 block 11, "
        "the same 1,000 training token activations, PCA whitening retaining 64–768 ICA components; "
        "evaluation uses 10,000 excluded token positions from the same captured corpus, not document-disjoint. "
        "B: three independent standard-Gaussian training matrices of shape 1,000 × 768, "
        "evaluated on independent 10,000-row Gaussian draws. Points report medians across fitted components. "
        "Symmetric-log y axis is linear from -1 to 1; dashed line is Gaussian excess kurtosis zero.\n")


if __name__ == "__main__":
    main()
