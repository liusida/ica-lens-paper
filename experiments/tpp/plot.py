#!/usr/bin/env python3
"""Plot intended, unintended, and net TPP effects from a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

METHODS = (
    ("ica_custom_sae", "ICA", "#3D5F99", "o"),
    ("unfitted_ica_custom_sae", "Unfitted ICA", "#8FA9CC", "s"),
    ("sae_custom_sae", "SAE", "#B45F4D", "o"),
    ("untrained_sae_matched_l0_custom_sae", "Untrained SAE", "#D29A8E", "s"),
    ("pca_custom_sae", "PCA", "#5B8C6A", "^"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload["methods"]
    first = next(iter(results.values()))
    budgets = [int(value) for value in first["eval_config"]["n_values"]]
    panels = (
        ("intended_diff_only", "Intended change"),
        ("unintended_diff_only", "Unintended change"),
        ("total_metric", "TPP score\n(intended − unintended)"),
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), sharex=True)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.76, wspace=0.33)

    for axis, (metric_suffix, title) in zip(axes, panels, strict=True):
        for method_id, label, color, marker in METHODS:
            metrics = results[method_id]["eval_result_metrics"]["tpp_metrics"]
            values = [metrics[f"tpp_threshold_{budget}_{metric_suffix}"] for budget in budgets]
            axis.plot(
                budgets,
                values,
                color=color,
                marker=marker,
                linewidth=1.35,
                markersize=3.5,
                label=label,
            )
        axis.axhline(0, color="0.65", linewidth=0.7, zorder=0)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(budgets)
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy change")
    figure.supxlabel("Top-k perturbed features", y=0.045)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=5,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
