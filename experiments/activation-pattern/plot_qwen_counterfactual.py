#!/usr/bin/env python3
"""Plot the focused Qwen L16 ICA counterfactual comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-counterfactual"
OUTPUT = HERE / "figures" / "qwen-l16-counterfactual.png"
COLORS = ("#315f93", "#2f8a5b")


def shade_tokens(ax: plt.Axes, count: int) -> None:
    for index in range(count):
        if index % 2 == 0:
            ax.axvspan(index - 0.5, index + 0.5, color="#eef0f2", zorder=0)


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    with np.load(RESULTS / summary["data"], allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}

    labels = {int(item["component"]): item["label"] for item in summary["components"]}
    components = [int(value) for value in data["component_ids"]]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.35), sharey="row")
    columns = (
        ("Original sentence", "observed"),
        ("Related sentence", "related"),
        ("Counterfactual sentence", "counterfactual"),
    )
    for row, component in enumerate(components):
        for column, (column_title, prefix) in enumerate(columns):
            ax = axes[row, column]
            values = np.abs(data[f"{prefix}_scores"][row])
            token_labels = [str(value) for value in data[f"{prefix}_token_labels"]]
            x = np.arange(len(values))
            shade_tokens(ax, len(values))
            ax.fill_between(x, values, color=COLORS[row], alpha=0.18, zorder=1)
            ax.plot(x, values, color=COLORS[row], marker="o", markersize=3.0, lw=1.8, zorder=2)
            ax.set_xlim(-0.5, len(values) - 0.5)
            ax.grid(axis="y", color="#d9dde1", linewidth=0.7, zorder=-1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(x)
            ax.set_xticklabels(token_labels, rotation=48, ha="right", fontsize=7.2)
            ax.set_title(
                f"{column_title}\nL16 C{component} ({labels[component]})",
                loc="left",
                fontsize=10.5,
                fontweight="bold",
                pad=5,
            )
        axes[row, 0].set_ylabel("|ICA score|", fontsize=10.5)
        row_max = max(
            float(np.abs(data[f"{prefix}_scores"][row]).max())
            for _, prefix in columns
        )
        axes[row, 0].set_ylim(0, row_max * 1.08)

    fig.suptitle("Qwen 3.5 9B Base", fontsize=12, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.225, top=0.87, wspace=0.10, hspace=0.94)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
