#!/usr/bin/env python3
"""Plot measured representation costs and clearly separated fitting-time estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ("gpt2", "gemma2", "qwen9b")
METHODS = ("ica", "sae")
COLORS = {"ica": "#3B609C", "sae": "#BA604B"}
LABELS = {"ica": "ICA", "sae": "Public SAE"}


def load_results(results: Path) -> dict[str, dict]:
    records = {}
    for model in MODELS:
        path = results / f"{model}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if set(record["methods"]) != set(METHODS):
            raise ValueError(f"unexpected methods in {path}")
        records[model] = record
    return records


def grouped_bars(ax, records: dict[str, dict], getter, ylabel: str) -> None:
    x = np.arange(len(MODELS), dtype=float)
    width = 0.34
    for offset, method in zip((-0.5, 0.5), METHODS, strict=True):
        values = [getter(records[model]["methods"][method]) for model in MODELS]
        ax.bar(
            x + offset * width,
            values,
            width,
            color=COLORS[method],
            label=LABELS[method],
            zorder=3,
        )
    ax.set_xticks(x, ("GPT-2\nSmall", "Gemma 2\n2B", "Qwen 3.5\n9B"))
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument(
        "--fit-estimates", type=Path, default=HERE / "fitting-time-estimates.json"
    )
    parser.add_argument("--output", type=Path, default=HERE / "figures" / "efficiency.png")
    args = parser.parse_args()

    records = load_results(args.results)
    estimates = json.loads(args.fit_estimates.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.65))

    grouped_bars(
        axes[0], records, lambda r: r["parameter_bytes"] / 2**20, "Memory (MiB / layer)"
    )
    grouped_bars(
        axes[1], records, lambda r: r["latency"]["encode"]["median"],
        "Encode latency (µs / activation)",
    )
    grouped_bars(
        axes[2], records, lambda r: r["latency"]["forward"]["median"],
        "Encode–decode latency (µs / activation)",
    )

    ax = axes[3]
    x = np.arange(len(MODELS), dtype=float)
    width = 0.34
    ranges = [estimates["models"][model]["ica"] for model in MODELS]
    lower = np.asarray([float(value["lower"]) for value in ranges])
    upper = np.asarray([float(value["upper"]) for value in ranges])
    midpoint = np.sqrt(lower * upper)
    ax.bar(x, midpoint, width, color=COLORS["ica"], zorder=3)
    ax.errorbar(
        x,
        midpoint,
        yerr=np.vstack((midpoint - lower, upper - midpoint)),
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=4,
    )
    ax.set_xticks(x, ("GPT-2\nSmall", "Gemma 2\n2B", "Qwen 3.5\n9B"))
    ax.set_ylabel("Fitting time (minutes / layer)")
    ax.set_yscale("log")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.text(0.5, 0.98, "two measured layers", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="#555555")

    titles = (
        "A · Parameter memory",
        "B · Encode",
        "C · Encode–decode",
        "D · Measured ICA fitting time",
    )
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title, loc="left", fontweight="bold", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.23, top=0.78, wspace=0.55)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
