"""Plot layerwise distributions of component-level gradient ERF."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

BIN_EDGES = np.asarray([1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, np.inf])
BIN_LABELS = ("[1, 2)", "[2, 3)", "[3, 4)", "[4, 8)", "[8, 16)", "[16, 32)", "[32, ∞)")
BIN_COLORS = (
    "#4F8A63",
    "#8FBE85",
    "#B7D3A8",
    "#E7B84B",
    "#CC7445",
    "#A58AB8",
    "#76558D",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="icalens experiment figure erf-gradient", description=__doc__
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panel-titles", default=None, help="Comma-separated titles.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    experiment = args.experiment.expanduser().resolve()
    output = (
        experiment / "figures" / "erf-gradient"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    paths = render(
        experiment,
        output_prefix=output,
        panel_titles=args.panel_titles,
        force=args.force,
    )
    for path in paths:
        print(path)


def render(
    experiment: Path,
    *,
    output_prefix: Path,
    panel_titles: str | None,
    force: bool,
) -> list[Path]:
    run_path = experiment / "run.json"
    if not run_path.is_file():
        raise ValueError(f"missing gradient ERF run manifest: {experiment}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    complete = run.get("status") == "complete"
    rows = _result_rows(experiment, complete=complete)
    if not rows:
        raise ValueError(f"gradient ERF experiment has no completed components: {experiment}")
    labels = list(run["resolved"].get("lens_order", run["resolved"]["lenses"]))
    titles = _titles(panel_titles, labels, run)
    outputs = [output_prefix.with_suffix(suffix) for suffix in (".png", ".pdf", ".txt")]
    for path in outputs:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    basis_sizes = _basis_sizes(labels, run)
    panel_widths = [len(run["resolved"]["lenses"][label]["layers"]) for label in labels]
    with plt.rc_context(_paper_style()):  # type: ignore[arg-type]
        figure, axes = plt.subplots(
            1,
            len(labels),
            figsize=(6.9, 2.75),
            sharey=True,
            squeeze=False,
            gridspec_kw={"width_ratios": panel_widths},
        )
        for panel_index, (axis, label, title) in enumerate(
            zip(axes.ravel(), labels, titles, strict=True)
        ):
            model_rows = [row for row in rows if row["model"] == label]
            layers = [int(layer) for layer in run["resolved"]["lenses"][label]["layers"]]
            fractions = np.zeros((len(BIN_LABELS), len(layers)), dtype=float)
            completed_per_layer: list[int] = []
            for column, layer in enumerate(layers):
                values = np.asarray(
                    [
                        float(row["gradient_erf_median"])
                        for row in model_rows
                        if int(row["layer"]) == layer
                    ]
                )
                completed_per_layer.append(len(values))
                if complete and len(values) != int(run["resolved"]["components_per_layer"]):
                    raise ValueError(f"{label} layer {layer} has incomplete component results")
                if not len(values):
                    continue
                bins = np.searchsorted(BIN_EDGES, values, side="right") - 1
                fractions[:, column] = [np.mean(bins == index) for index in range(len(BIN_LABELS))]
            fractions *= basis_sizes[panel_index]
            bottom = np.zeros(len(layers))
            for values, color in zip(fractions, BIN_COLORS, strict=True):
                axis.bar(
                    layers,
                    values,
                    bottom=bottom,
                    width=0.96,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                )
                bottom += values
            axis.set_title(title, loc="left", fontweight="bold", pad=3)
            if not complete:
                expected = len(layers) * int(run["resolved"]["components_per_layer"])
                axis.text(
                    0.99,
                    0.98,
                    f"partial: n={sum(completed_per_layer)}/{expected}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color="#687386",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
                )
            axis.set_ylim(0, max(basis_sizes))
            axis.set_xlim(min(layers) - 0.6, max(layers) + 0.6)
            tick_step = max(1, round(len(layers) / 6))
            axis.set_xticks(layers[::tick_step])
            axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if panel_index == 0:
                axis.set_ylabel("Estimated number of components")
        figure.supxlabel("Layer", y=0.06)
        figure.legend(
            handles=[
                Patch(facecolor=color, label=label)
                for color, label in zip(BIN_COLORS, BIN_LABELS, strict=True)
            ],
            loc="upper center",
            ncol=len(BIN_LABELS),
            frameon=False,
            bbox_to_anchor=(0.5, 1.015),
        )
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.23, top=0.82, wspace=0.18)
        figure.savefig(outputs[0], dpi=300, bbox_inches="tight")
        figure.savefig(outputs[1], bbox_inches="tight")
        plt.close(figure)
    outputs[2].write_text(
        _caption(labels, titles, run, rows=rows, complete=complete), encoding="utf-8"
    )
    return outputs


def _result_rows(experiment: Path, *, complete: bool) -> list[dict[str, Any]]:
    summary_path = experiment / "summary.csv"
    if complete and summary_path.is_file():
        return list(csv.DictReader(summary_path.open(encoding="utf-8")))
    rows = []
    for path in sorted((experiment / "components").glob("*/layer_*/C*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": str(item["model_label"]),
                    "layer": int(item["layer"]),
                    "component": int(item["component"]),
                    "gradient_erf_median": float(item["gradient_erf_median"]),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return rows


def _titles(panel_titles: str | None, labels: list[str], run: dict[str, Any]) -> list[str]:
    if panel_titles:
        values = [value.strip() for value in panel_titles.split(",")]
        if len(values) != len(labels):
            raise ValueError("--panel-titles must provide one title per model")
        return values
    known = {
        "openai-community/gpt2": "GPT-2 small",
        "google/gemma-2-2b": "Gemma 2 2B",
        "Qwen/Qwen3.5-9B-Base": "Qwen 3.5 9B Base",
    }
    return [
        known.get(run["resolved"]["lenses"][label]["model"]["repo_id"], label) for label in labels
    ]


def _basis_sizes(labels: list[str], run: dict[str, Any]) -> list[int]:
    known = {
        "openai-community/gpt2": 768,
        "google/gemma-2-2b": 2304,
        "Qwen/Qwen3.5-9B-Base": 4096,
    }
    sizes = [known.get(run["resolved"]["lenses"][label]["model"]["repo_id"]) for label in labels]
    if all(size is not None for size in sizes):
        return [int(size) for size in sizes if size is not None]
    return [1] * len(labels)


def _caption(
    labels: list[str],
    titles: list[str],
    run: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    complete: bool,
) -> str:
    count = int(run["resolved"]["components_per_layer"])
    seed = int(run["resolved"]["seed"])
    panels = ", ".join(f"{label} ({title})" for label, title in zip(labels, titles, strict=True))
    partial_note = ""
    if not complete:
        expected = count * sum(len(run["resolved"]["lenses"][label]["layers"]) for label in labels)
        partial_note = (
            f" Partial visualization based on {len(rows)} of {expected} planned components; "
            "each non-empty layer bar is normalized over the components completed in that layer."
        )
    return (
        "Layerwise distribution of gradient effective receptive field (ERF_grad). "
        f"Each layer contains {count} randomly sampled components (seed {seed}); each "
        "component is summarized by the median influence-weighted geometric token distance "
        "over its stored dominant-tail occurrences. Each stacked bar extrapolates the sampled "
        "fractions to the model's full ICA basis and therefore shows estimated component counts "
        "in the intervals "
        "[1,2), [2,3), [3,4), [4,8), [8,16), [16,32), and [32,infinity). "
        f"Panels: {panels}.{partial_note}\n"
    )


def _paper_style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
