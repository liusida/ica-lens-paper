"""Plot layerwise distributions of component-level suffix-sweep ERF."""

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

from .erf_gradient_figure import _basis_sizes, _paper_style, _titles

BIN_EDGES = np.asarray([1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, np.inf])
BIN_LABELS = ("[1, 2)", "[2, 3)", "[3, 4)", "[4, 8)", "[8, 16)", "[16, 32)", "[32, ∞)")
BIN_COLORS = ("#4F8A63", "#8FBE85", "#B5D1A4", "#E7B84B", "#CC7445", "#9A78AE", "#684783")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="icalens experiment figure erf-suffix-sweep", description=__doc__
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panel-titles", default=None, help="Comma-separated titles.")
    parser.add_argument(
        "--top-k",
        default="15",
        help="Recorded rank threshold to plot, or 'all' to create one figure set per threshold.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    experiment = args.experiment.expanduser().resolve()
    thresholds = _selected_thresholds(experiment, args.top_k)
    explicit_output = args.output.expanduser().resolve() if args.output is not None else None
    for top_k in thresholds:
        if explicit_output is None:
            output = experiment / "figures" / f"erf-suffix-sweep-top{top_k}"
        elif len(thresholds) == 1:
            output = explicit_output
        else:
            output = explicit_output.with_name(f"{explicit_output.name}-top{top_k}")
        for path in render(
            experiment,
            output_prefix=output,
            panel_titles=args.panel_titles,
            top_k=top_k,
            force=args.force,
        ):
            print(path)


def _selected_thresholds(experiment: Path, value: str) -> list[int]:
    run_path = experiment / "run.json"
    if not run_path.is_file():
        raise ValueError(f"missing suffix-sweep ERF run manifest: {experiment}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    recorded = [int(item) for item in run.get("resolved", {}).get("rank_thresholds", [])]
    if not recorded:
        raise ValueError(f"suffix-sweep ERF run records no rank thresholds: {experiment}")
    if value.lower() == "all":
        return recorded
    try:
        selected = int(value)
    except ValueError as error:
        raise ValueError("--top-k must be an integer or 'all'") from error
    if selected not in recorded:
        raise ValueError(
            f"top-k {selected} was not recorded; choose one of "
            f"{','.join(map(str, recorded))}, or 'all'"
        )
    return [selected]


def render(
    experiment: Path,
    *,
    output_prefix: Path,
    panel_titles: str | None,
    top_k: int,
    force: bool,
) -> list[Path]:
    run_path = experiment / "run.json"
    if not run_path.is_file():
        raise ValueError(f"missing suffix-sweep ERF run manifest: {experiment}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    complete = run.get("status") == "complete"
    thresholds = [int(value) for value in run["resolved"].get("rank_thresholds", [])]
    if top_k not in thresholds:
        raise ValueError(
            f"top-k {top_k} was not recorded; choose one of {','.join(map(str, thresholds))}"
        )
    rows = _result_rows(experiment, complete=complete)
    if not rows:
        raise ValueError(f"suffix-sweep ERF experiment has no completed components: {experiment}")
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
            model_rows = [
                row for row in rows if row["model"] == label and int(row["top_k"]) == top_k
            ]
            layers = [int(layer) for layer in run["resolved"]["lenses"][label]["layers"]]
            fractions = np.zeros((len(BIN_LABELS), len(layers)), dtype=float)
            completed_per_layer = []
            for column, layer in enumerate(layers):
                layer_rows = [row for row in model_rows if int(row["layer"]) == layer]
                completed_per_layer.append(len(layer_rows))
                if complete and len(layer_rows) != int(run["resolved"]["components_per_layer"]):
                    raise ValueError(f"{label} layer {layer} has incomplete component results")
                values = np.asarray(
                    [
                        float(row["suffix_erf_mean"])
                        for row in layer_rows
                        if row.get("suffix_erf_mean") not in (None, "")
                    ]
                )
                if not len(values):
                    continue
                bins = np.searchsorted(BIN_EDGES, values, side="right") - 1
                denominator = int(run["resolved"]["components_per_layer"])
                fractions[:, column] = [
                    np.sum(bins == index) / denominator for index in range(len(BIN_LABELS))
                ]
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
        _caption(labels, titles, run, rows=rows, complete=complete, top_k=top_k),
        encoding="utf-8",
    )
    return outputs


def _result_rows(experiment: Path, *, complete: bool) -> list[dict[str, Any]]:
    summary = experiment / "summary.csv"
    if complete and summary.is_file():
        return list(csv.DictReader(summary.open(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    for path in sorted((experiment / "components").glob("*/layer_*/C*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(
                {
                    "model": str(value["model_label"]),
                    "layer": int(value["layer"]),
                    "component": int(value["component"]),
                    "top_k": int(threshold),
                    "suffix_erf_mean": result["suffix_erf_mean"],
                }
                for threshold, result in value["threshold_results"].items()
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return rows


def _caption(
    labels: list[str],
    titles: list[str],
    run: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    complete: bool,
    top_k: int,
) -> str:
    count = int(run["resolved"]["components_per_layer"])
    seed = int(run["resolved"]["seed"])
    panels = ", ".join(f"{label} ({title})" for label, title in zip(labels, titles, strict=True))
    partial = ""
    if not complete:
        expected = count * sum(len(run["resolved"]["lenses"][label]["layers"]) for label in labels)
        plotted = sum(int(row["top_k"]) == top_k for row in rows)
        partial = f" Partial visualization based on {plotted} of {expected} planned components."
    return (
        "Layerwise distribution of suffix-sweep effective receptive field. "
        f"Each layer contains {count} randomly sampled components (seed {seed}); each component "
        f"is evaluated at the top-{top_k} rank threshold and summarized by the mean "
        "exact-or-bracketed first-recovery length over all dominant-tail occurrences. "
        "Suffix lengths 1 through 10 are exact; later lengths use progressively coarser geometric "
        "steps and geometric-midpoint estimates. An occurrence that never reaches the threshold "
        "is assigned its full available "
        "context length. Bars extrapolate sampled fractions to the model's full ICA basis. "
        f"Panels: {panels}.{partial}\n"
    )
