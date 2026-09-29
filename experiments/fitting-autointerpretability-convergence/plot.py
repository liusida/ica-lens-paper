#!/usr/bin/env python3
"""Plot individual and aggregate autointerpretability convergence trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.ticker import MaxNLocator
from prepare import EVALUATED_CHECKPOINTS, LAYERS
from trajectory import parse_layers

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "runs/evaluated-tinker"
DEFAULT_PREPARATION = ROOT / "runs/prepared"
DEFAULT_OUTPUT = ROOT / "figures"
SCORE_FILE = re.compile(r"feature_(\d+)\.json")
ICA_BLUE = "#3D5F99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument(
        "--fit-statistics",
        type=Path,
        help="Cached output of compute_fit_statistics.py for the all-layer figure.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument(
        "--n-components",
        type=int,
        help="Plot the first N usable matched cohort positions (default: all available).",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_features(preparation: Path) -> tuple[dict[int, dict[int, int]], int]:
    result: dict[int, dict[int, int]] = {}
    counts: set[int] = set()
    for iteration in EVALUATED_CHECKPOINTS:
        for layer in LAYERS:
            path = (
                preparation
                / f"iter-{iteration:03d}"
                / f"layer_{layer:02d}"
                / "ica"
                / "selection.json"
            )
            selection = json.loads(path.read_text(encoding="utf-8"))
            accepted = selection.get("accepted")
            if not isinstance(accepted, list) or not accepted:
                raise ValueError(f"expected a nonempty accepted-feature list: {path}")
            counts.add(len(accepted))
            mapping = {int(row["feature"]): position for position, row in enumerate(accepted)}
            if len(mapping) != len(accepted):
                raise ValueError(f"feature IDs are not unique: {path}")
            key = layer
            if key in result and result[key] != mapping:
                raise ValueError(f"cohort order changed at iteration {iteration}, layer {layer}")
            result[key] = mapping
    if len(counts) != 1:
        raise ValueError("prepared checkpoints do not have one consistent cohort size")
    return result, counts.pop()


def read_rows(input_root: Path, preparation: Path) -> tuple[list[dict[str, Any]], int, set[int]]:
    feature_positions, n_features = selected_features(preparation)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    disabled_positions: set[int] = set()
    for iteration in EVALUATED_CHECKPOINTS:
        for layer in LAYERS:
            directory = input_root / f"iter-{iteration:03d}" / f"layer_{layer:02d}/ica/results"
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("feature_*.json")):
                match = SCORE_FILE.fullmatch(path.name)
                if match is None:
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                feature = int(match.group(1))
                if feature not in feature_positions[layer]:
                    raise ValueError(f"unexpected feature C{feature}: {path}")
                position = feature_positions[layer][feature]
                if record.get("status") == "disabled":
                    disabled_positions.add(position)
                    continue
                if record.get("status") != "complete":
                    continue
                key = (iteration, layer, position)
                if key in seen:
                    raise ValueError(f"duplicate result for {key}: {path}")
                scores = {
                    name: float(record[name])
                    for name in ("combined_score", "top_score", "random_score")
                }
                if not all(math.isfinite(value) for value in scores.values()):
                    raise ValueError(f"non-finite score: {path}")
                seen.add(key)
                rows.append(
                    {
                        "iteration": iteration,
                        "layer": layer,
                        "cohort_position": position,
                        "feature": feature,
                        **scores,
                    }
                )
    if not rows:
        raise ValueError(f"no completed feature results found under {input_root}")
    complete_keys = {(row["iteration"], row["layer"], row["cohort_position"]) for row in rows}
    matched_positions = {
        position
        for position in range(n_features)
        if position not in disabled_positions
        and all(
            (iteration, layer, position) in complete_keys
            for iteration in EVALUATED_CHECKPOINTS
            for layer in LAYERS
        )
    }
    return rows, n_features, matched_positions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "iteration",
        "layer",
        "cohort_position",
        "feature",
        "combined_score",
        "top_score",
        "random_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["layer"], row["cohort_position"], row["iteration"]))
        )


def rc_params() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 7,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def style_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="0.89", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_xscale("symlog", linthresh=2, linscale=1)
    axis.set_xlim(-0.15, 215)
    axis.set_xticks(EVALUATED_CHECKPOINTS)
    axis.set_xticklabels([str(value) for value in EVALUATED_CHECKPOINTS], rotation=45)
    axis.set_ylim(-0.15, 1.0)


def trajectory_matrix(
    rows: list[dict[str, Any]],
    layer: int,
    n_features: int,
    score_name: str = "top_score",
) -> np.ndarray:
    matrix = np.full((n_features, len(EVALUATED_CHECKPOINTS)), np.nan)
    iteration_index = {value: index for index, value in enumerate(EVALUATED_CHECKPOINTS)}
    for row in rows:
        if row["layer"] == layer:
            matrix[row["cohort_position"], iteration_index[row["iteration"]]] = row[score_name]
    # A disabled or unfinished checkpoint removes the whole component trajectory
    # from longitudinal summaries rather than changing N across iterations.
    matrix[~np.all(np.isfinite(matrix), axis=1)] = np.nan
    return matrix


def save_figure(figure: Any, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def subplot_grid(plt: Any) -> tuple[Any, np.ndarray]:
    ncols = min(2, len(LAYERS))
    nrows = math.ceil(len(LAYERS) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.9, 2.45 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    return figure, axes.ravel()


def plot_detailed(plt: Any, rows: list[dict[str, Any]], n_features: int, stem: Path) -> None:
    with plt.rc_context(rc_params()):
        figure, axes = subplot_grid(plt)
        for panel, (axis, layer) in enumerate(zip(axes, LAYERS, strict=True)):
            matrix = trajectory_matrix(rows, layer, n_features)
            for values in matrix:
                axis.plot(
                    EVALUATED_CHECKPOINTS,
                    values,
                    color=ICA_BLUE,
                    linewidth=0.55,
                    alpha=0.28,
                )
            style_axis(axis)
            axis.set_title(
                f"{chr(ord('A') + panel)} · Qwen 3.5 9B Base · L{layer}",
                loc="left",
            )
            axis.set_xlabel("FastICA iteration")
        for axis in axes[len(LAYERS) :]:
            axis.set_visible(False)
        for axis in axes[::2]:
            axis.set_ylabel("Autointerpretability score")
        figure.subplots_adjust(
            left=0.09, right=0.99, bottom=0.11, top=0.95, wspace=0.13, hspace=0.38
        )
        save_figure(figure, stem)
        plt.close(figure)


def plot_aggregate(
    plt: Any,
    rows: list[dict[str, Any]],
    n_features: int,
    analysis_count: int,
    stem: Path,
) -> dict[int, list[int]]:
    counts: dict[int, list[int]] = {}
    with plt.rc_context(rc_params()):
        figure, axes = subplot_grid(plt)
        for panel, (axis, layer) in enumerate(zip(axes, LAYERS, strict=True)):
            matrix = trajectory_matrix(rows, layer, n_features)
            n = np.sum(np.isfinite(matrix), axis=0)
            counts[layer] = n.tolist()
            mean = np.full(matrix.shape[1], np.nan)
            low = np.full(matrix.shape[1], np.nan)
            high = np.full(matrix.shape[1], np.nan)
            for index in range(matrix.shape[1]):
                values = matrix[:, index]
                values = values[np.isfinite(values)]
                if len(values):
                    mean[index], low[index], high[index] = _bootstrap_mean(
                        values, _bootstrap_seed(layer, EVALUATED_CHECKPOINTS[index])
                    )
            color = ICA_BLUE
            axis.errorbar(
                EVALUATED_CHECKPOINTS,
                mean,
                yerr=[mean - low, high - mean],
                color=color,
                linewidth=1.5,
                marker="D",
                markersize=3.5,
                capsize=2.5,
                zorder=4,
            )
            partial = (n > 0) & (n < analysis_count)
            axis.scatter(
                np.asarray(EVALUATED_CHECKPOINTS)[partial],
                mean[partial],
                s=18,
                facecolors="white",
                edgecolors=color,
                linewidths=0.8,
                zorder=4,
            )
            style_axis(axis)
            axis.set_title(
                f"{chr(ord('A') + panel)} · Qwen 3.5 9B Base · L{layer}",
                loc="left",
            )
            axis.set_xlabel("FastICA iteration")
        for axis in axes[len(LAYERS) :]:
            axis.set_visible(False)
        for axis in axes[::2]:
            axis.set_ylabel("Mean autointerpretability score")
        figure.subplots_adjust(
            left=0.09, right=0.99, bottom=0.11, top=0.95, wspace=0.13, hspace=0.38
        )
        save_figure(figure, stem)
        plt.close(figure)
    return counts


def plot_all_layers(
    plt: Any,
    rows: list[dict[str, Any]],
    n_features: int,
    fit_statistics: dict[str, Any] | None,
    stem: Path,
) -> list[int]:
    matrix = np.vstack([trajectory_matrix(rows, layer, n_features) for layer in LAYERS])
    counts = np.sum(np.isfinite(matrix), axis=0)
    mean = np.full(matrix.shape[1], np.nan)
    low = np.full(matrix.shape[1], np.nan)
    high = np.full(matrix.shape[1], np.nan)
    for index, iteration in enumerate(EVALUATED_CHECKPOINTS):
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        if len(values):
            mean[index], low[index], high[index] = _bootstrap_mean(
                values, _bootstrap_seed(-1, iteration)
            )
    with plt.rc_context(rc_params()):
        figure, axis = plt.subplots(figsize=(5.25, 3.05))
        auto_artist = axis.errorbar(
            EVALUATED_CHECKPOINTS,
            mean,
            yerr=[mean - low, high - mean],
            color=ICA_BLUE,
            linewidth=1.5,
            marker="D",
            markersize=3.5,
            capsize=2.5,
            zorder=4,
            label="Autointerpretability",
        )
        style_axis(axis)
        auto_min = float(np.nanmin(low))
        auto_max = float(np.nanmax(high))
        auto_margin = 0.08 * (auto_max - auto_min)
        axis.set_ylim(auto_min - auto_margin, auto_max + auto_margin)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_title("Qwen 3.5 9B Base · all four layers", loc="left", pad=48)
        axis.set_ylabel("Mean autointerpretability score")
        handles = [auto_artist]
        labels = ["Autointerpretability"]
        if fit_statistics is not None:
            records = {int(row["layer"]): row for row in fit_statistics["layers"]}
            metrics = (
                ("logcosh_deviation", "Logcosh deviation", "#4F8A72"),
                ("excess_kurtosis", "Excess kurtosis", "#A65D57"),
            )
            for metric_index, (name, ylabel, color) in enumerate(metrics):
                values = np.concatenate(
                    [np.asarray(records[layer][name], dtype=np.float64) for layer in LAYERS], axis=1
                )
                mean_values = values.mean(axis=1)
                median_values = np.median(values, axis=1)
                metric_axis = axis.twinx()
                if metric_index == 1:
                    metric_axis.spines["right"].set_position(("axes", 1.16))
                (mean_artist,) = metric_axis.plot(
                    EVALUATED_CHECKPOINTS,
                    mean_values,
                    color=color,
                    linewidth=1.4,
                    marker="o",
                    markersize=3.0,
                )
                (median_artist,) = metric_axis.plot(
                    EVALUATED_CHECKPOINTS,
                    median_values,
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                    marker="s",
                    markersize=2.8,
                )
                lower = float(min(mean_values.min(), median_values.min()))
                upper = float(max(mean_values.max(), median_values.max()))
                margin = 0.08 * (upper - lower)
                metric_axis.set_ylim(lower - margin, upper + margin)
                metric_axis.set_ylabel(ylabel)
                metric_axis.spines["top"].set_visible(False)
                metric_axis.spines["right"].set_color(color)
                metric_axis.tick_params(axis="y", colors=color)
                metric_axis.yaxis.label.set_color(color)
                handles.extend([mean_artist, median_artist])
                labels.extend([f"{ylabel} mean", f"{ylabel} median"])
        axis.set_xlabel("FastICA iteration")
        axis.legend(
            handles,
            labels,
            frameon=False,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            columnspacing=1.1,
            handlelength=2.0,
        )
        figure.subplots_adjust(
            left=0.13,
            right=0.73 if fit_statistics is not None else 0.98,
            bottom=0.20,
            top=0.72,
        )
        save_figure(figure, stem)
        plt.close(figure)
    return counts.tolist()


def _bootstrap_mean(scores: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Match the main autointerpretability figure's 95% bootstrap mean CI."""
    mean = float(scores.mean())
    if len(scores) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(scores, size=(10_000, len(scores)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high)


def _bootstrap_seed(layer: int, iteration: int) -> int:
    digest = hashlib.sha256(f"17:{layer}:{iteration}:ica".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> None:
    global LAYERS
    args = parse_args()
    LAYERS = parse_layers(args.layers)
    input_root = args.input.expanduser().resolve()
    preparation = args.preparation.expanduser().resolve()
    fit_statistics_path = (
        args.fit_statistics.expanduser().resolve()
        if args.fit_statistics is not None
        else input_root.parent / "fit-statistics.json"
    )
    fit_statistics = (
        json.loads(fit_statistics_path.read_text(encoding="utf-8"))
        if fit_statistics_path.is_file()
        else None
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    detailed = output / "autointerpretability-trajectories"
    aggregate = output / "autointerpretability-convergence"
    all_layers = output / "autointerpretability-convergence-all-layers"
    targets = (
        [detailed.with_suffix(suffix) for suffix in (".png", ".pdf", ".txt")]
        + [aggregate.with_suffix(suffix) for suffix in (".png", ".pdf", ".txt")]
        + [all_layers.with_suffix(suffix) for suffix in (".png", ".pdf", ".txt")]
        + [output / "autointerpretability-convergence-data.csv"]
    )
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to replace existing outputs without --force: " + ", ".join(map(str, existing))
        )
    rows, n_features, matched_positions = read_rows(input_root, preparation)
    if args.n_components is not None:
        if args.n_components < 1:
            raise ValueError("--n-components must be positive")
        available = sorted(matched_positions)
        if args.n_components > len(available):
            raise ValueError(
                f"requested {args.n_components} components, but only "
                f"{len(available)} usable matched positions are available"
            )
        matched_positions = set(available[: args.n_components])
    rows = [row for row in rows if row["cohort_position"] in matched_positions]
    write_csv(output / "autointerpretability-convergence-data.csv", rows)
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib.pyplot as plt

        plot_detailed(plt, rows, n_features, detailed)
        counts = plot_aggregate(plt, rows, n_features, len(matched_positions), aggregate)
        all_layer_counts = plot_all_layers(plt, rows, n_features, fit_statistics, all_layers)
    complete_positions = len(matched_positions)
    detailed.with_suffix(".txt").write_text(
        f"Individual top-fragment autointerpretability scores for "
        f"{len(matched_positions)} matched "
        "persistent "
        f"FastICA rows at Qwen layers {', '.join(map(str, LAYERS))}. Thin lines identify "
        "cohort positions; a disabled component excludes its matched position from all "
        "layers, and partially completed positions are omitted. "
        "the iteration axis uses symmetric-log spacing to expose the densely sampled "
        "early trajectory, and missing evaluations remain gaps. Current complete "
        "matched trajectories: "
        f"{complete_positions}/{n_features}.\n",
        encoding="utf-8",
    )
    count_text = "; ".join(
        f"L{layer}: "
        + ", ".join(
            f"{iteration}={count}"
            for iteration, count in zip(EVALUATED_CHECKPOINTS, counts[layer], strict=True)
        )
        for layer in LAYERS
    )
    aggregate.with_suffix(".txt").write_text(
        "Mean top-fragment autointerpretability score across available matched FastICA rows. "
        "Capped error bars show deterministic 95% bootstrap confidence intervals "
        "for the mean; "
        f"hollow points have fewer than the selected {len(matched_positions)} rows. "
        "The iteration "
        "axis uses "
        "symmetric-log spacing. Counts by iteration: "
        f"{count_text}.\n",
        encoding="utf-8",
    )
    all_layers.with_suffix(".txt").write_text(
        "Mean top-fragment autointerpretability score pooled across Qwen layers "
        f"{', '.join(map(str, LAYERS))} and {len(matched_positions)} matched cohort "
        "positions per layer. Panel A's capped error bars show a deterministic 95% "
        "bootstrap confidence interval for the pooled mean. When fit statistics are "
        "provided, panels B and C show the pooled mean and median population logcosh "
        "deviation and excess kurtosis. Counts by iteration: "
        + ", ".join(
            f"{iteration}={count}"
            for iteration, count in zip(EVALUATED_CHECKPOINTS, all_layer_counts, strict=True)
        )
        + ".\n",
        encoding="utf-8",
    )
    print(f"Wrote detailed and aggregate figures to {output}")
    print(f"Complete matched trajectories: {complete_positions}/{n_features}")


if __name__ == "__main__":
    main()
