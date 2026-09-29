#!/usr/bin/env python3
"""Plot four layerwise statistics from a partial or complete pilot run."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ("ica", "sae", "pca", "random")
DRAW_ORDER = ("random", "pca", "sae", "ica")
MODEL_ORDER = ("gpt2", "gemma2", "qwen9b")
STYLE = {
    "ica": ("ICA", "#3D5F99", "D"),
    "sae": ("SAE", "#B45F4D", "o"),
    "pca": ("PCA", "#5B8C6A", "^"),
    "random": ("Random", "#777777", "s"),
}
TITLES = {
    "gpt2": "GPT-2 Small",
    "gemma2": "Gemma 2 2B",
    "qwen9b": "Qwen 3.5 9B Base",
}
STATISTICS = {
    "mean": ("projection-mean-over-layers.png", "Median |projection mean|", True),
    "variance": ("projection-variance-over-layers.png", "Median projection variance", False),
    "skewness": ("projection-skewness-over-layers.png", "Median |skewness|", True),
    "kurtosis": (
        "projection-kurtosis-over-layers.png",
        "Median kurtosis",
        False,
    ),
    "excess_kurtosis": (
        "projection-excess-kurtosis-over-layers.png",
        "Median excess kurtosis",
        False,
    ),
}


def stored_statistic(statistic: str) -> str:
    return "excess_kurtosis" if statistic == "kurtosis" else statistic


def load_partial_results(
    root: Path,
) -> tuple[dict, dict[tuple[str, int], dict[str, np.ndarray]]]:
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    resolved = run.get("resolved", {})
    if (
        resolved.get("format") != "icalens.kurtosis_variance_comparison"
        or resolved.get("schema_version") not in {2, 3, 4}
    ):
        raise ValueError(f"not a compatible kurtosis-variance run: {root}")
    loaded: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for model, model_config in resolved["models"].items():
        count = int(
            model_config.get("directions_per_method")
            or resolved.get("directions_per_method")
        )
        for layer in model_config["layers"]:
            path = root / model / f"layer_{int(layer):02d}.npz"
            if not path.is_file():
                continue
            try:
                with np.load(path, allow_pickle=False) as archive:
                    values = {key: archive[key] for key in archive.files}
            except (OSError, ValueError):
                continue
            required = [
                f"{method}_{stored_statistic(statistic)}"
                for method in METHODS
                for statistic in STATISTICS
            ]
            if any(
                key not in values
                or values[key].shape != (count,)
                or not np.isfinite(values[key]).all()
                for key in required
            ):
                continue
            loaded[model, int(layer)] = values
    if not loaded:
        raise ValueError(f"run has no valid completed layer checkpoints: {root}")
    return run, loaded


def paper_style() -> dict:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def plot_statistic(
    run: dict,
    loaded: dict[tuple[str, int], dict[str, np.ndarray]],
    statistic: str,
    path: Path,
    aggregation: str,
) -> None:
    _, ylabel, take_absolute = STATISTICS[statistic]
    ylabel = ylabel.replace("Median", "Mean") if aggregation == "mean" else ylabel
    models = [model for model in MODEL_ORDER if model in run["resolved"]["models"]]
    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(
            1, len(models), figsize=(7.15, 2.55), sharey=True, squeeze=False
        )
        axes = axes[0]
        for column, model in enumerate(models):
            axis = axes[column]
            expected = [int(value) for value in run["resolved"]["models"][model]["layers"]]
            completed = [layer for layer in expected if (model, layer) in loaded]
            for zorder, method in enumerate(DRAW_ORDER, start=2):
                points = []
                for layer in completed:
                    samples = loaded[model, layer][
                        f"{method}_{stored_statistic(statistic)}"
                    ]
                    if statistic == "kurtosis":
                        samples = samples + 3.0
                    if take_absolute:
                        samples = np.abs(samples)
                    reducer = np.mean if aggregation == "mean" else np.median
                    points.append(float(reducer(samples)))
                if points:
                    if statistic != "excess_kurtosis" and any(
                        value <= 0 for value in points
                    ):
                        raise ValueError(
                            f"log-scale plot requires positive {statistic} medians; "
                            f"found a nonpositive value for {model} {method}"
                        )
                    label, color, marker = STYLE[method]
                    axis.plot(
                        completed,
                        points,
                        color=color,
                        marker=marker,
                        linewidth=1.35,
                        markersize=3.0,
                        label=label,
                        zorder=zorder,
                    )
            if not completed:
                axis.text(
                    0.5,
                    0.5,
                    "Pending",
                    ha="center",
                    va="center",
                    color="#7a7a7a",
                    transform=axis.transAxes,
                )
            elif len(completed) < len(expected):
                axis.text(
                    0.98,
                    0.04,
                    f"partial: {len(completed)}/{len(expected)} layers",
                    ha="right",
                    va="bottom",
                    fontsize=6.4,
                    color="#6b7280",
                    transform=axis.transAxes,
                )
            axis.set_title(
                f"{chr(65 + column)} · {TITLES.get(model, model)}",
                loc="left",
                fontweight="bold",
                pad=3,
            )
            axis.set_xlim(min(expected) - 0.5, max(expected) + 0.5)
            if statistic == "excess_kurtosis":
                axis.set_yscale("symlog", linthresh=1.0)
                axis.axhline(0.0, color="0.65", linewidth=0.7, linestyle="--", zorder=0)
            else:
                axis.set_yscale("log")
            if statistic == "kurtosis":
                axis.axhline(3.0, color="0.65", linewidth=0.7, linestyle="--", zorder=0)
            axis.grid(axis="y", color="#e4e7eb", linewidth=0.45)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(ylabel)
        figure.supxlabel("Layer", y=0.04)
        handles, labels = next(
            (axis.get_legend_handles_labels() for axis in axes if axis.lines),
            ([], []),
        )
        if handles:
            by_label = dict(zip(labels, handles, strict=True))
            ordered_labels = [STYLE[method][0] for method in METHODS]
            figure.legend(
                [by_label[label] for label in ordered_labels],
                ordered_labels,
                loc="upper center",
                ncol=4,
                frameon=False,
                bbox_to_anchor=(0.5, 1.01),
            )
        figure.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.79, wspace=0.16)
        figure.savefig(path, dpi=300)
        plt.close(figure)


def plot_combined(
    run: dict,
    loaded: dict[tuple[str, int], dict[str, np.ndarray]],
    path: Path,
    aggregation: str,
) -> None:
    models = [model for model in MODEL_ORDER if model in run["resolved"]["models"]]
    statistics = list(STATISTICS)
    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(
            len(statistics),
            len(models),
            figsize=(7.15, 9.0),
            sharex="col",
            sharey="row",
            squeeze=False,
        )
        for row, statistic in enumerate(statistics):
            _, ylabel, take_absolute = STATISTICS[statistic]
            ylabel = ylabel.replace("Median", "Mean") if aggregation == "mean" else ylabel
            for column, model in enumerate(models):
                axis = axes[row, column]
                expected = [
                    int(value) for value in run["resolved"]["models"][model]["layers"]
                ]
                completed = [layer for layer in expected if (model, layer) in loaded]
                for zorder, method in enumerate(DRAW_ORDER, start=2):
                    points = []
                    for layer in completed:
                        samples = loaded[model, layer][
                            f"{method}_{stored_statistic(statistic)}"
                        ]
                        if statistic == "kurtosis":
                            samples = samples + 3.0
                        if take_absolute:
                            samples = np.abs(samples)
                        reducer = np.mean if aggregation == "mean" else np.median
                        points.append(float(reducer(samples)))
                    if points:
                        if statistic != "excess_kurtosis" and any(
                            value <= 0 for value in points
                        ):
                            raise ValueError(
                                f"log-scale plot requires positive {statistic} medians; "
                                f"found a nonpositive value for {model} {method}"
                            )
                        label, color, marker = STYLE[method]
                        axis.plot(
                            completed,
                            points,
                            color=color,
                            marker=marker,
                            linewidth=1.35,
                            markersize=3.0,
                            label=label,
                            zorder=zorder,
                        )
                if not completed:
                    axis.text(
                        0.5,
                        0.5,
                        "Pending",
                        ha="center",
                        va="center",
                        color="#7a7a7a",
                        transform=axis.transAxes,
                    )
                elif row == 0 and len(completed) < len(expected):
                    axis.text(
                        0.98,
                        0.04,
                        f"partial: {len(completed)}/{len(expected)} layers",
                        ha="right",
                        va="bottom",
                        fontsize=6.4,
                        color="#6b7280",
                        transform=axis.transAxes,
                    )
                if row == 0:
                    axis.set_title(
                        f"{chr(65 + column)} · {TITLES.get(model, model)}",
                        loc="left",
                        fontweight="bold",
                        pad=3,
                    )
                axis.set_xlim(min(expected) - 0.5, max(expected) + 0.5)
                if statistic == "excess_kurtosis":
                    axis.set_yscale("symlog", linthresh=1.0)
                    axis.axhline(
                        0.0, color="0.65", linewidth=0.7, linestyle="--", zorder=0
                    )
                else:
                    axis.set_yscale("log")
                if statistic == "kurtosis":
                    axis.axhline(
                        3.0, color="0.65", linewidth=0.7, linestyle="--", zorder=0
                    )
                axis.grid(axis="y", color="#e4e7eb", linewidth=0.45)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
            axes[row, 0].set_ylabel(ylabel)
        figure.supxlabel("Layer", y=0.025)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles, strict=True))
            ordered_labels = [STYLE[method][0] for method in METHODS]
            figure.legend(
                [by_label[label] for label in ordered_labels],
                ordered_labels,
                loc="upper center",
                ncol=4,
                frameon=False,
                bbox_to_anchor=(0.5, 0.998),
            )
        figure.subplots_adjust(
            left=0.085,
            right=0.995,
            bottom=0.075,
            top=0.94,
            hspace=0.22,
            wspace=0.16,
        )
        figure.savefig(path, dpi=300)
        plt.close(figure)


def plot_random_kurtosis_distribution(
    loaded: dict[tuple[str, int], dict[str, np.ndarray]],
    path: Path,
) -> bool:
    layers = (9, 12, 14)
    if any(("gemma2", layer) not in loaded for layer in layers):
        return False
    colors = ("#8FA9CC", "#3D5F99", "#777777")
    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), sharex=True, sharey=True)
        for index, (axis, layer, color) in enumerate(zip(axes, layers, colors, strict=True)):
            values = loaded["gemma2", layer]["random_excess_kurtosis"]
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
            axis.step(ordered, cumulative, where="post", color=color, linewidth=1.5)
            axis.axvline(0, color="0.65", linewidth=0.7, linestyle="--", zorder=0)
            axis.axvline(
                np.median(values), color=color, linewidth=0.9, linestyle=":", zorder=1
            )
            axis.set_xscale("symlog", linthresh=1.0)
            axis.set_title(
                f"{chr(65 + index)} · Gemma 2 2B · L{layer}",
                loc="left",
                fontweight="bold",
            )
            axis.text(
                0.98,
                0.05,
                f"mean {np.mean(values):.2f}\nmedian {np.median(values):.2f}",
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="#4b5563",
                transform=axis.transAxes,
            )
            axis.grid(axis="y", color="#e4e7eb", linewidth=0.45)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Cumulative fraction of directions")
        figure.supxlabel("Random-direction excess kurtosis", y=0.04)
        figure.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.88, wspace=0.16)
        figure.savefig(path, dpi=300)
        plt.close(figure)
    return True


def plot_random_kurtosis_histogram(
    loaded: dict[tuple[str, int], dict[str, np.ndarray]],
    path: Path,
) -> bool:
    layers = (9, 12, 14)
    if any(("gemma2", layer) not in loaded for layer in layers):
        return False
    samples = [loaded["gemma2", layer]["random_excess_kurtosis"] for layer in layers]
    colors = ("#8FA9CC", "#3D5F99", "#777777")
    def transform(value: np.ndarray) -> np.ndarray:
        return np.sign(value) * np.log10(1.0 + np.abs(value))

    def inverse(value: np.ndarray) -> np.ndarray:
        return np.sign(value) * (np.power(10.0, np.abs(value)) - 1.0)

    transformed = np.concatenate([transform(values) for values in samples])
    edges = inverse(np.linspace(transformed.min() - 0.02, transformed.max() + 0.02, 25))
    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), sharex=True, sharey=True)
        for index, (axis, layer, color, values) in enumerate(
            zip(axes, layers, colors, samples, strict=True)
        ):
            axis.hist(
                values,
                bins=edges,
                weights=np.full(len(values), 1.0 / len(values)),
                color=color,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.35,
            )
            axis.axvline(0, color="0.65", linewidth=0.7, linestyle="--", zorder=0)
            axis.axvline(
                np.median(values), color=color, linewidth=0.9, linestyle=":", zorder=2
            )
            axis.set_xscale("symlog", linthresh=1.0)
            axis.set_title(
                f"{chr(65 + index)} · Gemma 2 2B · L{layer}",
                loc="left",
                fontweight="bold",
            )
            axis.text(
                0.98,
                0.95,
                f"mean {np.mean(values):.2f}\nmedian {np.median(values):.2f}",
                ha="right",
                va="top",
                fontsize=6.5,
                color="#4b5563",
                transform=axis.transAxes,
            )
            axis.grid(axis="y", color="#e4e7eb", linewidth=0.45)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Fraction of directions")
        figure.supxlabel("Random-direction excess kurtosis", y=0.04)
        figure.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.88, wspace=0.16)
        figure.savefig(path, dpi=300)
        plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "runs/main")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    median_paths = [output / filename for filename, _, _ in STATISTICS.values()]
    mean_paths = [output / f"mean-{filename}" for filename, _, _ in STATISTICS.values()]
    median_combined = output / "kurtosis-variance-over-layers.png"
    mean_combined = output / "mean-kurtosis-variance-over-layers.png"
    distribution_path = output / "gemma-random-kurtosis-distribution-l09-l12-l14.png"
    histogram_path = output / "gemma-random-kurtosis-histogram-l09-l12-l14.png"
    paths = [
        *median_paths,
        *mean_paths,
        median_combined,
        mean_combined,
        distribution_path,
        histogram_path,
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"{existing[0]} exists; pass --force to replace figures")
    run, loaded = load_partial_results(root)
    with tempfile.TemporaryDirectory() as mpl_dir:
        os.environ["MPLCONFIGDIR"] = mpl_dir
        for aggregation, individual_paths, combined_path in (
            ("median", median_paths, median_combined),
            ("mean", mean_paths, mean_combined),
        ):
            for statistic, path in zip(STATISTICS, individual_paths, strict=True):
                plot_statistic(run, loaded, statistic, path, aggregation)
                print(path)
            plot_combined(run, loaded, combined_path, aggregation)
            print(combined_path)
        if plot_random_kurtosis_distribution(loaded, distribution_path):
            print(distribution_path)
        if plot_random_kurtosis_histogram(loaded, histogram_path):
            print(histogram_path)


if __name__ == "__main__":
    main()
