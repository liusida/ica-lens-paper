"""Compare matched ICA components across fitting iteration counts and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from analyze import _maximum_weight_assignment, _statistics
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent
SEEDS = (0, 1, 2, 3, 4)
ITERATIONS = (20, 50, 200, 500)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LAYER = 6
COMPONENTS = 768
ITERATION_PAIRS = ((20, 50), (50, 200), (200, 500))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directions = {
        seed: {
            iterations: _load(
                ROOT / f"output/seed-{seed}-iter-{iterations}", seed, iterations
            )
            for iterations in ITERATIONS
        }
        for seed in SEEDS
    }
    rows = _compare(directions)
    pairwise_rows, cross_seed_component_rows = _compare_seeds(directions)
    summary = _summarize(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "iteration-matched-component-similarities.csv"
    summary_path = RESULTS / "iteration-matched-summary.json"
    figure_path = FIGURES / "iteration-matched-similarity-by-reference-rank.png"
    aggregate_figure_path = FIGURES / "iteration-matched-mean-median.png"
    pairwise_csv_path = RESULTS / "iteration-seed-pairwise-mean.csv"
    pairwise_figure_path = FIGURES / "iteration-seed-pairwise-mean-heatmap.png"
    pairwise_median_figure_path = (
        FIGURES / "iteration-seed-pairwise-median-heatmap.png"
    )
    pairwise_median_pdf_path = (
        FIGURES / "iteration-seed-pairwise-median-heatmap.pdf"
    )
    pairwise_threshold_figure_path = (
        FIGURES / "iteration-seed-pairwise-fraction-gt-070-heatmap.png"
    )
    pairwise_component_csv_path = (
        RESULTS / "iteration-seed-pairwise-component-similarities.csv"
    )
    density_figure_path = FIGURES / "iteration-seed-pairwise-density.png"
    existing = [
        path
        for path in (
            csv_path,
            summary_path,
            figure_path,
            aggregate_figure_path,
            pairwise_csv_path,
            pairwise_figure_path,
            pairwise_median_figure_path,
            pairwise_median_pdf_path,
            pairwise_threshold_figure_path,
            pairwise_component_csv_path,
            density_figure_path,
        )
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError("results already exist; pass --force to replace them")
    FIGURES.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    _write_csv(pairwise_csv_path, pairwise_rows)
    _write_csv(pairwise_component_csv_path, cross_seed_component_rows)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot(figure_path, rows)
    _plot_summary(aggregate_figure_path, summary)
    _plot_seed_heatmaps(
        pairwise_figure_path,
        pairwise_rows,
        statistic="mean",
    )
    _plot_seed_heatmaps(
        pairwise_median_figure_path,
        pairwise_rows,
        statistic="median",
        pdf_path=pairwise_median_pdf_path,
    )
    _plot_seed_heatmaps(
        pairwise_threshold_figure_path,
        pairwise_rows,
        statistic="fraction_gt_0.70",
    )
    _plot_density(density_figure_path, cross_seed_component_rows)
    print(json.dumps(summary, indent=2))


def _load(
    path: Path, seed: int, iterations: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    root = path.expanduser().resolve()
    manifest = json.loads((root / "icalens.json").read_text(encoding="utf-8"))
    layer = manifest["layers"][str(LAYER)]
    fitting = layer["fitting"]
    expected = {
        "random_state": (fitting.get("random_state"), seed),
        "max_iter": (fitting.get("max_iter"), iterations),
        "n_iter": (fitting.get("n_iter"), iterations),
        "n_samples": (fitting.get("n_samples"), 1_000_000),
        "n_components": (layer.get("n_components"), COMPONENTS),
    }
    mismatches = [
        f"{key}: {actual!r} != {wanted!r}"
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            f"incompatible seed-{seed}, {iterations}-iteration artifact: "
            + "; ".join(mismatches)
        )
    reading = np.asarray(load_file(root / layer["file"])["reading_matrix"], dtype=np.float64)
    norms = np.linalg.norm(reading, axis=1, keepdims=True)
    if reading.shape != (COMPONENTS, COMPONENTS) or not np.isfinite(reading).all():
        raise ValueError(f"invalid reading matrix in {root}")
    if np.any(norms == 0):
        raise ValueError(f"zero reading direction in {root}")
    return reading / norms


def _compare(
    directions: dict[
        int, dict[int, np.ndarray[Any, np.dtype[np.float64]]]
    ],
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for seed in SEEDS:
        for iterations_a, iterations_b in ITERATION_PAIRS:
            similarities = np.abs(
                directions[seed][iterations_a] @ directions[seed][iterations_b].T
            )
            assignment = _maximum_weight_assignment(similarities)
            for component_a, component_b in enumerate(assignment):
                rows.append(
                    {
                        "seed": seed,
                        "iterations_a": iterations_a,
                        "iterations_b": iterations_b,
                        "component_a": component_a,
                        "component_b": int(component_b),
                        "absolute_cosine_similarity": float(
                            similarities[component_a, component_b]
                        ),
                    }
                )
    return rows


def _compare_seeds(
    directions: dict[
        int, dict[int, np.ndarray[Any, np.dtype[np.float64]]]
    ],
) -> tuple[
    list[dict[str, int | float]], list[dict[str, int | float]]
]:
    rows: list[dict[str, int | float]] = []
    component_rows: list[dict[str, int | float]] = []
    for iterations in ITERATIONS:
        for seed_a in SEEDS:
            for seed_b in SEEDS:
                if seed_b < seed_a:
                    continue
                if seed_a == seed_b:
                    mean_similarity = 1.0
                    median_similarity = 1.0
                    fraction_gt_070 = 1.0
                else:
                    similarities = np.abs(
                        directions[seed_a][iterations]
                        @ directions[seed_b][iterations].T
                    )
                    assignment = _maximum_weight_assignment(similarities)
                    matched = similarities[np.arange(COMPONENTS), assignment]
                    mean_similarity = float(np.mean(matched))
                    median_similarity = float(np.median(matched))
                    fraction_gt_070 = float(np.mean(matched > 0.7))
                    component_rows.extend(
                        {
                            "iterations": iterations,
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "component_a": component_a,
                            "component_b": int(component_b),
                            "absolute_cosine_similarity": float(
                                similarities[component_a, component_b]
                            ),
                        }
                        for component_a, component_b in enumerate(assignment)
                    )
                rows.append(
                    {
                        "iterations": iterations,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "mean_matched_absolute_cosine_similarity": mean_similarity,
                        "median_matched_absolute_cosine_similarity": (
                            median_similarity
                        ),
                        "fraction_gt_0.70": fraction_gt_070,
                    }
                )
    return rows, component_rows


def _summarize(rows: list[dict[str, int | float]]) -> dict[str, Any]:
    comparisons = []
    for iterations_a, iterations_b in ITERATION_PAIRS:
        per_seed = []
        for seed in SEEDS:
            values = np.asarray(
                [
                    row["absolute_cosine_similarity"]
                    for row in rows
                    if row["seed"] == seed
                    and row["iterations_a"] == iterations_a
                    and row["iterations_b"] == iterations_b
                ],
                dtype=np.float64,
            )
            per_seed.append({"seed": seed, **_statistics(values)})
        statistic_names = tuple(key for key in per_seed[0] if key != "seed")
        comparisons.append(
            {
                "iterations_a": iterations_a,
                "iterations_b": iterations_b,
                **{
                    name: float(np.mean([item[name] for item in per_seed]))
                    for name in statistic_names
                },
                "per_seed": per_seed,
            }
        )
    return {
        "comparison": "optimal one-to-one component matching",
        "matching_objective": "maximize total absolute cosine similarity",
        "seeds": list(SEEDS),
        "aggregation": "arithmetic mean of each seed-level statistic",
        "layer": LAYER,
        "iteration_pairs": [list(pair) for pair in ITERATION_PAIRS],
        "component_count": COMPONENTS,
        "comparisons": comparisons,
    }


def _plot(path: Path, rows: list[dict[str, int | float]]) -> None:
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache_dir:
        previous_cache = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache_dir
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            style = {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "axes.linewidth": 0.7,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
            encodings = {
                (20, 50): ("#4C72B0", 0.5, 0.85),
                (50, 200): ("#DD8452", 0.5, 0.85),
                (200, 500): ("#55A868", 0.5, 0.85),
            }
            with mpl.rc_context(style):
                figure, axis = plt.subplots(figsize=(6.9, 3.25))
                for pair in ITERATION_PAIRS:
                    mean_by_rank = [
                        float(
                            np.mean(
                                [
                                    row["absolute_cosine_similarity"]
                                    for row in rows
                                    if row["component_a"] == component
                                    and (
                                        row["iterations_a"],
                                        row["iterations_b"],
                                    )
                                    == pair
                                ]
                            )
                        )
                        for component in range(COMPONENTS)
                    ]
                    color, linewidth, alpha = encodings[pair]
                    axis.plot(
                        range(COMPONENTS),
                        mean_by_rank,
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        label=f"{pair[0]}–{pair[1]} iterations",
                    )
                axis.set(
                    xlabel="Component rank in earlier fit",
                    ylabel="Mean matched absolute cosine similarity",
                    ylim=(0, 1.01),
                )
                axis.set_axisbelow(True)
                axis.grid(axis="y", color="0.89", linewidth=0.5)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.legend(
                    title="Average across 5 seeds",
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=3,
                )
                figure.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.84)
                figure.savefig(path, dpi=300)
                plt.close(figure)
        finally:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache


def _write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(path: Path, summary: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache_dir:
        previous_cache = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache_dir
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            style = {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "axes.linewidth": 0.7,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
            comparisons = summary["comparisons"]
            labels = [
                f"{item['iterations_a']}–{item['iterations_b']}"
                for item in comparisons
            ]
            x = np.arange(len(comparisons))
            with mpl.rc_context(style):
                figure, axis = plt.subplots(figsize=(3.35, 2.6))
                axis.plot(
                    x,
                    [item["mean"] for item in comparisons],
                    color="#4C72B0",
                    marker="o",
                    linewidth=1.5,
                    markersize=3.5,
                    label="Mean",
                )
                axis.plot(
                    x,
                    [item["median"] for item in comparisons],
                    color="#DD8452",
                    marker="D",
                    linewidth=1.5,
                    markersize=3.5,
                    label="Median",
                )
                axis.set(
                    xlabel="Successive iteration pair",
                    ylabel="Matched absolute cosine similarity",
                    xticks=x,
                    xticklabels=labels,
                    xlim=(-0.18, len(comparisons) - 0.82),
                    ylim=(0, 1.01),
                )
                axis.set_axisbelow(True)
                axis.grid(axis="y", color="0.89", linewidth=0.5)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.legend(
                    title="Average across 5 seeds",
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=2,
                )
                figure.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.80)
                figure.savefig(path, dpi=300)
                plt.close(figure)
        finally:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache


def _plot_seed_heatmaps(
    path: Path,
    rows: list[dict[str, int | float]],
    *,
    statistic: str,
    pdf_path: Path | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache_dir:
        previous_cache = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache_dir
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            style = {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.linewidth": 0.7,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
            is_percentage = statistic == "fraction_gt_0.70"
            color_min = 0.0
            color_max = 100.0 if is_percentage else 1.0
            value_scale = 100.0 if is_percentage else 1.0
            value_key = (
                statistic
                if is_percentage
                else f"{statistic}_matched_absolute_cosine_similarity"
            )
            with mpl.rc_context(style):
                figure, axes = plt.subplots(1, len(ITERATIONS), figsize=(6.9, 1.9))
                image = None
                for axis, iterations in zip(axes, ITERATIONS, strict=True):
                    matrix = np.eye(len(SEEDS), dtype=np.float64)
                    for row in rows:
                        if row["iterations"] != iterations:
                            continue
                        seed_a = int(row["seed_a"])
                        seed_b = int(row["seed_b"])
                        value = float(row[value_key]) * value_scale
                        matrix[seed_a, seed_b] = value
                        matrix[seed_b, seed_a] = value
                    image = axis.imshow(
                        matrix,
                        cmap="Blues",
                        vmin=color_min,
                        vmax=color_max,
                        interpolation="nearest",
                    )
                    midpoint = (color_min + color_max) / 2
                    for seed_a in SEEDS:
                        for seed_b in SEEDS:
                            axis.text(
                                seed_b,
                                seed_a,
                                (
                                    f"{matrix[seed_a, seed_b]:.0f}%"
                                    if is_percentage
                                    else f"{matrix[seed_a, seed_b]:.2f}"
                                ),
                                ha="center",
                                va="center",
                                fontsize=5.5,
                                color=(
                                    "white"
                                    if matrix[seed_a, seed_b] > midpoint
                                    else "black"
                                ),
                            )
                    axis.set(
                        title=f"{iterations} iterations",
                        xlabel="Seed",
                        xticks=range(len(SEEDS)),
                        xticklabels=SEEDS,
                        yticks=range(len(SEEDS)),
                        yticklabels=SEEDS,
                    )
                    axis.tick_params(length=0)
                axes[0].set_ylabel("Seed")
                for axis in axes[1:]:
                    axis.tick_params(labelleft=False)
                assert image is not None
                figure.subplots_adjust(
                    left=0.06, right=0.85, bottom=0.20, top=0.88, wspace=0.18
                )
                colorbar_axis = figure.add_axes((0.88, 0.20, 0.012, 0.68))
                colorbar = figure.colorbar(image, cax=colorbar_axis)
                colorbar.set_label(
                    "Matched |cosine| > 0.7 (%)"
                    if is_percentage
                    else f"{statistic.capitalize()} matched |cosine|"
                )
                figure.savefig(path, dpi=300)
                if pdf_path is not None:
                    figure.savefig(pdf_path)
                plt.close(figure)
        finally:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache


def _plot_density(
    path: Path, rows: list[dict[str, int | float]]
) -> None:
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache_dir:
        previous_cache = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache_dir
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            style = {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "axes.linewidth": 0.7,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
            colors = {
                20: "#4C72B0",
                50: "#DD8452",
                200: "#55A868",
                500: "#8172B3",
            }
            bin_edges = np.linspace(0.0, 1.0, 51)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            with mpl.rc_context(style):
                figure, axis = plt.subplots(figsize=(3.35, 2.6))
                for iterations in ITERATIONS:
                    pair_densities = []
                    for seed_a in SEEDS:
                        for seed_b in SEEDS:
                            if seed_b <= seed_a:
                                continue
                            values = [
                                float(row["absolute_cosine_similarity"])
                                for row in rows
                                if row["iterations"] == iterations
                                and row["seed_a"] == seed_a
                                and row["seed_b"] == seed_b
                            ]
                            density, _ = np.histogram(
                                values, bins=bin_edges, density=True
                            )
                            pair_densities.append(density)
                    mean_density = np.mean(pair_densities, axis=0)
                    axis.plot(
                        bin_centers,
                        mean_density,
                        color=colors[iterations],
                        linewidth=1.5,
                        label=str(iterations),
                    )
                axis.set(
                    xlabel="Matched absolute cosine similarity",
                    ylabel="Mean density across seed pairs",
                    xlim=(0, 1),
                    ylim=(0, None),
                )
                axis.set_axisbelow(True)
                axis.grid(axis="y", color="0.89", linewidth=0.5)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.legend(
                    title="Iterations (10 pairs from 5 seeds)",
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=4,
                )
                figure.subplots_adjust(left=0.18, right=0.94, bottom=0.19, top=0.78)
                figure.savefig(path, dpi=300)
                plt.close(figure)
        finally:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache


if __name__ == "__main__":
    main()
