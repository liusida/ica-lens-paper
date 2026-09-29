#!/usr/bin/env python3
"""Plot complete and partial official TPP results across models."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    ("ica_custom_sae", "ICA", "#3D5F99", "^"),
    ("sae_custom_sae", "SAE", "#B45F4D", "o"),
    ("unfitted_ica_custom_sae", "Unfitted ICA", "#8FA9CC", "v"),
    ("untrained_sae_matched_l0_custom_sae", "Untrained SAE", "#D29A8E", "s"),
    ("pca_custom_sae", "PCA", "#7A8B63", "D"),
)
DRAW_ORDER = tuple(reversed(METHODS))
DEFAULT_METHOD_IDS = ("ica_custom_sae", "sae_custom_sae")
MODELS = (
    ("gpt2", "GPT-2 Small", (6, 10)),
    ("gemma2", "Gemma 2 2B", (12, 20)),
    ("qwen35-9b", "Qwen 3.5 9B Base", (12, 20)),
)
METRICS = (
    ("total_metric", "TPP score"),
    ("intended_diff_only", "Intended change"),
    ("unintended_diff_only", "Unintended change"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/tpp/official/results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/tpp/official/figures/tpp-model-comparison-partial.png"),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=3,
        help="Expected number of official seeds used in partial-status labels (default: 3).",
    )
    return parser.parse_args()


def discover_seed_roots(results: Path) -> list[tuple[str, Path]]:
    seed_roots = sorted(
        (path.name, path) for path in results.glob("seed-*") if path.is_dir()
    )
    return seed_roots or [("single", results)]


def validate_seed_configs(seed_roots: list[tuple[str, Path]], model_name: str) -> None:
    """Reject seed aggregation when anything except the evaluation seed differs."""
    reference: dict | None = None
    reference_seed = ""
    for seed_name, seed_root in seed_roots:
        path = seed_root / model_name / "config.json"
        if not path.is_file():
            continue
        config = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
        config.get("settings", {}).pop("random_seed", None)
        if reference is None:
            reference = config
            reference_seed = seed_name
        elif config != reference:
            raise ValueError(
                f"incompatible TPP configurations for {model_name}: "
                f"{reference_seed} and {seed_name} differ beyond random_seed"
            )


def load_available(root: Path, layers: tuple[int, ...]) -> dict[str, dict[int, dict]]:
    available: dict[str, dict[int, dict]] = {}
    for method_id, *_ in METHODS:
        by_layer = {}
        for layer in layers:
            path = (
                root
                / "layers"
                / f"layer_{layer:02d}"
                / "saebench"
                / "tpp"
                / f"{method_id}_eval_results.json"
            )
            if path.is_file():
                by_layer[layer] = json.loads(path.read_text(encoding="utf-8"))
        if by_layer:
            available[method_id] = by_layer
    return available


def metric_value(payload: dict, budget: int, suffix: str) -> float:
    key = f"tpp_threshold_{budget}_{suffix}"
    summary = payload["eval_result_metrics"]["tpp_metrics"].get(key)
    raw = payload["eval_result_unstructured"]
    dataset_means = [
        statistics.mean(float(class_result[key]) for class_result in classes.values())
        for classes in raw.values()
    ]
    reconstructed = statistics.mean(dataset_means)
    if summary is not None and not np.isclose(float(summary), reconstructed, atol=1e-7):
        raise ValueError(f"stored and reconstructed values disagree for {key}")
    return reconstructed


def plot_method(
    axis: plt.Axes,
    payloads_by_seed: dict[str, dict[int, dict]],
    *,
    suffix: str,
    label: str,
    color: str,
    marker: str,
    zorder: int,
) -> None:
    budgets, values = aggregate_seed_curves(payloads_by_seed, suffix=suffix)
    mean = values.mean(axis=0)
    if len(values) > 1:
        axis.fill_between(
            budgets,
            values.min(axis=0),
            values.max(axis=0),
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=zorder - 1,
        )
    axis.plot(
        budgets,
        mean,
        color=color,
        marker=marker,
        linewidth=1.35,
        markersize=3.4,
        label=label,
        zorder=zorder,
    )


def aggregate_seed_curves(
    payloads_by_seed: dict[str, dict[int, dict]], *, suffix: str
) -> tuple[list[int], np.ndarray]:
    """Average layers within each seed and return one curve per available seed."""
    first = next(iter(next(iter(payloads_by_seed.values())).values()))
    budgets = [int(value) for value in first["eval_config"]["n_values"]]
    seed_values = []
    for payloads in payloads_by_seed.values():
        for payload in payloads.values():
            payload_budgets = [int(value) for value in payload["eval_config"]["n_values"]]
            if payload_budgets != budgets:
                raise ValueError("TPP seed/layer results use different feature budgets")
        layer_values = np.asarray(
            [
                [metric_value(payload, budget, suffix) for budget in budgets]
                for payload in payloads.values()
            ]
        )
        seed_values.append(layer_values.mean(axis=0))
    return budgets, np.asarray(seed_values)


def main() -> None:
    args = parse_args()
    if args.expected_seeds < 1:
        raise ValueError("--expected-seeds must be positive")
    seed_roots = discover_seed_roots(args.results)
    for model_name, _, _ in MODELS:
        validate_seed_configs(seed_roots, model_name)
    model_data = [
        (
            model_name,
            title,
            layers,
            {
                seed_name: load_available(seed_root / model_name, layers)
                for seed_name, seed_root in seed_roots
            },
        )
        for model_name, title, layers in MODELS
    ]
    if not any(data for _, _, _, seeds in model_data for data in seeds.values()):
        raise ValueError("no TPP method results are available")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    figure = plt.figure(figsize=(11.2, 4.8))
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 0.72), hspace=0.48, wspace=0.42)
    axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for index in range(3):
        axes.append(
            (
                figure.add_subplot(grid[0, 2 * index : 2 * index + 2]),
                figure.add_subplot(grid[1, 2 * index]),
                figure.add_subplot(grid[1, 2 * index + 1]),
            )
        )

    for model_index, (_, title, expected_layers, seed_data) in enumerate(model_data):
        seeds_with_data = {
            seed: data for seed, data in seed_data.items() if any(data.values())
        }
        complete_seeds = sum(
            all(set(data.get(method, {})) == set(expected_layers) for method in DEFAULT_METHOD_IDS)
            for data in seeds_with_data.values()
        )
        available_seeds = len(seeds_with_data)
        subtitle = (
            f"{complete_seeds}/{args.expected_seeds} complete seeds; "
            f"{available_seeds}/{args.expected_seeds} with data"
        )
        top, intended, unintended = axes[model_index]
        top.set_title(f"{title}\n{subtitle}", fontweight="bold", loc="left")
        intended.set_title("Intended", fontweight="bold", loc="left")
        unintended.set_title("Unintended", fontweight="bold", loc="left")
        for metric_index, axis in enumerate((top, intended, unintended)):
            suffix, _ = METRICS[metric_index]
            for draw_index, (method_id, label, color, marker) in enumerate(DRAW_ORDER):
                method_payloads = {
                    seed: data[method_id]
                    for seed, data in seeds_with_data.items()
                    if method_id in data
                }
                if method_payloads:
                    plot_method(
                        axis,
                        method_payloads,
                        suffix=suffix,
                        label=label,
                        color=color,
                        marker=marker,
                        zorder=2 + draw_index,
                    )
            axis.axhline(0, color="0.65", linewidth=0.7, zorder=0)
            axis.set_xscale("log")
            axis.set_xticks((1, 2, 5, 10, 20, 50, 100))
            axis.set_xticklabels(("1", "2", "5", "10", "20", "50", "100"), rotation=38)
            axis.grid(axis="y", color="0.88", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        top.set_ylabel("TPP score" if model_index == 0 else "")
        intended.set_ylabel("TPP effect" if model_index == 0 else "")

    all_axes = [axis for group in axes for axis in group]
    for axis in all_axes:
        axis.set_ylim(0, 0.44)
    handles: list[object] = []
    labels: list[str] = []
    for axis in all_axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    by_label = dict(zip(labels, handles, strict=True))
    ordered_labels = [label for _, label, _, _ in METHODS if label in by_label]
    figure.legend(
        [by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    figure.supxlabel("Top-k perturbed features", y=0.025)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.16, top=0.82)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
