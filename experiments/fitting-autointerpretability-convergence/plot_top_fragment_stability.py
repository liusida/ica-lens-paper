#!/usr/bin/env python3
"""Plot cohort-average stability of top-activating fragment sets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from prepare import EVALUATED_CHECKPOINTS, LAYERS
from trajectory import parse_layers

ROOT = Path(__file__).resolve().parent
DEFAULT_PREPARATION = ROOT / "runs/prepared"
DEFAULT_OUTPUT = ROOT / "figures/top-20-fragment-stability"
ICA_BLUE = "#3D5F99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def top_sets(preparation: Path, layer: int, top_k: int) -> dict[int, list[set[int]]]:
    result: dict[int, list[set[int]]] = {}
    expected_ids: list[int] | None = None
    for iteration in EVALUATED_CHECKPOINTS:
        directory = preparation / f"iter-{iteration:03d}" / f"layer_{layer:02d}" / "ica"
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        rows = selection["accepted"]
        ids = [int(row["feature"]) for row in rows]
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"cohort order changed at iteration {iteration}, layer {layer}")
        values = np.load(directory / "candidate_activations.npy", mmap_mode="r")
        maxima = np.asarray(values).max(axis=1)
        result[iteration] = [
            set(np.argsort(-maxima[:, position], kind="stable")[:top_k].tolist())
            for position in range(maxima.shape[1])
        ]
    return result


def measure(preparation: Path, layers: tuple[int, ...], top_k: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for layer in layers:
        sets = top_sets(preparation, layer, top_k)
        late_100 = sets[100]
        final = sets[EVALUATED_CHECKPOINTS[-1]]
        for checkpoint_position, iteration in enumerate(EVALUATED_CHECKPOINTS):
            if checkpoint_position:
                previous = sets[EVALUATED_CHECKPOINTS[checkpoint_position - 1]]
                adjacent = np.asarray(
                    [
                        len(left & right) / top_k
                        for left, right in zip(previous, sets[iteration], strict=True)
                    ]
                )
            else:
                adjacent = np.full(len(final), np.nan)
            to_final = np.asarray(
                [
                    len(left & right) / top_k
                    for left, right in zip(sets[iteration], final, strict=True)
                ]
            )
            to_100 = np.asarray(
                [
                    len(left & right) / top_k
                    for left, right in zip(sets[iteration], late_100, strict=True)
                ]
            )
            for component_position, (local, overlap_100, overlap_200) in enumerate(
                zip(adjacent, to_100, to_final, strict=True)
            ):
                rows.append(
                    {
                        "layer": layer,
                        "iteration": iteration,
                        "component_position": component_position,
                        "adjacent_overlap": float(local),
                        "iteration_100_overlap": float(overlap_100),
                        "iteration_200_overlap": float(overlap_200),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, float | int]], layers: tuple[int, ...]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, len(layers), figsize=(3.45 * len(layers), 2.55), sharey=True)
    axes = np.atleast_1d(axes)
    for panel, (axis, layer) in enumerate(zip(axes, layers, strict=True)):
        layer_rows = [row for row in rows if row["layer"] == layer]
        for field, label, color, marker in (
            ("adjacent_overlap", "Previous checkpoint", ICA_BLUE, "o"),
            ("iteration_100_overlap", "Iteration 100", "#7397C2", "s"),
            ("iteration_200_overlap", "Iteration 200", "#A9BED8", "D"),
        ):
            means, lower, upper = [], [], []
            for iteration in EVALUATED_CHECKPOINTS:
                values = np.asarray(
                    [row[field] for row in layer_rows if row["iteration"] == iteration], dtype=float
                )
                values = values[np.isfinite(values)]
                if values.size:
                    rng = np.random.default_rng(layer * 100_000 + iteration * 10 + len(label))
                    bootstrap = rng.choice(values, size=(10_000, values.size), replace=True).mean(
                        axis=1
                    )
                    means.append(float(values.mean()))
                    lower.append(float(np.quantile(bootstrap, 0.025)))
                    upper.append(float(np.quantile(bootstrap, 0.975)))
                else:
                    means.append(np.nan)
                    lower.append(np.nan)
                    upper.append(np.nan)
            means_array = np.asarray(means)
            axis.errorbar(
                EVALUATED_CHECKPOINTS,
                means_array,
                yerr=np.vstack((means_array - lower, np.asarray(upper) - means_array)),
                color=color,
                marker=marker,
                markersize=3.0,
                linewidth=1.2,
                elinewidth=0.65,
                capsize=1.8,
                label=label,
            )
        axis.set_title(f"{chr(65 + panel)} · Qwen 3.5 9B Base · L{layer}", loc="left")
        axis.set_xscale("symlog", linthresh=2, linscale=1)
        axis.set_xticks(EVALUATED_CHECKPOINTS)
        axis.set_xticklabels(EVALUATED_CHECKPOINTS, rotation=45)
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", color="0.89", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("FastICA iteration")
    axes[0].set_ylabel("Top-20 fragment overlap")
    axes[-1].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = measure(args.preparation.expanduser().resolve(), layers, args.top_k)
    write_csv(output.with_suffix(".csv"), rows)
    plot(output, rows, layers)
    print(f"Wrote {output.with_suffix('.csv')}")
    print(f"Wrote {output.with_suffix('.png')}")
    print(f"Wrote {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
