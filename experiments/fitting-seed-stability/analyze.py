"""Match and compare ICA reading directions across fitting seeds."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

LAYER = 6
EXPECTED_COMPONENTS = 768
DEFAULT_INPUT = Path("pilot-experiments/fitting-seed-stability/output")
DEFAULT_RESULTS = Path("pilot-experiments/fitting-seed-stability/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    input_root = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    directions = {seed: _load_reading_directions(input_root, seed) for seed in seeds}
    rows = _matched_comparisons(directions)
    summary = _summarize(rows, seeds=seeds)

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "matched-component-similarities.csv"
    summary_path = output / "matched-summary.json"
    figure_path = output / "matched-similarity-by-reference-rank.png"
    existing = [path for path in (csv_path, summary_path, figure_path) if path.exists()]
    if existing and not args.force:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing results: {paths}; pass --force")
    _write_csv(csv_path, rows)
    _write_json(summary_path, summary)
    _plot(figure_path, rows)
    print(json.dumps(summary, indent=2))


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--seeds must be comma-separated integers") from error
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain at least two unique integers")
    return seeds


def _load_reading_directions(root: Path, seed: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    lens_root = root / f"seed-{seed}-iter-50"
    manifest = _read_json(lens_root / "icalens.json")
    layer = manifest.get("layers", {}).get(str(LAYER))
    if not isinstance(layer, dict):
        raise ValueError(f"seed {seed} has no fitted layer {LAYER}")
    fitting = layer.get("fitting", {})
    expected = {
        "n_components": (layer.get("n_components"), EXPECTED_COMPONENTS),
        "random_state": (fitting.get("random_state"), seed),
        "max_iter": (fitting.get("max_iter"), 50),
        "n_iter": (fitting.get("n_iter"), 50),
        "n_samples": (fitting.get("n_samples"), 1_000_000),
    }
    mismatches = [
        f"{key}: {actual!r} != {wanted!r}"
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(f"incompatible seed-{seed} artifact: " + "; ".join(mismatches))

    tensor_path = lens_root / str(layer["file"])
    reading = np.asarray(load_file(tensor_path)["reading_matrix"], dtype=np.float64)
    expected_shape = (EXPECTED_COMPONENTS, EXPECTED_COMPONENTS)
    if reading.shape != expected_shape:
        raise ValueError(
            f"seed {seed} reading matrix has shape {reading.shape}, expected {expected_shape}"
        )
    norms = np.linalg.norm(reading, axis=1, keepdims=True)
    if not np.isfinite(reading).all() or np.any(norms == 0):
        raise ValueError(f"seed {seed} reading matrix contains invalid directions")
    return reading / norms


def _matched_comparisons(
    directions: dict[int, np.ndarray[Any, np.dtype[np.float64]]],
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for seed_a, seed_b in itertools.combinations(directions, 2):
        similarities = np.abs(directions[seed_a] @ directions[seed_b].T)
        assignment = _maximum_weight_assignment(similarities)
        if len(set(assignment.tolist())) != len(assignment):
            raise RuntimeError("component assignment is not one-to-one")
        for component_a, component_b in enumerate(assignment):
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "component_a": component_a,
                    "component_b": int(component_b),
                    "absolute_cosine_similarity": float(
                        similarities[component_a, component_b]
                    ),
                }
            )
    return rows


def _maximum_weight_assignment(weights: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray:
    """Return the maximizing column for each row using the Hungarian algorithm."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("assignment weights must be a square matrix")
    if not np.isfinite(weights).all():
        raise ValueError("assignment weights must be finite")

    size = weights.shape[0]
    costs = np.max(weights) - weights
    row_potential = np.zeros(size + 1, dtype=np.float64)
    column_potential = np.zeros(size + 1, dtype=np.float64)
    matched_row = np.zeros(size + 1, dtype=np.int64)
    predecessor = np.zeros(size + 1, dtype=np.int64)

    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1, candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


def _statistics(values: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "minimum": float(np.min(values)),
        "fraction_ge_0.90": float(np.mean(values >= 0.90)),
        "fraction_ge_0.95": float(np.mean(values >= 0.95)),
        "fraction_ge_0.99": float(np.mean(values >= 0.99)),
    }


def _summarize(
    rows: list[dict[str, int | float]], *, seeds: tuple[int, ...]
) -> dict[str, Any]:
    pairwise: list[dict[str, Any]] = []
    for seed_a, seed_b in itertools.combinations(seeds, 2):
        values = np.asarray(
            [
                row["absolute_cosine_similarity"]
                for row in rows
                if row["seed_a"] == seed_a and row["seed_b"] == seed_b
            ],
            dtype=np.float64,
        )
        pairwise.append({"seed_a": seed_a, "seed_b": seed_b, **_statistics(values)})
    all_values = np.asarray(
        [row["absolute_cosine_similarity"] for row in rows], dtype=np.float64
    )
    return {
        "comparison": "optimal one-to-one component matching",
        "matching_objective": "maximize total absolute cosine similarity",
        "sign_handling": "absolute cosine similarity",
        "direction": "reading_matrix rows",
        "layer": LAYER,
        "seeds": list(seeds),
        "component_count": EXPECTED_COMPONENTS,
        "seed_pair_count": len(pairwise),
        "aggregate": _statistics(all_values),
        "pairwise": pairwise,
    }


def _plot(path: Path, rows: list[dict[str, int | float]]) -> None:
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache_dir:
        previous_cache = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache_dir
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

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
            with mpl.rc_context(style):
                figure, axis = plt.subplots(figsize=(6.9, 3.25))
                pairs = sorted(
                    {(int(row["seed_a"]), int(row["seed_b"])) for row in rows}
                )
                ordered_pairs = [pair for pair in pairs if pair != (0, 1)] + [(0, 1)]
                for seed_a, seed_b in ordered_pairs:
                    selected = [
                        row
                        for row in rows
                        if row["seed_a"] == seed_a and row["seed_b"] == seed_b
                    ]
                    highlighted = (seed_a, seed_b) == (0, 1)
                    axis.plot(
                        [int(row["component_a"]) for row in selected],
                        [float(row["absolute_cosine_similarity"]) for row in selected],
                        color="#3D5F99" if highlighted else "#777777",
                        linewidth=0.65 if highlighted else 0.3,
                        alpha=0.9 if highlighted else 0.16,
                    )
                axis.set(
                    xlabel="Component rank in first seed",
                    ylabel="Matched absolute cosine similarity",
                    ylim=(0, 1.01),
                )
                axis.set_axisbelow(True)
                axis.grid(axis="y", color="0.89", linewidth=0.5)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.legend(
                    handles=[
                        Line2D([0], [0], color="#3D5F99", linewidth=0.65, label="Seed 0–1"),
                        Line2D(
                            [0],
                            [0],
                            color="#777777",
                            linewidth=0.3,
                            alpha=0.5,
                            label="Other seed pairs",
                        ),
                    ],
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=2,
                )
                figure.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.88)
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


if __name__ == "__main__":
    main()
