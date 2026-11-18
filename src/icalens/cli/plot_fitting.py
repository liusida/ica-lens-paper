"""Plot model-level summaries of saved FastICA fitting curves."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..lens import ICALens

DEFAULT_OUTPUT = Path("figures")
DEFAULT_STEM = "fitting-curves"
KURTOSIS_STEM = "fitting-kurtosis"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens plot fitting-summary", description=__doc__)
    parser.add_argument(
        "lens",
        nargs="+",
        help="Local ICA Lens artifact directories or Hugging Face repository IDs.",
    )
    parser.add_argument(
        "--titles",
        "--panel-titles",
        dest="titles",
        default=None,
        help="Comma-separated panel titles (default: artifact model names).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: ./figures).",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing figure files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    lenses = [ICALens.from_pretrained(source) for source in args.lens]
    titles = _parse_titles(args.titles, lenses)
    outputs = render_fitting_summary(
        lenses,
        titles=titles,
        output=args.output.expanduser().resolve(),
        force=bool(args.force),
    )
    for path in outputs:
        print(path)


def render_fitting_summary(
    lenses: Sequence[ICALens],
    *,
    titles: Sequence[str],
    output: Path,
    force: bool,
) -> list[Path]:
    if not lenses:
        raise ValueError("at least one ICA Lens is required")
    if len(titles) != len(lenses):
        raise ValueError("the number of titles must match the number of lenses")

    paths = [output / f"{DEFAULT_STEM}.{suffix}" for suffix in ("png", "pdf")]
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")

    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    columns = min(3, len(lenses))
    rows = math.ceil(len(lenses) / columns)
    colors = plt.colormaps["Blues"](np.linspace(0.18, 0.72, 5))
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6.9, 2.45 * rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for axis, lens, title in zip(axes.flat, lenses, titles, strict=False):
            layer_count = _plot_model_summary(axis, lens, colors=colors)
            axis.set_title(title, loc="left", fontweight="bold", pad=5)
            axis.text(
                0.98,
                1.02,
                f"{layer_count} layers",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize="small",
                clip_on=False,
            )
            axis.grid(axis="y", color="0.89", linewidth=0.5)
            axis.set_axisbelow(True)
        for axis in axes.flat[len(lenses) :]:
            axis.remove()

        figure.supylabel("Logcosh contrast", x=0.015)
        figure.supxlabel("FastICA iteration", y=0.04)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            frameon=False,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            handlelength=1.8,
            columnspacing=1.15,
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.995,
            bottom=0.24,
            top=0.72,
            wspace=0.16,
            hspace=0.34,
        )
        for path in paths:
            figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)
        if any(_kurtosis_layers(lens) for lens in lenses):
            paths += _render_kurtosis(lenses, titles=titles, output=output, force=force)
    return paths


def _kurtosis_layers(lens: ICALens) -> list[tuple[int, dict[str, Any]]]:
    """Saved layers that carry a kurtosis history; others (older or partial) are skipped."""
    layers = lens.metadata.get("layers", {})
    found = []
    for layer in lens.available_layers:
        history = layers.get(str(layer), {}).get("fitting", {}).get("kurtosis_history")
        if isinstance(history, dict) and isinstance(history.get("train"), dict):
            found.append((layer, history))
    return found


def _curve(history: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray] | None:
    entry = history.get(split)
    if not isinstance(entry, dict) or not entry.get("iterations"):
        return None
    iterations = np.asarray(entry["iterations"], dtype=np.int64)
    means = np.asarray(entry["mean"], dtype=np.float64)
    if iterations.shape != means.shape:
        return None
    return iterations, means


def _layer_mean_curve(
    histories: list[tuple[int, dict[str, Any]]], split: str
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Average over the layers sharing the most common iteration grid."""
    curves = [curve for _, history in histories if (curve := _curve(history, split))]
    if not curves:
        return None
    grids: dict[tuple[int, ...], list[np.ndarray]] = {}
    for iterations, means in curves:
        grids.setdefault(tuple(iterations.tolist()), []).append(means)
    grid, members = max(grids.items(), key=lambda item: len(item[1]))
    return np.asarray(grid), np.mean(members, axis=0), len(members)


def _check_outputs(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")


def _style_axis(axis: Any) -> None:
    axis.grid(axis="y", color="0.89", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=6.5)


def _plot_train_heldout(axis: Any, train: Any, heldout: Any, *, linewidth: float = 1.3) -> None:
    if train is not None:
        axis.plot(train[0], train[1], color="#1d4ed8", linewidth=linewidth, label="Fitting (train)")
    if heldout is not None:
        axis.plot(
            heldout[0],
            heldout[1],
            color="#c2410c",
            linewidth=linewidth * 0.9,
            linestyle="--",
            marker="o",
            markersize=2.2,
            label="Held-out",
        )


def _render_kurtosis(
    lenses: Sequence[ICALens], *, titles: Sequence[str], output: Path, force: bool
) -> list[Path]:
    """Train vs held-out mean excess kurtosis: a layer average and one panel per layer."""
    import matplotlib.pyplot as plt

    selected = [
        (lens, title, histories)
        for lens, title in zip(lenses, titles, strict=True)
        if (histories := _kurtosis_layers(lens))
    ]
    average_paths = [output / f"{KURTOSIS_STEM}.{suffix}" for suffix in ("png", "pdf")]
    _check_outputs(average_paths, force)
    columns = min(3, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.9, 2.3 * rows), squeeze=False)
    for axis, (_lens, title, histories) in zip(axes.flat, selected, strict=False):
        train = _layer_mean_curve(histories, "train")
        heldout = _layer_mean_curve(histories, "heldout")
        _plot_train_heldout(axis, train and train[:2], heldout and heldout[:2], linewidth=1.5)
        count = max(item[2] for item in (train, heldout) if item is not None)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=8)
        axis.text(
            0.98,
            1.02,
            f"mean of {count} layer(s)",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
        )
        _style_axis(axis)
    for axis in axes.flat[len(selected) :]:
        axis.remove()
    figure.supylabel("Mean excess kurtosis", fontsize=8)
    figure.supxlabel("FastICA iteration (updates applied)", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="upper center", fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    for path in average_paths:
        figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    paths = list(average_paths)

    for index, (lens, title, histories) in enumerate(selected):
        suffix = "" if len(selected) == 1 else f"-{index + 1}"
        layer_paths = [
            output / f"{KURTOSIS_STEM}-layers{suffix}.{extension}" for extension in ("png", "pdf")
        ]
        _check_outputs(layer_paths, force)
        columns = min(4, len(histories))
        rows = math.ceil(len(histories) / columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(6.9, 1.55 * rows + 0.5), squeeze=False, sharex=True
        )
        for axis, (layer, history) in zip(axes.flat, histories, strict=False):
            _plot_train_heldout(axis, _curve(history, "train"), _curve(history, "heldout"))
            axis.set_title(f"Layer {layer}", loc="left", fontsize=7.5)
            _style_axis(axis)
        for axis in axes.flat[len(histories) :]:
            axis.remove()
        figure.suptitle(
            f"{title}: per-layer excess kurtosis ({len(histories)} of "
            f"{len(lens.available_layers)} saved layers)",
            fontsize=8,
            fontweight="bold",
            x=0.01,
            ha="left",
        )
        figure.supylabel("Mean excess kurtosis", fontsize=8)
        figure.supxlabel("FastICA iteration (updates applied)", fontsize=8)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, ncol=2, loc="upper right", fontsize=7)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        for path in layer_paths:
            figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)
        paths += layer_paths
    return paths


def _plot_model_summary(axis: Any, lens: ICALens, *, colors: Any) -> int:
    metadata = lens.metadata
    fitting = [metadata["layers"][str(layer)]["fitting"] for layer in lens.available_layers]
    if not fitting:
        raise ValueError(f"ICA Lens {lens.model_id!r} has no fitted layers")
    # Group well-formed layer histories by their (iterations, percentiles) grid and
    # average the largest group, so partial or mixed lenses still plot.
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[np.ndarray[Any, Any]]] = {}
    for item in fitting:
        history = item.get("objective_history")
        if not isinstance(history, dict):
            continue
        layer_iterations = tuple(int(value) for value in history.get("iterations") or ())
        layer_percentiles = tuple(int(value) for value in history.get("percentiles") or ())
        layer_values = np.asarray(history.get("values"), dtype=np.float64)
        if not layer_iterations or layer_values.shape != (
            len(layer_iterations),
            len(layer_percentiles),
        ):
            continue
        groups.setdefault((layer_iterations, layer_percentiles), []).append(layer_values)
    if not groups:
        raise ValueError(f"ICA Lens {lens.model_id!r} has no layers with objective histories")
    (grid_iterations, grid_percentiles), values = max(groups.items(), key=lambda item: len(item[1]))
    iterations = np.asarray(grid_iterations, dtype=np.int64)
    percentiles = np.asarray(grid_percentiles, dtype=np.int64)
    mean_curves = np.mean(values, axis=0)

    for color, lower in zip(colors, (0, 10, 20, 30, 40), strict=True):
        upper = 100 - lower
        low_index = _percentile_index(percentiles, lower, lens.model_id)
        high_index = _percentile_index(percentiles, upper, lens.model_id)
        axis.fill_between(
            iterations,
            mean_curves[:, low_index],
            mean_curves[:, high_index],
            color=color,
            alpha=0.55,
            linewidth=0,
            label=f"p{lower}–p{upper}",
        )
    median = mean_curves[:, _percentile_index(percentiles, 50, lens.model_id)]
    axis.plot(iterations, median, color="#172554", linewidth=1.5, label="Median")
    gaussian = [item.get("gaussian_objective") for item in fitting]
    if all(isinstance(value, (int, float)) and np.isfinite(value) for value in gaussian):
        axis.axhline(
            float(np.mean(gaussian)),
            color="#94a3b8",
            linestyle="--",
            linewidth=1.1,
            label="Gaussian baseline",
        )
    return len(values)


def _percentile_index(percentiles: np.ndarray[Any, Any], value: int, model_id: str) -> int:
    matches = np.flatnonzero(percentiles == value)
    if len(matches) != 1:
        raise ValueError(f"ICA Lens {model_id!r} objective history does not contain p{value}")
    return int(matches[0])


def _parse_titles(value: str | None, lenses: Sequence[ICALens]) -> list[str]:
    if value is None:
        return [lens.model_id.rsplit("/", 1)[-1] for lens in lenses]
    titles = [item.strip() for item in value.split(",")]
    if len(titles) != len(lenses) or any(not title for title in titles):
        raise ValueError("--titles must contain one non-empty title per lens")
    return titles
