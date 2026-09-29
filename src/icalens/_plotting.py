"""Lazy plotting helpers for fitted ICA Lens artifacts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def plot_fitting_curves(
    *,
    layers: list[tuple[int, dict[str, Any]]],
    model_id: str,
    columns: int | None,
) -> Any:
    """Return a figure of individual FastICA objective histories."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - protected by the package dependency
        raise ImportError(
            "plot_fitting_curve() requires matplotlib; install it with 'pip install matplotlib'"
        ) from error

    column_count = min(columns if columns is not None else 2, len(layers))
    rows = math.ceil(len(layers) / column_count)
    figure, axes = plt.subplots(
        rows,
        column_count,
        figsize=(7.2 * column_count, 4.4 * rows + 0.7),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for axis, (layer, fitting) in zip(axes.flat, layers, strict=False):
        _plot_fitting_axis(
            axis,
            layer=layer,
            fitting=fitting,
            plt=plt,
            show_axis_labels=len(layers) == 1,
        )
    for axis in axes.flat[len(layers) :]:
        axis.remove()

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if len(layers) == 1:
        axes.flat[0].set_title(f"FastICA fitting · {model_id} · layer {layers[0][0]}")
        axes.flat[0].legend(ncol=3, fontsize=9, frameon=False)
        figure.tight_layout()
    else:
        figure.suptitle(f"FastICA fitting · {model_id}", y=0.995)
        figure.supxlabel("Iteration")
        figure.supylabel("Logcosh contrast")
        figure.legend(
            handles,
            labels,
            ncol=7,
            fontsize=9,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
        )
        figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.92))

    # Prevent notebook backends from auto-displaying the still-open pyplot figure;
    # the returned Figure remains renderable as the cell's final expression.
    plt.close(figure)
    return figure


def _plot_fitting_axis(
    axis: Any,
    *,
    layer: int,
    fitting: dict[str, Any],
    plt: Any,
    show_axis_labels: bool,
) -> None:
    history = fitting.get("objective_history")
    if not isinstance(history, dict):
        raise ValueError(
            f"layer {layer} has no objective history; this lens may predate objective recording"
        )
    iterations = np.asarray(history.get("iterations"), dtype=np.int64)
    percentiles = np.asarray(history.get("percentiles"), dtype=np.int64)
    values = np.asarray(history.get("values"), dtype=np.float64)
    if iterations.ndim != 1 or percentiles.ndim != 1:
        raise ValueError(f"layer {layer} has malformed objective-history axes")
    if values.shape != (len(iterations), len(percentiles)):
        raise ValueError(f"layer {layer} has inconsistent objective-history dimensions")

    colors = plt.colormaps["Blues"](np.linspace(0.18, 0.72, 5))
    for index, lower in enumerate((0, 10, 20, 30, 40)):
        upper = 100 - lower
        axis.fill_between(
            iterations,
            values[:, _percentile_index(percentiles, lower, layer)],
            values[:, _percentile_index(percentiles, upper, layer)],
            color=colors[index],
            alpha=0.55,
            linewidth=0,
            label=f"p{lower}–p{upper}",
        )
    median = values[:, _percentile_index(percentiles, 50, layer)]
    axis.plot(iterations, median, color="#172554", linewidth=2.2, label="median")

    gaussian = fitting.get("gaussian_objective")
    if isinstance(gaussian, (int, float)) and np.isfinite(gaussian):
        axis.axhline(
            float(gaussian),
            color="#94a3b8",
            linestyle="--",
            linewidth=1.1,
            label="Gaussian baseline",
        )

    axis.set_title(f"Layer {layer}")
    if show_axis_labels:
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Logcosh contrast")
    axis.grid(axis="y", alpha=0.2)


def _percentile_index(percentiles: np.ndarray[Any, Any], value: int, layer: int) -> int:
    matches = np.flatnonzero(percentiles == value)
    if len(matches) != 1:
        raise ValueError(f"layer {layer} objective history does not contain p{value}")
    return int(matches[0])
