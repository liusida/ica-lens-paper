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
    return paths


def _plot_model_summary(axis: Any, lens: ICALens, *, colors: Any) -> int:
    metadata = lens.metadata
    fitting = [metadata["layers"][str(layer)]["fitting"] for layer in lens.available_layers]
    if not fitting:
        raise ValueError(f"ICA Lens {lens.model_id!r} has no fitted layers")
    histories = [item.get("objective_history") for item in fitting]
    if any(not isinstance(history, dict) for history in histories):
        raise ValueError(f"ICA Lens {lens.model_id!r} has layers without objective histories")

    first = histories[0]
    assert isinstance(first, dict)
    iterations = np.asarray(first.get("iterations"), dtype=np.int64)
    percentiles = np.asarray(first.get("percentiles"), dtype=np.int64)
    values: list[np.ndarray[Any, Any]] = []
    for history in histories:
        assert isinstance(history, dict)
        if not np.array_equal(iterations, np.asarray(history.get("iterations"))):
            raise ValueError(f"ICA Lens {lens.model_id!r} has inconsistent recorded iterations")
        if not np.array_equal(percentiles, np.asarray(history.get("percentiles"))):
            raise ValueError(f"ICA Lens {lens.model_id!r} has inconsistent percentiles")
        layer_values = np.asarray(history.get("values"), dtype=np.float64)
        if layer_values.shape != (len(iterations), len(percentiles)):
            raise ValueError(f"ICA Lens {lens.model_id!r} has malformed objective histories")
        values.append(layer_values)
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
    return len(fitting)


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
