#!/usr/bin/env python3
"""Build the paper overview from the toy-example projection data."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.stats import norm

COLORS = {
    "background": "#B8C1CC",
    "target": "#B45F4D",
    "concept": "#3568A8",
    "random": "#777777",
    "direction": "#4F8F78",
    "ica_direction": "#3568A8",
}
ORDER = ("a", "b", "e", "c", "ica")
TITLES = (
    "F · Random",
    "G · One token",
    "H · Random mean",
    "I · Related mean",
    "J · ICA",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "work/render")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--activation-index",
        type=int,
        help="Index from 0 to 299 within version 2's sampled background activations.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.input.expanduser().resolve()
    root = Path(__file__).resolve().parent.parent
    output_arg = args.output or root / f"figures/overview-v{args.version}"
    output = output_arg.expanduser().resolve()
    outputs = [output.with_suffix(suffix) for suffix in (".png", ".pdf", ".txt")]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"output exists: {existing[0]}; pass --force to replace it")
    data = np.load(source / "overview-data.npz")
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if args.version == 1:
        if args.activation_index is not None:
            raise ValueError("--activation-index is available only for --version 2")
        _render_v1(data, results, output, seed=args.seed)
    else:
        _render_v2(
            data,
            results,
            output,
            seed=args.seed,
            activation_index=args.activation_index,
        )
    outputs[2].write_text(_caption(args.version), encoding="utf-8")
    for path in outputs:
        print(path)


def _render_v1(
    data: np.lib.npyio.NpzFile, results: dict, output: Path, *, seed: int
) -> None:
    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    background = data["background"].astype(bool)
    concept = data["concept"].astype(bool)
    random_indices = data["random_indices"].astype(int)
    target = int(data["target_index"])
    values = {
        "a": data["projection_a"],
        "b": data["projection_b"],
        "c": data["projection_c"],
        "ica": data["projection_ica"],
        "e": data["projection_e"],
    }
    if not len(values["ica"]):
        raise ValueError("overview requires the ICA direction; rerun analyze.py with --ica-lens")
    maximum = max(float(np.abs(item).max()) for item in values.values())
    limit = float(np.ceil(maximum))
    bins = np.linspace(-limit, limit, 31)
    grid = np.linspace(-limit, limit, 500)
    projection_limit = float(np.ceil(np.abs(values["a"]).max()))
    projection_bins = np.linspace(-projection_limit, projection_limit, 31)
    projection_grid = np.linspace(-projection_limit, projection_limit, 500)
    ica_limit = 18.0
    ica_bins = np.linspace(-ica_limit, ica_limit, 61)
    ica_grid = np.linspace(-ica_limit, ica_limit, 800)
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(8.0, 5.3))
        outer = figure.add_gridspec(2, 1, height_ratios=(1.7, 0.68), hspace=0.62)
        top = outer[0].subgridspec(1, 3, width_ratios=(1.05, 0.82, 0.82), wspace=0.24)
        plane = figure.add_subplot(top[0])
        random_column = top[1].subgridspec(2, 1, height_ratios=(0.72, 1), hspace=0.08)
        strip = figure.add_subplot(random_column[0])
        distribution = figure.add_subplot(random_column[1], sharex=strip)
        ica_column = top[2].subgridspec(2, 1, height_ratios=(0.72, 1), hspace=0.08)
        ica_strip = figure.add_subplot(ica_column[0])
        ica_distribution = figure.add_subplot(ica_column[1], sharex=ica_strip)
        bottom = outer[1].subgridspec(1, 5, wspace=0.12)
        lower_axes = [figure.add_subplot(bottom[0])]
        lower_axes.extend(
            figure.add_subplot(bottom[index], sharex=lower_axes[0], sharey=lower_axes[0])
            for index in range(1, 5)
        )

        _plot_plane(plane, data, background, concept, seed=seed)
        _plot_projection_strip(
            strip, values["a"], background, projection_limit=projection_limit, seed=seed,
            title="B · Projected activations\n" + r"along $\mathbf{w}$",
            direction_color=COLORS["direction"],
        )
        _plot_distribution(
            distribution,
            values["a"],
            background,
            bins=projection_bins,
            grid=projection_grid,
            title="D · Projection distribution",
            xlabel=r"Projection onto $\mathbf{w}$",
        )
        related_for_ica = np.flatnonzero(concept)
        related_for_ica = related_for_ica[np.argsort(values["ica"][related_for_ica])[-4:]]
        _plot_projection_strip(
            ica_strip, values["ica"], background, projection_limit=ica_limit, seed=seed,
            title="C · Projected activations\n" + r"along $\mathbf{v}$",
            direction_color=COLORS["ica_direction"],
            highlighted=related_for_ica,
        )
        _plot_distribution(
            ica_distribution,
            values["ica"],
            background,
            bins=ica_bins,
            grid=ica_grid,
            title="E · Projection distribution",
            xlabel=r"Projection onto $\mathbf{v}$",
            highlighted=related_for_ica,
        )
        _plot_examples(
            lower_axes,
            values,
            background,
            concept,
            random_indices,
            target,
            results,
            bins,
            grid,
        )

        legend_handles = [
            Line2D([], [], color=COLORS["target"], marker="o", linestyle="None",
                   markersize=4.5, label="Selected token in G"),
            Patch(facecolor=COLORS["concept"], edgecolor="none",
                  label="Related tokens"),
            Patch(facecolor=COLORS["random"], edgecolor="none", label="Random tokens"),
        ]
        figure.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
                      bbox_to_anchor=(0.5, 0.362), handlelength=1.2, columnspacing=1.5)
        figure.text(0.5, 0.025, "Raw projection", ha="center", fontsize=9)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.subplots_adjust(left=0.055, right=0.995, bottom=0.09, top=0.91)
        _add_story_frame(
            figure,
            bounds=(-0.012, 0.385, 1.017, 0.595),
            label="Projection onto a direction",
        )
        _add_story_frame(
            figure,
            bounds=(-0.012, 0.015, 1.017, 0.34),
            label="Comparing directions",
        )
        figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def _render_v2(
    data: np.lib.npyio.NpzFile,
    results: dict,
    output: Path,
    *,
    seed: int,
    activation_index: int | None,
) -> None:
    """Render the compact version-2 design without the direction-comparison row."""
    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    background = data["background"].astype(bool)
    concept = data["concept"].astype(bool)
    projection_w = data["projection_a"]
    projection_v = data["projection_ica"]
    if not len(projection_v):
        raise ValueError("overview requires the ICA direction; rerun analyze.py with --ica-lens")

    projection_limit = float(np.ceil(np.abs(projection_w).max()))
    projection_bins = np.linspace(-projection_limit, projection_limit, 31)
    projection_grid = np.linspace(-projection_limit, projection_limit, 500)
    ica_limit = 11.0
    ica_bins = np.linspace(-ica_limit, ica_limit, 61)
    ica_grid = np.linspace(-ica_limit, ica_limit, 800)
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(8.0, 3.55))
        columns = figure.add_gridspec(
            1, 3, width_ratios=(1.05, 0.82, 0.82), wspace=0.24
        )
        plane = figure.add_subplot(columns[0])
        random_column = columns[1].subgridspec(
            2, 1, height_ratios=(0.72, 1), hspace=0.08
        )
        strip_w = figure.add_subplot(random_column[0])
        distribution_w = figure.add_subplot(random_column[1], sharex=strip_w)
        ica_column = columns[2].subgridspec(
            2, 1, height_ratios=(0.72, 1), hspace=0.08
        )
        strip_v = figure.add_subplot(ica_column[0])
        distribution_v = figure.add_subplot(
            ica_column[1], sharex=strip_v, sharey=distribution_w
        )

        token_lookup = {
            int(item["index"]): item["token"].strip()
            for item in results["direction_c_tokens"]
        }
        rng = np.random.default_rng(seed)
        angle = np.deg2rad(-60)
        display_x = (
            np.cos(angle) * projection_w
            - np.sin(angle) * data["projection_plane"]
        )
        display_y = (
            np.sin(angle) * projection_w
            + np.cos(angle) * data["projection_plane"]
        )
        background_indices = np.flatnonzero(
            background & (np.abs(display_x) <= 8.5) & (np.abs(display_y) <= 8.5)
        )
        shown = rng.choice(
            background_indices,
            size=min(300, len(background_indices)),
            replace=False,
        )
        selected_activation = None
        if activation_index is not None:
            if not 0 <= activation_index < len(shown):
                raise ValueError(
                    f"--activation-index must be in [0, {len(shown)})"
                )
            selected_activation = int(shown[activation_index])
        shown_mask = np.zeros_like(background)
        shown_mask[shown] = True
        related_for_ica = _plot_plane(
            plane,
            data,
            background,
            concept,
            seed=seed,
            token_lookup=token_lookup,
            display_count=300,
            related_count=5,
            shown_indices=shown,
            activation_index=selected_activation,
        )
        displayed = np.concatenate((shown, related_for_ica))
        _plot_projection_strip(
            strip_w,
            projection_w,
            background,
            projection_limit=projection_limit,
            seed=seed,
            title="B · Projected activations\n" + r"along Random $\mathbf{w}$",
            direction_color=COLORS["direction"],
            highlighted=related_for_ica,
            shown=shown,
        )
        _plot_distribution(
            distribution_w,
            projection_w,
            shown_mask,
            bins=projection_bins,
            grid=projection_grid,
            title="D · Projection distribution",
            xlabel="",
            fit_indices=displayed,
        )
        _mark_top_k_cutoff(distribution_w, projection_w[displayed], k=5)
        top_w = displayed[np.argsort(projection_w[displayed])[-5:][::-1]]
        top_examples = [
            {"token": str(data["tokens"][index]), "projection": projection_w[index]}
            for index in top_w
        ]
        last_token_x = _add_token_row(
            distribution_w,
            top_examples,
            color=COLORS["random"],
            facecolor="white",
            edgecolor="#B8BEC6",
        )
        distribution_w.annotate(
            "",
            xy=(top_examples[-1]["projection"], 0.01),
            xycoords=("data", "axes fraction"),
            xytext=(last_token_x, -0.28),
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#9AA1A9",
                "linewidth": 0.75,
                "shrinkA": 0,
                "shrinkB": 1,
            },
            annotation_clip=False,
        )
        _plot_projection_strip(
            strip_v,
            projection_v,
            background,
            projection_limit=ica_limit,
            seed=seed,
            title="C · Projected activations\n" + r"along ICA $\mathbf{v}$",
            direction_color=COLORS["ica_direction"],
            highlighted=related_for_ica,
            shown=shown,
        )
        _plot_distribution(
            distribution_v,
            projection_v,
            shown_mask,
            bins=ica_bins,
            grid=ica_grid,
            title="E · Projection distribution",
            xlabel="",
            highlighted=related_for_ica,
            fit_indices=displayed,
        )
        _mark_top_k_cutoff(distribution_v, projection_v[displayed], k=5)
        top_v = displayed[np.argsort(projection_v[displayed])[-5:][::-1]]
        ica_top_examples = [
            {"token": str(data["tokens"][index]), "projection": projection_v[index]}
            for index in top_v
        ]
        last_ica_token_x = _add_token_row(
            distribution_v,
            ica_top_examples,
            color=COLORS["ica_direction"],
            facecolor="#EFF4FA",
            edgecolor=COLORS["ica_direction"],
        )
        distribution_v.annotate(
            "",
            xy=(ica_top_examples[-1]["projection"], 0.01),
            xycoords=("data", "axes fraction"),
            xytext=(last_ica_token_x, -0.28),
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["ica_direction"],
                "linewidth": 0.75,
                "shrinkA": 0,
                "shrinkB": 1,
            },
            annotation_clip=False,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.subplots_adjust(left=0.055, right=0.995, bottom=0.27, top=0.88)
        figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def _add_token_row(
    axis: plt.Axes,
    examples: list[dict],
    *,
    color: str,
    facecolor: str,
    edgecolor: str,
) -> float:
    labels = [
        axis.text(
            0.5,
            -0.35,
            item["token"].strip(),
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=6.5,
            fontfamily="monospace",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.17",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": 0.65,
            },
            clip_on=False,
        )
        for item in examples
    ]
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    widths = [label.get_bbox_patch().get_window_extent(renderer).width for label in labels]
    gap = -1.0
    total_width = sum(widths) + gap * (len(widths) - 1)
    cursor = axis.bbox.x0 + (axis.bbox.width - total_width) / 2
    centers: list[float] = []
    for label, width in zip(labels, widths, strict=True):
        center = cursor + width / 2
        center_fraction = (center - axis.bbox.x0) / axis.bbox.width
        label.set_x(center_fraction)
        centers.append(center_fraction)
        cursor += width + gap
    return centers[-1]


def _mark_top_k_cutoff(
    axis: plt.Axes,
    projections: np.ndarray,
    *,
    k: int,
) -> float:
    cutoff = float(np.partition(projections, -k)[-k])
    axis.axvline(
        cutoff,
        ymin=0.0,
        ymax=0.64,
        color="#8D949C",
        linestyle=(0, (2.2, 2.2)),
        linewidth=0.75,
        zorder=3,
    )
    axis.annotate(
        f"top-{k}",
        xy=(cutoff, 0.56),
        xycoords=("data", "axes fraction"),
        xytext=(-2, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        rotation=90,
        fontsize=6.5,
        color="#707780",
    )
    return cutoff


def _add_story_frame(
    figure: plt.Figure,
    *,
    bounds: tuple[float, float, float, float],
    label: str,
) -> None:
    x, y, width, height = bounds
    frame = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        transform=figure.transFigure,
        facecolor="none",
        edgecolor="#B8BEC6",
        linewidth=0.8,
        zorder=-10,
        clip_on=False,
    )
    figure.add_artist(frame)
    figure.text(
        x + 0.012,
        y + height,
        label,
        color="#5E6670",
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )


def _plot_plane(
    axis: plt.Axes,
    data: np.lib.npyio.NpzFile,
    background: np.ndarray,
    concept: np.ndarray,
    *,
    seed: int,
    token_lookup: dict[int, str] | None = None,
    display_count: int = 100,
    related_count: int = 4,
    shown_indices: np.ndarray | None = None,
    activation_index: int | None = None,
) -> np.ndarray:
    x = data["projection_a"]
    y = data["projection_plane"]
    angle = np.deg2rad(-60)
    cosine, sine = np.cos(angle), np.sin(angle)

    def rotate(
        first: np.ndarray | float, second: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        first_array = np.asarray(first)
        second_array = np.asarray(second)
        return (
            cosine * first_array - sine * second_array,
            sine * first_array + cosine * second_array,
        )

    display_x, display_y = rotate(x, y)
    indices = np.flatnonzero(background)
    shown = shown_indices
    if shown is None:
        shown = np.random.default_rng(seed).choice(
            indices,
            size=min(display_count, len(indices)),
            replace=False,
        )
    axis.axhline(0, color="0.82", linewidth=0.6, zorder=0)
    axis.axvline(0, color="0.82", linewidth=0.6, zorder=0)
    axis.scatter(display_x[shown], display_y[shown], s=8, color=COLORS["background"], alpha=0.45,
                 linewidths=0, rasterized=True)
    related = np.flatnonzero(concept)
    visible = related[
        (np.abs(display_x[related]) <= 9.25) & (np.abs(display_y[related]) <= 9.25)
    ]
    if len(visible) < related_count:
        raise ValueError(
            f"fewer than {related_count} related-token activations fit in panel A"
        )
    related = visible[
        np.argsort(data["projection_ica"][visible])[-related_count:]
    ]
    axis.scatter(display_x[related], display_y[related], s=18, color=COLORS["concept"],
                 edgecolor="white", linewidth=0.5, zorder=3)
    if token_lookup is not None:
        rightmost = max(
            (int(index) for index in related),
            key=lambda index: (len(token_lookup[index]), display_x[index]),
        )
        axis.annotate(
            token_lookup[rightmost],
            xy=(display_x[rightmost], display_y[rightmost]),
            xytext=(16, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6.5,
            fontfamily="monospace",
            color=COLORS["ica_direction"],
            bbox={
                "boxstyle": "round,pad=0.17",
                "facecolor": "#EFF4FA",
                "edgecolor": COLORS["ica_direction"],
                "linewidth": 0.65,
            },
            zorder=5,
        )
    related_point = int(related[np.argmax(data["projection_ica"][related])])
    related_coordinates = np.array([x[related_point], y[related_point]])
    v_coordinates = data["plane_v_coordinates"]
    if len(v_coordinates) != 2:
        raise ValueError("overview data does not define the ICA direction in the display plane")
    related_projection_w = np.array([related_coordinates[0], 0.0])
    related_projection_v = np.dot(related_coordinates, v_coordinates) * v_coordinates
    related_display = np.asarray(rotate(*related_coordinates))
    projection_w_display = np.asarray(rotate(*related_projection_w))
    projection_v_display = np.asarray(rotate(*related_projection_v))
    for projected, direction_color in (
        (projection_w_display, COLORS["direction"]),
        (projection_v_display, COLORS["ica_direction"]),
    ):
        axis.plot(
            [related_display[0], projected[0]],
            [related_display[1], projected[1]],
            color=direction_color,
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
        axis.scatter(
            [projected[0]],
            [projected[1]],
            facecolor="white",
            edgecolor=COLORS["concept"],
            linewidth=1.1,
            s=22,
            zorder=4,
        )
    candidates = shown[
        (np.abs(display_x[shown]) <= 8.5)
        & (np.abs(display_y[shown]) <= 8.5)
        & (display_y[shown] >= 1.0)
    ]
    if activation_index is not None:
        if (
            abs(display_x[activation_index]) > 8.5
            or abs(display_y[activation_index]) > 8.5
        ):
            raise ValueError(
                "--activation-index must select a displayed activation inside panel A"
            )
        point = activation_index
    elif token_lookup is not None:
        point = int(candidates[np.argsort(display_y[candidates])[-3]])
    else:
        typical_distance = np.percentile(np.abs(y[indices]), 95)
        point = int(
            candidates[np.argmin(np.abs(np.abs(y[candidates]) - typical_distance))]
        )
    point_x, point_y = rotate(x[point], y[point])
    projection_x, projection_y = rotate(x[point], 0)
    point_coordinates = np.array([x[point], y[point]])
    point_projection_v = np.dot(point_coordinates, v_coordinates) * v_coordinates
    projection_v_x, projection_v_y = rotate(*point_projection_v)
    axis.plot([projection_x, point_x], [projection_y, point_y], color=COLORS["direction"],
              linestyle="--", linewidth=1.1)
    axis.plot([projection_v_x, point_x], [projection_v_y, point_y],
              color=COLORS["ica_direction"],
              linestyle="--", linewidth=1.1)
    axis.scatter(
        [point_x],
        [point_y],
        facecolor=COLORS["background"],
        edgecolor=COLORS["random"],
        linewidth=1.0,
        s=24,
        zorder=4,
    )
    axis.scatter([projection_x, projection_v_x], [projection_y, projection_v_y],
                 facecolor="white", edgecolor=COLORS["random"], s=22, zorder=4)
    span = 10.0
    tip_x, tip_y = rotate(span, 0)
    axis.annotate("", xy=(tip_x, tip_y), xytext=(-tip_x, -tip_y),
                  arrowprops={"arrowstyle": "-|>", "color": COLORS["direction"],
                              "linewidth": 1.4, "shrinkA": 0, "shrinkB": 0})
    axis.text(tip_x * 0.65 + 0.15, tip_y * 1.02 + 0.2, r"Random $\mathbf{w}$",
              ha="right", va="center",
              color=COLORS["direction"], fontweight="bold")
    v_x, v_y = rotate(span * v_coordinates[0], span * v_coordinates[1])
    axis.annotate("", xy=(v_x, v_y), xytext=(-v_x, -v_y),
                  arrowprops={"arrowstyle": "-|>", "color": COLORS["ica_direction"],
                              "linewidth": 1.4, "shrinkA": 0, "shrinkB": 0})
    axis.text(v_x * 0.98 + 0.5, v_y * 0.98 + 0.8, r"ICA $\mathbf{v}$",
              ha="right" if v_x > 0 else "left", va="bottom",
              color=COLORS["ica_direction"], fontweight="bold")
    axis.text(point_x, point_y, "activation  ", color=COLORS["random"],
              ha="right", va="center")
    axis.set_title("A · Activation space in a 2D plane", loc="left", fontweight="bold")
    axis.set_xlabel(r"Plane coordinate $u_1$")
    axis.set_ylabel(r"Plane coordinate $u_2$")
    axis.set_xlim(-10, 10)
    axis.set_ylim(-10, 10)
    axis.set_xticks(np.arange(-10, 11, 5))
    axis.set_yticks(np.arange(-10, 11, 5))
    axis.set_box_aspect(1)
    axis.spines[["top", "right"]].set_visible(False)
    return related


def _plot_projection_strip(
    axis: plt.Axes,
    values: np.ndarray,
    background: np.ndarray,
    *,
    projection_limit: float,
    seed: int,
    title: str,
    direction_color: str,
    highlighted: np.ndarray | None = None,
    shown: np.ndarray | None = None,
) -> None:
    if shown is None:
        rng = np.random.default_rng(seed)
        indices = np.flatnonzero(background)
        shown = rng.choice(indices, size=min(100, len(indices)), replace=False)
    left, right = -projection_limit, projection_limit
    axis.annotate("", xy=(right, 0), xytext=(left, 0),
                  arrowprops={"arrowstyle": "-|>", "color": direction_color,
                              "linewidth": 1.4, "shrinkA": 0, "shrinkB": 0},
                  zorder=0)
    axis.scatter(values[shown], np.zeros(len(shown)), s=13,
                 facecolors="white", edgecolors=COLORS["random"], alpha=0.75,
                 linewidths=0.7, rasterized=True, zorder=3)
    if highlighted is not None:
        axis.scatter(values[highlighted], np.zeros(len(highlighted)), s=18,
                     facecolor="white", edgecolor=COLORS["concept"],
                     linewidth=1.0, zorder=4)
    axis.set_ylim(-0.04, 0.04)
    axis.set_xlim(left, right)
    axis.set_yticks([])
    axis.tick_params(axis="x", labelbottom=False, bottom=False)
    axis.set_title(title, loc="left", y=0.82, fontweight="bold")
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)


def _plot_distribution(
    axis: plt.Axes,
    values: np.ndarray,
    background: np.ndarray,
    *,
    bins: np.ndarray,
    grid: np.ndarray,
    title: str,
    xlabel: str,
    highlighted: np.ndarray | None = None,
    fit_indices: np.ndarray | None = None,
) -> None:
    fit = values if fit_indices is None else values[fit_indices]
    axis.hist(values[background], bins=bins, density=True,
              color=COLORS["background"], alpha=0.8)
    axis.plot(grid, norm.pdf(grid, loc=fit.mean(), scale=fit.std(ddof=1)), color="0.45",
              linestyle="--", linewidth=1.2, label="Gaussian fit")
    top = axis.get_ylim()[1]
    if highlighted is not None:
        axis.scatter(values[highlighted], np.full(len(highlighted), 0.025 * top),
                     color=COLORS["concept"], marker="|", s=45, linewidths=1.1, zorder=3)
    axis.set_ylim(0, 1.18 * top)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", color="0.88", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_examples(axes: list[plt.Axes], values: dict[str, np.ndarray], background: np.ndarray,
                   concept: np.ndarray, random_indices: np.ndarray, target: int,
                   results: dict, bins: np.ndarray, grid: np.ndarray) -> None:
    stats = results["statistics"]
    labels = {"a": "A", "b": "B", "c": "C", "ica": "D", "e": "E"}
    for index, (axis, name, title) in enumerate(zip(axes, ORDER, TITLES, strict=True)):
        projection = values[name]
        fit = projection
        axis.hist(projection[background], bins=bins, density=True,
                  color=COLORS["background"], alpha=0.8)
        axis.plot(grid, norm.pdf(grid, loc=fit.mean(), scale=fit.std(ddof=1)), color="0.45",
                  linestyle="--", linewidth=1.2)
        if name == "b":
            axis.scatter(projection[concept], np.full(int(concept.sum()), 0.008),
                         color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
            axis.scatter(projection[target], 0.008, color=COLORS["target"], s=24, zorder=4)
        elif name in {"c", "ica"}:
            axis.scatter(projection[concept], np.full(int(concept.sum()), 0.008),
                         color=COLORS["concept"], marker="|", s=55, linewidths=1.2)
        elif name == "e":
            axis.scatter(projection[random_indices], np.full(len(random_indices), 0.008),
                         color=COLORS["random"], marker="|", s=55, linewidths=1.2)
        sigma = projection.std(ddof=1)
        kurtosis = stats[labels[name]]["excess_kurtosis"]
        stats_text = rf"$K$ = {kurtosis:.2f}" + f"\nσ = {sigma:.2f}"
        stats_x = 1.05 if name == "ica" else 0.97
        axis.text(stats_x, 0.95, stats_text,
                  transform=axis.transAxes, ha="right", va="top", fontsize=8)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="0.88", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            axis.set_ylabel("Density")
        else:
            axis.tick_params(axis="y", labelleft=False)


def _caption(version: int) -> str:
    background_count, related_count = (300, 5) if version == 2 else (100, 4)
    opening = (
        f"Projection and non-Gaussianity (figure design v{version}). "
        f"(A) A seeded sample of {background_count} centered GPT-2 Layer-0 "
        f"token activations, together with {related_count} related-token activations, "
        "in the plane spanned "
        r"by a seeded random direction $\mathbf{w}$ and "
        r"the ICA direction $\mathbf{v}$, shown in an arbitrary orthonormal coordinate system "
        "$(u_1,u_2)$ after a 60-degree clockwise display rotation. "
        "Orthogonal projection maps each activation to a scalar coordinate along a direction. "
    )
    if version == 2:
        return opening + (
            r"Panels (B) and (C) show projected points along $\mathbf{w}$ and $\mathbf{v}$, "
            r"respectively; panels (D) and (E) show their full projection distributions. "
            "All distributions use the same 50,256 non-special GPT-2 token "
            "activations.\n"
        )
    return opening + (
        r"Panels (B) and (C) show projected points along $\mathbf{w}$ and $\mathbf{v}$, "
        r"respectively; panels (D) and (E) show their full projection distributions. "
        "The lower panels show distributions from (F) a random direction, "
        "(G) one related-token direction, (H) the mean direction of random tokens, (I) the mean "
        "direction of related tokens, and (J) an ICA-discovered "
        "direction. $K$ denotes excess kurtosis. All distributions use the same 50,256 "
        "non-special GPT-2 token activations. "
        "The coherent token family forms a concentrated tail most strongly along the "
        "ICA-discovered direction.\n"
    )


if __name__ == "__main__":
    main()
