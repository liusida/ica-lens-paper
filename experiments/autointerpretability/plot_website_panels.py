#!/usr/bin/env python3
"""Render the four compact autointerpretability panels used by the project site."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from icalens.experiments.autointerpretability_figure import (
    STYLES,
    _bootstrap_mean,
    _load,
    _panel_rows,
    _seed,
)

PYTHIA_LAYERS = np.arange(6, dtype=np.int64)
PYTHIA_SERIES = (
    (
        "SAE",
        "#B45F4D",
        "o",
        np.asarray([0.70394331, 0.56012360, 0.49639247, 0.38281348, 0.33272601, 0.24662429]),
        np.asarray([0.03972324, 0.04443642, 0.04136648, 0.04073532, 0.03856153, 0.03257183]),
    ),
    (
        "ICA (arbitrary sign)",
        "#93A9CC",
        "D",
        np.asarray([0.37191437, 0.31686483, 0.25267557, 0.22717377, 0.25964163, 0.23028512]),
        np.asarray([0.04427850, 0.04168086, 0.03881106, 0.03480952, 0.03796052, 0.03088091]),
    ),
    (
        "ICA (stronger-tail sign)",
        "#3D5F99",
        "^",
        np.asarray([0.54923860, 0.48432994, 0.41984677, 0.39058566, 0.35800408, 0.36984066]),
        np.asarray([0.03989426, 0.03585512, 0.03485157, 0.03481029, 0.03401844, 0.02825864]),
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt2", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    models = (
        ("gpt2", "GPT-2 small", _load(args.gpt2.resolve())),
        ("gemma", "Gemma 2 2B", _load(args.gemma.resolve())),
        ("qwen", "Qwen 3.5 9B Base", _load(args.qwen.resolve())),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _render_pythia(output, force=args.force)
    for index, (stem, title, payload) in enumerate(models):
        _render_model(stem, title, payload, output, index=index, force=args.force)


def _paths(output: Path, stem: str, *, force: bool) -> tuple[Path, Path, Path]:
    paths = (
        output / f"auto-interpretability-{stem}-clean.png",
        output / f"auto-interpretability-{stem}-clean.pdf",
        output / f"auto-interpretability-{stem}-clean.txt",
    )
    if not force:
        for path in paths:
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --force to replace it")
    return paths


def _style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _figure(title: str) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.75, 2.55))
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.24, top=0.72)
    axis.set_title(title, loc="left", fontweight="bold", pad=4)
    axis.set_ylim(0.0, 0.8)
    axis.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8))
    axis.set_ylabel("Interpretability score")
    axis.set_xlabel("Layer")
    axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    return figure, axis


def _save(figure: Any, paths: tuple[Path, Path, Path], companion: str) -> None:
    import matplotlib.pyplot as plt

    png, pdf, text = paths
    figure.savefig(png, dpi=100)
    figure.savefig(pdf)
    plt.close(figure)
    text.write_text(companion, encoding="utf-8")
    print(png)
    print(pdf)
    print(text)


def _render_pythia(output: Path, *, force: bool) -> None:
    import matplotlib.pyplot as plt

    paths = _paths(output, "pythia", force=force)
    with plt.rc_context(_style()):
        figure, axis = _figure("Pythia 70M reproduction")
        for label, color, marker, means, ci95 in PYTHIA_SERIES:
            axis.errorbar(
                PYTHIA_LAYERS,
                means,
                yerr=ci95,
                label=label,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=3.5,
                elinewidth=0.8,
                capsize=2.0,
            )
        axis.set_xticks(PYTHIA_LAYERS)
        figure.legend(
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.56, 0.98),
            columnspacing=1.1,
            handlelength=1.8,
        )
        companion = (
            "Pythia 70M reproduction of the Cunningham et al. comparison.\n"
            "Lines show mean automated-interpretability score over 150 features; error bars "
            "show 95% intervals.\n"
        )
        _save(figure, paths, companion)


def _render_model(
    stem: str,
    title: str,
    payload: dict[str, Any],
    output: Path,
    *,
    index: int,
    force: bool,
) -> None:
    import matplotlib.pyplot as plt

    paths = _paths(output, stem, force=force)
    rows = _panel_rows(payload)
    lines = [
        f"{title}: mean automated-interpretability score and feature-bootstrap 95% intervals.",
        "layer\tmethod\tdefined/selected\tmean\tbootstrap_95_low\tbootstrap_95_high",
    ]
    with plt.rc_context(_style()):
        figure, axis = _figure(title)
        for method in ("sae", "ica"):
            observed = sorted(
                (row for row in rows if row["method"] == method and row["scores"].size),
                key=lambda row: row["layer"],
            )
            label, color, marker = STYLES[method]
            summaries = [
                _bootstrap_mean(row["scores"], _seed(index, row["layer"], method))
                for row in observed
            ]
            means = np.asarray([item[0] for item in summaries])
            lows = np.asarray([item[1] for item in summaries])
            highs = np.asarray([item[2] for item in summaries])
            layers = np.asarray([row["layer"] for row in observed])
            axis.errorbar(
                layers,
                means,
                yerr=(means - lows, highs - means),
                label=label,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=3.5,
                elinewidth=0.8,
                capsize=2.0,
            )
            for row, (mean, low, high) in zip(observed, summaries, strict=True):
                lines.append(
                    f"{row['layer']}\t{label}\t{row['defined']}/{row['selected']}\t"
                    f"{mean:.6f}\t{low:.6f}\t{high:.6f}"
                )
        axis.set_xticks(sorted({row["layer"] for row in rows}))
        figure.legend(
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.56, 0.98),
            columnspacing=1.5,
            handlelength=2.0,
        )
        _save(figure, paths, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
