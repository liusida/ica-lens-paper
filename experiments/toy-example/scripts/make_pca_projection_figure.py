#!/usr/bin/env python3
"""Plot toy-example token activations along two principal components."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from safetensors.torch import load_file

COLORS = {"background": "#B8C1CC", "gaussian": "#777777"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=root / "work/selected")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/pca-projection-distributions.png",
    )
    parser.add_argument("--components", default="1,150")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    capture = args.capture.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    component_numbers = tuple(int(value) for value in args.components.split(","))
    if len(component_numbers) != 2 or min(component_numbers) < 1:
        raise ValueError("--components must contain two positive one-based ranks")

    samples = json.loads((capture / "samples.json").read_text(encoding="utf-8"))
    x = load_file(capture / "activations.safetensors")["layer_00"].double()
    background = torch.tensor([sample["role"] == "background" for sample in samples])
    center = x[background].mean(dim=0)
    centered = x - center
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    indices = torch.tensor([component - 1 for component in component_numbers])
    if int(indices.max()) >= x.shape[1]:
        raise ValueError(f"component rank exceeds hidden size {x.shape[1]}")
    selected = order[indices]
    directions = eigenvectors[:, selected]
    projections = centered @ directions

    _render(
        projections.numpy(),
        background.numpy(),
        component_numbers,
        eigenvalues[selected].numpy(),
        tuple(str(sample["token"]) for sample in samples),
        output,
    )
    print(output)


def _render(
    projections: np.ndarray,
    background: np.ndarray,
    component_numbers: tuple[int, int],
    eigenvalues: np.ndarray,
    tokens: tuple[str, ...],
    output: Path,
) -> None:
    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    limit = float(np.ceil(np.abs(projections).max()))
    bins = np.linspace(-limit, limit, 61)
    grid = np.linspace(-limit, limit, 800)
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharex=True, sharey=True)
        for panel, (axis, component, values, eigenvalue) in enumerate(
            zip(axes, component_numbers, projections.T, eigenvalues, strict=True)
        ):
            background_values = values[background]
            mean = float(background_values.mean())
            sigma = float(background_values.std(ddof=1))
            centered_values = values - values.mean()
            second_moment = float(np.mean(centered_values**2))
            excess = float(np.mean(centered_values**4) / second_moment**2 - 3.0)
            axis.hist(
                background_values,
                bins=bins,
                density=True,
                color=COLORS["background"],
                alpha=0.75,
                edgecolor="none",
                zorder=1,
            )
            axis.plot(
                grid,
                np.exp(-0.5 * ((grid - mean) / sigma) ** 2)
                / (sigma * np.sqrt(2.0 * np.pi)),
                color=COLORS["gaussian"],
                linestyle=(0, (4, 2)),
                linewidth=1.5,
                zorder=2,
            )
            standardized = (values - mean) / sigma
            outlier_indices = np.argsort(np.abs(standardized))[-3:][::-1]
            label_x = (0.60, 0.60, 0.60)
            label_y = (0.72, 0.57, 0.42)
            for rank, index in enumerate(outlier_indices):
                token = tokens[int(index)].replace(" ", "␣")
                axis.annotate(
                    token,
                    xy=(values[index], 0.002),
                    xycoords="data",
                    xytext=(label_x[rank], label_y[rank]),
                    textcoords="axes fraction",
                    ha="left",
                    va="center",
                    fontsize=6.2,
                    fontfamily="monospace",
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "facecolor": "white",
                        "edgecolor": "#6B7280",
                        "linewidth": 0.55,
                    },
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#6B7280",
                        "linewidth": 0.55,
                    },
                    zorder=5,
                )
            axis.text(
                0.03,
                0.94,
                rf"$K={excess:.2f}$" + "\n" + rf"$\sigma={sigma:.2f}$",
                ha="left",
                va="top",
                transform=axis.transAxes,
            )
            axis.text(
                0.03,
                0.69,
                rf"$\lambda={eigenvalue:.2f}$",
                ha="left",
                va="top",
                fontsize=8,
                color="#4b5563",
                transform=axis.transAxes,
            )
            axis.set_title(
                f"{chr(65 + panel)} · PC{component}",
                loc="left",
                fontweight="bold",
                pad=4,
            )
            axis.grid(axis="y", color="0.88", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Density")
        figure.supxlabel("Raw projection", y=0.04)
        figure.legend(
            handles=[
                Line2D(
                    [], [], color=COLORS["gaussian"], linestyle=(0, (4, 2)),
                    linewidth=1.5, label="Gaussian fit",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.52, 1.01),
            ncol=1,
            frameon=False,
        )
        figure.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.79, wspace=0.16)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300)
        plt.close(figure)


if __name__ == "__main__":
    main()
