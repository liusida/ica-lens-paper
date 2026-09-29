#!/usr/bin/env python3
"""Render tokenwise activation patterns for cached feature ranks 1 through 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"ica": "#3D5F99", "sae": "#B45F4D"}
RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def ordered_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitudes = np.abs(values)
    maxima = magnitudes.max(axis=1, keepdims=True)
    normalized = np.divide(magnitudes, maxima, out=np.zeros_like(magnitudes), where=maxima > 0)
    peaks = normalized.argmax(axis=1)
    order = np.lexsort((-normalized.max(axis=1), peaks))
    return normalized[order], order


def selected_rows(feature_ids: np.ndarray, rankings: np.ndarray, top_n: int) -> np.ndarray:
    selected = set(int(value) for value in rankings[:, :top_n].ravel() if value >= 0)
    return np.asarray([index for index, value in enumerate(feature_ids) if int(value) in selected])


def ridge_panel(
    axis: plt.Axes,
    values: np.ndarray,
    feature_ids: np.ndarray,
    token_labels: np.ndarray,
    *,
    color: str,
    prefix: str,
    show_tokens: bool,
) -> None:
    normalized, order = ordered_rows(values)
    feature_ids = feature_ids[order]
    count, token_count = normalized.shape
    x = np.arange(token_count)
    for token in range(0, token_count, 2):
        axis.axvspan(token - 0.5, token + 0.5, color="0.965", zorder=0)
    for row, response in enumerate(normalized):
        baseline = count - 1 - row
        axis.axhline(baseline, color="0.72", linewidth=0.35, zorder=1)
        axis.fill_between(
            x, baseline, baseline + 0.82 * response, color=color, alpha=0.60, linewidth=0, zorder=2
        )
        axis.plot(x, baseline + 0.82 * response, color=color, linewidth=0.45, zorder=3)
    axis.set_xlim(-0.5, token_count - 0.5)
    axis.set_ylim(-0.15, count - 0.05 + 0.82)
    axis.set_yticks(np.arange(count), [f"{prefix}{value}" for value in feature_ids[::-1]])
    axis.tick_params(axis="y", length=0, pad=1, labelsize=3.7)
    axis.set_xticks(x)
    if show_tokens:
        axis.set_xticklabels(token_labels, rotation=50, ha="right", rotation_mode="anchor")
        axis.tick_params(axis="x", length=1.8, pad=1, labelsize=4.3)
    else:
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.55)


def render_model(results: Path, output: Path, run: dict, model_key: str, top_n: int) -> Path:
    model = run["resolved"]["models"][model_key]
    layers = [int(value) for value in model["layers"]]
    positions = ("First layer", "Middle layer", "Last layer")
    with plt.rc_context(RC_PARAMS):
        figure, axes = plt.subplots(3, 2, figsize=(5.5, 6.35))
        figure.subplots_adjust(
            left=0.075, right=0.993, bottom=0.105, top=0.92, hspace=0.26, wspace=0.18
        )
        figure.text(
            0.29,
            0.975,
            "ICA scores",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=COLORS["ica"],
        )
        figure.text(
            0.755,
            0.975,
            "SAE activations",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=COLORS["sae"],
        )
        for row, (layer, position) in enumerate(zip(layers, positions, strict=True)):
            with np.load(
                results / model_key / f"layer_{layer:02d}.npz", allow_pickle=False
            ) as data:
                labels = data["token_labels"]
                ica_rows = selected_rows(data["ica_feature_ids"], data["ica_top_features"], top_n)
                sae_rows = selected_rows(data["sae_feature_ids"], data["sae_top_features"], top_n)
                ridge_panel(
                    axes[row, 0],
                    data["ica_scores"][ica_rows],
                    data["ica_feature_ids"][ica_rows],
                    labels,
                    color=COLORS["ica"],
                    prefix="C",
                    show_tokens=row == 2,
                )
                ridge_panel(
                    axes[row, 1],
                    data["sae_activations"][sae_rows],
                    data["sae_feature_ids"][sae_rows],
                    labels,
                    color=COLORS["sae"],
                    prefix="F",
                    show_tokens=row == 2,
                )
            axes[row, 0].text(
                -0.02,
                1.03,
                f"{'ACE'[row]} · {position} (L{layer})",
                transform=axes[row, 0].transAxes,
                ha="left",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                clip_on=False,
            )
            axes[row, 1].text(
                -0.02,
                1.03,
                f"{'BDF'[row]} · {position} (L{layer})",
                transform=axes[row, 1].transAxes,
                ha="left",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                clip_on=False,
            )
        destination = output / f"activation-pattern-{model_key}-top-{top_n}.png"
        figure.savefig(destination, dpi=300, facecolor="white")
        plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results-top5")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    output = args.output.expanduser().resolve()
    run = json.loads((results / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("resolved", {}).get("schema_version") != 2:
        raise ValueError(f"top-5 activation-pattern run is not complete: {results}")
    output.mkdir(parents=True, exist_ok=True)
    destinations = [
        output / f"activation-pattern-{model}-top-{top_n}.png"
        for model in run["resolved"]["models"]
        for top_n in range(1, 6)
    ]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"{existing[0]} exists; pass --force to replace outputs")
    for model in run["resolved"]["models"]:
        for top_n in range(1, 6):
            print(render_model(results, output, run, model, top_n))


if __name__ == "__main__":
    main()
