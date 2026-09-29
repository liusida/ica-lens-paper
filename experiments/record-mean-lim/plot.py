#!/usr/bin/env python3
"""Plot aggregate FastICA limit and non-Gaussianity trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "runs/main"
DEFAULT_OUTPUT = ROOT / "figures/mean-limit-vs-fit-quality.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    summary = json.loads((source / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError(f"incomplete pilot run: {source}")
    layers = summary["resolved"]["layers"]
    loaded = [np.load(source / "layers" / f"layer-{layer:02d}.npz") for layer in layers]
    try:
        iterations = loaded[0]["iterations"]
        for data in loaded[1:]:
            if not np.array_equal(data["iterations"], iterations):
                raise ValueError("layer iteration grids differ")
        limits = np.concatenate([data["component_limit"] for data in loaded], axis=1)
        logcosh = np.concatenate([data["logcosh_objective"] for data in loaded], axis=1)
        kurtosis = np.concatenate([data["excess_kurtosis"] for data in loaded], axis=1)
    finally:
        for data in loaded:
            data.close()
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    lim_axis = axis
    quality_axis = axis.twinx()
    kurtosis_axis = axis.twinx()
    kurtosis_axis.spines["right"].set_position(("axes", 1.12))
    lim_axis.plot(
        iterations[1:], np.nanmax(limits[1:], axis=1),
        label="Maximum limit", color="#355F9D", linewidth=1.8,
    )
    lim_axis.plot(
        iterations[1:], np.nanmean(limits[1:], axis=1),
        label="Mean limit", color="#7EA1CC", linewidth=1.5,
    )
    quality_axis.plot(
        iterations, logcosh.mean(axis=1),
        label="Mean Logcosh objective", color="#4F8A70", linewidth=1.5,
    )
    kurtosis_axis.plot(
        iterations, kurtosis.mean(axis=1),
        label="Mean excess kurtosis", color="#B45D4C", linewidth=1.5,
    )
    quality_axis.axhline(
        0.374567207491457, color="0.55", linestyle="--", linewidth=1.0,
        label="Gaussian",
    )
    kurtosis_axis.axhline(0.0, color="0.55", linestyle="--", linewidth=1.0)
    lim_axis.set_yscale("log")
    lim_axis.set_xlabel("FastICA iteration")
    lim_axis.set_ylabel("Direction change (limit)", color="#355F9D")
    quality_axis.set_ylabel("Mean Logcosh objective", color="#4F8A70")
    kurtosis_axis.set_ylabel("Mean excess kurtosis", color="#B45D4C")
    lim_axis.tick_params(axis="y", colors="#355F9D")
    quality_axis.tick_params(axis="y", colors="#4F8A70")
    kurtosis_axis.tick_params(axis="y", colors="#B45D4C")
    quality_axis.set_ylim(0.0, 0.4)
    kurtosis_axis.set_ylim(0.0, 90.0)
    lim_axis.grid(axis="y", color="0.88", linewidth=0.6)
    handles = []
    labels = []
    for current in (lim_axis, quality_axis, kurtosis_axis):
        current_handles, current_labels = current.get_legend_handles_labels()
        handles.extend(current_handles)
        labels.extend(current_labels)
    lim_axis.legend(handles, labels, frameon=False, loc="upper right")
    figure.suptitle(
        f"Qwen 3.5 9B Base · layers {', '.join(map(str, layers))} · {limits.shape[1]:,} components"
    )
    figure.subplots_adjust(left=0.11, right=0.79, bottom=0.15, top=0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
