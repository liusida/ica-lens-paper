#!/usr/bin/env python3
"""Plot an iteration-50 layer-7 component distribution from cached activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

from icalens._activation_dataset import ActivationDataset

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/fixed-panel"
TRAJECTORY = RUN / "trajectory"
OUTPUT = ROOT / "results"
ACTIVATIONS = Path(
    "/media/liusida/Expansion/research/ICA-data/icalens-activations/"
    "qwen3.5-9b-base-pile10k-1m"
)
TAILS = Path(
    "/media/liusida/Expansion/research/ICA-data/"
    "fitting-autointerpretability-convergence/fixed-panel/tails.json"
)
LAYER, ITERATION = 7, 50
N_SAMPLES = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=int, default=2035)
    return parser.parse_args()


def gaussian_density(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def main() -> None:
    args = parse_args()
    component = int(args.component)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shared = load_file(TRAJECTORY / "shared/layer-07.safetensors")
    checkpoint = load_file(
        TRAJECTORY / "checkpoints/layer-07/iter-050.safetensors"
    )
    center = shared["center"].to(dtype=torch.float32)
    whitening = shared["whitening"].to(dtype=torch.float32)
    reading = checkpoint["unmixing"][component].to(dtype=torch.float32) @ whitening

    cohort = json.loads((RUN / "cohort.json").read_text(encoding="utf-8"))
    row_ids = list(map(int, cohort["layers"][str(LAYER)]["row_ids"]))
    cohort_position = row_ids.index(component)
    tails = json.loads(TAILS.read_text(encoding="utf-8"))
    tail_sign = int(tails["layers"][str(LAYER)]["tail_sign"][cohort_position])
    reading *= tail_sign

    dataset = ActivationDataset(ACTIVATIONS)
    if dataset.sample_count < N_SAMPLES:
        raise ValueError(f"capture has only {dataset.sample_count} samples")
    # The capture is already a deterministic token sample. A systematic stride
    # covers its full extent without loading the 8 GB layer into memory.
    step = dataset.sample_count / N_SAMPLES
    indices = np.floor(np.arange(N_SAMPLES, dtype=np.float64) * step).astype(np.int64)
    raw = dataset.layer(LAYER)
    scores = np.empty(N_SAMPLES, dtype=np.float32)
    batch = 4096
    for start in range(0, N_SAMPLES, batch):
        stop = min(start + batch, N_SAMPLES)
        selected = raw.index_select(0, torch.from_numpy(indices[start:stop]))
        values = selected.to(dtype=torch.float32)
        scores[start:stop] = ((values - center) @ reading).numpy()

    mean = float(scores.mean(dtype=np.float64))
    std = float(scores.std(dtype=np.float64))
    z = (scores.astype(np.float64) - mean) / std
    excess_kurtosis = float(np.mean(z**4) - 3.0)
    skewness = float(np.mean(z**3))
    quantiles = {
        f"p{100*q:g}": float(np.quantile(z, q))
        for q in (0, 0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999, 1)
    }
    statistics = {
        "model": "Qwen/Qwen3.5-9B-Base",
        "layer": LAYER,
        "component": component,
        "iteration": ITERATION,
        "n_samples": N_SAMPLES,
        "sampling": "systematic indices spanning the one-million-token capture",
        "tail_sign": tail_sign,
        "raw_mean": mean,
        "raw_std": std,
        "sample_skewness": skewness,
        "sample_excess_kurtosis": excess_kurtosis,
        "standardized_quantiles": quantiles,
        "fraction_abs_z_gt_3": float(np.mean(np.abs(z) > 3)),
        "fraction_abs_z_gt_5": float(np.mean(np.abs(z) > 5)),
        "fraction_abs_z_gt_10": float(np.mean(np.abs(z) > 10)),
    }
    stem = f"c{component}"
    (OUTPUT / f"{stem}-distribution-statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT / f"{stem}-distribution-data.npz",
        sample_indices=indices,
        signed_scores=scores,
        standardized_scores=z.astype(np.float32),
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))
    color = "#9A5A4A"
    grid = np.linspace(-5, 5, 500)
    axes[0].hist(z, bins=np.linspace(-5, 5, 121), density=True,
                 color=color, alpha=0.68, edgecolor="none", label=f"C{component}")
    axes[0].plot(grid, gaussian_density(grid), color="#555555", linestyle="--",
                 linewidth=1.25, label="Gaussian")
    axes[0].set(xlim=(-5, 5), xlabel="Standardized signed score", ylabel="Density",
                title="A · Central distribution")
    axes[0].legend(frameon=False, fontsize=8)

    full_left = float(np.floor(z.min()))
    full_right = float(np.ceil(z.max()))
    full_edges = np.linspace(full_left, full_right, 241)
    axes[1].hist(z, bins=full_edges, density=True, color=color, alpha=0.55,
                 edgecolor="none", label=f"C{component}")
    full_grid = np.linspace(full_left, full_right, 1200)
    axes[1].plot(full_grid, gaussian_density(full_grid), color="#555555",
                 linestyle="--", linewidth=1.25, label="Gaussian")
    axes[1].set_yscale("log")
    axes[1].set(xlim=(full_left, full_right), xlabel="Standardized signed score",
                ylabel="Density", title="B · Log-density histogram")
    axes[1].set_ylim(1e-6, 1)
    axes[1].legend(frameon=False, fontsize=8)

    absolute = np.abs(z)
    positive = absolute[absolute > 0]
    upper = max(10.0, float(positive.max()) * 1.03)
    bins = np.geomspace(max(1e-2, float(positive.min())), upper, 90)
    counts, edges = np.histogram(positive, bins=bins)
    probability = counts / counts.sum()
    centers = np.sqrt(edges[:-1] * edges[1:])
    keep = counts > 0
    axes[2].stairs(probability, edges, color=color, linewidth=1.5, fill=True,
                   alpha=0.36, label=f"C{component}")
    # Expected bin masses for |N(0,1)|, evaluated numerically at bin centers.
    gaussian_mass = 2 * gaussian_density(centers) * np.diff(edges)
    axes[2].plot(centers, gaussian_mass, color="#555555", linestyle="--",
                 linewidth=1.25, label="Gaussian")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Absolute standardized score", ylabel="Probability per bin",
                title="C · Tail histogram")
    axes[2].set_xlim(max(1e-2, float(positive.min())), upper)
    axes[2].set_ylim(max(5e-6, probability[keep].min() / 2), 1)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Qwen iteration-50 · L7/C{component} · n={N_SAMPLES:,} · "
        rf"excess kurtosis $\kappa={excess_kurtosis:.1f}$",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}-score-distribution.{suffix}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(statistics, indent=2))
    print(OUTPUT / f"{stem}-score-distribution.png")


if __name__ == "__main__":
    main()
