#!/usr/bin/env python3
"""Plot three real agreement regimes for kurtosis and autointerpretability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

from icalens._activation_dataset import ActivationDataset

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/fixed-panel"
RESULTS = ROOT / "results"
ITERATION = 50
ACTIVATIONS = Path(
    "/media/liusida/Expansion/research/ICA-data/icalens-activations/"
    "qwen3.5-9b-base-pile10k-1m"
)
TAILS = Path(
    "/media/liusida/Expansion/research/ICA-data/"
    "fitting-autointerpretability-convergence/fixed-panel/tails.json"
)
SIGNED_EVAL_SAMPLE = RESULTS / "signed-eval-token-scores-100k.npy"


@dataclass(frozen=True)
class Case:
    layer: int
    component: int
    label: str


CASES = (
    Case(7, 338, "(a) Expected agreement"),
    Case(7, 2035, "(b) Tail-coverage error"),
    Case(15, 1503, "(c) Dataset-shortcut error"),
)


def full_population_statistics(case: Case) -> tuple[float, float]:
    fit = json.loads((RUN / "fit-statistics.json").read_text(encoding="utf-8"))
    layer = next(row for row in fit["layers"] if int(row["layer"]) == case.layer)
    component_position = list(map(int, layer["row_ids"])).index(case.component)
    iteration_position = list(map(int, layer["iterations"])).index(ITERATION)
    kurtosis = float(layer["excess_kurtosis"][iteration_position][component_position])

    evaluation = json.loads(
        (
            RUN
            / "evaluated-tinker"
            / f"iter-{ITERATION:03d}"
            / f"layer_{case.layer:02d}"
            / "ica/results"
            / f"feature_{case.component}.json"
        ).read_text(encoding="utf-8")
    )
    return kurtosis, float(evaluation["combined_score"])


def gaussian_density(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values * values) / np.sqrt(2.0 * np.pi)


def fitting_distribution(case: Case) -> tuple[np.ndarray, float, float]:
    """Return signed scores for the complete one-million-token fitting population."""
    path = RESULTS / f"c{case.component}-fitting-1m-distribution-data.npz"
    if path.is_file():
        with np.load(path) as data:
            scores = np.asarray(data["signed_scores"], dtype=np.float64)
            return scores, float(scores.mean()), float(scores.std())

    shared = load_file(RUN / "trajectory/shared" / f"layer-{case.layer:02d}.safetensors")
    checkpoint = load_file(
        RUN / "trajectory/checkpoints" / f"layer-{case.layer:02d}"
        / f"iter-{ITERATION:03d}.safetensors"
    )
    center = shared["center"].to(dtype=torch.float32)
    whitening = shared["whitening"].to(dtype=torch.float32)
    reading = checkpoint["unmixing"][case.component].to(dtype=torch.float32) @ whitening
    cohort = json.loads((RUN / "cohort.json").read_text(encoding="utf-8"))
    row_ids = list(map(int, cohort["layers"][str(case.layer)]["row_ids"]))
    position = row_ids.index(case.component)
    tails = json.loads(TAILS.read_text(encoding="utf-8"))
    reading *= int(tails["layers"][str(case.layer)]["tail_sign"][position])

    dataset = ActivationDataset(ACTIVATIONS)
    raw = dataset.layer(case.layer)
    scores = np.empty(dataset.sample_count, dtype=np.float32)
    for start in range(0, dataset.sample_count, 4096):
        stop = min(start + 4096, dataset.sample_count)
        activations = raw[start:stop].to(dtype=torch.float32)
        scores[start:stop] = ((activations - center) @ reading).numpy()
    mean = float(scores.mean(dtype=np.float64))
    std = float(scores.std(dtype=np.float64))
    np.savez_compressed(
        path,
        signed_scores=scores,
        mean=mean,
        std=std,
    )
    return scores.astype(np.float64), mean, std


def evaluation_distribution(case: Case) -> tuple[np.ndarray, float, float]:
    """Return the systematic 100,032-token OpenWebText evaluation sample."""
    values = np.load(SIGNED_EVAL_SAMPLE, mmap_mode="r")
    scores = np.asarray(values[:, :, CASES.index(case)], dtype=np.float32).reshape(-1)
    scores = scores.astype(np.float64)
    return scores, float(scores.mean()), float(scores.std())


def weakest_selected_fragment(case: Case, mean: float, std: float) -> float:
    """Return the weakest selected evaluation-fragment maximum on fitting z scale."""
    layer_dir = (
        RUN
        / "prepared"
        / f"iter-{ITERATION:03d}"
        / f"layer_{case.layer:02d}"
        / "ica"
    )
    selection = json.loads((layer_dir / "selection.json").read_text(encoding="utf-8"))
    record = next(
        row for row in selection["accepted"] if int(row["feature"]) == case.component
    )
    values = np.load(layer_dir / "candidate_activations.npy", mmap_mode="r")
    position = int(record["candidate_position"])
    selected_indices = np.asarray(
        [*map(int, record["train_top"]), *map(int, record["valid_top"])], dtype=np.int64
    )
    selected_maxima = np.asarray(
        values[selected_indices, :, position], dtype=np.float32
    ).max(axis=1)
    return (float(selected_maxima.min()) - mean) / std


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    color = "#B66A55"

    for axis, case in zip(axes, CASES, strict=True):
        raw_scores, mean, std = fitting_distribution(case)
        values = (raw_scores - mean) / std
        threshold = weakest_selected_fragment(case, mean, std)
        kurtosis, score = full_population_statistics(case)
        left = float(np.floor(values.min()))
        right = float(np.ceil(max(float(values.max()), threshold) + 0.25))
        edges = np.linspace(left, right, 76)
        axis.hist(
            values,
            bins=edges,
            density=True,
            color=color,
            alpha=0.55,
            edgecolor=color,
            linewidth=0.35,
            label="Fitting tokens",
        )
        grid = np.linspace(left, right, 1200)
        axis.plot(
            grid,
            gaussian_density(grid),
            color="#4C4C4C",
            linestyle="--",
            linewidth=1.15,
            label="Gaussian",
        )
        axis.axvline(
            threshold,
            color="#8F3F32",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            label="Selection cutoff",
        )
        axis.set_yscale("log")
        axis.set_xlim(left, right)
        axis.set_ylim(1e-6, 1.0)
        axis.set_xlabel("Standardized signed token activation")
        axis.set_ylabel("Density")
        axis.legend(frameon=False, fontsize=8, loc="upper right")
        axis.set_title(case.label, y=-0.36, fontsize=11)
        axis.text(
            0.03,
            0.04,
            rf"$\kappa={kurtosis:.2f}$, $r={score:.2f}$",
            transform=axis.transAxes,
            fontsize=8.5,
        )

    fig.subplots_adjust(left=0.065, right=0.995, top=0.98, bottom=0.30, wspace=0.30)
    output = RESULTS / "autointerpretability-failure-modes"
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
