#!/usr/bin/env python3
"""Measure token energy concentration and evaluate unseen-token kurtosis."""
import json
from pathlib import Path

import numpy as np
import torch
from icalens import ICALens
from icalens._activation_dataset import ActivationDataset

HERE = Path(__file__).resolve().parent


def measure(data, indices, reading, center):
    total = torch.zeros(len(reading), dtype=torch.float64, device="cuda")
    for start in range(0, len(indices), 8192):
        x = data[indices[start:start+8192]].cuda().float()
        total += ((x - center) @ reading.T).double().sum(0)
    mean = total / len(indices)
    squares = torch.zeros_like(total)
    fourths = torch.zeros_like(total)
    peak = torch.zeros_like(total)
    for start in range(0, len(indices), 8192):
        x = data[indices[start:start+8192]].cuda().float()
        s = ((x - center) @ reading.T).double() - mean
        sq = s.square()
        squares += sq.sum(0)
        fourths += sq.square().sum(0)
        peak = torch.maximum(peak, sq.max(0).values)
    kurt = len(indices) * fourths / squares.square() - 3
    top1 = peak / squares
    top1_m4 = peak.square() / fourths
    return {"median_excess_kurtosis": float(torch.quantile(kurt, .5)),
            "median_top1_squared_energy_fraction": float(torch.quantile(top1, .5)),
            "median_top1_fourth_moment_fraction": float(torch.quantile(top1_m4, .5)),
            "tokens": len(indices)}


def main():
    root = HERE / "work/runs"
    dataset = ActivationDataset(HERE / "work/activations/gpt2-pile10k-layer11-1m")
    data = dataset.layer(11)
    # All fits through 100K use prefixes of this same permutation. The next
    # 100K are unseen token positions for every fit evaluated here.
    permutation = np.load(root / "1m/fit-row-indices.npy")
    heldout = torch.from_numpy(permutation[100_000:200_000].copy())
    report = {"heldout_definition": "100K token positions excluded from all fits through 100K; not document-disjoint", "settings": {}}
    for budget in ("1k", "5k", "10k", "100k"):
        indices = torch.from_numpy(np.load(root / budget / "fit-row-indices.npy"))
        assert not np.intersect1d(indices.numpy(), heldout.numpy()).size
        lens = ICALens.from_pretrained(root / budget / "lens")
        artifact = lens._get_layer(11)
        reading = torch.from_numpy(artifact.reading_matrix).cuda().float()
        center = torch.from_numpy(artifact.center).cuda().float()
        report["settings"][budget] = {"training": measure(data, indices, reading, center),
                                        "unseen_tokens": measure(data, heldout, reading, center)}
        print(budget, report["settings"][budget], flush=True)
    (root / "concentration-diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
