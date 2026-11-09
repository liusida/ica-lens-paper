#!/usr/bin/env python3
"""Audit experimental coverage, nested token subsets, and saved native lenses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from icalens import ICALens
from safetensors.torch import load_file

from run import BUDGETS, DEFAULT_ACTIVATIONS, component_moments
from icalens._activation_dataset import ActivationDataset


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here / "work/runs"
    dataset = ActivationDataset(DEFAULT_ACTIVATIONS)
    pool = dataset.layer(11)
    previous = None
    report = {}
    for label, count in BUDGETS.items():
        path = root / label
        complete = json.loads((path / "complete.json").read_text())
        assert complete["budget"] == complete["diagnostic_samples"] == count
        assert complete["steps_completed"] == 50
        with np.load(path / "trajectory.npz") as archive:
            trace = {name: archive[name].copy() for name in archive.files}
        assert np.array_equal(trace["steps"], np.arange(51))
        for name, value in trace.items():
            assert np.isfinite(value).all(), (label, name)
        assert trace["component_excess_kurtosis"].shape == (51, 768)
        assert np.allclose(trace["median_excess_kurtosis"], np.median(trace["component_excess_kurtosis"], axis=1))
        indices = np.load(path / "fit-row-indices.npy")
        assert len(indices) == len(np.unique(indices)) == count
        if previous is not None:
            assert np.array_equal(indices[:len(previous)], previous)
        previous = indices
        lens = ICALens.from_pretrained(path / "lens")
        assert lens.available_layers == (11,)
        # Replay every fitting row at the final state and compare to the saved
        # trace; this catches sample-count, checkpoint, and metric mismatches.
        final = load_file(path / "checkpoints/step-050.safetensors", device="cuda")
        preprocessing = load_file(path / "preprocessing.safetensors", device="cuda")
        skew, kurt, contrast = component_moments(pool[torch.from_numpy(indices)], final["unmixing"],
            preprocessing["whitening"], preprocessing["center"], 16_384)
        assert np.allclose(kurt, trace["component_excess_kurtosis"][-1], rtol=1e-6, atol=1e-6)
        assert np.allclose(contrast, trace["component_logcosh"][-1], rtol=1e-6, atol=1e-6)
        sample = pool[torch.from_numpy(indices[:128])].float()
        native = lens.transform(sample, layer=11).numpy()
        reading = (final["unmixing"] @ preprocessing["whitening"]).cpu()
        expected = ((sample - preprocessing["center"].cpu()) @ reading.T).numpy()
        assert np.allclose(np.sort(native, axis=1), np.sort(expected, axis=1), atol=2e-3, rtol=2e-3)
        report[label] = {"tokens": count, "steps": 50,
            "initial_median_excess_kurtosis": float(trace["median_excess_kurtosis"][0]),
            "final_median_excess_kurtosis": float(np.median(kurt)),
            "final_mean_excess_kurtosis": float(np.mean(kurt)),
            "final_median_absolute_skewness": float(np.median(np.abs(skew))),
            "elapsed_seconds": complete["elapsed_seconds"]}
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
