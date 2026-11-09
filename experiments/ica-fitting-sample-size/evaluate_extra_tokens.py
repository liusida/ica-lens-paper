#!/usr/bin/env python3
"""Evaluate saved ICA states on all 1M newly captured, disjoint-document tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from icalens._activation_dataset import ActivationDataset
from icalens._fastica import GAUSSIAN_OBJECTIVES
from run import BUDGETS

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUT = WORK / "extra-evaluation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(white, mean, std, weights, batch_size):
    count = len(white)
    totals = torch.zeros((4, len(weights)), dtype=torch.float64, device="cuda")
    raw_contrast = torch.zeros_like(totals[0])
    standardized_contrast = torch.zeros_like(totals[0])
    for start in range(0, count, batch_size):
        projected = (weights @ white[start:start+batch_size].T).T
        centered = projected.double() - mean
        for order in range(1, 5):
            totals[order-1] += centered.pow(order).sum(0)
        absolute = projected.abs()
        raw_contrast += (absolute + F.softplus(-2*absolute) - math.log(2)).double().sum(0)
        standardized = ((projected - mean.float()) / std.float()).abs()
        standardized_contrast += (standardized + F.softplus(-2*standardized) - math.log(2)).double().sum(0)
    a, b, c, d = totals / count
    variance = b - a.square()
    m3 = c - 3*a*b + 2*a.pow(3)
    m4 = d - 4*a*c + 6*a.square()*b - 3*a.pow(4)
    assert bool((variance > 0).all())
    return {"component_mean": (mean+a).cpu().numpy(),
            "component_variance": variance.cpu().numpy(),
            "component_skewness": (m3 / variance.pow(1.5)).cpu().numpy(),
            "component_excess_kurtosis": (m4 / variance.square() - 3).cpu().numpy(),
            "component_logcosh_raw": (raw_contrast/count).cpu().numpy(),
            "component_logcosh_standardized": (standardized_contrast/count).cpu().numpy(),
            "component_logcosh_deviation": (standardized_contrast/count - GAUSSIAN_OBJECTIVES["logcosh"]).abs().cpu().numpy()}


def evaluate(label, data, dataset, batch_size):
    target = OUT / label
    target.mkdir(parents=True, exist_ok=True)
    source = WORK / "runs" / label
    shared = load_file(source / "preprocessing.safetensors", device="cuda")
    paths = [source / f"checkpoints/step-{step:03d}.safetensors" for step in range(51)]
    configuration = {
        "format": "icalens.extra_token_evaluation", "model": dataset.model,
        "fitting_tokens": BUDGETS[label], "evaluation_tokens": len(data), "steps": list(range(51)),
        "evaluation_manifest_sha256": dataset.provenance["activation_dataset"]["manifest_sha256"],
        "training_run_sha256": sha(source / "run.json"),
        "preprocessing_sha256": sha(source / "preprocessing.safetensors"),
        "checkpoint_sha256": {path.name: sha(path) for path in paths},
        "metric_population": "all 1M additional evaluation tokens at every saved state",
        "fitted_transform": "original training center, whitening, and unmixing; no refitting",
        "moment_centering": "evaluation component mean and variance for standardized central moments",
        "logcosh": "raw score contrast and evaluation-mean/variance-standardized contrast are both retained",
        "tf32": False, "fit_projection_dtype": "float32", "moment_accumulation_dtype": "float64"}
    config_path = target / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise RuntimeError(f"evaluation configuration changed: {target}")
    config_path.write_text(json.dumps(configuration, indent=2)+"\n")
    if (target / "complete.json").exists():
        print(label, "already complete", flush=True)
        return
    tic = time.perf_counter()
    white = torch.empty_like(data)
    for start in range(0, len(data), batch_size):
        white[start:start+batch_size] = (data[start:start+batch_size] - shared["center"]) @ shared["whitening"].T
    total = torch.zeros(white.shape[1], dtype=torch.float64, device="cuda")
    for start in range(0, len(white), batch_size):
        total += white[start:start+batch_size].double().sum(0)
    white_mean = total / len(white)
    covariance = torch.zeros((white.shape[1],)*2, dtype=torch.float64, device="cuda")
    for start in range(0, len(white), batch_size):
        centered = white[start:start+batch_size].double() - white_mean
        covariance.addmm_(centered.T, centered)
    covariance /= len(white)
    history = []
    for step, path in enumerate(paths):
        weights = load_file(path, device="cuda")["unmixing"]
        w64 = weights.double()
        mean = w64 @ white_mean
        variance = ((w64 @ covariance) * w64).sum(1)
        assert bool((variance > 0).all())
        record = moments(white, mean, variance.sqrt(), weights, batch_size)
        assert all(np.isfinite(value).all() for value in record.values())
        record["step"] = step
        record["elapsed_seconds"] = time.perf_counter()-tic
        history.append(record)
        package = {key: np.stack([row[key] for row in history]) for key in history[0]}
        temporary = target / "trajectory.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **package)
        temporary.replace(target / "trajectory.npz")
        if step % 5 == 0:
            print(f"{label} step {step}/50: median evaluation excess kurtosis={np.median(record['component_excess_kurtosis']):.5g}", flush=True)
    final = history[-1]
    (target / "complete.json").write_text(json.dumps({"fitting_tokens": BUDGETS[label],
        "evaluation_tokens": len(data), "states": len(history),
        "elapsed_seconds": time.perf_counter()-tic,
        "median_final_excess_kurtosis": float(np.median(final["component_excess_kurtosis"])),
        "median_final_logcosh_deviation": float(np.median(final["component_logcosh_deviation"]))}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", choices=BUDGETS, default=list(BUDGETS))
    parser.add_argument("--batch-size", type=int, default=16_384)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = ActivationDataset(WORK / "activations/gpt2-pile10k-extra-1m")
    assert dataset.sample_count == 1_000_000
    data = dataset.layer(11).float().cuda()
    for label in args.budgets:
        evaluate(label, data, dataset, args.batch_size)


if __name__ == "__main__":
    main()
