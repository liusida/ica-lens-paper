#!/usr/bin/env python3
"""Measure Figure 3's ICA statistic on held-out OpenWebText activations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens.cli.capture import main as capture

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "kurtosis-variance-comparison"
DATASET = "Skylion007/openwebtext"
REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
CACHE_ROOT = Path("/home/liusida/Expansion/research/ICA-data/heldout-kurtosis")
LOCAL_ARTIFACTS = HERE.parents[2] / "icalens_pypi"

spec = importlib.util.spec_from_file_location("figure3", REFERENCE / "run.py")
assert spec is not None and spec.loader is not None
figure3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure3)


@torch.inference_mode()
def excess_kurtosis(activations, reading_matrix, batch_size):
    directions = torch.nn.functional.normalize(
        torch.tensor(reading_matrix, device="cuda", dtype=torch.float32), dim=1
    )
    state = None
    for batch in activations.split(batch_size):
        scores = batch.to(device="cuda", dtype=torch.float32) @ directions.T
        state = figure3._merge_moments(state, scores)
    assert state is not None
    n, _, m2, _, m4 = state
    values = (n * m4 / m2.square() - 3).cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Undefined excess kurtosis in one or more directions")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=figure3.MODEL_SPECS, default=list(figure3.MODEL_SPECS)
    )
    parser.add_argument("--layers", default="all", help="all or comma-separated layer indices")
    parser.add_argument("--tokens", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--capture-layers-at-once", default="all")
    parser.add_argument("--output", type=Path, default=HERE / "runs/openwebtext")
    parser.add_argument("--activation-cache", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=LOCAL_ARTIFACTS)
    args = parser.parse_args()
    if args.tokens < 4 or args.batch_size < 1:
        parser.error("Use at least four tokens and a positive batch size")

    cache = args.activation_cache or CACHE_ROOT / args.output.name
    reference = args.artifact_root / "experiments/kurtosis-variance-comparison/runs/main"
    with (reference / "summary.csv").open(newline="") as handle:
        fit = {
            (row["model"], int(row["layer"])): float(row["ica_excess_kurtosis_mean"])
            for row in csv.DictReader(handle)
        }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "layer",
                "heldout_tokens",
                "fit_mean_excess_kurtosis",
                "heldout_mean_excess_kurtosis",
                "heldout_minus_fit",
            ]
        )
        for label in dict.fromkeys(args.models):
            config = figure3.MODEL_SPECS[label]
            lens_path = args.artifact_root / "local-icalens-models/official" / config["lens"].name
            lens = ICALens.from_pretrained(lens_path)
            training = ActivationDataset(config["activations"])
            layers = (
                lens.available_layers
                if args.layers == "all"
                else tuple(dict.fromkeys(map(int, args.layers.split(","))))
            )
            for layer in layers:
                if layer not in lens.available_layers or (label, layer) not in fit:
                    raise ValueError(f"No lens or Figure 3 reference for {label} layer {layer}")

            destination = args.output / label
            destination.mkdir(parents=True, exist_ok=True)
            activation_path = cache / label / "activations"
            capture(
                "text",
                [
                    "--model",
                    training.model["repo_id"],
                    "--model-revision",
                    training.model["revision"],
                    "--dataset",
                    DATASET,
                    "--dataset-revision",
                    REVISION,
                    "--layers",
                    ",".join(map(str, layers)),
                    "--context-length",
                    str(training.provenance["context_length"]),
                    "--document-framing",
                    training.provenance["document_framing"]["strategy"],
                    "--token-budget",
                    str(args.tokens),
                    "--candidate-tokens",
                    str(args.tokens),
                    "--capture-layers-at-once",
                    args.capture_layers_at_once,
                    "--output",
                    str(activation_path),
                ],
            )
            heldout = ActivationDataset(activation_path)
            for layer in layers:
                artifact = lens._get_layer(layer)
                values = excess_kurtosis(
                    heldout.layer(layer), artifact.reading_matrix, args.batch_size
                )
                np.save(destination / f"layer-{layer:02d}-excess-kurtosis.npy", values)
                baseline = fit[label, layer]
                mean = float(values.mean())
                writer.writerow(
                    [label, layer, heldout.sample_count, baseline, mean, mean - baseline]
                )
                handle.flush()
                print(f"{label} layer {layer}: fit={baseline:.4g}, held-out={mean:.4g}", flush=True)


if __name__ == "__main__":
    main()
