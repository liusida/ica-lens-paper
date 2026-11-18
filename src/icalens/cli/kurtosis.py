"""Score a fitted ICA Lens's component excess kurtosis on captured activations."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens.kurtosis import lens_excess_kurtosis

from ._status import log


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens kurtosis", description=__doc__)
    parser.add_argument("--lens", required=True, help="Local artifact directory or Hub repo ID.")
    parser.add_argument(
        "--activations", type=Path, required=True, help="Captured activation directory."
    )
    parser.add_argument("--layers", default="all", help="'all' or comma-separated indices.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=None, help="Directory for per-layer .npy and summary.csv."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    lens = ICALens.from_pretrained(args.lens)
    dataset = ActivationDataset(args.activations)
    if dataset.manifest["activation_site"] != lens.activation_site:
        raise ValueError(
            f"activation site mismatch: lens {lens.activation_site!r}, "
            f"activations {dataset.manifest['activation_site']!r}"
        )
    if args.layers.strip().lower() == "all":
        layers = tuple(
            layer for layer in lens.available_layers if layer in dataset.available_layers
        )
    else:
        layers = tuple(sorted({int(item) for item in args.layers.split(",")}))
    missing = [
        layer
        for layer in layers
        if layer not in lens.available_layers or layer not in dataset.available_layers
    ]
    if not layers or missing:
        raise ValueError(f"layers unavailable in lens or activations: {missing or 'none selected'}")
    rows = []
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
    for layer in tqdm(layers, desc="Excess kurtosis", unit="layer", dynamic_ncols=True):
        values = lens_excess_kurtosis(
            lens,
            dataset.layer(layer),
            layer=layer,
            batch_size=args.batch_size,
            device=args.device,
        )
        mean = float(np.mean(values))
        rows.append((layer, dataset.sample_count, mean, float(np.median(values))))
        log(f"Layer {layer}: mean excess kurtosis {mean:.6g} over {dataset.sample_count} tokens")
        if args.output is not None:
            np.save(args.output / f"layer-{layer:02d}-excess-kurtosis.npy", values)
    if args.output is not None:
        with (args.output / "summary.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "tokens", "mean_excess_kurtosis", "median_excess_kurtosis"])
            writer.writerows(rows)
        log(f"Saved kurtosis summary to {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
