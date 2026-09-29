#!/usr/bin/env python3
"""Compute fitted-direction logcosh and kurtosis along the ICA trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from prepare import EVALUATED_CHECKPOINTS
from safetensors.torch import load_file
from select_cohort import checkpoint_path
from trajectory import parse_layers

from icalens._activation_dataset import ActivationDataset
from icalens.experiments._run import atomic_write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/fixed-panel"
GAUSSIAN_LOGCOSH = 0.374567207491457


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def disabled_positions(evaluation: Path, layers: tuple[int, ...]) -> set[int]:
    disabled: set[int] = set()
    for iteration in EVALUATED_CHECKPOINTS:
        for layer in layers:
            selection_path = (
                evaluation.parent
                / "prepared"
                / f"iter-{iteration:03d}"
                / f"layer_{layer:02d}/ica/selection.json"
            )
            if not selection_path.is_file():
                continue
            accepted = json.loads(selection_path.read_text())["accepted"]
            positions = {int(row["feature"]): i for i, row in enumerate(accepted)}
            result_dir = evaluation / f"iter-{iteration:03d}" / f"layer_{layer:02d}/ica/results"
            for path in result_dir.glob("feature_*.json") if result_dir.is_dir() else ():
                value = json.loads(path.read_text())
                if value.get("status") == "disabled":
                    feature = int(value["feature"])
                    disabled.add(positions[feature])
    return disabled


def selected_positions(capacity: int, disabled: set[int], count: int) -> list[int]:
    result = [position for position in range(capacity) if position not in disabled][:count]
    if len(result) != count:
        raise ValueError(f"only {len(result)} usable cohort positions, requested {count}")
    return result


def layer_statistics(
    *,
    dataset: ActivationDataset,
    trajectory: Path,
    layer: int,
    row_ids: list[int],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    shared = load_file(trajectory / "shared" / f"layer-{layer:02d}.safetensors")
    center = shared["center"].to(device=device, dtype=torch.float32)
    whitening = shared["whitening"].to(device=device, dtype=torch.float32)
    indices = torch.tensor(row_ids, device="cpu")
    directions = []
    for iteration in EVALUATED_CHECKPOINTS:
        unmixing = load_file(checkpoint_path(trajectory, layer, iteration))["unmixing"]
        directions.append(unmixing.index_select(0, indices).to(device) @ whitening)
    stacked = torch.cat(directions)
    width = stacked.shape[0]
    sums = torch.zeros((5, width), dtype=torch.float64, device=device)
    values = dataset.layer(layer)
    for start in range(0, dataset.sample_count, batch_size):
        batch = values[start : start + batch_size].to(device=device, dtype=torch.float32)
        scores = ((batch - center) @ stacked.T).to(torch.float64)
        sums[0] += scores.sum(0)
        sums[1] += scores.square().sum(0)
        sums[2] += scores.pow(3).sum(0)
        sums[3] += scores.pow(4).sum(0)
        sums[4] += (torch.logaddexp(scores, -scores) - math.log(2.0)).sum(0)
    mean, raw2, raw3, raw4, logcosh = sums / dataset.sample_count
    variance = (raw2 - mean.square()).clamp_min(0)
    central4 = raw4 - 4 * mean * raw3 + 6 * mean.square() * raw2 - 3 * mean.pow(4)
    excess_kurtosis = torch.where(
        variance > 0, central4 / variance.square() - 3, torch.zeros_like(variance)
    )
    logcosh_deviation = (logcosh - GAUSSIAN_LOGCOSH).abs()
    shape = (len(EVALUATED_CHECKPOINTS), len(row_ids))
    return {
        "layer": layer,
        "iterations": list(EVALUATED_CHECKPOINTS),
        "row_ids": row_ids,
        "logcosh_deviation": logcosh_deviation.reshape(shape).cpu().tolist(),
        "excess_kurtosis": excess_kurtosis.reshape(shape).cpu().tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.n_components < 1 or args.batch_size < 1:
        raise ValueError("component and batch counts must be positive")
    layers = parse_layers(args.layers)
    run = args.run.expanduser().resolve()
    trajectory = run / "trajectory"
    cohort_path = run / "cohort.json"
    evaluation = run / "evaluated-tinker"
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run / "fit-statistics.json"
    )
    summary = json.loads((trajectory / "summary.json").read_text())
    cohort = json.loads(cohort_path.read_text())
    capacity = int(cohort["n_components_per_layer"])
    disabled = disabled_positions(evaluation, layers)
    positions = selected_positions(capacity, disabled, args.n_components)
    resolved = {
        "format": "icalens.fitting-convergence-fit-statistics",
        "format_version": 1,
        "trajectory": str(trajectory),
        "trajectory_summary_sha256": sha256(trajectory / "summary.json"),
        "cohort": str(cohort_path),
        "cohort_sha256": sha256(cohort_path),
        "activation_dataset": summary["activations"],
        "layers": list(layers),
        "iterations": list(EVALUATED_CHECKPOINTS),
        "selected_cohort_positions_zero_based": positions,
        "disabled_cohort_positions_zero_based": sorted(disabled),
        "n_components_per_layer": args.n_components,
        "statistics": {
            "logcosh": (
                "absolute deviation of population E[log(cosh(score))] from Gaussian baseline"
            ),
            "gaussian_logcosh": GAUSSIAN_LOGCOSH,
            "kurtosis": "population excess kurtosis of signed component scores",
        },
    }
    previous: dict[str, Any] = {}
    if output.is_file():
        previous = json.loads(output.read_text())
        if previous.get("resolved") != resolved:
            raise ValueError(f"incompatible existing statistics artifact: {output}")
    dataset = ActivationDataset(summary["activations"])
    records = {str(row["layer"]): row for row in previous.get("layers", [])}
    for layer in layers:
        if str(layer) in records:
            print(f"PASS L{layer} already computed", flush=True)
            continue
        row_ids = [int(x) for x in cohort["layers"][str(layer)]["row_ids"]]
        selected = [row_ids[position] for position in positions]
        print(f"START L{layer} population statistics ({timestamp()})", flush=True)
        records[str(layer)] = layer_statistics(
            dataset=dataset,
            trajectory=trajectory,
            layer=layer,
            row_ids=selected,
            batch_size=args.batch_size,
            device=args.device,
        )
        atomic_write_json(
            output,
            {
                "status": "running",
                "resolved": resolved,
                "layers": [records[str(value)] for value in layers if str(value) in records],
            },
        )
        print(f"PASS L{layer} population statistics ({timestamp()})", flush=True)
    atomic_write_json(
        output,
        {
            "status": "complete",
            "resolved": resolved,
            "layers": [records[str(layer)] for layer in layers],
        },
    )
    print(f"PASS fit statistics: {output}")


if __name__ == "__main__":
    main()
