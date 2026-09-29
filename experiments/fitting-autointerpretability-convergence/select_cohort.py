#!/usr/bin/env python3
"""Select persistent FastICA rows for longitudinal autointerpretability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from trajectory import parse_layers

from icalens.experiments._run import atomic_write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = ROOT / "runs/trajectory"
DEFAULT_OUTPUT = ROOT / "runs/cohort.json"
LAYERS = (15, 31)
WIDTH = 4096
CHECKPOINTS = (0, *range(1, 11), *range(20, 201, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_rows(*, layer: int, count: int) -> list[int]:
    if not 0 < count <= WIDTH:
        raise ValueError(f"--n-components must be between 1 and {WIDTH}")
    generator = np.random.default_rng(layer)
    # A permutation has a stable prefix: increasing count preserves every
    # previously selected row and its cohort position.
    return generator.permutation(WIDTH)[:count].tolist()


def checkpoint_path(trajectory: Path, layer: int, iteration: int) -> Path:
    return trajectory / "checkpoints" / f"layer-{layer:02d}" / f"iter-{iteration:03d}.safetensors"


def validate_trajectory(trajectory: Path, layers: tuple[int, ...] | None = None) -> dict:
    summary_path = trajectory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"trajectory summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "format": "icalens.fastica_trajectory",
        "format_version": 1,
        "checkpoints": list(CHECKPOINTS),
        "component_order": "persistent optimization row; no per-checkpoint reordering",
    }
    mismatches = [
        f"{key}: {summary.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise ValueError("incompatible trajectory: " + "; ".join(mismatches))
    if layers is not None and summary.get("layers") != list(layers):
        raise ValueError(
            f"incompatible trajectory layers: {summary.get('layers')!r} != {list(layers)!r}"
        )
    return summary


def normalized_directions(
    trajectory: Path,
    *,
    layer: int,
    iteration: int,
    rows: list[int],
    whitening: torch.Tensor,
) -> torch.Tensor:
    checkpoint = load_file(checkpoint_path(trajectory, layer, iteration))["unmixing"]
    selected = checkpoint.index_select(0, torch.tensor(rows)).to(whitening.device)
    directions = selected @ whitening
    return torch.nn.functional.normalize(directions, dim=1)


def continuity(trajectory: Path, *, layer: int, rows: list[int], device: str) -> dict:
    whitening = load_file(trajectory / "shared" / f"layer-{layer:02d}.safetensors")[
        "whitening"
    ].to(device)
    previous = normalized_directions(
        trajectory,
        layer=layer,
        iteration=CHECKPOINTS[0],
        rows=rows,
        whitening=whitening,
    )
    records = []
    for iteration in CHECKPOINTS[1:]:
        current = normalized_directions(
            trajectory,
            layer=layer,
            iteration=iteration,
            rows=rows,
            whitening=whitening,
        )
        signed_cosine = (current * previous).sum(dim=1)
        signs = torch.where(signed_cosine < 0, -1.0, 1.0)
        current = current * signs[:, None]
        records.append(
            {
                "iteration": iteration,
                "previous_iteration": records[-1]["iteration"] if records else CHECKPOINTS[0],
                "signed_cosine_before_alignment": signed_cosine.tolist(),
                "sign_from_previous": signs.to(torch.int8).tolist(),
                "absolute_cosine": signed_cosine.abs().tolist(),
            }
        )
        previous = current
    return {
        "alignment": "sequential sign alignment by persistent optimization row",
        "records": records,
    }


def main() -> None:
    args = parse_args()
    layers_requested = parse_layers(args.layers)
    trajectory = args.trajectory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = validate_trajectory(trajectory, layers_requested)
    layers = {
        str(layer): {"row_ids": selected_rows(layer=layer, count=args.n_components)}
        for layer in layers_requested
    }
    resolved = {
        "format": "icalens.fastica_autointerpretability_cohort",
        "format_version": 1,
        "trajectory": str(trajectory),
        "trajectory_activation_manifest_sha256": summary["activation_manifest_sha256"],
        "selection": "prefix of a uniform permutation of persistent optimization rows",
        "layer_seed_rule": "seed = layer index",
        "n_components_per_layer": args.n_components,
        "width": WIDTH,
        "layers": layers,
        "tail_orientation": {
            "status": "pending",
            "policy": "iteration-200 profiled tail propagated through sequentially aligned rows",
        },
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    for layer in layers_requested:
        rows = layers[str(layer)]["row_ids"]
        layers[str(layer)]["continuity"] = continuity(
            trajectory, layer=layer, rows=rows, device=args.device
        )
    atomic_write_json(output, resolved)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
