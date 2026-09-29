#!/usr/bin/env python3
"""Show one persistent component's explanations across fitting checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from prepare import EVALUATED_CHECKPOINTS, LAYERS
from safetensors.torch import load_file
from select_cohort import checkpoint_path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "runs/evaluated-tinker"
DEFAULT_PREPARATION = ROOT / "runs/prepared"
DEFAULT_TRAJECTORY = ROOT / "runs/trajectory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True, choices=LAYERS)
    parser.add_argument("--component", type=int, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--show-scores", action="store_true")
    return parser.parse_args()


def cosine_to_final(trajectory: Path, layer: int, component: int) -> dict[int, float]:
    whitening = load_file(
        trajectory / "shared" / f"layer-{layer:02d}.safetensors"
    )["whitening"]
    directions = {}
    for iteration in EVALUATED_CHECKPOINTS:
        unmixing = load_file(checkpoint_path(trajectory, layer, iteration))["unmixing"]
        if not 0 <= component < unmixing.shape[0]:
            raise ValueError(
                f"component C{component} is outside checkpoint width {unmixing.shape[0]}"
            )
        reading = unmixing[component : component + 1] @ whitening
        directions[iteration] = torch.nn.functional.normalize(reading, dim=1)
    final = directions[EVALUATED_CHECKPOINTS[-1]]
    return {
        iteration: float((direction * final).sum())
        for iteration, direction in directions.items()
    }


def main() -> None:
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    preparation = args.preparation.expanduser().resolve()
    trajectory = args.trajectory.expanduser().resolve()
    cosines = cosine_to_final(trajectory, args.layer, args.component)
    found = 0
    print(f"Layer {args.layer}, component C{args.component}")
    print(f"Evaluation: {input_root}")
    for iteration in EVALUATED_CHECKPOINTS:
        selection_path = (
            preparation
            / f"iter-{iteration:03d}"
            / f"layer_{args.layer:02d}"
            / "ica"
            / "selection.json"
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = {int(row["feature"]) for row in selection["accepted"]}
        if args.component not in selected:
            raise ValueError(
                f"C{args.component} is not in the fixed layer-{args.layer} cohort"
            )
        result_path = (
            input_root
            / f"iter-{iteration:03d}"
            / f"layer_{args.layer:02d}"
            / "ica"
            / "results"
            / f"feature_{args.component}.json"
        )
        if not result_path.is_file():
            print(
                f"\nIteration {iteration:>3} | "
                f"cosine to iter-200={cosines[iteration]:.4f} | pending"
            )
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            print(
                f"\nIteration {iteration:>3} | "
                f"cosine to iter-200={cosines[iteration]:.4f} | "
                f"{result.get('status', 'incomplete')}"
            )
            continue
        found += 1
        heading = (
            f"\nIteration {iteration:>3} | "
            f"cosine to iter-200={cosines[iteration]:.4f}"
        )
        if args.show_scores:
            heading += (
                f" | combined={float(result['combined_score']):.4f}"
                f" | top={float(result['top_score']):.4f}"
                f" | random={float(result['random_score']):.4f}"
            )
        print(heading)
        print(str(result["explanation"]))
    print(f"\nCompleted checkpoints: {found}/{len(EVALUATED_CHECKPOINTS)}")


if __name__ == "__main__":
    main()
