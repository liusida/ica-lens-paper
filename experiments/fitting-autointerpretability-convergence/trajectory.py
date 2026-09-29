#!/usr/bin/env python3
"""Fit continuous Qwen FastICA trajectories and checkpoint selected iterations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from icalens._activation_dataset import ActivationDataset
from icalens._fastica import FastICAResult, fit_fastica
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parent
DEFAULT_ACTIVATIONS = Path(
    "/home/liusida/Expansion/research/ICA-data/icalens-activations/"
    "qwen3.5-9b-base-pile10k-1m"
)
DEFAULT_OUTPUT = ROOT / "runs/trajectory"
DEFAULT_LAYERS = (15, 31)
CHECKPOINTS = (0, *range(1, 11), *range(20, 201, 10))
EXPECTED_MODEL = "Qwen/Qwen3.5-9B-Base"
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_SAMPLE_COUNT = 1_000_000
FIT_BATCH_SIZE = 16_384
SEED = 0
MAX_ITER = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-batch-size", type=int, default=FIT_BATCH_SIZE)
    return parser.parse_args()


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--layers must be comma-separated integers") from error
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique layer indices")
    return layers


def validate_dataset(dataset: ActivationDataset, layers: tuple[int, ...]) -> None:
    expected = {
        "model": (dataset.model.get("repo_id"), EXPECTED_MODEL),
        "hidden_size": (dataset.hidden_size, EXPECTED_HIDDEN_SIZE),
        "sample_count": (dataset.sample_count, EXPECTED_SAMPLE_COUNT),
        "dtype": (dataset.dtype, torch.bfloat16),
        "activation_site": (dataset.manifest.get("activation_site"), "resid_post"),
    }
    mismatches = [
        f"{key}: {actual!r} != {wanted!r}"
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    missing = sorted(set(layers) - set(dataset.available_layers))
    if missing:
        mismatches.append(f"missing layers: {missing}")
    if mismatches:
        raise ValueError("incompatible activation dataset: " + "; ".join(mismatches))


def checkpoint_path(output: Path, layer: int, iteration: int) -> Path:
    return output / "checkpoints" / f"layer-{layer:02d}" / f"iter-{iteration:03d}.safetensors"


def shared_path(output: Path, layer: int) -> Path:
    return output / "shared" / f"layer-{layer:02d}.safetensors"


def atomic_save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, temporary)
    os.replace(temporary, path)


def load_latest_checkpoint(output: Path, layer: int) -> tuple[int, torch.Tensor] | None:
    for iteration in reversed(CHECKPOINTS):
        path = checkpoint_path(output, layer, iteration)
        if not path.is_file():
            continue
        values = load_file(path)
        unmixing = values.get("unmixing")
        if unmixing is None or tuple(unmixing.shape) != (
            EXPECTED_HIDDEN_SIZE,
            EXPECTED_HIDDEN_SIZE,
        ):
            raise ValueError(f"invalid trajectory checkpoint: {path}")
        return iteration, unmixing
    return None


def result_summary(result: FastICAResult, resumed_from: int) -> dict[str, Any]:
    return {
        "resumed_from_iteration": resumed_from,
        "n_iter": result.n_iter,
        "objective_iterations_in_this_process": result.objective_iterations,
        "objective_percentiles_in_this_process": result.objective_history,
        "gaussian_objective": result.gaussian_objective,
    }


def fit_layer(
    dataset: ActivationDataset,
    output: Path,
    *,
    layer: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    latest = load_latest_checkpoint(output, layer)
    if latest is not None and latest[0] == MAX_ITER:
        return {
            "layer": layer,
            "status": "complete",
            "last_checkpoint": MAX_ITER,
            "reused": True,
        }
    resumed_from, initial = latest if latest is not None else (0, None)

    def checkpoint(
        iteration: int,
        unmixing: torch.Tensor,
        whitening: torch.Tensor,
        center: torch.Tensor,
    ) -> None:
        shared = shared_path(output, layer)
        if not shared.is_file():
            atomic_save_tensors(shared, {"center": center, "whitening": whitening})
        if iteration in CHECKPOINTS:
            atomic_save_tensors(
                checkpoint_path(output, layer, iteration),
                {"unmixing": unmixing},
            )
            log(f"Checkpointed layer {layer} at iteration {iteration}.")

    result = fit_fastica(
        dataset.layer(layer),
        n_components=dataset.hidden_size,
        algorithm="parallel",
        fun="logcosh",
        max_iter=MAX_ITER,
        random_state=SEED,
        progress=True,
        device=device,
        batch_size=batch_size,
        row_normalize=False,
        objective_every=1,
        checkpoint_callback=checkpoint,
        initial_unmixing=initial,
        start_iteration=resumed_from,
    )
    summary = {
        "layer": layer,
        "status": "complete",
        "last_checkpoint": MAX_ITER,
        "reused": False,
        **result_summary(result, resumed_from),
    }
    atomic_write_json(output / "layers" / f"layer-{layer:02d}.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("trajectory fitting requires CUDA")
    if args.fit_batch_size <= 0:
        raise ValueError("--fit-batch-size must be positive")
    layers = parse_layers(args.layers)
    activations = args.activations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    dataset = ActivationDataset(activations)
    validate_dataset(dataset, layers)
    resolved = {
        "format": "icalens.fastica_trajectory",
        "format_version": 1,
        "activations": str(activations),
        "activation_manifest_sha256": dataset.provenance["activation_dataset"][
            "manifest_sha256"
        ],
        "model": dataset.model,
        "layers": list(layers),
        "checkpoints": list(CHECKPOINTS),
        "max_iter": MAX_ITER,
        "seed": SEED,
        "algorithm": "parallel",
        "fun": "logcosh",
        "whiten": "unit-variance",
        "icalens_preprocessing": "none",
        "fit_batch_size": args.fit_batch_size,
        "stored_dtype": "float32",
        "component_order": "persistent optimization row; no per-checkpoint reordering",
    }
    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="fitting")
    warn_if_dirty(source)
    completed = {
        layer
        for layer in layers
        if checkpoint_path(output, layer, MAX_ITER).is_file()
    }
    summaries = []
    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · fitting–autointerpretability convergence",
        completed=len(completed),
        total=len(layers),
        completed_unit_ids=completed,
        unit_label="layer trajectories",
        recent_label="Recent fitting output",
        detail_filename="trajectory-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        for layer in layers:
            latest = load_latest_checkpoint(output, layer)
            display.phase(
                "Fitting continuous FastICA trajectory",
                layer=layer,
                iteration=(latest[0] if latest is not None else 0),
            )
            log(f"Starting layer {layer} trajectory.")
            summary = fit_layer(
                dataset,
                output,
                layer=layer,
                device=args.device,
                batch_size=args.fit_batch_size,
            )
            summaries.append(summary)
            display.complete_unit(layer, refresh=True)
            log(f"Completed layer {layer} trajectory.")
        display.phase("Writing trajectory summary")
    atomic_write_json(output / "summary.json", {**resolved, "layer_results": summaries})
    run.set_status("complete", complete=True)
    log(f"Trajectory complete: {output}")


if __name__ == "__main__":
    main()
