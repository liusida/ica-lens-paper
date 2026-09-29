#!/usr/bin/env python3
"""Repeat the Qwen FastICA trajectory while recording population diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from icalens._activation_dataset import ActivationDataset
from icalens._fastica import (
    GAUSSIAN_OBJECTIVES,
    _contrast,
    _covariance,
    _mean,
    _nonlinearity,
    _symmetric_decorrelation,
    _whitened_batches,
    _whitening_matrix,
)
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parent
DEFAULT_ACTIVATIONS = Path(
    "/home/liusida/Expansion/research/ICA-data/icalens-activations/"
    "qwen3.5-9b-base-pile10k-1m"
)
DEFAULT_OUTPUT = ROOT / "runs/main"
DEFAULT_LAYERS = (7, 15, 23, 31)
EXPECTED_MODEL = "Qwen/Qwen3.5-9B-Base"
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_SAMPLE_COUNT = 1_000_000
FIT_BATCH_SIZE = 16_384
MAX_ITER = 200
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-batch-size", type=int, default=FIT_BATCH_SIZE)
    return parser.parse_args()


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("--layers must contain unique comma-separated indices")
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
        f"{name}: {actual!r} != {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    missing = sorted(set(layers) - set(dataset.available_layers))
    if missing:
        mismatches.append(f"missing layers: {missing}")
    if mismatches:
        raise ValueError("incompatible activation dataset: " + "; ".join(mismatches))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_path(output: Path, layer: int) -> Path:
    return output / "layers" / f"layer-{layer:02d}.npz"


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary) as stored:
        if set(stored.files) != set(arrays):
            raise ValueError(f"failed to validate temporary result: {temporary}")
    os.replace(temporary, path)


def population_statistics(
    source: torch.Tensor,
    weights: torch.Tensor,
    *,
    batch_kwargs: dict[str, object],
    objective: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(source.shape[0])
    raw2 = torch.zeros(weights.shape[0], dtype=torch.float64, device=weights.device)
    raw4 = torch.zeros_like(raw2)
    objective_sum = torch.zeros_like(raw2)
    for whitened in _whitened_batches(source, **batch_kwargs):
        projected = weights @ whitened
        squared = projected.square()
        raw2 += squared.sum(dim=1, dtype=torch.float64)
        raw4 += squared.square().sum(dim=1, dtype=torch.float64)
        objective_sum += objective(projected).to(torch.float64) * int(projected.shape[1])
    variance = (raw2 / count).clamp_min(torch.finfo(torch.float64).eps)
    excess_kurtosis = raw4 / count / variance.square() - 3.0
    logcosh_objective = objective_sum / count
    logcosh_contrast = (logcosh_objective - GAUSSIAN_OBJECTIVES["logcosh"]).abs()
    return logcosh_objective, logcosh_contrast, excess_kurtosis


def fit_layer(
    dataset: ActivationDataset,
    *,
    layer: int,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    source = dataset.layer(layer)
    fit_device = torch.device(device)
    dtype = torch.float32
    center = _mean(
        source, device=fit_device, dtype=dtype, batch_size=batch_size,
        row_normalize=False, preprocessing_center=None, norm_eps=1e-12, progress=True,
    )
    covariance = _covariance(
        source, center=center, device=fit_device, dtype=dtype, batch_size=batch_size,
        row_normalize=False, preprocessing_center=None, norm_eps=1e-12, progress=True,
    )
    whitening = _whitening_matrix(
        covariance,
        n_samples=int(source.shape[0]),
        n_components=dataset.hidden_size,
        dtype=dtype,
    )
    generator = torch.Generator(device=fit_device).manual_seed(SEED)
    initial = torch.randn(
        (dataset.hidden_size, dataset.hidden_size),
        dtype=dtype,
        device=fit_device,
        generator=generator,
    )
    weights = _symmetric_decorrelation(initial)
    nonlinearity = _nonlinearity("logcosh")
    objective = _contrast("logcosh")
    batch_kwargs: dict[str, object] = {
        "center": center,
        "whitening": whitening,
        "device": fit_device,
        "dtype": dtype,
        "batch_size": batch_size,
        "row_normalize": False,
        "preprocessing_center": None,
        "norm_eps": 1e-12,
    }
    shape = (MAX_ITER + 1, dataset.hidden_size)
    limits = torch.full(shape, torch.nan, dtype=torch.float32)
    logcosh_objective = torch.empty(shape, dtype=torch.float32)
    logcosh = torch.empty(shape, dtype=torch.float32)
    kurtosis = torch.empty(shape, dtype=torch.float32)
    n_samples = int(source.shape[0])
    iterations = tqdm(
        range(MAX_ITER),
        desc=f"Layer {layer} FastICA",
        unit="iter",
        dynamic_ncols=True,
    )
    for iteration_index in iterations:
        term_sum = torch.zeros_like(weights)
        derivative_sum = torch.zeros(weights.shape[0], dtype=dtype, device=fit_device)
        raw2 = torch.zeros(weights.shape[0], dtype=torch.float64, device=fit_device)
        raw4 = torch.zeros_like(raw2)
        objective_sum = torch.zeros_like(raw2)
        for whitened in _whitened_batches(source, **batch_kwargs):
            projected = weights @ whitened
            transformed, derivative_mean = nonlinearity(projected)
            batch_count = int(whitened.shape[1])
            term_sum.addmm_(transformed, whitened.T)
            derivative_sum += derivative_mean * batch_count
            squared = projected.square()
            raw2 += squared.sum(dim=1, dtype=torch.float64)
            raw4 += squared.square().sum(dim=1, dtype=torch.float64)
            objective_sum += objective(projected).to(torch.float64) * batch_count
        variance = (raw2 / n_samples).clamp_min(torch.finfo(torch.float64).eps)
        raw_objective = objective_sum / n_samples
        logcosh_objective[iteration_index] = raw_objective.float().cpu()
        logcosh[iteration_index] = (
            raw_objective - GAUSSIAN_OBJECTIVES["logcosh"]
        ).abs().float().cpu()
        kurtosis[iteration_index] = (
            raw4 / n_samples / variance.square() - 3.0
        ).float().cpu()
        updated = term_sum / n_samples - (derivative_sum / n_samples)[:, None] * weights
        updated = _symmetric_decorrelation(updated)
        component_limit = torch.abs(
            torch.abs(torch.sum(updated * weights, dim=1)) - 1.0
        )
        limits[iteration_index + 1] = component_limit.float().cpu()
        iterations.set_postfix(
            max_lim=f"{float(component_limit.max()):.2e}",
            mean_lim=f"{float(component_limit.mean()):.2e}",
        )
        weights = updated
    final_objective, final_logcosh, final_kurtosis = population_statistics(
        source, weights, batch_kwargs=batch_kwargs, objective=objective
    )
    logcosh_objective[MAX_ITER] = final_objective.float().cpu()
    logcosh[MAX_ITER] = final_logcosh.float().cpu()
    kurtosis[MAX_ITER] = final_kurtosis.float().cpu()
    return {
        "iterations": np.arange(MAX_ITER + 1, dtype=np.int64),
        "component_limit": limits.numpy(),
        "logcosh_objective": logcosh_objective.numpy(),
        "logcosh_contrast": logcosh.numpy(),
        "excess_kurtosis": kurtosis.numpy(),
    }


def validate_layer_result(path: Path) -> bool:
    with np.load(path) as data:
        if "logcosh_objective" not in data:
            return False
        expected = {
            "iterations": (MAX_ITER + 1,),
            "component_limit": (MAX_ITER + 1, EXPECTED_HIDDEN_SIZE),
            "logcosh_objective": (MAX_ITER + 1, EXPECTED_HIDDEN_SIZE),
            "logcosh_contrast": (MAX_ITER + 1, EXPECTED_HIDDEN_SIZE),
            "excess_kurtosis": (MAX_ITER + 1, EXPECTED_HIDDEN_SIZE),
        }
        for key, shape in expected.items():
            if key not in data or data[key].shape != shape:
                raise ValueError(f"invalid {key} in {path}")
        if not np.isnan(data["component_limit"][0]).all():
            raise ValueError(f"iteration-0 limit must be undefined: {path}")
        for key in (
            "component_limit", "logcosh_objective", "logcosh_contrast", "excess_kurtosis"
        ):
            values = data[key][1:] if key == "component_limit" else data[key]
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {key} in {path}")
        expected_contrast = np.abs(
            data["logcosh_objective"] - GAUSSIAN_OBJECTIVES["logcosh"]
        )
        if not np.allclose(data["logcosh_contrast"], expected_contrast, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Logcosh objective and contrast disagree in {path}")
    return True


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the default pilot run")
    if args.fit_batch_size < 1:
        raise ValueError("--fit-batch-size must be positive")
    layers = parse_layers(args.layers)
    activations = args.activations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    dataset = ActivationDataset(activations)
    validate_dataset(dataset, layers)
    manifest_path = activations / "activations.json"
    resolved = {
        "format": "icalens.mean-limit-pilot",
        "format_version": 1,
        "activations": str(activations),
        "activation_manifest_sha256": sha256(manifest_path),
        "model": EXPECTED_MODEL,
        "activation_site": "resid_post",
        "layers": list(layers),
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "algorithm": "parallel",
        "fun": "logcosh",
        "max_iter": MAX_ITER,
        "seed": SEED,
        "fit_batch_size": args.fit_batch_size,
        "row_normalize": False,
        "whiten": "unit-variance",
    }
    source_state = source_provenance()
    run = ResumableRun.open(
        output=output, resolved=resolved, source=source_state, status="running"
    )
    warn_if_dirty(source_state)
    completed: set[int] = set()
    for layer in layers:
        path = layer_path(output, layer)
        if path.is_file() and validate_layer_result(path):
            completed.add(layer)
    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · FastICA convergence diagnostics",
        completed=len(completed),
        total=len(layers),
        completed_unit_ids=completed,
        unit_label="layer trajectories",
        recent_label="Recent fitting output",
        detail_filename="run-detail.log",
        source_dirty=bool(source_state.get("dirty")),
    ) as display:
        warn_if_dirty(source_state)
        for layer in layers:
            path = layer_path(output, layer)
            display.phase("Recording component-wise FastICA diagnostics", layer=layer)
            if layer in completed:
                log(f"Reusing completed layer {layer}: {path}")
                continue
            if path.is_file():
                log(f"Recomputing layer {layer}: saved result predates raw Logcosh recording.")
            log(f"Starting layer {layer}.")
            arrays = fit_layer(
                dataset, layer=layer, device=args.device, batch_size=args.fit_batch_size
            )
            atomic_save_npz(path, layer=np.asarray(layer), **arrays)
            if not validate_layer_result(path):
                raise ValueError(f"new layer result lacks raw Logcosh objective: {path}")
            display.complete_unit(layer, refresh=True)
            log(f"Completed layer {layer}: {path}")
        display.phase("Writing summary")
    layer_files = [layer_path(output, layer) for layer in layers]
    atomic_write_json(
        output / "summary.json",
        {
            "status": "complete",
            "resolved": resolved,
            "layer_files": [str(path) for path in layer_files],
            "layer_file_sha256": {str(path): sha256(path) for path in layer_files},
            "definitions": {
                "component_limit": "abs(abs(dot(w_t, w_(t-1))) - 1)",
                "logcosh_objective": "E[log(cosh(score))]",
                "logcosh_contrast": "abs(E[log(cosh(score))] - Gaussian reference)",
                "excess_kurtosis": "E[score^4] / E[score^2]^2 - 3 on centered whitened projections",
            },
        },
    )
    run.set_status("complete", complete=True)
    log(f"Experiment complete: {output}")


if __name__ == "__main__":
    main()
