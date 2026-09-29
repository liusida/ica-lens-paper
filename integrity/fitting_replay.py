"""Numerically replay C11 fitting from the accepted D10 activation cache."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens.cli.fit_activations import main as fit_activations


def sampled_indices(size: int, count: int, seed: int) -> torch.Tensor:
    if count < 1 or count > size:
        raise ValueError(f"sample size must be in 1..{size}, got {count}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[:count].sort().values


def maximum_error(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(expected - actual)))


def run(
    *,
    lens_root: Path,
    activation_root: Path,
    layer: int,
    verification_rows: int,
    verification_components: int,
    verification_seed: int,
    rtol: float,
    atol: float,
    output: Path,
    canary_id: str = "gpt2",
) -> dict[str, Any]:
    """Refit one complete layer, then compare a deterministic numerical fragment."""
    if not torch.cuda.is_available():
        raise RuntimeError("C11 fitting replay requires CUDA")
    if rtol < 0 or atol < 0:
        raise ValueError("comparison tolerances must be non-negative")

    reference_dataset = ActivationDataset(activation_root)
    reference_lens = ICALens.from_pretrained(lens_root)
    reference_layer = reference_lens._get_layer(layer)
    fitting = reference_lens.metadata["layers"][str(layer)]["fitting"]
    row_indices = sampled_indices(
        reference_dataset.sample_count,
        min(verification_rows, reference_dataset.sample_count),
        verification_seed,
    )
    component_indices = sampled_indices(
        reference_layer.n_components,
        min(verification_components, reference_layer.n_components),
        verification_seed,
    )

    output.mkdir(parents=True, exist_ok=True)
    replay_lens_root = output / "working-lens"
    if replay_lens_root.exists():
        shutil.rmtree(replay_lens_root)
    fit_activations(
        [
            "--input",
            str(activation_root),
            "--layers",
            str(layer),
            "--output",
            str(replay_lens_root),
            "--icalens-preprocessing",
            str(fitting.get("icalens_preprocessing", "none")),
            "--max-iter",
            str(fitting["max_iter"]),
            "--objective-every",
            str(fitting.get("objective_every", 1)),
            "--fit-batch-size",
            str(fitting["batch_size"]),
            "--seed",
            str(fitting["random_state"]),
        ]
    )
    replay_lens = ICALens.from_pretrained(replay_lens_root)
    replay_layer = replay_lens._get_layer(layer)

    components = component_indices.numpy()
    assert reference_layer.center is not None and replay_layer.center is not None
    assert reference_layer.reading_matrix is not None
    assert replay_layer.reading_matrix is not None
    assert reference_layer.writing_matrix is not None
    assert replay_layer.writing_matrix is not None
    expected_center = np.asarray(reference_layer.center)
    actual_center = np.asarray(replay_layer.center)
    expected_reading = np.asarray(reference_layer.reading_matrix)[components]
    actual_reading = np.asarray(replay_layer.reading_matrix)[components]
    expected_writing = np.asarray(reference_layer.writing_matrix)[:, components]
    actual_writing = np.asarray(replay_layer.writing_matrix)[:, components]

    values = reference_dataset.layer(layer).index_select(0, row_indices).float().numpy()
    expected_scores = np.asarray(reference_lens.transform(values, layer=layer))[:, components]
    actual_scores = np.asarray(replay_lens.transform(values, layer=layer))[:, components]
    comparisons = {
        "center": bool(np.allclose(expected_center, actual_center, rtol=rtol, atol=atol)),
        "reading_matrix": bool(np.allclose(expected_reading, actual_reading, rtol=rtol, atol=atol)),
        "writing_matrix": bool(np.allclose(expected_writing, actual_writing, rtol=rtol, atol=atol)),
        "scores": bool(np.allclose(expected_scores, actual_scores, rtol=rtol, atol=atol)),
    }
    errors = {
        "center": maximum_error(expected_center, actual_center),
        "reading_matrix": maximum_error(expected_reading, actual_reading),
        "writing_matrix": maximum_error(expected_writing, actual_writing),
        "scores": maximum_error(expected_scores, actual_scores),
    }
    status = "pass" if all(comparisons.values()) else "fail"
    fragment_path = output / "fragment.npz"
    np.savez_compressed(
        fragment_path,
        row_indices=row_indices.numpy(),
        component_indices=components,
        expected_scores=expected_scores,
        actual_scores=actual_scores,
    )
    shutil.rmtree(replay_lens_root)

    return {
        "check": f"C11-fit-{canary_id}-layer{layer}",
        "status": status,
        "scope": {
            "input_data_ids": ["D10"],
            "code_ids": ["C11"],
            "output_data_ids": ["D11"],
            "canary": canary_id,
            "layer": layer,
            "fitting_rows": reference_dataset.sample_count,
            "fitted_components": reference_layer.n_components,
            "verification_rows": len(row_indices),
            "verification_components": len(component_indices),
            "verification_seed": verification_seed,
        },
        "comparison": {
            "close_by_field": comparisons,
            "maximum_absolute_error_by_field": errors,
            "rtol": rtol,
            "atol": atol,
        },
        "artifacts": {"comparison_fragment": str(fragment_path)},
        "failed_checks": [name for name, passed in comparisons.items() if not passed],
    }
