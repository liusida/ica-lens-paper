"""Numerically replay the population-statistics portion of C12 profiling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset

STATISTIC_SECTIONS = ("score_statistics", "sign_statistics")


def sampled_indices(size: int, count: int, seed: int) -> np.ndarray:
    if count < 1 or count > size:
        raise ValueError(f"sample size must be in 1..{size}, got {count}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[:count].sort().values.numpy()


def section_matrix(
    profile: dict[str, Any], components: np.ndarray, section: str
) -> tuple[list[str], np.ndarray]:
    entries = profile["components"]
    keys = sorted(entries[int(components[0])][section])
    values = np.asarray(
        [[float(entries[int(index)][section][key]) for key in keys] for index in components],
        dtype=np.float64,
    )
    return keys, values


def run(
    *,
    lens_root: Path,
    activation_root: Path,
    layer: int,
    verification_components: int,
    verification_seed: int,
    rtol: float,
    atol: float,
    output: Path,
    canary_id: str = "gpt2",
) -> dict[str, Any]:
    """Recompute full-population profile statistics and compare a small fragment."""
    if not torch.cuda.is_available():
        raise RuntimeError("C12 profile-statistics replay requires CUDA")
    if rtol < 0 or atol < 0:
        raise ValueError("comparison tolerances must be non-negative")

    dataset = ActivationDataset(activation_root)
    lens = ICALens.from_pretrained(lens_root)
    artifact = lens._get_layer(layer)
    expected = copy.deepcopy(lens._get_profile(artifact))
    components = sampled_indices(
        artifact.n_components,
        min(verification_components, artifact.n_components),
        verification_seed,
    )

    provenance = dataset.provenance
    provenance["statistics_sampling"] = {
        "policy": "all",
        "seed": None,
        "selected_tokens": dataset.sample_count,
        "population_tokens": dataset.sample_count,
    }
    actual = lens.refresh_profile_statistics_from_activations(
        dataset.layer(layer),
        layer=layer,
        rows=None,
        batch_size=8192,
        provenance=provenance,
        device="cuda",
        progress=True,
    )

    expected_directions = np.asarray(
        [expected["components"][int(index)]["tail_direction"] for index in components]
    )
    actual_directions = np.asarray(
        [actual["components"][int(index)]["tail_direction"] for index in components]
    )
    comparisons: dict[str, bool] = {
        "statistics_provenance": expected.get("score_statistics_provenance") == provenance,
        "tail_direction": bool(np.array_equal(expected_directions, actual_directions)),
    }
    maximum_errors: dict[str, float] = {}
    fragment: dict[str, Any] = {
        "component_indices": components,
        "expected_tail_direction": expected_directions,
        "actual_tail_direction": actual_directions,
    }
    for section in STATISTIC_SECTIONS:
        expected_keys, expected_values = section_matrix(expected, components, section)
        actual_keys, actual_values = section_matrix(actual, components, section)
        keys_match = expected_keys == actual_keys
        comparisons[f"{section}_keys"] = keys_match
        comparisons[section] = keys_match and bool(
            np.allclose(expected_values, actual_values, rtol=rtol, atol=atol)
        )
        maximum_errors[section] = float(np.max(np.abs(expected_values - actual_values)))
        fragment[f"{section}_keys"] = np.asarray(expected_keys)
        fragment[f"expected_{section}"] = expected_values
        fragment[f"actual_{section}"] = actual_values

    output.mkdir(parents=True, exist_ok=True)
    fragment_path = output / "fragment.npz"
    np.savez_compressed(fragment_path, **fragment)
    status = "pass" if all(comparisons.values()) else "fail"
    return {
        "check": f"C12-profile-statistics-{canary_id}-layer{layer}",
        "status": status,
        "scope": {
            "input_data_ids": ["D10", "D11"],
            "code_ids": ["C12"],
            "output_data_ids": ["D12"],
            "canary": canary_id,
            "layer": layer,
            "profiled_rows": dataset.sample_count,
            "profiled_components": artifact.n_components,
            "verification_components": len(components),
            "verification_seed": verification_seed,
            "covered_behavior": [
                "population_score_statistics",
                "sign_and_energy_statistics",
                "tail_orientation",
            ],
            "not_covered": ["occurrence_selection", "logit_lens", "r_lens"],
        },
        "comparison": {
            "close_by_field": comparisons,
            "maximum_absolute_error_by_field": maximum_errors,
            "rtol": rtol,
            "atol": atol,
        },
        "artifacts": {"comparison_fragment": str(fragment_path)},
        "failed_checks": [name for name, passed in comparisons.items() if not passed],
    }
