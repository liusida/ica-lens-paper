"""Replay the selected-occurrence portion of C12 profiling."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens.cli.profile import _recover_cached_records

IDENTITY_FIELDS = (
    "token",
    "text",
    "token_id",
    "position",
    "context",
    "context_target_start",
    "context_target_end",
    "source_index",
    "absolute_score_rank",
)
NUMERIC_FIELDS = ("score", "energy")


def sampled_indices(size: int, count: int, seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[: min(count, size)].sort().values.numpy()


def compare_component_occurrences(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float]:
    if expected["tail_direction"] != actual["tail_direction"]:
        return False, float("inf")
    maximum_error = 0.0
    for sign in ("positive", "negative"):
        expected_examples = expected["examples"][sign]
        actual_examples = actual["examples"][sign]
        if expected_examples["tokens"] != actual_examples["tokens"]:
            return False, float("inf")
        expected_occurrences = expected_examples["occurrences"]
        actual_occurrences = actual_examples["occurrences"]
        if len(expected_occurrences) != len(actual_occurrences):
            return False, float("inf")
        for left, right in zip(expected_occurrences, actual_occurrences, strict=True):
            if any(left.get(field) != right.get(field) for field in IDENTITY_FIELDS):
                return False, float("inf")
            for field in NUMERIC_FIELDS:
                left_value = float(left[field])
                right_value = float(right[field])
                maximum_error = max(maximum_error, abs(left_value - right_value))
                if not np.isclose(left_value, right_value, rtol=rtol, atol=atol):
                    return False, maximum_error
    return True, maximum_error


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
    if not torch.cuda.is_available():
        raise RuntimeError("C12 profile-occurrence replay requires CUDA")
    dataset = ActivationDataset(activation_root)
    lens = ICALens.from_pretrained(lens_root)
    artifact = lens._get_layer(layer)
    expected = copy.deepcopy(lens._get_profile(artifact))
    components = sampled_indices(artifact.n_components, verification_components, verification_seed)
    selection = expected["selection"]

    rows = torch.arange(dataset.sample_count, dtype=torch.int64)
    records = _recover_cached_records(dataset, rows, lens)
    expected_sampling = expected["example_provenance"]["profile_sampling"]
    provenance = dataset.provenance
    provenance["profile_sampling"] = {
        "policy": "uniform_without_replacement",
        "seed": int(expected_sampling["seed"]),
        "selected_tokens": dataset.sample_count,
        "population_tokens": dataset.sample_count,
    }
    actual = lens.refresh_profile_examples_from_activations(
        dataset.layer(layer),
        records,
        layer=layer,
        rows=rows,
        batch_size=8192,
        top_k_examples=int(selection["top_k_examples_on_selected_tail"]),
        provenance=provenance,
        device="cuda",
        progress=True,
    )

    provenance_exact = expected.get("example_provenance") == provenance
    selection_exact = all(
        expected["selection"].get(key) == actual["selection"].get(key)
        for key in (
            "top_k_examples_on_selected_tail",
            "example_selection",
            "example_absolute_score_rank",
        )
    )
    component_exact: dict[str, bool] = {}
    maximum_error = 0.0
    for component in components:
        exact, error = compare_component_occurrences(
            expected["components"][int(component)],
            actual["components"][int(component)],
            rtol=rtol,
            atol=atol,
        )
        component_exact[str(int(component))] = exact
        maximum_error = max(maximum_error, error)

    output.mkdir(parents=True, exist_ok=True)
    fragment_path = output / "fragment.json.gz"
    fragment = {
        "component_indices": components.tolist(),
        "expected": [expected["components"][int(index)] for index in components],
        "actual": [actual["components"][int(index)] for index in components],
    }
    with gzip.open(fragment_path, "wt", encoding="utf-8") as handle:
        json.dump(fragment, handle, sort_keys=True)

    comparisons = {
        "example_provenance": provenance_exact,
        "selection_protocol": selection_exact,
        "component_occurrences": all(component_exact.values()),
    }
    status = "pass" if all(comparisons.values()) else "fail"
    return {
        "check": f"C12-profile-occurrences-{canary_id}-layer{layer}",
        "status": status,
        "scope": {
            "input_data_ids": ["D02", "D03", "D10", "D11"],
            "code_ids": ["C12"],
            "output_data_ids": ["D12"],
            "canary": canary_id,
            "layer": layer,
            "profiled_rows": dataset.sample_count,
            "profiled_components": artifact.n_components,
            "verification_components": len(components),
            "verification_seed": verification_seed,
            "covered_behavior": [
                "dataset_and_token_context_replay",
                "selected_tail_occurrence_search",
                "absolute_score_ranking",
            ],
        },
        "comparison": {
            "close_by_field": comparisons,
            "component_exact": component_exact,
            "maximum_absolute_score_or_energy_error": maximum_error,
            "rtol": rtol,
            "atol": atol,
        },
        "artifacts": {"comparison_fragment": str(fragment_path)},
        "failed_checks": [name for name, passed in comparisons.items() if not passed],
    }
