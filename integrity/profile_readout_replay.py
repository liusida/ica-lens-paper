"""Replay the logit-lens and R-lens readout portion of C12 profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from icalens import ICALens
from icalens.analysis import _resolve_model_and_tokenizer
from icalens.profiling import (
    _load_r_lens,
    _logit_lens,
    _r_lens_source_map,
    _r_lens_tokens,
)

SIGNS = ("positive", "negative")
SIDES = ("top_tokens", "bottom_tokens")


def sampled_indices(size: int, count: int, seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[: min(count, size)].sort().values.numpy()


def readout_fragment(
    readouts: list[dict[str, Any]], components: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    ids: list[int] = []
    logits: list[float] = []
    identities: list[str] = []
    for component in components:
        for sign in SIGNS:
            for side in SIDES:
                for entry in readouts[int(component)][sign][side]:
                    ids.append(int(entry["token_id"]))
                    logits.append(float(entry["logit"]))
                    identities.append(f"{entry['token']}\0{entry['text']}")
    return np.asarray(ids), np.asarray(logits), tuple(identities)


def stored_readouts(profile: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [{sign: component[name][sign] for sign in SIGNS} for component in profile["components"]]


def run(
    *,
    lens_root: Path,
    r_lens_path: Path,
    layer: int,
    verification_components: int,
    verification_seed: int,
    rtol: float,
    atol: float,
    output: Path,
    canary_id: str = "gpt2",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("C12 profile-readout replay requires CUDA")
    lens = ICALens.from_pretrained(lens_root)
    artifact = lens._get_layer(layer)
    profile = lens._get_profile(artifact)
    selection = profile["selection"]
    components = sampled_indices(artifact.n_components, verification_components, verification_seed)

    try:
        _resolve_model_and_tokenizer(lens, None, None, "cuda")
        actual_logit = _logit_lens(
            lens,
            artifact,
            top_k=int(selection["logit_lens_top_k"]),
            batch_size=int(selection["logit_lens_batch_size"]),
            progress=True,
        )
        r_lens, _ = _load_r_lens(lens, artifact, r_lens_path)
        source_map = _r_lens_source_map(r_lens, layer)
        if source_map is None:
            raise ValueError(f"R-lens has no source map for layer {layer}")
        actual_r_lens = _r_lens_tokens(
            lens,
            artifact,
            source_map=source_map,
            top_k=int(selection["r_lens_top_k"]),
            batch_size=int(selection["r_lens_batch_size"]),
            progress=True,
        )
    finally:
        lens.unload_model()

    fragments: dict[str, np.ndarray] = {"component_indices": components}
    comparisons: dict[str, bool] = {}
    maximum_errors: dict[str, float] = {}
    for name, expected_values, actual_values in (
        ("logit_lens", stored_readouts(profile, "logit_lens"), actual_logit),
        ("r_lens", stored_readouts(profile, "r_lens"), actual_r_lens),
    ):
        expected_ids, expected_logits, expected_identity = readout_fragment(
            expected_values, components
        )
        actual_ids, actual_logits, actual_identity = readout_fragment(actual_values, components)
        comparisons[f"{name}_token_ids"] = bool(np.array_equal(expected_ids, actual_ids))
        comparisons[f"{name}_token_text"] = expected_identity == actual_identity
        comparisons[f"{name}_logits"] = bool(
            np.allclose(expected_logits, actual_logits, rtol=rtol, atol=atol)
        )
        maximum_errors[name] = float(np.max(np.abs(expected_logits - actual_logits)))
        fragments[f"expected_{name}_ids"] = expected_ids
        fragments[f"actual_{name}_ids"] = actual_ids
        fragments[f"expected_{name}_logits"] = expected_logits
        fragments[f"actual_{name}_logits"] = actual_logits

    output.mkdir(parents=True, exist_ok=True)
    fragment_path = output / "fragment.npz"
    np.savez_compressed(fragment_path, **fragments)
    status = "pass" if all(comparisons.values()) else "fail"
    return {
        "check": f"C12-profile-readouts-{canary_id}-layer{layer}",
        "status": status,
        "scope": {
            "input_data_ids": ["D01", "D02", "D11", "D15"],
            "code_ids": ["C12"],
            "output_data_ids": ["D12"],
            "canary": canary_id,
            "layer": layer,
            "verification_components": len(components),
            "verification_seed": verification_seed,
            "covered_behavior": ["logit_lens", "r_lens"],
            "not_covered": ["occurrence_selection"],
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
