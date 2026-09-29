"""Validate upstream GPT-2 reference artifacts and their relations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def run(
    *,
    lens_root: Path,
    activation_root: Path,
    layer: int,
    canary_id: str = "gpt2",
    r_lens_path: Path | None = None,
) -> dict[str, Any]:
    lens_manifest_path = lens_root / "icalens.json"
    activation_manifest_path = activation_root / "activations.json"
    lens = load_json(lens_manifest_path)
    activations = load_json(activation_manifest_path)
    layer_key = str(layer)
    lens_layer = lens.get("layers", {}).get(layer_key)
    activation_layer = activations.get("layers", {}).get(layer_key)
    if not isinstance(lens_layer, dict) or not isinstance(activation_layer, dict):
        raise ValueError(f"layer {layer} is missing from a reference artifact")

    fitting = lens_layer.get("fitting", {}).get("provenance")
    capture = activations.get("provenance")
    if not isinstance(fitting, dict) or not isinstance(capture, dict):
        raise ValueError("reference artifacts are missing fitting/capture provenance")

    checks: dict[str, bool] = {
        "model_identity": lens.get("model") == activations.get("model"),
        "activation_site": lens.get("activation_site") == activations.get("activation_site"),
        "layer_indexing": lens.get("layer_indexing") == activations.get("layer_indexing"),
        "hidden_size": int(activation_layer.get("shape", [0, 0])[1])
        == int(activations.get("hidden_size", -1)),
        "component_count": int(lens_layer.get("n_components", -1))
        == int(activations.get("hidden_size", -2)),
        "activation_layer_complete": activation_layer.get("status") == "complete",
    }
    for field in (
        "candidate_tokens",
        "context_length",
        "dataset",
        "fitting_tokens",
        "sampling_seed",
        "text_field",
        "token_scope",
    ):
        checks[f"provenance_{field}"] = fitting.get(field) == capture.get(field)
    fitting_framing = fitting.get("document_framing", {})
    capture_framing = capture.get("document_framing", {})
    for field in ("strategy", "token", "token_id", "included_in_fitting_samples"):
        checks[f"document_framing_{field}"] = (
            isinstance(fitting_framing, dict)
            and isinstance(capture_framing, dict)
            and fitting_framing.get(field) == capture_framing.get(field)
        )

    artifact_path = lens_root / str(lens_layer["file"])
    profile_path = lens_root / str(lens_layer["component_profile"])
    activation_path = activation_root / str(activation_layer["file"])
    sample_path = activation_root / "samples.safetensors"
    files = (artifact_path, profile_path, activation_path, sample_path)
    missing = [str(path) for path in files if not path.is_file()]
    checks["required_files_exist"] = not missing
    expected_shape = [int(activations["sample_count"]), int(activations["hidden_size"])]
    checks["activation_shape"] = activation_layer.get("shape") == expected_shape
    r_lens_expected = None
    r_lens_actual = None
    if r_lens_path is not None:
        profile = load_json_gzip(profile_path)
        r_lens_expected = (profile.get("r_lens_provenance") or {}).get("sha256")
        r_lens_actual = sha256(r_lens_path) if r_lens_path.is_file() else None
        checks["r_lens_file_exists"] = r_lens_path.is_file()
        checks["r_lens_identity"] = bool(r_lens_expected) and r_lens_expected == r_lens_actual

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "check": f"reference-preflight-{canary_id}-layer{layer}",
        "status": "pass" if not failed else "fail",
        "scope": {
            "data_ids": [
                "D01",
                "D02",
                "D03",
                "D10",
                "D11",
                "D12",
                *(["D15"] if r_lens_path is not None else []),
            ],
            "canary": canary_id,
            "layer": layer,
            "note": "artifact identity preflight required before downstream replay",
        },
        "references": {
            "lens": str(lens_root),
            "lens_manifest_sha256": sha256(lens_manifest_path),
            "activations": str(activation_root),
            "activation_manifest_sha256": sha256(activation_manifest_path),
            "model": lens.get("model"),
            "dataset": fitting.get("dataset"),
            "layer_artifact": str(artifact_path),
            "layer_artifact_sha256": sha256(artifact_path) if artifact_path.is_file() else None,
            "component_profile": str(profile_path),
            "component_profile_sha256": sha256(profile_path) if profile_path.is_file() else None,
            "r_lens": str(r_lens_path) if r_lens_path is not None else None,
            "r_lens_expected_sha256": r_lens_expected,
            "r_lens_actual_sha256": r_lens_actual,
        },
        "checks": checks,
        "failed_checks": failed,
        "missing_files": missing,
    }


def load_json_gzip(path: Path) -> dict[str, Any]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
