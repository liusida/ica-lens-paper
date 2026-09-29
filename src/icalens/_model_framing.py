"""Load the versioned raw-text document-framing registry."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

REGISTRY_FILENAME = "model_framing.json"
REGISTRY_URL = "https://raw.githubusercontent.com/liusida/ica-lens-paper/main/model_framing.json"
MAX_REGISTRY_BYTES = 1_000_000


@dataclass(frozen=True)
class FramingPolicy:
    """One exact model's framing policy and registry provenance."""

    model_id: str
    entry: dict[str, Any]
    source: str
    sha256: str
    schema_version: int


def resolve_framing_policy(
    model_id: str,
    *,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
) -> FramingPolicy:
    """Resolve an exact model entry, refreshing GitHub when requested or missing."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    model_id = model_id.strip()
    cache_path = _cache_path(cache_dir)

    if refresh:
        registry, raw = _download_registry()
        _write_cache(cache_path, raw)
        return _policy_from_registry(model_id, registry, raw, REGISTRY_URL)

    bundled, bundled_raw, bundled_source = _load_bundled_registry()
    if model_id in bundled["models"]:
        return _policy_from_registry(model_id, bundled, bundled_raw, bundled_source)

    cached = _read_registry_file(cache_path)
    if cached is not None:
        registry, raw = cached
        if model_id in registry["models"]:
            return _policy_from_registry(model_id, registry, raw, f"cache:{REGISTRY_FILENAME}")

    try:
        registry, raw = _download_registry()
    except Exception as error:
        raise ValueError(
            f"No document-framing policy is known for {model_id!r}, and the current "
            f"GitHub registry could not be checked: {error}. Specify --document-framing "
            "explicitly or add the model to model_framing.json."
        ) from error
    _write_cache(cache_path, raw)
    if model_id not in registry["models"]:
        raise ValueError(
            f"No document-framing policy is known for {model_id!r}. Specify "
            "--document-framing explicitly or add the model to model_framing.json."
        )
    return _policy_from_registry(model_id, registry, raw, REGISTRY_URL)


def _load_bundled_registry() -> tuple[dict[str, Any], bytes, str]:
    project_registry = Path(__file__).parents[2] / REGISTRY_FILENAME
    if project_registry.is_file():
        raw = project_registry.read_bytes()
        return _parse_registry(raw), raw, f"project:{REGISTRY_FILENAME}"
    resource = files("icalens").joinpath(REGISTRY_FILENAME)
    raw = resource.read_bytes()
    return _parse_registry(raw), raw, f"package:{REGISTRY_FILENAME}"


def _download_registry() -> tuple[dict[str, Any], bytes]:
    request = urllib.request.Request(
        REGISTRY_URL,
        headers={"User-Agent": "icalens-model-framing-registry"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        raw = response.read(MAX_REGISTRY_BYTES + 1)
    if len(raw) > MAX_REGISTRY_BYTES:
        raise ValueError("downloaded model-framing registry is unexpectedly large")
    return _parse_registry(raw), raw


def _read_registry_file(path: Path) -> tuple[dict[str, Any], bytes] | None:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        return _parse_registry(raw), raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(raw)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_path(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser() / REGISTRY_FILENAME
    configured = os.environ.get("ICALENS_CACHE_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache" / "icalens"
    return root / REGISTRY_FILENAME


def _parse_registry(raw: bytes) -> dict[str, Any]:
    registry = json.loads(raw)
    if not isinstance(registry, dict):
        raise TypeError("model-framing registry must be a JSON object")
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported model-framing registry schema version")
    if registry.get("policy_scope") != "independent_pretraining_document":
        raise ValueError("unsupported model-framing registry policy scope")
    models = registry.get("models")
    if not isinstance(models, dict):
        raise TypeError("model-framing registry models must be an object")
    for model_id, entry in models.items():
        if not isinstance(model_id, str) or not isinstance(entry, dict):
            raise TypeError("model-framing registry entries must be objects")
        strategy = entry.get("document_framing")
        if strategy not in {"prepend-bos", "prepend-eos"}:
            raise ValueError(f"invalid framing strategy for {model_id!r}")
        role = strategy.removeprefix("prepend-")
        if entry.get("token_role") != role:
            raise ValueError(f"token role does not match strategy for {model_id!r}")
        if not isinstance(entry.get("expected_token"), str):
            raise TypeError(f"expected token is missing for {model_id!r}")
        evidence_url = entry.get("evidence_url")
        if not isinstance(evidence_url, str) or not evidence_url.startswith("https://"):
            raise ValueError(f"official evidence URL is missing for {model_id!r}")
    return registry


def _policy_from_registry(
    model_id: str,
    registry: dict[str, Any],
    raw: bytes,
    source: str,
) -> FramingPolicy:
    entry = registry["models"].get(model_id)
    if not isinstance(entry, dict):
        raise ValueError(f"No document-framing policy is known for {model_id!r}")
    return FramingPolicy(
        model_id=model_id,
        entry=dict(entry),
        source=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=int(registry["schema_version"]),
    )
