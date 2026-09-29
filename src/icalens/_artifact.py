"""Portable ICA Lens manifest and tensor persistence."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file

from .exceptions import ArtifactError

FORMAT_NAME = "icalens"
FORMAT_VERSION = 4
MINIMUM_PACKAGE_VERSION = "0.3.4"
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3, 4)
MANIFEST_FILENAME = "icalens.json"


@dataclass
class LayerArtifact:
    """Parameters and provenance for one fitted layer."""

    layer: int
    file: str
    n_components: int
    fitting: dict[str, Any]
    profile_file: str | None = None
    profile: dict[str, Any] | None = None
    center: NDArray[np.float32] | None = None
    reading_matrix: NDArray[np.float32] | None = None
    writing_matrix: NDArray[np.float32] | None = None
    preprocessing_center: NDArray[np.float32] | None = None

    @property
    def loaded(self) -> bool:
        return self.center is not None


def parse_manifest(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ArtifactError("icalens.json must contain a JSON object")
    if data.get("format") != FORMAT_NAME:
        raise ArtifactError(f"unsupported artifact format: {data.get('format')!r}")
    format_version = data.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ArtifactError(
            f"unsupported artifact format version: {format_version!r}; "
            f"this package supports versions {SUPPORTED_FORMAT_VERSIONS}"
        )
    expected_minimum = "0.3.2" if format_version == 3 else MINIMUM_PACKAGE_VERSION
    if format_version >= 3 and data.get("minimum_package_version") != expected_minimum:
        raise ArtifactError(
            f"format version 3 requires minimum_package_version={expected_minimum!r}"
        )
    model_key = "base_model" if format_version == 1 else "model"
    required = (model_key, "activation_site", "hidden_size", "input_preprocessing", "layers")
    missing = [key for key in required if key not in data]
    if missing:
        raise ArtifactError(f"manifest is missing required fields: {', '.join(missing)}")
    if not isinstance(data[model_key], dict) or not data[model_key].get("repo_id"):
        raise ArtifactError(f"manifest {model_key}.repo_id must be a non-empty string")
    if format_version >= 2 and data[model_key].get("type") not in ("base", "instruct"):
        raise ArtifactError("manifest model.type must be 'base' or 'instruct'")
    if not isinstance(data["layers"], dict):
        raise ArtifactError("manifest layers must be an object")
    return data


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        return parse_manifest(json.loads(path.read_text(encoding="utf-8")))
    except ArtifactError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read manifest {path}: {error}") from error


def layer_from_manifest(layer_key: str, data: Any) -> LayerArtifact:
    try:
        layer = int(layer_key)
        if not isinstance(data, dict):
            raise TypeError
        filename = str(data["file"])
        n_components = int(data["n_components"])
        fitting = dict(data["fitting"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(f"invalid manifest entry for layer {layer_key!r}") from error
    if layer < 0 or n_components <= 0 or not filename:
        raise ArtifactError(f"invalid manifest entry for layer {layer_key!r}")
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactError(f"unsafe tensor path for layer {layer}: {filename!r}")
    profile_file = data.get("component_profile")
    if profile_file is not None:
        profile_file = str(profile_file)
        profile_path = Path(profile_file)
        if profile_path.is_absolute() or ".." in profile_path.parts:
            raise ArtifactError(
                f"unsafe component profile path for layer {layer}: {profile_file!r}"
            )
    return LayerArtifact(
        layer=layer,
        file=filename,
        n_components=n_components,
        fitting=fitting,
        profile_file=profile_file,
    )


def load_profile(path: Path) -> dict[str, Any]:
    """Load one component-profile JSON file."""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read component profile {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("components"), list):
        raise ArtifactError(f"invalid component profile: {path}")
    return value


def load_layer(path: Path, artifact: LayerArtifact, hidden_size: int) -> None:
    try:
        tensors = load_file(path)
        center = _tensor(tensors, "center", (hidden_size,))
        reading = _tensor(tensors, "reading_matrix", (artifact.n_components, hidden_size))
        writing = _tensor(tensors, "writing_matrix", (hidden_size, artifact.n_components))
        preprocessing_center = (
            _tensor(tensors, "preprocessing_center", (hidden_size,))
            if "preprocessing_center" in tensors
            else None
        )
    except ArtifactError:
        raise
    except Exception as error:
        raise ArtifactError(f"could not load layer artifact {path}: {error}") from error
    artifact.center = center
    artifact.reading_matrix = reading
    artifact.writing_matrix = writing
    artifact.preprocessing_center = preprocessing_center


def save_directory(path: Path, manifest: dict[str, Any], layers: dict[int, LayerArtifact]) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ArtifactError(f"save destination exists and is not a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    backup: Path | None = None
    try:
        for artifact in layers.values():
            if not artifact.loaded:
                raise ArtifactError(f"layer {artifact.layer} tensors are not loaded")
            assert artifact.center is not None
            assert artifact.reading_matrix is not None
            assert artifact.writing_matrix is not None
            tensor_path = stage / artifact.file
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            tensors = {
                "center": artifact.center,
                "reading_matrix": artifact.reading_matrix,
                "writing_matrix": artifact.writing_matrix,
            }
            if artifact.preprocessing_center is not None:
                tensors["preprocessing_center"] = artifact.preprocessing_center
            save_file(tensors, tensor_path)
            if artifact.profile_file is not None and artifact.profile is not None:
                profile_path = stage / artifact.profile_file
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                _write_profile(profile_path, artifact.profile)
        (stage / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "README.md").write_text(_model_card(manifest), encoding="utf-8")
        if path.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{path.name}-backup-", dir=path.parent))
            backup.rmdir()
            os.replace(path, backup)
        os.replace(stage, path)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if backup is not None and backup.exists() and not path.exists():
            os.replace(backup, path)
        raise


def save_profile_checkpoint(path: Path, manifest: dict[str, Any], artifact: LayerArtifact) -> None:
    """Atomically add or replace one profile in an existing artifact directory."""
    path = path.expanduser().resolve()
    if not (path / MANIFEST_FILENAME).is_file():
        raise ArtifactError(f"profile checkpoint destination is not an ICA Lens: {path}")
    if artifact.profile_file is None or artifact.profile is None:
        raise ArtifactError(f"layer {artifact.layer} has no component profile to checkpoint")
    profile_path = path / artifact.profile_file
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_profile(profile_path, artifact.profile)
    _atomic_write_json(path / MANIFEST_FILENAME, manifest)
    (path / "README.md").write_text(_model_card(manifest), encoding="utf-8")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_profile(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        _write_profile(temporary, value, compressed=path.suffix == ".gz")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_profile(path: Path, value: dict[str, Any], *, compressed: bool | None = None) -> None:
    use_gzip = path.suffix == ".gz" if compressed is None else compressed
    if use_gzip:
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(value, handle, separators=(",", ":"), sort_keys=True)
    else:
        path.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True), encoding="utf-8")


def _tensor(
    tensors: dict[str, NDArray[Any]], name: str, shape: tuple[int, ...]
) -> NDArray[np.float32]:
    if name not in tensors:
        raise ArtifactError(f"layer artifact is missing tensor {name!r}")
    value = np.asarray(tensors[name], dtype=np.float32)
    if value.shape != shape:
        raise ArtifactError(f"tensor {name!r} has shape {value.shape}, expected {shape}")
    if not np.all(np.isfinite(value)):
        raise ArtifactError(f"tensor {name!r} contains non-finite values")
    return np.ascontiguousarray(value)


def _model_card(manifest: dict[str, Any]) -> str:
    model_entry = manifest.get("model", manifest.get("base_model"))
    if not isinstance(model_entry, dict):
        raise ArtifactError("manifest model metadata must be an object")
    model = model_entry["repo_id"]
    revision = model_entry.get("revision")
    model_type = model_entry.get("type", "base")
    site = manifest["activation_site"]
    layers = ", ".join(str(layer) for layer in sorted(map(int, manifest["layers"])))
    package_version = manifest.get("package_version", "unknown")
    preprocessing = manifest.get("input_preprocessing", {})
    normalization = preprocessing.get("row_normalization", "unknown")
    r_lens_profiles = manifest.get("r_lens_profiles", {})
    transfer_entries = (
        [
            (layer, provenance)
            for layer, provenance in r_lens_profiles.items()
            if isinstance(provenance, dict) and isinstance(provenance.get("transfer"), dict)
        ]
        if isinstance(r_lens_profiles, dict)
        else []
    )
    if transfer_entries:
        first_provenance = transfer_entries[0][1]
        transfer_layers = ", ".join(
            str(layer) for layer, _ in sorted(transfer_entries, key=lambda item: int(item[0]))
        )
        r_lens_note = f"""
## R-lens readouts

R-lens component-token readouts for layers {transfer_layers} reuse the R-lens
fitted for base model `{first_provenance.get("model_id", "unknown")}`. They were
transferred to this instruction-tuned model to reduce fitting compute. The
transfer was explicitly requested and passed hidden-size, activation-site, and
layer-map compatibility checks; its complete provenance is stored in
`icalens.json` and each component-profile file.
"""
    else:
        r_lens_note = ""
    layer_details = []
    provenances = []
    for layer, entry in sorted(manifest["layers"].items(), key=lambda item: int(item[0])):
        fitting = entry.get("fitting", {})
        provenances.append(fitting.get("provenance"))
        layer_details.append(
            f"| {layer} | {entry['n_components']} | "
            f"{fitting.get('n_samples', 'unknown')} | {fitting.get('max_iter', 'unknown')} |"
        )
    fitting_summary = "\n".join(layer_details)
    common_provenance = (
        provenances[0]
        if provenances and all(provenance == provenances[0] for provenance in provenances)
        else None
    )
    dataset_entry = (
        common_provenance.get("dataset") if isinstance(common_provenance, dict) else None
    )
    dataset_id = dataset_entry.get("repo_id") if isinstance(dataset_entry, dict) else None
    dataset_metadata = f"datasets:\n- {dataset_id}\n" if dataset_id else ""
    if (
        isinstance(dataset_id, str)
        and isinstance(dataset_entry, dict)
        and isinstance(common_provenance, dict)
    ):
        dataset_link = f"https://huggingface.co/datasets/{dataset_id}"
        dataset_revision = dataset_entry.get("revision", "not pinned")
        dataset_split = dataset_entry.get("split", "not recorded")
        fitting_data_rows = (
            f"| Fitting dataset | [{dataset_id}]({dataset_link}) |\n"
            f"| Dataset revision | `{dataset_revision}` |\n"
            f"| Dataset split | `{dataset_split}` |\n"
            f"| Fitting token scope | `{common_provenance.get('token_scope', 'not recorded')}` |\n"
            f"| Candidate tokens | {common_provenance.get('candidate_tokens', 'not recorded')} |\n"
            f"| Fitting tokens | {common_provenance.get('fitting_tokens', 'not recorded')} |"
        )
    else:
        fitting_data_rows = "| Fitting dataset | See per-layer provenance in `icalens.json` |"
    provenance_summary = (
        f"```json\n{json.dumps(common_provenance, indent=2, sort_keys=True)}\n```"
        if common_provenance is not None
        else "See the per-layer fitting metadata in `icalens.json`."
    )
    revision_text = f"`{revision}`" if revision is not None else "not pinned"
    model_link = f"https://huggingface.co/{model}"
    example_layer = min(map(int, manifest["layers"]))
    return f"""---
library_name: icalens
{dataset_metadata}\
tags:
- interpretability
- independent-component-analysis
- activations
- arxiv:2606.11722
---

# ICA Lens for {model}

This repository contains a fitted **ICA Lens** for analyzing internal activations
of [{model}]({model_link}). It provides layer-wise ICA transformations for mapping
residual-stream activations to independent-component scores and energy shares.

## Artifact summary

| Field | Value |
|---|---|
| Analyzed model | [{model}]({model_link}) |
| Analyzed model revision | {revision_text} |
| Model kind | `{model_type}` |
| Activation site | `{site}` |
| Layer indexing | `{manifest.get("layer_indexing", "unknown")}` |
| Available layers | {layers} |
| Hidden size | {manifest.get("hidden_size", "unknown")} |
| Input row normalization | `{normalization}` |
| ICALens package version | `{package_version}` |
{fitting_data_rows}

The analyzed model identity and revision are stored authoritatively in
`icalens.json`; model-card metadata is not used when loading the lens.

## Usage

Analyze text end to end:

```python
from icalens import ICALens

lens = ICALens.from_pretrained(\"REPOSITORY_ID\")
result = lens.analyze(\"She deposited the check at the bank.\", layer={example_layer})

print(result.tokens)
print(result.scores)  # signed standard ICA scores
print(result.energy)  # per-token squared-score fractions
```

Or transform activations captured separately:

```python
scores = lens.transform(activations, layer={example_layer})
```

Externally captured activations must use the model revision, activation site,
layer indexing, and preprocessing recorded in `icalens.json`.

## Score definition

Signed scores are the centered, whitened activations followed by the learned
orthogonal ICA rotation. No post-ICA source scaling is applied to v0.2 fits.
For a token, component energy share is `score² / sum(all component scores²)`.

{r_lens_note}

## Fitting

| Layer | Components | Fitting tokens | FastICA iterations |
|---:|---:|---:|---:|
{fitting_summary}

### Fitting provenance

{provenance_summary}

## Limitations

- Component IDs are specific to a layer and fitted artifact.
- Standard ICA scores are signed and are not probabilities.

## Paper

[ICA Lens: Interpreting Language Models Without Training Another Dictionary](https://arxiv.org/abs/2606.11722)

## Citation

```bibtex
@article{{liu2026icalens,
  title={{ICA Lens: Interpreting Language Models Without Training Another Dictionary}},
  author={{Liu, Sida and Han, Feijiang}},
  journal={{arXiv preprint arXiv:2606.11722}},
  year={{2026}}
}}
```
"""
