"""Reusable, disk-backed activation datasets for ICA fitting."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast

import torch
from safetensors.torch import load_file, save_file

FORMAT = "icalens.activations"
FORMAT_VERSION = 1
MANIFEST_NAME = "activations.json"
SAMPLES_NAME = "samples.safetensors"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class ActivationDataset:
    """A validated activation dataset whose layer tensors are memory-mapped."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"activation dataset manifest not found: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        _validate_manifest(self.manifest)
        if self.manifest["status"] != "complete":
            raise ValueError(f"activation dataset is not complete: {self.path}")
        self._manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(sorted(int(layer) for layer in self.manifest["layers"]))

    @property
    def sample_count(self) -> int:
        return int(self.manifest["sample_count"])

    @property
    def hidden_size(self) -> int:
        return int(self.manifest["hidden_size"])

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPES[str(self.manifest["dtype"])]

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.manifest["model"])

    @property
    def provenance(self) -> dict[str, Any]:
        value = json.loads(json.dumps(self.manifest["provenance"]))
        value["activation_dataset"] = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "dtype": str(self.manifest["dtype"]),
            "manifest_sha256": self._manifest_sha256,
        }
        return cast(dict[str, Any], value)

    def layer(self, layer: int) -> torch.Tensor:
        """Return one read-only-in-practice, disk-backed layer tensor."""
        entry = self.manifest["layers"].get(str(int(layer)))
        if entry is None:
            raise ValueError(
                f"layer {layer} is unavailable; available layers: {self.available_layers}"
            )
        path = self.path / str(entry["file"])
        expected_bytes = self.sample_count * self.hidden_size * self.dtype.itemsize
        actual_bytes = path.stat().st_size if path.is_file() else -1
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"activation layer file has {actual_bytes} bytes, expected {expected_bytes}: {path}"
            )
        values = torch.from_file(
            str(path),
            shared=False,
            size=self.sample_count * self.hidden_size,
            dtype=self.dtype,
        )
        return values.reshape(self.sample_count, self.hidden_size)

    def samples(self) -> dict[str, torch.Tensor]:
        """Return validated token-position metadata aligned with activation rows."""
        path = self.path / str(self.manifest["samples_file"])
        if not path.is_file():
            raise FileNotFoundError(f"activation sample metadata not found: {path}")
        values = load_file(path)
        required = {"document_index", "position", "token_id"}
        missing = required - values.keys()
        if missing:
            raise ValueError(f"activation sample metadata is missing: {sorted(missing)}")
        for name in required:
            tensor = values[name]
            if tensor.ndim != 1 or int(tensor.shape[0]) != self.sample_count:
                raise ValueError(
                    f"sample metadata {name!r} has shape {tuple(tensor.shape)}; "
                    f"expected ({self.sample_count},)"
                )
        return values


class ActivationDatasetWriter:
    """Append captures into raw layer files and checkpoint a portable manifest."""

    def __init__(
        self,
        path: str | Path,
        *,
        model: Mapping[str, Any],
        activation_site: str,
        layer_indexing: str,
        layers: Sequence[int],
        sample_count: int,
        hidden_size: int,
        dtype: torch.dtype,
        provenance: Mapping[str, Any],
        samples: Mapping[str, torch.Tensor],
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "layers").mkdir(exist_ok=True)
        dtype_name = _dtype_name(dtype)
        layer_entries = {
            str(int(layer)): {
                "file": f"layers/layer_{int(layer):02d}.{dtype_name}.bin",
                "shape": [int(sample_count), int(hidden_size)],
                "status": "pending",
            }
            for layer in layers
        }
        requested = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "status": "capturing",
            "model": dict(model),
            "activation_site": str(activation_site),
            "layer_indexing": str(layer_indexing),
            "sample_count": int(sample_count),
            "hidden_size": int(hidden_size),
            "dtype": dtype_name,
            "samples_file": SAMPLES_NAME,
            "layers": layer_entries,
            "provenance": json.loads(json.dumps(provenance)),
        }
        manifest_path = self.path / MANIFEST_NAME
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            _validate_compatible(existing, requested)
            self.manifest = existing
            samples_path = self.path / SAMPLES_NAME
            if not samples_path.is_file():
                save_file(
                    {name: value.detach().cpu().contiguous() for name, value in samples.items()},
                    samples_path,
                )
        else:
            self.manifest = requested
            _write_json_atomic(manifest_path, self.manifest)
            save_file(
                {name: value.detach().cpu().contiguous() for name, value in samples.items()},
                self.path / SAMPLES_NAME,
            )

    @property
    def missing_layers(self) -> tuple[int, ...]:
        return tuple(
            int(layer)
            for layer, entry in self.manifest["layers"].items()
            if entry["status"] != "complete"
        )

    @property
    def required_bytes(self) -> int:
        return (
            len(self.missing_layers)
            * int(self.manifest["sample_count"])
            * int(self.manifest["hidden_size"])
            * _DTYPES[self.manifest["dtype"]].itemsize
        )

    @contextmanager
    def group(self, layers: Sequence[int]) -> Iterator[ActivationGroupWriter]:
        sink = ActivationGroupWriter(self, layers)
        try:
            yield sink
            sink.finish()
        finally:
            sink.close()

    def finish(self) -> None:
        missing = self.missing_layers
        if missing:
            raise RuntimeError(f"cannot finish activation dataset; missing layers: {missing}")
        self.manifest["status"] = "complete"
        _write_json_atomic(self.path / MANIFEST_NAME, self.manifest)


class ActivationGroupWriter:
    def __init__(self, dataset: ActivationDatasetWriter, layers: Sequence[int]) -> None:
        self.dataset = dataset
        self.layers = tuple(int(layer) for layer in layers)
        self.offset = 0
        self.handles: dict[int, BinaryIO] = {}
        for layer in self.layers:
            entry = dataset.manifest["layers"][str(layer)]
            final = dataset.path / entry["file"]
            partial = final.with_suffix(final.suffix + ".partial")
            partial.unlink(missing_ok=True)
            self.handles[layer] = partial.open("wb", buffering=1024 * 1024)

    def append(self, values: Mapping[int, torch.Tensor]) -> None:
        row_count: int | None = None
        for layer in self.layers:
            tensor = values[layer].detach().to(device="cpu").contiguous()
            if tensor.ndim != 2 or int(tensor.shape[1]) != self.dataset.manifest["hidden_size"]:
                raise ValueError(
                    f"unexpected activation shape for layer {layer}: {tuple(tensor.shape)}"
                )
            if _dtype_name(tensor.dtype) != self.dataset.manifest["dtype"]:
                raise ValueError(f"unexpected activation dtype for layer {layer}: {tensor.dtype}")
            if row_count is None:
                row_count = int(tensor.shape[0])
            elif int(tensor.shape[0]) != row_count:
                raise ValueError("captured layers have different row counts")
            self.handles[layer].write(tensor.view(torch.uint8).numpy().tobytes())
        assert row_count is not None
        self.offset += row_count

    def finish(self) -> None:
        expected = int(self.dataset.manifest["sample_count"])
        if self.offset != expected:
            raise RuntimeError(f"wrote {self.offset} activation rows; expected {expected}")
        self.close()
        for layer in self.layers:
            entry = self.dataset.manifest["layers"][str(layer)]
            final = self.dataset.path / entry["file"]
            partial = final.with_suffix(final.suffix + ".partial")
            expected_bytes = (
                expected
                * int(self.dataset.manifest["hidden_size"])
                * _DTYPES[self.dataset.manifest["dtype"]].itemsize
            )
            if partial.stat().st_size != expected_bytes:
                raise RuntimeError(f"incomplete activation file: {partial}")
            os.replace(partial, final)
            entry["status"] = "complete"
        _write_json_atomic(self.dataset.path / MANIFEST_NAME, self.dataset.manifest)

    def close(self) -> None:
        for handle in self.handles.values():
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()


def sample_metadata(
    documents: Sequence[Any], selected_positions: Mapping[int, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Build token-aligned indices sufficient to recover samples from the source dataset."""
    document_indices: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    token_ids: list[torch.Tensor] = []
    for document_index, selected in selected_positions.items():
        selected = selected.to(dtype=torch.long, device="cpu")
        document_indices.append(torch.full_like(selected, int(document_index)))
        positions.append(selected)
        token_ids.append(documents[document_index].input_ids.index_select(0, selected).long())
    return {
        "document_index": torch.cat(document_indices),
        "position": torch.cat(positions),
        "token_id": torch.cat(token_ids),
    }


def check_disk_space(path: Path, *, required_bytes: int) -> tuple[int, int]:
    """Return required and available bytes, failing below a 10% safety margin."""
    ancestor = path.expanduser().resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    available = shutil.disk_usage(ancestor).free
    recommended = int(required_bytes * 1.1)
    if available < recommended:
        raise RuntimeError(
            f"insufficient disk space for activation dataset: need about "
            f"{recommended / 1024**3:.1f} GiB including margin, but only "
            f"{available / 1024**3:.1f} GiB is available under {ancestor}"
        )
    return recommended, available


def _dtype_name(dtype: torch.dtype) -> str:
    for name, candidate in _DTYPES.items():
        if dtype == candidate:
            return name
    raise ValueError(f"unsupported activation dtype: {dtype}")


def _validate_manifest(value: Mapping[str, Any]) -> None:
    if value.get("format") != FORMAT or value.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported activation dataset format")
    if value.get("dtype") not in _DTYPES:
        raise ValueError(f"unsupported activation dataset dtype: {value.get('dtype')!r}")
    if not isinstance(value.get("layers"), dict) or not value["layers"]:
        raise ValueError("activation dataset has no layers")


def _validate_compatible(existing: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    _validate_manifest(existing)
    keys = (
        "format",
        "format_version",
        "model",
        "activation_site",
        "layer_indexing",
        "sample_count",
        "hidden_size",
        "dtype",
        "provenance",
    )
    mismatches = [key for key in keys if existing.get(key) != requested.get(key)]
    if set(existing["layers"]) != set(requested["layers"]):
        mismatches.append("layers")
    if mismatches:
        raise ValueError(
            f"existing activation dataset is incompatible ({', '.join(mismatches)}): "
            f"{MANIFEST_NAME}"
        )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
