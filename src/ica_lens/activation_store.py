from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


def load_activation_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def activation_layers(manifest: dict[str, Any]) -> list[str]:
    layers = manifest["capture"]["layers"]
    if not isinstance(layers, list) or not all(isinstance(layer, str) for layer in layers):
        raise ValueError("Manifest does not contain a valid capture.layers list.")
    return layers


def iter_layer_shards(
    *,
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: str,
    map_location: str | torch.device = "cpu",
) -> Iterator[tuple[int, torch.Tensor]]:
    for shard in layer_shard_records(manifest, layer):
        shard_index = int(shard["index"])
        layer_path = shard["layers"].get(layer)
        if not isinstance(layer_path, str):
            raise KeyError(f"Layer {layer!r} missing from shard {shard_index}.")
        tensor = torch.load(capture_dir / layer_path, map_location=map_location)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor in {layer_path}, got {type(tensor).__name__}.")
        yield shard_index, tensor


def iter_layer_shards_with_metadata(
    *,
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: str,
    map_location: str | torch.device = "cpu",
) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
    for shard in layer_shard_records(manifest, layer):
        layer_path = shard["layers"].get(layer)
        if not isinstance(layer_path, str):
            raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}.")
        tensor = torch.load(capture_dir / layer_path, map_location=map_location)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor in {layer_path}, got {type(tensor).__name__}.")
        yield shard, tensor


def layer_shard_records(manifest: dict[str, Any], layer: str) -> list[dict[str, Any]]:
    layer_shards = manifest.get("layer_shards")
    if isinstance(layer_shards, dict) and isinstance(layer_shards.get(layer), list):
        return [dict(shard) for shard in layer_shards[layer]]
    return [dict(shard) for shard in manifest["shards"]]
