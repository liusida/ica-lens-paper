#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "ica_lens").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Could not find v6 project root.")


V6_ROOT = _project_root()
V6_SRC = V6_ROOT / "src"
if str(V6_SRC) not in sys.path:
    sys.path.insert(0, str(V6_SRC))

from ica_lens.activation_store import iter_layer_shards, layer_shard_records, load_activation_manifest  # noqa: E402
from ica_lens.config import load_toml  # noqa: E402
from ica_lens.paths import CONFIGS_DIR, RESULTS_DIR  # noqa: E402
from ica_lens.saebench_adapters import (  # noqa: E402
    _decoder_weight,
    _load_sae_weights,
    _resolve_sae_weights_path,
    canonical_ica_preprocess,
)


DEFAULT_MODELS = ("gpt2", "gemma2_2b", "qwen3_5_2b_base")
DEFAULT_ACTIVATION_ROOT = RESULTS_DIR / "reproduced" / "activations"
DEFAULT_ICA_ROOT = RESULTS_DIR / "reproduced" / "ica"
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "reproduced" / "nongaussianity"
NORM_EPS = 1e-12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute projection excess kurtosis for ICA, SAE decoder, and random directions."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=DEFAULT_MODELS)
    parser.add_argument("--layers", nargs="*", default=None, help="Layer names like layer_06, or integer ids.")
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--families", nargs="+", default=["ica", "sae", "random"], choices=("ica", "sae", "random"))
    parser.add_argument("--random-directions", type=int, default=32768)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--direction-chunk-size", type=int, default=1024)
    parser.add_argument("--activation-chunk-rows", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--row-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    started_at = time.time()
    results: dict[str, Any] = {}
    for model_name in args.models:
        results[model_name] = _process_model(model_name=str(model_name), args=args)

    manifest_path = args.output_root.resolve() / "manifest.json"
    _write_json(
        manifest_path,
        {
            "analysis": "projection_nongaussianity",
            "models": list(args.models),
            "settings": {
                "token_budget": int(args.token_budget),
                "activation_root": str(args.activation_root.resolve()),
                "ica_root": str(args.ica_root.resolve()),
                "families": list(dict.fromkeys(args.families)),
                "layers": args.layers,
                "random_directions": int(args.random_directions),
                "random_seed": int(args.random_seed),
                "direction_chunk_size": int(args.direction_chunk_size),
                "activation_chunk_rows": int(args.activation_chunk_rows),
                "max_tokens": args.max_tokens,
                "device": str(_resolve_device(str(args.device))),
                "row_normalize": bool(args.row_normalize),
            },
            "results": results,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
    )
    print(f"wrote {manifest_path}")
    return 0


def _process_model(*, model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = CONFIGS_DIR / "comparisons" / f"{model_name}.toml"
    cfg = load_toml(cfg_path)
    hidden_size = int(cfg["model"]["hidden_size"])
    manifest_path = args.activation_root.resolve() / f"{model_name}_tok{args.token_budget}" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing activation manifest: {manifest_path}")
    manifest = load_activation_manifest(manifest_path)
    layers = _selected_layers(args.layers, manifest)
    output_dir = args.output_root.resolve() / model_name
    layer_cache_dir = output_dir / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_cache_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(args.device))
    families = list(dict.fromkeys(args.families))
    all_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, Any]] = []
    layer_results: dict[str, Any] = {}
    for layer in layers:
        layer_results[layer] = {}
        for family in families:
            cache_path = layer_cache_dir / layer / f"{family}.csv"
            metadata_path = cache_path.with_suffix(".json")
            if cache_path.exists() and not args.force:
                rows = _read_csv(cache_path)
                summary = _summary_from_values(torch.tensor([float(row["excess_kurtosis"]) for row in rows]))
            else:
                directions, projection_mean, source = _load_directions(
                    family=family,
                    cfg=cfg,
                    ica_root=args.ica_root.resolve(),
                    model_name=model_name,
                    token_budget=int(args.token_budget),
                    layer=layer,
                    hidden_size=hidden_size,
                    random_directions=int(args.random_directions),
                    random_seed=int(args.random_seed),
                )
                values = _projection_excess_kurtosis(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    layer=layer,
                    directions=directions,
                    projection_mean=projection_mean,
                    device=device,
                    direction_chunk_size=int(args.direction_chunk_size),
                    activation_chunk_rows=int(args.activation_chunk_rows),
                    max_tokens=args.max_tokens,
                    row_normalize=bool(args.row_normalize),
                    desc=f"{model_name} {layer} {family}",
                )
                rows = [
                    {
                        "model": model_name,
                        "layer": layer,
                        "family": family,
                        "direction_index": str(index),
                        "excess_kurtosis": f"{float(value):.9g}",
                    }
                    for index, value in enumerate(values.tolist())
                ]
                summary = _summary_from_values(values)
                _write_csv(cache_path, rows, ["model", "layer", "family", "direction_index", "excess_kurtosis"])
                _write_json(
                    metadata_path,
                    {
                        "analysis": "projection_excess_kurtosis",
                        "model": model_name,
                        "layer": layer,
                        "family": family,
                        "n_directions": int(directions.shape[0]),
                        "hidden_size": int(directions.shape[1]),
                        "source": source,
                        "summary": summary,
                    },
                )
                print(f"wrote {cache_path}")
                print(f"wrote {metadata_path}")
            all_rows.extend(rows)
            summary_row = {"model": model_name, "layer": layer, "family": family, **summary}
            summary_rows.append(summary_row)
            layer_results[layer][family] = summary_row

    direction_table = output_dir / "direction_excess_kurtosis.csv"
    summary_table = output_dir / "layer_summary.csv"
    manifest_out = output_dir / "manifest.json"
    _write_csv(direction_table, all_rows, ["model", "layer", "family", "direction_index", "excess_kurtosis"])
    _write_csv(summary_table, summary_rows)
    _write_json(
        manifest_out,
        {
            "analysis": "projection_nongaussianity",
            "model": model_name,
            "config": str(cfg_path.resolve()),
            "activation_manifest": str(manifest_path),
            "layers": layers,
            "files": {
                "direction_excess_kurtosis": str(direction_table),
                "layer_summary": str(summary_table),
            },
            "results": layer_results,
        },
    )
    print(f"wrote {direction_table}")
    print(f"wrote {summary_table}")
    print(f"wrote {manifest_out}")
    return {"files": {"direction_excess_kurtosis": str(direction_table), "layer_summary": str(summary_table), "manifest": str(manifest_out)}}


def _load_directions(
    *,
    family: str,
    cfg: dict[str, Any],
    ica_root: Path,
    model_name: str,
    token_budget: int,
    layer: str,
    hidden_size: int,
    random_directions: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if family == "ica":
        artifact_path = _ica_artifact_path(ica_root=ica_root, model_name=model_name, token_budget=token_budget, layer=layer)
        metadata_path = artifact_path.with_suffix(".json")
        blob = torch.load(artifact_path, map_location="cpu")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else blob.get("metadata", {})
        preprocess = canonical_ica_preprocess(metadata.get("preprocess"))
        if preprocess != "with_normalization":
            raise ValueError(f"Expected with_normalization ICA artifact for nongaussianity, got {preprocess}: {artifact_path}")
        components = _normalize_rows(blob["tensors"]["components"].to(dtype=torch.float32))
        mean = blob["tensors"]["mean"].to(dtype=torch.float32)
        if mean.dim() == 1:
            mean = mean.reshape(1, -1)
        return components.contiguous(), mean.contiguous(), {
            "artifact": str(artifact_path),
            "metadata": str(metadata_path),
            "direction_source": "ica_components_unit_normalized",
            "preprocess": preprocess,
        }
    if family == "sae":
        sae_cfg = dict(cfg["sae"])
        weights_path = _resolve_sae_weights_path(sae_config=sae_cfg, layer=_layer_index(layer))
        weights = _load_sae_weights(weights_path, sae_config=sae_cfg)
        decoder = _decoder_weight(weights, hidden_size=hidden_size, preferred_key=sae_cfg.get("decoder_key")).to(dtype=torch.float32)
        return _normalize_rows(decoder).contiguous(), None, {
            "weights": str(weights_path),
            "repo_id": sae_cfg.get("repo_id"),
            "direction_source": "sae_decoder_rows_unit_normalized",
        }
    if family == "random":
        layer_seed = int(random_seed) + 1009 * _layer_index(layer)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(layer_seed)
        directions = torch.randn((random_directions, hidden_size), generator=generator, dtype=torch.float32)
        return _normalize_rows(directions).contiguous(), None, {
            "random_seed": int(random_seed),
            "layer_seed": int(layer_seed),
            "direction_source": "gaussian_random_unit_rows",
        }
    raise ValueError(f"Unsupported family: {family}")


def _projection_excess_kurtosis(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    layer: str,
    directions: torch.Tensor,
    projection_mean: torch.Tensor | None,
    device: torch.device,
    direction_chunk_size: int,
    activation_chunk_rows: int,
    max_tokens: int | None,
    row_normalize: bool,
    desc: str,
) -> torch.Tensor:
    directions = directions.to(dtype=torch.float32).contiguous()
    n_directions = int(directions.shape[0])
    sum1 = torch.zeros(n_directions, dtype=torch.float64, device=device)
    sum2 = torch.zeros_like(sum1)
    sum3 = torch.zeros_like(sum1)
    sum4 = torch.zeros_like(sum1)
    count = 0
    mean_device = None if projection_mean is None else projection_mean.to(device=device, dtype=torch.float32)
    direction_ranges = list(range(0, n_directions, direction_chunk_size))
    total = max_tokens or sum(int(shard.get("tokens", 0)) for shard in layer_shard_records(manifest, layer))
    pbar = tqdm(total=total, unit="tok", dynamic_ncols=True, desc=desc)
    try:
        for _shard_index, activations in iter_layer_shards(capture_dir=manifest_path.parent, manifest=manifest, layer=layer):
            remaining = None if max_tokens is None else max_tokens - count
            if remaining is not None and remaining <= 0:
                break
            take = int(activations.shape[0]) if remaining is None else min(remaining, int(activations.shape[0]))
            for activation_chunk in torch.split(activations[:take], activation_chunk_rows, dim=0):
                x = activation_chunk.to(device=device, dtype=torch.float32, non_blocking=True)
                if row_normalize:
                    x = _normalize_rows(x)
                if mean_device is not None:
                    x = x - mean_device
                rows = int(x.shape[0])
                count += rows
                for start in direction_ranges:
                    end = min(start + direction_chunk_size, n_directions)
                    chunk = directions[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
                    values = (x @ chunk.T).to(dtype=torch.float64)
                    sum1[start:end] += values.sum(dim=0)
                    sum2[start:end] += values.square().sum(dim=0)
                    sum3[start:end] += values.pow(3).sum(dim=0)
                    sum4[start:end] += values.pow(4).sum(dim=0)
            pbar.update(take)
    finally:
        pbar.close()
    if count <= 0:
        raise RuntimeError(f"No activation rows found for {layer}.")
    mean = sum1 / count
    second = sum2 / count
    third = sum3 / count
    fourth = sum4 / count
    variance = (second - mean.square()).clamp_min(1e-30)
    central_fourth = fourth - 4 * mean * third + 6 * mean.square() * second - 3 * mean.pow(4)
    return (central_fourth / variance.square() - 3.0).detach().to(device="cpu", dtype=torch.float32)


def _selected_layers(values: list[str] | None, manifest: dict[str, Any]) -> list[str]:
    available = list(manifest["capture"]["layers"])
    if values:
        layers = [_layer_name(value) for value in values]
        missing = sorted(set(layers) - set(available))
        if missing:
            raise ValueError(f"Requested layer(s) not found in manifest: {', '.join(missing)}")
        return layers
    return [layer for layer in available if layer != "embedding"]


def _ica_artifact_path(*, ica_root: Path, model_name: str, token_budget: int, layer: str) -> Path:
    candidates = [
        ica_root / f"{model_name}_tok{token_budget}" / f"{layer}_fastica.pt",
        ica_root / model_name / f"{layer}_fastica.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _layer_name(value: str) -> str:
    return value if value.startswith("layer_") else f"layer_{int(value):02d}"


def _layer_index(layer: str) -> int:
    return int(layer.removeprefix("layer_"))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return torch.device(value)


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(NORM_EPS)


def _summary_from_values(values: torch.Tensor) -> dict[str, float | int]:
    x = values.detach().cpu().to(dtype=torch.float64)
    q = torch.quantile(x, torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=torch.float64))
    return {
        "n": int(x.numel()),
        "min": float(x.min().item()),
        "p01": float(q[0].item()),
        "p05": float(q[1].item()),
        "p25": float(q[2].item()),
        "median": float(q[3].item()),
        "p75": float(q[4].item()),
        "p95": float(q[5].item()),
        "p99": float(q[6].item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
