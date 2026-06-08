#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ica_lens.activation_store import activation_layers, iter_layer_shards, layer_shard_records, load_activation_manifest
from ica_lens.config import load_toml
from ica_lens.fastica import (
    VENDOR_FASTICA_SRC,
    fit_fastica,
    fit_fastica_without_normalization,
    save_ica_artifact,
)
from ica_lens.models import resolve_device, torch_dtype
from ica_lens.paths import CONFIGS_DIR, RESULTS_DIR, V6_ROOT


DEFAULT_CONFIG = CONFIGS_DIR / "fit_ica" / "gpt2.toml"
DEFAULT_ACTIVATION_ROOT = RESULTS_DIR / "reproduced" / "activations"
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "reproduced" / "ica"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit FastICA models for ICA Lens reproduction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--activation-manifest", type=Path, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preprocess", choices=("with_normalization", "without_normalization"), default=None)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--fit-rows", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--algorithm", choices=("parallel", "deflation"), default=None)
    parser.add_argument("--n-components", default=None)
    parser.add_argument("--nonlinearity", default=None)
    parser.add_argument("--norm-eps", type=float, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    started_at = time.time()
    cfg = load_toml(args.config)
    activation_cfg = _load_activation_config(args.config, cfg)
    fit_cfg = dict(cfg.get("fit", {}))

    preprocess = str(args.preprocess or fit_cfg.get("preprocess", "with_normalization"))
    token_budget = int(args.token_budget or activation_cfg["activation"]["token_budget"])
    manifest_path = args.activation_manifest or _activation_manifest_path(
        activation_root=args.activation_root,
        activation_cfg=activation_cfg,
        token_budget=token_budget,
    )
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing activation manifest: {manifest_path}. Capture activations first.")

    manifest = load_activation_manifest(manifest_path)
    model_short_name = str(manifest["model"]["short_name"])
    run_dir = args.output_root.resolve() / f"{model_short_name}_tok{token_budget}"
    run_dir.mkdir(parents=True, exist_ok=True)

    layers = _resolve_layers(args.layers, fit_cfg.get("layers"), manifest)
    device = resolve_device(str(args.device))
    dtype = torch_dtype(str(args.dtype or fit_cfg.get("dtype", "float32")))
    hidden_size = int(manifest["model"]["hidden_size"])
    n_components = _n_components(args.n_components or fit_cfg.get("n_components", "hidden_dim"), hidden_size)
    fit_rows = args.fit_rows

    summaries: dict[str, Any] = {}
    for layer in layers:
        output_path = run_dir / f"{layer}_fastica.pt"
        if output_path.exists() and not args.force:
            print(f"skipped {layer}: existing artifact at {output_path}")
            summaries[layer] = _load_metadata(output_path.with_suffix(".json"))
            continue

        activations = _load_layer_activations(
            capture_dir=manifest_path.parent,
            manifest=manifest,
            layer=layer,
            fit_rows=fit_rows,
            device=device,
            dtype=dtype,
        )
        if n_components > min(int(activations.shape[0]), int(activations.shape[1])):
            raise ValueError(
                f"n_components={n_components} is too large for {layer} activations with shape {tuple(activations.shape)}."
            )

        layer_index = activation_layers(manifest).index(layer)
        layer_seed = int(args.seed if args.seed is not None else fit_cfg.get("seed", 0)) + layer_index
        fit_fn = fit_fastica_without_normalization if preprocess == "without_normalization" else fit_fastica
        artifact = fit_fn(
            activations,
            n_components=n_components,
            seed=layer_seed,
            max_iter=int(args.max_iter if args.max_iter is not None else fit_cfg.get("max_iter", 1000)),
            tol=float(args.tol if args.tol is not None else fit_cfg.get("tol", 1e-4)),
            algorithm=str(args.algorithm or fit_cfg.get("algorithm", "parallel")),
            fun=str(args.nonlinearity or fit_cfg.get("nonlinearity", "logcosh")),
            norm_eps=float(args.norm_eps if args.norm_eps is not None else fit_cfg.get("norm_eps", 1e-12)),
            progress=bool(args.progress if args.progress is not None else fit_cfg.get("progress", True)),
        )
        metadata = {
            "layer": layer,
            "activation_manifest": str(manifest_path),
            "activation_shape": list(activations.shape),
            "artifact_format": "torch_save_tensors_and_metadata",
            "config": str(args.config.resolve()),
            "fastica_torch": _vendor_report(),
            "component_id_convention": "Component id is the row index in saved FastICA tensors.",
        }
        save_ica_artifact(output_path, artifact, metadata)
        summaries[layer] = _load_metadata(output_path.with_suffix(".json"))
        del activations, artifact
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_manifest = {
        "artifact": f"v6_reproduced_ica_{preprocess}_{model_short_name}_tok{token_budget}",
        "model": manifest["model"],
        "activation_manifest": str(manifest_path),
        "config": str(args.config.resolve()),
        "output_dir": str(run_dir),
        "layers": layers,
        "settings": {
            "preprocess": preprocess,
            "token_budget": token_budget,
            "fit_rows": fit_rows,
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "n_components": n_components,
            "seed": int(args.seed if args.seed is not None else fit_cfg.get("seed", 0)),
            "layer_seed_policy": "base_seed + layer_index_in_activation_manifest",
            "max_iter": int(args.max_iter if args.max_iter is not None else fit_cfg.get("max_iter", 1000)),
            "tol": float(args.tol if args.tol is not None else fit_cfg.get("tol", 1e-4)),
            "algorithm": str(args.algorithm or fit_cfg.get("algorithm", "parallel")),
            "nonlinearity": str(args.nonlinearity or fit_cfg.get("nonlinearity", "logcosh")),
        },
        "fastica_torch": _vendor_report(),
        "environment": _environment_report(),
        "layer_summaries": summaries,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote ICA artifacts and manifest: {run_dir}")
    return 0


def _load_activation_config(config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    rel = str(dict(cfg.get("fit", {})).get("activation_config", ""))
    if not rel:
        raise ValueError("ICA config must set [fit].activation_config.")
    path = Path(rel)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return load_toml(path)


def _activation_manifest_path(*, activation_root: Path, activation_cfg: dict[str, Any], token_budget: int) -> Path:
    model_cfg = _load_model_config_from_activation(activation_cfg)
    run_name = str(model_cfg["model"]["short_name"])
    return activation_root.resolve() / f"{run_name}_tok{token_budget}" / "manifest.json"


def _load_model_config_from_activation(activation_cfg: dict[str, Any]) -> dict[str, Any]:
    activation = dict(activation_cfg["activation"])
    path = Path(str(activation["model_config"]))
    if not path.is_absolute():
        path = (CONFIGS_DIR / "activations" / path).resolve()
    return load_toml(path)


def _resolve_layers(cli_layers: list[str] | None, config_layers: object, manifest: dict[str, Any]) -> list[str]:
    all_layers = activation_layers(manifest)
    if cli_layers is not None:
        layers = cli_layers
    elif config_layers is None or str(config_layers) == "all_hidden":
        layers = [layer for layer in all_layers if layer != "embedding"]
    elif str(config_layers) == "all":
        layers = all_layers
    elif isinstance(config_layers, list):
        layers = [str(layer) for layer in config_layers]
    else:
        layers = [str(config_layers)]
    missing = [layer for layer in layers if layer not in all_layers]
    if missing:
        raise ValueError(f"Layer(s) not found in activation manifest: {', '.join(missing)}")
    return layers


def _n_components(value: object, hidden_size: int) -> int:
    if str(value) == "hidden_dim":
        return hidden_size
    n_components = int(value)
    if n_components <= 0 or n_components > hidden_size:
        raise ValueError(f"n_components must be in [1, {hidden_size}], got {n_components}.")
    return n_components


def _load_layer_activations(
    *,
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: str,
    fit_rows: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    rows_seen = 0
    total = fit_rows or sum(int(shard.get("tokens", 0)) for shard in layer_shard_records(manifest, layer))
    pbar = tqdm(total=total, unit="tok", dynamic_ncols=True, desc=f"load {layer}")
    try:
        for _shard_index, activations in iter_layer_shards(capture_dir=capture_dir, manifest=manifest, layer=layer):
            remaining = None if fit_rows is None else fit_rows - rows_seen
            if remaining is not None and remaining <= 0:
                break
            take = int(activations.shape[0]) if remaining is None else min(remaining, int(activations.shape[0]))
            chunks.append(activations[:take].to(device=device, dtype=dtype, non_blocking=True))
            rows_seen += take
            pbar.update(take)
    finally:
        pbar.close()
    if not chunks:
        raise RuntimeError(f"No activations loaded for {layer}.")
    return torch.cat(chunks, dim=0)


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _vendor_report() -> dict[str, Any]:
    return {
        "path": str((V6_ROOT / "vendor" / "FastICA_torch").resolve()),
        "commit": _git_commit(V6_ROOT / "vendor" / "FastICA_torch"),
        "import_path": str(VENDOR_FASTICA_SRC),
    }


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment_report() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": _package_version("numpy"),
        "fastica_torch": _package_version("fastica_torch"),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
