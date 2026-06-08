from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ica_lens.activation_store import activation_layers, iter_layer_shards, load_activation_manifest
from ica_lens.models import resolve_device, torch_dtype
from ica_lens.saebench_adapters import canonical_ica_preprocess


def v6_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "ica_lens").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Could not find v6 root.")


V6_ROOT = v6_root()
DEFAULT_OUTPUT_ROOT = V6_ROOT / "results" / "demo" / "saebench_sparse_probe"
DEFAULT_ACTIVATION_ROOT = V6_ROOT / "results" / "demo" / "activations"
DEFAULT_ICA_ROOT = V6_ROOT / "artifacts" / "fetched" / "models"
DEFAULT_SAEBENCH_ARTIFACTS = V6_ROOT / "results" / "demo" / "saebench_artifacts"
SAEBENCH_ROOT = V6_ROOT / "vendor" / "SAEBench"
SAEBENCH_PYTHON = SAEBENCH_ROOT / ".venv" / "bin" / "python"
QWEN35_SAEBENCH_ROOT = V6_ROOT / "vendor" / "SAEBench-qwen35"
QWEN35_SAEBENCH_PYTHON = QWEN35_SAEBENCH_ROOT / ".venv" / "bin" / "python"
CONFIG_ROOT = V6_ROOT / "configs" / "comparisons"
MATRYOSHKA_REPO_ID = "chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40"
MATRYOSHKA_VARIANT = "snap"
MATRYOSHKA_LAYER = 12
MATRYOSHKA_HOOK = "blocks.12.hook_resid_post"
MATRYOSHKA_WIDTHS = (128, 512)
DEFAULT_K_VALUES = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_DATASETS = [
    "LabHC/bias_in_bios_class_set1",
    "LabHC/bias_in_bios_class_set2",
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "canrager/amazon_reviews_mcauley_1and5_sentiment",
    "codeparrot/github-code",
    "fancyzhx/ag_news",
    "Helsinki-NLP/europarl",
]
MODEL_DIRS = {
    "gpt2": "gpt2_tok3000",
    "gemma2_2b": "gemma2_2b_tok3000",
    "qwen3_5_2b_base": "qwen3_5_2b_base_tok3000",
}
MODEL_LAYER_SPECS = [
    ("gpt2", "layer_06"),
    ("gpt2", "layer_10"),
    ("gemma2_2b", "layer_12"),
    ("gemma2_2b", "layer_20"),
    ("qwen3_5_2b_base", "layer_12"),
    ("qwen3_5_2b_base", "layer_20"),
]
CORE_METHODS = ("pca", "ica", "sae_baseline", "itda")
ALL_METHODS = CORE_METHODS + ("matryoshka_128", "matryoshka_512")
METHOD_LABELS = {
    "pca_two_sign": "PCA",
    "ica_two_sign": "ICA",
    "sae_baseline": "SAE",
    "itda_two_sign": "ITDA",
    "matryoshka_128": "SAE-Matryoshka-128",
    "matryoshka_512": "SAE-Matryoshka-512",
}
METHOD_COLORS = {
    "pca_two_sign": "#5B8C6A",
    "ica_two_sign": "#3D5F99",
    "sae_baseline": "#B45F4D",
    "itda_two_sign": "#6F6F6F",
    "matryoshka_128": "#8B6BBE",
    "matryoshka_512": "#D49A2A",
}
METHOD_ORDER = ("ica_two_sign", "sae_baseline", "itda_two_sign", "matryoshka_512", "matryoshka_128", "pca_two_sign")
CORE_METHOD_ORDER = ("ica_two_sign", "sae_baseline", "itda_two_sign", "pca_two_sign")
SPARSE_PROBE_MARKERS = {
    "ica_two_sign": "^",
    "pca_two_sign": "^",
    "itda_two_sign": "s",
    "sae_baseline": "o",
    "matryoshka_128": "o",
    "matryoshka_512": "o",
}
MODEL_LABELS = {
    "gpt2": "GPT-2 Small",
    "gemma2_2b": "Gemma 2 2B",
    "qwen3_5_2b_base": "Qwen 3.5 2B Base",
}


@dataclass(frozen=True)
class RunTarget:
    model: str
    layer: str
    method: str


def canonical_layer(layer: str | int) -> str:
    if isinstance(layer, str) and layer.startswith("layer_"):
        return f"layer_{int(layer.removeprefix('layer_')):02d}"
    return f"layer_{int(layer):02d}"


def official_targets(*, models: list[str] | None = None, layers: list[str] | None = None, methods: list[str] | None = None) -> list[RunTarget]:
    selected_methods = list(methods or ALL_METHODS)
    if "all" in selected_methods:
        selected_methods = list(ALL_METHODS)
    selected_models = set(models or sorted(MODEL_DIRS))
    selected_layers = {canonical_layer(layer) for layer in layers} if layers else None
    targets: list[RunTarget] = []
    for model, layer in MODEL_LAYER_SPECS:
        if model not in selected_models:
            continue
        if selected_layers is not None and layer not in selected_layers:
            continue
        for method in selected_methods:
            if method in {"matryoshka_128", "matryoshka_512"} and (model, layer) != ("gemma2_2b", "layer_12"):
                continue
            targets.append(RunTarget(model=model, layer=layer, method=method))
    return targets


def artifact_root(output_root: Path) -> Path:
    return output_root / "artifacts"


def pca_prefix(output_root: Path, model: str, layer: str) -> Path:
    return artifact_root(output_root) / "pca" / model / f"{layer}_pca"


def itda_name(layer: str, *, k: int, tau: float, max_atoms: int) -> str:
    tau_text = f"{tau:g}".replace(".", "p").replace("-", "m")
    return f"{layer}_itda_k{k}_tau{tau_text}_atoms{max_atoms}"


def itda_prefix(output_root: Path, model: str, layer: str, *, k: int = 40, tau: float = 4e-4, max_atoms: int = 4096) -> Path:
    return artifact_root(output_root) / "itda" / model / itda_name(layer, k=k, tau=tau, max_atoms=max_atoms)


def matryoshka_checkpoint_dir(output_root: Path) -> Path:
    return artifact_root(output_root) / "matryoshka" / "gemma2_2b" / "layer_12" / MATRYOSHKA_VARIANT / MATRYOSHKA_HOOK


def activation_manifest_path(activation_root: Path, model: str, *, token_budget: int | None = None) -> Path:
    if token_budget is not None:
        return activation_root / f"{model}_tok{token_budget}" / "manifest.json"
    return activation_root / MODEL_DIRS[model] / "manifest.json"


def load_model_config(model: str) -> dict[str, Any]:
    return tomllib.loads((CONFIG_ROOT / f"{model}.toml").read_text(encoding="utf-8"))


def saebench_env(model: str) -> tuple[Path, Path]:
    if model == "qwen3_5_2b_base":
        return QWEN35_SAEBENCH_ROOT, QWEN35_SAEBENCH_PYTHON
    return SAEBENCH_ROOT, SAEBENCH_PYTHON


def ensure_paths_for_saebench(saebench_root: Path) -> None:
    for path in (V6_ROOT / "src", saebench_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def prepare_dry_run(*, output_root: Path, activation_root: Path, targets: list[RunTarget], itda_k: int, itda_tau: float, itda_max_atoms: int) -> list[dict[str, object]]:
    rows = []
    seen_artifacts: set[tuple[str, str, str]] = set()
    for target in targets:
        if target.method == "pca" and (target.model, target.layer, "pca") not in seen_artifacts:
            seen_artifacts.add((target.model, target.layer, "pca"))
            rows.append({"action": "fit_pca", "model": target.model, "layer": target.layer, "path": str(pca_prefix(output_root, target.model, target.layer).with_suffix(".pt"))})
        if target.method == "itda" and (target.model, target.layer, "itda") not in seen_artifacts:
            seen_artifacts.add((target.model, target.layer, "itda"))
            rows.append({"action": "train_itda", "model": target.model, "layer": target.layer, "path": str(itda_prefix(output_root, target.model, target.layer, k=itda_k, tau=itda_tau, max_atoms=itda_max_atoms).with_suffix(".json"))})
    if any(target.method.startswith("matryoshka") for target in targets):
        rows.append({"action": "download_matryoshka", "model": "gemma2_2b", "layer": "layer_12", "path": str(matryoshka_checkpoint_dir(output_root))})
    rows.append({"action": "activation_root", "path": str(activation_root)})
    return rows


def prepare_artifacts(
    *,
    output_root: Path,
    activation_root: Path,
    targets: list[RunTarget],
    itda_k: int = 40,
    itda_tau: float = 4e-4,
    itda_max_atoms: int = 4096,
    itda_batch_size: int = 1024,
    itda_max_train_tokens: int = 1_000_000,
    seed: int = 0,
    device: str = "auto",
    force: bool = False,
    token_budget: int | None = None,
    ica_root: Path = DEFAULT_ICA_ROOT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"analysis": "official_sparse_probe_baseline_artifact_prep", "artifacts": []}
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        if target.method == "pca" and (target.model, target.layer, "pca") not in seen:
            seen.add((target.model, target.layer, "pca"))
            metadata = fit_pca_artifact(
                model=target.model,
                layer=target.layer,
                activation_root=activation_root,
                output_prefix=pca_prefix(output_root, target.model, target.layer),
                ica_root=ica_root,
                token_budget=token_budget,
                force=force,
            )
            manifest["artifacts"].append(metadata)
        if target.method == "itda" and (target.model, target.layer, "itda") not in seen:
            seen.add((target.model, target.layer, "itda"))
            metadata = train_itda_artifact(
                model=target.model,
                layer=target.layer,
                activation_root=activation_root,
                output_prefix=itda_prefix(output_root, target.model, target.layer, k=itda_k, tau=itda_tau, max_atoms=itda_max_atoms),
                token_budget=token_budget,
                k=itda_k,
                loss_threshold=itda_tau,
                max_atoms=itda_max_atoms,
                batch_size=itda_batch_size,
                max_train_tokens=itda_max_train_tokens,
                seed=seed,
                device=device,
                force=force,
            )
            manifest["artifacts"].append(metadata)
    if any(target.method.startswith("matryoshka") for target in targets):
        checkpoint_dir = download_matryoshka_checkpoint(output_root=artifact_root(output_root) / "matryoshka" / "gemma2_2b" / "layer_12")
        manifest["artifacts"].append({"method": "matryoshka", "checkpoint_dir": str(checkpoint_dir), "repo_id": MATRYOSHKA_REPO_ID})
    manifest["created_at_unix"] = time.time()
    manifest_path = artifact_root(output_root) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def fit_pca_artifact(
    *,
    model: str,
    layer: str,
    activation_root: Path,
    output_prefix: Path,
    preprocess: str = "with_normalization",
    ica_root: Path = DEFAULT_ICA_ROOT,
    token_budget: int | None = None,
    device: str = "auto",
    dtype_name: str = "float32",
    covariance_dtype_name: str = "float64",
    fit_rows: int | None = None,
    norm_eps: float = 1e-12,
    force: bool = False,
) -> dict[str, Any]:
    started_at = time.time()
    output_path = output_prefix.with_suffix(".pt")
    if output_path.exists() and not force:
        return json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    manifest_path = activation_manifest_path(activation_root, model, token_budget=token_budget)
    manifest = load_activation_manifest(manifest_path)
    if layer not in activation_layers(manifest):
        raise ValueError(f"Layer {layer!r} not found in {manifest_path}")
    hidden_size = int(manifest["model"]["hidden_size"])
    n_components = _matched_ica_components(model=model, layer=layer, ica_root=ica_root, token_budget=token_budget)
    device_obj = resolve_device(device)
    dtype = torch_dtype(dtype_name)
    covariance_dtype = torch.float64 if covariance_dtype_name == "float64" else torch.float32
    mean, rows = _stream_mean(
        capture_dir=manifest_path.parent,
        manifest=manifest,
        layer=layer,
        preprocess=preprocess,
        fit_rows=fit_rows,
        device=device_obj,
        dtype=dtype,
        covariance_dtype=covariance_dtype,
        norm_eps=norm_eps,
    )
    covariance = _stream_covariance(
        capture_dir=manifest_path.parent,
        manifest=manifest,
        layer=layer,
        mean=mean,
        preprocess=preprocess,
        fit_rows=fit_rows,
        device=device_obj,
        dtype=dtype,
        covariance_dtype=covariance_dtype,
        norm_eps=norm_eps,
    )
    eigvals, eigvecs = torch.linalg.eigh(covariance.cpu())
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals.index_select(0, order)
    eigvecs = eigvecs.index_select(1, order)
    components = eigvecs[:, :n_components].T.contiguous().to(torch.float32)
    explained = eigvals[:n_components].to(torch.float64)
    total_variance = torch.clamp(eigvals.sum().to(torch.float64), min=0.0)
    tensors = {
        "mean": mean.reshape(1, -1).cpu().to(torch.float32),
        "components": components.cpu(),
        "explained_variance": explained.cpu(),
        "explained_variance_ratio": (explained / total_variance.clamp_min(1e-30)).cpu(),
    }
    metadata = {
        "method": "pca",
        "analysis": "official_sparse_probe_baseline",
        "model": model,
        "layer": layer,
        "activation_manifest": str(manifest_path),
        "preprocess": preprocess,
        "rows": int(rows),
        "hidden_size": hidden_size,
        "n_components": int(n_components),
        "fit_rows": fit_rows,
        "norm_eps": float(norm_eps),
        "covariance_dtype": covariance_dtype_name,
        "component_id_convention": "Component id is rank order by descending covariance eigenvalue.",
        "explained_variance_sum": float(explained.sum().item()),
        "total_variance": float(total_variance.item()),
        "explained_variance_ratio_sum": float((explained / total_variance.clamp_min(1e-30)).sum().item()),
        "environment": environment_report(),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensors": tensors, "metadata": metadata}, output_path)
    output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote PCA artifact: {output_path}")
    return metadata


def _matched_ica_components(*, model: str, layer: str, ica_root: Path, token_budget: int | None = None) -> int:
    candidates = []
    if token_budget is not None:
        candidates.append(ica_root / f"{model}_tok{token_budget}" / f"{layer}_fastica.json")
    candidates.append(ica_root / model / f"{layer}_fastica.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(f"Cannot match ICA component count; missing {path}")
    return int(json.loads(path.read_text(encoding="utf-8"))["n_components"])


def _stream_mean(**kwargs: Any) -> tuple[torch.Tensor, int]:
    total = None
    rows = 0
    for shard_rows in _iter_preprocessed_shards(**kwargs):
        total = shard_rows.sum(dim=0) if total is None else total + shard_rows.sum(dim=0)
        rows += int(shard_rows.shape[0])
    if total is None or rows == 0:
        raise RuntimeError("No activation rows available for PCA.")
    return total / rows, rows


def _stream_covariance(*, mean: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    device = kwargs["device"]
    covariance_dtype = kwargs["covariance_dtype"]
    covariance = torch.zeros((int(mean.shape[0]), int(mean.shape[0])), dtype=covariance_dtype, device=device)
    rows = 0
    mean = mean.to(device=device, dtype=covariance_dtype)
    for shard_rows in _iter_preprocessed_shards(**kwargs):
        centered = shard_rows - mean
        covariance.add_(centered.T @ centered)
        rows += int(centered.shape[0])
    if rows < 2:
        raise RuntimeError(f"Need at least two rows for covariance, got {rows}.")
    return covariance / (rows - 1)


def _iter_preprocessed_shards(
    *,
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: str,
    preprocess: str,
    fit_rows: int | None,
    device: torch.device,
    dtype: torch.dtype,
    covariance_dtype: torch.dtype,
    norm_eps: float,
):
    remaining = fit_rows
    for _shard_index, shard in iter_layer_shards(capture_dir=capture_dir, manifest=manifest, layer=layer, map_location="cpu"):
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and int(shard.shape[0]) > remaining:
            shard = shard[:remaining]
        values = shard.to(device=device, dtype=dtype)
        if preprocess == "with_normalization":
            values = values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(norm_eps)
        elif preprocess != "without_normalization":
            raise ValueError(f"Unsupported preprocess mode: {preprocess!r}")
        values = values.to(dtype=covariance_dtype)
        if remaining is not None:
            remaining -= int(values.shape[0])
        yield values
        del values, shard
        if device.type == "cuda":
            torch.cuda.empty_cache()


def train_itda_artifact(
    *,
    model: str,
    layer: str,
    activation_root: Path,
    output_prefix: Path,
    k: int,
    loss_threshold: float,
    max_atoms: int,
    batch_size: int,
    max_train_tokens: int,
    seed: int,
    device: str,
    force: bool,
    token_budget: int | None = None,
    save_dtype: str = "float32",
) -> dict[str, Any]:
    started_at = time.time()
    atoms_path = output_prefix.parent / f"{output_prefix.name}_atoms.pt"
    sources_path = output_prefix.parent / f"{output_prefix.name}_atom_sources.pt"
    metadata_path = output_prefix.with_suffix(".json")
    if atoms_path.exists() and sources_path.exists() and metadata_path.exists() and not force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    device_obj = resolve_device(device)
    corpus_dir = activation_manifest_path(activation_root, model, token_budget=token_budget).parent
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing activation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if layer not in manifest["capture"]["layers"]:
        raise ValueError(f"Layer {layer!r} is not in manifest layers: {manifest['capture']['layers']}")
    print(f"loading {model} {layer} activations from {corpus_dir}")
    activations, source_columns, shard_records = _load_layer_corpus(corpus_dir, manifest, layer)
    n_rows, d_model = activations.shape
    n_train = min(n_rows, int(max_train_tokens)) if max_train_tokens > 0 else n_rows
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(n_rows, generator=generator)[:n_train]
    print(f"training rows: {n_train:,} sampled from {n_rows:,}; d_model={d_model}; device={device_obj}")
    atoms, atom_order_indices, train_stats = train_itda(
        activations=activations,
        order=order,
        max_atoms=max_atoms,
        k=k,
        loss_threshold=loss_threshold,
        batch_size=batch_size,
        device=device_obj,
    )
    atom_sources = _source_rows_from_global_indices(atom_order_indices.cpu(), source_columns)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(atoms.to(dtype=torch.float32 if save_dtype == "float32" else torch.bfloat16, device="cpu"), atoms_path)
    torch.save(atom_sources, sources_path)
    metadata = {
        "method": "itda",
        "analysis": "official_sparse_probe_baseline",
        "model": model,
        "layer": layer,
        "hook_name": f"blocks.{int(layer.removeprefix('layer_'))}.hook_resid_post",
        "activation_corpus": str(corpus_dir),
        "manifest": str(manifest_path),
        "source_manifest": {"model": manifest.get("model"), "dataset": manifest.get("dataset"), "capture": manifest.get("capture")},
        "shards": shard_records,
        "artifact_files": {"atoms": str(atoms_path), "atom_sources": str(sources_path)},
        "n_atoms": int(atoms.shape[0]),
        "d_model": int(d_model),
        "k": int(k),
        "loss_threshold": float(loss_threshold),
        "max_atoms": int(max_atoms),
        "batch_size": int(batch_size),
        "max_train_tokens": int(max_train_tokens),
        "rows_available": int(n_rows),
        "rows_sampled": int(n_train),
        "seed": int(seed),
        "device": str(device_obj),
        "save_dtype": save_dtype,
        "train_stats": train_stats,
        "environment": environment_report(),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {atoms_path}")
    print(f"wrote {sources_path}")
    print(f"wrote {metadata_path}")
    return metadata


@torch.no_grad()
def train_itda(*, activations: torch.Tensor, order: torch.Tensor, max_atoms: int, k: int, loss_threshold: float, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    n_train = int(order.numel())
    d_model = int(activations.shape[1])
    init_count = min(max_atoms, d_model, n_train)
    init_indices = order[:init_count]
    atoms = _normalize_rows(activations[init_indices].to(device=device, dtype=torch.float32))
    atom_order_indices = init_indices.clone()
    rows_seen = init_count
    added_after_init = 0
    batches = 0
    last_batch_mean_error: float | None = None
    print(f"initialized {init_count} atoms from shuffled activation rows")
    cursor = init_count
    while cursor < n_train and atoms.shape[0] < max_atoms:
        batch_indices = order[cursor : cursor + batch_size]
        x = activations[batch_indices].to(device=device, dtype=torch.float32)
        recon = _matching_pursuit_reconstruct(atoms, x, k=min(k, int(atoms.shape[0])))
        errors = _normalized_mse(x, recon)
        last_batch_mean_error = float(errors.mean().item())
        remaining = max_atoms - int(atoms.shape[0])
        selected = torch.nonzero(errors > loss_threshold, as_tuple=True)[0]
        if selected.numel() > remaining:
            selected = selected[:remaining]
        if selected.numel() > 0:
            atoms = torch.cat([atoms, _normalize_rows(x[selected])], dim=0)
            atom_order_indices = torch.cat([atom_order_indices, batch_indices[selected.cpu()].cpu()])
            added_after_init += int(selected.numel())
        rows_seen += int(batch_indices.numel())
        batches += 1
        if batches % 10 == 0 or atoms.shape[0] >= max_atoms:
            print(f"batch={batches} rows_seen={rows_seen:,} atoms={atoms.shape[0]:,}/{max_atoms:,} mean_error={last_batch_mean_error:.6g} added={added_after_init:,}")
        cursor += batch_size
    return atoms, atom_order_indices, {
        "rows_seen": int(rows_seen),
        "batches": int(batches),
        "initial_atoms": int(init_count),
        "added_after_init": int(added_after_init),
        "stopped_because_full": bool(atoms.shape[0] >= max_atoms),
        "last_batch_mean_error": last_batch_mean_error,
    }


@torch.no_grad()
def _matching_pursuit_reconstruct(atoms: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    residual = x.clone()
    recon = torch.zeros_like(x)
    atoms_t = atoms.T.contiguous()
    for _ in range(k):
        correlations = residual @ atoms_t
        best_atoms = torch.argmax(torch.abs(correlations), dim=1)
        coeffs = correlations[torch.arange(x.shape[0], device=x.device), best_atoms]
        update = coeffs[:, None] * atoms[best_atoms]
        recon += update
        residual -= update
    return recon


def _normalized_mse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    x_norm = x.norm(dim=1, keepdim=True).clamp_min(1e-9)
    recon_norm = recon.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return ((x / x_norm) - (recon / recon_norm)).pow(2).mean(dim=1)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)


def _load_layer_corpus(corpus_dir: Path, manifest: dict[str, Any], layer: str) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, Any]]]:
    activation_parts, input_id_parts, doc_id_parts, position_parts, shard_id_parts, local_index_parts = [], [], [], [], [], []
    shard_records: list[dict[str, Any]] = []
    row_offset = 0
    for shard in manifest["shards"]:
        shard_index = int(shard["index"])
        layer_path = corpus_dir / shard["layers"][layer]
        acts = torch.load(layer_path, map_location="cpu")
        input_ids = torch.load(corpus_dir / shard["input_ids"], map_location="cpu")
        doc_ids = torch.load(corpus_dir / shard["doc_ids"], map_location="cpu")
        positions = torch.load(corpus_dir / shard["positions"], map_location="cpu")
        n = int(acts.shape[0])
        activation_parts.append(acts)
        input_id_parts.append(input_ids.to(dtype=torch.long))
        doc_id_parts.append(doc_ids.to(dtype=torch.long))
        position_parts.append(positions.to(dtype=torch.long))
        shard_id_parts.append(torch.full((n,), shard_index, dtype=torch.long))
        local_index_parts.append(torch.arange(n, dtype=torch.long))
        shard_records.append({"index": shard_index, "rows": n, "row_offset": row_offset, "activation_path": str(layer_path)})
        row_offset += n
        print(f"loaded shard {shard_index}: rows={n:,} dtype={acts.dtype}")
    activations = torch.cat(activation_parts, dim=0)
    source_columns = {
        "global_index": torch.arange(activations.shape[0], dtype=torch.long),
        "shard": torch.cat(shard_id_parts, dim=0),
        "local_index": torch.cat(local_index_parts, dim=0),
        "input_id": torch.cat(input_id_parts, dim=0),
        "doc_id": torch.cat(doc_id_parts, dim=0),
        "position": torch.cat(position_parts, dim=0),
    }
    return activations, source_columns, shard_records


def _source_rows_from_global_indices(indices: torch.Tensor, source_columns: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    indices = indices.to(dtype=torch.long)
    return {key: values[indices].clone() for key, values in source_columns.items()}


def canonical_pca_preprocess(value: str | None) -> str:
    aliases = {"with_normalization": "with_normalization", "row_normalize_then_center": "with_normalization", "without_normalization": "without_normalization", "raw_center_pca": "without_normalization"}
    if value is None or value == "":
        return "with_normalization"
    if value not in aliases:
        raise ValueError(f"Unsupported PCA preprocess mode: {value!r}")
    return aliases[value]


def load_pca_sae(*, artifact_prefix: Path, model_name: str, hook_layer: int, hook_name: str, device: str, dtype: Any, expected_preprocess: str | None = None):
    import torch.nn as nn
    from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig

    class PCASAELike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            saved = torch.load(artifact_prefix.with_suffix(".pt"), map_location="cpu")
            self.metadata = json.loads(artifact_prefix.with_suffix(".json").read_text(encoding="utf-8"))
            self.preprocess = canonical_pca_preprocess(self.metadata.get("preprocess"))
            expected = canonical_pca_preprocess(expected_preprocess)
            if self.preprocess != expected:
                raise ValueError(f"PCA preprocess mismatch for {artifact_prefix}: expected {expected}, got {self.preprocess}")
            tensors = saved["tensors"] if isinstance(saved, dict) and "tensors" in saved else saved
            components = tensors["components"].to(dtype=torch.float32)
            mean = tensors["mean"].to(dtype=torch.float32)
            if mean.dim() == 1:
                mean = mean.reshape(1, -1)
            n_components, d_in = components.shape
            norm_eps = float(self.metadata.get("norm_eps", 1e-12))
            decoder = components.contiguous()
            w_dec = torch.cat([decoder, -decoder], dim=0)
            w_dec = w_dec / w_dec.norm(dim=1, keepdim=True).clamp_min(norm_eps)
            self.W_enc = nn.Parameter(components.T.contiguous())
            self.W_dec = nn.Parameter(w_dec.contiguous())
            self.b_enc = nn.Parameter(torch.zeros(2 * n_components, dtype=torch.float32))
            self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=torch.float32))
            self.register_buffer("components", components.contiguous())
            self.register_buffer("decoder", decoder.contiguous())
            self.register_buffer("mean", mean.contiguous())
            self.device = torch.device(device)
            self.dtype = dtype
            self.norm_eps = norm_eps
            self._last_input_norm = None
            self._last_nonzero_mask = None
            self._last_leading_shape = None
            self.cfg = CustomSAEConfig(model_name=model_name, d_in=int(d_in), d_sae=int(2 * n_components), hook_layer=int(hook_layer), hook_name=hook_name)
            self.cfg.architecture = "pca_two_sign_sae_like"
            self.cfg.activation_fn_str = "relu"
            self.cfg.dtype = dtype.__str__().split(".")[1]
            self.cfg.device = str(self.device)
            self.cfg.pca_artifact_prefix = str(artifact_prefix)
            self.cfg.pca_n_components = int(n_components)
            self.cfg.pca_preprocess = self.preprocess
            self.cfg.pca_decode = "components_add_mean_restore_input_norm" if self.preprocess == "with_normalization" else "components_add_mean"
            self.to(device=self.device, dtype=self.dtype)

        def encode(self, x):
            x = x.to(device=self.components.device, dtype=self.components.dtype)
            self._last_leading_shape = tuple(x.shape[:-1])
            if self.preprocess == "with_normalization":
                norm = x.norm(dim=-1, keepdim=True)
                nonzero_mask = norm > self.norm_eps
                x_score = x / norm.clamp_min(self.norm_eps)
                scores = (x_score - self.mean) @ self.components.T
                scores = torch.where(nonzero_mask, scores, torch.zeros_like(scores))
                self._last_input_norm = norm.clamp_min(self.norm_eps).detach()
                self._last_nonzero_mask = nonzero_mask.detach()
            else:
                scores = (x - self.mean) @ self.components.T
            return torch.cat([torch.relu(scores), torch.relu(-scores)], dim=-1)

        def decode(self, feature_acts):
            n_components = int(self.components.shape[0])
            scores = feature_acts[..., :n_components] - feature_acts[..., n_components:]
            x_hat = scores.to(self.decoder.dtype) @ self.decoder + self.mean
            if self.preprocess == "with_normalization":
                x_hat = x_hat * self._last_input_norm.to(dtype=x_hat.dtype, device=x_hat.device)
                return torch.where(self._last_nonzero_mask.to(device=x_hat.device), x_hat, torch.zeros_like(x_hat))
            return x_hat

        def forward(self, x):
            return self.decode(self.encode(x))

    return PCASAELike()


def load_itda_sae(*, artifact_prefix: Path, model_name: str, hook_layer: int, hook_name: str, device: str, dtype: Any, encode_k: int | None = None, encode_chunk_size: int = 2048):
    import torch.nn as nn
    from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig

    class ITDASAELike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.metadata = json.loads(artifact_prefix.with_suffix(".json").read_text(encoding="utf-8"))
            atoms_path = artifact_prefix.parent / f"{artifact_prefix.name}_atoms.pt"
            atoms = torch.load(atoms_path, map_location="cpu").to(dtype=torch.float32)
            atoms = atoms / atoms.norm(dim=1, keepdim=True).clamp_min(1e-9)
            n_atoms, d_in = atoms.shape
            self.register_buffer("atoms", atoms.contiguous())
            self.W_dec = nn.Parameter(torch.cat([atoms, -atoms], dim=0).contiguous())
            self.W_enc = nn.Parameter(torch.zeros(d_in, 2 * n_atoms, dtype=torch.float32))
            self.b_enc = nn.Parameter(torch.zeros(2 * n_atoms, dtype=torch.float32))
            self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=torch.float32))
            self.device = torch.device(device)
            self.dtype = dtype
            self.encode_k = int(encode_k if encode_k is not None else self.metadata.get("k", 40))
            self.encode_chunk_size = int(encode_chunk_size)
            self.cfg = CustomSAEConfig(model_name=model_name, d_in=int(d_in), d_sae=int(2 * n_atoms), hook_layer=int(hook_layer), hook_name=hook_name)
            self.cfg.architecture = "itda_matching_pursuit_two_sign_sae_like"
            self.cfg.activation_fn_str = "relu"
            self.cfg.dtype = dtype.__str__().split(".")[1]
            self.cfg.device = str(self.device)
            self.cfg.itda_n_atoms = int(n_atoms)
            self.cfg.itda_training_k = int(self.metadata.get("k", self.encode_k))
            self.cfg.itda_encode_k = int(self.encode_k)
            self.cfg.itda_encode_chunk_size = int(self.encode_chunk_size)
            self.to(device=self.device, dtype=self.dtype)

        @torch.no_grad()
        def encode(self, x):
            original_shape = tuple(x.shape)
            x_flat = x.reshape(-1, original_shape[-1]).to(device=self.atoms.device, dtype=torch.float32)
            features = []
            for start in range(0, x_flat.shape[0], self.encode_chunk_size):
                coeffs = self._encode_flat(x_flat[start : start + self.encode_chunk_size])
                features.append(torch.cat([torch.relu(coeffs), torch.relu(-coeffs)], dim=-1))
            return torch.cat(features, dim=0).reshape(*original_shape[:-1], -1).to(dtype=self.dtype)

        @torch.no_grad()
        def _encode_flat(self, x_flat):
            atoms = self.atoms.to(device=x_flat.device, dtype=x_flat.dtype)
            residual = x_flat.clone()
            coeffs = torch.zeros(x_flat.shape[0], atoms.shape[0], device=x_flat.device, dtype=x_flat.dtype)
            atoms_t = atoms.T.contiguous()
            row_indices = torch.arange(x_flat.shape[0], device=x_flat.device)
            for _ in range(min(self.encode_k, int(atoms.shape[0]))):
                correlations = residual @ atoms_t
                best_atoms = torch.argmax(torch.abs(correlations), dim=1)
                values = correlations[row_indices, best_atoms]
                coeffs[row_indices, best_atoms] += values
                residual -= values[:, None] * atoms[best_atoms]
            return coeffs

        def decode(self, feature_acts):
            n_atoms = int(self.atoms.shape[0])
            coeffs = feature_acts[..., :n_atoms] - feature_acts[..., n_atoms:]
            return coeffs.to(self.atoms.dtype) @ self.atoms

        def forward(self, x):
            return self.decode(self.encode(x))

    return ITDASAELike()


def download_matryoshka_checkpoint(*, output_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    checkpoint_dir = f"{MATRYOSHKA_VARIANT}/{MATRYOSHKA_HOOK}"
    for filename in ("cfg.json", "sae_weights.safetensors", "sparsity.safetensors"):
        hf_hub_download(repo_id=MATRYOSHKA_REPO_ID, filename=f"{checkpoint_dir}/{filename}", local_dir=str(output_root))
    return output_root / checkpoint_dir


def load_matryoshka_prefix_sae(*, checkpoint_dir: Path, width: int, device: str, dtype: Any):
    import torch.nn as nn
    from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
    from safetensors.torch import load_file

    cfg_dict = json.loads((checkpoint_dir / "cfg.json").read_text(encoding="utf-8"))
    state = load_file(str(checkpoint_dir / "sae_weights.safetensors"), device="cpu")

    class MatryoshkaPrefixSAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_in = int(cfg_dict["d_in"])
            self.W_enc = torch.nn.Parameter(state["W_enc"][:, :width].contiguous())
            w_dec = state["W_dec"][:width, :].contiguous()
            self.W_dec = torch.nn.Parameter(w_dec / w_dec.norm(dim=1, keepdim=True).clamp_min(1e-12))
            self.b_enc = torch.nn.Parameter(state["b_enc"][:width].contiguous())
            self.b_dec = torch.nn.Parameter(state["b_dec"].contiguous())
            self.threshold = torch.nn.Parameter(state["threshold"][:width].contiguous(), requires_grad=False)
            self.device = torch.device(device)
            self.dtype = dtype
            self.cfg = CustomSAEConfig(model_name="gemma-2-2b", d_in=d_in, d_sae=width, hook_name=MATRYOSHKA_HOOK, hook_layer=MATRYOSHKA_LAYER)
            self.cfg.architecture = "matryoshka_jumprelu_prefix"
            self.cfg.activation_fn_str = "relu"
            self.cfg.dtype = dtype.__str__().split(".")[1]
            self.cfg.device = str(self.device)
            self.cfg.repo_id = MATRYOSHKA_REPO_ID
            self.cfg.checkpoint_dir = str(checkpoint_dir)
            self.cfg.full_d_sae = int(cfg_dict["d_sae"])
            self.cfg.prefix_width = int(width)
            self.to(device=self.device, dtype=self.dtype)

        def encode(self, x):
            x = x.to(device=self.W_enc.device, dtype=self.W_enc.dtype)
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc
            acts = torch.relu(pre)
            return acts * (pre > self.threshold.to(device=pre.device, dtype=pre.dtype))

        def decode(self, feature_acts):
            return feature_acts.to(device=self.W_dec.device, dtype=self.W_dec.dtype) @ self.W_dec + self.b_dec

        def forward(self, x):
            return self.decode(self.encode(x))

    return MatryoshkaPrefixSAE()


def run_sparse_probe_worker(
    *,
    target: RunTarget,
    output_root: Path,
    saebench_artifacts_path: Path,
    activation_root: Path = DEFAULT_ACTIVATION_ROOT,
    ica_root: Path = DEFAULT_ICA_ROOT,
    token_budget: int | None = None,
    datasets: list[str] | None = None,
    k_values: list[int] | None = None,
    probe_train_size: int = 4000,
    probe_test_size: int = 1000,
    context_length: int = 128,
    llm_batch_size: int | None = None,
    sae_batch_size: int | None = None,
    llm_dtype: str | None = None,
    saebench_model_name: str | None = None,
    hook_name_template: str | None = None,
    run_name: str = "official_sparse_probe_baselines",
    itda_k: int = 40,
    itda_tau: float = 4e-4,
    itda_max_atoms: int = 4096,
    itda_encode_chunk_size: int = 1024,
    force_rerun: bool = False,
    save_activations: bool = True,
    train_full_feature_probe: bool = False,
) -> dict[str, Any]:
    ensure_paths_for_saebench(saebench_env(target.model)[0])
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    from ica_lens.saebench_adapters import load_fastica_sae, selected_baseline_saes
    from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
    import sae_bench.evals.sparse_probing.main as sparse_probe_main
    from sae_bench.sae_bench_utils.general_utils import setup_environment

    cfg = load_model_config(target.model)
    tpp_config = dict(cfg["saebench_tpp"])
    model_cfg = dict(cfg["model"])
    sae_config = dict(cfg["sae"])
    ica_config = dict(cfg.get("ica", {}))
    layer_index = int(target.layer.removeprefix("layer_"))
    resolved_model_name = str(saebench_model_name or tpp_config["saebench_model_name"])
    resolved_hook_template = str(hook_name_template or tpp_config["hook_name_template"])
    hook_name = resolved_hook_template.format(layer=layer_index)
    device = setup_environment()
    _patch_qwen35_sparse_probe_loader(sparse_probe_main=sparse_probe_main, saebench_model_name=resolved_model_name)
    eval_config = SparseProbingEvalConfig(model_name=resolved_model_name)
    eval_config.dataset_names = list(datasets or DEFAULT_DATASETS)
    eval_config.k_values = list(k_values or DEFAULT_K_VALUES)
    eval_config.probe_train_set_size = int(probe_train_size)
    eval_config.probe_test_set_size = int(probe_test_size)
    eval_config.context_length = int(context_length)
    eval_config.llm_batch_size = int(llm_batch_size if llm_batch_size is not None else tpp_config["llm_batch_size"])
    eval_config.sae_batch_size = int(sae_batch_size if sae_batch_size is not None else tpp_config["sae_batch_size"])
    eval_config.llm_dtype = str(llm_dtype if llm_dtype is not None else tpp_config["llm_dtype"])
    eval_config.random_seed = 42
    eval_config.lower_vram_usage = not bool(train_full_feature_probe)
    dtype = _str_to_dtype(eval_config.llm_dtype)
    run_dir = output_root / "runs" / run_name
    saebench_output = run_dir / "saebench" / target.model / target.layer / target.method
    started = time.time()
    selected_saes, method_name, metadata = _selected_probe_sae(
        target=target,
        output_root=output_root,
        ica_root=ica_root,
        token_budget=token_budget,
        cfg=cfg,
        model_cfg=model_cfg,
        sae_config=sae_config,
        saebench_model_name=resolved_model_name,
        ica_config=ica_config,
        layer_index=layer_index,
        hook_name=hook_name,
        device=device,
        dtype=dtype,
        itda_k=itda_k,
        itda_tau=itda_tau,
        itda_max_atoms=itda_max_atoms,
        itda_encode_chunk_size=itda_encode_chunk_size,
    )
    result = sparse_probe_main.run_eval(
        eval_config,
        selected_saes=selected_saes,
        device=device,
        output_path=str(saebench_output),
        force_rerun=force_rerun,
        clean_up_activations=False,
        save_activations=save_activations,
        artifacts_path=str(saebench_artifacts_path),
    )
    result_payload = next(iter(result.values())) if len(result) == 1 else result[selected_saes[0][0]]
    row = layer_result_row(
        model=target.model,
        layer=target.layer,
        method=method_name,
        elapsed_seconds=round(time.time() - started, 3),
        result=result_payload,
        metadata=metadata,
    )
    write_worker_result(output_root=output_root, run_name=run_name, row=row, raw_result=result_payload, target=target)
    return row


def _selected_probe_sae(*, target: RunTarget, output_root: Path, ica_root: Path, token_budget: int | None, cfg: dict[str, Any], model_cfg: dict[str, Any], sae_config: dict[str, Any], saebench_model_name: str, ica_config: dict[str, Any], layer_index: int, hook_name: str, device: str, dtype: Any, itda_k: int, itda_tau: float, itda_max_atoms: int, itda_encode_chunk_size: int) -> tuple[list[tuple[str, Any]], str, dict[str, object]]:
    from ica_lens.saebench_adapters import load_fastica_sae, selected_baseline_saes

    if target.method == "pca":
        prefix = pca_prefix(output_root, target.model, target.layer)
        sae = load_pca_sae(artifact_prefix=prefix, model_name=saebench_model_name, hook_layer=layer_index, hook_name=hook_name, device=device, dtype=dtype, expected_preprocess="with_normalization")
        return [(f"official_pca_{target.model}_{target.layer}", sae)], "pca_two_sign", {"n_saebench_features": int(sae.cfg.d_sae), "artifact_prefix": str(prefix)}
    if target.method == "ica":
        prefix = _ica_artifact_prefix(ica_root=ica_root, model=target.model, layer=target.layer, token_budget=token_budget)
        sae = load_fastica_sae(artifact_prefix=prefix, model_name=saebench_model_name, hook_layer=layer_index, hook_name=hook_name, device=device, dtype=dtype, sign_mode="two_sign", expected_preprocess=canonical_ica_preprocess(ica_config.get("preprocess")))
        return [(f"official_ica_{target.model}_{target.layer}", sae)], "ica_two_sign", {"n_saebench_features": int(sae.cfg.d_sae), "artifact_prefix": str(prefix)}
    if target.method == "itda":
        prefix = itda_prefix(output_root, target.model, target.layer, k=itda_k, tau=itda_tau, max_atoms=itda_max_atoms)
        sae = load_itda_sae(artifact_prefix=prefix, model_name=saebench_model_name, hook_layer=layer_index, hook_name=hook_name, device=device, dtype=dtype, encode_k=itda_k, encode_chunk_size=itda_encode_chunk_size)
        return [(f"official_itda_{target.model}_{target.layer}", sae)], "itda_two_sign", {"n_saebench_features": int(sae.cfg.d_sae), "n_itda_atoms": int(sae.cfg.itda_n_atoms), "artifact_prefix": str(prefix), "itda_encode_k": itda_k}
    if target.method == "sae_baseline":
        baseline_config = {"model": model_cfg, "sae": sae_config, "saebench_tpp": {**dict(cfg["saebench_tpp"]), "saebench_model_name": saebench_model_name}}
        selected = selected_baseline_saes(baseline_config, dict(baseline_config["saebench_tpp"]), layer_index, device=device)
        return selected, "sae_baseline", {"sae_release": selected[0][0]}
    if target.method.startswith("matryoshka_"):
        width = int(target.method.rsplit("_", 1)[-1])
        checkpoint_dir = matryoshka_checkpoint_dir(output_root)
        sae = load_matryoshka_prefix_sae(checkpoint_dir=checkpoint_dir, width=width, device=device, dtype=dtype)
        return [(f"official_matryoshka_{width}_gemma2_2b_layer_12", sae)], f"matryoshka_{width}", {"n_saebench_features": width, "matryoshka_width": width, "checkpoint_dir": str(checkpoint_dir)}
    raise ValueError(f"Unknown method: {target.method}")


def _ica_artifact_prefix(*, ica_root: Path, model: str, layer: str, token_budget: int | None) -> Path:
    candidates = []
    if token_budget is not None:
        candidates.append(ica_root / f"{model}_tok{token_budget}" / f"{layer}_fastica")
    candidates.append(ica_root / model / f"{layer}_fastica")
    for candidate in candidates:
        if candidate.with_suffix(".pt").is_file():
            return candidate
    return candidates[0]


def _patch_qwen35_sparse_probe_loader(*, sparse_probe_main: Any, saebench_model_name: str) -> None:
    if saebench_model_name != "Qwen/Qwen3.5-2B-Base":
        return
    from sae_bench.sae_bench_utils import activation_collection
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class Qwen35HookedTransformerShim:
        @staticmethod
        def from_pretrained_no_processing(model_name: str, device: str, dtype: Any) -> Any:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
            hf_model.to(device)
            hf_model.eval()
            return activation_collection.HFCausalLMActivationModel(hf_model, tokenizer)

    sparse_probe_main.HookedTransformer = Qwen35HookedTransformerShim


def layer_result_row(*, model: str, layer: str, method: str, elapsed_seconds: float, result: dict[str, Any], metadata: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {"model_name": model, "layer": layer, "method": method, "elapsed_seconds": elapsed_seconds}
    row.update(metadata)
    sae_cfg = result.get("sae_cfg_dict", {})
    if isinstance(sae_cfg, dict) and "d_sae" in sae_cfg:
        row.setdefault("n_saebench_features", sae_cfg.get("d_sae"))
    if result.get("sae_lens_id") is not None:
        row.setdefault("sae_id", result.get("sae_lens_id"))
    row.update(flatten_sparse_probe_metrics(result.get("eval_result_metrics", {})))
    return row


def flatten_sparse_probe_metrics(metrics: dict[str, Any]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for category, values in metrics.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[key if str(key).startswith(f"{category}_") else f"{category}_{key}"] = value
    return flat


def write_worker_result(*, output_root: Path, run_name: str, row: dict[str, object], raw_result: dict[str, Any], target: RunTarget) -> None:
    run_dir = output_root / "runs" / run_name
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{target.model}_{target.layer}_{target.method}"
    (raw_dir / f"{stem}.json").write_text(json.dumps({"target": target.__dict__, "row": row, "raw_result": raw_result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    collect_run_rows(output_root=output_root, run_name=run_name)


def collect_run_rows(*, output_root: Path, run_name: str = "official_sparse_probe_baselines") -> list[dict[str, object]]:
    run_dir = output_root / "runs" / run_name
    rows = []
    for path in sorted((run_dir / "raw").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8"))["row"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / f"{run_name}_by_method.csv", rows)
    _write_json(run_dir / f"{run_name}.json", {"analysis": "official_sparse_probe_baselines", "rows": rows})
    _write_csv(run_dir / f"{run_name}_summary.csv", summary_rows(rows))
    return rows


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: (str(row.get("model_name")), str(row.get("layer")), METHOD_ORDER.index(str(row.get("method"))) if str(row.get("method")) in METHOD_ORDER else 99))


def collect_summary_outputs(*, output_root: Path) -> list[dict[str, object]]:
    rows = collect_run_rows(output_root=output_root)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "sparse_probe_baselines_long.csv", rows)
    _write_csv(summary_dir / "sparse_probe_baselines_summary.csv", summary_rows(rows))
    _write_json(summary_dir / "sparse_probe_baselines.json", {"analysis": "official_sparse_probe_baselines_summary", "rows": rows})
    return rows


def plot_outputs(*, output_root: Path, figures_dir: Path | None = None, formats: list[str] | None = None) -> None:
    formats = formats or ["pdf", "png"]
    rows = collect_summary_outputs(output_root=output_root)
    figures_dir = figures_dir or output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_all_layers(rows, figures_dir, formats)
    _plot_merged_model_panels(rows, figures_dir, formats)
    _plot_gemma_focus(rows, figures_dir, formats)


def _plot_all_layers(rows: list[dict[str, object]], figures_dir: Path, formats: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_style_rcparams())
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharey=True, squeeze=False)
    panel_specs = [
        ("gpt2", "layer_06"),
        ("gemma2_2b", "layer_12"),
        ("qwen3_5_2b_base", "layer_12"),
        ("gpt2", "layer_10"),
        ("gemma2_2b", "layer_20"),
        ("qwen3_5_2b_base", "layer_20"),
    ]
    for ax, (model, layer) in zip(axes.ravel(), panel_specs, strict=True):
        _draw_panel(ax, rows, model=model, layer=layer, methods=CORE_METHOD_ORDER, title=f"{MODEL_LABELS[model]} {layer.replace('_', ' ')}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=4)
    fig.supxlabel(r"top-$k$ features used by probe", y=0.03)
    fig.supylabel("probe accuracy", x=0.01)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.13, top=0.86, wspace=0.16, hspace=0.35)
    for fmt in formats:
        fig.savefig(figures_dir / f"sparse_probe_baselines_all_layers.{fmt}", bbox_inches="tight", dpi=220 if fmt != "pdf" else None)
    plt.close(fig)


def _plot_gemma_focus(rows: list[dict[str, object]], figures_dir: Path, formats: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_style_rcparams())
    fig, ax = plt.subplots(1, 1, figsize=(4.75, 2.65))
    _draw_panel(
        ax,
        rows,
        model="gemma2_2b",
        layer="layer_12",
        methods=METHOD_ORDER,
        title="Gemma 2 2B layer 12",
        marker_by_method=SPARSE_PROBE_MARKERS,
    )
    ax.set_ylabel("probe accuracy")
    ax.set_xlabel(r"top-$k$ features used by probe")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(left=0.12, right=0.66, bottom=0.20, top=0.93)
    for fmt in formats:
        fig.savefig(figures_dir / f"gemma2_layer12_official_baselines.{fmt}", bbox_inches="tight", dpi=220 if fmt != "pdf" else None)
        fig.savefig(figures_dir / f"gemma2_layer12_probe_with_matryoshka_and_itda.{fmt}", bbox_inches="tight", dpi=220 if fmt != "pdf" else None)
    plt.close(fig)


def _plot_merged_model_panels(rows: list[dict[str, object]], figures_dir: Path, formats: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(_style_rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharey=True, squeeze=False)
    layer_specs = {
        "gpt2": ("layer_06", "layer_10"),
        "gemma2_2b": ("layer_12", "layer_20"),
        "qwen3_5_2b_base": ("layer_12", "layer_20"),
    }
    for ax, model in zip(axes.ravel(), ("gpt2", "gemma2_2b", "qwen3_5_2b_base"), strict=True):
        for method in CORE_METHOD_ORDER:
            xs, ys = _mean_series(rows, model=model, layers=layer_specs[model], method=method)
            if not xs:
                continue
            ax.plot(xs, ys, color=METHOD_COLORS[method], marker=SPARSE_PROBE_MARKERS[method], linewidth=1.35, markersize=2.4, label=METHOD_LABELS[method])
        ax.set_title(MODEL_LABELS[model], loc="left", fontweight="semibold", pad=2)
        ax.axvline(20, color="#c9ced6", linewidth=0.65, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks(DEFAULT_K_VALUES)
        ax.set_xticklabels([str(k) for k in DEFAULT_K_VALUES], rotation=45, ha="right")
        ax.grid(axis="y", color="#e4e7eb", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker=SPARSE_PROBE_MARKERS[method], linewidth=1.35, markersize=2.4, label=METHOD_LABELS[method])
        for method in CORE_METHOD_ORDER
    ]
    fig.legend(handles=method_handles, frameon=False, loc="upper center", ncols=4)
    fig.supxlabel(r"top-$k$ features used by probe", y=0.04)
    fig.supylabel("mean probe accuracy", x=0.01)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.76, wspace=0.16)
    for fmt in formats:
        fig.savefig(figures_dir / f"sparse_probe_baselines_by_model.{fmt}", bbox_inches="tight", dpi=220 if fmt != "pdf" else None)
    plt.close(fig)


def _draw_panel(
    ax: Any,
    rows: list[dict[str, object]],
    *,
    model: str,
    layer: str,
    methods: tuple[str, ...],
    title: str,
    marker_by_method: dict[str, str] | None = None,
) -> None:
    for method in methods:
        row = next((candidate for candidate in rows if candidate.get("model_name") == model and candidate.get("layer") == layer and candidate.get("method") == method), None)
        if row is None:
            continue
        xs, ys = _series(row)
        marker = marker_by_method.get(method, "o") if marker_by_method is not None else "o"
        ax.plot(xs, ys, color=METHOD_COLORS[method], marker=marker, linewidth=1.35, markersize=2.4, linestyle="-", label=METHOD_LABELS[method])
    ax.set_title(title, loc="left", fontweight="semibold", pad=2)
    ax.axvline(20, color="#c9ced6", linewidth=0.65, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks(DEFAULT_K_VALUES)
    ax.set_xticklabels([str(k) for k in DEFAULT_K_VALUES], rotation=45, ha="right")
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _series(row: dict[str, object]) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for k in DEFAULT_K_VALUES:
        value = row.get(f"sae_top_{k}_test_accuracy")
        if value not in {"", None}:
            xs.append(k)
            ys.append(float(value))
    return xs, ys


def _mean_series(rows: list[dict[str, object]], *, model: str, layers: tuple[str, ...], method: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    selected_rows = [
        row
        for row in rows
        if row.get("model_name") == model and row.get("layer") in layers and row.get("method") == method
    ]
    for k in DEFAULT_K_VALUES:
        values = [
            float(row[f"sae_top_{k}_test_accuracy"])
            for row in selected_rows
            if row.get(f"sae_top_{k}_test_accuracy") not in {"", None}
        ]
        if values:
            xs.append(k)
            ys.append(sum(values) / len(values))
    return xs, ys


def _style_rcparams() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _str_to_dtype(value: str) -> Any:
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return dtypes[value]


def artifact_id(path: Path, metadata: dict[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    if metadata:
        digest.update(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()[:12]


def environment_report() -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": package_version("torch"), "v6_git_commit": git_commit(V6_ROOT)}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = ["model_name", "layer", "method", "n_saebench_features", "n_itda_atoms", "itda_encode_k", "matryoshka_width", "elapsed_seconds"]
    seen = set(preferred)
    rest = sorted({key for row in rows for key in row if key not in seen})
    return [key for key in preferred if any(key in row for row in rows)] + rest
