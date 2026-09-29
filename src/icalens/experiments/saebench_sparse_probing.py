"""Run pinned SAEBench sparse probing for an ICA Lens artifact."""

from __future__ import annotations

import argparse
import codecs
import csv
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from icalens import ICALens, __version__
from icalens.cli._status import log

from ._run import ResumableRun
from ._saebench_environment import (
    backend_description,
    prepare_backend,
    resolve_backend,
)
from ._source_provenance import source_provenance, warn_if_dirty

DEFAULT_ACTIVATION_CACHE_ROOT = Path("~/Expansion/research/ICA-data/sparse-probing").expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="icalens experiment saebench-sparse-probing", description=__doc__
    )
    parser.add_argument("--lens", required=True, help="Local ICA Lens path or Hugging Face repo.")
    parser.add_argument(
        "--layers", required=True, help="Comma-separated fitted layer indices, or 'all'."
    )
    parser.add_argument("--preset", choices=("smoke", "paper"), default="smoke")
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated feature budgets overriding the selected preset, e.g. 200,500.",
    )
    parser.add_argument(
        "--baselines",
        default="",
        help="Comma-separated comparison baselines: sae,pca,random,all (default: ICA only).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--activation-cache-root",
        type=Path,
        default=DEFAULT_ACTIVATION_CACHE_ROOT,
        help=(
            "Root for persistent sparse-probing activation caches "
            f"(default: {DEFAULT_ACTIVATION_CACHE_ROOT})."
        ),
    )
    parser.add_argument(
        "--saebench-path", type=Path, default=None, help="Use an existing SAEBench checkout."
    )
    parser.add_argument("--refresh-environment", action="store_true")
    parser.add_argument(
        "--allow-low-disk",
        action="store_true",
        help="Run even when free disk is below the activation-cache safety estimate.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_started_at = time.time()
    args = parse_args(argv)
    source = source_provenance()
    warn_if_dirty(source)
    lens = ICALens.from_pretrained(args.lens)
    hidden_size = lens.hidden_size
    if hidden_size is None:
        raise ValueError("lens manifest does not record hidden_size")
    layers = _parse_layers(args.layers, lens.available_layers)
    backend = resolve_backend(lens.model_id)
    settings = _load_preset(args.preset)
    if args.k_values is not None:
        settings["k_values"] = _parse_k_values(args.k_values)
    baselines = _resolve_baselines(lens.model_id, args.baselines)
    input_protocol = _evaluation_input_protocol(lens._get_profile(lens._get_layer(layers[0])))
    cache_estimate = _estimate_activation_cache_bytes(settings, hidden_size)
    resolved: dict[str, Any] = {
        "experiment": "saebench-sparse-probing",
        "experiment_schema_version": 4,
        "icalens_version": __version__,
        "lens": str(args.lens),
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "activation_site": lens.activation_site,
        "evaluation_input_protocol": input_protocol,
        "layers": layers,
        "preset": settings,
        "backend": backend_description(backend, args.cache_dir),
        "baselines": baselines,
        "activation_cache": {
            "estimated_bytes_per_layer": cache_estimate,
            "estimated_per_layer": _format_bytes(cache_estimate),
            "shared_capture_layers": len(layers),
            "estimated_peak_bytes": cache_estimate * len(layers),
            "estimated_peak": _format_bytes(cache_estimate * len(layers)),
            "recommended_free_bytes": int(cache_estimate * len(layers) * 1.20),
            "recommended_free": _format_bytes(int(cache_estimate * len(layers) * 1.20)),
            "used_when_multiple_methods_are_pending": True,
        },
        "output": str(args.output.expanduser().resolve()),
    }
    if args.saebench_path is not None:
        resolved["backend"]["override_path"] = str(args.saebench_path.expanduser().resolve())
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    activation_cache_root = args.activation_cache_root.expanduser().resolve()
    cache_identity = {
        "schema_version": 1,
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "layers": layers,
        "preset": settings,
        "evaluation_input_protocol": input_protocol,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    activation_cache_dir = activation_cache_root / _path_slug(lens.model_id) / cache_key
    activation_cache_dir.mkdir(parents=True, exist_ok=True)
    resolved["activation_cache"].update(
        {
            "root": str(activation_cache_root),
            "run_cache_key": cache_key,
            "run_cache_dir": str(activation_cache_dir),
            "persistent": True,
        }
    )
    run_path = output / "run.json"
    run = _load_or_initialize_run(run_path, resolved, source=source)
    run["icalens_source"] = source
    _write_json(run_path, run)

    expected_methods = ["ica", *baselines]
    completed_by_layer = {layer: _completed_methods(output, layer) for layer in layers}
    pending_layers = [
        layer for layer in layers if not set(expected_methods).issubset(completed_by_layer[layer])
    ]
    if not pending_layers:
        log("All requested layers are already complete; rebuilding summary files.")
        _finish_run(output, run_path, run, resolved, layers)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("SAEBench sparse probing currently requires CUDA")

    missing_by_layer = {
        layer: [name for name in expected_methods if name not in completed_by_layer[layer]]
        for layer in pending_layers
    }
    if any(len(methods) > 1 for methods in missing_by_layer.values()):
        _check_activation_cache_space(
            activation_cache_dir,
            settings=settings,
            hidden_size=hidden_size,
            layers_at_once=len(pending_layers),
            allow_low_disk=bool(args.allow_low_disk),
        )

    log(f"Selected {backend.name}@{backend.commit[:8]} for {lens.model_id}.")
    prepared = prepare_backend(
        backend,
        cache_dir=args.cache_dir,
        saebench_path=args.saebench_path,
        refresh=bool(args.refresh_environment),
    )
    log(f"Using SAEBench at {prepared.root}")
    config_path = output / "experiment-config.json"
    _write_json(config_path, settings)
    methods_per_layer = len(expected_methods)
    datasets_per_method = len(settings["datasets"])
    total_evaluations = len(layers) * methods_per_layer * datasets_per_method
    completed_evaluations = sum(
        len(set(expected_methods).intersection(completed_by_layer[layer])) * datasets_per_method
        for layer in layers
    )
    run_initial_evaluations = _completed_evaluations_at_start(
        output,
        layers=layers,
        methods=expected_methods,
        datasets=settings["datasets"],
        completed_by_layer=completed_by_layer,
    )

    jobs: list[dict[str, Any]] = []
    for layer in layers:
        key = str(layer)
        layer_state = run["layer_runs"].setdefault(key, {})
        raw_path = output / "layers" / f"layer_{layer:02d}" / "raw-result.json"
        completed_methods = _completed_methods(output, layer)
        missing_methods = [name for name in expected_methods if name not in completed_methods]
        if not missing_methods:
            log(f"Layer {layer}: all methods already complete; reusing saved results.")
            continue
        layer_state.update({"status": "running", "started_at": _timestamp(), "completed_at": None})
        _write_json(run_path, run)
        layer_dir = raw_path.parent
        layer_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = _write_layer_snapshot(
            lens,
            layer=layer,
            output=output / "checkpoints" / f"layer_{layer:02d}",
            saebench_model_name=backend.saebench_model_name,
            baselines=baselines,
        )
        jobs.append(
            {
                "layer": layer,
                "snapshot": str(snapshot_path),
                "output": str(layer_dir),
                "methods": missing_methods,
            }
        )
        log(f"Layer {layer}: queued {args.preset} sparse probing for {', '.join(missing_methods)}.")
        _write_json(run_path, run)

    jobs_path = output / "checkpoints" / "multilayer-jobs.json"
    _write_json(jobs_path, {"jobs": jobs})
    command = [
        str(prepared.python),
        str(files("icalens.experiments").joinpath("_saebench_multilayer_worker.py")),
        "--saebench-root",
        str(prepared.root),
        "--jobs",
        str(jobs_path),
        "--config",
        str(config_path),
        "--artifacts",
        str(activation_cache_dir),
        "--keep-artifacts",
        "--progress-initial",
        str(completed_evaluations),
        "--progress-total",
        str(total_evaluations),
        "--progress-run-initial",
        str(run_initial_evaluations),
        "--progress-started-at",
        str(run_started_at),
    ]
    if source.get("dirty"):
        command.append("--source-dirty")
    detail_log = output / "logs" / "experiment-detail.log"
    try:
        log(
            "Running dataset-first sparse probing with shared activation capture for layers "
            + ",".join(str(job["layer"]) for job in jobs)
            + "..."
        )
        _run_logged(command, detail_log)
        for job in jobs:
            layer = int(job["layer"])
            raw_path = Path(job["output"]) / "raw-result.json"
            run["layer_runs"][str(layer)].update(
                {"status": "complete", "completed_at": _timestamp(), "result": str(raw_path)}
            )
    except Exception as error:
        for job in jobs:
            layer_state = run["layer_runs"][str(job["layer"])]
            if layer_state.get("status") == "running":
                layer_state.update(
                    {"status": "failed", "completed_at": _timestamp(), "error": str(error)}
                )
        _write_json(run_path, run)
        raise
    finally:
        log(f"Full output: {detail_log}")
    _write_json(run_path, run)

    _finish_run(output, run_path, run, resolved, layers)
    log(f"Experiment complete: {output}")
    log(f"Create a figure with: icalens experiment figure sparse-probing {output}")


def _parse_k_values(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--k-values must be a comma-separated list of integers") from error
    if not values or any(item < 1 for item in values):
        raise ValueError("--k-values must contain positive integers")
    return sorted(set(values))


def collect_result_rows(output: Path, layers: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        path = output / "layers" / f"layer_{layer:02d}" / "raw-result.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        methods = payload.get("methods")
        if not isinstance(methods, dict):
            methods = {"ica": payload}
        for method, raw in methods.items():
            metrics = raw.get("eval_result_metrics", {}).get("sae", {})
            config = raw.get("eval_config", {})
            for k in config.get("k_values", []):
                metric_name = f"sae_top_{k}_test_accuracy"
                value = metrics.get(metric_name)
                if value is None:
                    value = _mean_unstructured_metric(raw, metric_name)
                if value is not None:
                    rows.append(
                        {
                            "method": str(method),
                            "layer": int(layer),
                            "k": int(k),
                            "mean_probe_accuracy": float(value),
                        }
                    )
    return rows


def _mean_unstructured_metric(raw: dict[str, Any], metric_name: str) -> float | None:
    """Recover arbitrary-k metrics omitted by SAEBench's fixed result schema."""
    dataset_means: list[float] = []
    unstructured = raw.get("eval_result_unstructured", {})
    if not isinstance(unstructured, dict):
        return None
    for dataset_result in unstructured.values():
        if not isinstance(dataset_result, dict):
            continue
        per_class = dataset_result.get(metric_name)
        if not isinstance(per_class, dict) or not per_class:
            continue
        values = [float(value) for value in per_class.values()]
        dataset_means.append(sum(values) / len(values))
    if not dataset_means:
        return None
    return sum(dataset_means) / len(dataset_means)


def _completed_methods(output: Path, layer: int) -> set[str]:
    """Return methods with durable SAEBench result files for a layer."""
    completed: set[str] = set()
    raw_path = output / "layers" / f"layer_{layer:02d}" / "raw-result.json"
    if raw_path.is_file():
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        methods = payload.get("methods") if isinstance(payload, dict) else None
        if isinstance(methods, dict):
            completed.update(
                str(name) for name, result in methods.items() if _valid_result_payload(result)
            )
        elif _valid_result_payload(payload):
            completed.add("ica")
    saebench_dir = raw_path.parent / "saebench"
    for name in ("ica", "sae", "pca", "random"):
        if _valid_result_file(saebench_dir / f"{name}_custom_sae_eval_results.json"):
            completed.add(name)
    return completed


def _valid_result_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return _valid_result_payload(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return False


def _valid_result_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("eval_result_metrics"), dict)


def _completed_evaluations_at_start(
    output: Path,
    *,
    layers: Sequence[int],
    methods: Sequence[str],
    datasets: Sequence[str],
    completed_by_layer: dict[int, set[str]],
) -> int:
    """Count durable dataset-method jobs without treating resumed work as new work."""
    completed = 0
    for layer in layers:
        complete_methods = completed_by_layer[layer]
        dataset_root = output / "layers" / f"layer_{layer:02d}" / "saebench-datasets"
        for method in methods:
            if method in complete_methods:
                completed += len(datasets)
                continue
            for index, dataset in enumerate(datasets):
                safe_name = dataset.replace("/", "__")
                path = (
                    dataset_root
                    / f"{index:02d}_{safe_name}"
                    / f"{method}_custom_sae_eval_results.json"
                )
                completed += int(_valid_result_file(path))
    return completed


def _finish_run(
    output: Path,
    run_path: Path,
    run: dict[str, Any],
    resolved: dict[str, Any],
    layers: Sequence[int],
) -> None:
    rows = collect_result_rows(output, layers)
    _write_json(
        output / "results.json",
        {
            "experiment": resolved,
            "icalens_source": run.get("icalens_source"),
            "rows": rows,
        },
    )
    _write_csv(output / "results.csv", rows)
    run["status"] = "complete"
    run["completed_at"] = _timestamp()
    _write_json(run_path, run)


def _write_layer_snapshot(
    lens: ICALens,
    *,
    layer: int,
    output: Path,
    saebench_model_name: str,
    baselines: dict[str, dict[str, Any]],
) -> Path:
    artifact = lens._get_layer(layer)
    assert artifact.center is not None
    assert artifact.reading_matrix is not None
    assert artifact.writing_matrix is not None
    profile = lens._get_profile(artifact)
    tail_signs = torch.empty(artifact.n_components, dtype=torch.float32)
    seen: set[int] = set()
    for component in profile.get("components", []):
        index = int(component["component"])
        direction = component.get("tail_direction")
        if not 0 <= index < artifact.n_components or index in seen:
            raise ValueError(f"layer {layer} has an invalid profiled component index {index}")
        if direction not in {"positive", "negative"}:
            raise ValueError(f"layer {layer} C{index} has no profiled tail_direction")
        tail_signs[index] = 1.0 if direction == "positive" else -1.0
        seen.add(index)
    if len(seen) != artifact.n_components:
        missing = sorted(set(range(artifact.n_components)).difference(seen))
        raise ValueError(f"layer {layer} profile is missing components {missing[:10]}")
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "layer.safetensors"
    save_file(
        {
            "center": torch.from_numpy(np.asarray(artifact.center)),
            "reading_matrix": torch.from_numpy(np.asarray(artifact.reading_matrix)),
            "writing_matrix": torch.from_numpy(np.asarray(artifact.writing_matrix)),
            "tail_signs": tail_signs,
        },
        tensor_path,
    )
    snapshot = {
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "saebench_model_name": saebench_model_name,
        "activation_site": lens.activation_site,
        "hidden_size": lens.hidden_size,
        "layer": layer,
        "row_normalize": lens.row_normalize,
        "norm_eps": lens.norm_eps,
        "layer_file": str(tensor_path),
        "ica_orientation": "profile_tail_direction",
        "ica_feature_sides": "one_signed_coordinate",
        "evaluation_input_protocol": _evaluation_input_protocol(profile),
        "baselines": _prepare_layer_baselines(baselines, layer=layer),
    }
    path = output / "snapshot.json"
    _write_json(path, snapshot)
    return path


def _evaluation_input_protocol(profile: dict[str, Any]) -> dict[str, Any]:
    framing = profile.get("provenance", {}).get("document_framing")
    if not isinstance(framing, dict):
        raise ValueError("component profile is missing document_framing provenance")
    strategy = str(framing.get("strategy"))
    if strategy not in {"prepend-bos", "prepend-eos"}:
        raise ValueError(f"unsupported evaluation document framing: {strategy!r}")
    return {
        "protocol_version": 1,
        "strategy": strategy,
        "token_id": int(framing["token_id"]),
        "token": str(framing["token"]),
        "boundary_source": "tokenizer" if strategy == "prepend-bos" else "adapter",
        "exclude_boundary_activation": True,
        "attention_mask": "pad_tokens_hidden_boundary_visible",
    }


def _path_slug(value: str) -> str:
    return "-".join(part for part in re.split(r"[^a-zA-Z0-9]+", value.lower()) if part)


def _parse_layers(value: str, available: Sequence[int]) -> list[int]:
    if value.strip().lower() == "all":
        return list(available)
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--layers must be comma-separated integers or 'all'") from error
    if not layers:
        raise ValueError("--layers must select at least one layer")
    missing = sorted(set(layers).difference(available))
    if missing:
        raise ValueError(f"lens does not contain layers {missing}; available layers: {available}")
    return list(dict.fromkeys(layers))


def _load_preset(name: str) -> dict[str, Any]:
    path = files("icalens.experiments").joinpath("configs", f"{name}.json")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


_SPARSE_PROBING_CLASS_COUNTS = {
    "LabHC/bias_in_bios_class_set1": 5,
    "LabHC/bias_in_bios_class_set2": 5,
    "LabHC/bias_in_bios_class_set3": 5,
    "canrager/amazon_reviews_mcauley_1and5": 5,
    "canrager/amazon_reviews_mcauley_1and5_sentiment": 2,
    "codeparrot/github-code": 5,
    "fancyzhx/ag_news": 4,
    "Helsinki-NLP/europarl": 5,
}


def _estimate_activation_cache_bytes(settings: dict[str, Any], hidden_size: int) -> int:
    """Estimate SAEBench's raw hidden-state cache for one layer.

    Sparse probing builds one-vs-rest binary tasks, so each selected class stores
    half of the configured train and test sizes. The saved tensors retain the full
    context axis and use ``llm_dtype``.
    """
    datasets = [str(name) for name in settings["datasets"]]
    unknown = [name for name in datasets if name not in _SPARSE_PROBING_CLASS_COUNTS]
    if unknown:
        raise ValueError(
            "cannot estimate sparse-probing cache for unregistered datasets: " + ", ".join(unknown)
        )
    # Dataset-first execution deletes each cache before starting the next one, so
    # peak storage is determined by the largest selected dataset.
    class_count = max(_SPARSE_PROBING_CLASS_COUNTS[name] for name in datasets)
    samples_per_class = (
        int(settings["probe_train_size"]) // 2 + int(settings["probe_test_size"]) // 2
    )
    dtype_bytes = {
        "float64": 8,
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }.get(str(settings["llm_dtype"]))
    if dtype_bytes is None:
        raise ValueError(f"unsupported cache-estimation dtype: {settings['llm_dtype']!r}")
    return (
        class_count
        * samples_per_class
        * int(settings["context_length"])
        * int(hidden_size)
        * dtype_bytes
    )


def _check_activation_cache_space(
    output: Path,
    *,
    settings: dict[str, Any],
    hidden_size: int,
    layers_at_once: int = 1,
    allow_low_disk: bool,
) -> None:
    estimate = _estimate_activation_cache_bytes(settings, hidden_size) * layers_at_once
    # torch.save metadata is small, but partial files and filesystem overhead make
    # running exactly at the raw tensor estimate unsafe.
    safety_target = int(estimate * 1.20)
    free = shutil.disk_usage(output).free
    log(
        f"Activation-cache disk preflight ({layers_at_once} layer"
        f"{'s' if layers_at_once != 1 else ''} captured together): "
        f"estimated {_format_bytes(estimate)}, "
        f"recommended free {_format_bytes(safety_target)}, "
        f"available {_format_bytes(free)}."
    )
    if free < safety_target:
        message = (
            "insufficient free disk for the shared SAEBench activation cache: "
            f"recommended {_format_bytes(safety_target)}, available {_format_bytes(free)}. "
            "Free disk, choose an output on a larger filesystem, or pass "
            "--allow-low-disk to proceed at your own risk."
        )
        if not allow_low_disk:
            raise RuntimeError(message)
        log(f"WARNING: {message}")


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _resolve_baselines(model_id: str, value: str) -> dict[str, dict[str, Any]]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    if "all" in selected:
        selected = ["sae", "pca", "random"]
    selected = list(dict.fromkeys(selected))
    unknown = sorted(set(selected).difference({"sae", "pca", "random"}))
    if unknown:
        raise ValueError(f"unknown baselines {unknown}; expected sae,pca,random,all")
    if not selected:
        return {}
    path = files("icalens.experiments").joinpath("baseline_registry.json")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise RuntimeError("unsupported experiment baseline registry schema")
    model = registry.get("models", {}).get(model_id)
    registered_selected = [name for name in selected if name != "random"]
    if registered_selected and not isinstance(model, dict):
        raise ValueError(f"no sparse-probing baselines are registered for {model_id!r}")
    if not isinstance(model, dict):
        model = {}
    missing = [name for name in selected if name != "random" and name not in model]
    if missing:
        raise ValueError(f"baselines {missing} are not registered for {model_id!r}")
    resolved = {name: cast(dict[str, Any], model[name]) for name in selected if name != "random"}
    if "random" in selected:
        resolved["random"] = {
            "name": "Random orthogonal basis",
            "components": "model_hidden_size",
            "preprocessing": "match_ica_lens",
            "feature_sides": "one_signed_coordinate",
            "seed": 0,
        }
    return {name: resolved[name] for name in selected}


def _prepare_layer_baselines(
    baselines: dict[str, dict[str, Any]], *, layer: int
) -> dict[str, dict[str, Any]]:
    prepared = cast(dict[str, dict[str, Any]], json.loads(json.dumps(baselines)))
    sae = prepared.get("sae")
    if sae is not None:
        layer_checkpoints = sae.get("layer_checkpoints")
        if isinstance(layer_checkpoints, dict):
            filename = layer_checkpoints.get(str(layer))
            if filename is None:
                raise ValueError(f"SAE baseline has no checkpoint registered for layer {layer}")
            filename = str(filename)
        else:
            filename = str(sae["checkpoint_template"]).format(layer=layer)
        sae["checkpoint"] = filename
        sae["weights_file"] = hf_hub_download(
            repo_id=str(sae["repo_id"]),
            filename=filename,
            revision=str(sae["revision"]),
        )
    return prepared


def _load_or_initialize_run(
    path: Path,
    resolved: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def normalize(previous: dict[str, Any], current: dict[str, Any]) -> None:
        for value in (previous.get("backend"), current.get("backend")):
            if isinstance(value, dict):
                value.pop("cached", None)
                value.pop("cache_path", None)
        previous_baselines = previous.get("baselines")
        current_baselines = current.get("baselines")
        if isinstance(previous_baselines, dict) and isinstance(current_baselines, dict):
            for name, config in current_baselines.items():
                previous_baselines.setdefault(name, config)

    run = ResumableRun.open(
        output=path.parent,
        resolved=resolved,
        source=source or {},
        status="running",
        filename=path.name,
        normalize_previous=normalize,
    )
    run.state.setdefault("layer_runs", {})
    run.save()
    return run.state


def _configuration_mismatches(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Compare result-defining settings while ignoring runtime cache metadata."""
    mismatches: list[str] = []
    keys = (
        "experiment",
        "lens",
        "model_id",
        "model_revision",
        "activation_site",
        "layers",
        "preset",
        "baselines",
    )
    for key in keys:
        previous_value = previous.get(key, {} if key == "baselines" else None)
        current_value = current.get(key, {} if key == "baselines" else None)
        if (
            key == "baselines"
            and isinstance(previous_value, dict)
            and isinstance(current_value, dict)
        ):
            compatible = all(
                current_value.get(name) == config for name, config in previous_value.items()
            )
        else:
            compatible = previous_value == current_value
        if not compatible:
            mismatches.append(f"{key}: {previous_value!r} != {current_value!r}")

    backend_keys = ("name", "repository", "commit", "saebench_model_name")
    previous_backend = previous.get("backend", {})
    current_backend = current.get("backend", {})
    for key in backend_keys:
        if previous_backend.get(key) != current_backend.get(key):
            mismatches.append(
                f"backend.{key}: {previous_backend.get(key)!r} != {current_backend.get(key)!r}"
            )
    return mismatches


def _run_logged(command: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if sys.stdout.isatty() and os.name == "posix":
        _run_logged_in_terminal(command, path, environment)
        return
    with path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"SAEBench worker exited with status {code}; see {path}")


def _run_logged_in_terminal(command: list[str], path: Path, environment: dict[str, str]) -> None:
    """Run through a pseudo-terminal so child progress bars can redraw in place."""
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    columns, lines = shutil.get_terminal_size(fallback=(120, 24))
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", lines, columns, 0, 0))
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=slave,
            env=environment,
            close_fds=True,
        )
    finally:
        os.close(slave)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with path.open("a", encoding="utf-8") as log:
        while True:
            try:
                chunk = os.read(master, 16_384)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            text = decoder.decode(chunk)
            sys.stdout.write(text)
            sys.stdout.flush()
            log.write(text)
            log.flush()
        remainder = decoder.decode(b"", final=True)
        if remainder:
            sys.stdout.write(remainder)
            sys.stdout.flush()
            log.write(remainder)
    os.close(master)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"SAEBench worker exited with status {code}; see {path}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "layer", "k", "mean_probe_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main(sys.argv[1:])
