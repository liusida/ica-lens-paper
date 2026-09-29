"""SAEBench sparse probing with one shared multi-layer capture per dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from icalens.experiments._sae import SAEFeatureEncoder
    from icalens.experiments._saebench_worker import (
        ICAFeatureEncoder,
        PCAFeatureEncoder,
        RandomFeatureEncoder,
        _BenchmarkDisplay,
        _blocks,
        _dataset_output,
        _dataset_result_path,
        _merge_dataset_results,
        _prepare_model_inputs,
        _remove_dataset_artifacts,
        _unwrap_runtime_types,
    )
except ModuleNotFoundError:  # Direct execution inside the isolated SAEBench environment.
    from _sae import SAEFeatureEncoder  # type: ignore[import-not-found,no-redef]
    from _saebench_worker import (  # type: ignore[import-not-found,no-redef]
        ICAFeatureEncoder,
        PCAFeatureEncoder,
        RandomFeatureEncoder,
        _BenchmarkDisplay,
        _blocks,
        _dataset_output,
        _dataset_result_path,
        _merge_dataset_results,
        _prepare_model_inputs,
        _remove_dataset_artifacts,
        _unwrap_runtime_types,
    )


class _SharedModel:
    """The minimal TransformerLens-like object consumed by SAEBench."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: str) -> _SharedModel:
        self.model.to(device)
        return self


class _StopAfterLastLayer(Exception):
    pass


def _valid_result_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("eval_result_metrics"), dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saebench-root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--progress-initial", type=int, default=0)
    parser.add_argument("--progress-total", type=int, default=1)
    parser.add_argument("--progress-run-initial", type=int, default=0)
    parser.add_argument("--progress-started-at", type=float, default=None)
    parser.add_argument("--source-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.saebench_root))
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if not jobs:
        return
    snapshots = {
        int(job["layer"]): json.loads(Path(job["snapshot"]).read_text(encoding="utf-8"))
        for job in jobs
    }
    first = snapshots[int(jobs[0]["layer"])]
    for snapshot in snapshots.values():
        for key in ("model_id", "model_revision", "saebench_model_name"):
            if snapshot[key] != first[key]:
                raise ValueError(f"multi-layer jobs disagree on {key}")
        if snapshot["evaluation_input_protocol"] != first["evaluation_input_protocol"]:
            raise ValueError("multi-layer jobs disagree on evaluation_input_protocol")

    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE  # type: ignore[attr-defined]

    import sae_bench.evals.sparse_probing.main as sparse_main  # type: ignore[import-not-found]
    import sae_bench.evals.sparse_probing.probe_training as probe_training  # type: ignore[import-not-found]
    from sae_bench.evals.sparse_probing.eval_config import (  # type: ignore[import-not-found]
        SparseProbingEvalConfig,
    )
    from sae_bench.sae_bench_utils import (  # type: ignore[import-not-found]
        activation_collection,
        dataset_info,
        dataset_utils,
    )

    for name in ("get_sae_meaned_activations", "create_meaned_model_activations"):
        setattr(
            activation_collection,
            name,
            _unwrap_runtime_types(getattr(activation_collection, name)),
        )
    # Shared caches are deliberately stored on CPU so multiple layers do not
    # occupy accelerator memory simultaneously. Preserve SAEBench's exact
    # mean-pooling rule while moving only each encoder batch to its device.
    activation_collection.get_sae_meaned_activations = _get_sae_meaned_cpu_cache

    tokenizer = AutoTokenizer.from_pretrained(first["model_id"], revision=first["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, settings["llm_dtype"])
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        first["model_id"],
        device="cuda",
        dtype=dtype,
        touch="auto",
        low_cpu_mem_usage=True,
        revision=first["model_revision"],
    )
    model.eval()
    hooked_model = _SharedModel(model, tokenizer)

    configs: dict[int, Any] = {}
    encoders: dict[int, dict[str, torch.nn.Module]] = {}
    feature_configs: dict[int, dict[str, dict[str, Any]]] = {}
    for job in jobs:
        layer = int(job["layer"])
        snapshot = snapshots[layer]
        selected = list(job["methods"])
        layer_encoders: dict[str, torch.nn.Module] = {}
        if "ica" in selected:
            layer_encoders["ica"] = ICAFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        if "sae" in selected:
            layer_encoders["sae"] = SAEFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        if "pca" in selected:
            layer_encoders["pca"] = PCAFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        if "random" in selected:
            layer_encoders["random"] = RandomFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        unknown = sorted(set(selected).difference(layer_encoders))
        if unknown:
            raise ValueError(f"requested methods are unavailable at layer {layer}: {unknown}")
        encoders[layer] = layer_encoders
        feature_configs[layer] = {
            name: asdict(cast(Any, encoder).cfg) for name, encoder in layer_encoders.items()
        }
        config = SparseProbingEvalConfig(model_name=snapshot["saebench_model_name"])
        config.dataset_names = list(settings["datasets"])
        config.k_values = list(settings["k_values"])
        config.probe_train_set_size = int(settings["probe_train_size"])
        config.probe_test_set_size = int(settings["probe_test_size"])
        config.context_length = int(settings["context_length"])
        config.random_seed = int(settings["random_seed"])
        config.llm_batch_size = int(settings["llm_batch_size"])
        config.sae_batch_size = int(settings["sae_batch_size"])
        config.llm_dtype = str(settings["llm_dtype"])
        config.lower_vram_usage = False
        configs[layer] = config

    datasets = list(settings["datasets"])
    existing = sum(
        _valid_result_file(
            _dataset_result_path(Path(job["output"]) / "saebench-datasets", index, dataset, method)
        )
        for job in jobs
        for index, dataset in enumerate(datasets)
        for method in job["methods"]
    )
    display = _BenchmarkDisplay(
        output=Path(jobs[0]["output"]),
        completed=args.progress_initial + existing,
        total=args.progress_total,
        run_initial=args.progress_run_initial,
        run_started_at=args.progress_started_at or time.time(),
        source_dirty=bool(args.source_dirty),
    )
    original_tqdm = sparse_main.tqdm
    sparse_main.tqdm = display.track_methods
    try:
        with display:
            for index, dataset in enumerate(datasets):
                missing_by_layer: dict[int, list[str]] = {}
                for job in jobs:
                    layer = int(job["layer"])
                    root = Path(job["output"]) / "saebench-datasets"
                    root.mkdir(parents=True, exist_ok=True)
                    missing_by_layer[layer] = [
                        method
                        for method in job["methods"]
                        if not _valid_result_file(
                            _dataset_result_path(root, index, dataset, method)
                        )
                    ]
                pending_layers = [layer for layer, missing in missing_by_layer.items() if missing]
                dataset_artifacts = args.artifacts / f"dataset_{index:02d}"
                if not pending_layers and not args.keep_artifacts:
                    _remove_dataset_artifacts(dataset_artifacts)
                    continue

                all_methods = list(
                    dict.fromkeys(method for job in jobs for method in job["methods"])
                )
                completed_for_dataset = {
                    method
                    for method in all_methods
                    if all(
                        not missing_by_layer[int(job["layer"])] or method not in job["methods"]
                        for job in jobs
                    )
                }
                display.set_dataset(
                    dataset,
                    index=index + 1,
                    total=len(datasets),
                    methods=all_methods,
                    completed_methods=completed_for_dataset,
                )
                display.set_phase(
                    "shared capture (layers "
                    + ",".join(str(layer) for layer in pending_layers)
                    + ")"
                )
                _prepare_shared_dataset_cache(
                    dataset=dataset,
                    layers=pending_layers,
                    snapshots=snapshots,
                    config=configs[pending_layers[0]],
                    model=model,
                    tokenizer=tokenizer,
                    artifacts=dataset_artifacts,
                    sparse_main=sparse_main,
                    probe_training=probe_training,
                    activation_collection=activation_collection,
                    dataset_info=dataset_info,
                    dataset_utils=dataset_utils,
                )

                for layer_position, layer in enumerate(pending_layers, start=1):
                    job = next(item for item in jobs if int(item["layer"]) == layer)
                    missing = missing_by_layer[layer]
                    display.set_dataset(
                        f"{dataset} · layer {layer} ({layer_position}/{len(pending_layers)})",
                        index=index + 1,
                        total=len(datasets),
                        methods=list(job["methods"]),
                        completed_methods=set(job["methods"]).difference(missing),
                    )
                    config = configs[layer]
                    config.dataset_names = [dataset]

                    class Factory:
                        @staticmethod
                        def from_pretrained_no_processing(
                            _: str, device: str, dtype: torch.dtype
                        ) -> Any:
                            del device, dtype
                            return hooked_model

                    sparse_main.HookedTransformer = Factory
                    result = sparse_main.run_eval(
                        config,
                        selected_saes=[(name, encoders[layer][name]) for name in missing],
                        device="cuda",
                        output_path=str(
                            _dataset_output(
                                Path(job["output"]) / "saebench-datasets",
                                index,
                                dataset,
                            )
                        ),
                        force_rerun=False,
                        clean_up_activations=False,
                        save_activations=False,
                        artifacts_path=str(dataset_artifacts),
                    )
                    del result
                if not args.keep_artifacts:
                    _remove_dataset_artifacts(dataset_artifacts)
    finally:
        sparse_main.tqdm = original_tqdm

    for job in jobs:
        layer = int(job["layer"])
        output = Path(job["output"])
        config = configs[layer]
        config.dataset_names = datasets
        final = output / "saebench"
        final.mkdir(parents=True, exist_ok=True)
        for method in job["methods"]:
            payloads = [
                json.loads(
                    _dataset_result_path(
                        output / "saebench-datasets", index, dataset, method
                    ).read_text(encoding="utf-8")
                )
                for index, dataset in enumerate(datasets)
            ]
            _atomic_json(
                final / f"{method}_custom_sae_eval_results.json",
                _merge_dataset_results(payloads, datasets),
            )
        methods = {
            name: json.loads(
                (final / f"{name}_custom_sae_eval_results.json").read_text(encoding="utf-8")
            )
            for name in ["ica", *snapshots[layer].get("baselines", {})]
            if (final / f"{name}_custom_sae_eval_results.json").is_file()
        }
        _atomic_json(output / "raw-result.json", {"methods": methods})
        _atomic_json(
            output / "worker.json",
            {
                "feature_configs": {
                    **_existing_feature_configs(output / "worker.json"),
                    **feature_configs[layer],
                },
                "preset": settings,
                "baselines": snapshots[layer].get("baselines", {}),
                "execution_order": "dataset_then_shared_multi_layer_capture",
            },
        )


def _atomic_json(path: Path, value: Any) -> None:
    """Validate JSON before replacing the last durable result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def _prepare_shared_dataset_cache(
    *,
    dataset: str,
    layers: list[int],
    snapshots: dict[int, dict[str, Any]],
    config: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    artifacts: Path,
    sparse_main: Any,
    probe_training: Any,
    activation_collection: Any,
    dataset_info: Any,
    dataset_utils: Any,
) -> None:
    cache_paths = {
        layer: _canonical_cache_path(artifacts, snapshots[layer], dataset) for layer in layers
    }
    missing_layers = [layer for layer, path in cache_paths.items() if not path.is_file()]
    if missing_layers:
        train, test = dataset_utils.get_multi_label_train_test_data(
            dataset,
            config.probe_train_set_size,
            config.probe_test_set_size,
            config.random_seed,
        )
        chosen = dataset_info.chosen_classes_per_dataset[dataset]
        train = dataset_utils.filter_dataset(train, chosen)
        test = dataset_utils.filter_dataset(test, chosen)
        train = dataset_utils.tokenize_data_dictionary(
            train, tokenizer, config.context_length, "cuda"
        )
        test = dataset_utils.tokenize_data_dictionary(
            test, tokenizer, config.context_length, "cuda"
        )
        protocol = snapshots[missing_layers[0]]["evaluation_input_protocol"]
        train_acts = _capture_layers(
            train, model, tokenizer, missing_layers, config.llm_batch_size, protocol
        )
        test_acts = _capture_layers(
            test, model, tokenizer, missing_layers, config.llm_batch_size, protocol
        )
    for layer in missing_layers:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        mean_train = _mean_activations_on_cuda(train_acts[layer], activation_collection)
        mean_test = _mean_activations_on_cuda(test_acts[layer], activation_collection)
        _, full_accuracy = probe_training.train_probe_on_activations(
            mean_train,
            mean_test,
            select_top_k=None,
            use_sklearn=False,
            batch_size=250,
            epochs=100,
            lr=1e-2,
        )
        llm_results: dict[str, Any] = {"llm_test_accuracy": full_accuracy}
        for k in config.k_values:
            _, accuracy = probe_training.train_probe_on_activations(
                mean_train, mean_test, select_top_k=k
            )
            llm_results[f"llm_top_{k}_test_accuracy"] = accuracy
        payload = {
            "train": {name: value.cpu() for name, value in train_acts[layer].items()},
            "test": {name: value.cpu() for name, value in test_acts[layer].items()},
            "llm_results": llm_results,
        }
        path = cache_paths[layer]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    # Recreate aliases even when a prior run already completed shared capture
    # but stopped before evaluating every method.
    for layer, path in cache_paths.items():
        for hook_name in _hook_aliases(snapshots[layer]):
            alias = _cache_path(artifacts, snapshots[layer], dataset, hook_name)
            if alias == path or alias.exists():
                continue
            alias.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, alias)


def _capture_layers(
    tokenized: dict[str, dict[str, torch.Tensor]],
    model: torch.nn.Module,
    tokenizer: Any,
    layers: list[int],
    batch_size: int,
    input_protocol: dict[str, Any],
) -> dict[int, dict[str, torch.Tensor]]:
    from tqdm import tqdm

    result: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    blocks = _blocks(model)
    final_layer = max(layers)
    for class_name, encoded in tokenized.items():
        tokens = encoded["input_ids"]
        chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        for start in tqdm(range(0, len(tokens), batch_size), desc="Collecting shared activations"):
            batch = tokens[start : start + batch_size].to(next(model.parameters()).device)
            model_batch, attention_mask, strip_prefix = _prepare_model_inputs(
                batch, tokenizer, input_protocol
            )
            captured: dict[int, torch.Tensor] = {}
            handles = []
            for layer in layers:

                def capture(
                    _: Any,
                    __: Any,
                    output: Any,
                    *,
                    layer: int = layer,
                    captured: dict[int, torch.Tensor] = captured,
                    strip_prefix: int = strip_prefix,
                ) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[layer] = hidden[:, strip_prefix:].detach()
                    if layer == final_layer:
                        raise _StopAfterLastLayer

                handles.append(blocks[layer].register_forward_hook(capture))
            try:
                with torch.inference_mode():
                    try:
                        model(
                            input_ids=model_batch,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                    except _StopAfterLastLayer:
                        pass
            finally:
                for handle in handles:
                    handle.remove()
            mask = ~(
                (batch == tokenizer.pad_token_id)
                | (batch == tokenizer.eos_token_id)
                | (batch == tokenizer.bos_token_id)
            )
            for layer in layers:
                chunks[layer].append((captured[layer] * mask[:, :, None]).cpu())
        for layer in layers:
            result[layer][class_name] = torch.cat(chunks[layer], dim=0)
    return result


@torch.no_grad()
def _get_sae_meaned_cpu_cache(
    activations: dict[str, torch.Tensor], encoder: Any, batch_size: int
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for class_name, values in activations.items():
        batches: list[torch.Tensor] = []
        for start in range(0, len(values), batch_size):
            source = values[start : start + batch_size]
            encoded = encoder.encode(source)
            nonzero = (source.sum(dim=-1) != 0).to(device=encoded.device, dtype=encoder.dtype)
            counts = nonzero.sum(dim=-1).clamp_min(1)
            pooled = (encoded * nonzero[:, :, None]).sum(dim=1) / counts[:, None]
            batches.append(pooled.to(dtype=encoder.dtype))
        result[class_name] = torch.cat(batches, dim=0)
    return result


def _mean_activations_on_cuda(
    activations: dict[str, torch.Tensor], activation_collection: Any
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for class_name, values in activations.items():
        result.update(
            activation_collection.create_meaned_model_activations({class_name: values.to("cuda")})
        )
    return result


def _canonical_cache_path(artifacts: Path, snapshot: dict[str, Any], dataset: str) -> Path:
    layer = int(snapshot["layer"])
    return _cache_path(artifacts, snapshot, dataset, f"blocks.{layer}.hook_resid_post")


def _cache_path(artifacts: Path, snapshot: dict[str, Any], dataset: str, hook_name: str) -> Path:
    filename = f"{dataset}_activations.pt".replace("/", "_")
    return (
        artifacts / "sparse_probing" / str(snapshot["saebench_model_name"]) / hook_name / filename
    )


def _hook_aliases(snapshot: dict[str, Any]) -> set[str]:
    layer = int(snapshot["layer"])
    names = {f"blocks.{layer}.hook_resid_post"}
    sae = snapshot.get("baselines", {}).get("sae")
    if sae:
        names.add(str(sae["hook_name_template"]).format(layer=layer))
    return names


def _existing_feature_configs(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("feature_configs", {})
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
