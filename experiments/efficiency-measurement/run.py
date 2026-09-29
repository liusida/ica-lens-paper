#!/usr/bin/env python3
"""Measure current ICA and public-SAE memory and feature-pass latency."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from icalens import ICALens
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._sae import SAEFeatureEncoder
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.saebench_sparse_probing import (
    _prepare_layer_baselines,
    _resolve_baselines,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORMAT = "icalens.efficiency_measurement"
SCHEMA_VERSION = 1
MODEL_SPECS = {
    "gpt2": ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k",
    "gemma2": ROOT / "local-icalens-models/official/icalens-gemma-2-2b-pile10k",
    "qwen9b": ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k",
}
MODEL_TITLES = {
    "gpt2": "GPT-2 Small",
    "gemma2": "Gemma 2 2B",
    "qwen9b": "Qwen 3.5 9B Base",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ICACodec(torch.nn.Module):
    """Resident torch form of the public ICALens transform/inverse transform."""

    def __init__(self, lens: ICALens, layer: int, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        artifact = lens._get_layer(layer)
        assert artifact.center is not None
        assert artifact.reading_matrix is not None
        assert artifact.writing_matrix is not None
        self.register_buffer("center", torch.from_numpy(artifact.center.copy()))
        self.register_buffer("reading", torch.from_numpy(artifact.reading_matrix.copy()))
        self.register_buffer("writing", torch.from_numpy(artifact.writing_matrix.copy()))
        if artifact.preprocessing_center is not None:
            self.register_buffer(
                "preprocessing_center",
                torch.from_numpy(artifact.preprocessing_center.copy()),
            )
        else:
            self.preprocessing_center = None
        self.row_normalize = bool(lens.row_normalize)
        self.norm_eps = float(lens.norm_eps)
        self.to(device=device, dtype=dtype)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        work = activations
        if self.preprocessing_center is not None:
            work = work - self.preprocessing_center
        if self.row_normalize:
            work = work / torch.linalg.vector_norm(work, dim=-1, keepdim=True).clamp_min(
                self.norm_eps
            )
        return (work - self.center) @ self.reading.T

    def decode(self, scores: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
        del reference
        return scores @ self.writing.T + self.center


def tensor_bytes(module: torch.nn.Module) -> int:
    """Count unique persistent parameter and buffer storage."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in (*module.parameters(), *module.buffers()):
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            seen.add(identity)
            total += storage.nbytes()
    return total


@torch.inference_mode()
def time_cuda(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    batch_size: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    elapsed_us = np.empty(repeats, dtype=np.float64)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for index in range(repeats):
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        elapsed_us[index] = float(start.elapsed_time(end)) * 1000.0 / batch_size
        del result
    return {
        "unit": "microseconds_per_activation",
        "median": float(np.median(elapsed_us)),
        "mean": float(np.mean(elapsed_us)),
        "p25": float(np.percentile(elapsed_us, 25)),
        "p75": float(np.percentile(elapsed_us, 75)),
        "trials": elapsed_us.tolist(),
    }


@torch.inference_mode()
def peak_extra_bytes(operation: Callable[[], torch.Tensor]) -> int:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return max(0, int(peak - baseline))


def benchmark_codec(
    codec: Any,
    inputs: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        codes = codec.encode(inputs)
        torch.cuda.synchronize()
    operations = {
        "encode": lambda: codec.encode(inputs),
        "decode": lambda codes=codes: codec.decode(codes, reference=inputs),
        "forward": lambda: codec.decode(codec.encode(inputs), reference=inputs),
    }
    timings = {
        name: time_cuda(
            operation,
            warmup=warmup,
            repeats=repeats,
            batch_size=len(inputs),
        )
        for name, operation in operations.items()
    }
    workspace = {name: peak_extra_bytes(operation) for name, operation in operations.items()}
    result = {
        "parameter_bytes": tensor_bytes(codec),
        "code_width": int(codes.shape[-1]),
        "latency": timings,
        "peak_extra_cuda_bytes": workspace,
    }
    del codes
    return result


def result_is_valid(path: Path, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity:
            return False
        for method in ("ica", "sae"):
            record = value["methods"][method]
            if int(record["parameter_bytes"]) <= 0 or int(record["code_width"]) <= 0:
                return False
            for operation in ("encode", "decode", "forward"):
                timing = record["latency"][operation]
                trials = np.asarray(timing["trials"], dtype=np.float64)
                if trials.shape != (int(identity["timing"]["repeats"]),):
                    return False
                if not np.isfinite(trials).all() or np.any(trials <= 0):
                    return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=tuple(MODEL_SPECS)
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("efficiency measurement requires CUDA")
    if min(args.batch_size, args.warmup, args.repeats) <= 0:
        raise ValueError("batch size, warmup, and repeats must be positive")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    labels = list(args.models)
    lenses = {label: ICALens.from_pretrained(MODEL_SPECS[label]) for label in labels}
    resolved_models: dict[str, Any] = {}
    prepared_baselines: dict[str, Any] = {}
    for label, lens in lenses.items():
        layers = list(lens.available_layers)
        layer = int(layers[len(layers) // 2])
        lens_path = MODEL_SPECS[label].resolve()
        manifest_path = lens_path / "icalens.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_path = lens_path / manifest["layers"][str(layer)]["file"]
        baseline = _prepare_layer_baselines(_resolve_baselines(lens.model_id, "sae"), layer=layer)[
            "sae"
        ]
        prepared_baselines[label] = baseline
        resolved_models[label] = {
            "title": MODEL_TITLES[label],
            "model_id": lens.model_id,
            "model_revision": lens.model_revision,
            "layer": layer,
            "hidden_size": int(lens.hidden_size or 0),
            "ica_lens": str(lens_path),
            "ica_artifact": str(artifact_path),
            "ica_artifact_sha256": sha256(artifact_path),
            "sae": {
                key: baseline.get(key)
                for key in (
                    "name",
                    "repo_id",
                    "revision",
                    "checkpoint",
                    "checkpoint_format",
                    "width",
                    "activation",
                    "top_k",
                    "normalize_activations",
                    "apply_b_dec_to_input",
                )
            },
            "sae_weights_file": str(baseline["weights_file"]),
            "sae_weights_sha256": sha256(Path(baseline["weights_file"])),
        }
    resolved = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "device": args.device,
        "dtype": args.dtype,
        "input": "deterministic standard-normal synthetic residual vectors",
        "timing": {
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cuda_synchronize": True,
            "summary": "median CUDA-event time divided by batch size",
        },
        "memory": "unique persistent parameter and buffer storage used for encode and decode",
        "layer_selection": "middle available ICA layer",
        "models": resolved_models,
    }
    output = args.output.expanduser().resolve()
    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="measuring")
    completed = set()
    for label in labels:
        path = output / f"{label}.json"
        identity = {key: resolved[key] for key in resolved if key != "models"} | {
            "model": resolved_models[label]
        }
        if not args.force and result_is_valid(path, identity):
            completed.add(label)

    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · efficiency measurement",
        completed=len(completed),
        total=len(labels),
        completed_unit_ids=completed,
        unit_label="models",
        detail_filename="efficiency-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        for model_index, label in enumerate(labels):
            if label in completed:
                log(f"{label}: reusing validated result.")
                continue
            lens = lenses[label]
            specification = resolved_models[label]
            layer = int(specification["layer"])
            display.phase("Benchmarking current codecs", model=label, layer=layer)
            generator = torch.Generator(device=args.device)
            generator.manual_seed(1729 + model_index)
            inputs = torch.randn(
                (args.batch_size, int(specification["hidden_size"])),
                generator=generator,
                device=args.device,
                dtype=dtype,
            )
            snapshot = {
                "hidden_size": specification["hidden_size"],
                "layer": layer,
                "saebench_model_name": lens.model_id,
                "baselines": {"sae": prepared_baselines[label]},
            }
            methods: dict[str, Any] = {}
            try:
                for method in ("ica", "sae"):
                    codec = (
                        ICACodec(lens, layer, args.device, dtype)
                        if method == "ica"
                        else SAEFeatureEncoder(snapshot, device=args.device, dtype=dtype)
                    )
                    methods[method] = benchmark_codec(
                        codec, inputs, warmup=args.warmup, repeats=args.repeats
                    )
                    log(
                        f"{label} {method}: {methods[method]['parameter_bytes'] / 2**20:.2f} MiB; "
                        f"encode {methods[method]['latency']['encode']['median']:.3f}, "
                        f"decode {methods[method]['latency']['decode']['median']:.3f}, "
                        "forward "
                        f"{methods[method]['latency']['forward']['median']:.3f} us/activation."
                    )
                    del codec
                    gc.collect()
                    torch.cuda.empty_cache()
                identity = {key: resolved[key] for key in resolved if key != "models"} | {
                    "model": specification
                }
                path = output / f"{label}.json"
                atomic_write_json(path, {"identity": identity, "methods": methods})
                if not result_is_valid(path, identity):
                    raise ValueError(f"new result failed validation: {path}")
                display.complete_unit(label, refresh=True)
            finally:
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
        run.set_status("complete", complete=True)
        log(f"Experiment complete: {output}")


if __name__ == "__main__":
    main()
