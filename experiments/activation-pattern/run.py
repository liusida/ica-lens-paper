#!/usr/bin/env python3
"""Measure token-level ICA scores and SAE activations on one shared sentence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens.analysis import capture
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
FORMAT = "icalens.activation_pattern"
SCHEMA_VERSION = 2
MAX_FEATURE_RANK = 5
DEFAULT_SENTENCE = (
    "Maya stopped at the bank before the trip, waiting in line to deposit a check "
    "and withdraw enough cash for the weekend."
)
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


def identity_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def representative_layers(lens: ICALens) -> list[int]:
    available = list(lens.available_layers)
    if not available:
        raise ValueError("ICA Lens has no fitted layers")
    selected = [available[0], available[len(available) // 2], available[-1]]
    return list(dict.fromkeys(selected))


def sae_identity(config: dict[str, Any], layer: int) -> dict[str, Any]:
    checkpoints = config.get("layer_checkpoints")
    checkpoint = (
        str(checkpoints[str(layer)])
        if isinstance(checkpoints, dict)
        else str(config["checkpoint_template"]).format(layer=layer)
    )
    return {
        key: config[key]
        for key in (
            "name",
            "repo_id",
            "revision",
            "checkpoint_format",
            "width",
            "activation",
            "top_k",
            "apply_b_dec_to_input",
            "normalize_activations",
        )
        if key in config
    } | {"checkpoint": checkpoint}


def resolved_configuration(
    labels: list[str],
    lenses: dict[str, ICALens],
    layers: dict[str, list[int]],
    sentence: str,
) -> dict[str, Any]:
    models = {}
    for label in labels:
        lens = lenses[label]
        lens_path = MODEL_SPECS[label].resolve()
        manifest_path = lens_path / "icalens.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        baseline = _resolve_baselines(lens.model_id, "sae")["sae"]
        layer_dependencies = {}
        for layer in layers[label]:
            artifact_path = lens_path / manifest["layers"][str(layer)]["file"]
            layer_dependencies[str(layer)] = {
                "ica_artifact": str(artifact_path),
                "ica_artifact_sha256": sha256(artifact_path),
                "ica_components": int(manifest["layers"][str(layer)]["n_components"]),
                "sae": sae_identity(baseline, layer),
            }
        models[label] = {
            "title": MODEL_TITLES[label],
            "lens": str(lens_path),
            "lens_manifest_sha256": sha256(manifest_path),
            "model_id": lens.model_id,
            "model_revision": lens.model_revision,
            "activation_site": lens.activation_site,
            "layers": layers[label],
            "layer_dependencies": layer_dependencies,
        }
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "sentence": sentence,
        "tokenization": {
            "add_special_tokens": False,
            "document_framing": "use each ICA layer's recorded fitting policy",
            "reported_tokens": "exclude the framing token",
        },
        "layer_selection": "first, index floor(number of layers / 2), last",
        "ica_quantity": "signed score from ICALens.transform",
        "sae_quantity": "native checkpoint post-activation before decoder-row-norm scaling",
        "cached_feature_ranks": MAX_FEATURE_RANK,
        "feature_selection": (
            "within each method, union of features attaining ranks 1 through 5 by "
            "absolute ICA score or positive SAE activation at any reported token"
        ),
        "models": models,
    }


def unit_identity(resolved: dict[str, Any], label: str, layer: int) -> dict[str, Any]:
    model = resolved["models"][label]
    return {
        key: resolved[key]
        for key in (
            "format",
            "schema_version",
            "sentence",
            "tokenization",
            "layer_selection",
            "ica_quantity",
            "sae_quantity",
            "cached_feature_ranks",
            "feature_selection",
        )
    } | {
        "model": {
            key: model[key]
            for key in (
                "lens_manifest_sha256",
                "model_id",
                "model_revision",
                "activation_site",
            )
        },
        "layer": layer,
        "dependencies": model["layer_dependencies"][str(layer)],
    }


def ordered_union(indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        list(dict.fromkeys(int(value) for value in indices if value >= 0)),
        dtype=np.int64,
    )


def validate_checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as raw:
            data = {key: raw[key] for key in raw.files}
    except (OSError, KeyError, ValueError):
        return None
    required = {
        "identity_sha256",
        "token_ids",
        "positions",
        "tokens",
        "token_texts",
        "token_labels",
        "ica_feature_ids",
        "ica_scores",
        "ica_top_feature",
        "ica_top_features",
        "sae_feature_ids",
        "sae_activations",
        "sae_top_feature",
        "sae_top_features",
    }
    if not required.issubset(data):
        return None
    token_count = len(data["token_ids"])
    ica_width = int(identity["dependencies"]["ica_components"])
    sae_width = int(identity["dependencies"]["sae"]["width"])
    if (
        str(data["identity_sha256"]) != identity_sha256(identity)
        or data["positions"].shape != (token_count,)
        or data["tokens"].shape != (token_count,)
        or data["token_texts"].shape != (token_count,)
        or data["token_labels"].shape != (token_count,)
        or data["ica_scores"].shape != (len(data["ica_feature_ids"]), token_count)
        or data["sae_activations"].shape != (len(data["sae_feature_ids"]), token_count)
        or data["ica_top_feature"].shape != (token_count,)
        or data["ica_top_features"].shape != (token_count, MAX_FEATURE_RANK)
        or data["sae_top_feature"].shape != (token_count,)
        or data["sae_top_features"].shape != (token_count, MAX_FEATURE_RANK)
        or not np.isfinite(data["ica_scores"]).all()
        or not np.isfinite(data["sae_activations"]).all()
        or np.any(data["sae_activations"] < 0)
        or np.any((data["ica_feature_ids"] < 0) | (data["ica_feature_ids"] >= ica_width))
        or np.any((data["sae_feature_ids"] < 0) | (data["sae_feature_ids"] >= sae_width))
        or np.any((data["ica_top_feature"] < 0) | (data["ica_top_feature"] >= ica_width))
        or np.any((data["sae_top_feature"] < -1) | (data["sae_top_feature"] >= sae_width))
        or np.any((data["ica_top_features"] < 0) | (data["ica_top_features"] >= ica_width))
        or np.any((data["sae_top_features"] < -1) | (data["sae_top_features"] >= sae_width))
        or not np.array_equal(data["ica_top_feature"], data["ica_top_features"][:, 0])
        or not np.array_equal(data["sae_top_feature"], data["sae_top_features"][:, 0])
        or set(int(value) for value in data["ica_feature_ids"])
        != set(int(value) for value in data["ica_top_features"].ravel())
        or set(int(value) for value in data["sae_feature_ids"])
        != set(int(value) for value in data["sae_top_features"].ravel() if value >= 0)
    ):
        return None
    return data


def atomic_write_checkpoint(
    path: Path, identity: dict[str, Any], payload: dict[str, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        identity_sha256=np.asarray(identity_sha256(identity)),
        **payload,
    )
    if validate_checkpoint(temporary, identity) is None:
        raise ValueError(f"temporary checkpoint failed validation: {temporary}")
    temporary.replace(path)


def load_model(lens: ICALens, device: str) -> torch.nn.Module:
    if not device.startswith("cuda"):
        raise ValueError("this GB10 experiment currently requires a CUDA device")
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        lens.model_id,
        revision=lens.model_revision,
        device=device,
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def measure_layer(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    baseline: dict[str, Any],
    sentence: str,
    device: str,
) -> dict[str, np.ndarray]:
    captured = capture(
        lens,
        sentence,
        layer=layer,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    ica = lens.transform(captured.activations.float(), layer=layer)
    if not isinstance(ica, torch.Tensor):
        ica = torch.as_tensor(ica)
    ica = ica.detach().cpu().float().numpy()
    ica_top_features = np.argsort(-np.abs(ica), axis=1, kind="stable")[:, :MAX_FEATURE_RANK].astype(
        np.int64
    )
    ica_ids = ordered_union(ica_top_features.ravel())

    prepared = _prepare_layer_baselines({"sae": baseline}, layer=layer)
    snapshot = {
        "hidden_size": int(lens.hidden_size or captured.activations.shape[-1]),
        "layer": layer,
        "saebench_model_name": lens.model_id,
        "baselines": prepared,
    }
    encoder = SAEFeatureEncoder(snapshot, device=device, dtype=torch.float32)
    scaled_sae = encoder.encode(captured.activations.float())
    sae = (scaled_sae / encoder.decoder_norms.clamp_min(1e-12)).detach().cpu().float().numpy()
    sae_top_features = np.argsort(-sae, axis=1, kind="stable")[:, :MAX_FEATURE_RANK].astype(
        np.int64
    )
    sae_top_values = np.take_along_axis(sae, sae_top_features, axis=1)
    sae_top_features[sae_top_values <= 0] = -1
    sae_ids = ordered_union(sae_top_features.ravel())

    return {
        "token_ids": captured.token_ids.cpu().numpy().astype(np.int64),
        "positions": captured.positions.cpu().numpy().astype(np.int64),
        "tokens": np.asarray(captured.tokens),
        "token_texts": np.asarray(captured.token_texts),
        "token_labels": np.asarray(captured.token_labels),
        "ica_feature_ids": ica_ids,
        "ica_scores": ica[:, ica_ids].T.astype(np.float32),
        "ica_top_feature": ica_top_features[:, 0],
        "ica_top_features": ica_top_features,
        "sae_feature_ids": sae_ids,
        "sae_activations": sae[:, sae_ids].T.astype(np.float32),
        "sae_top_feature": sae_top_features[:, 0],
        "sae_top_features": sae_top_features,
    }


def write_summary(
    output: Path,
    resolved: dict[str, Any],
    units: list[tuple[str, int]],
    cached: dict[tuple[str, int], dict[str, np.ndarray]],
) -> None:
    rows = []
    model_tokens: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {}
    for label, layer in units:
        data = cached[label, layer]
        token_identity = (
            tuple(int(value) for value in data["token_ids"]),
            tuple(str(value) for value in data["token_texts"]),
        )
        previous = model_tokens.setdefault(label, token_identity)
        if token_identity != previous:
            raise ValueError(f"token alignment differs across {label} layers")
        rows.append(
            {
                "model": label,
                "layer": layer,
                "n_tokens": len(data["token_ids"]),
                "selected_ica_features": len(data["ica_feature_ids"]),
                "selected_sae_features": len(data["sae_feature_ids"]),
                "checkpoint": str(output / label / f"layer_{layer:02d}.npz"),
            }
        )
    atomic_write_json(
        output / "summary.json",
        {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "sentence": resolved["sentence"],
            "models": list(resolved["models"]),
            "rows": rows,
        },
    )
    path = output / "summary.csv"
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=tuple(MODEL_SPECS),
    )
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    labels = list(args.models)
    lenses = {label: ICALens.from_pretrained(MODEL_SPECS[label]) for label in labels}
    for label, lens in lenses.items():
        if lens.activation_site != "resid_post":
            raise ValueError(f"{label} lens does not use resid_post activations")
    layers = {label: representative_layers(lenses[label]) for label in labels}
    resolved = resolved_configuration(labels, lenses, layers, args.sentence)
    output = args.output.expanduser().resolve()
    source = source_provenance()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="measuring",
    )
    units = [(label, layer) for label in labels for layer in layers[label]]
    cached: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    completed = set()
    for label, layer in units:
        identity = unit_identity(resolved, label, layer)
        checkpoint = output / label / f"layer_{layer:02d}.npz"
        data = None if args.force else validate_checkpoint(checkpoint, identity)
        if data is not None:
            cached[label, layer] = data
            completed.add((label, layer))

    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · activation-pattern comparison",
        completed=len(completed),
        total=len(units),
        completed_unit_ids=completed,
        unit_label="model-layers",
        recent_label="Recent activation-pattern output",
        detail_filename="activation-pattern-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        log(f"Validated configuration: {len(labels)} models, {len(units)} layers.")
        for label in labels:
            pending = [layer for layer in layers[label] if (label, layer) not in completed]
            if not pending:
                log(f"{label}: all requested layers complete; reusing checkpoints.")
                continue
            lens = lenses[label]
            display.phase("Loading language model", model=label)
            tokenizer = AutoTokenizer.from_pretrained(
                lens.model_id,
                revision=lens.model_revision,
                use_fast=True,
            )
            model = load_model(lens, args.device)
            baseline = _resolve_baselines(lens.model_id, "sae")["sae"]
            try:
                for layer in pending:
                    display.phase("Measuring token responses", model=label, layer=layer)
                    data = measure_layer(
                        lens=lens,
                        model=model,
                        tokenizer=tokenizer,
                        layer=layer,
                        baseline=baseline,
                        sentence=args.sentence,
                        device=args.device,
                    )
                    identity = unit_identity(resolved, label, layer)
                    checkpoint = output / label / f"layer_{layer:02d}.npz"
                    atomic_write_checkpoint(checkpoint, identity, data)
                    validated = validate_checkpoint(checkpoint, identity)
                    if validated is None:
                        raise ValueError(f"new checkpoint failed validation: {checkpoint}")
                    cached[label, layer] = validated
                    display.complete_unit((label, layer), refresh=True)
                    log(
                        f"{label} layer {layer}: checkpointed {len(data['token_ids'])} tokens, "
                        f"{len(data['ica_feature_ids'])} ICA and "
                        f"{len(data['sae_feature_ids'])} SAE selected features."
                    )
            finally:
                del model
                torch.cuda.empty_cache()

        display.phase("Writing summaries")
        write_summary(output, resolved, units, cached)
        run.set_status("complete", complete=True)
        log(f"Experiment complete: {output}")


if __name__ == "__main__":
    main()
