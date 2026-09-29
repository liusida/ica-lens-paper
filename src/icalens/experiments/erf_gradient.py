"""Run the gradient effective-receptive-field experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens, __version__
from icalens._capture import transformer_blocks
from icalens.cli._status import log

from ._display import ExperimentDisplay
from ._run import ResumableRun, atomic_write_json
from ._source_provenance import source_provenance, warn_if_dirty

SCHEMA_VERSION = 1
METHOD = "squared-input-embedding-gradient-geometric-distance"


class _CapturedLayer(RuntimeError):
    """Stop a model forward after capturing the requested residual stream."""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="icalens experiment erf-gradient", description=__doc__)
    parser.add_argument(
        "--lens",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each local profiled ICA Lens.",
    )
    parser.add_argument("--components-per-layer", type=int, default=100)
    parser.add_argument("--occurrences-per-component", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.components_per_layer, args.occurrences_per_component, args.batch_size) < 1:
        raise ValueError("component, occurrence, and batch counts must be positive")

    lens_specs = _parse_lenses(args.lens)
    output = args.output.expanduser().resolve()
    cached = _load_prepared_run(output, lens_specs=lens_specs, args=args)
    lenses = {label: ICALens.from_pretrained(path) for label, path in lens_specs.items()}
    if cached is None:
        selections: dict[str, dict[str, list[int]]] = {}
        prepared_inputs: dict[str, dict[str, dict[str, Any]]] = {}
        for label, lens in lenses.items():
            selection, prepared = _prepare_component_inputs(
                lens,
                label=label,
                count=args.components_per_layer,
                seed=args.seed,
                occurrence_limit=args.occurrences_per_component,
            )
            selections[label] = selection
            prepared_inputs[label] = prepared
        resolved = _resolved_configuration(
            lens_specs=lens_specs,
            lenses=lenses,
            selections=selections,
            components_per_layer=args.components_per_layer,
            occurrences_per_component=args.occurrences_per_component,
            seed=args.seed,
        )
    else:
        resolved, selections, prepared_inputs = cached
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    source = source_provenance()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="measuring",
    )
    if cached is None:
        _write_prepared_inputs(output, prepared_inputs)
    units = _component_units(selections)
    completed_ids = {
        unit_id
        for unit_id, label, layer, component in units
        if _component_checkpoint_valid(
            _component_path(output, label, layer, component),
            label=label,
            layer=layer,
            component=component,
            method=METHOD,
        )
    }
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · gradient effective receptive field",
        completed=len(completed_ids),
        total=len(units),
        completed_unit_ids=completed_ids,
        unit_label="components",
        recent_label="Recent ERF output",
        detail_filename="erf-gradient-detail.log",
        source_dirty=bool(source.get("dirty")),
    )
    with display:
        warn_if_dirty(source)
        log(
            f"Validated ERF configuration: {len(lenses)} models, {len(units)} components, "
            f"up to {args.occurrences_per_component} occurrences each."
        )
        for label, lens in lenses.items():
            pending = [
                (int(layer), component)
                for layer, components in _layers_in_numeric_order(selections[label])
                for component in components
                if _unit_id(label, int(layer), component) not in completed_ids
            ]
            if not pending:
                log(f"{label}: all selected components already complete; skipping model load.")
                continue
            display.phase("Loading model", model=label)
            log(f"Loading {label}: {lens.model_id}@{lens.model_revision}.")
            model = _load_model(lens, device=args.device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            tokenizer = AutoTokenizer.from_pretrained(
                lens.model_id,
                revision=lens.model_revision,
                use_fast=True,
                trust_remote_code=True,
            )
            datasets: dict[str, Any] = {}
            token_cache: dict[tuple[str, int, int], list[int]] = {}
            try:
                for layer, component in pending:
                    display.phase(
                        "Measuring gradients", model=label, layer=layer, component=f"C{component}"
                    )
                    prepared_component = prepared_inputs[label][str(layer)][str(component)]
                    direction = str(prepared_component["tail_direction"])
                    occurrences = prepared_component["occurrences"]
                    result = _measure_component(
                        lens=lens,
                        model=model,
                        tokenizer=tokenizer,
                        datasets=datasets,
                        token_cache=token_cache,
                        layer=layer,
                        component=component,
                        direction=direction,
                        occurrences=occurrences,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                    result.update(model_label=label, method=METHOD)
                    path = _component_path(output, label, layer, component)
                    atomic_write_json(path, result)
                    display.complete_unit(_unit_id(label, layer, component), refresh=True)
                    log(
                        f"Checkpointed {label} layer {layer} C{component}: "
                        f"median ERF {result['gradient_erf_median']:.4f}, "
                        f"n={result['n_occurrences']}."
                    )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        display.phase("Summarizing")
        _write_summaries(output, units)
        run.set_status("complete", complete=True)
        log(f"Gradient ERF experiment complete: {output}")


def _parse_lenses(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--lens must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or not label.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"invalid lens label {label!r}")
        if label in parsed:
            raise ValueError(f"duplicate lens label {label!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"lens path does not exist: {path}")
        parsed[label] = path
    return parsed


def _prepare_component_inputs(
    lens: ICALens,
    *,
    label: str,
    count: int,
    seed: int,
    occurrence_limit: int,
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    selections: dict[str, list[int]] = {}
    prepared: dict[str, dict[str, Any]] = {}
    for layer in lens.available_layers:
        profile = lens._get_profile(lens._get_layer(layer))
        eligible: list[int] = []
        by_component: dict[int, dict[str, Any]] = {}
        for item in profile["components"]:
            component = int(item["component"])
            direction = str(item["tail_direction"])
            occurrences = item["examples"][direction]["occurrences"][:occurrence_limit]
            if occurrences:
                eligible.append(component)
                by_component[component] = {
                    "component": component,
                    "tail_direction": direction,
                    "occurrences": occurrences,
                }
        if len(eligible) < count:
            raise ValueError(
                f"{label} layer {layer} has only {len(eligible)} profiled components with "
                f"dominant-tail occurrences; cannot sample {count}"
            )
        layer_seed = _stable_seed(seed, label, layer)
        rng = np.random.default_rng(layer_seed)
        selected = sorted(int(value) for value in rng.choice(eligible, size=count, replace=False))
        selections[str(layer)] = selected
        prepared[str(layer)] = {str(component): by_component[component] for component in selected}
    return selections, prepared


def _resolved_configuration(
    *,
    lens_specs: dict[str, Path],
    lenses: dict[str, ICALens],
    selections: dict[str, dict[str, list[int]]],
    components_per_layer: int,
    occurrences_per_component: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format": "icalens.erf_gradient",
        "schema_version": SCHEMA_VERSION,
        "icalens_version": __version__,
        "method": METHOD,
        "influence": "squared L2 norm of d(component score)/d(input token embedding)",
        "distance": "target token is 1; preceding lexical positions increase by 1",
        "sample_erf": "exp(sum_i normalized_influence_i * log(distance_i))",
        "component_erf": "median sample_erf over stored dominant-tail occurrences",
        "document_framing": "reported separately and excluded from lexical normalization",
        "components_per_layer": components_per_layer,
        "occurrences_per_component": occurrences_per_component,
        "seed": seed,
        "lens_order": list(lens_specs),
        "lenses": {
            label: _lens_identity(path, lenses[label], selections[label])
            for label, path in lens_specs.items()
        },
        "selections": selections,
    }


def _prepared_layer_path(output: Path, label: str, layer: int) -> Path:
    return output / "prepared" / label / f"layer_{layer:02d}.json"


def _write_prepared_inputs(
    output: Path, prepared_inputs: dict[str, dict[str, dict[str, Any]]]
) -> None:
    for label, layers in prepared_inputs.items():
        for layer_text, components in layers.items():
            atomic_write_json(
                _prepared_layer_path(output, label, int(layer_text)),
                {
                    "format": "icalens.erf_gradient.prepared_layer",
                    "schema_version": SCHEMA_VERSION,
                    "model_label": label,
                    "layer": int(layer_text),
                    "components": components,
                },
            )


def _load_prepared_run(
    output: Path, *, lens_specs: dict[str, Path], args: argparse.Namespace
) -> (
    tuple[
        dict[str, Any],
        dict[str, dict[str, list[int]]],
        dict[str, dict[str, dict[str, Any]]],
    ]
    | None
):
    run_path = output / "run.json"
    if not run_path.is_file():
        return None
    try:
        state = json.loads(run_path.read_text(encoding="utf-8"))
        resolved = state["resolved"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{run_path} has no valid resolved configuration") from error
    requested = {
        "format": "icalens.erf_gradient",
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "components_per_layer": args.components_per_layer,
        "occurrences_per_component": args.occurrences_per_component,
        "seed": args.seed,
        "lens_order": list(lens_specs),
        "lens_paths": {label: str(path) for label, path in lens_specs.items()},
    }
    existing = {
        "format": resolved.get("format"),
        "schema_version": resolved.get("schema_version"),
        "method": resolved.get("method"),
        "components_per_layer": resolved.get("components_per_layer"),
        "occurrences_per_component": resolved.get("occurrences_per_component"),
        "seed": resolved.get("seed"),
        "lens_order": resolved.get("lens_order"),
        "lens_paths": {
            label: value.get("path") for label, value in resolved.get("lenses", {}).items()
        },
    }
    if existing != requested:
        differences = [
            f"{key}: {existing[key]!r} -> {requested[key]!r}"
            for key in requested
            if existing[key] != requested[key]
        ]
        raise ValueError(
            f"{run_path} belongs to a different configuration "
            f"({'; '.join(differences)}); choose another output"
        )
    _validate_lens_fingerprints(resolved, lens_specs)
    selections = resolved.get("selections")
    if not isinstance(selections, dict):
        raise ValueError(f"{run_path} has no valid component selections")
    prepared_inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for label, layers in selections.items():
        prepared_inputs[label] = {}
        for layer_text, selected in layers.items():
            path = _prepared_layer_path(output, label, int(layer_text))
            if not path.is_file():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                components = payload["components"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                return None
            if (
                payload.get("format") != "icalens.erf_gradient.prepared_layer"
                or payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("model_label") != label
                or payload.get("layer") != int(layer_text)
                or sorted(int(component) for component in components) != sorted(selected)
            ):
                return None
            prepared_inputs[label][layer_text] = components
    return resolved, selections, prepared_inputs


def _validate_lens_fingerprints(resolved: dict[str, Any], lens_specs: dict[str, Path]) -> None:
    """Reject prepared inputs whose Lens manifest or profiles have changed."""
    identities = resolved.get("lenses")
    if not isinstance(identities, dict):
        raise ValueError("prepared ERF run has no Lens fingerprints")
    differences: list[str] = []
    for label, path in lens_specs.items():
        identity = identities.get(label)
        if not isinstance(identity, dict):
            differences.append(f"{label}: identity is missing")
            continue
        manifest_path = path / "icalens.json"
        if not manifest_path.is_file():
            differences.append(f"{label}: icalens.json is missing")
            continue
        recorded_manifest = identity.get("manifest_sha256")
        if recorded_manifest != _sha256(manifest_path):
            differences.append(f"{label}: Lens manifest changed")
        profile_hashes = identity.get("profile_sha256")
        if not isinstance(profile_hashes, dict):
            differences.append(f"{label}: profile fingerprints are missing")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layers = manifest.get("layers", {})
        for layer_text, recorded_hash in profile_hashes.items():
            entry = layers.get(str(layer_text))
            relative = entry.get("component_profile") if isinstance(entry, dict) else None
            profile_path = path / relative if isinstance(relative, str) else None
            if profile_path is None or not profile_path.is_file():
                differences.append(f"{label} layer {layer_text}: profile is missing")
            elif _sha256(profile_path) != recorded_hash:
                differences.append(f"{label} layer {layer_text}: profile changed")
    if differences:
        preview = "; ".join(differences[:8])
        if len(differences) > 8:
            preview += f"; and {len(differences) - 8} more"
        raise ValueError(
            "prepared ERF inputs are stale because their Lens dependencies changed ("
            + preview
            + "); choose another output or deliberately clear this derived run"
        )


def _stable_seed(seed: int, label: str, layer: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}:{layer}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _lens_identity(path: Path, lens: ICALens, selection: dict[str, list[int]]) -> dict[str, Any]:
    profile_hashes = {}
    for layer_text in selection:
        artifact = lens._get_layer(int(layer_text))
        if not artifact.profile_file:
            raise ValueError(f"{path} layer {layer_text} has no component profile")
        profile_path = path / artifact.profile_file
        profile_hashes[layer_text] = _sha256(profile_path)
    return {
        "path": str(path),
        "model": {"repo_id": lens.model_id, "revision": lens.model_revision},
        "activation_site": lens.activation_site,
        "layer_indexing": lens.layer_indexing,
        "layers": list(lens.available_layers),
        "manifest_sha256": _sha256(path / "icalens.json"),
        "profile_sha256": profile_hashes,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_units(
    selections: dict[str, dict[str, list[int]]],
) -> list[tuple[str, str, int, int]]:
    return [
        (_unit_id(label, int(layer), component), label, int(layer), component)
        for label, layers in selections.items()
        for layer, components in _layers_in_numeric_order(layers)
        for component in components
    ]


def _layers_in_numeric_order(
    layers: dict[str, list[int]],
) -> list[tuple[str, list[int]]]:
    return sorted(layers.items(), key=lambda item: int(item[0]))


def _unit_id(label: str, layer: int, component: int) -> str:
    return f"{label}:layer-{layer}:C{component}"


def _component_path(output: Path, label: str, layer: int, component: int) -> Path:
    return output / "components" / label / f"layer_{layer:02d}" / f"C{component:05d}.json"


def _component_checkpoint_valid(
    path: Path, *, label: str, layer: int, component: int, method: str
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("model_label") == label
        and value.get("layer") == layer
        and value.get("component") == component
        and value.get("method") == method
        and isinstance(value.get("gradient_erf_median"), (int, float))
        and isinstance(value.get("occurrences"), list)
        and len(value["occurrences"]) == value.get("n_occurrences")
    )


def _load_model(lens: Any, *, device: str) -> torch.nn.Module:
    if device.startswith("cuda"):
        model = load_model_to_cuda(
            AutoModelForCausalLM,
            lens.model_id,
            revision=lens.model_revision,
            device=device,
            dtype=torch.bfloat16,
            touch="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = (
            cast(Any, AutoModelForCausalLM)
            .from_pretrained(lens.model_id, revision=lens.model_revision, trust_remote_code=True)
            .to(device)
        )
    model.eval()
    return cast(torch.nn.Module, model)


def _measure_component(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    datasets: dict[str, Any],
    token_cache: dict[tuple[str, int, int], list[int]],
    layer: int,
    component: int,
    direction: str,
    occurrences: list[dict[str, Any]],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    provenance = lens.metadata["layers"][str(layer)]["fitting"]["provenance"]
    prepared = [
        _prepare_occurrence(
            occurrence,
            occurrence_rank=occurrence_rank,
            provenance=provenance,
            tokenizer=tokenizer,
            datasets=datasets,
            token_cache=token_cache,
        )
        for occurrence_rank, occurrence in enumerate(occurrences, start=1)
    ]
    prepared.sort(key=lambda item: len(item["content_ids"]))
    results = []
    block = transformer_blocks(model)[layer]
    for start in range(0, len(prepared), batch_size):
        results.extend(
            _measure_batch(
                lens=lens,
                model=model,
                block=block,
                tokenizer=tokenizer,
                layer=layer,
                component=component,
                direction=direction,
                prepared=prepared[start : start + batch_size],
                framing=provenance["document_framing"],
                device=device,
            )
        )
    results.sort(key=lambda item: item["occurrence_rank"])
    values = [float(item["gradient_erf"]) for item in results]
    return {
        "layer": layer,
        "component": component,
        "tail_direction": direction,
        "n_occurrences": len(results),
        "gradient_erf_median": statistics.median(values),
        "gradient_erf_mean": statistics.mean(values),
        "gradient_erf_min": min(values),
        "gradient_erf_max": max(values),
        "occurrences": results,
    }


def _prepare_occurrence(
    occurrence: dict[str, Any],
    *,
    occurrence_rank: int,
    provenance: dict[str, Any],
    tokenizer: Any,
    datasets: dict[str, Any],
    token_cache: dict[tuple[str, int, int], list[int]],
) -> dict[str, Any]:
    source = provenance["dataset"]
    dataset_key = json.dumps(source, sort_keys=True)
    if dataset_key not in datasets:
        datasets[dataset_key] = load_dataset(
            source["repo_id"],
            split=source["split"],
            revision=source["revision"],
        )
    source_index = int(occurrence["source_index"])
    framing = provenance["document_framing"]
    framing_offset = 0 if framing["strategy"] == "none" else 1
    content_position = int(occurrence["position"]) - framing_offset
    cache_key = (dataset_key, source_index, content_position)
    if cache_key not in token_cache:
        text = str(datasets[dataset_key][source_index][provenance["text_field"]])
        token_cache[cache_key] = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=content_position + 1,
        )["input_ids"][: content_position + 1]
    content_ids = token_cache[cache_key]
    if not content_ids or content_ids[-1] != int(occurrence["token_id"]):
        raise ValueError("stored profile occurrence does not match reconstructed source token")
    return {
        "occurrence": occurrence,
        "occurrence_rank": occurrence_rank,
        "content_ids": content_ids,
        "framing_offset": framing_offset,
    }


def _measure_batch(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    block: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    component: int,
    direction: str,
    prepared: list[dict[str, Any]],
    framing: dict[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    sequences = []
    for item in prepared:
        ids = list(item["content_ids"])
        if item["framing_offset"]:
            ids.insert(0, int(framing["token_id"]))
        sequences.append(ids)
    lengths = [len(ids) for ids in sequences]
    maximum = max(lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    input_ids = torch.full((len(sequences), maximum), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), maximum), dtype=torch.long, device=device)
    for row, ids in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, : len(ids)] = 1
    embedding_layer = cast(torch.nn.Module, cast(Any, model).get_input_embeddings())
    embeddings = (
        cast(torch.Tensor, embedding_layer.forward(input_ids)).detach().requires_grad_(True)
    )
    captured: dict[str, torch.Tensor] = {}

    def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        row = torch.arange(len(sequences), device=hidden.device)
        positions = torch.tensor(lengths, device=hidden.device) - 1
        captured["target"] = hidden[row, positions].float()
        raise _CapturedLayer

    handle = block.register_forward_hook(hook)
    try:
        try:
            model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False)
        except _CapturedLayer:
            pass
    finally:
        handle.remove()
    if "target" not in captured:
        raise RuntimeError("requested residual stream was not captured")
    scores = lens.transform(captured["target"], layer=layer)
    selected_scores = scores[:, component]
    multiplier = 1.0 if direction == "positive" else -1.0
    if bool((multiplier * selected_scores <= 0).any()):
        raise RuntimeError("computed score does not preserve the profiled dominant-tail sign")
    selected_scores.sum().backward()
    if embeddings.grad is None:
        raise RuntimeError("input embedding gradients were not retained")
    results = []
    for row, item in enumerate(prepared):
        content_ids = item["content_ids"]
        offset = int(item["framing_offset"])
        influences = (
            embeddings.grad[row, offset : offset + len(content_ids)].float().square().sum(-1)
        )
        total = influences.sum()
        if float(total) <= 0:
            raise RuntimeError("component has zero lexical input-gradient influence")
        probabilities = influences / total
        statistics_payload, distances = _gradient_statistics(probabilities)
        top_count = min(10, len(content_ids))
        top_indices = torch.topk(probabilities, k=top_count).indices.tolist()
        occurrence = item["occurrence"]
        results.append(
            {
                "occurrence_rank": int(item["occurrence_rank"]),
                "source_index": int(occurrence["source_index"]),
                "position": int(occurrence["position"]),
                "token_id": int(occurrence["token_id"]),
                "text": str(occurrence["text"]),
                "context": str(occurrence["context"]),
                "stored_score": float(occurrence["score"]),
                "computed_score": float(selected_scores[row].detach()),
                "stored_energy": float(occurrence["energy"]),
                "content_tokens": len(content_ids),
                "target_influence_probability": float(probabilities[-1]),
                **statistics_payload,
                "top_influences": [
                    {
                        "distance": int(distances[index]),
                        "token_id": int(content_ids[index]),
                        "token": str(tokenizer.convert_ids_to_tokens(content_ids[index])),
                        "probability": float(probabilities[index]),
                    }
                    for index in top_indices
                ],
            }
        )
    return results


def _gradient_statistics(
    probabilities: torch.Tensor,
) -> tuple[dict[str, float | int], torch.Tensor]:
    distances = torch.arange(
        len(probabilities), 0, -1, dtype=torch.float32, device=probabilities.device
    )
    mean_log = float((probabilities * distances.log()).sum())
    cumulative = torch.cumsum(probabilities.flip(0), dim=0)
    radii = {
        f"influence_radius_{percent}": int(
            torch.searchsorted(cumulative, percent / 100.0).item() + 1
        )
        for percent in (50, 90, 95)
    }
    return (
        {
            "mean_log_distance": mean_log,
            "gradient_erf": math.exp(mean_log),
            "arithmetic_gradient_erf": float((probabilities * distances).sum()),
            **radii,
        },
        distances,
    )


def _write_summaries(output: Path, units: list[tuple[str, str, int, int]]) -> None:
    rows = []
    for _, label, layer, component in units:
        path = _component_path(output, label, layer, component)
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": label,
                "layer": layer,
                "component": component,
                "tail_direction": value["tail_direction"],
                "n_occurrences": value["n_occurrences"],
                "gradient_erf_median": value["gradient_erf_median"],
                "gradient_erf_mean": value["gradient_erf_mean"],
                "gradient_erf_min": value["gradient_erf_min"],
                "gradient_erf_max": value["gradient_erf_max"],
            }
        )
    atomic_write_json(output / "summary.json", {"components": rows})
    _atomic_write_csv(output / "summary.csv", rows)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
