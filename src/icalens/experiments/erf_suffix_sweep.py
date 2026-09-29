"""Run the suffix-sweep effective-receptive-field experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from icalens import ICALens, __version__
from icalens._capture import transformer_blocks
from icalens.cli._status import log

from ._display import ExperimentDisplay
from ._run import ResumableRun, atomic_write_json
from ._source_provenance import source_provenance, warn_if_dirty
from .erf_gradient import (
    _component_units,
    _lens_identity,
    _load_model,
    _parse_lenses,
    _prepare_component_inputs,
    _prepare_occurrence,
    _unit_id,
)

SCHEMA_VERSION = 4
FORMAT = "icalens.erf_suffix_sweep"
PREPARED_FORMAT = "icalens.erf_suffix_sweep.prepared_layer"
METHOD = "suffix-length-sweep-multirank-selected-tail"
DEFAULT_RANK_THRESHOLDS = (1, 3, 5, 10, 15)


class _CapturedLayer(RuntimeError):
    """Stop a model forward after capturing the requested residual stream."""


def _parse_rank_thresholds(value: str) -> tuple[int, ...]:
    try:
        thresholds = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "rank thresholds must be comma-separated integers"
        ) from error
    if not thresholds or thresholds[0] < 1:
        raise argparse.ArgumentTypeError("rank thresholds must be positive")
    return thresholds


def _validate_full_context_ranks(prepared_inputs: dict[str, Any]) -> None:
    missing = []
    for label, layers in prepared_inputs.items():
        for layer, components in layers.items():
            for component, value in components.items():
                for occurrence_index, occurrence in enumerate(value["occurrences"], start=1):
                    rank = occurrence.get("absolute_score_rank")
                    if not isinstance(rank, int) or rank < 1:
                        missing.append(
                            f"{label} layer {layer} C{component} occurrence {occurrence_index}"
                        )
    if missing:
        examples = ", ".join(missing[:3])
        raise ValueError(
            "suffix-sweep ERF requires absolute_score_rank on every stored occurrence; "
            f"missing for {examples}. Run `icalens profile refresh-examples-rank` first."
        )


def _format_mean(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="icalens experiment erf-suffix-sweep", description=__doc__
    )
    parser.add_argument(
        "--lens",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each local profiled ICA Lens.",
    )
    parser.add_argument("--components-per-layer", type=int, default=100)
    parser.add_argument("--occurrences-per-component", type=int, default=20)
    parser.add_argument(
        "--exact-suffix-length",
        type=int,
        default=10,
        help=(
            "Test every suffix length through this value, then test 2x and 4x; "
            "subsequent lengths grow by 4x."
        ),
    )
    parser.add_argument(
        "--rank-thresholds",
        default=",".join(str(value) for value in DEFAULT_RANK_THRESHOLDS),
        help="Comma-separated absolute-score rank thresholds recorded in one sweep.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--batch-token-budget", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.rank_thresholds = _parse_rank_thresholds(args.rank_thresholds)
    if (
        min(
            args.components_per_layer,
            args.occurrences_per_component,
            args.exact_suffix_length,
            args.max_batch_size,
            args.batch_token_budget,
        )
        < 1
    ):
        raise ValueError("component, occurrence, suffix, and batch values must be positive")

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
        _validate_full_context_ranks(prepared_inputs)
        resolved = _resolved_configuration(
            lens_specs=lens_specs,
            lenses=lenses,
            selections=selections,
            args=args,
        )
    else:
        resolved, selections, prepared_inputs = cached
        _validate_full_context_ranks(prepared_inputs)
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="measuring")
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
        )
    }
    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · suffix-sweep effective receptive field",
        completed=len(completed_ids),
        total=len(units),
        completed_unit_ids=completed_ids,
        unit_label="components",
        recent_label="Recent ERF output",
        detail_filename="erf-suffix-sweep-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        log(
            f"Validated suffix-sweep ERF configuration: {len(lenses)} models, "
            f"{len(units)} components, up to {args.occurrences_per_component} occurrences each."
        )
        for label, lens in lenses.items():
            pending = [
                (int(layer), component)
                for layer, components in sorted(
                    selections[label].items(), key=lambda item: int(item[0])
                )
                for component in components
                if _unit_id(label, int(layer), component) not in completed_ids
            ]
            if not pending:
                log(f"{label}: all selected components already complete; skipping model load.")
                continue
            display.phase("Loading model", model=label)
            log(f"Loading {label}: {lens.model_id}@{lens.model_revision}.")
            model = _load_model(lens, device=args.device)
            tokenizer = AutoTokenizer.from_pretrained(
                lens.model_id,
                revision=lens.model_revision,
                use_fast=True,
                trust_remote_code=True,
            )
            datasets: dict[str, Any] = {}
            token_cache: dict[tuple[str, int, int], list[int]] = {}
            try:
                pending_by_layer: dict[int, list[int]] = {}
                for layer, component in pending:
                    pending_by_layer.setdefault(layer, []).append(component)
                for layer, components in pending_by_layer.items():
                    display.phase("Sweeping suffixes", model=label, layer=layer)

                    def checkpoint(
                        component: int,
                        result: dict[str, Any],
                        *,
                        _label: str = label,
                        _layer: int = layer,
                    ) -> None:
                        result.update(
                            model_label=_label,
                            method=METHOD,
                            schema_version=SCHEMA_VERSION,
                        )
                        atomic_write_json(
                            _component_path(output, _label, _layer, component), result
                        )
                        display.complete_unit(_unit_id(_label, _layer, component), refresh=True)
                        display_threshold = max(args.rank_thresholds)
                        display_result = result["threshold_results"][str(display_threshold)]
                        log(
                            f"Checkpointed {_label} layer {_layer} C{component}: "
                            f"top-{display_threshold} mean ERF "
                            f"{_format_mean(display_result['suffix_erf_mean'])}, "
                            f"recovered {display_result['n_recovered']}/"
                            f"{result['n_occurrences']}."
                        )

                    _measure_layer(
                        lens=lens,
                        model=model,
                        tokenizer=tokenizer,
                        datasets=datasets,
                        token_cache=token_cache,
                        layer=layer,
                        prepared_components={
                            component: prepared_inputs[label][str(layer)][str(component)]
                            for component in components
                        },
                        exact_suffix_length=args.exact_suffix_length,
                        rank_thresholds=args.rank_thresholds,
                        max_batch_size=args.max_batch_size,
                        batch_token_budget=args.batch_token_budget,
                        device=args.device,
                        checkpoint=checkpoint,
                    )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        display.phase("Summarizing")
        _write_summaries(output, units)
        run.set_status("complete", complete=True)
        log(f"Suffix-sweep ERF experiment complete: {output}")


def _measure_layer(
    *,
    lens: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    datasets: dict[str, Any],
    token_cache: dict[tuple[str, int, int], list[int]],
    layer: int,
    prepared_components: dict[int, dict[str, Any]],
    exact_suffix_length: int,
    rank_thresholds: tuple[int, ...],
    max_batch_size: int,
    batch_token_budget: int,
    device: str,
    checkpoint: Any,
) -> None:
    """Sweep one layer, sharing each suffix forward across components and thresholds."""
    provenance = lens.metadata["layers"][str(layer)]["fitting"]["provenance"]
    states: dict[int, dict[str, Any]] = {}
    for component, value in prepared_components.items():
        prepared = [
            _prepare_occurrence(
                occurrence,
                occurrence_rank=rank,
                provenance=provenance,
                tokenizer=tokenizer,
                datasets=datasets,
                token_cache=token_cache,
            )
            for rank, occurrence in enumerate(value["occurrences"], start=1)
        ]
        occurrence_states = {}
        for item in prepared:
            occurrence = item["occurrence"]
            full_rank = int(occurrence["absolute_score_rank"])
            occurrence_states[int(item["occurrence_rank"])] = {
                "item": item,
                "full_context_rank": full_rank,
                "unresolved": set(rank_thresholds),
                "last_failure": {threshold: 0 for threshold in rank_thresholds},
                "recoveries": {},
                "observations": [],
            }
        states[component] = {
            "direction": str(value["tail_direction"]),
            "occurrences": occurrence_states,
        }
    block = transformer_blocks(model)[layer]
    maximum_context = max(
        len(value["item"]["content_ids"])
        for state in states.values()
        for value in state["occurrences"].values()
    )
    completed: set[int] = set()
    for suffix_length in _suffix_schedule(
        exact_suffix_length=exact_suffix_length, maximum_context=maximum_context
    ):
        active = [
            (component, state["direction"], occurrence_state["item"])
            for component, state in states.items()
            if component not in completed
            for occurrence_state in state["occurrences"].values()
            if occurrence_state["unresolved"]
            and len(occurrence_state["item"]["content_ids"]) >= suffix_length
        ]
        batch_size = _adaptive_batch_size(
            suffix_length,
            max_batch_size=max_batch_size,
            token_budget=batch_token_budget,
        )
        for start in range(0, len(active), batch_size):
            batch = active[start : start + batch_size]
            measured = _measure_mixed_batch(
                lens=lens,
                model=model,
                block=block,
                tokenizer=tokenizer,
                layer=layer,
                batch=batch,
                suffix_length=suffix_length,
                framing=provenance["document_framing"],
                device=device,
            )
            for component, value in measured:
                occurrence_state = states[component]["occurrences"][int(value["occurrence_rank"])]
                if not occurrence_state["unresolved"]:
                    continue
                occurrence_state["observations"].append(value)
                if not value["sign_matches_selected_tail"]:
                    for threshold in occurrence_state["unresolved"]:
                        occurrence_state["last_failure"][threshold] = suffix_length
                    continue
                crossed = {
                    threshold
                    for threshold in occurrence_state["unresolved"]
                    if int(value["absolute_score_rank"]) <= threshold
                }
                for threshold in crossed:
                    occurrence_state["recoveries"][threshold] = _recovery_estimate(
                        lower=occurrence_state["last_failure"][threshold],
                        upper=suffix_length,
                        exact_suffix_length=exact_suffix_length,
                        source="measured_suffix",
                    )
                occurrence_state["unresolved"].difference_update(crossed)
                for threshold in occurrence_state["unresolved"]:
                    occurrence_state["last_failure"][threshold] = suffix_length

        for component, state in states.items():
            if component in completed or any(
                value["unresolved"] for value in state["occurrences"].values()
            ):
                continue
            checkpoint(
                component,
                _finish_component_result(
                    layer=layer,
                    component=component,
                    direction=state["direction"],
                    occurrence_states=state["occurrences"],
                    exact_suffix_length=exact_suffix_length,
                    rank_thresholds=rank_thresholds,
                ),
            )
            completed.add(component)
        if len(completed) == len(states):
            return

    # The stored full-context rank is the authoritative final endpoint. Thresholds
    # that still fail there receive the occurrence's actual full context length.
    for component, state in states.items():
        if component in completed:
            continue
        for occurrence_state in state["occurrences"].values():
            full_length = len(occurrence_state["item"]["content_ids"])
            for threshold in tuple(occurrence_state["unresolved"]):
                if occurrence_state["full_context_rank"] <= threshold:
                    result = _recovery_estimate(
                        lower=min(occurrence_state["last_failure"][threshold], full_length - 1),
                        upper=full_length,
                        exact_suffix_length=exact_suffix_length,
                        source="stored_full_context_rank",
                    )
                else:
                    result = _unrecovered_assignment(full_length=full_length)
                occurrence_state["recoveries"][threshold] = result
                occurrence_state["unresolved"].remove(threshold)
        checkpoint(
            component,
            _finish_component_result(
                layer=layer,
                component=component,
                direction=state["direction"],
                occurrence_states=state["occurrences"],
                exact_suffix_length=exact_suffix_length,
                rank_thresholds=rank_thresholds,
            ),
        )


def _suffix_schedule(*, exact_suffix_length: int, maximum_context: int) -> list[int]:
    """Test 1..N exactly, then N*2, N*4, N*16, ... below the longest context."""
    if min(exact_suffix_length, maximum_context) < 1:
        raise ValueError("exact suffix length and maximum context must be positive")
    schedule = list(range(1, min(exact_suffix_length, maximum_context) + 1))
    length = exact_suffix_length * 2
    while length < maximum_context:
        schedule.append(length)
        length *= 2 if length == exact_suffix_length * 2 else 4
    return schedule


def _recovery_estimate(
    *, lower: int, upper: int, exact_suffix_length: int, source: str
) -> dict[str, Any]:
    """Represent an exact or multiplicatively bracketed first-recovery length."""
    if not 0 <= lower < upper or exact_suffix_length < 1:
        raise ValueError("recovery bounds must satisfy 0 <= lower < upper")
    exact = upper <= exact_suffix_length or upper == lower + 1
    estimate = float(upper) if exact else math.sqrt(lower * upper)
    return {
        "recovered": True,
        "erf_estimate": estimate,
        "lower_bound_exclusive": None if exact else lower,
        "upper_bound_inclusive": upper,
        "exact": exact,
        "source": source,
    }


def _unrecovered_assignment(*, full_length: int) -> dict[str, Any]:
    if full_length < 1:
        raise ValueError("full context length must be positive")
    return {
        "recovered": False,
        "erf_estimate": float(full_length),
        "lower_bound_exclusive": None,
        "upper_bound_inclusive": full_length,
        "exact": False,
        "source": "unrecovered_assigned_full_context_length",
    }


def _adaptive_batch_size(suffix_length: int, *, max_batch_size: int, token_budget: int) -> int:
    """Keep approximately the same number of lexical tokens in each forward batch."""
    if min(suffix_length, max_batch_size, token_budget) < 1:
        raise ValueError("suffix length, maximum batch size, and token budget must be positive")
    return max(1, min(max_batch_size, token_budget // suffix_length))


def _measure_mixed_batch(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    block: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    batch: list[tuple[int, str, dict[str, Any]]],
    suffix_length: int,
    framing: dict[str, Any],
    device: str,
) -> list[tuple[int, dict[str, Any]]]:
    prepared = [item for _component, _direction, item in batch]
    sequences = []
    for item in prepared:
        ids = list(item["content_ids"][-suffix_length:])
        if item["framing_offset"]:
            ids.insert(0, int(framing["token_id"]))
        sequences.append(ids)
    scores = _suffix_scores(
        lens=lens,
        model=model,
        block=block,
        tokenizer=tokenizer,
        layer=layer,
        sequences=sequences,
        device=device,
    )
    components = torch.tensor([component for component, _direction, _item in batch], device=device)
    rows = torch.arange(len(batch), device=device)
    selected = scores[rows, components]
    ranks = (scores.abs() > selected.abs().unsqueeze(1)).sum(1) + 1
    positive = torch.tensor(
        [direction == "positive" for _component, direction, _item in batch], device=device
    )
    sign_ok = torch.where(positive, selected > 0, selected < 0)
    return [
        (
            component,
            {
                "occurrence_rank": int(item["occurrence_rank"]),
                "suffix_length": suffix_length,
                "score": float(selected[row]),
                "absolute_score_rank": int(ranks[row]),
                "sign_matches_selected_tail": bool(sign_ok[row]),
            },
        )
        for row, (component, _direction, item) in enumerate(batch)
    ]


def _suffix_scores(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    block: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    sequences: list[list[int]],
    device: str,
) -> torch.Tensor:
    lengths = [len(ids) for ids in sequences]
    maximum = max(lengths)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    input_ids = torch.full((len(sequences), maximum), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, : len(ids)] = 1
    captured: dict[str, torch.Tensor] = {}

    def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        rows = torch.arange(len(sequences), device=hidden.device)
        positions = torch.tensor(lengths, device=hidden.device) - 1
        captured["target"] = hidden[rows, positions].float()
        raise _CapturedLayer

    handle = block.register_forward_hook(hook)
    try:
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _CapturedLayer:
            pass
    finally:
        handle.remove()
    if "target" not in captured:
        raise RuntimeError("requested residual stream was not captured")
    transformed = lens.transform(captured["target"], layer=layer)
    if not isinstance(transformed, torch.Tensor):
        raise TypeError("lens transform must return a tensor")
    return transformed.detach()


def _finish_component_result(
    *,
    layer: int,
    component: int,
    direction: str,
    occurrence_states: dict[int, dict[str, Any]],
    exact_suffix_length: int,
    rank_thresholds: tuple[int, ...],
) -> dict[str, Any]:
    results = []
    for rank, state in sorted(occurrence_states.items()):
        item = state["item"]
        occurrence = item["occurrence"]
        threshold_results = {}
        for threshold in rank_thresholds:
            recovery = state["recoveries"].get(threshold)
            if recovery is None:
                raise RuntimeError("suffix-sweep threshold has no recovery result")
            threshold_results[str(threshold)] = recovery
        results.append(
            {
                "occurrence_rank": rank,
                "source_index": int(occurrence["source_index"]),
                "position": int(occurrence["position"]),
                "token_id": int(occurrence["token_id"]),
                "text": str(occurrence["text"]),
                "context": str(occurrence["context"]),
                "stored_score": float(occurrence["score"]),
                "stored_energy": float(occurrence["energy"]),
                "full_context_absolute_score_rank": int(state["full_context_rank"]),
                "content_tokens": len(item["content_ids"]),
                "thresholds": threshold_results,
                "sweep": state["observations"],
            }
        )
    component_thresholds = {}
    for threshold in rank_thresholds:
        recoveries = [item["thresholds"][str(threshold)] for item in results]
        values = [float(value["erf_estimate"]) for value in recoveries]
        recovered_count = sum(bool(value["recovered"]) for value in recoveries)
        component_thresholds[str(threshold)] = {
            "top_k": threshold,
            "n_occurrences": len(results),
            "n_recovered": recovered_count,
            "recovered_fraction": recovered_count / len(results),
            "n_unrecovered": len(results) - recovered_count,
            "n_exact_recoveries": sum(
                bool(value["exact"]) for value in recoveries if value["recovered"]
            ),
            "suffix_erf_median": statistics.median(values),
            "suffix_erf_mean": statistics.mean(values),
            "suffix_erf_min": min(values),
            "suffix_erf_max": max(values),
        }
    return {
        "layer": layer,
        "component": component,
        "tail_direction": direction,
        "exact_suffix_length": exact_suffix_length,
        "rank_thresholds": list(rank_thresholds),
        "n_occurrences": len(results),
        "threshold_results": component_thresholds,
        "occurrences": results,
    }


def _recovery_from_scores(
    scores: torch.Tensor, *, component: int, direction: str, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return absolute-score rank, selected-tail match, and recovery."""
    if scores.ndim != 2 or not 0 <= component < scores.shape[1] or top_k < 1:
        raise ValueError("scores, component, and top_k are incompatible")
    if direction not in ("positive", "negative"):
        raise ValueError("direction must be positive or negative")
    selected = scores[:, component]
    ranks = (scores.abs() > selected.abs().unsqueeze(1)).sum(1) + 1
    sign_ok = selected > 0 if direction == "positive" else selected < 0
    return ranks, sign_ok, (ranks <= top_k) & sign_ok


def _resolved_configuration(
    *,
    lens_specs: dict[str, Path],
    lenses: dict[str, ICALens],
    selections: dict[str, dict[str, list[int]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "icalens_version": __version__,
        "method": METHOD,
        "recovery": "score on selected tail and absolute-score rank <= each recorded threshold",
        "suffix_schedule": (
            "1..exact_suffix_length, then 2x and 4x exact_suffix_length, "
            "then multiply the tested length by 4"
        ),
        "occurrence_erf": (
            "exact first recovery through the exact sweep; geometric midpoint of later "
            "recovery bracket"
        ),
        "unrecovered_assignment": "the occurrence's full available context length",
        "component_erf": "mean occurrence ERF over all occurrences",
        "components_per_layer": args.components_per_layer,
        "occurrences_per_component": args.occurrences_per_component,
        "exact_suffix_length": args.exact_suffix_length,
        "rank_thresholds": list(args.rank_thresholds),
        "seed": args.seed,
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
                    "format": PREPARED_FORMAT,
                    "schema_version": SCHEMA_VERSION,
                    "model_label": label,
                    "layer": int(layer_text),
                    "components": components,
                },
            )


def _load_prepared_run(
    output: Path, *, lens_specs: dict[str, Path], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    run_path = output / "run.json"
    if not run_path.is_file():
        return None
    try:
        resolved = json.loads(run_path.read_text(encoding="utf-8"))["resolved"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{run_path} has no valid resolved configuration") from error
    keys = (
        "format",
        "schema_version",
        "method",
        "components_per_layer",
        "occurrences_per_component",
        "exact_suffix_length",
        "rank_thresholds",
        "seed",
        "lens_order",
    )
    requested = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "components_per_layer": args.components_per_layer,
        "occurrences_per_component": args.occurrences_per_component,
        "exact_suffix_length": args.exact_suffix_length,
        "rank_thresholds": list(args.rank_thresholds),
        "seed": args.seed,
        "lens_order": list(lens_specs),
    }
    existing = {key: resolved.get(key) for key in keys}
    if existing != requested:
        differences = [
            f"{key}: {existing[key]!r} -> {requested[key]!r}"
            for key in keys
            if existing[key] != requested[key]
        ]
        raise ValueError(
            f"{run_path} belongs to a different configuration "
            f"({'; '.join(differences)}); choose another output"
        )
    for label, path in lens_specs.items():
        identity = resolved.get("lenses", {}).get(label, {})
        if identity.get("path") != str(path):
            raise ValueError(f"{run_path} uses a different Lens path for {label}")
        current = _lens_identity(path, ICALens.from_pretrained(path), resolved["selections"][label])
        if current["manifest_sha256"] != identity.get("manifest_sha256") or current[
            "profile_sha256"
        ] != identity.get("profile_sha256"):
            raise ValueError(
                f"prepared suffix-sweep inputs are stale because {label} changed; "
                "choose another output"
            )
    selections = resolved.get("selections")
    if not isinstance(selections, dict):
        raise ValueError(f"{run_path} has no valid component selections")
    prepared: dict[str, Any] = {}
    for label, layers in selections.items():
        prepared[label] = {}
        for layer_text, selected in layers.items():
            path = _prepared_layer_path(output, label, int(layer_text))
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            components = payload.get("components")
            if (
                payload.get("format") != PREPARED_FORMAT
                or payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("model_label") != label
                or payload.get("layer") != int(layer_text)
                or not isinstance(components, dict)
                or sorted(int(value) for value in components) != sorted(selected)
            ):
                return None
            prepared[label][layer_text] = components
    return resolved, selections, prepared


def _component_path(output: Path, label: str, layer: int, component: int) -> Path:
    return output / "components" / label / f"layer_{layer:02d}" / f"C{component:05d}.json"


def _component_checkpoint_valid(path: Path, *, label: str, layer: int, component: int) -> bool:
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
        and value.get("method") == METHOD
        and value.get("schema_version") == SCHEMA_VERSION
        and isinstance(value.get("threshold_results"), dict)
        and isinstance(value.get("occurrences"), list)
        and len(value["occurrences"]) == value.get("n_occurrences")
    )


def _write_summaries(output: Path, units: list[tuple[str, str, int, int]]) -> None:
    rows = []
    for _, label, layer, component in units:
        value = json.loads(_component_path(output, label, layer, component).read_text())
        for threshold_text, result in value["threshold_results"].items():
            rows.append(
                {
                    "model": label,
                    "layer": layer,
                    "component": component,
                    "top_k": int(threshold_text),
                    "suffix_erf_median": result["suffix_erf_median"],
                    "suffix_erf_mean": result["suffix_erf_mean"],
                    "suffix_erf_min": result["suffix_erf_min"],
                    "suffix_erf_max": result["suffix_erf_max"],
                    "n_occurrences": result["n_occurrences"],
                    "n_recovered": result["n_recovered"],
                    "recovered_fraction": result["recovered_fraction"],
                    "n_unrecovered": result["n_unrecovered"],
                    "n_exact_recoveries": result["n_exact_recoveries"],
                }
            )
    atomic_write_json(output / "summary.json", {"format": FORMAT, "rows": rows})
    path = output / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
