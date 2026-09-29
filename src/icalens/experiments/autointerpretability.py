"""Prepare and run the LLM-based feature autointerpretability experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from dotenv import load_dotenv
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens, __version__
from icalens._capture import transformer_blocks
from icalens.cli._status import log

from ._display import ExperimentDisplay
from ._run import ResumableRun, atomic_write_json
from ._sae import SAEFeatureEncoder
from ._source_provenance import source_provenance, warn_if_dirty
from .autointerpretability_protocol import (
    EXAMPLES_PER_SPLIT,
    FRAGMENT_LENGTH,
    explanation_messages,
    select_record_indices,
    simulator_messages,
)
from .saebench_sparse_probing import _prepare_layer_baselines, _resolve_baselines

SCHEMA_VERSION = 1
DEFAULT_OPENAI_EVALUATOR_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_TINKER_EXPLAINER_MODEL = "thinkingmachines/Inkling"
DEFAULT_TINKER_SIMULATOR_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_DATASET = "Skylion007/openwebtext"

_HELP = """usage: icalens experiment autointerpretability COMMAND [OPTIONS]

commands:
  prepare   Capture a shared fragment pool and prepare ICA/SAE feature records
  evaluate  Generate explanations, simulate held-out activations, and score them
  summarize Aggregate completed per-feature results
"""


def main(argv: Sequence[str] | None = None) -> None:
    values = list(argv or ())
    if not values or values[0] in {"-h", "--help"}:
        print(_HELP)
        return
    command = values.pop(0)
    if command == "prepare":
        prepare_main(values)
    elif command == "evaluate":
        evaluate_main(values)
    elif command == "summarize":
        summarize_main(values)
    else:
        raise SystemExit(f"unknown autointerpretability command {command!r}")


def _prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icalens experiment autointerpretability prepare")
    parser.add_argument("--lens", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated zero-based layers.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-fragments", type=int, default=50_000)
    parser.add_argument("--n-features", type=int, default=150)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def prepare_main(argv: Sequence[str] | None = None) -> None:
    args = _prepare_parser().parse_args(argv)
    if min(args.n_fragments, args.n_features, args.candidate_multiplier, args.batch_size) < 1:
        raise ValueError("fragment, feature, candidate, and batch counts must be positive")
    if args.n_fragments < 20:
        raise ValueError("--n-fragments must be at least 20")
    source = source_provenance()
    lens = ICALens.from_pretrained(args.lens)
    layers = _parse_layers(args.layers, lens.available_layers)
    if lens.activation_site != "resid_post":
        raise ValueError("autointerpretability currently supports resid_post only")
    baseline = _resolve_baselines(lens.model_id, "sae")["sae"]
    revision = args.dataset_revision or HfApi().dataset_info(args.dataset).sha
    if not revision:
        raise RuntimeError(f"could not resolve dataset revision for {args.dataset}")
    candidate_count = args.n_features * args.candidate_multiplier
    estimated_candidate_bytes = sum(
        args.n_fragments
        * FRAGMENT_LENGTH
        * (
            min(candidate_count, lens._layers[layer].n_components)
            + min(candidate_count, int(baseline["width"]))
        )
        * np.dtype(np.float16).itemsize
        for layer in layers
    )
    resolved = {
        "format": "icalens.autointerpretability",
        "schema_version": SCHEMA_VERSION,
        "icalens_version": __version__,
        "lens": str(args.lens),
        "model": {"repo_id": lens.model_id, "revision": lens.model_revision},
        "activation_site": lens.activation_site,
        "layer_indexing": lens.layer_indexing,
        "layers": layers,
        "dataset": {"repo_id": args.dataset, "revision": revision, "split": args.split},
        "n_fragments": args.n_fragments,
        "fragment_length": FRAGMENT_LENGTH,
        "n_features": args.n_features,
        "candidate_multiplier": args.candidate_multiplier,
        "estimated_candidate_activation_bytes": estimated_candidate_bytes,
        "seed": args.seed,
        "sae": baseline,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    output = args.output.expanduser().resolve()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="preparing",
        normalize_previous=_normalize_legacy_openwebtext,
    )
    pending = [
        layer
        for layer in layers
        if not _prepared_layer_is_valid(
            output, layer=layer, n_fragments=args.n_fragments, n_features=args.n_features
        )
    ]
    completed_fragments = _fragment_checkpoint_count(output, expected=args.n_fragments)
    completed_layers = len(layers) - len(pending)
    initial = completed_fragments + completed_layers
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · autointerpretability preparation",
        completed=initial,
        total=args.n_fragments + len(layers),
        recent_label="Recent preparation output",
        detail_filename="preparation-detail.log",
        source_dirty=bool(source.get("dirty")),
    )
    with display:
        warn_if_dirty(source)
        log(
            f"Validated preparation configuration: {args.n_fragments} fragments, "
            f"layers {','.join(map(str, layers))}, ICA and SAE."
        )
        display.phase("Sampling fragments", dataset=DEFAULT_DATASET)
        log("Starting or resuming deterministic fragment sampling.")
        tokenizer = (
            None
            if completed_fragments == args.n_fragments
            else AutoTokenizer.from_pretrained(
                lens.model_id, revision=lens.model_revision, use_fast=True
            )
        )
        fragments = _load_or_sample_fragments(
            output=output,
            tokenizer=tokenizer,
            dataset=args.dataset,
            revision=str(revision),
            split=args.split,
            count=args.n_fragments,
            seed=args.seed,
            on_checkpoint=display.advance,
        )
        log(f"Fragment pool complete: {len(fragments)} durable fragments.")
        if pending:
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            display.phase("Loading model and feature dictionaries", layers=pending)
            log(f"Loading model and feature dictionaries for pending layers {pending}.")
            model = _load_model(lens, device=args.device)
            try:
                _prepare_layers(
                    lens=lens,
                    model=model,
                    fragments=fragments,
                    layers=pending,
                    baseline=baseline,
                    output=output,
                    n_features=args.n_features,
                    candidate_multiplier=args.candidate_multiplier,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    device=args.device,
                    display=display,
                )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            log(f"All requested layers are already prepared: {output}")
        run.set_status("prepared", complete=True)
        log(f"Autointerpretability records prepared: {output}")


def _load_model(lens: ICALens, *, device: str) -> torch.nn.Module:
    model: torch.nn.Module
    if device.startswith("cuda"):
        model = cast(
            torch.nn.Module,
            load_model_to_cuda(
                AutoModelForCausalLM,
                lens.model_id,
                revision=lens.model_revision,
                device=device,
                dtype=torch.bfloat16,
                touch="auto",
                low_cpu_mem_usage=True,
            ),
        )
    else:
        model = cast(
            torch.nn.Module,
            cast(Any, AutoModelForCausalLM)
            .from_pretrained(lens.model_id, revision=lens.model_revision)
            .to(device),
        )
    model.eval()
    return model


def _load_or_sample_fragments(
    *,
    output: Path,
    tokenizer: Any,
    dataset: str,
    revision: str,
    split: str,
    count: int,
    seed: int,
    on_checkpoint: Any | None = None,
) -> list[dict[str, Any]]:
    path = output / "fragments.jsonl"
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(existing) != count:
            raise ValueError(f"{path} contains {len(existing)} fragments, expected {count}")
        return cast(list[dict[str, Any]], existing)
    partial = output / "fragments.partial.jsonl"
    rows = _read_fragment_checkpoint(partial)
    if len(rows) > count:
        raise ValueError(f"{partial} contains {len(rows)} fragments, expected at most {count}")
    if rows and [int(row["fragment"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{partial} has non-contiguous fragment indices")
    last_document = int(rows[-1]["document_index"]) if rows else -1
    if rows:
        log(f"Resuming fragment sampling from {len(rows)}/{count} durable fragments.")
    if len(rows) == count:
        partial.replace(path)
        return rows
    stream = load_dataset(dataset, split=split, revision=revision, streaming=True)
    rejected = 0
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", encoding="utf-8", buffering=1) as checkpoint:
        for document_index, item in enumerate(stream):
            if document_index <= last_document:
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text:
                rejected += 1
                continue
            # We intentionally tokenize the complete document before choosing a random
            # 64-token window. `verbose=False` suppresses the tokenizer's model-length
            # warning; only the selected window is ever passed to the language model.
            ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
            if len(ids) < FRAGMENT_LENGTH:
                rejected += 1
                continue
            document_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{document_index}".encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(document_seed)
            start = int(rng.integers(0, len(ids) - FRAGMENT_LENGTH + 1))
            selected = [int(value) for value in ids[start : start + FRAGMENT_LENGTH]]
            tokens = tokenizer.convert_ids_to_tokens(selected)
            if len(tokens) != FRAGMENT_LENGTH or any("�" in token for token in tokens):
                rejected += 1
                continue
            row = {
                "fragment": len(rows),
                "document_index": document_index,
                "token_start": start,
                "token_ids": selected,
                "tokens": tokens,
            }
            checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if on_checkpoint is not None:
                on_checkpoint()
            if len(rows) == count:
                break
    if len(rows) != count:
        raise RuntimeError(f"dataset ended after {len(rows)} usable fragments")
    partial.replace(path)
    _write_json(
        output / "fragment_sampling.json",
        {"accepted": count, "rejected_this_invocation": rejected},
    )
    return rows


def _read_fragment_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    valid: list[dict[str, Any]] = []
    repaired = False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            repaired = True
            break
        if not isinstance(value, dict):
            raise ValueError(f"invalid fragment checkpoint row in {path}")
        valid.append(cast(dict[str, Any], value))
    if repaired:
        temporary = path.with_name(path.name + ".repair")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid),
            encoding="utf-8",
        )
        temporary.replace(path)
        log(f"Discarded an incomplete final line from {path}.")
    return valid


def _fragment_checkpoint_count(output: Path, *, expected: int) -> int:
    final = output / "fragments.jsonl"
    if final.is_file():
        count = sum(1 for line in final.read_text(encoding="utf-8").splitlines() if line)
        if count != expected:
            raise ValueError(f"{final} contains {count} fragments, expected {expected}")
        return count
    return len(_read_fragment_checkpoint(output / "fragments.partial.jsonl"))


def _prepare_layers(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    fragments: list[dict[str, Any]],
    layers: list[int],
    baseline: dict[str, Any],
    output: Path,
    n_features: int,
    candidate_multiplier: int,
    seed: int,
    batch_size: int,
    device: str,
    display: ExperimentDisplay,
) -> None:
    candidate_count = n_features * candidate_multiplier
    encoders: dict[int, SAEFeatureEncoder] = {}
    candidates: dict[tuple[int, str], np.ndarray] = {}
    stores: dict[tuple[int, str], np.memmap] = {}
    temporary_paths: dict[tuple[int, str], Path] = {}
    orientations: dict[int, np.ndarray] = {}
    for layer in layers:
        prepared = _prepare_layer_baselines({"sae": baseline}, layer=layer)
        snapshot = {
            "hidden_size": int(lens.hidden_size or 0),
            "layer": layer,
            "saebench_model_name": lens.model_id,
            "baselines": prepared,
        }
        encoder = SAEFeatureEncoder(snapshot, device=device, dtype=torch.float32)
        encoders[layer] = encoder
        artifact = lens._get_layer(layer)
        profile = lens._get_profile(artifact)
        signs = np.empty(artifact.n_components, dtype=np.float32)
        for component in profile["components"]:
            index = int(component["component"])
            direction = component.get("tail_direction")
            if direction not in {"positive", "negative"}:
                raise ValueError(f"layer {layer} C{index} has no profiled tail_direction")
            signs[index] = 1.0 if direction == "positive" else -1.0
        orientations[layer] = signs
        widths = {"ica": artifact.n_components, "sae": int(encoder.cfg.d_sae)}
        for method, width in widths.items():
            count = min(candidate_count, width)
            permutation = np.random.default_rng(seed + 1009 * layer + (0 if method == "ica" else 1))
            ids = permutation.permutation(width)[:count].astype(np.int64)
            candidates[(layer, method)] = ids
            directory = output / f"layer_{layer:02d}" / method
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "candidate_activations.npy.tmp"
            temporary_paths[(layer, method)] = temporary
            stores[(layer, method)] = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float16,
                shape=(len(fragments), FRAGMENT_LENGTH, count),
            )
    blocks = transformer_blocks(model)
    handles: list[Any] = []
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int) -> Any:
        def hook(_: Any, __: Any, value: Any) -> None:
            hidden = value[0] if isinstance(value, tuple) else value
            captured[layer] = hidden.detach()

        return hook

    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        for start in range(0, len(fragments), batch_size):
            batch = fragments[start : start + batch_size]
            input_ids = torch.tensor(
                [row["token_ids"] for row in batch], dtype=torch.long, device=device
            )
            captured.clear()
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                )
            for layer in layers:
                hidden = captured[layer].to(dtype=torch.float32)
                ica_ids = torch.as_tensor(candidates[(layer, "ica")], device=device)
                orientation_tensor = torch.as_tensor(orientations[layer], device=device)
                scores = lens.transform(hidden, layer=layer)
                oriented = (scores * orientation_tensor).clamp_min(0).index_select(-1, ica_ids)
                sae_ids = torch.as_tensor(candidates[(layer, "sae")], device=device)
                sae = encoders[layer].encode(hidden).index_select(-1, sae_ids)
                stores[(layer, "ica")][start : start + len(batch)] = oriented.cpu().numpy()
                stores[(layer, "sae")][start : start + len(batch)] = sae.cpu().numpy()
            if start == 0 or (start // batch_size) % 100 == 0:
                display.phase(
                    "Encoding candidate features",
                    layers=",".join(map(str, layers)),
                    fragments=f"{start + len(batch)}/{len(fragments)}",
                )
    finally:
        for handle in handles:
            handle.remove()
    for store in stores.values():
        store.flush()
    for layer in layers:
        for method in ("ica", "sae"):
            directory = output / f"layer_{layer:02d}" / method
            final = directory / "candidate_activations.npy"
            temporary_paths[(layer, method)].replace(final)
            values = np.load(final, mmap_mode="r")
            accepted: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for position, feature_id in enumerate(candidates[(layer, method)].tolist()):
                try:
                    train, top, random = select_record_indices(
                        np.asarray(values[:, :, position], dtype=np.float32),
                        seed=seed + feature_id,
                    )
                except ValueError as error:
                    rejected.append({"feature": feature_id, "reason": str(error)})
                    continue
                accepted.append(
                    {
                        "feature": feature_id,
                        "candidate_position": position,
                        "train_top": train,
                        "valid_top": top,
                        "valid_random": random,
                    }
                )
                if len(accepted) == n_features:
                    break
            if len(accepted) < n_features:
                raise RuntimeError(
                    f"layer {layer} {method}: only {len(accepted)}/{n_features} eligible "
                    "features; increase --candidate-multiplier or --n-fragments"
                )
            _write_json(
                directory / "selection.json",
                {
                    "candidate_ids": candidates[(layer, method)].tolist(),
                    "accepted": accepted,
                    "rejected": rejected,
                },
            )
        _write_json(
            output / f"layer_{layer:02d}" / "prepared.json",
            {
                "format": "icalens.autointerpretability-prepared-layer",
                "schema_version": SCHEMA_VERSION,
                "layer": layer,
                "methods": ["ica", "sae"],
                "n_fragments": len(fragments),
                "n_features": n_features,
            },
        )
        display.advance(refresh=True)
        log(f"Checkpointed prepared layer {layer}.")


def _prepared_layer_is_valid(
    output: Path, *, layer: int, n_fragments: int, n_features: int
) -> bool:
    directory = output / f"layer_{layer:02d}"
    marker = directory / "prepared.json"
    if not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        current_metadata = {
            "format": "icalens.autointerpretability-prepared-layer",
            "schema_version": SCHEMA_VERSION,
            "layer": layer,
            "methods": ["ica", "sae"],
            "n_fragments": n_fragments,
            "n_features": n_features,
        }
        legacy_metadata = {
            "layer": layer,
            "methods": ["ica", "sae"],
            "n_fragments": n_fragments,
            "n_features": n_features,
        }
        if metadata not in (current_metadata, legacy_metadata):
            return False
        for method in ("ica", "sae"):
            activations = np.load(directory / method / "candidate_activations.npy", mmap_mode="r")
            if activations.ndim != 3 or activations.shape[:2] != (n_fragments, FRAGMENT_LENGTH):
                return False
            selection = json.loads(
                (directory / method / "selection.json").read_text(encoding="utf-8")
            )
            accepted = selection.get("accepted")
            if not isinstance(accepted, list) or len(accepted) != n_features:
                return False
            if any(
                not isinstance(item, dict)
                or not 0 <= int(item.get("candidate_position", -1)) < activations.shape[2]
                for item in accepted
            ):
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if metadata == legacy_metadata:
        _write_json(marker, current_metadata)
    return True


def _evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icalens experiment autointerpretability evaluate")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--methods", default="ica,sae")
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument(
        "--feature-position",
        type=int,
        default=None,
        help="Evaluate one zero-based position in each layer's accepted feature cohort.",
    )
    parser.add_argument("--provider", choices=("tinker", "openai"), default="tinker")
    parser.add_argument("--model", default=None)
    parser.add_argument("--explainer-model", default=None)
    parser.add_argument("--simulator-model", default=None)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Evaluation output directory; required to preserve another evaluator condition.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between OpenAI request starts (default: 1.0).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent simulator requests within each feature (default: 10).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def evaluate_main(argv: Sequence[str] | None = None) -> None:
    args = _evaluate_parser().parse_args(argv)
    if args.n_features is not None and args.feature_position is not None:
        raise ValueError("--n-features and --feature-position are mutually exclusive")
    if args.feature_position is not None and args.feature_position < 0:
        raise ValueError("--feature-position must be nonnegative")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be nonnegative")
    if args.provider == "openai":
        if args.explainer_model is not None or args.simulator_model is not None:
            raise ValueError("OpenAI uses --model, not the Tinker model-role options")
        args.model = args.model or DEFAULT_OPENAI_EVALUATOR_MODEL
        if args.output is None:
            raise ValueError("OpenAI evaluation requires --output to preserve prior conditions")
        asyncio.run(_evaluate_openai_main(args))
        return
    if args.model is not None:
        raise ValueError("Tinker uses --explainer-model and --simulator-model, not --model")
    if args.output is None:
        raise ValueError("Tinker evaluation requires --output to preserve prior conditions")
    args.explainer_model = args.explainer_model or DEFAULT_TINKER_EXPLAINER_MODEL
    args.simulator_model = args.simulator_model or DEFAULT_TINKER_SIMULATOR_MODEL
    _evaluate_tinker_main(args)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _selected_records(selection: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    accepted = selection["accepted"]
    if not isinstance(accepted, list) or not all(isinstance(row, dict) for row in accepted):
        raise ValueError("selection accepted records must be a list of objects")
    if args.feature_position is not None:
        position = int(args.feature_position)
        if position >= len(accepted):
            raise ValueError(
                f"--feature-position {position} is unavailable; "
                f"the cohort contains {len(accepted)} features"
            )
        return [accepted[position]]
    return cast(list[dict[str, Any]], accepted[: args.n_features] if args.n_features else accepted)


def _modern_simulator_messages(explanation: str, tokens: Sequence[str]) -> list[dict[str, str]]:
    indexed = [{"index": index, "token": token} for index, token in enumerate(tokens)]
    return [
        {
            "role": "system",
            "content": (
                "Predict one activation for every token using only the supplied feature "
                "explanation. Return valid JSON with exactly one key, activations, containing "
                "exactly 64 integer labels in token-index order."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Feature explanation: {explanation}\n\n"
                "Use integer labels from 0 (inactive) to 10 (strongest activation).\n\n"
                f"Tokens:\n{json.dumps(indexed, ensure_ascii=False)}"
            ),
        },
    ]


def _object_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _modern_expected_activations(choice: Any) -> tuple[str, np.ndarray]:
    """Port Cunningham's modern output-logprob expected-label calculation."""
    message = _object_field(choice, "message")
    content = str(_object_field(message, "content", "") or "")
    payload = json.loads(content)
    hard_labels = payload.get("activations")
    if not isinstance(hard_labels, list) or len(hard_labels) != FRAGMENT_LENGTH:
        raise ValueError(f"expected {FRAGMENT_LENGTH} activation labels")
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 10 for x in hard_labels):
        raise ValueError("activation labels must be integers in 0 through 10")
    logprobs = _object_field(choice, "logprobs")
    logprob_content = _object_field(logprobs, "content")
    if not logprob_content:
        raise ValueError("model response did not include output-token log probabilities")
    array_start, array_end = content.find("["), content.rfind("]")
    spans = list(re.finditer(r"(?<![\d.])-?\d+(?![\d.])", content[array_start : array_end + 1]))
    if len(spans) != FRAGMENT_LENGTH:
        raise ValueError("could not align JSON activation labels with response text")
    absolute = [(array_start + match.start(), array_start + match.end()) for match in spans]
    ranges: list[tuple[int, int, Any]] = []
    cursor = 0
    for item in logprob_content:
        token = str(_object_field(item, "token", ""))
        ranges.append((cursor, cursor + len(token), item))
        cursor += len(token)
    expected: list[float] = []
    for start, end in absolute:
        item = next(
            (entry for left, right, entry in ranges if left <= start and end <= right), None
        )
        if item is None:
            raise ValueError("could not align an activation label with a logprob token")
        candidates: dict[int, float] = {}
        alternatives = list(_object_field(item, "top_logprobs", []) or []) + [item]
        for alternative in alternatives:
            stripped = str(_object_field(alternative, "token", "")).strip(' \t\r\n,[]{}:"')
            if re.fullmatch(r"\d+", stripped) and 0 <= int(stripped) <= 10:
                label = int(stripped)
                probability = float(_object_field(alternative, "logprob"))
                candidates[label] = max(candidates.get(label, -math.inf), probability)
        if not candidates:
            raise ValueError("no valid activation labels were present in top_logprobs")
        labels = np.asarray(list(candidates), dtype=np.float64)
        values = np.asarray(list(candidates.values()), dtype=np.float64)
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        expected.append(float(np.dot(labels, probabilities)))
    return content, np.asarray(expected, dtype=np.float64)


async def _openai_retry(call: Any, *, retries: int) -> Any:
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            return await call()
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise AssertionError("unreachable")


class _AsyncRequestPacer:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.lock = asyncio.Lock()
        self.last_start = 0.0

    async def wait(self) -> None:
        async with self.lock:
            now = asyncio.get_running_loop().time()
            delay = self.interval - (now - self.last_start)
            if delay > 0:
                await asyncio.sleep(delay)
            self.last_start = asyncio.get_running_loop().time()


async def _evaluate_openai_main(args: argparse.Namespace) -> None:
    preparation = args.input.expanduser().resolve()
    output = (args.output or args.input).expanduser().resolve()
    stored = json.loads((preparation / "run.json").read_text(encoding="utf-8"))
    prepared = stored["resolved"]
    available = [int(value) for value in prepared["layers"]]
    layers = available if args.layers == "all" else _parse_layers(args.layers, available)
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    if not methods or set(methods).difference({"ica", "sae"}):
        raise ValueError("--methods must contain ica and/or sae")
    tasks: list[tuple[int, str, dict[str, Any]]] = []
    for layer in layers:
        for method in methods:
            selection = json.loads(
                (preparation / f"layer_{layer:02d}" / method / "selection.json").read_text()
            )
            selected = _selected_records(selection, args)
            tasks.extend((layer, method, record) for record in selected)
    simulation_count = sum(
        len(record["valid_top"]) + len(record["valid_random"]) for _, _, record in tasks
    )
    print(f"{len(tasks)} features; {len(tasks)} explanations; {simulation_count} simulations")
    if args.dry_run:
        return
    source = source_provenance()
    resolved = {
        "format": "icalens.autointerpretability-modern-evaluation",
        "schema_version": SCHEMA_VERSION,
        "preparation": str(preparation),
        "preparation_resolved": prepared,
        "layers": layers,
        "methods": methods,
        "n_features": args.n_features,
    }
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="evaluating")
    prompt_hash = hashlib.sha256(
        json.dumps(
            {
                "version": "cunningham-modern-v1",
                "explanation": explanation_messages(
                    [["x"] * FRAGMENT_LENGTH] * EXAMPLES_PER_SPLIT,
                    np.zeros((EXAMPLES_PER_SPLIT, FRAGMENT_LENGTH)),
                ),
                "simulator": _modern_simulator_messages("example", ["x"] * FRAGMENT_LENGTH),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    evaluation = {
        "provider": "openai",
        "explainer_model": args.model,
        "simulator_model": args.model,
        "protocol": "cunningham-modern",
        "prompt_hash": prompt_hash,
        "prediction": "output-token-top-logprob-expected-label",
        "ground_truth": "continuous-feature-activation",
    }
    run.validate_section("evaluation", evaluation)
    completed = _modern_completed_units(output, tasks, evaluation)
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · Cunningham-modern autointerpretability",
        completed=completed,
        total=len(tasks) + simulation_count,
        source_dirty=bool(source.get("dirty")),
        detail_filename="evaluation-detail.log",
    )
    with display:
        warn_if_dirty(source)
        load_dotenv(args.env_file)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(f"OPENAI_API_KEY is missing from {args.env_file}")
        try:
            openai = importlib.import_module("openai")
        except ImportError as error:
            raise RuntimeError("Install with `uv sync --extra autointerpretability`") from error
        client = openai.AsyncOpenAI()
        semaphore = asyncio.Semaphore(args.max_concurrent)
        pacer = _AsyncRequestPacer(args.request_delay)
        fragments = [json.loads(line) for line in (preparation / "fragments.jsonl").open()]
        try:
            for number, (layer, method, record) in enumerate(tasks, 1):
                feature = int(record["feature"])
                display.phase(
                    "Evaluating feature", layer=layer, method=method.upper(), feature=f"C{feature}"
                )
                source_dir = preparation / f"layer_{layer:02d}" / method
                results = output / f"layer_{layer:02d}" / method / "results"
                results.mkdir(parents=True, exist_ok=True)
                destination = results / f"feature_{feature}.json"
                if _modern_result_valid(destination, evaluation, feature):
                    log(f"Reusing {layer}/{method}/C{feature} ({number}/{len(tasks)})")
                    continue
                try:
                    await _evaluate_openai_feature(
                        client=client,
                        semaphore=semaphore,
                        source_dir=source_dir,
                        record=record,
                        fragments=fragments,
                        destination=destination,
                        model=args.model,
                        retries=args.retries,
                        prompt_hash=prompt_hash,
                        display=display,
                        pacer=pacer,
                    )
                except Exception as error:
                    _write_json(
                        destination.with_suffix(".error.json"),
                        {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "reason": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    log(f"Feature failed: layer {layer} {method} {feature}: {error}")
                else:
                    destination.with_suffix(".error.json").unlink(missing_ok=True)
                    log(f"Evaluated {layer}/{method}/C{feature} ({number}/{len(tasks)})")
        finally:
            await client.close()
        _summarize_modern(output, preparation, layers=layers, methods=methods)
        complete = display.completed == display.total
        run.set_status("complete" if complete else "incomplete", complete=complete)
        display.set_outcome("complete" if complete else "incomplete")


async def _evaluate_openai_feature(
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    source_dir: Path,
    record: dict[str, Any],
    fragments: list[dict[str, Any]],
    destination: Path,
    model: str,
    retries: int,
    prompt_hash: str,
    display: ExperimentDisplay,
    pacer: _AsyncRequestPacer,
) -> None:
    values = np.load(source_dir / "candidate_activations.npy", mmap_mode="r")
    feature_values = np.asarray(values[:, :, int(record["candidate_position"])], dtype=np.float32)
    feature = int(record["feature"])
    train = [int(value) for value in record["train_top"]]
    explanation_path = destination.parent / f"feature_{feature}.explanation.json"
    explanation_record = _read_json_object(explanation_path)
    if (
        explanation_record
        and explanation_record.get("model") == model
        and explanation_record.get("prompt_hash") == prompt_hash
    ):
        explanation = str(explanation_record["explanation"])
    else:

        async def explain() -> str:
            await pacer.wait()
            response = await client.chat.completions.create(
                model=model,
                messages=explanation_messages(
                    [fragments[i]["tokens"] for i in train], feature_values[train]
                ),
                temperature=0,
                max_tokens=160,
            )
            value = str(response.choices[0].message.content or "").strip()
            if not value:
                raise ValueError("explainer returned an empty explanation")
            return value

        explanation = await _openai_retry(explain, retries=retries)
        _write_json(
            explanation_path,
            {
                "provider": "openai",
                "model": model,
                "prompt_hash": prompt_hash,
                "feature": feature,
                "explanation": explanation,
            },
        )
        display.advance(refresh=True)
    validation = [*map(int, record["valid_top"]), *map(int, record["valid_random"])]
    simulation_dir = destination.parent / f"feature_{feature}.simulations"
    simulation_dir.mkdir(exist_ok=True)

    async def simulate(item: tuple[int, int]) -> np.ndarray:
        position, fragment_index = item
        cache = simulation_dir / f"fragment_{position:02d}.json"
        cached = _read_json_object(cache)
        if _modern_simulation_valid(cached, prompt_hash, fragment_index):
            assert cached is not None
            return np.asarray(cached["activations"], dtype=np.float64)

        async def request() -> tuple[str, np.ndarray]:
            async with semaphore:
                await pacer.wait()
                response = await client.chat.completions.create(
                    model=model,
                    messages=_modern_simulator_messages(
                        explanation, fragments[fragment_index]["tokens"]
                    ),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "token_activations",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "activations": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 0, "maximum": 10},
                                        "minItems": FRAGMENT_LENGTH,
                                        "maxItems": FRAGMENT_LENGTH,
                                    }
                                },
                                "required": ["activations"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    logprobs=True,
                    top_logprobs=20,
                    temperature=0,
                    max_tokens=256,
                )
            return _modern_expected_activations(response.choices[0])

        raw, predicted = await _openai_retry(request, retries=retries)
        _write_json(
            cache,
            {
                "fragment_index": fragment_index,
                "activations": predicted.tolist(),
                "prompt_hash": prompt_hash,
                "raw_completion": raw,
            },
        )
        display.advance(refresh=True)
        return np.asarray(predicted, dtype=np.float64)

    predicted = np.stack(await asyncio.gather(*(simulate(item) for item in enumerate(validation))))
    actual = feature_values[validation].astype(np.float64)
    split = len(record["valid_top"])
    _write_json(
        destination,
        {
            "status": "complete",
            "provider": "openai",
            "explainer_model": model,
            "simulator_model": model,
            "protocol": "cunningham-modern",
            "prompt_hash": prompt_hash,
            "feature": feature,
            "explanation": explanation,
            "combined_score": _modern_pearson(actual, predicted),
            "top_score": _modern_pearson(actual[:split], predicted[:split]),
            "random_score": _modern_pearson(actual[split:], predicted[split:]),
            "train_top": train,
            "valid_top": record["valid_top"],
            "valid_random": record["valid_random"],
            "actual_activations": actual.tolist(),
            "predicted_activations": predicted.tolist(),
        },
    )


def _flatten_tinker_input(model_input: Any) -> list[int]:
    return [token for chunk in model_input.chunks for token in chunk.tokens]


def _tinker_sample_text(
    *,
    sampler: Any,
    tokenizer: Any,
    renderer: Any,
    renderers_module: Any,
    types: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[list[int], Any, str]:
    if renderer.__class__.__name__ == "TmlV0Renderer":
        prompt = renderer.build_generation_prompt(messages, effort=0.0)
    else:
        prompt = renderer.build_generation_prompt(messages)
    prompt_tokens = _flatten_tinker_input(prompt)
    result = sampler.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    ).result()
    sequence = result.sequences[0]
    response, _ = renderer.parse_response(list(sequence.tokens))
    raw = renderers_module.get_text_content(response).strip()
    return prompt_tokens, sequence, raw


def _balanced_json_end(raw: str, start: int) -> int | None:
    opener = raw[start]
    closer = {"[": "]", "{": "}"}.get(opener)
    if closer is None:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        character = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _tinker_label_spans(raw: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for match in re.finditer(r'"activations"\s*:\s*', raw):
        start = match.end()
        if start < len(raw) and raw[start] in "[{":
            end = _balanced_json_end(raw, start)
            if end is not None:
                candidates.append((start, end))
    if not candidates:
        for match in re.finditer(r"\[", raw):
            end = _balanced_json_end(raw, match.start())
            if end is not None:
                candidates.append((match.start(), end))
    details: list[str] = []
    for start, end in candidates:
        value = raw[start:end]
        if value.startswith("{"):
            pairs = list(re.finditer(r'"(\d+)"\s*:\s*(-?\d+)', value))
            keys = [int(pair.group(1)) for pair in pairs]
            details.append(f"keys={keys!r}")
            if keys == list(range(FRAGMENT_LENGTH)):
                return [(start + pair.start(2), start + pair.end(2)) for pair in pairs]
        else:
            numbers = list(re.finditer(r"(?<![\d.])-?\d+(?![\d.])", value))
            details.append(f"array_length={len(numbers)}")
            if len(numbers) == FRAGMENT_LENGTH:
                return [(start + item.start(), start + item.end()) for item in numbers]
            # Some simulators render an indexed array as
            # [[0, score], [1, score], ..., [63, score]].  This is equivalent
            # to the indexed mapping accepted above; score only the value in
            # each pair so the teacher-forced expectation remains unchanged.
            if len(numbers) == 2 * FRAGMENT_LENGTH:
                indices = [int(item.group()) for item in numbers[::2]]
                if indices == list(range(FRAGMENT_LENGTH)):
                    return [(start + item.start(), start + item.end()) for item in numbers[1::2]]
    detail = ", ".join(details) if details else "no activations object or array found"
    raise ValueError(f"expected labels 0..63 or an exact 64-value array; got {detail}")


def _tinker_expected_activations(
    *,
    sampler: Any,
    tokenizer: Any,
    types: Any,
    prompt_tokens: list[int],
    sequence: Any,
    seed: int | None = None,
) -> tuple[str, np.ndarray]:
    completion_tokens = list(sequence.tokens)
    raw = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    spans = _tinker_label_spans(raw)
    offsets = [0]
    for index in range(1, len(completion_tokens) + 1):
        offsets.append(len(tokenizer.decode(completion_tokens[:index], skip_special_tokens=True)))
    token_indices: list[int] = []
    for left, _ in spans:
        token_index = next(
            (
                index
                for index in range(len(completion_tokens))
                if offsets[index] <= left < offsets[index + 1]
            ),
            None,
        )
        if token_index is None:
            raise ValueError("could not align a numeric label to a generated token")
        token_indices.append(token_index)
    scored = sampler.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens + completion_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=1,
            seed=seed,
            temperature=0.0,
        ),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=20,
    ).result()
    alternatives = scored.topk_prompt_logprobs
    expected: list[float] = []
    for token_index in token_indices:
        position = len(prompt_tokens) + token_index
        candidates: dict[int, float] = {}
        for token_id, logprob in alternatives[position] or []:
            stripped = tokenizer.decode([token_id], skip_special_tokens=True).strip(
                ' \t\r\n,[]{}:"'
            )
            if re.fullmatch(r"\d+", stripped) and 0 <= int(stripped) <= 10:
                label = int(stripped)
                candidates[label] = max(candidates.get(label, -math.inf), float(logprob))
        if not candidates:
            raise ValueError(f"no valid label alternatives at generated token {token_index}")
        labels = np.asarray(list(candidates), dtype=np.float64)
        logps = np.asarray(list(candidates.values()), dtype=np.float64)
        probabilities = np.exp(logps - logps.max())
        probabilities /= probabilities.sum()
        expected.append(float(np.dot(labels, probabilities)))
    return raw, np.asarray(expected, dtype=np.float64)


def _retry_tinker(call: Any, *, retries: int) -> Any:
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            return call(attempt)
        except Exception as error:
            if attempt == retries:
                raise
            wait = 1.0 if isinstance(error, ValueError) else delay
            time.sleep(wait)
            if not isinstance(error, ValueError):
                delay = min(delay * 2, 30.0)
    raise AssertionError("unreachable")


def _evaluate_tinker_main(args: argparse.Namespace) -> None:
    from ._tinker_environment import enter_isolated_tinker_environment

    enter_isolated_tinker_environment(sys.argv[1:])
    preparation = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    stored = json.loads((preparation / "run.json").read_text(encoding="utf-8"))
    prepared = stored["resolved"]
    available = [int(value) for value in prepared["layers"]]
    layers = available if args.layers == "all" else _parse_layers(args.layers, available)
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    if not methods or set(methods).difference({"ica", "sae"}):
        raise ValueError("--methods must contain ica and/or sae")
    tasks: list[tuple[int, str, dict[str, Any]]] = []
    for layer in layers:
        for method in methods:
            selection = json.loads(
                (preparation / f"layer_{layer:02d}" / method / "selection.json").read_text()
            )
            selected = _selected_records(selection, args)
            tasks.extend((layer, method, record) for record in selected)
    simulation_count = sum(
        len(record["valid_top"]) + len(record["valid_random"]) for _, _, record in tasks
    )
    print(
        f"{len(tasks)} features; {len(tasks)} explanations; "
        f"{simulation_count} simulations; {simulation_count} logprob scoring calls"
    )
    if args.dry_run:
        return
    source = source_provenance()
    resolved = {
        "format": "icalens.autointerpretability-modern-evaluation",
        "schema_version": SCHEMA_VERSION,
        "preparation": str(preparation),
        "preparation_resolved": prepared,
        "layers": layers,
        "methods": methods,
        "n_features": args.n_features,
    }
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="evaluating")
    prompt_identity = {
        "version": (
            "cunningham-tinker-v1" if args.sampling_seed is None else "cunningham-tinker-seeded-v1"
        ),
        "explanation": explanation_messages(
            [["x"] * FRAGMENT_LENGTH] * EXAMPLES_PER_SPLIT,
            np.zeros((EXAMPLES_PER_SPLIT, FRAGMENT_LENGTH)),
        ),
        "simulator": simulator_messages("example", ["x"] * FRAGMENT_LENGTH),
    }
    if args.sampling_seed is not None:
        prompt_identity["sampling_seed"] = args.sampling_seed
    prompt_hash = hashlib.sha256(json.dumps(prompt_identity, sort_keys=True).encode()).hexdigest()
    evaluation = {
        "provider": "tinker",
        "explainer_model": args.explainer_model,
        "simulator_model": args.simulator_model,
        "protocol": "cunningham-modern",
        "prompt_hash": prompt_hash,
        "prediction": "teacher-forced-top-logprob-expected-label",
        "ground_truth": "continuous-feature-activation",
    }
    if args.sampling_seed is not None:
        evaluation["sampling_seed"] = args.sampling_seed
    run.validate_section("evaluation", evaluation)
    completed = _modern_completed_units(output, tasks, evaluation)
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · Cunningham-modern · Tinker",
        completed=completed,
        total=len(tasks) + simulation_count,
        source_dirty=bool(source.get("dirty")),
        detail_filename="evaluation-detail.log",
    )
    with display:
        warn_if_dirty(source)
        load_dotenv(args.env_file)
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError(f"TINKER_API_KEY is missing from {args.env_file}")
        try:
            tinker = importlib.import_module("tinker")
            types = importlib.import_module("tinker.types")
            renderers = importlib.import_module("tinker_cookbook.renderers")
            tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
        except ImportError as error:
            raise RuntimeError("The isolated Tinker environment is incomplete") from error
        service = tinker.ServiceClient()
        explainer_sampler = service.create_sampling_client(base_model=args.explainer_model)
        simulator_sampler = service.create_sampling_client(base_model=args.simulator_model)
        explainer_tokenizer = tokenizer_utils.get_tokenizer(args.explainer_model)
        simulator_tokenizer = tokenizer_utils.get_tokenizer(args.simulator_model)
        explainer_renderer = renderers.get_renderer(
            "tml_v0", explainer_tokenizer, model_name=args.explainer_model
        )
        simulator_renderer = renderers.get_renderer(
            "qwen3_8_disable_thinking", simulator_tokenizer, model_name=args.simulator_model
        )
        fragments = [json.loads(line) for line in (preparation / "fragments.jsonl").open()]
        for number, (layer, method, record) in enumerate(tasks, 1):
            feature = int(record["feature"])
            display.phase(
                "Evaluating feature", layer=layer, method=method.upper(), feature=f"C{feature}"
            )
            source_dir = preparation / f"layer_{layer:02d}" / method
            results = output / f"layer_{layer:02d}" / method / "results"
            results.mkdir(parents=True, exist_ok=True)
            destination = results / f"feature_{feature}.json"
            if _modern_result_valid(destination, evaluation, feature):
                log(f"Reusing {layer}/{method}/C{feature} ({number}/{len(tasks)})")
                continue
            try:
                _evaluate_tinker_feature(
                    source_dir=source_dir,
                    record=record,
                    fragments=fragments,
                    destination=destination,
                    explainer_model=args.explainer_model,
                    simulator_model=args.simulator_model,
                    explainer_sampler=explainer_sampler,
                    simulator_sampler=simulator_sampler,
                    explainer_tokenizer=explainer_tokenizer,
                    simulator_tokenizer=simulator_tokenizer,
                    explainer_renderer=explainer_renderer,
                    simulator_renderer=simulator_renderer,
                    renderers_module=renderers,
                    types=types,
                    retries=args.retries,
                    prompt_hash=prompt_hash,
                    sampling_seed=args.sampling_seed,
                    display=display,
                    max_concurrent=args.max_concurrent,
                )
            except Exception as error:
                _write_json(
                    destination.with_suffix(".error.json"),
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                log(f"Feature failed: layer {layer} {method} {feature}: {error}")
            else:
                destination.with_suffix(".error.json").unlink(missing_ok=True)
                log(f"Evaluated {layer}/{method}/C{feature} ({number}/{len(tasks)})")
        _summarize_modern(output, preparation, layers=layers, methods=methods)
        complete = display.completed == display.total
        run.set_status("complete" if complete else "incomplete", complete=complete)
        display.set_outcome("complete" if complete else "incomplete")


def _evaluate_tinker_feature(
    *,
    source_dir: Path,
    record: dict[str, Any],
    fragments: list[dict[str, Any]],
    destination: Path,
    explainer_model: str,
    simulator_model: str,
    explainer_sampler: Any,
    simulator_sampler: Any,
    explainer_tokenizer: Any,
    simulator_tokenizer: Any,
    explainer_renderer: Any,
    simulator_renderer: Any,
    renderers_module: Any,
    types: Any,
    retries: int,
    prompt_hash: str,
    sampling_seed: int | None = None,
    display: ExperimentDisplay,
    max_concurrent: int,
) -> None:
    values = np.load(source_dir / "candidate_activations.npy", mmap_mode="r")
    feature_values = np.asarray(values[:, :, int(record["candidate_position"])], dtype=np.float32)
    feature = int(record["feature"])
    train = [int(value) for value in record["train_top"]]
    explanation_path = destination.parent / f"feature_{feature}.explanation.json"
    explanation_record = _read_json_object(explanation_path)
    if (
        explanation_record
        and explanation_record.get("model") == explainer_model
        and explanation_record.get("prompt_hash") == prompt_hash
    ):
        explanation = str(explanation_record["explanation"])
    else:
        explanation = _retry_tinker(
            lambda attempt: _tinker_sample_text(
                sampler=explainer_sampler,
                tokenizer=explainer_tokenizer,
                renderer=explainer_renderer,
                renderers_module=renderers_module,
                types=types,
                messages=explanation_messages(
                    [fragments[index]["tokens"] for index in train], feature_values[train]
                ),
                max_tokens=512,
                temperature=0.0 if attempt == 0 else 0.2,
                seed=sampling_seed,
            )[2],
            retries=retries,
        )
        _write_json(
            explanation_path,
            {
                "provider": "tinker",
                "model": explainer_model,
                "prompt_hash": prompt_hash,
                "feature": feature,
                "explanation": explanation,
            },
        )
        display.advance(refresh=True)
    validation = [*map(int, record["valid_top"]), *map(int, record["valid_random"])]
    simulation_dir = destination.parent / f"feature_{feature}.simulations"
    simulation_dir.mkdir(exist_ok=True)

    def simulate(item: tuple[int, int]) -> np.ndarray:
        position, fragment_index = item
        cache = simulation_dir / f"fragment_{position:02d}.json"
        cached = _read_json_object(cache)
        if _modern_simulation_valid(cached, prompt_hash, fragment_index):
            assert cached is not None
            return np.asarray(cached["activations"], dtype=np.float64)

        def request(attempt: int) -> tuple[str, np.ndarray]:
            messages = simulator_messages(explanation, fragments[fragment_index]["tokens"])
            if attempt:
                messages[-1]["content"] += (
                    f"\n\nRetry {attempt}: the previous response had an invalid index count. "
                    "Rebuild the object from scratch and verify that every key 0 through 63 "
                    "appears exactly once."
                )
            prompt_tokens, sequence, parsed_raw = _tinker_sample_text(
                sampler=simulator_sampler,
                tokenizer=simulator_tokenizer,
                renderer=simulator_renderer,
                renderers_module=renderers_module,
                types=types,
                messages=messages,
                max_tokens=768,
                temperature=0.0 if attempt == 0 else 0.2,
                seed=sampling_seed,
            )
            decoded_raw, predicted = _tinker_expected_activations(
                sampler=simulator_sampler,
                tokenizer=simulator_tokenizer,
                types=types,
                prompt_tokens=prompt_tokens,
                sequence=sequence,
                seed=sampling_seed,
            )
            return parsed_raw or decoded_raw, predicted

        raw, predicted = _retry_tinker(request, retries=retries)
        _write_json(
            cache,
            {
                "fragment_index": fragment_index,
                "activations": predicted.tolist(),
                "prompt_hash": prompt_hash,
                "raw_completion": raw,
            },
        )
        display.advance(refresh=True)
        return cast(np.ndarray[Any, Any], predicted)

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        predicted = np.stack(list(executor.map(simulate, enumerate(validation))))
    actual = feature_values[validation].astype(np.float64)
    split = len(record["valid_top"])
    _write_json(
        destination,
        {
            "status": "complete",
            "provider": "tinker",
            "explainer_model": explainer_model,
            "simulator_model": simulator_model,
            "sampling_seed": sampling_seed,
            "protocol": "cunningham-modern",
            "prompt_hash": prompt_hash,
            "feature": feature,
            "explanation": explanation,
            "combined_score": _modern_pearson(actual, predicted),
            "top_score": _modern_pearson(actual[:split], predicted[:split]),
            "random_score": _modern_pearson(actual[split:], predicted[split:]),
            "train_top": train,
            "valid_top": record["valid_top"],
            "valid_random": record["valid_random"],
            "actual_activations": actual.tolist(),
            "predicted_activations": predicted.tolist(),
        },
    )


def _modern_pearson(actual: np.ndarray, predicted: np.ndarray) -> float:
    left, right = np.asarray(actual).reshape(-1), np.asarray(predicted).reshape(-1)
    if left.size == 0 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _modern_simulation_valid(value: dict[str, Any] | None, prompt_hash: str, fragment: int) -> bool:
    if (
        not value
        or value.get("prompt_hash") != prompt_hash
        or value.get("fragment_index") != fragment
    ):
        return False
    activations = value.get("activations")
    return bool(
        isinstance(activations, list)
        and len(activations) == FRAGMENT_LENGTH
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in activations)
    )


def _modern_result_valid(path: Path, evaluation: dict[str, Any], feature: int) -> bool:
    value = _read_json_object(path)
    return bool(
        value
        and value.get("status") in {"complete", "disabled"}
        and value.get("protocol") == "cunningham-modern"
        and value.get("prompt_hash") == evaluation["prompt_hash"]
        and value.get("feature") == feature
    )


def _modern_completed_units(
    root: Path, tasks: Sequence[tuple[int, str, dict[str, Any]]], evaluation: dict[str, Any]
) -> int:
    count = 0
    for layer, method, record in tasks:
        feature = int(record["feature"])
        results = root / f"layer_{layer:02d}" / method / "results"
        if _modern_result_valid(results / f"feature_{feature}.json", evaluation, feature):
            count += 1 + len(record["valid_top"]) + len(record["valid_random"])
            continue
        explanation = _read_json_object(results / f"feature_{feature}.explanation.json")
        count += int(
            bool(
                explanation
                and explanation.get("model") == evaluation["explainer_model"]
                and explanation.get("prompt_hash") == evaluation["prompt_hash"]
            )
        )
        validation = [*map(int, record["valid_top"]), *map(int, record["valid_random"])]
        count += sum(
            _modern_simulation_valid(
                _read_json_object(
                    results / f"feature_{feature}.simulations" / f"fragment_{i:02d}.json"
                ),
                str(evaluation["prompt_hash"]),
                fragment,
            )
            for i, fragment in enumerate(validation)
        )
    return count


def _summarize_modern(
    root: Path, preparation: Path, *, layers: list[int], methods: list[str]
) -> None:
    rows: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    for layer in layers:
        for method in methods:
            result_dir = root / f"layer_{layer:02d}" / method / "results"
            condition = []
            for path in sorted(result_dir.glob("feature_*.json")) if result_dir.is_dir() else []:
                if ".explanation." in path.name or ".error." in path.name:
                    continue
                value = json.loads(path.read_text())
                if value.get("status") != "complete":
                    continue
                condition.append(value)
                rows.append({"layer": layer, "method": method, **value})
            scores = np.asarray([row["combined_score"] for row in condition], dtype=np.float64)
            finite = scores[np.isfinite(scores)]
            selection = json.loads(
                (preparation / f"layer_{layer:02d}" / method / "selection.json").read_text()
            )
            conditions.append(
                {
                    "layer": layer,
                    "method": method,
                    "completed": len(condition),
                    "defined_scores": len(finite),
                    "undefined_scores": int(len(scores) - len(finite)),
                    "mean_top_random_score": float(scores.mean()) if len(scores) else float("nan"),
                    "selected_features": len(selection["accepted"]),
                    "rejected_candidates": len(selection["rejected"]),
                }
            )
    _write_json(
        root / "results.json",
        {
            "format": "icalens.autointerpretability-modern-results",
            "schema_version": SCHEMA_VERSION,
            "conditions": conditions,
            "features": rows,
        },
    )


def _summarize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icalens experiment autointerpretability summarize")
    parser.add_argument("--input", type=Path, required=True)
    return parser


def summarize_main(argv: Sequence[str] | None = None) -> None:
    args = _summarize_parser().parse_args(argv)
    root = args.input.expanduser().resolve()
    run = json.loads((root / "run.json").read_text())
    resolved = run["resolved"]
    layers = [int(x) for x in resolved["layers"]]
    if resolved.get("format") == "icalens.autointerpretability-modern-evaluation":
        _summarize_modern(
            root,
            Path(resolved["preparation"]),
            layers=layers,
            methods=list(resolved["methods"]),
        )
    else:
        raise ValueError("the input is a preparation directory, not an evaluation output")


def _parse_layers(value: str, available: Sequence[int]) -> list[int]:
    try:
        selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("layers must be comma-separated integers") from error
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("layers must be a nonempty list without duplicates")
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"unavailable layers {missing}; available: {list(available)}")
    return selected


def _normalize_legacy_openwebtext(previous: dict[str, Any], current: dict[str, Any]) -> None:
    previous_dataset = previous.get("dataset")
    current_dataset = current.get("dataset")
    if (
        isinstance(previous_dataset, dict)
        and isinstance(current_dataset, dict)
        and previous_dataset.get("repo_id") == "openwebtext"
        and current_dataset.get("repo_id") == DEFAULT_DATASET
        and previous_dataset.get("revision") == current_dataset.get("revision")
    ):
        previous_dataset["repo_id"] = DEFAULT_DATASET


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)
