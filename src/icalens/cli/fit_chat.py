"""Fit an ICA Lens from a streamed conversation dataset."""

from __future__ import annotations

import argparse
import resource
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens._capture import capture_resid_post

from ._status import log

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
TOKEN_BUDGET = 1_000
CONTEXT_LENGTH = 1_024
DEFAULT_OUTPUT = Path("icalens-output") / "icalens-qwen2.5-0.5b-instruct"


@dataclass(frozen=True)
class ChatDocument:
    """One template-formatted conversation and its eligible token positions."""

    input_ids: torch.Tensor
    candidate_positions: torch.Tensor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens fit chat", description=__doc__)
    parser.add_argument(
        "--model", default=MODEL_ID, help=f"Model repository (default: {MODEL_ID})."
    )
    parser.add_argument(
        "--dataset", default=DATASET_ID, help=f"Dataset repository (default: {DATASET_ID})."
    )
    parser.add_argument(
        "--split", default=DATASET_SPLIT, help=f"Dataset split (default: {DATASET_SPLIT})."
    )
    parser.add_argument(
        "--messages-field",
        default="messages",
        help="Dataset field containing role/content messages (default: messages).",
    )
    parser.add_argument(
        "--token-scope",
        choices=("assistant", "user", "content", "all"),
        default="all",
        help="Tokens eligible for fitting (default: all, including template tokens).",
    )
    parser.add_argument(
        "--layers",
        default="12",
        help="Comma-separated transformer-layer indices, or 'all' (default: 12).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--icalens-preprocessing",
        choices=("none", "l2", "geometric-median-l2"),
        default="none",
        help=(
            "Activation transform before standard FastICA centering and whitening (default: none)."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        metavar="OWNER/REPO",
        help="Upload the saved lens to a Hugging Face Model repository.",
    )
    parser.add_argument("--private", action="store_true", help="Create a private Hub repository.")
    parser.add_argument("--hub-revision", default="main")
    parser.add_argument("--commit-message", default="Upload ICA Lens artifacts")
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Maximum formatted conversation length (default: 1024).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Fixed number of FastICA iterations (default: 20).",
    )
    parser.add_argument(
        "--objective-every",
        type=int,
        default=1,
        help="Record objective percentiles every N iterations (default: 1).",
    )
    parser.add_argument(
        "--candidate-tokens",
        type=int,
        default=None,
        help="Candidate pool size; defaults to --token-budget.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=TOKEN_BUDGET,
        help="Number of sampled token activations used to fit ICA.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=None,
        help="Cap this process's PyTorch CUDA allocator, for example 16.",
    )
    parser.add_argument(
        "--fit-batch-size",
        type=int,
        default=8192,
        help="Activation rows processed on CUDA at once; 0 uses all rows (default: 8192).",
    )
    parser.add_argument(
        "--capture-layers-at-once",
        type=int,
        default=0,
        help="Capture and fit this many layers per model pass; 0 uses all requested layers.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("'icalens fit chat' requires a CUDA device")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if args.max_vram_gb is not None:
        set_cuda_memory_limit(args.max_vram_gb)
    if args.fit_batch_size < 0:
        raise ValueError("--fit-batch-size must be non-negative")
    if args.objective_every <= 0:
        raise ValueError("--objective-every must be positive")
    if args.capture_layers_at_once < 0:
        raise ValueError("--capture-layers-at-once must be non-negative")
    torch.cuda.reset_peak_memory_stats()

    log(f"Resolving model {args.model} and dataset {args.dataset} revisions...")
    api = HfApi()
    model_revision = api.model_info(args.model).sha
    dataset_revision = api.dataset_info(args.dataset).sha
    if model_revision is None or dataset_revision is None:
        raise RuntimeError("Could not resolve exact model and dataset revisions.")

    log(f"Loading tokenizer and {args.dataset} candidate conversations...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_revision, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fit_chat.py requires a fast tokenizer for role-aware token offsets.")
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{args.model} tokenizer does not define a chat template.")
    candidate_tokens = args.token_budget if args.candidate_tokens is None else args.candidate_tokens
    documents = load_chat_documents(
        tokenizer,
        dataset_id=args.dataset,
        dataset_revision=str(dataset_revision),
        split=args.split,
        messages_field=args.messages_field,
        token_scope=args.token_scope,
        candidate_token_budget=candidate_tokens,
        context_length=args.context_length,
    )
    selected_positions = sample_positions(
        documents,
        token_budget=args.token_budget,
        seed=args.seed,
    )

    log(f"Loading {args.model}@{model_revision} on CUDA...")
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        revision=model_revision,
    )
    model.eval()

    layers = parse_layers(args.layers, layer_count=int(model.config.num_hidden_layers))
    log(f"Requested layers: {','.join(map(str, layers))}")
    lens = ICALens(
        model_id=args.model,
        model_revision=str(model_revision),
        model_type="instruct",
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        icalens_preprocessing=args.icalens_preprocessing,
    )
    output = args.output.expanduser().resolve()

    verification_probes: dict[int, torch.Tensor] = {}
    capture_group_size = args.capture_layers_at_once or len(layers)
    for group_start in range(0, len(layers), capture_group_size):
        layer_group = layers[group_start : group_start + capture_group_size]
        log(
            f"Capturing {args.token_budget} sampled {args.token_scope} activations for layers "
            f"{','.join(map(str, layer_group))}..."
        )
        activations_by_layer = capture_activations(
            model,
            documents=documents,
            selected_positions=selected_positions,
            layers=layer_group,
        )
        for layer in layer_group:
            activations = activations_by_layer[layer]
            verification_probes[layer] = activations[:8].clone()
            sample_count = int(activations.shape[0])
            hidden_size = int(activations.shape[1])
            minimum_samples = hidden_size + 1
            if sample_count < minimum_samples:
                raise ValueError(
                    f"Cannot fit a full {hidden_size}-component ICA Lens from only "
                    f"{sample_count} token activations: centering limits the rank to at most "
                    f"{sample_count - 1}. Increase --token-budget to at least "
                    f"{minimum_samples}; if --candidate-tokens is set, it must be at least "
                    "as large."
                )
            fit_batch_size = sample_count if args.fit_batch_size == 0 else args.fit_batch_size
            log(
                f"Fitting layer {layer} from {sample_count} {args.token_scope} tokens "
                f"with {hidden_size} components, max_iter={args.max_iter}, "
                f"fit_batch_size={fit_batch_size}..."
            )
            lens.fit(
                activations,
                layer=layer,
                n_components=hidden_size,
                algorithm="parallel",
                fun="logcosh",
                max_iter=args.max_iter,
                random_state=args.seed,
                progress=True,
                device="cuda",
                batch_size=fit_batch_size,
                objective_every=args.objective_every,
                provenance={
                    "dataset": {
                        "repo_id": args.dataset,
                        "revision": str(dataset_revision),
                        "split": args.split,
                    },
                    "messages_field": args.messages_field,
                    "token_scope": args.token_scope,
                    "candidate_tokens": candidate_tokens,
                    "fitting_tokens": args.token_budget,
                    "sampling_seed": args.seed,
                    "context_length": args.context_length,
                },
            )
            output = lens.save(output)
            log(
                f"Checkpointed layer {layer} to {output}; available layers: {lens.available_layers}"
            )
        del activations, activations_by_layer

    log(f"Saved {len(layers)} layer(s) to {output}")
    log(f"Available layers: {lens.available_layers}")
    log(f"Dataset: {args.dataset}@{dataset_revision} ({args.split})")
    log(f"Token scope: {args.token_scope}")
    if args.push_to_hub is not None:
        log(f"Uploading to Hugging Face Model repository {args.push_to_hub}...")
        uploaded = lens.push_to_hub(
            args.push_to_hub,
            private=args.private,
            revision=args.hub_revision,
            commit_message=args.commit_message,
        )
        cloud = ICALens.from_pretrained(
            args.push_to_hub, revision=args.hub_revision, force_download=True
        )
        for layer in layers:
            probe = verification_probes[layer]
            torch.testing.assert_close(
                cloud.transform(probe, layer=layer), lens.transform(probe, layer=layer)
            )
        log(f"Verified cloud round-trip: {uploaded}")
    peak_gib = torch.cuda.max_memory_reserved() / 1024**3
    log(f"Peak PyTorch CUDA memory reserved: {peak_gib:.2f} GiB")
    log(f"Peak process resident memory (RSS): {peak_rss_gib():.2f} GiB")


def load_chat_documents(
    tokenizer: Any,
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    messages_field: str,
    token_scope: str,
    candidate_token_budget: int,
    context_length: int,
) -> list[ChatDocument]:
    """Stream and format conversations until enough eligible tokens are collected."""
    if candidate_token_budget <= 0:
        raise ValueError("--candidate-tokens must be positive")
    dataset = load_dataset(
        dataset_id,
        split=split,
        revision=dataset_revision,
        streaming=True,
    )
    documents: list[ChatDocument] = []
    captured_tokens = 0
    skipped = 0
    progress = tqdm(
        total=candidate_token_budget,
        desc="Tokenize conversations",
        unit="tok",
        dynamic_ncols=True,
    )
    try:
        for row in dataset:
            messages = row.get(messages_field) if isinstance(row, dict) else None
            try:
                document = format_conversation(
                    tokenizer,
                    messages,
                    token_scope=token_scope,
                    context_length=context_length,
                )
            except (TypeError, ValueError):
                skipped += 1
                continue
            if document is None:
                skipped += 1
                continue

            remaining = candidate_token_budget - captured_tokens
            positions = document.candidate_positions[:remaining]
            if positions.numel() == 0:
                skipped += 1
                continue
            documents.append(ChatDocument(document.input_ids, positions))
            count = int(positions.shape[0])
            captured_tokens += count
            progress.update(count)
            progress.set_postfix(conversations=len(documents), skipped=skipped, refresh=False)
            if captured_tokens >= candidate_token_budget:
                break
    finally:
        progress.close()

    if captured_tokens < candidate_token_budget:
        raise RuntimeError(
            f"{dataset_id} yielded only {captured_tokens} eligible tokens; "
            f"expected {candidate_token_budget}."
        )
    log(
        f"Loaded {captured_tokens} candidate {token_scope} tokens from "
        f"{len(documents)} conversations ({skipped} skipped)."
    )
    return documents


def format_conversation(
    tokenizer: Any,
    messages: Any,
    *,
    token_scope: str,
    context_length: int,
) -> ChatDocument | None:
    """Apply the chat template and locate eligible content tokens by offsets."""
    normalized = normalize_messages(messages)
    rendered = tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not rendered:
        return None

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=context_length,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if not isinstance(input_ids, list) or not isinstance(offsets, list):
        raise TypeError("tokenizer did not return flat input IDs and offsets")
    if not input_ids:
        return None

    if token_scope == "all":
        candidate_positions = list(range(len(input_ids)))
    else:
        spans = content_spans(rendered, normalized, token_scope=token_scope)
        candidate_positions = [
            index for index, offset in enumerate(offsets) if token_overlaps_spans(offset, spans)
        ]
    if not candidate_positions:
        return None
    return ChatDocument(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        candidate_positions=torch.tensor(candidate_positions, dtype=torch.long),
    )


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Validate the common Hugging Face role/content conversation schema."""
    if not isinstance(messages, list) or not messages:
        raise TypeError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError("message content must be a non-empty string")
        normalized.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError("conversation has no assistant message")
    return normalized


def content_spans(
    rendered: str,
    messages: list[dict[str, str]],
    *,
    token_scope: str,
) -> list[tuple[int, int]]:
    """Find selected message contents in the rendered template string."""
    if token_scope not in ("assistant", "user", "content"):
        raise ValueError(f"unsupported token scope: {token_scope!r}")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        content = message["content"]
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError("chat template transformed message content; offsets are ambiguous")
        end = start + len(content)
        if token_scope == "content" or message["role"] == token_scope:
            spans.append((start, end))
        cursor = end
    return spans


def token_overlaps_spans(offset: Any, spans: list[tuple[int, int]]) -> bool:
    """Return whether a non-empty tokenizer character offset intersects a span."""
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        raise TypeError("invalid tokenizer offset")
    start, end = int(offset[0]), int(offset[1])
    return end > start and any(
        end > span_start and start < span_end for span_start, span_end in spans
    )


def sample_positions(
    documents: list[ChatDocument],
    *,
    token_budget: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    """Sample eligible positions uniformly without replacement."""
    total_tokens = sum(int(document.candidate_positions.shape[0]) for document in documents)
    if token_budget <= 0:
        raise ValueError("--token-budget must be positive")
    if token_budget > total_tokens:
        raise ValueError(
            f"--token-budget ({token_budget}) exceeds candidate tokens ({total_tokens})"
        )
    rng = np.random.default_rng(seed)
    flat_indices = np.sort(rng.choice(total_tokens, size=token_budget, replace=False))
    selected: defaultdict[int, list[int]] = defaultdict(list)
    document_start = 0
    selection_index = 0
    for document_index, document in enumerate(documents):
        document_end = document_start + int(document.candidate_positions.shape[0])
        while selection_index < len(flat_indices) and flat_indices[selection_index] < document_end:
            local_index = int(flat_indices[selection_index] - document_start)
            selected[document_index].append(int(document.candidate_positions[local_index]))
            selection_index += 1
        document_start = document_end
    return {
        document_index: torch.tensor(positions, dtype=torch.long)
        for document_index, positions in selected.items()
    }


def capture_activations(
    model: torch.nn.Module,
    *,
    documents: list[ChatDocument],
    selected_positions: dict[int, torch.Tensor],
    layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Capture selected chat-token activations while preserving left context."""
    selected_count = sum(int(positions.shape[0]) for positions in selected_positions.values())
    buffers: dict[int, torch.Tensor] | None = None
    offset = 0
    progress = tqdm(
        total=selected_count,
        desc="Capture chat activations",
        unit="tok",
        dynamic_ncols=True,
    )
    try:
        for document_index, positions_cpu in selected_positions.items():
            input_ids = documents[document_index].input_ids.unsqueeze(0).to("cuda")
            positions = positions_cpu.to("cuda")
            selected_by_layer = capture_resid_post(
                model,
                model_inputs={"input_ids": input_ids},
                layers=layers,
                positions=positions,
            )
            if buffers is None:
                buffers = {
                    layer: torch.empty(
                        (selected_count, int(selected_by_layer[layer].shape[-1])),
                        dtype=selected_by_layer[layer].dtype,
                        device="cpu",
                    )
                    for layer in layers
                }
            count = int(positions.shape[0])
            for layer in layers:
                buffers[layer][offset : offset + count].copy_(selected_by_layer[layer])
            offset += count
            progress.update(count)
            progress.set_postfix(conversation=document_index, refresh=False)
    finally:
        progress.close()
    if buffers is None or offset != selected_count:
        raise RuntimeError(f"captured {offset} activations; expected {selected_count}")
    return buffers


def parse_layers(value: str, *, layer_count: int) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(layer_count))
    try:
        layers = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("--layers must be comma-separated integers or 'all'") from error
    if not layers:
        raise ValueError("--layers cannot be empty")
    invalid = [layer for layer in layers if layer < 0 or layer >= layer_count]
    if invalid:
        raise ValueError(f"layer indices out of range 0..{layer_count - 1}: {invalid}")
    return layers


def peak_rss_gib() -> float:
    """Return this process's lifetime peak resident-set size in GiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def set_cuda_memory_limit(limit_gb: float) -> None:
    """Limit this process through PyTorch's CUDA caching allocator."""
    if not np.isfinite(limit_gb) or limit_gb <= 0:
        raise ValueError("--max-vram-gb must be a finite positive number")
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    requested_bytes = limit_gb * 1024**3
    if requested_bytes > total_bytes:
        total_gib = total_bytes / 1024**3
        raise ValueError(
            f"--max-vram-gb={limit_gb:g} exceeds visible CUDA memory ({total_gib:.2f} GiB)"
        )
    fraction = requested_bytes / total_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    log(
        f"Limited the PyTorch CUDA allocator to {limit_gb:.2f} GiB "
        f"({fraction:.2%} of visible memory)."
    )


if __name__ == "__main__":
    main()
