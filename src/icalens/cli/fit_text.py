"""Fit an ICA Lens from a streamed raw-text dataset."""

from __future__ import annotations

import argparse
import hashlib
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
from icalens._model_framing import resolve_framing_policy

from ._status import log

MODEL_ID = "openai-community/gpt2"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"
TOKEN_BUDGET = 1_000
CONTEXT_LENGTH = 1_024
DEFAULT_OUTPUT = Path("icalens-output") / "icalens-gpt2-small"


@dataclass(frozen=True)
class TextDocument:
    """One model input and the positions eligible for ICA fitting."""

    input_ids: torch.Tensor
    candidate_positions: torch.Tensor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens fit text", description=__doc__)
    parser.add_argument(
        "--model", default=MODEL_ID, help=f"Model repository (default: {MODEL_ID})."
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_ID,
        help=f"Dataset repository or local JSONL/Parquet file (default: {DATASET_ID}).",
    )
    parser.add_argument(
        "--split", default=DATASET_SPLIT, help=f"Dataset split (default: {DATASET_SPLIT})."
    )
    parser.add_argument(
        "--text-field",
        default=TEXT_FIELD,
        help=f"Dataset field containing text (default: {TEXT_FIELD}).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help=f"Maximum tokens captured per document (default: {CONTEXT_LENGTH}).",
    )
    parser.add_argument(
        "--document-framing",
        choices=("auto", "none", "prepend-bos", "prepend-eos"),
        default="auto",
        help=(
            "Context placed before each raw document. 'auto' uses the exact-model "
            "ICALens registry and fails when the model is unknown (default: auto)."
        ),
    )
    parser.add_argument(
        "--refresh-model-registry",
        action="store_true",
        help="Fetch and cache the current model-framing registry from GitHub.",
    )
    parser.add_argument(
        "--layers",
        default="6",
        help="Comma-separated transformer-layer indices, or 'all' (default: 6).",
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
        type=parse_token_budget,
        default=argparse.SUPPRESS,
        help="Candidate pool size, or 'all' for the entire dataset; defaults to --token-budget.",
    )
    parser.add_argument(
        "--token-budget",
        type=parse_token_budget,
        default=TOKEN_BUDGET,
        help="Number of fitting tokens, or 'all' to use the entire dataset.",
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
        raise RuntimeError("'icalens fit text' requires a CUDA device")
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
    revision = api.model_info(args.model).sha
    dataset_revision, dataset_provenance = resolve_text_dataset(
        args.dataset, split=args.split, api=api
    )
    if revision is None:
        raise RuntimeError("Could not resolve the exact model revision.")
    log(f"Loading tokenizer and {args.dataset} candidate tokens...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    document_framing = resolve_document_framing(
        tokenizer,
        args.document_framing,
        model_id=args.model,
        refresh_registry=args.refresh_model_registry,
    )
    log(
        "Document framing: "
        f"{document_framing['strategy']}"
        + (
            f" {document_framing['token']!r} (excluded from fitting samples)."
            if document_framing["strategy"] != "none"
            else "."
        )
    )
    log(f"Document-framing policy source: {document_framing['policy_source']}")
    candidate_tokens = getattr(args, "candidate_tokens", args.token_budget)
    documents = load_pile_documents(
        tokenizer,
        dataset_id=args.dataset,
        dataset_revision=str(dataset_revision),
        split=args.split,
        text_field=args.text_field,
        candidate_token_budget=candidate_tokens,
        context_length=args.context_length,
        document_framing=document_framing,
    )
    candidate_token_count = sum(
        int(document.candidate_positions.shape[0]) for document in documents
    )
    selected_positions = sample_positions(
        documents,
        token_budget=args.token_budget,
        seed=args.seed,
    )
    token_budget = sum(int(positions.shape[0]) for positions in selected_positions.values())
    if args.token_budget is None:
        log(f"Resolved --token-budget all to {token_budget} usable tokens.")

    log(f"Loading {args.model}@{revision} on CUDA...")
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        revision=revision,
    )
    model.eval()

    # Direct block hooks capture resid_post, including the last transformer block.
    layers = parse_layers(args.layers, layer_count=int(model.config.num_hidden_layers))
    log(f"Requested layers: {','.join(map(str, layers))}")
    lens = ICALens(
        model_id=args.model,
        model_revision=str(revision),
        model_type="base",
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        icalens_preprocessing=args.icalens_preprocessing,
    )
    output = args.output.expanduser().resolve()

    capture_group_size = args.capture_layers_at_once or len(layers)
    for group_start in range(0, len(layers), capture_group_size):
        layer_group = layers[group_start : group_start + capture_group_size]
        log(
            f"Capturing {token_budget} sampled activations for layers "
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
                f"Fitting layer {layer} from {sample_count} tokens "
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
                    "dataset": dataset_provenance,
                    "text_field": args.text_field,
                    "token_scope": "all",
                    "candidate_tokens": candidate_token_count,
                    "fitting_tokens": token_budget,
                    "sampling_seed": args.seed,
                    "context_length": args.context_length,
                    "document_framing": document_framing,
                },
            )
            output = lens.save(output)
            log(
                f"Checkpointed layer {layer} to {output}; available layers: {lens.available_layers}"
            )
        del activations, activations_by_layer

    log(f"Saved {len(layers)} layer(s) to {output}")
    log(f"Available layers: {lens.available_layers}")
    peak_gib = torch.cuda.max_memory_reserved() / 1024**3
    log(f"Peak PyTorch CUDA memory reserved: {peak_gib:.2f} GiB")
    log(f"Peak process resident memory (RSS): {peak_rss_gib():.2f} GiB")


def load_pile_documents(
    tokenizer: Any,
    *,
    dataset_id: str = DATASET_ID,
    dataset_revision: str | None = None,
    split: str = DATASET_SPLIT,
    text_field: str = TEXT_FIELD,
    candidate_token_budget: int | None,
    context_length: int,
    document_framing: dict[str, Any] | None = None,
) -> list[TextDocument]:
    """Tokenize text documents until the requested candidate pool is full."""
    if candidate_token_budget is not None and candidate_token_budget <= 0:
        raise ValueError("--candidate-tokens must be positive")
    local_path = Path(dataset_id).expanduser()
    if local_path.is_file():
        suffix = local_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            loader = "json"
        elif suffix == ".parquet":
            loader = "parquet"
        else:
            raise ValueError(
                f"local text datasets must be JSONL, JSON, or Parquet files: {local_path}"
            )
        dataset = load_dataset(
            loader,
            data_files={split: str(local_path.resolve())},
            split=split,
            streaming=True,
        )
    else:
        dataset = load_dataset(dataset_id, split=split, revision=dataset_revision, streaming=True)
    framing = document_framing or {"strategy": "none", "token_id": None}
    prefix_ids = [] if framing["strategy"] == "none" else [int(framing["token_id"])]
    content_limit = context_length - len(prefix_ids)
    if content_limit <= 0:
        raise ValueError("--context-length must leave room for document framing")
    documents: list[TextDocument] = []
    captured_tokens = 0
    progress = tqdm(
        total=candidate_token_budget,
        desc=f"Tokenize {dataset_id}",
        unit="tok",
        dynamic_ncols=True,
    )
    try:
        for row in dataset:
            text = row.get(text_field) if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            remaining = (
                content_limit
                if candidate_token_budget is None
                else candidate_token_budget - captured_tokens
            )
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=min(content_limit, remaining),
            )["input_ids"]
            if encoded:
                input_ids = torch.tensor(prefix_ids + encoded, dtype=torch.long)
                candidate_positions = torch.arange(
                    len(prefix_ids), len(prefix_ids) + len(encoded), dtype=torch.long
                )
                document = TextDocument(
                    input_ids=input_ids,
                    candidate_positions=candidate_positions,
                )
                documents.append(document)
                token_count = int(candidate_positions.shape[0])
                captured_tokens += token_count
                progress.update(token_count)
                progress.set_postfix(documents=len(documents), refresh=False)
            if candidate_token_budget is not None and captured_tokens >= candidate_token_budget:
                break
    finally:
        progress.close()

    if candidate_token_budget is not None and captured_tokens < candidate_token_budget:
        raise RuntimeError(
            f"{dataset_id} yielded only {captured_tokens} candidate tokens; "
            f"expected {candidate_token_budget}."
        )
    log(f"Loaded {captured_tokens} candidate tokens from {len(documents)} documents.")
    return documents


def resolve_text_dataset(
    dataset_id: str,
    *,
    split: str,
    api: HfApi | None = None,
    revision: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve an immutable identity and manifest provenance for a text dataset."""
    local_path = Path(dataset_id).expanduser()
    if local_path.is_file():
        if revision is not None:
            raise ValueError("--dataset-revision cannot be used with a local text dataset")
        resolved = local_path.resolve()
        digest = _file_sha256(resolved)
        return digest, {
            "path": str(resolved),
            "sha256": digest,
            "split": split,
        }
    if local_path.exists():
        raise ValueError(f"local text dataset must be a file: {local_path}")
    resolved_revision = (api or HfApi()).dataset_info(dataset_id, revision=revision).sha
    if resolved_revision is None:
        raise RuntimeError(f"Could not resolve an exact revision for {dataset_id}.")
    return str(resolved_revision), {
        "repo_id": dataset_id,
        "revision": str(resolved_revision),
        "split": split,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_positions(
    documents: list[TextDocument] | list[torch.Tensor],
    *,
    token_budget: int | None,
    seed: int,
) -> dict[int, torch.Tensor]:
    """Sample token positions uniformly without replacement across documents."""
    candidates = [_candidate_positions(document) for document in documents]
    total_tokens = sum(int(positions.shape[0]) for positions in candidates)
    if token_budget is None:
        return {index: positions.clone() for index, positions in enumerate(candidates)}
    if token_budget <= 0:
        raise ValueError("--token-budget must be positive")
    if token_budget > total_tokens:
        raise ValueError(
            f"--token-budget ({token_budget}) exceeds candidate tokens ({total_tokens})"
        )
    rng = np.random.default_rng(seed)
    flat_positions = np.sort(rng.choice(total_tokens, size=token_budget, replace=False))
    selected: defaultdict[int, list[int]] = defaultdict(list)
    document_start = 0
    position_index = 0
    for document_index, positions in enumerate(candidates):
        document_end = document_start + int(positions.shape[0])
        while (
            position_index < len(flat_positions) and flat_positions[position_index] < document_end
        ):
            local_index = int(flat_positions[position_index] - document_start)
            selected[document_index].append(int(positions[local_index]))
            position_index += 1
        document_start = document_end
    return {
        document_index: torch.tensor(positions, dtype=torch.long)
        for document_index, positions in selected.items()
    }


def _candidate_positions(document: TextDocument | torch.Tensor) -> torch.Tensor:
    if isinstance(document, TextDocument):
        return document.candidate_positions
    return torch.arange(document.shape[0], dtype=torch.long)


def resolve_document_framing(
    tokenizer: Any,
    requested: str,
    *,
    model_id: str | None = None,
    refresh_registry: bool = False,
) -> dict[str, Any]:
    """Resolve and validate a portable raw-document prefix policy."""
    if requested not in {"auto", "none", "prepend-bos", "prepend-eos"}:
        raise ValueError(f"unsupported document framing: {requested!r}")

    strategy = requested
    policy = None
    if strategy == "auto":
        if model_id is None:
            raise ValueError("model_id is required when --document-framing is auto")
        policy = resolve_framing_policy(model_id, refresh=refresh_registry)
        strategy = str(policy.entry["document_framing"])

    if strategy == "none":
        result: dict[str, Any] = {
            "requested": requested,
            "strategy": "none",
            "token": None,
            "token_id": None,
            "included_in_fitting_samples": False,
            "policy_source": "explicit-cli",
        }
        return result

    token_kind = "bos" if strategy == "prepend-bos" else "eos"
    token = getattr(tokenizer, f"{token_kind}_token", None)
    token_id = getattr(tokenizer, f"{token_kind}_token_id", None)
    if token is None or token_id is None:
        raise ValueError(f"--document-framing {strategy} requires tokenizer.{token_kind}_token")
    if policy is not None and str(token) != policy.entry["expected_token"]:
        raise ValueError(
            f"The registry expects {policy.entry['expected_token']!r} for {model_id}, "
            f"but its tokenizer exposes {str(token)!r} as {token_kind}_token. "
            "Refusing to fit with an unverified framing token."
        )
    result = {
        "requested": requested,
        "strategy": strategy,
        "token": str(token),
        "token_id": int(token_id),
        "included_in_fitting_samples": False,
        "policy_source": "explicit-cli" if policy is None else policy.source,
    }
    if policy is not None:
        result.update(
            {
                "registry_sha256": policy.sha256,
                "registry_schema_version": policy.schema_version,
                "registry_model_entry": policy.model_id,
                "evidence_url": policy.entry["evidence_url"],
                "evidence_note": policy.entry["evidence_note"],
            }
        )
    return result


def parse_token_budget(value: str) -> int | None:
    """Parse a positive token count or the sentinel 'all'."""
    if value.strip().lower() == "all":
        return None
    try:
        budget = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'") from error
    if budget <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'")
    return budget


def capture_activations(
    model: torch.nn.Module,
    *,
    documents: list[TextDocument] | list[torch.Tensor],
    selected_positions: dict[int, torch.Tensor],
    layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Capture selected positions while preserving each document's left context."""
    selected_count = sum(int(positions.shape[0]) for positions in selected_positions.values())
    buffers: dict[int, torch.Tensor] | None = None
    offset = 0
    progress = tqdm(
        total=selected_count,
        desc="Capture activations",
        unit="tok",
        dynamic_ncols=True,
    )
    try:
        for document_index, positions_cpu in selected_positions.items():
            document = documents[document_index]
            document_ids = document.input_ids if isinstance(document, TextDocument) else document
            input_ids = document_ids.unsqueeze(0).to("cuda")
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
            progress.set_postfix(document=document_index, refresh=False)
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
