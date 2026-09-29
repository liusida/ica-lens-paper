"""Capture reusable, disk-backed model activations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .._activation_dataset import (
    ActivationDataset,
    ActivationDatasetWriter,
    check_disk_space,
    sample_metadata,
)
from .._capture import capture_resid_post
from ._status import log
from .fit_chat import load_chat_documents
from .fit_chat import sample_positions as sample_chat_positions
from .fit_text import (
    load_pile_documents,
    parse_layers,
    parse_token_budget,
    resolve_document_framing,
    resolve_text_dataset,
    set_cuda_memory_limit,
)
from .fit_text import (
    sample_positions as sample_text_positions,
)


def main(kind: str, argv: Sequence[str] | None = None) -> None:
    args = _parse_args(kind, argv)
    if not torch.cuda.is_available():
        raise RuntimeError("'icalens capture' requires a CUDA device")
    if args.max_vram_gb is not None:
        set_cuda_memory_limit(args.max_vram_gb)

    api = HfApi()
    log(f"Resolving model {args.model} and dataset {args.dataset} revisions...")
    model_revision = api.model_info(args.model, revision=args.model_revision).sha
    if model_revision is None:
        raise RuntimeError("Could not resolve the exact model revision.")
    if kind == "text":
        dataset_revision, dataset_provenance = resolve_text_dataset(
            args.dataset, split=args.split, api=api, revision=args.dataset_revision
        )
    else:
        resolved_dataset_revision = api.dataset_info(
            args.dataset, revision=args.dataset_revision
        ).sha
        if resolved_dataset_revision is None:
            raise RuntimeError("Could not resolve the exact dataset revision.")
        dataset_revision = resolved_dataset_revision
        dataset_provenance = {
            "repo_id": args.dataset,
            "revision": str(dataset_revision),
            "split": args.split,
        }
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_revision, use_fast=True)
    config = AutoConfig.from_pretrained(args.model, revision=model_revision, trust_remote_code=True)
    text_config = _text_config(config)
    layers = parse_layers(args.layers, layer_count=int(text_config.num_hidden_layers))
    documents: Any

    if kind == "text":
        framing = resolve_document_framing(
            tokenizer,
            args.document_framing,
            model_id=args.model,
            refresh_registry=args.refresh_model_registry,
        )
        candidates = getattr(args, "candidate_tokens", args.token_budget)
        if _preflight_capture(
            args,
            kind=kind,
            model_revision=str(model_revision),
            dataset_provenance=dataset_provenance,
            model_type="base",
            layers=layers,
            hidden_size=int(text_config.hidden_size),
            candidate_tokens=candidates,
            framing=framing,
        ):
            return
        documents = load_pile_documents(
            tokenizer,
            dataset_id=args.dataset,
            dataset_revision=str(dataset_revision),
            split=args.split,
            text_field=args.text_field,
            candidate_token_budget=candidates,
            context_length=args.context_length,
            document_framing=framing,
        )
        selected = sample_text_positions(documents, token_budget=args.token_budget, seed=args.seed)
        candidate_count = sum(int(doc.candidate_positions.numel()) for doc in documents)
        provenance: dict[str, Any] = {
            "dataset": dataset_provenance,
            "text_field": args.text_field,
            "token_scope": "all",
            "candidate_tokens": candidate_count,
            "fitting_tokens": sum(int(value.numel()) for value in selected.values()),
            "sampling_seed": args.seed,
            "context_length": args.context_length,
            "document_framing": framing,
            "tokenization_runtime": _tokenization_runtime(),
        }
        model_type = "base"
        progress_desc = "Capture text activations"
    else:
        if not tokenizer.is_fast or tokenizer.chat_template is None:
            raise RuntimeError("chat capture requires a fast tokenizer with a chat template")
        candidates = args.token_budget if args.candidate_tokens is None else args.candidate_tokens
        if _preflight_capture(
            args,
            kind=kind,
            model_revision=str(model_revision),
            dataset_provenance=dataset_provenance,
            model_type="instruct",
            layers=layers,
            hidden_size=int(text_config.hidden_size),
            candidate_tokens=candidates,
            framing=None,
        ):
            return
        documents = load_chat_documents(
            tokenizer,
            dataset_id=args.dataset,
            dataset_revision=str(dataset_revision),
            split=args.split,
            messages_field=args.messages_field,
            token_scope=args.token_scope,
            candidate_token_budget=candidates,
            context_length=args.context_length,
        )
        selected = sample_chat_positions(documents, token_budget=args.token_budget, seed=args.seed)
        provenance = {
            "dataset": dataset_provenance,
            "messages_field": args.messages_field,
            "token_scope": args.token_scope,
            "candidate_tokens": candidates,
            "fitting_tokens": args.token_budget,
            "sampling_seed": args.seed,
            "context_length": args.context_length,
            "tokenization_runtime": _tokenization_runtime(),
        }
        model_type = "instruct"
        progress_desc = "Capture chat activations"

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
    writer = ActivationDatasetWriter(
        args.output,
        model={"repo_id": args.model, "revision": str(model_revision), "type": model_type},
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        layers=layers,
        sample_count=sum(int(value.numel()) for value in selected.values()),
        hidden_size=int(model.config.hidden_size),
        dtype=torch.bfloat16,
        provenance=provenance,
        samples=sample_metadata(documents, selected),
    )
    recommended, available = check_disk_space(args.output, required_bytes=writer.required_bytes)
    log(
        f"Activation files need about {recommended / 1024**3:.1f} GiB including margin; "
        f"{available / 1024**3:.1f} GiB is available."
    )
    missing = writer.missing_layers
    log(f"Requested layers: {','.join(map(str, layers))}")
    if not missing:
        writer.finish()
        log(f"Activation dataset is already complete: {writer.path}")
        return
    group_size = (
        len(missing) if args.capture_layers_at_once == "all" else int(args.capture_layers_at_once)
    )
    for start in range(0, len(missing), group_size):
        group = missing[start : start + group_size]
        log(f"Capturing layers {','.join(map(str, group))} directly to disk...")
        with writer.group(group) as sink:
            progress = tqdm(
                total=writer.manifest["sample_count"],
                desc=progress_desc,
                unit="tok",
                dynamic_ncols=True,
            )
            try:
                for document_index, positions_cpu in selected.items():
                    input_ids = documents[document_index].input_ids.unsqueeze(0).to("cuda")
                    captured = capture_resid_post(
                        model,
                        model_inputs={"input_ids": input_ids},
                        layers=group,
                        positions=positions_cpu.to("cuda"),
                    )
                    sink.append(captured)
                    progress.update(int(positions_cpu.numel()))
            finally:
                progress.close()
        for layer in group:
            log(f"Checkpointed activation layer {layer} to {writer.path}")
    writer.finish()
    log(f"Saved reusable activation dataset to {writer.path}")


def _preflight_capture(
    args: argparse.Namespace,
    *,
    kind: str,
    model_revision: str,
    dataset_provenance: dict[str, Any],
    model_type: str,
    layers: tuple[int, ...],
    hidden_size: int,
    candidate_tokens: int | None,
    framing: dict[str, Any] | None,
) -> bool:
    """Validate an existing capture before replaying or loading expensive data."""
    manifest_path = args.output.expanduser().resolve() / "activations.json"
    if not manifest_path.is_file():
        return False
    existing = json.loads(manifest_path.read_text())
    provenance = existing.get("provenance", {})
    expected: dict[str, Any] = {
        "model": {
            "repo_id": args.model,
            "revision": model_revision,
            "type": model_type,
        },
        "activation_site": "resid_post",
        "layer_indexing": "transformer_blocks_zero_based",
        "hidden_size": hidden_size,
        "layers": set(map(str, layers)),
        "dataset": dataset_provenance,
        "sampling_seed": args.seed,
        "context_length": args.context_length,
        "token_scope": "all" if kind == "text" else args.token_scope,
    }
    actual: dict[str, Any] = {
        "model": existing.get("model"),
        "activation_site": existing.get("activation_site"),
        "layer_indexing": existing.get("layer_indexing"),
        "hidden_size": existing.get("hidden_size"),
        "layers": set(existing.get("layers", {})),
        "dataset": provenance.get("dataset"),
        "sampling_seed": provenance.get("sampling_seed"),
        "context_length": provenance.get("context_length"),
        "token_scope": provenance.get("token_scope"),
    }
    if candidate_tokens is not None:
        expected["candidate_tokens"] = candidate_tokens
        actual["candidate_tokens"] = provenance.get("candidate_tokens")
    if args.token_budget is not None:
        expected["fitting_tokens"] = args.token_budget
        actual["fitting_tokens"] = provenance.get("fitting_tokens")
    if kind == "text":
        expected["text_field"] = args.text_field
        expected["document_framing"] = framing
        actual["text_field"] = provenance.get("text_field")
        actual["document_framing"] = provenance.get("document_framing")
    else:
        expected["messages_field"] = args.messages_field
        actual["messages_field"] = provenance.get("messages_field")
    if "tokenization_runtime" in provenance:
        expected["tokenization_runtime"] = _tokenization_runtime()
        actual["tokenization_runtime"] = provenance.get("tokenization_runtime")
    mismatches = [name for name, value in expected.items() if actual.get(name) != value]
    if mismatches:
        raise ValueError(
            "existing activation dataset is incompatible before capture "
            f"({', '.join(mismatches)}): {manifest_path}"
        )
    if existing.get("status") != "complete":
        log(
            f"Validated compatible partial activation dataset at {manifest_path.parent}; "
            "continuing with its missing layers."
        )
        return False
    # This verifies every declared layer file and the token metadata before the
    # completed dataset is accepted as a durable checkpoint.
    completed = ActivationDataset(manifest_path.parent)
    for layer in layers:
        completed.layer(layer)
    completed.samples()
    log(f"Activation dataset is already complete and compatible: {completed.path}")
    return True


def _tokenization_runtime() -> dict[str, str]:
    """Versions that can affect conversion from source text to token positions."""
    return {
        package: importlib.metadata.version(package)
        for package in ("datasets", "tokenizers", "transformers")
    }


def _parse_args(kind: str, argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"icalens capture {kind}", description=__doc__)
    default_model = "openai-community/gpt2" if kind == "text" else "Qwen/Qwen2.5-0.5B-Instruct"
    default_dataset = "NeelNanda/pile-10k" if kind == "text" else "HuggingFaceH4/ultrachat_200k"
    parser.add_argument("--model", default=default_model)
    parser.add_argument(
        "--model-revision",
        help="Exact model revision; primarily used by reproducibility workflows.",
    )
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="Dataset repository, or a local JSONL/Parquet file for text capture.",
    )
    parser.add_argument(
        "--dataset-revision",
        help="Exact Hub dataset revision; primarily used by reproducibility workflows.",
    )
    parser.add_argument("--split", default="train" if kind == "text" else "train_sft")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument(
        "--token-budget", type=parse_token_budget if kind == "text" else int, default=1000
    )
    parser.add_argument(
        "--candidate-tokens",
        type=parse_token_budget if kind == "text" else int,
        default=argparse.SUPPRESS if kind == "text" else None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture-layers-at-once", default="all", type=_capture_group_size)
    parser.add_argument("--max-vram-gb", type=float, default=None)
    if kind == "text":
        parser.add_argument("--text-field", default="text")
        parser.add_argument(
            "--document-framing",
            choices=("auto", "none", "prepend-bos", "prepend-eos"),
            default="auto",
        )
        parser.add_argument("--refresh-model-registry", action="store_true")
    else:
        parser.add_argument("--messages-field", default="messages")
        parser.add_argument(
            "--token-scope", choices=("assistant", "user", "content", "all"), default="all"
        )
    return parser.parse_args(argv)


def _capture_group_size(value: str) -> str | int:
    if value.lower() == "all":
        return "all"
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'")
    return result


def _text_config(config: Any) -> Any:
    """Return the causal-text config from flat or composite model configs."""
    nested = getattr(config, "text_config", None)
    candidate = config if nested is None else nested
    missing = [
        name
        for name in ("num_hidden_layers", "hidden_size")
        if getattr(candidate, name, None) is None
    ]
    if missing:
        raise ValueError(f"model text config is missing required fields: {', '.join(missing)}")
    return candidate
