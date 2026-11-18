"""Capture GPT-2 MLP outputs (``mlp_out``) into an ICA Lens ActivationDataset.

Standalone experiment script. It mirrors ``icalens capture text`` exactly
(same tokenization, document framing, uniform position sampling and seed), so
the sampled tokens match a resid_post capture with the same arguments. The only
difference is the hook site: ``transformer.h[l].mlp`` output, i.e.
``c_proj(gelu(c_fc(ln_2(x))))`` after dropout (identity in eval), before the
residual add.

Fit the result with the unchanged library CLI:

    icalens fit activations --input <output> --layers all ...
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens._activation_dataset import (
    ActivationDatasetWriter,
    check_disk_space,
    sample_metadata,
)
from icalens._capture import transformer_blocks
from icalens.cli.fit_text import (
    load_pile_documents,
    parse_layers,
    parse_token_budget,
    resolve_document_framing,
    resolve_text_dataset,
    sample_positions,
)

ACTIVATION_SITE = "mlp_out"


class _CaptureComplete(Exception):
    pass


def mlp_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    blocks = transformer_blocks(model)
    mlps = [getattr(block, "mlp", None) for block in blocks]
    if any(mlp is None for mlp in mlps):
        raise ValueError(f"{type(model).__name__}: not every block has an .mlp submodule")
    return mlps  # type: ignore[return-value]


def capture_mlp_out(
    model: torch.nn.Module,
    mlps: list[torch.nn.Module],
    *,
    input_ids: torch.Tensor,
    layers: tuple[int, ...],
    positions: torch.Tensor,
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    final_layer = max(layers)

    def make_hook(layer: int) -> Any:
        def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[0].index_select(0, positions)
            if layer == final_layer:
                raise _CaptureComplete

        return hook

    handles = [mlps[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    try:
        with torch.inference_mode():
            try:
                model(input_ids=input_ids, use_cache=False)
            except _CaptureComplete:
                pass
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise RuntimeError(f"MLP hooks did not capture layers: {missing}")
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--model-revision", default=None, help="Exact model commit to pin.")
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--dataset-revision", default=None, help="Exact dataset commit to pin.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--token-budget", type=parse_token_budget, default=1000)
    parser.add_argument("--candidate-tokens", type=parse_token_budget, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--document-framing",
        choices=("auto", "none", "prepend-bos", "prepend-eos"),
        default="auto",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    api = HfApi()
    model_revision = api.model_info(args.model, revision=args.model_revision).sha
    if model_revision is None:
        raise RuntimeError("Could not resolve the exact model revision.")
    dataset_revision, dataset_provenance = resolve_text_dataset(
        args.dataset, split=args.split, api=api, revision=args.dataset_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_revision, use_fast=True)
    framing = resolve_document_framing(tokenizer, args.document_framing, model_id=args.model)
    candidates = getattr(args, "candidate_tokens", args.token_budget)
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
    selected = sample_positions(documents, token_budget=args.token_budget, seed=args.seed)
    sample_count = sum(int(value.numel()) for value in selected.values())
    provenance: dict[str, Any] = {
        "dataset": dataset_provenance,
        "text_field": args.text_field,
        "token_scope": "all",
        "candidate_tokens": sum(int(doc.candidate_positions.numel()) for doc in documents),
        "fitting_tokens": sample_count,
        "sampling_seed": args.seed,
        "context_length": args.context_length,
        "document_framing": framing,
        "tokenization_runtime": {
            package: importlib.metadata.version(package)
            for package in ("datasets", "tokenizers", "transformers")
        },
        "capture_script": "experiments/mlp-out-ica/capture_mlp_out.py",
        "activation_site_definition": (
            "output of transformer block .mlp submodule (after c_proj and dropout, "
            "before the residual add)"
        ),
    }

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
    mlps = mlp_modules(model)
    layers = parse_layers(args.layers, layer_count=len(mlps))

    writer = ActivationDatasetWriter(
        args.output,
        model={"repo_id": args.model, "revision": str(model_revision), "type": "base"},
        activation_site=ACTIVATION_SITE,
        layer_indexing="transformer_blocks_zero_based",
        layers=layers,
        sample_count=sample_count,
        hidden_size=int(model.config.hidden_size),
        dtype=torch.bfloat16,
        provenance=provenance,
        samples=sample_metadata(documents, selected),
    )
    check_disk_space(args.output, required_bytes=writer.required_bytes)
    missing = writer.missing_layers
    if missing:
        with writer.group(missing) as sink:
            for document_index, positions in tqdm(
                selected.items(), desc="Capture mlp_out", unit="doc", dynamic_ncols=True
            ):
                input_ids = documents[document_index].input_ids.unsqueeze(0).to("cuda")
                sink.append(
                    capture_mlp_out(
                        model,
                        mlps,
                        input_ids=input_ids,
                        layers=missing,
                        positions=positions.to("cuda"),
                    )
                )
    writer.finish()
    print(f"Saved {ACTIVATION_SITE} activation dataset to {writer.path}")


if __name__ == "__main__":
    main()
