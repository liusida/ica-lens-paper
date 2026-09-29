#!/usr/bin/env python3
"""Capture GPT-2 Layer-0 activations for its non-special vocabulary."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens._capture import transformer_blocks

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
class _StopAfterLastLayer(Exception):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "work/source/output")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    paths = [output / name for name in ("activations.safetensors", "samples.json", "capture.json")]
    if any(path.exists() for path in paths) and not args.force:
        raise FileExistsError(f"output exists under {output}; pass --force to replace it")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    special = set(tokenizer.all_special_ids)
    token_ids = [token_id for token_id in range(len(tokenizer)) if token_id not in special]
    samples = [
        {"index": index, "token_id": token_id, "token": tokenizer.decode([token_id]),
         "role": "background"}
        for index, token_id in enumerate(token_ids)
    ]

    model = load_model_to_cuda(
        AutoModelForCausalLM, MODEL_ID, revision=MODEL_REVISION, device="cuda",
        dtype=torch.bfloat16, low_cpu_mem_usage=True, touch="auto",
    )
    model.eval()
    layer_indices = (0,)
    activations = _capture(
        model, transformer_blocks(model), torch.tensor(token_ids),
        layers=layer_indices, eos_token_id=int(tokenizer.eos_token_id),
        batch_size=args.batch_size,
    )
    output.mkdir(parents=True, exist_ok=True)
    save_file({f"layer_{layer:02d}": value for layer, value in activations.items()}, paths[0])
    paths[1].write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    paths[2].write_text(json.dumps({
        "format": "icalens-toy-example-gpt2-vocab-v1",
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "layers": list(layer_indices), "activation_site": "resid_post",
        "document_framing": {"strategy": "prepend-eos", "token_id": tokenizer.eos_token_id,
                             "included_in_samples": False},
        "counts": {"background": len(samples), "concept": 0, "total": len(samples)},
        "samples_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    for path in paths:
        print(path)


def _capture(
    model: torch.nn.Module, blocks: Sequence[torch.nn.Module], token_ids: torch.Tensor, *,
    layers: tuple[int, ...], eos_token_id: int, batch_size: int,
) -> dict[int, torch.Tensor]:
    chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    def make_hook(layer: int):
        def hook(_: torch.nn.Module, __: tuple[Any, ...], result: Any) -> None:
            hidden = result[0] if isinstance(result, tuple) else result
            chunks[layer].append(hidden[:, 1].detach().float().cpu())
            if layer == layers[-1]:
                raise _StopAfterLastLayer

        return hook

    handles = [blocks[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    try:
        for batch in token_ids.split(batch_size):
            input_ids = torch.stack((torch.full_like(batch, eos_token_id), batch), dim=1).cuda()
            try:
                with torch.inference_mode():
                    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                          use_cache=False)
            except _StopAfterLastLayer:
                pass
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(layer_chunks).contiguous() for layer, layer_chunks in chunks.items()}


if __name__ == "__main__":
    main()
