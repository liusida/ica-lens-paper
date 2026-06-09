from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .models import torch_dtype


def capture_random_token_hidden_state_shards(
    *,
    texts: Sequence[str],
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    run_name: str,
    model_id: str,
    model_short_name: str,
    dataset_manifest: dict[str, object],
    context_length: int,
    token_budget: int,
    activation_dtype: str,
    shard_token_budget: int,
    seed: int,
    layers: Sequence[str] | None = None,
) -> Path:
    if shard_token_budget <= 0:
        raise ValueError("shard_token_budget must be positive.")

    device = next(model.parameters()).device
    dtype = torch_dtype(activation_dtype)
    capture_dir = output_dir / f"{run_name}_tok{token_budget}"
    all_layer_names = hidden_state_layer_names(int(model.config.num_hidden_layers) + 1)
    selected_layers = list(layers) if layers is not None else all_layer_names
    unknown_layers = sorted(set(selected_layers) - set(all_layer_names))
    if unknown_layers:
        raise ValueError(f"Requested unknown activation layer(s): {', '.join(unknown_layers)}")
    if not selected_layers:
        raise ValueError("At least one activation layer must be selected.")
    selected_layer_set = set(selected_layers)
    explicit_layer_capture = layers is not None
    hook_plan = _hook_capture_plan(model, selected_layers) if explicit_layer_capture else None
    capture_dir.mkdir(parents=True, exist_ok=True)

    tokenized_docs: list[torch.Tensor] = []
    doc_lengths: list[int] = []
    for text in tqdm(texts, dynamic_ncols=True, desc="tokenize docs"):
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_length)
        input_ids = encoded["input_ids"][0].to(dtype=torch.long, device="cpu")
        if int(input_ids.shape[0]) > 0:
            tokenized_docs.append(input_ids)
            doc_lengths.append(int(input_ids.shape[0]))

    if not tokenized_docs:
        raise RuntimeError("No tokenized documents available.")

    selected_by_doc = _sample_positions_by_doc(doc_lengths, token_budget=token_budget, seed=seed)
    selected_docs = sorted(selected_by_doc)
    buffers: dict[str, list[torch.Tensor]] = {}
    input_id_buffer: list[torch.Tensor] = []
    doc_id_buffer: list[torch.Tensor] = []
    position_buffer: list[torch.Tensor] = []
    shards: list[dict[str, object]] = []
    token_count = 0
    shard_index = 0
    shard_tokens = 0
    started_at = time.time()

    pbar = tqdm(total=token_budget, unit="tok", dynamic_ncols=True, desc="capture hidden states")
    try:
        for doc_id in selected_docs:
            positions = selected_by_doc[doc_id]
            input_ids_cpu = tokenized_docs[doc_id]
            input_ids = input_ids_cpu.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            position_tensor = torch.tensor(positions, dtype=torch.long, device=device)

            take = int(position_tensor.shape[0])
            if hook_plan is None:
                with torch.inference_mode():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)

                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("Model did not return hidden states.")

                for key, hidden in zip(hidden_state_layer_names(len(hidden_states)), hidden_states[1:], strict=True):
                    if key not in selected_layer_set:
                        continue
                    selected = hidden[0].index_select(0, position_tensor)
                    buffers.setdefault(key, []).append(selected.detach().to(dtype=dtype).cpu())
            else:
                _capture_layers_with_hooks(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_tensor=position_tensor,
                    dtype=dtype,
                    buffers=buffers,
                    plan=hook_plan,
                )
            input_id_buffer.append(input_ids_cpu.index_select(0, position_tensor.cpu()))
            doc_id_buffer.append(torch.full((take,), doc_id, dtype=torch.long))
            position_buffer.append(position_tensor.cpu())

            token_count += take
            shard_tokens += take
            pbar.update(take)
            pbar.set_postfix(docs=len(selected_docs), shard_tokens=shard_tokens)

            if shard_tokens >= shard_token_budget:
                shard_index = _flush_shard(
                    capture_dir=capture_dir,
                    shard_index=shard_index,
                    buffers=buffers,
                    input_id_buffer=input_id_buffer,
                    doc_id_buffer=doc_id_buffer,
                    position_buffer=position_buffer,
                    shards=shards,
                )
                shard_tokens = 0
    finally:
        pbar.close()

    if token_count != token_budget:
        raise RuntimeError(f"Expected {token_budget} sampled tokens, captured {token_count}.")

    if shard_tokens > 0:
        _flush_shard(
            capture_dir=capture_dir,
            shard_index=shard_index,
            buffers=buffers,
            input_id_buffer=input_id_buffer,
            doc_id_buffer=doc_id_buffer,
            position_buffer=position_buffer,
            shards=shards,
        )

    manifest = {
        "run_name": run_name,
        "model": {
            "id": model_id,
            "short_name": model_short_name,
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "vocab_size": int(model.config.vocab_size),
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "dataset": {
            **dataset_manifest,
            "candidate_documents": len(tokenized_docs),
            "candidate_tokens": int(sum(doc_lengths)),
        },
        "capture": {
            "context_length": context_length,
            "requested_tokens": token_budget,
            "captured_tokens": token_count,
            "documents": len(selected_docs),
            "activation_dtype": activation_dtype,
            "shard_token_budget": shard_token_budget,
            "available_layers": all_layer_names,
            "layers": selected_layers,
            "activation_site": "transformer_block_outputs" if hook_plan is not None else "hidden_states_without_initial_embedding_state",
            "capture_backend": "forward_hooks_early_stop" if hook_plan is not None else "output_hidden_states",
            "embedding_layer_source": "pass --include-embedding to store model.get_input_embeddings().weight as layer 'embedding'",
            "storage_layout": "per_layer_shards",
            "sampling_policy": "random_token_positions_without_exclusion",
            "seed": seed,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "shards": shards,
    }
    manifest_path = capture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class _StopForwardAfterTargetLayer(Exception):
    pass


def _hook_capture_plan(model: torch.nn.Module, selected_layers: Sequence[str]) -> dict[str, Any]:
    blocks = _decoder_blocks(model)
    plan_layers: list[tuple[str, int]] = []
    for layer_name in selected_layers:
        layer_index = _layer_name_to_index(layer_name)
        if layer_index >= len(blocks):
            raise ValueError(f"Requested {layer_name}, but model exposes only {len(blocks)} decoder blocks.")
        plan_layers.append((layer_name, layer_index))
    max_index = max(index for _, index in plan_layers)
    return {"blocks": blocks, "layers": plan_layers, "stop_index": max_index}


def _decoder_blocks(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("backbone", "layers"),
    )
    for first, second in candidates:
        parent = getattr(model, first, None)
        blocks = getattr(parent, second, None) if parent is not None else None
        if blocks is not None and len(blocks) == int(model.config.num_hidden_layers):
            return blocks
    raise RuntimeError(
        "Could not locate decoder blocks for hook capture. "
        "Use all-layer capture, or add this model architecture to _decoder_blocks()."
    )


def _layer_name_to_index(layer_name: str) -> int:
    prefix = "layer_"
    if not layer_name.startswith(prefix):
        raise ValueError(f"Expected layer name like 'layer_30', got {layer_name!r}.")
    try:
        return int(layer_name[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"Expected layer name like 'layer_30', got {layer_name!r}.") from exc


def _capture_layers_with_hooks(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_tensor: torch.Tensor,
    dtype: torch.dtype,
    buffers: dict[str, list[torch.Tensor]],
    plan: dict[str, Any],
) -> None:
    blocks: Sequence[torch.nn.Module] = plan["blocks"]
    stop_index = int(plan["stop_index"])
    layer_by_index = {index: layer_name for layer_name, index in plan["layers"]}
    captured: set[str] = set()
    handles: list[Any] = []

    def make_hook(layer_name: str, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            selected = hidden[0].index_select(0, position_tensor)
            buffers.setdefault(layer_name, []).append(selected.detach().to(dtype=dtype).cpu())
            captured.add(layer_name)
            if layer_index == stop_index:
                raise _StopForwardAfterTargetLayer()

        return hook

    for layer_index, layer_name in layer_by_index.items():
        handles.append(blocks[layer_index].register_forward_hook(make_hook(layer_name, layer_index)))

    try:
        with torch.inference_mode():
            try:
                model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, use_cache=False)
            except _StopForwardAfterTargetLayer:
                pass
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(layer_by_index.values()) - captured)
    if missing:
        raise RuntimeError(f"Hook capture did not see requested layer(s): {', '.join(missing)}")


def hidden_state_layer_names(n_hidden_states: int) -> list[str]:
    if n_hidden_states < 2:
        raise ValueError("n_hidden_states must include the initial state and at least one transformer layer.")
    return [f"layer_{idx:02d}" for idx in range(n_hidden_states - 1)]


def _flush_shard(
    *,
    capture_dir: Path,
    shard_index: int,
    buffers: dict[str, list[torch.Tensor]],
    input_id_buffer: list[torch.Tensor],
    doc_id_buffer: list[torch.Tensor],
    position_buffer: list[torch.Tensor],
    shards: list[dict[str, object]],
) -> int:
    if not input_id_buffer:
        return shard_index

    hidden_states = {key: torch.cat(chunks, dim=0) for key, chunks in buffers.items()}
    input_ids = torch.cat(input_id_buffer, dim=0)
    doc_ids = torch.cat(doc_id_buffer, dim=0)
    positions = torch.cat(position_buffer, dim=0)
    n_tokens = int(input_ids.shape[0])

    shard_name = f"shard_{shard_index:05d}.pt"
    metadata_paths = {
        "input_ids": _save_tensor(capture_dir, "input_ids", shard_name, input_ids),
        "doc_ids": _save_tensor(capture_dir, "doc_ids", shard_name, doc_ids),
        "positions": _save_tensor(capture_dir, "positions", shard_name, positions),
    }
    layer_paths = {key: _save_tensor(capture_dir, key, shard_name, tensor) for key, tensor in hidden_states.items()}
    shards.append(
        {
            "index": shard_index,
            "tokens": n_tokens,
            **metadata_paths,
            "layers": layer_paths,
            "hidden_size": int(next(iter(hidden_states.values())).shape[1]),
        }
    )

    buffers.clear()
    input_id_buffer.clear()
    doc_id_buffer.clear()
    position_buffer.clear()
    return shard_index + 1


def _save_tensor(capture_dir: Path, subdir: str, shard_name: str, tensor: torch.Tensor) -> str:
    output_dir = capture_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_dir / shard_name)
    return f"{subdir}/{shard_name}"


def _sample_positions_by_doc(doc_lengths: list[int], *, token_budget: int, seed: int) -> dict[int, list[int]]:
    total_tokens = int(sum(doc_lengths))
    if token_budget > total_tokens:
        raise ValueError(f"Requested {token_budget} tokens, but only {total_tokens} candidates exist.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampled = torch.randperm(total_tokens, generator=generator)[:token_budget]
    sampled, _ = sampled.sort()
    cumulative = torch.tensor(doc_lengths, dtype=torch.long).cumsum(dim=0)
    doc_indices = torch.bucketize(sampled, cumulative, right=True)
    starts = torch.zeros_like(sampled)
    nonzero = doc_indices > 0
    starts[nonzero] = cumulative[doc_indices[nonzero] - 1]
    positions = sampled - starts

    lengths = torch.tensor(doc_lengths, dtype=torch.long)
    if bool((positions < 0).any()) or bool((positions >= lengths[doc_indices]).any()):
        raise RuntimeError("Sampled an out-of-bounds token position.")

    selected: dict[int, list[int]] = {}
    for doc_id, position in zip(doc_indices.tolist(), positions.tolist(), strict=True):
        selected.setdefault(int(doc_id), []).append(int(position))
    return selected
