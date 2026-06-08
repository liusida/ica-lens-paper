from __future__ import annotations

from collections.abc import Collection
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


def list_ica_layer_keys(ica_dir: Path) -> list[str]:
    if not ica_dir.is_dir():
        return []
    layers = [path.name[: -len("_fastica.pt")] for path in ica_dir.glob("*_fastica.pt")]
    return sorted(layers, key=_layer_sort_key)


def fastica_artifact_path(ica_dir: Path, layer: str) -> Path:
    return ica_dir / f"{layer}_fastica.pt"


def interpret_text_probe(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    layer: str,
    ica_artifact_path: Path,
    top_k: int,
    highlight_components: Collection[int] | None,
    max_length: int,
    raw_gpt2_block_index: int | None = None,
) -> dict[str, Any]:
    num_transformer_layers = int(model.config.num_hidden_layers)
    hidden_index = _layer_key_to_hidden_index(layer, num_transformer_layers=num_transformer_layers)
    show_predictions = layer == f"layer_{num_transformer_layers - 1:02d}"
    full_ids = tokenizer.encode(text, add_special_tokens=True)
    truncated = len(full_ids) > max_length
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    predictions: list[dict[str, Any]] | None = None
    with torch.inference_mode():
        if layer == "embedding":
            hidden = model.get_input_embeddings()(input_ids)[0]
        elif raw_gpt2_block_index is not None:
            captured: dict[str, torch.Tensor] = {}
            blocks = getattr(getattr(model, "transformer", None), "h", None)
            if blocks is None:
                raise RuntimeError("raw_gpt2_block_index requires a GPT-2 style model.transformer.h module list.")

            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                captured["hidden"] = output[0].detach() if isinstance(output, tuple) else output.detach()

            handle = blocks[int(raw_gpt2_block_index)].register_forward_hook(hook)
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            finally:
                handle.remove()
            captured_hidden = captured.get("hidden")
            if captured_hidden is None:
                raise RuntimeError("GPT-2 raw block hook did not capture hidden states.")
            hidden = captured_hidden[0]
            if show_predictions:
                pred_ids = torch.argmax(outputs.logits[0], dim=-1).detach().cpu().tolist()
                predictions = [
                    {
                        "token_id": int(pred_id),
                        "token": tokenizer.convert_ids_to_tokens(int(pred_id)),
                        "token_text": _decode_token_text(tokenizer, int(pred_id)),
                    }
                    for pred_id in pred_ids
                ]
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            hidden = hidden_states[hidden_index][0]
            if show_predictions:
                pred_ids = torch.argmax(outputs.logits[0], dim=-1).detach().cpu().tolist()
                predictions = [
                    {
                        "token_id": int(pred_id),
                        "token": tokenizer.convert_ids_to_tokens(int(pred_id)),
                        "token_text": _decode_token_text(tokenizer, int(pred_id)),
                    }
                    for pred_id in pred_ids
                ]

    artifact = _load_fastica_artifact(ica_artifact_path)
    scores = _all_source_scores(hidden, **artifact)
    idx, vals = _topk_components_per_token(scores, top_k=top_k)
    forced = sorted({int(component) for component in (highlight_components or [])})
    out_of_range = [component for component in forced if component < 0 or component >= int(scores.shape[1])]
    if out_of_range:
        raise ValueError(f"highlight component out of range: {out_of_range[0]}")

    ids = input_ids[0].detach().cpu().tolist()
    idx_cpu = idx.detach().cpu().tolist()
    vals_cpu = vals.detach().cpu().tolist()
    forced_scores = scores[:, forced].detach().cpu().tolist() if forced else []

    tokens = []
    for pos, token_id in enumerate(ids):
        top = [
            {"component": int(idx_cpu[pos][j]), "score": float(vals_cpu[pos][j])}
            for j in range(len(idx_cpu[pos]))
        ]
        seen = {item["component"] for item in top}
        for j, component in enumerate(forced):
            if component not in seen:
                top.append({"component": component, "score": float(forced_scores[pos][j]), "highlighted": True})
        token_item = {
            "position": pos,
            "token_id": int(token_id),
            "token": tokenizer.convert_ids_to_tokens(int(token_id)),
            "token_text": _decode_token_text(tokenizer, int(token_id)),
            "top": top,
        }
        if predictions is not None:
            token_item["prediction"] = predictions[pos]
        tokens.append(token_item)

    return {
        "layer": layer,
        "top_k": int(top_k),
        "max_length": int(max_length),
        "tokens": tokens,
        "seq_len": len(tokens),
        "truncated": truncated,
        "n_components": int(scores.shape[1]),
        "predictions_available": predictions is not None,
    }


@lru_cache(maxsize=None)
def _load_fastica_artifact(path: Path) -> dict[str, torch.Tensor | float]:
    blob = torch.load(path, map_location="cpu")
    tensors = blob["tensors"]
    meta = blob.get("metadata") or {}
    mean = tensors["mean"].to(torch.float32)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    return {
        "mean": mean,
        "components": tensors["components"].to(torch.float32),
        "norm_eps": float(meta.get("norm_eps", 1e-12)),
    }


def _all_source_scores(
    activations: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    x = activations.to(dtype=torch.float32)
    normalized = x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(norm_eps)
    return (normalized - mean.to(x.device)) @ components.to(x.device).T


def _topk_components_per_token(scores: torch.Tensor, *, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(int(top_k), int(scores.shape[1]))
    if k <= 0:
        raise ValueError("top_k must be positive.")
    idx = torch.topk(scores.abs(), k, dim=1).indices
    row_idx = torch.arange(scores.shape[0], device=scores.device, dtype=torch.long).unsqueeze(1).expand_as(idx)
    return idx, scores[row_idx, idx]


def _layer_key_to_hidden_index(layer: str, *, num_transformer_layers: int) -> int:
    if layer == "embedding":
        return 0
    if not layer.startswith("layer_"):
        raise ValueError(f"Unknown layer key: {layer!r}")
    idx = int(layer.split("_", maxsplit=1)[1])
    if idx < 0 or idx >= num_transformer_layers:
        raise ValueError(f"Layer index out of range: {layer!r}")
    return idx + 1


def _layer_sort_key(layer: str) -> tuple[int, int | str]:
    if layer == "embedding":
        return (0, 0)
    if layer.startswith("layer_"):
        return (1, int(layer.removeprefix("layer_")))
    return (2, layer)


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        decoded = str(tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    except Exception:
        decoded = ""
    raw_token = str(tokenizer.convert_ids_to_tokens(token_id))
    if "\ufffd" in decoded:
        byte_text = _decode_byte_level_token(raw_token)
        if byte_text is not None:
            return byte_text
    return decoded or raw_token


def _decode_byte_level_token(token: str) -> str | None:
    byte_decoder = _gpt2_byte_decoder()
    try:
        raw = bytes(byte_decoder[ch] for ch in token)
    except KeyError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{byte:02X}" for byte in raw)


def _gpt2_byte_decoder() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {chr(char): byte for byte, char in zip(bs, cs, strict=True)}
