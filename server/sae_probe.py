from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .probe import _decode_token_text, _layer_key_to_hidden_index


V6_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = V6_ROOT / "configs" / "comparisons"


def list_sae_layers(model_name: str) -> list[str]:
    config = load_sae_config(model_name)
    model_config = load_comparison_config(model_name).get("model", {})
    n_layers = int(model_config.get("num_hidden_layers") or 0)
    layer_checkpoints = config.get("layer_checkpoints")
    if isinstance(layer_checkpoints, dict) and layer_checkpoints:
        indices = sorted(int(key) for key in layer_checkpoints)
    else:
        indices = list(range(n_layers))
    return [f"layer_{idx:02d}" for idx in indices]


def load_comparison_config(model_name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / f"{model_name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"No SAE comparison config for {model_name!r}: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_sae_config(model_name: str) -> dict[str, Any]:
    data = load_comparison_config(model_name)
    if "sae" not in data:
        raise KeyError(f"Comparison config for {model_name!r} has no [sae] section.")
    return dict(data["sae"])


def interpret_text_sae_probe(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    model_name: str,
    layer: str,
    top_k: int,
    max_length: int,
) -> dict[str, Any]:
    if layer == "embedding":
        raise ValueError("SAE Explorer currently supports transformer layers, not embedding.")
    layer_index = int(layer.removeprefix("layer_"))
    hidden_index = _layer_key_to_hidden_index(layer, num_transformer_layers=int(model.config.num_hidden_layers))
    full_ids = tokenizer.encode(text, add_special_tokens=True)
    truncated = len(full_ids) > max_length
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        hidden = hidden_states[hidden_index][0].to(dtype=torch.float32)
        sae = load_sae(model_name, layer_index, hidden_size=int(hidden.shape[-1]), device=hidden.device)
        pre, acts = sae_forward(hidden, sae=sae)
        idx, vals = topk_features(acts, top_k=top_k)

    ids = input_ids[0].detach().cpu().tolist()
    idx_cpu = idx.detach().cpu().tolist()
    vals_cpu = vals.detach().cpu().tolist()
    pre_cpu = pre.detach().cpu()
    tokens = []
    for pos, token_id in enumerate(ids):
        top = []
        for j in range(len(idx_cpu[pos])):
            feature = int(idx_cpu[pos][j])
            top.append(
                {
                    "feature": feature,
                    "activation": float(vals_cpu[pos][j]),
                    "preactivation": float(pre_cpu[pos, feature].item()),
                }
            )
        tokens.append(
            {
                "position": pos,
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "token_text": _decode_token_text(tokenizer, int(token_id)),
                "top": top,
            }
        )
    return {
        "model_name": model_name,
        "layer": layer,
        "top_k": int(top_k),
        "max_length": int(max_length),
        "tokens": tokens,
        "seq_len": len(tokens),
        "truncated": truncated,
        "sae_width": int(sae["w_enc"].shape[1]),
        "activation_rule": sae["activation_rule"],
    }


@lru_cache(maxsize=64)
def _load_sae_cpu(model_name: str, layer: int, hidden_size: int) -> dict[str, Any]:
    config = load_sae_config(model_name)
    path = _resolve_sae_weights_path(config, layer=layer)
    weights = _load_sae_weights(path, config=config)
    w_dec = _decoder_weight(weights, hidden_size=hidden_size, preferred_key=config.get("decoder_key")).to(dtype=torch.float32)
    w_enc = _encoder_weight(weights, hidden_size=hidden_size, d_sae=int(w_dec.shape[0])).to(dtype=torch.float32).contiguous()
    b_enc = _vector_weight(weights, names=("b_enc", "encoder.bias"), length=int(w_enc.shape[1]))
    b_dec = _vector_weight(weights, names=("b_dec", "decoder.bias", "bias"), length=hidden_size)
    threshold = weights.get("threshold")
    return {
        "w_enc": w_enc,
        "b_enc": b_enc.to(dtype=torch.float32).contiguous(),
        "b_dec": b_dec.to(dtype=torch.float32).contiguous(),
        "threshold": threshold.to(dtype=torch.float32).contiguous() if threshold is not None else None,
        "top_k": _sae_top_k(config),
        "activation_rule": _activation_rule_label(config),
    }


def load_sae(model_name: str, layer: int, *, hidden_size: int, device: torch.device) -> dict[str, Any]:
    cpu = _load_sae_cpu(model_name, layer, hidden_size)
    return {
        key: value.to(device=device, dtype=torch.float32, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in cpu.items()
    }


def sae_forward(hidden: torch.Tensor, *, sae: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    centered = hidden.to(dtype=torch.float32) - sae["b_dec"].unsqueeze(0)
    pre = centered @ sae["w_enc"] + sae["b_enc"].unsqueeze(0)
    threshold = sae["threshold"]
    if threshold is not None:
        return pre, torch.where(pre > threshold.unsqueeze(0), pre, torch.zeros_like(pre))
    top_k = sae["top_k"]
    if top_k is not None:
        k = min(int(top_k), int(pre.shape[1]))
        values, indices = torch.topk(pre, k=k, dim=1)
        acts = torch.zeros_like(pre)
        acts.scatter_(1, indices, torch.clamp(values, min=0.0))
        return pre, acts
    return pre, torch.relu(pre)


def topk_features(acts: torch.Tensor, *, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(int(top_k), int(acts.shape[1]))
    if k <= 0:
        raise ValueError("top_k must be positive.")
    vals, idx = torch.topk(acts, k=k, dim=1)
    return idx, vals


def _resolve_sae_weights_path(config: dict[str, Any], *, layer: int) -> str:
    layer_checkpoints = config.get("layer_checkpoints")
    if isinstance(layer_checkpoints, dict) and str(layer) in layer_checkpoints:
        filename = str(layer_checkpoints[str(layer)])
    else:
        filename = str(config["checkpoint_template"]).format(layer=layer)
    return hf_hub_download(repo_id=str(config["repo_id"]), filename=filename, revision=config.get("revision"))


def _load_sae_weights(path: str, *, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    checkpoint_format = str(config.get("checkpoint_format", "")).lower()
    if checkpoint_format == "safetensors":
        return load_file(path, device="cpu")
    if checkpoint_format == "npz":
        with np.load(path) as arrays:
            return {key: torch.from_numpy(arrays[key]) for key in arrays.files}
    if checkpoint_format in {"torch", "pt", "pth"}:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected torch checkpoint {path} to contain a dict, got {type(loaded).__name__}.")
        return _flatten_tensor_dict(loaded)
    raise ValueError(f"Unsupported SAE checkpoint_format {checkpoint_format!r}.")


def _flatten_tensor_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, torch.Tensor]:
    flattened: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, torch.Tensor):
            flattened[name] = value
            flattened[str(key)] = value
        elif isinstance(value, dict):
            flattened.update(_flatten_tensor_dict(value, name))
    return flattened


def _encoder_weight(weights: dict[str, torch.Tensor], *, hidden_size: int, d_sae: int) -> torch.Tensor:
    for key in ("W_enc", "encoder.weight", "W_enc.weight", "encoder.W_enc"):
        tensor = weights.get(key)
        if tensor is not None:
            return _orient_encoder(tensor, hidden_size=hidden_size, d_sae=d_sae)
    return _decoder_weight(weights, hidden_size=hidden_size).T.contiguous()


def _decoder_weight(
    weights: dict[str, torch.Tensor],
    *,
    hidden_size: int,
    preferred_key: object = None,
) -> torch.Tensor:
    if preferred_key is not None:
        key = str(preferred_key)
        if key not in weights:
            raise KeyError(f"Configured decoder_key {key!r} not found. Available keys: {sorted(weights)}")
        return _orient_decoder(weights[key], hidden_size=hidden_size)
    for key in ("W_dec", "decoder.weight", "W_dec.weight", "decoder.W_dec"):
        tensor = weights.get(key)
        if tensor is not None:
            return _orient_decoder(tensor, hidden_size=hidden_size)
    candidates = [tensor for tensor in weights.values() if tensor.ndim == 2 and hidden_size in tensor.shape]
    if len(candidates) == 1:
        return _orient_decoder(candidates[0], hidden_size=hidden_size)
    raise KeyError(f"Could not identify SAE decoder weight. Available keys: {sorted(weights)}")


def _vector_weight(weights: dict[str, torch.Tensor], *, names: tuple[str, ...], length: int) -> torch.Tensor:
    for name in names:
        tensor = weights.get(name)
        if tensor is not None:
            if int(tensor.numel()) != int(length):
                raise ValueError(f"SAE vector {name!r} has length {tensor.numel()}, expected {length}.")
            return tensor.reshape(length).to(dtype=torch.float32)
    return torch.zeros(length, dtype=torch.float32)


def _orient_decoder(tensor: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    if int(tensor.shape[1]) == hidden_size:
        return tensor
    if int(tensor.shape[0]) == hidden_size:
        return tensor.T
    raise ValueError(f"Decoder weight shape {tuple(tensor.shape)} does not contain hidden size {hidden_size}.")


def _orient_encoder(tensor: torch.Tensor, *, hidden_size: int, d_sae: int) -> torch.Tensor:
    if tuple(tensor.shape) == (hidden_size, d_sae):
        return tensor
    if tuple(tensor.shape) == (d_sae, hidden_size):
        return tensor.T
    if int(tensor.shape[0]) == hidden_size:
        return tensor
    if int(tensor.shape[1]) == hidden_size:
        return tensor.T
    raise ValueError(f"Encoder weight shape {tuple(tensor.shape)} does not match hidden={hidden_size}, d_sae={d_sae}.")


def _sae_top_k(config: dict[str, Any]) -> int | None:
    if config.get("top_k") is not None:
        return int(config["top_k"])
    repo = str(config.get("repo_id") or "")
    if "GPT2-Small-OAI-v5-32k-resid-post-SAEs" in repo:
        return 32
    return None


def _activation_rule_label(config: dict[str, Any]) -> str:
    if config.get("top_k") is not None:
        return f"topk-{int(config['top_k'])}"
    if str(config.get("activation") or "").lower() == "topk":
        return "topk"
    return "relu_or_threshold"
