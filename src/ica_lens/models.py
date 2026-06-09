from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_toml


def load_model_config(path: Path) -> dict:
    return load_toml(path)


def torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype {name!r}") from exc


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token


def load_model_and_tokenizer(model_id: str, *, device: str, dtype: str) -> tuple[torch.nn.Module, Any]:
    torch_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ensure_pad_token(tokenizer)

    model_dtype = torch_dtype(dtype)
    if torch_device.type == "cpu" and model_dtype == torch.float16:
        model_dtype = torch.float32

    load_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    if torch_device.type == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
                device_map={"": str(torch_device)},
            )
        except (ImportError, TypeError, ValueError) as exc:
            print(
                f"Direct device_map load failed for {model_id!r}; falling back to model.to({torch_device}). "
                f"This can be slow for large models. Reason: {exc}",
                flush=True,
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model.to(torch_device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.to(torch_device)
    model.eval()
    return model, tokenizer
