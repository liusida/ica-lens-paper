"""Internal transformer-block activation capture helpers."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, cast

import torch


class _CaptureComplete(Exception):
    """Stop a model forward after the last requested block has run."""


def transformer_blocks(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """Return the ordered transformer blocks for supported causal LMs."""
    candidates = (
        ("transformer", "h"),  # GPT-2
        ("model", "layers"),  # Llama, Mistral, text-only Qwen
        ("model", "language_model", "layers"),  # Qwen3.5 conditional generation
        ("language_model", "layers"),  # Other conditional-generation wrappers
        ("gpt_neox", "layers"),  # GPT-NeoX/Pythia
    )
    for path in candidates:
        blocks: Any = model
        for name in path:
            blocks = getattr(blocks, name, None)
            if blocks is None:
                break
        if isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)):
            return cast(Sequence[torch.nn.Module], blocks)
    raise ValueError(
        f"unsupported model architecture {type(model).__name__}: "
        "could not locate transformer blocks"
    )


def capture_resid_post(
    model: torch.nn.Module,
    *,
    model_inputs: dict[str, torch.Tensor],
    layers: tuple[int, ...],
    positions: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Capture transformer-block outputs before any model-level final norm."""
    blocks = transformer_blocks(model)
    if not layers:
        raise ValueError("at least one layer must be requested")
    invalid = [layer for layer in layers if layer < 0 or layer >= len(blocks)]
    if invalid:
        raise ValueError(f"layer indices out of range 0..{len(blocks) - 1}: {invalid}")
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    final_layer = max(layers)

    def make_hook(layer: int) -> Any:
        def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"transformer block {layer} did not return a tensor")
            captured[layer] = hidden[0].index_select(0, positions.to(hidden.device))
            if layer == final_layer:
                raise _CaptureComplete

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        with torch.inference_mode():
            try:
                model(**model_inputs, use_cache=False)
            except _CaptureComplete:
                pass
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise RuntimeError(f"forward hooks did not capture layers: {missing}")
    return captured


@contextmanager
def clamp_resid_post(
    model: torch.nn.Module,
    *,
    layer: int,
    edit: Any,
) -> Any:
    """Apply an activation edit to one block output for the context lifetime."""
    blocks = transformer_blocks(model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"layer index out of range 0..{len(blocks) - 1}: {layer}")

    def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"transformer block {layer} did not return a tensor")
        edited = edit(hidden)
        if not isinstance(edited, torch.Tensor) or edited.shape != hidden.shape:
            raise ValueError("resid_post edit must return a tensor with the original shape")
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
