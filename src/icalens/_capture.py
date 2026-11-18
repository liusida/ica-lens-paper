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


RESID_POST = "resid_post"
MLP_OUT = "mlp_out"
HOOKED_SITES = (RESID_POST, MLP_OUT)

# Architectures whose ``block.mlp`` output is exactly what is added to the residual
# stream. Others (e.g. Gemma 2, which applies a post-feedforward norm) are refused
# so that ``mlp_out`` always means "the MLP's write into the residual stream".
MLP_OUT_MODEL_TYPES = frozenset({"gpt2"})


def site_module(model: torch.nn.Module, site: str, layer: int) -> torch.nn.Module:
    """Return the module whose forward output is ``site`` at ``layer``."""
    blocks = transformer_blocks(model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"layer index out of range 0..{len(blocks) - 1}: {layer}")
    if site == RESID_POST:
        return blocks[layer]
    if site == MLP_OUT:
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type not in MLP_OUT_MODEL_TYPES:
            raise ValueError(
                f"activation_site='mlp_out' is verified only for model types "
                f"{sorted(MLP_OUT_MODEL_TYPES)}; got {model_type!r}"
            )
        mlp = getattr(blocks[layer], "mlp", None)
        if not isinstance(mlp, torch.nn.Module):
            raise ValueError(f"transformer block {layer} has no .mlp submodule")
        return mlp
    raise ValueError(f"unsupported hooked activation site {site!r}; expected {HOOKED_SITES}")


def capture_sites(
    model: torch.nn.Module,
    *,
    model_inputs: dict[str, torch.Tensor],
    requests: Sequence[tuple[str, int]],
    positions: torch.Tensor,
) -> dict[tuple[str, int], torch.Tensor]:
    """Capture several (site, layer) outputs in one forward pass at ``positions``.

    The forward pass stops as soon as every requested output has been captured.
    """
    wanted = tuple(dict.fromkeys((str(site), int(layer)) for site, layer in requests))
    if not wanted:
        raise ValueError("at least one (site, layer) must be requested")
    modules = {key: site_module(model, *key) for key in wanted}
    captured: dict[tuple[str, int], torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(key: tuple[str, int]) -> Any:
        def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"{key[0]} at layer {key[1]} did not return a tensor")
            captured[key] = hidden[0].index_select(0, positions.to(hidden.device))
            if len(captured) == len(wanted):
                raise _CaptureComplete

        return hook

    try:
        for key, module in modules.items():
            handles.append(module.register_forward_hook(make_hook(key)))
        with torch.inference_mode():
            try:
                model(**model_inputs, use_cache=False)
            except _CaptureComplete:
                pass
    finally:
        for handle in handles:
            handle.remove()
    missing = [key for key in wanted if key not in captured]
    if missing:
        raise RuntimeError(f"forward hooks did not capture: {missing}")
    return captured


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
