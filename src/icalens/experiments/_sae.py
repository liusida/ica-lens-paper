"""Shared public-SAE checkpoint loading and feature encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


@dataclass
class SAEFeatureConfig:
    """Minimal SAE configuration exposed to experiment integrations."""

    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    architecture: str
    dtype: str = "float32"
    device: str = "cuda"


class SAEFeatureEncoder(torch.nn.Module):
    """Load a pinned SAE and reproduce its checkpoint preprocessing and encoder."""

    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    threshold: torch.Tensor | None
    decoder_norms: torch.Tensor

    def __init__(self, snapshot: dict[str, Any], *, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        baseline = snapshot["baselines"]["sae"]
        tensors = _load_sae_tensors(
            Path(baseline["weights_file"]), checkpoint_format=str(baseline["checkpoint_format"])
        )
        hidden_size = int(snapshot["hidden_size"])
        width = int(baseline["width"])
        encoder = _orient_encoder(tensors["W_enc"], hidden_size=hidden_size, width=width)
        decoder = _orient_decoder(tensors["W_dec"], hidden_size=hidden_size, width=width)
        decoder_norms = torch.linalg.vector_norm(decoder, dim=-1).clamp_min(1e-12)
        # Experiment integrations use unit-norm decoder directions. Preserve the
        # checkpoint function by scaling encoded features by the removed norms.
        decoder = decoder / decoder_norms[:, None]
        self.W_enc = torch.nn.Parameter(encoder.to(torch.float32), requires_grad=False)
        self.W_dec = torch.nn.Parameter(decoder.to(torch.float32), requires_grad=False)
        self.b_enc = torch.nn.Parameter(tensors["b_enc"].to(torch.float32), requires_grad=False)
        self.b_dec = torch.nn.Parameter(tensors["b_dec"].to(torch.float32), requires_grad=False)
        self.register_buffer("decoder_norms", decoder_norms.to(torch.float32))
        threshold = tensors.get("threshold")
        if threshold is None:
            self.threshold = None
        else:
            self.threshold = torch.nn.Parameter(threshold.to(torch.float32), requires_grad=False)
        self.dtype = dtype
        self.device = torch.device(device)
        self.activation = str(baseline.get("activation", "relu"))
        self.top_k = int(baseline["top_k"]) if baseline.get("top_k") is not None else None
        self.apply_b_dec_to_input = bool(baseline.get("apply_b_dec_to_input", True))
        self.normalize_activations = str(baseline.get("normalize_activations", "none"))
        if self.normalize_activations not in {"none", "layer_norm"}:
            raise ValueError(
                f"unsupported SAE activation normalization: {self.normalize_activations!r}"
            )
        self.cfg = SAEFeatureConfig(
            model_name=str(snapshot["saebench_model_name"]),
            d_in=hidden_size,
            d_sae=width,
            hook_layer=int(snapshot["layer"]),
            hook_name=str(baseline["hook_name_template"]).format(layer=snapshot["layer"]),
            architecture=f"{self.activation}_sae_checkpoint",
            dtype=str(dtype).removeprefix("torch."),
            device=device,
        )
        self.to(device=self.device, dtype=dtype)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        work, _, _ = self._preprocess(activations)
        if self.apply_b_dec_to_input:
            work = work - self.b_dec
        encoded = work @ self.W_enc + self.b_enc
        if self.activation == "topk":
            if self.top_k is None:
                raise ValueError("TopK SAE is missing top_k")
            values, indices = torch.topk(encoded, k=self.top_k, dim=-1)
            selected = torch.zeros_like(encoded)
            selected.scatter_(-1, indices, torch.relu(values))
            encoded = selected
        elif self.threshold is None:
            encoded = torch.relu(encoded)
        else:
            encoded = torch.where(encoded > self.threshold, encoded, torch.zeros_like(encoded))
        return encoded * self.decoder_norms

    def decode(self, codes: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
        """Decode feature acts and undo checkpoint activation preprocessing."""
        _, mean, std = self._preprocess(reference)
        reconstructed = codes @ self.W_dec + self.b_dec
        if self.normalize_activations == "layer_norm":
            assert mean is not None and std is not None
            reconstructed = reconstructed * std + mean
        return reconstructed

    def _preprocess(
        self, activations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        work = activations.to(device=self.device, dtype=self.dtype)
        if self.normalize_activations == "none":
            return work, None, None
        # Match sae_lens' runtime layer_norm convention exactly: torch.std uses
        # Bessel's correction by default, and epsilon is added after std.
        mean = work.mean(dim=-1, keepdim=True)
        std = (work - mean).std(dim=-1, keepdim=True)
        return (work - mean) / (std + 1e-5), mean, std


def _load_sae_tensors(path: Path, *, checkpoint_format: str) -> dict[str, torch.Tensor]:
    if checkpoint_format == "safetensors":
        return load_file(path, device="cpu")
    if checkpoint_format == "npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    if checkpoint_format == "torch":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in value.items()
        ):
            raise ValueError(f"unsupported torch SAE checkpoint contents in {path}")
        return value
    raise ValueError(f"unsupported SAE checkpoint format: {checkpoint_format!r}")


def _orient_encoder(tensor: torch.Tensor, *, hidden_size: int, width: int) -> torch.Tensor:
    if tuple(tensor.shape) == (hidden_size, width):
        return tensor
    if tuple(tensor.shape) == (width, hidden_size):
        return tensor.T.contiguous()
    raise ValueError(
        f"SAE encoder shape {tuple(tensor.shape)} does not match ({hidden_size}, {width})"
    )


def _orient_decoder(tensor: torch.Tensor, *, hidden_size: int, width: int) -> torch.Tensor:
    if tuple(tensor.shape) == (width, hidden_size):
        return tensor
    if tuple(tensor.shape) == (hidden_size, width):
        return tensor.T.contiguous()
    raise ValueError(
        f"SAE decoder shape {tuple(tensor.shape)} does not match ({width}, {hidden_size})"
    )
