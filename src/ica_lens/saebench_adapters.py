from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PREPROCESS_ALIASES = {
    "with_normalization": "with_normalization",
    "row_normalize_then_center_whiten": "with_normalization",
    "without_normalization": "without_normalization",
    "raw_center_whiten": "without_normalization",
}


def canonical_ica_preprocess(value: str | None) -> str:
    if value is None or value == "":
        return "with_normalization"
    key = str(value)
    if key not in _PREPROCESS_ALIASES:
        raise ValueError(f"Unsupported ICA preprocess mode: {value!r}")
    return _PREPROCESS_ALIASES[key]


def load_fastica_sae(
    *,
    artifact_prefix: Path,
    model_name: str,
    hook_layer: int,
    hook_name: str,
    device: str,
    dtype: Any,
    sign_mode: str,
    dominant_signs: list[int] | None = None,
    expected_preprocess: str | None = None,
):
    import torch
    import torch.nn as nn

    try:
        from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
    except ModuleNotFoundError:
        class CustomSAEConfig:  # type: ignore[no-redef]
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

    class FastICASAELike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            saved = torch.load(artifact_prefix.with_suffix(".pt"), map_location="cpu")
            metadata_path = artifact_prefix.with_suffix(".json")
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            self.preprocess = canonical_ica_preprocess(self.metadata.get("preprocess"))
            expected = canonical_ica_preprocess(expected_preprocess)
            if self.preprocess != expected:
                raise ValueError(
                    f"ICA preprocess mismatch for {artifact_prefix}: config expects {expected!r}, "
                    f"artifact metadata has {self.metadata.get('preprocess')!r} -> {self.preprocess!r}."
                )
            components = saved["tensors"]["components"].to(dtype=torch.float32)
            mean = saved["tensors"]["mean"].to(dtype=torch.float32)
            if mean.dim() == 1:
                mean = mean.reshape(1, -1)
            n_components, d_in = components.shape
            norm_eps = float(self.metadata.get("norm_eps", 1e-12))
            mixing = torch.linalg.pinv(components)
            score_decode_directions = mixing.T.contiguous()
            if sign_mode == "two_sign":
                w_dec = torch.cat([score_decode_directions, -score_decode_directions], dim=0)
                signs = None
            elif sign_mode == "dominant_sign":
                if dominant_signs is None or len(dominant_signs) != int(n_components):
                    raise ValueError(
                        f"dominant_sign mode requires {n_components} signs, got "
                        f"{0 if dominant_signs is None else len(dominant_signs)}"
                    )
                signs = torch.tensor(dominant_signs, dtype=torch.float32).reshape(-1, 1)
                w_dec = score_decode_directions * signs
            else:
                raise ValueError(f"Unsupported ICA sign_mode {sign_mode!r}")
            feature_scale = w_dec.norm(dim=1).clamp_min(norm_eps)
            w_dec = w_dec / feature_scale[:, None]
            self.W_enc = nn.Parameter(components.T.contiguous())
            self.W_dec = nn.Parameter(w_dec.contiguous())
            self.b_enc = nn.Parameter(torch.zeros(w_dec.shape[0], dtype=torch.float32))
            self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=torch.float32))
            self.register_buffer("components", components.contiguous())
            self.register_buffer("mean", mean.contiguous())
            self.register_buffer("mixing", mixing.contiguous())
            self.register_buffer("feature_scale", feature_scale.reshape(1, -1).contiguous())
            if signs is not None:
                self.register_buffer("dominant_signs", signs.reshape(1, -1).contiguous())
            else:
                self.dominant_signs = None
            self.device = torch.device(device)
            self.dtype = dtype
            self.sign_mode = sign_mode
            self.norm_eps = norm_eps
            self._last_input_norm = None
            self._last_nonzero_mask = None
            self._last_leading_shape = None
            self.cfg = CustomSAEConfig(model_name=model_name, d_in=int(d_in), d_sae=int(w_dec.shape[0]), hook_layer=int(hook_layer), hook_name=hook_name)
            self.cfg.architecture = "fastica_inverse_sae_like"
            self.cfg.activation_fn_str = "relu"
            self.cfg.dtype = dtype.__str__().split(".")[1]
            self.cfg.device = str(self.device)
            self.cfg.fastica_artifact_prefix = str(artifact_prefix)
            self.cfg.fastica_n_components = int(n_components)
            self.cfg.fastica_sign_mode = str(sign_mode)
            self.cfg.fastica_signed_relu_pairs = bool(sign_mode == "two_sign")
            self.cfg.fastica_adaptor = "inverse"
            self.cfg.fastica_preprocess = self.preprocess
            self.cfg.fastica_decode = (
                "pinv_components_add_mean_restore_input_norm"
                if self.preprocess == "with_normalization"
                else "pinv_components_add_mean"
            )
            self.cfg.fastica_mixing_source = "pinv_components"
            self.cfg.fastica_row_normalize = bool(self.preprocess == "with_normalization")
            self.cfg.fastica_feature_scaling = (
                "decoder_norm_times_input_norm"
                if self.preprocess == "with_normalization"
                else "decoder_norm"
            )
            self.to(device=self.device, dtype=self.dtype)

        def encode(self, x):
            x = x.to(device=self.components.device, dtype=self.components.dtype)
            self._last_leading_shape = tuple(x.shape[:-1])
            feature_scale = self.feature_scale.to(dtype=x.dtype, device=x.device)
            if self.preprocess == "with_normalization":
                norm = x.norm(dim=-1, keepdim=True)
                nonzero_mask = norm > self.norm_eps
                norm_clamped = norm.clamp_min(self.norm_eps)
                x_score = x / norm_clamped
                scores = (x_score - self.mean) @ self.components.T
                scores = torch.where(nonzero_mask, scores, torch.zeros_like(scores))
                self._last_input_norm = norm_clamped.detach()
                self._last_nonzero_mask = nonzero_mask.detach()
                input_scale = norm_clamped.detach().to(dtype=scores.dtype, device=scores.device)
            elif self.preprocess == "without_normalization":
                scores = (x - self.mean) @ self.components.T
                self._last_input_norm = None
                self._last_nonzero_mask = None
                input_scale = None
            else:
                raise ValueError(f"Unsupported ICA preprocess mode: {self.preprocess!r}")

            if self.sign_mode == "two_sign":
                features = torch.cat([torch.relu(scores), torch.relu(-scores)], dim=-1)
            elif self.sign_mode == "dominant_sign":
                if self.dominant_signs is None:
                    raise RuntimeError("dominant_sign mode is missing dominant_signs buffer")
                features = torch.relu(scores * self.dominant_signs.to(dtype=scores.dtype, device=scores.device))
            else:
                raise ValueError(f"Unsupported ICA sign_mode {self.sign_mode!r}")
            features = features * feature_scale
            if input_scale is not None:
                features = features * input_scale
            return features

        def decode(self, feature_acts):
            if self._last_leading_shape is None:
                raise RuntimeError("FastICA inverse decode requires encode(x) to be called immediately before decode(features).")
            feature_acts = feature_acts.to(device=self.components.device, dtype=self.components.dtype)
            if tuple(feature_acts.shape[:-1]) != self._last_leading_shape:
                raise RuntimeError(
                    "FastICA inverse decode got feature activations with leading shape "
                    f"{tuple(feature_acts.shape[:-1])}, but cached input shape is {self._last_leading_shape}."
                )
            n_components = int(self.components.shape[0])
            feature_scale = self.feature_scale.to(dtype=feature_acts.dtype, device=feature_acts.device)
            unscaled_feature_acts = feature_acts / feature_scale
            if self.preprocess == "with_normalization":
                if self._last_input_norm is None or self._last_nonzero_mask is None:
                    raise RuntimeError("Missing cached input norm for with_normalization decode.")
                input_norm = self._last_input_norm.to(dtype=feature_acts.dtype, device=feature_acts.device)
                unscaled_feature_acts = unscaled_feature_acts / input_norm
            elif self.preprocess != "without_normalization":
                raise ValueError(f"Unsupported ICA preprocess mode: {self.preprocess!r}")

            if self.sign_mode == "two_sign":
                if int(feature_acts.shape[-1]) != 2 * n_components:
                    raise RuntimeError(f"Expected {2 * n_components} two-sign features, got {feature_acts.shape[-1]}.")
                pos = unscaled_feature_acts[..., :n_components]
                neg = unscaled_feature_acts[..., n_components:]
                scores = pos - neg
            elif self.sign_mode == "dominant_sign":
                if int(feature_acts.shape[-1]) != n_components:
                    raise RuntimeError(f"Expected {n_components} dominant-sign features, got {feature_acts.shape[-1]}.")
                if self.dominant_signs is None:
                    raise RuntimeError("dominant_sign mode is missing dominant_signs buffer")
                scores = unscaled_feature_acts * self.dominant_signs.to(dtype=feature_acts.dtype, device=feature_acts.device)
            else:
                raise ValueError(f"Unsupported ICA sign_mode {self.sign_mode!r}")

            x_hat = scores @ self.mixing.T + self.mean
            if self.preprocess == "with_normalization":
                x_hat = x_hat * self._last_input_norm.to(dtype=x_hat.dtype, device=x_hat.device)
                return torch.where(self._last_nonzero_mask.to(device=x_hat.device), x_hat, torch.zeros_like(x_hat))
            return x_hat

        def forward(self, x):
            return self.decode(self.encode(x))

        def to(self, *args: Any, **kwargs: Any):
            super().to(*args, **kwargs)
            if "device" in kwargs and kwargs["device"] is not None:
                self.device = torch.device(kwargs["device"])
                self.cfg.device = str(self.device)
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                self.dtype = kwargs["dtype"]
                self.cfg.dtype = self.dtype.__str__().split(".")[1]
            return self

    return FastICASAELike()


def selected_baseline_saes(config: dict[str, Any], tpp_config: dict[str, Any], layer_index: int, *, device: str) -> list[tuple[str, Any]]:
    from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

    baseline = tpp_config.get("sae_baseline")
    if not isinstance(baseline, dict):
        raise ValueError("Missing [saebench_tpp.sae_baseline] config for --methods sae_baseline.")
    source = str(baseline.get("source", "sae_lens_registry"))
    if source == "custom_checkpoint":
        sae = load_checkpoint_sae(
            sae_config=config["sae"],
            model_name=str(tpp_config["saebench_model_name"]),
            hook_layer=layer_index,
            hook_name=str(baseline.get("hook_name_template", tpp_config["hook_name_template"])).format(layer=layer_index),
            hidden_size=int(config["model"]["hidden_size"]),
            device=device,
            dtype=str_to_dtype(str(tpp_config["llm_dtype"])),
        )
        release = str(baseline.get("release_name", f"custom_sae_baseline_layer_{layer_index}")).format(layer=layer_index)
        return [(release, sae)]
    if source != "sae_lens_registry":
        raise ValueError(f"Unsupported SAE baseline source {source!r}.")
    release_pattern = str(baseline["release_pattern"])
    id_pattern = _baseline_id_pattern(baseline, config, layer_index)
    selected = get_saes_from_regex(release_pattern, id_pattern)
    if not selected:
        raise ValueError(
            "No SAE baseline matched "
            f"release_pattern={release_pattern!r}, id_pattern={id_pattern!r}. "
            "Use SAEBench's print_all_sae_releases()/print_release_details() to find the loadable SAE Lens names."
        )
    if len(selected) > 1 and not bool(baseline.get("allow_multiple", False)):
        raise ValueError(
            f"SAE baseline matched {len(selected)} SAEs for layer {layer_index}: {selected}. "
            "Narrow id_pattern_template or set allow_multiple=true."
        )
    return selected


def load_checkpoint_sae(
    *,
    sae_config: dict[str, Any],
    model_name: str,
    hook_layer: int,
    hook_name: str,
    hidden_size: int,
    device: str,
    dtype: Any,
):
    import torch
    import torch.nn as nn

    try:
        from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
    except ModuleNotFoundError:
        class CustomSAEConfig:  # type: ignore[no-redef]
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

    weights_path = _resolve_sae_weights_path(sae_config=sae_config, layer=hook_layer)
    weights = _load_sae_weights(weights_path, sae_config=sae_config)
    w_dec = _decoder_weight(weights, hidden_size=hidden_size, preferred_key=sae_config.get("decoder_key")).to(torch.float32)
    w_enc = _encoder_weight(weights, hidden_size=hidden_size, d_sae=int(w_dec.shape[0])).to(torch.float32)
    b_enc = _vector_weight(weights, names=("b_enc", "encoder.bias"), length=int(w_dec.shape[0]))
    b_dec = _vector_weight(weights, names=("b_dec", "decoder.bias"), length=hidden_size)
    decoder_norms = w_dec.norm(dim=1).clamp_min(1e-12)
    w_dec = w_dec / decoder_norms[:, None]
    activation = str(sae_config.get("activation", "relu"))
    top_k = sae_config.get("top_k")

    class CheckpointSAELike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W_enc = nn.Parameter(w_enc.contiguous())
            self.W_dec = nn.Parameter(w_dec.contiguous())
            self.b_enc = nn.Parameter(b_enc.contiguous())
            self.b_dec = nn.Parameter(b_dec.contiguous())
            self.register_buffer("decoder_norms", decoder_norms.reshape(1, -1).contiguous())
            self.device = torch.device(device)
            self.dtype = dtype
            self.activation = activation
            self.top_k = int(top_k) if top_k is not None else None
            self.cfg = CustomSAEConfig(model_name=model_name, d_in=hidden_size, d_sae=int(w_dec.shape[0]), hook_layer=int(hook_layer), hook_name=hook_name)
            self.cfg.architecture = "checkpoint_sae_like_post_activation_feature_scaling"
            self.cfg.activation_fn_str = activation
            self.cfg.dtype = dtype.__str__().split(".")[1]
            self.cfg.device = str(self.device)
            self.cfg.checkpoint_path = str(weights_path)
            self.cfg.checkpoint_format = str(sae_config.get("checkpoint_format", ""))
            self.cfg.top_k = self.top_k
            self.cfg.checkpoint_decoder_norm_handling = "post_activation_feature_scaling"
            self.to(device=self.device, dtype=self.dtype)

        def encode(self, x):
            acts = x.to(dtype=self.W_enc.dtype) @ self.W_enc + self.b_enc
            if self.activation == "topk":
                acts = torch.relu(acts)
                if self.top_k is not None and self.top_k < acts.shape[-1]:
                    values, indices = torch.topk(acts, k=self.top_k, dim=-1)
                    filtered = torch.zeros_like(acts)
                    filtered.scatter_(-1, indices, values)
                    acts = filtered
            elif self.activation == "identity":
                pass
            else:
                acts = torch.relu(acts)
            return acts * self.decoder_norms.to(dtype=acts.dtype, device=acts.device)

        def decode(self, feature_acts):
            return feature_acts.to(dtype=self.W_dec.dtype) @ self.W_dec + self.b_dec

        def forward(self, x):
            return self.decode(self.encode(x))

        def to(self, *args: Any, **kwargs: Any):
            super().to(*args, **kwargs)
            if "device" in kwargs and kwargs["device"] is not None:
                self.device = torch.device(kwargs["device"])
                self.cfg.device = str(self.device)
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                self.dtype = kwargs["dtype"]
                self.cfg.dtype = self.dtype.__str__().split(".")[1]
            return self

    return CheckpointSAELike()


def str_to_dtype(value: str):
    import torch

    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if value not in dtypes:
        raise ValueError(f"Unsupported dtype {value!r}; expected one of {sorted(dtypes)}")
    return dtypes[value]


def _baseline_id_pattern(baseline: dict[str, Any], config: dict[str, Any], layer_index: int) -> str:
    if bool(baseline.get("id_from_sae_layer_checkpoints", False)):
        layer_checkpoints = config.get("sae", {}).get("layer_checkpoints")
        if not isinstance(layer_checkpoints, dict) or str(layer_index) not in layer_checkpoints:
            raise ValueError(f"Missing sae.layer_checkpoints entry for layer {layer_index}.")
        checkpoint = str(layer_checkpoints[str(layer_index)])
        return checkpoint.removesuffix("/params.npz").removesuffix(".npz")
    return str(baseline["id_pattern_template"]).format(layer=layer_index)


def _resolve_sae_weights_path(*, sae_config: dict[str, Any], layer: int) -> str:
    from huggingface_hub import hf_hub_download

    if "local_checkpoint_template" in sae_config:
        path = Path(str(sae_config["local_checkpoint_template"]).format(layer=layer))
        return str(path.resolve())
    layer_checkpoints = sae_config.get("layer_checkpoints")
    if isinstance(layer_checkpoints, dict) and str(layer) in layer_checkpoints:
        filename = str(layer_checkpoints[str(layer)])
    else:
        filename = str(sae_config["checkpoint_template"]).format(layer=layer)
    return hf_hub_download(repo_id=str(sae_config["repo_id"]), filename=filename, revision=sae_config.get("revision"))


def _load_sae_weights(path: str, *, sae_config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from safetensors.torch import load_file

    checkpoint_format = str(sae_config.get("checkpoint_format", "")).lower()
    if not checkpoint_format:
        suffix = Path(path).suffix.lower()
        checkpoint_format = {".safetensors": "safetensors", ".npz": "npz", ".pt": "torch", ".pth": "torch"}.get(suffix, "")
    if checkpoint_format == "safetensors":
        return dict(load_file(path, device="cpu"))
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
    raise ValueError(f"Could not determine SAE checkpoint format for {path}.")


def _flatten_tensor_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    import torch

    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, torch.Tensor):
            flattened[name] = value
            flattened[str(key)] = value
        elif isinstance(value, dict):
            flattened.update(_flatten_tensor_dict(value, name))
    return flattened


def _decoder_weight(weights: dict[str, Any], *, hidden_size: int, preferred_key: object = None):
    if preferred_key is not None:
        key = str(preferred_key)
        if key not in weights:
            raise KeyError(f"Configured decoder_key {key!r} not found. Available keys: {sorted(weights)}")
        return _orient_decoder(weights[key], hidden_size=hidden_size)
    for key in ("W_dec", "decoder.weight", "W_dec.weight", "decoder.W_dec"):
        tensor = weights.get(key)
        if tensor is not None:
            return _orient_decoder(tensor, hidden_size=hidden_size)
    candidates = [tensor for tensor in weights.values() if getattr(tensor, "ndim", None) == 2 and hidden_size in tensor.shape]
    if len(candidates) == 1:
        return _orient_decoder(candidates[0], hidden_size=hidden_size)
    raise KeyError(f"Could not identify SAE decoder weight. Available keys: {sorted(weights)}")


def _encoder_weight(weights: dict[str, Any], *, hidden_size: int, d_sae: int):
    for key in ("W_enc", "encoder.weight", "W_enc.weight", "encoder.W_enc"):
        tensor = weights.get(key)
        if tensor is not None:
            return _orient_encoder(tensor, hidden_size=hidden_size, d_sae=d_sae)
    return _decoder_weight(weights, hidden_size=hidden_size).T.contiguous()


def _orient_decoder(tensor: Any, *, hidden_size: int):
    if int(tensor.shape[1]) == hidden_size:
        return tensor
    if int(tensor.shape[0]) == hidden_size:
        return tensor.T
    raise ValueError(f"Decoder weight shape {tuple(tensor.shape)} does not contain hidden size {hidden_size}.")


def _orient_encoder(tensor: Any, *, hidden_size: int, d_sae: int):
    if tuple(tensor.shape) == (hidden_size, d_sae):
        return tensor
    if tuple(tensor.shape) == (d_sae, hidden_size):
        return tensor.T
    if int(tensor.shape[0]) == hidden_size:
        return tensor
    if int(tensor.shape[1]) == hidden_size:
        return tensor.T
    raise ValueError(f"Encoder weight shape {tuple(tensor.shape)} does not match hidden={hidden_size}, d_sae={d_sae}.")


def _vector_weight(weights: dict[str, Any], *, names: tuple[str, ...], length: int):
    import torch

    for name in names:
        tensor = weights.get(name)
        if tensor is not None and getattr(tensor, "ndim", None) == 1 and int(tensor.shape[0]) == length:
            return tensor.to(torch.float32)
    return torch.zeros(length, dtype=torch.float32)
