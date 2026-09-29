"""Replay the shared SAE adapter against accepted token-level activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from icalens import ICALens
from icalens.analysis import _resolve_model_and_tokenizer, capture
from icalens.experiments._sae import SAEFeatureEncoder
from icalens.experiments.saebench_sparse_probing import (
    _prepare_layer_baselines,
    _resolve_baselines,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_LABELS = {"gpt2": "gpt2", "gemma": "gemma2", "qwen": "qwen9b"}


def _checkpoint_tensors(path: Path, checkpoint_format: str) -> dict[str, torch.Tensor]:
    """Load checkpoint tensors without using the production SAE adapter helpers."""
    if checkpoint_format == "safetensors":
        return load_file(path, device="cpu")
    if checkpoint_format == "npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    if checkpoint_format == "torch":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict):
            raise ValueError(f"expected tensor dictionary in {path}")
        return value
    raise ValueError(f"unsupported checkpoint format {checkpoint_format!r}")


def _decoder_rows(tensor: torch.Tensor, hidden_size: int, width: int) -> torch.Tensor:
    if tuple(tensor.shape) == (width, hidden_size):
        return tensor.float()
    if tuple(tensor.shape) == (hidden_size, width):
        return tensor.T.contiguous().float()
    raise ValueError(f"unexpected decoder shape {tuple(tensor.shape)}")


def run(
    *,
    lens_root: Path,
    layer: int,
    rtol: float,
    atol: float,
    output: Path,
    canary_id: str,
) -> dict[str, Any]:
    """Re-encode one accepted sentence and compare its stored SAE response."""
    if not torch.cuda.is_available():
        raise RuntimeError("C13 SAE-adapter replay requires CUDA")

    result_root = ROOT / "experiments/activation-pattern/results"
    label = RESULT_LABELS[canary_id]
    reference_path = result_root / label / f"layer_{layer:02d}.npz"
    run_state = json.loads((result_root / "run.json").read_text(encoding="utf-8"))
    sentence = str(run_state["resolved"]["sentence"])
    with np.load(reference_path, allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}

    lens = ICALens.from_pretrained(lens_root)
    model = tokenizer = None
    try:
        model, tokenizer = _resolve_model_and_tokenizer(lens, None, None, "cuda")
        captured = capture(
            lens,
            sentence,
            layer=layer,
            model=model,
            tokenizer=tokenizer,
            device="cuda",
        )
        ica = lens.transform(captured.activations.float(), layer=layer)
        if not isinstance(ica, torch.Tensor):
            ica = torch.as_tensor(ica)
        ica = ica.detach().cpu().float().numpy()
        baseline = _resolve_baselines(lens.model_id, "sae")["sae"]
        prepared = _prepare_layer_baselines({"sae": baseline}, layer=layer)
        encoder = SAEFeatureEncoder(
            {
                "hidden_size": int(lens.hidden_size or captured.activations.shape[-1]),
                "layer": layer,
                "saebench_model_name": lens.model_id,
                "baselines": prepared,
            },
            device="cuda",
            dtype=torch.float32,
        )
        scaled = encoder.encode(captured.activations.float())
        captured_activations = captured.activations.detach().cpu().float()
        activations = (
            (scaled / encoder.decoder_norms.clamp_min(1e-12)).detach().cpu().float().numpy()
        )

        # Independent checkpoint oracle for the normalization and decoder contract.
        sae_config = prepared["sae"]
        tensors = _checkpoint_tensors(
            Path(sae_config["weights_file"]), str(sae_config["checkpoint_format"])
        )
        hidden_size = int(captured_activations.shape[-1])
        width = int(sae_config["width"])
        raw_decoder = _decoder_rows(tensors["W_dec"], hidden_size, width)
        oracle_norms = torch.linalg.vector_norm(raw_decoder, dim=-1).clamp_min(1e-12)
        probe_ids = torch.linspace(0, width - 1, steps=min(64, width)).long().unique()
        oracle_decoder = raw_decoder[probe_ids] / oracle_norms[probe_ids, None]
        actual_decoder = encoder.W_dec.detach().cpu().float()[probe_ids]
        actual_norms = encoder.decoder_norms.detach().cpu().float()[probe_ids]

        generator = torch.Generator(device="cpu").manual_seed(0)
        probe_values = torch.rand((2, len(probe_ids)), generator=generator)
        probe_codes = torch.zeros((2, width), dtype=torch.float32)
        probe_codes[:, probe_ids] = probe_values
        decoded = (
            encoder.decode(probe_codes.to("cuda"), reference=captured.activations[:2].float())
            .detach()
            .cpu()
            .float()
        )
        oracle_reconstruction = probe_values @ oracle_decoder + tensors["b_dec"].float()
        if str(sae_config.get("normalize_activations", "none")) == "layer_norm":
            reference = captured_activations[:2]
            mean = reference.mean(dim=-1, keepdim=True)
            std = (reference - mean).std(dim=-1, keepdim=True)
            oracle_reconstruction = oracle_reconstruction * std + mean
    finally:
        lens.unload_model()

    feature_ids = expected["sae_feature_ids"].astype(np.int64)
    ica_feature_ids = expected["ica_feature_ids"].astype(np.int64)
    actual_ica_selected = ica[:, ica_feature_ids].T.astype(np.float32)
    actual_ica_top = np.abs(ica).argmax(axis=1).astype(np.int64)
    actual_selected = activations[:, feature_ids].T.astype(np.float32)
    actual_top = activations.argmax(axis=1).astype(np.int64)
    actual_top[activations.max(axis=1) <= 0] = -1
    token_ids = captured.token_ids.cpu().numpy().astype(np.int64)

    checks = {
        "token_ids_exact": np.array_equal(token_ids, expected["token_ids"]),
        "top_feature_exact": np.array_equal(actual_top, expected["sae_top_feature"]),
        "selected_activations_close": np.allclose(
            actual_selected, expected["sae_activations"], rtol=rtol, atol=atol
        ),
        "ica_top_feature_exact": np.array_equal(actual_ica_top, expected["ica_top_feature"]),
        "ica_selected_scores_close": np.allclose(
            actual_ica_selected, expected["ica_scores"], rtol=rtol, atol=atol
        ),
        "decoder_norms_match_checkpoint": torch.allclose(
            actual_norms, oracle_norms[probe_ids], rtol=rtol, atol=atol
        ),
        "unit_decoder_rows_match_checkpoint": torch.allclose(
            actual_decoder, oracle_decoder, rtol=rtol, atol=atol
        ),
        "scaled_coefficients_match_contract": np.allclose(
            scaled.detach().cpu().float().numpy()[:, feature_ids],
            activations[:, feature_ids] * oracle_norms[feature_ids].numpy(),
            rtol=rtol,
            atol=atol,
        ),
        "decode_matches_checkpoint_contract": torch.allclose(
            decoded, oracle_reconstruction, rtol=rtol, atol=atol
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    maximum_error = float(np.max(np.abs(actual_selected - expected["sae_activations"])))
    maximum_ica_error = float(np.max(np.abs(actual_ica_selected - expected["ica_scores"])))

    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "fragment.tmp.npz"
    np.savez_compressed(
        temporary,
        token_ids=token_ids,
        feature_ids=feature_ids,
        expected_activations=expected["sae_activations"],
        actual_activations=actual_selected,
        expected_top_feature=expected["sae_top_feature"],
        actual_top_feature=actual_top,
        ica_feature_ids=ica_feature_ids,
        expected_ica_scores=expected["ica_scores"],
        actual_ica_scores=actual_ica_selected,
        expected_ica_top_feature=expected["ica_top_feature"],
        actual_ica_top_feature=actual_ica_top,
        decoder_probe_ids=probe_ids.numpy(),
        expected_decoder_norms=oracle_norms[probe_ids].numpy(),
        actual_decoder_norms=actual_norms.numpy(),
        expected_decoded=oracle_reconstruction.numpy(),
        actual_decoded=decoded.numpy(),
    )
    fragment = output / "fragment.npz"
    temporary.replace(fragment)

    return {
        "check": f"C13-sae-adapter-{canary_id}-layer{layer}",
        "status": "pass" if not failed else "fail",
        "scope": {
            "input_data_ids": ["D01", "D02", "D04"],
            "code_ids": ["C13", "C28"],
            "output_data_ids": ["D27"],
            "canary": canary_id,
            "layer": layer,
            "note": "accepted middle-layer activation-pattern sentence and SAE response",
        },
        "reference": str(reference_path),
        "checks": checks,
        "maximum_absolute_error": maximum_error,
        "maximum_ica_absolute_error": maximum_ica_error,
        "failed_checks": failed,
        "artifacts": {"fragment": str(fragment)},
    }
