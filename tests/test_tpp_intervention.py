from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.tpp.worker import SignedLinearDictionary, TrainedSAE


def _intervene(
    wrapper: Any, x: torch.Tensor, feature: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the reconstruction-error-preserving intervention used by TPP."""
    codes = wrapper.encode(x)
    reconstruction = wrapper.decode(codes)
    reconstruction_error = x - reconstruction
    ablated_codes = codes.clone()
    if feature is not None:
        ablated_codes[..., feature] = 0
    modified = wrapper.decode(ablated_codes) + reconstruction_error
    return modified, codes, ablated_codes


def _assert_intervention_contract(wrapper: Any, x: torch.Tensor, feature: int) -> None:
    unchanged, _, _ = _intervene(wrapper, x, feature=None)
    torch.testing.assert_close(unchanged, x, atol=1e-6, rtol=1e-6)

    modified, codes, ablated_codes = _intervene(wrapper, x, feature=feature)
    decoded_contribution = wrapper.decode(codes) - wrapper.decode(ablated_codes)
    torch.testing.assert_close(x - modified, decoded_contribution, atol=1e-6, rtol=1e-6)

    expected_contribution = codes[..., feature, None] * wrapper.W_dec[feature]
    torch.testing.assert_close(x - modified, expected_contribution, atol=1e-6, rtol=1e-6)


def test_tpp_relu_ica_intervention_contract() -> None:
    center = torch.tensor([0.25, -0.5])
    reading = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    writing = torch.linalg.inv(reading)
    orientation = torch.tensor([-1.0, 1.0])
    wrapper = SignedLinearDictionary(
        center=center,
        reading=reading,
        writing=writing,
        snapshot={
            "saebench_model_name": "test-model",
            "layer": 0,
            "row_normalize": False,
            "norm_eps": 1e-12,
        },
        architecture="fitted_ica_profile_oriented_relu",
        activation="relu",
        orientation=orientation,
        device="cpu",
        dtype=torch.float32,
    )
    x = torch.tensor([[[1.25, 1.5], [-0.75, -2.5]]])

    raw_scores = (x - center) @ reading.T
    expected_codes = torch.relu(raw_scores * orientation) * wrapper.decoder_norms
    torch.testing.assert_close(wrapper.encode(x), expected_codes)
    assert wrapper.cfg.activation_fn_str == "relu"
    assert torch.allclose(wrapper.W_dec.norm(dim=-1), torch.ones(2))

    # The first token has a negative profile-oriented score for component 0,
    # so its ReLU feature is inactive and ablating it changes nothing there.
    assert expected_codes[0, 0, 0].item() == 0
    modified, _, _ = _intervene(wrapper, x, feature=0)
    torch.testing.assert_close(modified[0, 0], x[0, 0], atol=1e-6, rtol=1e-6)

    _assert_intervention_contract(wrapper, x, feature=0)


def test_tpp_sae_intervention_contract(tmp_path: Path) -> None:
    weights = tmp_path / "tiny-sae.npz"
    np.savez(
        weights,
        W_enc=np.diag(np.array([0.5, 1.0 / 3.0], dtype=np.float32)),
        W_dec=np.diag(np.array([2.0, 3.0], dtype=np.float32)),
        b_enc=np.zeros(2, dtype=np.float32),
        b_dec=np.zeros(2, dtype=np.float32),
    )
    wrapper = TrainedSAE(
        {
            "baselines": {
                "sae": {
                    "weights_file": str(weights),
                    "checkpoint_format": "npz",
                    "width": 2,
                    "activation": "relu",
                    "apply_b_dec_to_input": False,
                    "normalize_activations": "none",
                    "hook_name_template": "blocks.{layer}.hook_resid_post",
                }
            },
            "hidden_size": 2,
            "saebench_model_name": "test-model",
            "layer": 0,
        },
        device="cpu",
        dtype=torch.float32,
    )
    x = torch.tensor([[[4.0, 6.0], [-2.0, 3.0]]])

    assert torch.allclose(wrapper.W_dec.norm(dim=-1), torch.ones(2))
    _assert_intervention_contract(wrapper, x, feature=1)
