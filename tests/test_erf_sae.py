"""SAE ERF validation and threshold-independent batching without model downloads."""

import pytest
import torch

from icalens.experiments import erf_suffix_sweep as sweep
from icalens.experiments.erf_sae import _identity_sha256, _load_layer, validate_result


def test_zero_activation_and_competition_rank():
    rank, sign, recovered = sweep._recovery_from_scores(
        torch.zeros(1, 8), component=0, direction="positive", top_k=5
    )
    assert rank.item() == 1 and not sign.item() and not recovered.item()
    rank, _, recovered = sweep._recovery_from_scores(
        torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 5.0]]), component=5, direction="positive", top_k=5
    )
    assert rank.item() == 5 and recovered.item()


def test_layer_bundle_identity(tmp_path):
    identity = {"k": 5}
    p = tmp_path / "layer.json"
    assert _load_layer(p, identity, "result") is None
    p.write_text(
        '{"identity_sha256":"'
        + _identity_sha256(identity)
        + '","kind":"result"}'
    )
    assert _load_layer(p, identity, "result")["kind"] == "result"
    with pytest.raises(ValueError, match="incompatible"):
        _load_layer(p, {"k": 1}, "result")


def test_inactive_and_missing_threshold_validation():
    validate_result({"n_occurrences": 0, "status": "no_positive_examples"}, 0)
    with pytest.raises(ValueError, match="incorrect result thresholds"):
        validate_result({"n_occurrences": 1, "occurrences": [{}], "rank_thresholds": [5]}, 1)
