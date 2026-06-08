from __future__ import annotations

from ica_lens.activations import hidden_state_layer_names


def test_hidden_state_layer_names() -> None:
    assert hidden_state_layer_names(3) == ["layer_00", "layer_01", "layer_02"]
