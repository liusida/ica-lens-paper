from __future__ import annotations


def hidden_state_layer_names(num_hidden_layers: int) -> list[str]:
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be positive.")
    return [f"layer_{idx:02d}" for idx in range(num_hidden_layers)]
