from __future__ import annotations

from types import SimpleNamespace

import torch

from ica_lens.activation_capture import capture_random_token_hidden_state_shards


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, *, return_tensors: str, truncation: bool, max_length: int) -> dict[str, torch.Tensor]:
        del return_tensors, truncation
        ids = [idx + 2 for idx, _ in enumerate(text.split())][:max_length] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class CountingBlock(torch.nn.Module):
    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + self.value


class TinyInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([CountingBlock(idx + 1) for idx in range(4)])


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.model = TinyInner()
        self.config = SimpleNamespace(num_hidden_layers=4, hidden_size=3, vocab_size=128)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        states = [x]
        for block in self.model.layers:
            x = block(x)
            if output_hidden_states:
                states.append(x)
        return SimpleNamespace(hidden_states=tuple(states) if output_hidden_states else None)


def test_selected_layer_capture_uses_hooks_and_stops_after_deepest_layer(tmp_path) -> None:
    model = TinyModel()

    manifest = capture_random_token_hidden_state_shards(
        texts=["aa bb cc dd ee", "ff gg hh ii"],
        model=model,
        tokenizer=TinyTokenizer(),
        output_dir=tmp_path,
        run_name="tiny",
        model_id="tiny",
        model_short_name="tiny",
        dataset_manifest={"path": "inline", "split": "test", "text_column": "text"},
        context_length=8,
        token_budget=3,
        activation_dtype="float32",
        shard_token_budget=10,
        seed=0,
        layers=["layer_02"],
    )

    shard = torch.load(manifest.parent / "layer_02" / "shard_00000.pt")

    assert tuple(shard.shape) == (3, 3)
    assert [block.calls for block in model.model.layers] == [2, 2, 2, 0]
