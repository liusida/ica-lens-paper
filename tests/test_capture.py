from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from icalens._capture import capture_resid_post, transformer_blocks


class AddOne(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor]:
        return (values + 1,)


class HookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([AddOne(), AddOne()])
        self.transformer.ln_f = torch.nn.Identity()

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        hidden = input_ids.float().unsqueeze(-1)
        for block in self.transformer.h:
            hidden = block(hidden)[0]
        final = hidden * 100  # Deliberately distinguish final-norm output.
        return SimpleNamespace(last_hidden_state=final)


def test_last_resid_post_is_captured_before_final_norm() -> None:
    model = HookModel()
    captured = capture_resid_post(
        model,
        model_inputs={"input_ids": torch.tensor([[1, 2]])},
        layers=(0, 1),
        positions=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(captured[0].squeeze(-1), torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(captured[1].squeeze(-1), torch.tensor([3.0, 4.0]))


class CountingBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor]:
        self.calls += 1
        return (values + 1,)


class CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([CountingBlock() for _ in range(4)])
        self.completed_forward = False

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        hidden = input_ids.float().unsqueeze(-1)
        for block in self.transformer.h:
            hidden = block(hidden)[0]
        self.completed_forward = True
        return SimpleNamespace(last_hidden_state=hidden)


def test_capture_stops_after_highest_requested_layer() -> None:
    model = CountingModel()
    captured = capture_resid_post(
        model,
        model_inputs={"input_ids": torch.tensor([[1, 2]])},
        layers=(0, 1),
        positions=torch.tensor([0, 1]),
    )

    assert tuple(captured) == (0, 1)
    assert [block.calls for block in model.transformer.h] == [1, 1, 0, 0]
    assert not model.completed_forward


def test_capture_rejects_an_empty_layer_selection() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        capture_resid_post(
            HookModel(),
            model_inputs={"input_ids": torch.tensor([[1]])},
            layers=(),
            positions=torch.tensor([0]),
        )


class NestedLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([AddOne(), AddOne()])


def test_finds_nested_conditional_generation_language_blocks() -> None:
    model = NestedLanguageModel()
    assert transformer_blocks(model) is model.model.language_model.layers
