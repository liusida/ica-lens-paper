from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icalens import ICALens
from icalens.analysis import (
    _chat_group_titles,
    _resolve_device,
    _resolve_model_and_tokenizer,
    _token_presentations,
)


class DummyTokenizer:
    model_max_length = 16

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        length = len(text.split())
        return {
            "input_ids": torch.arange(1, length + 1).unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"t{value}" for value in ids]

    def decode(self, ids: list[int]) -> str:
        return f"t{ids[0]}"


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        values = input_ids.float()
        hidden = torch.stack((values, values.square(), values + 1), dim=-1)
        return SimpleNamespace(hidden_states=(hidden,))


class DummyLogitModel(DummyModel):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = SimpleNamespace(ln_f=torch.nn.Identity())
        self.output_embeddings = torch.nn.Linear(3, 7, bias=False)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.output_embeddings


class DummyLogitTokenizer(DummyTokenizer):
    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return f"v{ids}"
        return [f"v{value}" for value in ids]


class DummyGenerationTokenizer(DummyTokenizer):
    def decode(self, ids: object, **_: object) -> str:
        values = torch.as_tensor(ids).tolist()
        return " ".join(f"t{value}" for value in values)


class TupleBlock(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden + 1,)


class DummyGenerationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([TupleBlock()])
        self.block_output: torch.Tensor | None = None

    def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        values = input_ids.float()
        hidden = torch.stack((values, values.square(), values + 1), dim=-1)
        self.block_output = self.transformer.h[0](hidden)[0]
        new_token = torch.full((input_ids.shape[0], 1), 9, device=input_ids.device)
        return torch.cat((input_ids, new_token), dim=1)


class DummyAutoregressiveGenerationModel(DummyGenerationModel):
    def __init__(self) -> None:
        super().__init__()
        self.block_outputs: list[torch.Tensor] = []

    def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        values = input_ids.float()
        prefill = torch.stack((values, values.square(), values + 1), dim=-1)
        self.block_outputs.append(self.transformer.h[0](prefill)[0])
        decode_values = torch.full((input_ids.shape[0], 1), 9.0, device=input_ids.device)
        decode = torch.stack((decode_values, decode_values.square(), decode_values + 1), dim=-1)
        self.block_outputs.append(self.transformer.h[0](decode)[0])
        new_token = torch.full((input_ids.shape[0], 1), 9, device=input_ids.device)
        return torch.cat((input_ids, new_token), dim=1)


def test_analyze_raw_text_returns_scores_and_energy(mixed_signals: np.ndarray) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(mixed_signals, layer=0)
    result = lens.analyze(
        "one two three",
        layer=0,
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
    )
    assert result.tokens == ("t1", "t2", "t3")
    assert result.token_texts == ("t1", "t2", "t3")
    assert result.scores.shape == (3, 3)
    assert result.model == "example/model@abc"
    assert result.layer == 0
    assert result.input_text == "one two three"
    torch.testing.assert_close(result.energy.sum(dim=-1), torch.ones(3))


def test_analyze_uses_recorded_document_framing_by_default(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(
        mixed_signals,
        layer=0,
        provenance={
            "document_framing": {
                "strategy": "prepend-bos",
                "token": "<bos>",
                "token_id": 42,
            }
        },
    )

    result = lens.analyze(
        "one two",
        layer=0,
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
    )

    assert result.tokens == ("t1", "t2")
    torch.testing.assert_close(result.activations[:, 0], torch.tensor([1.0, 2.0]))


def test_analyze_document_framing_controls_tokenizer_special_tokens(
    mixed_signals: np.ndarray,
) -> None:
    class RecordingTokenizer(DummyTokenizer):
        def __init__(self) -> None:
            self.add_special_tokens: list[bool] = []

        def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
            add_special_tokens = bool(kwargs.get("add_special_tokens"))
            self.add_special_tokens.append(add_special_tokens)
            encoded = super().__call__(text, **kwargs)
            if add_special_tokens:
                prefix = torch.tensor([[42]])
                encoded["input_ids"] = torch.cat((prefix, encoded["input_ids"]), dim=1)
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
                encoded["special_tokens_mask"] = torch.tensor([[1, 0, 0]])
            return encoded

    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(mixed_signals, layer=0)
    tokenizer = RecordingTokenizer()

    model_result = lens.analyze(
        "one two",
        layer=0,
        document_framing="model",
        model=DummyModel(),
        tokenizer=tokenizer,
    )
    none_result = lens.analyze(
        "one two",
        layer=0,
        document_framing="none",
        model=DummyModel(),
        tokenizer=tokenizer,
    )

    assert tokenizer.add_special_tokens == [True, False]
    assert model_result.tokens == ("t1", "t2")
    assert none_result.tokens == ("t1", "t2")


def test_analyze_rejects_invalid_document_framing(mixed_signals: np.ndarray) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc").fit(
        mixed_signals, layer=0
    )
    with pytest.raises(ValueError, match="document_framing"):
        lens.analyze(
            "one two",
            layer=0,
            document_framing="automatic",
            model=DummyModel(),
            tokenizer=DummyTokenizer(),
        )


def test_analyze_records_initial_component_selection(mixed_signals: np.ndarray) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(mixed_signals, layer=0)

    result = lens.analyze(
        "one two three",
        layer=0,
        selected_components=2,
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
    )

    assert result.selected_components == (2,)


def test_analyze_rejects_invalid_initial_component_selections(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(mixed_signals, layer=0)
    kwargs = {"layer": 0, "model": DummyModel(), "tokenizer": DummyTokenizer()}

    with pytest.raises(ValueError, match="currently supports one"):
        lens.analyze("one two three", selected_components=[0, 1], **kwargs)
    with pytest.raises(ValueError, match="between 0 and 2"):
        lens.analyze("one two three", selected_components=3, **kwargs)


def test_analyze_verbose_reports_major_stages(
    mixed_signals: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(mixed_signals, layer=0)

    lens.analyze(
        "one two three",
        layer=0,
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "[ICALens] Preparing example/model..." in output
    assert "Capturing 3 token activations" in output
    assert "Computing ICA scores and component energy" in output
    assert "Analysis complete" in output


def test_add_logit_effects_covers_top_components_at_every_token(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(mixed_signals, layer=0)
    model = DummyLogitModel()
    tokenizer = DummyLogitTokenizer()
    result = lens.analyze("one two three", layer=0, model=model, tokenizer=tokenizer)
    lens._analysis_model = model
    lens._analysis_tokenizer = tokenizer

    enriched = lens.add_logit_effects(
        result,
        components_per_token=2,
        effect_tokens_per_component=4,
        batch_size=2,
    )

    assert result.logit_effects == ()
    assert len(enriched.logit_effects) == 6
    assert {(item["token_index"], item["component"]) for item in enriched.logit_effects} == {
        (token_index, int(component))
        for token_index, row in enumerate(result.scores.abs().topk(2, dim=-1).indices)
        for component in row
    }
    assert all(len(item["tokens"]) == 4 for item in enriched.logit_effects)
    assert all(item["multiplier"] == 1.1 for item in enriched.logit_effects)


def test_auto_device_uses_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"
    assert _resolve_device(None) == "cpu"


def test_auto_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("auto") == "cuda"
    assert _resolve_device("cpu") == "cpu"


def test_analysis_model_and_tokenizer_are_reused() -> None:
    lens = ICALens(model_id="example/model", model_revision="abc")
    model = DummyModel()
    tokenizer = DummyTokenizer()
    lens._analysis_model = model
    lens._analysis_tokenizer = tokenizer
    lens._analysis_device = "cpu"

    resolved_model, resolved_tokenizer = _resolve_model_and_tokenizer(lens, None, None, "cpu")

    assert resolved_model is model
    assert resolved_tokenizer is tokenizer


def test_unload_model_clears_analysis_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc")
    lens._analysis_model = DummyModel()
    lens._analysis_tokenizer = DummyTokenizer()
    lens._analysis_device = "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    lens.unload_model()

    assert lens._analysis_model is None
    assert lens._analysis_tokenizer is None
    assert lens._analysis_device is None


def test_generate_clamps_one_component_at_every_position(mixed_signals: np.ndarray) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc").fit(mixed_signals, layer=0)
    model = DummyGenerationModel()
    output = lens.generate(
        "one two",
        layer=0,
        clamp=(1, 2.5),
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert output == "t9"
    assert model.block_output is not None
    values = torch.tensor([[1.0, 2.0]])
    original = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    expected_scores = lens.transform(original, layer=0)
    expected_scores[..., 1] = 2.5
    reconstructed = lens.inverse_transform(expected_scores, layer=0)
    expected = lens.restore_norm(reconstructed, reference=original)
    torch.testing.assert_close(model.block_output, expected)


def test_generate_clamps_multiple_components_at_every_position(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc").fit(mixed_signals, layer=0)
    model = DummyGenerationModel()
    lens.generate(
        "one two",
        layer=0,
        clamp={0: -1.5, 2: 3.0},
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert model.block_output is not None
    values = torch.tensor([[1.0, 2.0]])
    original = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    expected_scores = lens.transform(original, layer=0)
    expected_scores[..., 0] = -1.5
    expected_scores[..., 2] = 3.0
    reconstructed = lens.inverse_transform(expected_scores, layer=0)
    expected = lens.restore_norm(reconstructed, reference=original)
    torch.testing.assert_close(model.block_output, expected)


def test_generate_steers_current_position_by_default(mixed_signals: np.ndarray) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    model = DummyGenerationModel()
    lens.generate(
        "one two",
        layer=0,
        steer=(1, 2.5),
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert model.block_output is not None
    values = torch.tensor([[1.0, 2.0]])
    original = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    expected = original.clone()
    artifact = lens._get_layer(0)
    assert artifact.writing_matrix is not None
    expected[:, -1, :] += 2.5 * torch.from_numpy(artifact.writing_matrix[:, 1])
    torch.testing.assert_close(model.block_output, expected)


def test_generate_steers_all_positions(mixed_signals: np.ndarray) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    model = DummyGenerationModel()
    lens.generate(
        "one two",
        layer=0,
        steer={0: -1.5, 2: 3.0},
        steering_scope="all-positions",
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert model.block_output is not None
    values = torch.tensor([[1.0, 2.0]])
    original = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    artifact = lens._get_layer(0)
    assert artifact.writing_matrix is not None
    direction = -1.5 * torch.from_numpy(artifact.writing_matrix[:, 0]) + 3.0 * torch.from_numpy(
        artifact.writing_matrix[:, 2]
    )
    expected = original + direction
    torch.testing.assert_close(model.block_output, expected)


def test_generate_can_clamp_and_steer_together(mixed_signals: np.ndarray) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    model = DummyGenerationModel()
    lens.generate(
        "one two",
        layer=0,
        clamp=(0, 0.0),
        steer=(1, 2.5),
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert model.block_output is not None
    values = torch.tensor([[1.0, 2.0]])
    original = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    clamped_scores = lens.transform(original, layer=0)
    clamped_scores[..., 0] = 0.0
    reconstructed = lens.inverse_transform(clamped_scores, layer=0)
    expected = lens.restore_norm(reconstructed, reference=original)
    artifact = lens._get_layer(0)
    assert artifact.writing_matrix is not None
    expected[:, -1, :] += 2.5 * torch.from_numpy(artifact.writing_matrix[:, 1])
    torch.testing.assert_close(model.block_output, expected)


def test_generate_debug_prints_scores_before_and_after(
    mixed_signals: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    lens.generate(
        "one two",
        layer=0,
        steer=(1, 2.5),
        steering_scope="all-positions",
        max_new_tokens=1,
        device="cpu",
        model=DummyGenerationModel(),
        tokenizer=DummyGenerationTokenizer(),
        debug=True,
    )

    output = capsys.readouterr().out
    assert "[ICALens debug] step=0 token='t2' C1" in output
    assert "token='t2'" in output
    assert "last=" in output
    assert "affected=2" in output
    assert "before[min/mean/max]=" in output
    assert "after[min/mean/max]=" in output


def test_generate_current_position_also_steers_each_decode_step(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    model = DummyAutoregressiveGenerationModel()
    lens.generate(
        "one two",
        layer=0,
        steer=(1, 2.5),
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert len(model.block_outputs) == 2
    artifact = lens._get_layer(0)
    assert artifact.writing_matrix is not None
    direction = 2.5 * torch.from_numpy(artifact.writing_matrix[:, 1])
    prefill, decode = model.block_outputs
    values = torch.tensor([[1.0, 2.0]])
    original_prefill = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    expected_prefill = original_prefill.clone()
    expected_prefill[:, -1, :] += direction
    torch.testing.assert_close(prefill, expected_prefill)
    decode_values = torch.tensor([[9.0]])
    expected_decode = (
        torch.stack((decode_values, decode_values.square(), decode_values + 1), dim=-1)
        + 1
        + direction
    )
    torch.testing.assert_close(decode, expected_decode)


def test_generate_supports_stronger_initial_steer_and_later_cap(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    model = DummyAutoregressiveGenerationModel()
    lens.generate(
        "one two",
        layer=0,
        steer=(1, 10.0),
        initial_steer=(1, 20.0),
        steer_cap=(1, 10.0),
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert len(model.block_outputs) == 2
    prefill, decode = model.block_outputs
    values = torch.tensor([[1.0, 2.0]])
    original_prefill = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    original_decode_values = torch.tensor([[9.0]])
    original_decode = (
        torch.stack(
            (
                original_decode_values,
                original_decode_values.square(),
                original_decode_values + 1,
            ),
            dim=-1,
        )
        + 1
    )
    decode_scores = lens.transform(original_decode, layer=0)
    artifact = lens._get_layer(0)
    assert artifact.writing_matrix is not None
    direction = torch.from_numpy(artifact.writing_matrix[:, 1])
    expected_prefill = original_prefill.clone()
    expected_prefill[:, -1, :] += 20.0 * direction
    decode_offset = torch.clamp(10.0 - decode_scores[:, -1, 1], min=0.0, max=10.0)
    expected_decode = original_decode.clone()
    expected_decode[:, -1, :] += decode_offset.unsqueeze(-1) * direction
    torch.testing.assert_close(prefill, expected_prefill)
    torch.testing.assert_close(decode, expected_decode)


def test_generate_three_item_steer_is_cap_shorthand(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    explicit_model = DummyAutoregressiveGenerationModel()
    shorthand_model = DummyAutoregressiveGenerationModel()

    common = {
        "layer": 0,
        "max_new_tokens": 1,
        "device": "cpu",
        "tokenizer": DummyGenerationTokenizer(),
    }
    lens.generate(
        "one two",
        steer=(1, 10.0),
        steer_cap=(1, 20.0),
        model=explicit_model,
        **common,
    )
    lens.generate(
        "one two",
        steer=(1, 10.0, 20.0),
        model=shorthand_model,
        **common,
    )

    assert len(explicit_model.block_outputs) == len(shorthand_model.block_outputs)
    for explicit, shorthand in zip(
        explicit_model.block_outputs, shorthand_model.block_outputs, strict=True
    ):
        torch.testing.assert_close(explicit, shorthand)


def test_generate_initial_steer_uses_its_own_optional_cap(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        icalens_preprocessing="none",
    ).fit(mixed_signals, layer=0)
    model = DummyAutoregressiveGenerationModel()
    lens.generate(
        "one two",
        layer=0,
        initial_steer=(1, 20.0, 25.0),
        steer=(1, 5.0, 10.0),
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    prefill, decode = model.block_outputs
    values = torch.tensor([[1.0, 2.0]])
    original_prefill = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    decode_values = torch.tensor([[9.0]])
    original_decode = (
        torch.stack((decode_values, decode_values.square(), decode_values + 1), dim=-1) + 1
    )
    prefill_scores = lens.transform(original_prefill, layer=0)
    decode_scores = lens.transform(original_decode, layer=0)
    artifact = lens._get_layer(0)
    assert artifact.writing_matrix is not None
    direction = torch.from_numpy(artifact.writing_matrix[:, 1])
    prefill_offset = torch.clamp(25.0 - prefill_scores[:, -1, 1], min=0.0, max=20.0)
    decode_offset = torch.clamp(10.0 - decode_scores[:, -1, 1], min=0.0, max=5.0)
    expected_prefill = original_prefill.clone()
    expected_prefill[:, -1, :] += prefill_offset.unsqueeze(-1) * direction
    expected_decode = original_decode.clone()
    expected_decode[:, -1, :] += decode_offset.unsqueeze(-1) * direction
    torch.testing.assert_close(prefill, expected_prefill)
    torch.testing.assert_close(decode, expected_decode)


def test_generate_empty_text_uses_recorded_document_framing(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc").fit(
        mixed_signals,
        layer=0,
        provenance={
            "document_framing": {
                "strategy": "prepend-bos",
                "token": "<bos>",
                "token_id": 42,
            }
        },
    )
    model = DummyGenerationModel()

    output = lens.generate(
        "",
        document_framing="recorded",
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert output == "t9"
    assert model.block_output is not None
    expected = torch.tensor([[[43.0, 1765.0, 44.0]]])
    torch.testing.assert_close(model.block_output, expected)


def test_generate_uses_model_document_framing_by_default(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc").fit(
        mixed_signals,
        layer=0,
        provenance={
            "document_framing": {
                "strategy": "prepend-bos",
                "token": "<bos>",
                "token_id": 42,
            }
        },
    )
    model = DummyGenerationModel()

    lens.generate(
        "one two",
        max_new_tokens=1,
        device="cpu",
        model=model,
        tokenizer=DummyGenerationTokenizer(),
    )

    assert model.block_output is not None
    values = torch.tensor([[1.0, 2.0]])
    expected = torch.stack((values, values.square(), values + 1), dim=-1) + 1
    torch.testing.assert_close(model.block_output, expected)


def test_generate_document_framing_controls_tokenizer_special_tokens(
    mixed_signals: np.ndarray,
) -> None:
    class RecordingTokenizer(DummyGenerationTokenizer):
        def __init__(self) -> None:
            self.add_special_tokens: list[bool] = []

        def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
            self.add_special_tokens.append(bool(kwargs.get("add_special_tokens")))
            return super().__call__(text, **kwargs)

    lens = ICALens(model_id="example/model", model_revision="abc").fit(
        mixed_signals, layer=0
    )
    tokenizer = RecordingTokenizer()
    for framing in ("model", "none"):
        lens.generate(
            "one two",
            document_framing=framing,
            max_new_tokens=1,
            device="cpu",
            model=DummyGenerationModel(),
            tokenizer=tokenizer,
        )

    assert tokenizer.add_special_tokens == [True, False]


def test_generate_empty_text_requires_recorded_document_framing(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc").fit(mixed_signals, layer=0)
    with pytest.raises(ValueError, match="requires a recorded BOS/EOS"):
        lens.generate(
            "",
            document_framing="recorded",
            max_new_tokens=1,
            device="cpu",
            model=DummyGenerationModel(),
            tokenizer=DummyGenerationTokenizer(),
        )


def test_generate_validates_clamp_arguments() -> None:
    lens = ICALens(model_id="example/model", model_revision="abc")
    with pytest.raises(ValueError, match="layer is required"):
        lens.generate("prompt", clamp=(0, 1.0))


def test_generate_validates_steering_arguments() -> None:
    lens = ICALens(model_id="example/model", model_revision="abc")
    with pytest.raises(ValueError, match="layer is required"):
        lens.generate("prompt", steer=(0, 1.0))
    with pytest.raises(ValueError, match="initial_steer requires steer"):
        lens.generate("prompt", initial_steer=(0, 1.0))
    with pytest.raises(ValueError, match="steer_cap requires steer"):
        lens.generate("prompt", steer_cap=(0, 1.0))
    with pytest.raises(ValueError, match="steering_scope"):
        lens.generate(
            "prompt",
            layer=0,
            steer=(0, 1.0),
            steering_scope="decode-only",
        )
    with pytest.raises(ValueError, match="document_framing"):
        lens.generate("prompt", document_framing="automatic")


def test_generate_additive_steering_rejects_normalized_lens(
    mixed_signals: np.ndarray,
) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc").fit(mixed_signals, layer=0)
    with pytest.raises(ValueError, match="unnormalized Lens"):
        lens.generate(
            "prompt",
            layer=0,
            steer=(0, 1.0),
            max_new_tokens=1,
            device="cpu",
            model=DummyGenerationModel(),
            tokenizer=DummyGenerationTokenizer(),
        )


def test_invalid_token_fragment_uses_placeholder_and_contextual_tooltip() -> None:
    class FragmentTokenizer:
        def decode(self, ids: list[int]) -> str:
            if len(ids) > 1:
                return "你"
            return "�"

    texts, labels, tooltips = _token_presentations(
        FragmentTokenizer(), [99, 100], [0, 1], ("fragment-a", "fragment-b")
    )

    assert texts == ("�", "�")
    assert labels == ("<?>", "<?>")
    assert "Token ID 99 (0x0063)" in tooltips[0]
    assert "tokens 0–1 decode together as '你'" in tooltips[0]


def test_chat_tokens_are_grouped_with_their_message_template() -> None:
    rendered = "<start>user\nHi<end><start>assistant\nHello<end>"
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    pieces = [
        "<start>",
        "user",
        "\n",
        "Hi",
        "<end>",
        "<start>",
        "assistant",
        "\n",
        "Hello",
        "<end>",
    ]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)

    groups = _chat_group_titles(rendered, messages, offsets, pieces)

    assert groups[:5] == ["User 1"] * 5
    assert groups[5:] == ["Assistant 1"] * 5
