from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from icalens import ICALens
from icalens._capture import MLP_OUT, RESID_POST, capture_sites, site_module
from icalens.analysis import add_logit_effects

VOCAB = 40
WIDTH = 8


class WordTokenizer:
    model_max_length = 64

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        ids = [(sum(map(ord, word)) % (VOCAB - 1)) + 1 for word in text.split()]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return f"v{ids}"
        return [f"v{value}" for value in ids]

    def decode(self, ids: list[int], **_: object) -> str:
        return " ".join(f"v{value}" for value in ids)


@pytest.fixture(scope="module")
def model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=WIDTH, n_layer=2, n_head=2)
    return GPT2LMHeadModel(config).eval()


def _inputs(length: int = 48, seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randint(0, VOCAB, (1, length), generator=generator)}


def test_mlp_out_hook_is_the_mlp_write_into_the_residual_stream(model: GPT2LMHeadModel) -> None:
    inputs = _inputs()
    positions = torch.arange(inputs["input_ids"].shape[1])
    captured = capture_sites(
        model,
        model_inputs=inputs,
        requests=((MLP_OUT, 1), (RESID_POST, 1)),
        positions=positions,
    )
    mlp_out = captured[(MLP_OUT, 1)]
    resid_mid = captured[(RESID_POST, 1)] - mlp_out
    block = model.transformer.h[1]
    with torch.inference_mode():
        expected = block.mlp(block.ln_2(resid_mid))
    torch.testing.assert_close(mlp_out, expected, rtol=1e-5, atol=1e-5)


def test_mlp_out_refuses_unverified_architectures(model: GPT2LMHeadModel) -> None:
    original = model.config.model_type
    try:
        model.config.model_type = "gemma2"
        with pytest.raises(ValueError, match="verified only"):
            site_module(model, MLP_OUT, 0)
    finally:
        model.config.model_type = original
    with pytest.raises(ValueError, match="unsupported hooked activation site"):
        site_module(model, "attn_out", 0)


def _mlp_out_lens(model: GPT2LMHeadModel, layer: int) -> ICALens:
    inputs = _inputs(length=64, seed=2)
    activations = capture_sites(
        model,
        model_inputs=inputs,
        requests=((MLP_OUT, layer),),
        positions=torch.arange(64),
    )[(MLP_OUT, layer)]
    lens = ICALens(
        model_id="tiny/gpt2",
        model_revision="test",
        activation_site=MLP_OUT,
        layer_indexing="transformer_blocks_zero_based",
        icalens_preprocessing="none",
    )
    return lens.fit(activations.float(), layer=layer, max_iter=5, batch_size=64)


def test_analyze_uses_mlp_out_and_records_residual(model: GPT2LMHeadModel) -> None:
    lens = _mlp_out_lens(model, layer=1)
    tokenizer = WordTokenizer()
    text = "alpha beta gamma delta epsilon zeta"
    result = lens.analyze(
        text, layer=1, model=model, tokenizer=tokenizer, document_framing="none", device=None
    )

    encoded = tokenizer(text)
    positions = torch.arange(encoded["input_ids"].shape[1])
    expected = capture_sites(
        model,
        model_inputs=encoded,
        requests=((MLP_OUT, 1), (RESID_POST, 1)),
        positions=positions,
    )
    torch.testing.assert_close(result.activations, expected[(MLP_OUT, 1)].float())
    assert result.residual is not None
    torch.testing.assert_close(result.residual, expected[(RESID_POST, 1)].float())
    assert result.scores.shape == (len(positions), WIDTH)


def test_mlp_out_logit_effects_propagate_through_the_residual_stream(
    model: GPT2LMHeadModel,
) -> None:
    lens = _mlp_out_lens(model, layer=0)
    tokenizer = WordTokenizer()
    result = lens.analyze(
        "one two three four",
        layer=0,
        model=model,
        tokenizer=tokenizer,
        document_framing="none",
        device=None,
    )
    lens._analysis_model = model
    lens._analysis_tokenizer = tokenizer
    effects = add_logit_effects(
        lens, result, components_per_token=1, effect_tokens_per_component=VOCAB
    ).logit_effects

    effect = effects[0]
    token, component = effect["token_index"], effect["component"]
    scores = result.scores[token : token + 1].clone()
    scores[0, component] *= effect["multiplier"]
    reference = result.activations[token : token + 1]
    edited = lens.restore_norm(lens.inverse_transform(scores, layer=0), reference=reference)
    residual = result.residual[token : token + 1]
    with torch.inference_mode():
        delta = model.transformer.ln_f(residual + (edited - reference)) - model.transformer.ln_f(
            residual
        )
        expected = delta @ model.get_output_embeddings().weight.T
    reported = {entry["token_id"]: entry["logit"] for entry in effect["tokens"]}
    for token_id, value in reported.items():
        assert value == pytest.approx(float(expected[0, token_id]), abs=1e-4)
