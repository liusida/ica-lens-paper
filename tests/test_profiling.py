from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icalens import ICALens
from icalens.profiling import _absolute_score_ranks, _final_norm, _TopScoreOccurrences


class TinyTokenizer:
    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"T{token_id}"

    def decode(self, token_ids: list[int]) -> str:
        return f" token-{token_ids[0]}"


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(2, 4, bias=False)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.head


def test_final_norm_supports_gpt_neox_layout() -> None:
    model = torch.nn.Module()
    model.gpt_neox = torch.nn.Module()
    model.gpt_neox.final_layer_norm = torch.nn.LayerNorm(2)

    assert _final_norm(model) is model.gpt_neox.final_layer_norm


def test_top_score_occurrences_ignore_relative_energy_rank() -> None:
    selector = _TopScoreOccurrences(directions=["positive", "negative"], top_k=2)
    selector.update(
        torch.tensor([[0.01, 0.90], [0.20, 0.01], [0.10, 0.08]], dtype=torch.float64),
        torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], dtype=torch.float64),
        row_offset=0,
    )

    examples = selector.finish(lambda row: {"row": row})

    assert [item[0] for item in examples[0]["positive"]] == [3.0, 2.0]
    assert [item[2]["energy"] for item in examples[0]["positive"]] == [0.10, 0.20]
    assert [item[0] for item in examples[1]["negative"]] == [3.0, 2.0]
    assert [item[2]["energy"] for item in examples[1]["negative"]] == [0.08, 0.01]


def test_absolute_score_ranks_use_one_based_competition_ranking() -> None:
    scores = torch.tensor([[3.0, -3.0, 2.0, -1.0], [0.0, -4.0, 2.0, -3.0]])

    assert _absolute_score_ranks(scores).tolist() == [[1, 1, 3, 4], [4, 1, 3, 2]]


def test_profiles_and_round_trips_component_metadata(tmp_path, monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)

    profile = lens.profile_components(["example"], layer=1, top_k_examples=2)

    assert profile["n_tokens"] == 2
    assert profile["components"][0]["sign_statistics"]["positive_fraction"] == 0.5
    assert profile["components"][0]["dominant_sign"] == "negative"
    assert profile["components"][0]["score_statistics"]["excess_kurtosis"] == pytest.approx(-2.0)
    assert profile["components"][0]["score_statistics"]["skewness"] == pytest.approx(0.0)
    assert profile["components"][0]["score_statistics"]["excess_kurtosis_rank"] == 1
    assert profile["components"][1]["score_statistics"]["excess_kurtosis_rank"] == 2
    assert profile["components"][0]["examples"]["negative"]["tokens"] == [
        {"text": " beta", "count": 1}
    ]
    assert profile["components"][0]["examples"]["negative"]["occurrences"]
    assert profile["components"][0]["examples"]["positive"]["occurrences"] == []
    assert profile["selection"]["example_selection"] == ("top_absolute_score_on_selected_tail")
    artifact = lens.save(tmp_path / "profiled")
    assert (artifact / "component_profiles/resid_post/layer_01.json.gz").is_file()
    lens.checkpoint_component_profile(artifact, layer=1)

    loaded = ICALens.from_pretrained(artifact)
    component = loaded.component_profile(layer=1, component=0)
    assert component["dominant_sign"] == "negative"
    assert component["logit_lens"]["method"] == "final_norm_then_unembed"
    assert component["fitting_statistics"]["contrast"] == "logcosh"
    assert component["fitting_statistics"]["absolute_deviation_rank"] in (1, 2)
    top_occurrence = loaded.top_occurrence(layer=1, component=0)
    assert top_occurrence == component["examples"][component["tail_direction"]]["occurrences"][0]
    top_occurrence["text"] = "changed"
    assert loaded.top_occurrence(layer=1, component=0)["text"] != "changed"
    with pytest.raises(IndexError, match="occurrence id"):
        loaded.top_occurrence(layer=1, component=0, id=20)
    with pytest.raises(ValueError, match="non-negative integer"):
        loaded.top_occurrence(layer=1, component=0, id=-1)
    loaded._get_layer(1).profile["example_provenance"] = {
        "dataset": {"repo_id": "owner/data", "split": "train", "revision": "abc"},
        "text_field": "body",
    }
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *args, **kwargs: [{"body": "The complete source sample."}],
    )
    assert loaded.top_occurrence(layer=1, component=0, full=True)["full_text"] == (
        "The complete source sample."
    )
    summaries = loaded._component_profile_summaries(1)
    assert summaries is not None
    summary_token = summaries[0]["logit_tokens"][0]
    assert {"text", "token", "token_id", "logit"} <= summary_token.keys()
    summary_occurrence = summaries[0]["occurrences"][0]
    assert {
        "text",
        "token",
        "token_id",
        "context",
        "context_target_start",
        "context_target_end",
        "score",
        "energy",
    } <= (summary_occurrence.keys())
    assert summary_occurrence["context"] == " alpha beta"
    assert summary_occurrence["context_target_start"] == len(" alpha")
    assert summary_occurrence["context_target_end"] == len(" alpha beta")
    assert component.component == 0
    assert component.layer == 1
    rendered = component._repr_html_()
    assert rendered.startswith('<iframe title="ICA Lens Component Profile"')
    assert "Component profile — C0 · layer 1 · negative tail" in rendered
    assert "Tail selection" not in rendered
    assert "Skewness" in rendered
    assert "+0.00" in rendered
    assert "Logit-lens tokens · negative" in rendered
    assert "Token ID:" in rendered
    assert "Raw token:" in rendered
    assert "Bytes:" in rendered
    assert "Excess kurtosis" in rendered
    assert "Logcosh deviation" in rendered
    assert "|Logcosh contrast/objective − 0.374567|" in rendered
    assert "grid-template-columns:minmax(0,1fr) 150px" in rendered
    assert "profile-score-statistics {display:flex;flex-direction:column;gap:12px" in rendered
    assert "profile-score-stat-value {color:#273244;font-size:15px}" in rendered
    assert "profile-score-stat-rank {margin-left:6px;color:#647084;font-size:10px}" in rendered
    assert "profile-score-stat-pair" not in rendered
    assert rendered.index("Logit-lens tokens") < rendered.index("Skewness")
    report = component.to_html(tmp_path / "component-profile.html")
    assert report.is_file()
    assert "Top-score occurrences · negative" in report.read_text()
    assert 'class="profile-occurrence-token" title="Token ID:' in report.read_text()
    assert "Absolute-score rank among all components at this token" in report.read_text()
    assert profile["format_version"] == 1


def test_profiles_from_cached_activations_without_analyze(monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    lens._analysis_device = "cpu"
    monkeypatch.setattr(
        lens,
        "analyze",
        lambda *args, **kwargs: pytest.fail("cached profiling must not call analyze"),
    )
    records = [
        {
            "token": f"T{index}",
            "text": f" token-{index}",
            "token_id": index,
            "position": index,
            "context": "cached context",
            "source_index": index // 2,
        }
        for index in range(4)
    ]

    profile = lens.profile_components_from_activations(
        torch.from_numpy(signals),
        records,
        layer=1,
        batch_size=2,
        device="cpu",
    )

    assert profile["n_tokens"] == 4
    assert profile["n_inputs"] == 2
    assert "excess_kurtosis" in profile["components"][0]["score_statistics"]
    assert profile["components"][0]["examples"]["positive"]["occurrences"]


def test_refresh_profile_statistics_uses_skewness_for_tail_direction(monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)
    lens.profile_components(["example"], layer=1, top_k_examples=2)
    monkeypatch.setattr(lens, "transform", lambda values, *, layer: values)
    activations = torch.tensor([[1.0, -1.0], [1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [-1.8, 1.0]])

    refreshed = lens.refresh_profile_statistics_from_activations(
        activations,
        layer=1,
        batch_size=2,
        provenance={"source": "test"},
        device="cpu",
    )

    component = refreshed["components"][0]
    assert component["sign_statistics"]["positive_energy_fraction"] > 0.5
    assert component["score_statistics"]["skewness"] < 0
    assert component["tail_direction"] == "negative"
    assert component["dominant_sign"] == "negative"
    assert component["logit_lens"]["dominant"] == component["logit_lens"]["negative"]
    assert refreshed["selection"]["sign_selection"] == "population_skewness"
    assert refreshed["score_statistics_provenance"] == {"source": "test"}


def test_refresh_examples_uses_existing_tail_and_preserves_statistics(monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    lens._analysis_device = "cpu"
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)
    profile = lens.profile_components(["example"], layer=1, top_k_examples=2)
    before = dict(profile["components"][0]["score_statistics"])
    profile["components"][0]["tail_direction"] = "positive"
    profile["components"][0]["dominant_sign"] = "positive"
    monkeypatch.setattr(lens, "transform", lambda values, *, layer: values)
    records = [{"text": f" row-{i}", "source_index": i} for i in range(4)]

    refreshed = lens.refresh_profile_examples_from_activations(
        torch.tensor([[1.0, -1.0], [4.0, 0.5], [-3.0, 2.0], [2.0, -4.0]]),
        records,
        layer=1,
        top_k_examples=2,
        provenance={"source": "cached"},
        device="cpu",
    )

    assert [
        item["score"] for item in refreshed["components"][0]["examples"]["positive"]["occurrences"]
    ] == [4.0, 2.0]
    assert refreshed["components"][0]["examples"]["negative"]["occurrences"] == []
    assert refreshed["components"][0]["score_statistics"] == before
    assert refreshed["example_provenance"] == {"source": "cached"}
    assert all(
        occurrence["absolute_score_rank"] >= 1
        for component in refreshed["components"]
        for sign in ("positive", "negative")
        for occurrence in component["examples"][sign]["occurrences"]
    )


def test_refresh_example_ranks_enriches_existing_occurrences_without_reselection(
    monkeypatch,
) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)
    profile = lens.profile_components(["example"], layer=1, top_k_examples=2)
    occurrence = profile["components"][0]["examples"]["negative"]["occurrences"][0]
    occurrence.pop("absolute_score_rank")
    before = dict(occurrence)
    monkeypatch.setattr(lens, "transform", lambda values, *, layer: values)

    enriched = lens.refresh_profile_example_ranks_from_activations(
        torch.tensor([[1.0, 4.0], [5.0, 2.0]]),
        [(0, "negative", 0, 1)],
        layer=1,
        device="cpu",
    )

    enriched_occurrence = enriched["components"][0]["examples"]["negative"]["occurrences"][0]
    assert {key: enriched_occurrence[key] for key in before} == before
    assert enriched_occurrence["absolute_score_rank"] == 1
    assert enriched["selection"]["example_absolute_score_rank"] == (
        "competition_rank_by_absolute_score"
    )


def test_add_r_lens_profile_preserves_existing_information(monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    lens._analysis_device = "cpu"
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)
    original = lens.profile_components(["example"], layer=1, top_k_examples=2)
    original_examples = original["components"][0]["examples"]

    enriched = lens.add_r_lens_profile(
        layer=1,
        r_lens={
            "J": {1: torch.eye(2)},
            "d_model": 2,
            "provenance": {
                "model_id": "example/model",
                "model_revision": "revision",
            },
        },
        top_k=2,
        batch_size=1,
        device="cpu",
    )

    assert enriched["components"][0]["examples"] == original_examples
    assert enriched["components"][0]["r_lens"]["method"] == ("relp_then_final_norm_then_unembed")
    assert len(enriched["components"][0]["r_lens"]["dominant"]["top_tokens"]) == 2
    assert enriched["selection"]["r_lens_top_k"] == 2


def test_add_r_lens_profile_records_explicit_base_to_instruct_transfer(
    monkeypatch,
) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(
        model_id="example/model-instruct",
        model_revision="instruct-revision",
        model_type="instruct",
    ).fit(signals, layer=1, n_components=2, max_iter=2)
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    lens._analysis_device = "cpu"
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)
    lens.profile_components(["example"], layer=1, top_k_examples=2)
    r_lens = {
        "J": {1: torch.eye(2)},
        "d_model": 2,
        "provenance": {
            "model_id": "example/model-base",
            "model_revision": "base-revision",
        },
    }

    with pytest.raises(ValueError, match="allow_base_model_transfer"):
        lens.add_r_lens_profile(layer=1, r_lens=r_lens, device="cpu")

    directly_profiled = lens.profile_components(
        ["example"],
        layer=1,
        top_k_examples=2,
        r_lens=r_lens,
        r_lens_top_k=2,
        device="cpu",
        allow_base_model_transfer=True,
    )
    assert directly_profiled["r_lens_provenance"]["transfer"]["kind"] == ("base_to_instruct")

    enriched = lens.add_r_lens_profile(
        layer=1,
        r_lens=r_lens,
        allow_base_model_transfer=True,
        top_k=2,
        device="cpu",
    )
    provenance = enriched["r_lens_provenance"]
    assert provenance["model_id"] == "example/model-base"
    assert provenance["transfer"]["kind"] == "base_to_instruct"
    assert provenance["transfer"]["target_model_id"] == "example/model-instruct"
