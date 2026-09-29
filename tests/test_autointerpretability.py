from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from icalens.experiments import autointerpretability
from icalens.experiments._run import ResumableRun
from icalens.experiments.autointerpretability import (
    _evaluate_tinker_feature,
    _load_or_sample_fragments,
    _modern_expected_activations,
    _modern_pearson,
    _normalize_legacy_openwebtext,
    _tinker_expected_activations,
    _tinker_label_spans,
)
from icalens.experiments.autointerpretability_figure import (
    _bootstrap_mean,
    _merge_model_payloads,
)
from icalens.experiments.autointerpretability_figure import (
    render as render_autointerpretability,
)
from icalens.experiments.autointerpretability_protocol import (
    FRAGMENT_LENGTH,
    normalize_activations,
    parse_indexed_activations,
    pearson_correlation,
    select_record_indices,
    simulator_messages,
)


def test_normalize_activations_uses_zero_to_ten_scale() -> None:
    values = np.asarray([[-1.0, 0.0, 1.0, 2.0]])
    assert normalize_activations(values).tolist() == [[0, 0, 5, 10]]


def test_select_record_indices_is_deterministic_and_disjoint() -> None:
    values = np.zeros((40, FRAGMENT_LENGTH), dtype=np.float32)
    values[:, 0] = np.arange(1, 41)
    first = select_record_indices(values, seed=7)
    second = select_record_indices(values, seed=7)
    assert first == second
    train, top, random = first
    assert len(train) == len(top) == len(random) == 5
    assert set(train).isdisjoint(top)


def test_select_record_indices_rejects_sparse_pool() -> None:
    values = np.zeros((40, FRAGMENT_LENGTH), dtype=np.float32)
    values[:19, 0] = 1
    with pytest.raises(ValueError, match="fewer than 20 fragments activate"):
        select_record_indices(values, seed=0)


def test_simulator_prompt_has_explicit_indices() -> None:
    messages = simulator_messages("punctuation", [f"t{index}" for index in range(64)])
    assert '"index": 63' in messages[1]["content"]
    assert "string keys 0 through 63" in messages[0]["content"]


def test_parse_indexed_activations_requires_all_ordered_keys() -> None:
    payload = {"activations": {str(index): index % 11 for index in range(64)}}
    parsed = parse_indexed_activations(json.dumps(payload))
    assert parsed.shape == (64,)
    assert parsed[63] == 8

    del payload["activations"]["17"]
    with pytest.raises(ValueError, match="keys 0 through 63"):
        parse_indexed_activations(json.dumps(payload))


def test_parse_indexed_activations_rejects_non_integer_and_out_of_range() -> None:
    mapping = {str(index): 0 for index in range(64)}
    mapping["3"] = 1.5
    with pytest.raises(ValueError, match="must be integers"):
        parse_indexed_activations(json.dumps({"activations": mapping}))
    mapping["3"] = 11
    with pytest.raises(ValueError, match="range"):
        parse_indexed_activations(json.dumps({"activations": mapping}))


def test_pearson_correlation_reports_undefined_constants() -> None:
    assert pearson_correlation(np.arange(5), np.arange(5)) == pytest.approx(1.0)
    assert pearson_correlation(np.ones(5), np.arange(5)) is None


def test_tinker_evaluation_runs_simulations_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "method"
    results = directory / "results"
    results.mkdir(parents=True)
    values = np.arange(15 * FRAGMENT_LENGTH, dtype=np.float32).reshape(15, FRAGMENT_LENGTH, 1)
    np.save(directory / "candidate_activations.npy", values)
    record = {
        "feature": 7,
        "candidate_position": 0,
        "train_top": list(range(5)),
        "valid_top": list(range(5, 10)),
        "valid_random": list(range(10, 15)),
    }
    prompt_hash = "prompt-hash"
    (results / "feature_7.explanation.json").write_text(
        json.dumps(
            {
                "model": "Inkling",
                "prompt_hash": prompt_hash,
                "feature": 7,
                "explanation": "test feature",
            }
        )
    )
    lock = threading.Lock()
    active = 0
    peak = 0

    def sample(call: object, *, retries: int) -> tuple[str, np.ndarray]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return "raw", np.zeros(FRAGMENT_LENGTH, dtype=np.int64)

    class Display:
        def advance(self, count: int = 1, *, refresh: bool = False) -> None:
            return None

    monkeypatch.setattr(autointerpretability, "_retry_tinker", sample)
    _evaluate_tinker_feature(
        source_dir=directory,
        record=record,
        fragments=[{"tokens": ["x"] * FRAGMENT_LENGTH} for _ in range(15)],
        destination=results / "feature_7.json",
        explainer_model="Inkling",
        simulator_model="Qwen",
        explainer_sampler=object(),
        simulator_sampler=object(),
        explainer_tokenizer=object(),
        simulator_tokenizer=object(),
        explainer_renderer=object(),
        simulator_renderer=object(),
        renderers_module=object(),
        types=object(),
        retries=0,
        prompt_hash=prompt_hash,
        display=Display(),  # type: ignore[arg-type]
        max_concurrent=10,
    )

    assert peak > 1
    assert len(list((results / "feature_7.simulations").glob("fragment_*.json"))) == 10


def test_evaluate_rejects_nonpositive_concurrency() -> None:
    with pytest.raises(ValueError, match="--max-concurrent must be positive"):
        autointerpretability.evaluate_main(["--input", "unused", "--max-concurrent", "0"])


def test_modern_expected_activations_uses_label_logprobs() -> None:
    content = json.dumps({"activations": [0] * FRAGMENT_LENGTH}, separators=(",", ":"))
    items = []
    for character in content:
        alternatives = [SimpleNamespace(token="1", logprob=0.0)] if character == "0" else []
        items.append(SimpleNamespace(token=character, logprob=0.0, top_logprobs=alternatives))
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=SimpleNamespace(content=items),
    )

    raw, expected = _modern_expected_activations(choice)

    assert raw == content
    assert expected.tolist() == pytest.approx([0.5] * FRAGMENT_LENGTH)


def test_modern_pearson_uses_continuous_values_and_preserves_nan() -> None:
    assert _modern_pearson(np.array([0.1, 0.2, 0.4]), np.array([1.0, 2.0, 4.0])) == pytest.approx(
        1.0
    )
    assert np.isnan(_modern_pearson(np.arange(3), np.zeros(3)))


def test_tinker_label_spans_accepts_indexed_mapping_and_array() -> None:
    mapping = json.dumps(
        {"activations": {str(index): index % 11 for index in range(FRAGMENT_LENGTH)}},
        separators=(",", ":"),
    )
    array = json.dumps({"activations": [0] * FRAGMENT_LENGTH}, separators=(",", ":"))
    indexed_array = json.dumps(
        {"activations": [[index, index % 11] for index in range(FRAGMENT_LENGTH)]},
        separators=(",", ":"),
    )

    assert len(_tinker_label_spans(mapping)) == FRAGMENT_LENGTH
    assert len(_tinker_label_spans(array)) == FRAGMENT_LENGTH
    assert len(_tinker_label_spans(indexed_array)) == FRAGMENT_LENGTH


def test_tinker_expected_activations_uses_teacher_forced_logprobs() -> None:
    raw = json.dumps({"activations": [0] * FRAGMENT_LENGTH}, separators=(",", ":"))
    prompt_tokens = [999]
    completion_tokens = [ord(character) for character in raw]

    class Tokenizer:
        def decode(self, tokens: list[int], **kwargs: object) -> str:
            return "".join(chr(token) for token in tokens)

    alternatives: list[list[tuple[int, float]] | None] = [None] * (
        len(prompt_tokens) + len(completion_tokens)
    )
    for index, character in enumerate(raw):
        if character == "0":
            alternatives[len(prompt_tokens) + index] = [(ord("0"), 0.0), (ord("1"), 0.0)]

    class SampleResult:
        topk_prompt_logprobs = alternatives

    class Pending:
        def result(self) -> SampleResult:
            return SampleResult()

    class Sampler:
        def sample(self, **kwargs: object) -> Pending:
            return Pending()

    class ModelInput:
        @staticmethod
        def from_ints(tokens: list[int]) -> list[int]:
            return tokens

    types = SimpleNamespace(
        ModelInput=ModelInput,
        SamplingParams=lambda **kwargs: kwargs,
    )
    sequence = SimpleNamespace(tokens=completion_tokens)

    decoded, expected = _tinker_expected_activations(
        sampler=Sampler(),
        tokenizer=Tokenizer(),
        types=types,
        prompt_tokens=prompt_tokens,
        sequence=sequence,
    )

    assert decoded == raw
    assert expected.tolist() == pytest.approx([0.5] * FRAGMENT_LENGTH)


def test_autointerpretability_figure_plots_distribution_and_summary(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    payload = {
        "run": {
            "resolved": {
                "layers": [5],
                "methods": ["ica", "sae"],
                "preparation_resolved": {"model": {"repo_id": "openai-community/gpt2"}},
            },
            "evaluation": {
                "provider": "tinker",
                "explainer_model": "thinkingmachines/Inkling",
                "simulator_model": "Qwen/Qwen3.8-27B",
            },
        },
        "results": {
            "conditions": [
                {"layer": 5, "method": "ica", "selected_features": 3},
                {"layer": 5, "method": "sae", "selected_features": 3},
            ],
            "features": [
                {"layer": 5, "method": "ica", "combined_score": value}
                for value in [0.1, 0.2, 0.3]
            ]
            + [
                {"layer": 5, "method": "sae", "combined_score": value}
                for value in [0.4, 0.5, 0.6]
            ],
        },
    }

    outputs = render_autointerpretability(
        [payload], titles=["GPT-2 small"], output=tmp_path, formats=["png"], force=False
    )

    assert outputs == [tmp_path / "autointerpretability.png", tmp_path / "autointerpretability.txt"]
    assert outputs[0].stat().st_size > 0
    companion = outputs[1].read_text()
    assert "feature-level bootstrap" in companion
    assert "GPT-2 small\t5\tICA\t3/3" in companion
    with pytest.raises(FileExistsError, match="--force"):
        render_autointerpretability(
            [payload], titles=["GPT-2 small"], output=tmp_path, formats=["png"], force=False
        )


def test_autointerpretability_bootstrap_is_deterministic() -> None:
    scores = np.asarray([0.1, 0.2, 0.8])
    assert _bootstrap_mean(scores, 7) == _bootstrap_mean(scores, 7)


def test_autointerpretability_figure_merges_layers_for_same_model(tmp_path: Path) -> None:
    def payload(layer: int) -> dict[str, object]:
        return {
            "path": tmp_path / f"layer-{layer}",
            "run": {
                "evaluation": {"provider": "tinker", "protocol": "cunningham-modern"},
                "resolved": {
                    "layers": [layer],
                    "methods": ["ica", "sae"],
                    "preparation_resolved": {
                        "model": {"repo_id": "openai-community/gpt2"}
                    },
                },
            },
            "results": {
                "conditions": [
                    {"layer": layer, "method": method, "selected_features": 1}
                    for method in ["ica", "sae"]
                ],
                "features": [
                    {"layer": layer, "method": method, "combined_score": 0.5}
                    for method in ["ica", "sae"]
                ],
            },
        }

    merged = _merge_model_payloads([payload(2), payload(8)])  # type: ignore[list-item]

    assert len(merged) == 1
    assert merged[0]["run"]["resolved"]["layers"] == [2, 8]
    assert len(merged[0]["results"]["conditions"]) == 4


def test_run_manifest_migrates_legacy_openwebtext_alias(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    previous = {
        "status": "preparing",
        "resolved": {"dataset": {"repo_id": "openwebtext", "revision": "abc", "split": "train"}},
    }
    path.write_text(json.dumps(previous))
    resolved = {
        "dataset": {
            "repo_id": "Skylion007/openwebtext",
            "revision": "abc",
            "split": "train",
        }
    }

    resumed = ResumableRun.open(
        output=tmp_path,
        resolved=resolved,
        source={},
        status="preparing",
        normalize_previous=_normalize_legacy_openwebtext,
    )

    assert resumed.state["resolved"] == resolved


def test_run_manifest_reports_configuration_difference(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"resolved": {"n_fragments": 1000}}))

    with pytest.raises(ValueError, match=r"resolved\.n_fragments: 1000 -> 50000"):
        ResumableRun.open(
            output=tmp_path,
            resolved={"n_fragments": 50_000},
            source={},
            status="preparing",
        )


def test_fragment_sampling_silences_full_document_length_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
            calls.append(kwargs)
            return {"input_ids": list(range(1217))}

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [str(value) for value in ids]

    monkeypatch.setattr(
        autointerpretability,
        "load_dataset",
        lambda *args, **kwargs: iter([{"text": "long document"}]),
    )

    rows = _load_or_sample_fragments(
        output=tmp_path,
        tokenizer=Tokenizer(),
        dataset="owner/dataset",
        revision="abc",
        split="train",
        count=1,
        seed=0,
    )

    assert calls == [{"add_special_tokens": False, "verbose": False}]
    assert len(rows[0]["token_ids"]) == FRAGMENT_LENGTH


def test_fragment_sampling_resumes_from_durable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = [{"text": f"document {index}"} for index in range(3)]
    available = documents[:2]

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
            offset = int(text.rsplit(" ", 1)[1]) * 2000
            return {"input_ids": list(range(offset, offset + 128))}

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [str(value) for value in ids]

    monkeypatch.setattr(
        autointerpretability,
        "load_dataset",
        lambda *args, **kwargs: iter(available),
    )
    checkpoints: list[int] = []
    with pytest.raises(RuntimeError, match="after 2 usable fragments"):
        _load_or_sample_fragments(
            output=tmp_path,
            tokenizer=Tokenizer(),
            dataset="owner/dataset",
            revision="abc",
            split="train",
            count=3,
            seed=7,
            on_checkpoint=lambda: checkpoints.append(1),
        )
    assert len((tmp_path / "fragments.partial.jsonl").read_text().splitlines()) == 2

    available = documents
    rows = _load_or_sample_fragments(
        output=tmp_path,
        tokenizer=Tokenizer(),
        dataset="owner/dataset",
        revision="abc",
        split="train",
        count=3,
        seed=7,
        on_checkpoint=lambda: checkpoints.append(1),
    )

    assert [row["document_index"] for row in rows] == [0, 1, 2]
    assert len(checkpoints) == 3
    assert (tmp_path / "fragments.jsonl").is_file()
    assert not (tmp_path / "fragments.partial.jsonl").exists()
