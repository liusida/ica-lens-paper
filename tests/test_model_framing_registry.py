from __future__ import annotations

import json
from pathlib import Path

import pytest

from icalens import _model_framing
from icalens._model_framing import resolve_framing_policy

REGISTRY = Path(__file__).parents[1] / "model_framing.json"


def test_model_framing_registry_is_internally_consistent() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["schema_version"] == 1
    assert registry["policy_scope"] == "independent_pretraining_document"
    models = registry["models"]
    assert models
    for model_id, policy in models.items():
        assert "/" in model_id
        strategy = policy["document_framing"]
        assert strategy in {"prepend-bos", "prepend-eos"}
        assert policy["token_role"] == strategy.removeprefix("prepend-")
        assert policy["expected_token"]
        assert policy["evidence_url"].startswith("https://")
        assert policy["evidence_note"]


def test_registry_contains_verified_project_models() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    models = registry["models"]

    assert models["EleutherAI/pythia-70m"]["document_framing"] == "prepend-eos"
    assert models["openai-community/gpt2"]["document_framing"] == "prepend-eos"
    assert models["google/gemma-2-2b"]["document_framing"] == "prepend-bos"
    assert models["Qwen/Qwen3.5-2B-Base"]["document_framing"] == "prepend-eos"
    assert models["Qwen/Qwen3.5-9B-Base"]["document_framing"] == "prepend-eos"
    assert models["Qwen/Qwen3.5-35B-A3B-Base"]["document_framing"] == "prepend-eos"


def test_exact_bundled_policy_records_content_hash() -> None:
    policy = resolve_framing_policy("openai-community/gpt2")

    assert policy.entry["expected_token"] == "<|endoftext|>"
    assert len(policy.sha256) == 64
    assert policy.schema_version == 1


def test_pythia_policy_uses_its_eod_token_as_document_prefix() -> None:
    policy = resolve_framing_policy("EleutherAI/pythia-70m")

    assert policy.entry["document_framing"] == "prepend-eos"
    assert policy.entry["expected_token"] == "<|endoftext|>"


def test_missing_model_can_be_resolved_and_cached_from_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry["models"]["owner/new-model"] = {
        "document_framing": "prepend-eos",
        "token_role": "eos",
        "expected_token": "<eos>",
        "evidence_url": "https://example.com/official-model-documentation",
        "evidence_note": "Official test policy.",
    }
    raw = json.dumps(registry).encode()
    monkeypatch.setattr(
        _model_framing,
        "_download_registry",
        lambda: (_model_framing._parse_registry(raw), raw),
    )

    policy = resolve_framing_policy("owner/new-model", cache_dir=tmp_path)

    assert policy.entry["expected_token"] == "<eos>"
    assert (tmp_path / "model_framing.json").read_bytes() == raw


def test_refresh_uses_remote_registry_for_existing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = REGISTRY.read_bytes()
    monkeypatch.setattr(
        _model_framing,
        "_download_registry",
        lambda: (_model_framing._parse_registry(raw), raw),
    )

    policy = resolve_framing_policy(
        "openai-community/gpt2", refresh=True, cache_dir=tmp_path
    )

    assert policy.source == _model_framing.REGISTRY_URL
    assert (tmp_path / "model_framing.json").is_file()
