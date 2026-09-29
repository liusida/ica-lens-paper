from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from icalens.cli.profile import (
    _pending_example_layers,
    _pending_statistics_layers,
    _replay_dataset_source,
    parse_args,
)


def test_parses_statistics_refresh_operation() -> None:
    args = parse_args(
        [
            "refresh-statistics",
            "--lens",
            "artifact",
            "--layers",
            "all",
            "--activations",
            "capture",
            "--max-tokens",
            "1000000",
        ]
    )

    assert args.operation == "refresh-statistics"
    assert args.activations.name == "capture"
    assert args.max_tokens == 1_000_000


def test_parses_example_refresh_operation() -> None:
    args = parse_args(
        ["refresh-examples", "--lens", "artifact", "--layers", "all", "--activations", "capture"]
    )
    assert args.operation == "refresh-examples"


def test_parses_example_rank_refresh_operation() -> None:
    args = parse_args(
        [
            "refresh-examples-rank",
            "--lens",
            "artifact",
            "--layers",
            "all",
            "--activations",
            "capture",
        ]
    )
    assert args.operation == "refresh-examples-rank"


def test_example_refresh_treats_legacy_examples_as_pending() -> None:
    profile = {
        "selection": {"sign_selection": "population_skewness"},
        "components": [{"tail_direction": "negative"}],
    }
    artifact = SimpleNamespace(profile_file="profiles/0.json.gz")
    lens = SimpleNamespace(_layers={0: artifact}, _get_profile=lambda value: profile)
    pending, completed = _pending_example_layers(
        SimpleNamespace(force=False, top_k_examples=20), lens, (0,), provenance={"source": "new"}
    )
    assert pending == (0,)
    assert completed == ()


def test_replay_dataset_source_accepts_matching_local_file(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"alpha"}\n')
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()

    assert _replay_dataset_source({"path": str(dataset), "sha256": digest, "split": "train"}) == (
        str(dataset.resolve()),
        None,
    )


def test_replay_dataset_source_rejects_changed_local_file(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"changed"}\n')

    with pytest.raises(ValueError, match="checksum differs"):
        _replay_dataset_source({"path": str(dataset), "sha256": "0" * 64, "split": "train"})


def test_statistics_refresh_resumes_only_matching_durable_layers() -> None:
    provenance = {"activation_dataset": {"manifest_sha256": "abc"}}
    legacy = {
        "selection": {"score_statistics": "population_excess_kurtosis"},
    }
    refreshed = {
        "selection": {
            "score_statistics": "population_mean_variance_skewness_excess_kurtosis",
            "sign_selection": "population_skewness",
        },
        "score_statistics_provenance": provenance,
    }
    lens = SimpleNamespace(
        _layers={
            0: SimpleNamespace(profile_file="profiles/0.json.gz"),
            1: SimpleNamespace(profile_file="profiles/1.json.gz"),
        },
        _get_profile=lambda artifact: legacy if artifact is lens._layers[0] else refreshed,
    )

    pending, completed = _pending_statistics_layers(
        SimpleNamespace(force=False),
        lens,
        (0, 1),
        provenance=provenance,
    )

    assert pending == (0,)
    assert completed == (1,)


def test_statistics_refresh_rejects_mixed_provenance_before_work() -> None:
    profile = {
        "selection": {
            "score_statistics": "population_mean_variance_skewness_excess_kurtosis",
            "sign_selection": "population_skewness",
        },
        "score_statistics_provenance": {"activation_dataset": "old"},
    }
    artifact = SimpleNamespace(profile_file="profiles/0.json.gz")
    lens = SimpleNamespace(_layers={0: artifact}, _get_profile=lambda value: profile)

    with pytest.raises(ValueError, match="statistics provenance differs"):
        _pending_statistics_layers(
            SimpleNamespace(force=False),
            lens,
            (0,),
            provenance={"activation_dataset": "new"},
        )


def test_statistics_refresh_force_replaces_mixed_provenance() -> None:
    profile = {
        "selection": {
            "score_statistics": "population_mean_variance_skewness_excess_kurtosis",
            "sign_selection": "population_skewness",
        },
        "score_statistics_provenance": {"activation_dataset": "old"},
    }
    artifact = SimpleNamespace(profile_file="profiles/0.json.gz")
    lens = SimpleNamespace(_layers={0: artifact}, _get_profile=lambda value: profile)

    pending, completed = _pending_statistics_layers(
        SimpleNamespace(force=True),
        lens,
        (0,),
        provenance={"activation_dataset": "new"},
    )

    assert pending == (0,)
    assert completed == ()
