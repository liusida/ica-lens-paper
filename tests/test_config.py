from __future__ import annotations

from pathlib import Path

from ica_lens.config import load_json, load_toml


def test_artifact_manifest_loads() -> None:
    manifest = load_json(Path("artifacts/manifest.json"))
    assert manifest["schema_version"] == 1
    assert set(manifest["artifact_sets"]) == {"models", "databases"}


def test_demo_config_loads() -> None:
    config = load_toml(Path("configs/demo/mini_3k.toml"))
    assert config["demo"]["model"] == "all"
    assert config["demo"]["token_budget"] == 3000
    assert config["demo"]["analysis"] == ["fit_ica", "sparse_probe", "saebench_tpp"]
