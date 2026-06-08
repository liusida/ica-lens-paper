from __future__ import annotations

from ica_lens.artifacts import artifact_sets


def test_artifact_sets_parse() -> None:
    sets = artifact_sets()
    assert sets["models"].name == "models"
    assert sets["databases"].name == "databases"
    assert sets["models"].is_configured
    assert sets["models"].dataset_id == "sida/ica-lens-paper"
    assert sets["models"].allow_patterns == ("models/**",)
