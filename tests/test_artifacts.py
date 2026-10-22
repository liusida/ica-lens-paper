from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from icalens import ArtifactError, ICALens


def make_lens() -> ICALens:
    return ICALens(
        model_id="example/model",
        model_revision="abc123",
        activation_site="resid_post",
    )


def test_save_and_local_load_round_trip(tmp_path, mixed_signals: np.ndarray) -> None:
    original = make_lens().fit(mixed_signals, layer=6, n_components=2, max_iter=500)
    expected = original.transform(mixed_signals[:10], layer=6)

    artifact_dir = original.save(tmp_path / "artifact")
    loaded = ICALens.from_pretrained(artifact_dir)

    assert (artifact_dir / "icalens.json").is_file()
    assert (artifact_dir / "README.md").is_file()
    assert (artifact_dir / "artifacts/resid_post/layer_06.safetensors").is_file()
    assert loaded.available_layers == (6,)
    assert loaded.metadata["model"]["revision"] == "abc123"
    assert loaded.metadata["model"]["type"] == "base"
    model_card = (artifact_dir / "README.md").read_text(encoding="utf-8")
    front_matter = model_card.split("---", maxsplit=2)[1]
    assert "base_model:" not in front_matter
    assert "layer-wise ICA transformations" in model_card
    np.testing.assert_allclose(loaded.transform(mixed_signals[:10], layer=6), expected)


def test_save_replaces_existing_artifact(tmp_path, mixed_signals: np.ndarray) -> None:
    target = tmp_path / "artifact"
    make_lens().fit(mixed_signals, layer=1, n_components=2).save(target)
    make_lens().fit(mixed_signals, layer=3, n_components=2).save(target)
    loaded = ICALens.from_pretrained(target)
    assert loaded.available_layers == (3,)


def test_model_card_promotes_common_fitting_dataset(tmp_path, mixed_signals: np.ndarray) -> None:
    provenance = {
        "dataset": {
            "repo_id": "example/fitting-data",
            "revision": "data123",
            "split": "train",
        },
        "token_scope": "all",
        "candidate_tokens": 1000,
        "fitting_tokens": 800,
    }
    artifact_dir = (
        make_lens()
        .fit(mixed_signals, layer=6, n_components=2, provenance=provenance)
        .save(tmp_path / "artifact")
    )
    model_card = (artifact_dir / "README.md").read_text(encoding="utf-8")
    front_matter = model_card.split("---", maxsplit=2)[1]

    assert "datasets:\n- example/fitting-data" in front_matter
    dataset_link = "[example/fitting-data](https://huggingface.co/datasets/example/fitting-data)"
    assert dataset_link in model_card
    assert "| Dataset revision | `data123` |" in model_card
    assert "| Fitting tokens | 800 |" in model_card


def test_rejects_unsupported_format_version(tmp_path) -> None:
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "icalens.json").write_text(
        json.dumps({"format": "icalens", "format_version": 999}), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="format version"):
        ICALens.from_pretrained(target)


def test_v4_manifest_records_minimum_package_version(
    tmp_path, mixed_signals: np.ndarray
) -> None:
    artifact_dir = (
        make_lens().fit(mixed_signals, layer=6, n_components=2).save(tmp_path / "artifact")
    )
    manifest = json.loads((artifact_dir / "icalens.json").read_text(encoding="utf-8"))

    assert manifest["format_version"] == 4
    assert manifest["minimum_package_version"] == "0.3.4"


def test_rejects_v4_manifest_without_minimum_package_version(
    tmp_path, mixed_signals: np.ndarray
) -> None:
    artifact_dir = (
        make_lens().fit(mixed_signals, layer=6, n_components=2).save(tmp_path / "artifact")
    )
    manifest_path = artifact_dir / "icalens.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["minimum_package_version"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="minimum_package_version"):
        ICALens.from_pretrained(artifact_dir)


def test_loads_v1_base_model_manifest(tmp_path, mixed_signals: np.ndarray) -> None:
    artifact_dir = (
        make_lens().fit(mixed_signals, layer=6, n_components=2).save(tmp_path / "artifact")
    )
    manifest_path = artifact_dir / "icalens.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    manifest["base_model"] = {
        "repo_id": manifest["model"]["repo_id"],
        "revision": manifest["model"]["revision"],
    }
    del manifest["model"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = ICALens.from_pretrained(artifact_dir)
    assert loaded.model_id == "example/model"
    assert loaded.model_revision == "abc123"
    assert loaded.model_type == "base"


def test_manifest_does_not_expose_mutable_internal_state(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=1, n_components=2)
    metadata = lens.metadata
    metadata["layers"].clear()
    assert lens.available_layers == (1,)


def _save_multilayer_artifact(
    path: Path, mixed_signals: np.ndarray
) -> tuple[Path, np.ndarray]:
    original = make_lens()
    for layer in (1, 2):
        original.fit(mixed_signals[:32], layer=layer, n_components=2, max_iter=2)
    expected = original.transform(mixed_signals[:10], layer=2)
    artifact_dir = original.save(path)

    # A portable profile fixture avoids loading any language model to build it.
    profile_file = "artifacts/resid_post/layer_02.profile.json"
    profile = {
        "components": [
            {"component": 0, "label": "preserved component"},
            {"component": 1, "label": "other component"},
        ]
    }
    (artifact_dir / profile_file).write_text(json.dumps(profile), encoding="utf-8")
    manifest_path = artifact_dir / "icalens.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"]["2"]["component_profile"] = profile_file
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return artifact_dir, expected


def _load_artifact_source(
    artifact_dir: Path, source: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[ICALens, list[str]]:
    downloads: list[str] = []
    if source == "local":
        return ICALens.from_pretrained(artifact_dir), downloads

    def download(*, filename: str, **kwargs: object) -> str:
        assert kwargs["repo_id"] == "example/lens"
        downloads.append(filename)
        return str(artifact_dir / filename)

    monkeypatch.setattr("icalens.lens.hf_hub_download", download)
    return ICALens.from_pretrained("example/lens", revision="abc123"), downloads


@pytest.mark.parametrize("source", ["local", "hub"])
@pytest.mark.parametrize("fitted_layer", [1, 3], ids=["replace-layer", "add-layer"])
def test_fit_preserves_lazy_layers_and_profiles(
    tmp_path,
    mixed_signals: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    fitted_layer: int,
) -> None:
    artifact_dir, expected = _save_multilayer_artifact(tmp_path / "source", mixed_signals)
    loaded, downloads = _load_artifact_source(artifact_dir, source, monkeypatch)

    loaded.fit(mixed_signals[32:64], layer=fitted_layer, n_components=2, max_iter=2)

    assert not loaded._layers[2].loaded
    assert loaded._layers[2].profile is None
    if source == "hub":
        assert downloads == ["icalens.json"]
    np.testing.assert_allclose(loaded.transform(mixed_signals[:10], layer=2), expected)
    assert loaded.component_profile(layer=2, component=0)["label"] == "preserved component"

    reloaded = ICALens.from_pretrained(loaded.save(tmp_path / "updated"))
    assert reloaded.available_layers == ((1, 2) if fitted_layer == 1 else (1, 2, 3))
    np.testing.assert_allclose(reloaded.transform(mixed_signals[:10], layer=2), expected)
    np.testing.assert_allclose(
        reloaded.transform(mixed_signals[:10], layer=fitted_layer),
        loaded.transform(mixed_signals[:10], layer=fitted_layer),
    )
    assert reloaded.component_profile(layer=2, component=0)["label"] == "preserved component"


@pytest.mark.parametrize("source", ["local", "hub"])
@pytest.mark.parametrize("fitted_layer", [1, 3], ids=["replace-layer", "add-layer"])
def test_fit_preserves_lazy_profile_after_loading_layer_tensors(
    tmp_path,
    mixed_signals: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    fitted_layer: int,
) -> None:
    artifact_dir, expected = _save_multilayer_artifact(tmp_path / "source", mixed_signals)
    loaded, downloads = _load_artifact_source(artifact_dir, source, monkeypatch)
    np.testing.assert_allclose(loaded.transform(mixed_signals[:10], layer=2), expected)
    assert loaded._layers[2].profile is None

    loaded.fit(mixed_signals[32:64], layer=fitted_layer, n_components=2, max_iter=2)

    assert loaded._layers[2].profile is None
    if source == "hub":
        assert downloads == ["icalens.json", "artifacts/resid_post/layer_02.safetensors"]
    # Saving must still be able to load the profile even though its tensors were cached.
    reloaded = ICALens.from_pretrained(loaded.save(tmp_path / "updated"))
    np.testing.assert_allclose(reloaded.transform(mixed_signals[:10], layer=2), expected)
    assert reloaded.component_profile(layer=2, component=0)["label"] == "preserved component"
