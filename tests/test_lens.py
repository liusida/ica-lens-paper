from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from icalens import ICALens, NotFittedError


def make_lens() -> ICALens:
    return ICALens(
        model_id="example/model",
        model_revision="abc123",
        activation_site="resid_post",
    )


def test_instruct_model_metadata() -> None:
    lens = ICALens(
        model_id="example/model-instruct",
        model_revision="def456",
        model_type="instruct",
    )
    assert lens.model_id == "example/model-instruct"
    assert lens.model_revision == "def456"
    assert lens.model_type == "instruct"
    assert lens.metadata["model"]["type"] == "instruct"


def test_model_revision_is_optional() -> None:
    lens = ICALens(model_id="example/model")
    assert lens.model_revision is None
    assert lens.metadata["model"]["revision"] is None


def test_legacy_constructor_names_remain_compatible() -> None:
    lens = ICALens(base_model="example/model", base_model_revision="abc123")
    assert lens.model_id == "example/model"
    assert lens.base_model == lens.model_id
    assert lens.base_model_revision == lens.model_revision


def test_rejects_invalid_or_conflicting_model_arguments() -> None:
    with pytest.raises(ValueError, match="model_type"):
        ICALens(model_id="example/model", model_revision="abc123", model_type="chat")
    with pytest.raises(ValueError, match="not both"):
        ICALens(
            model_id="example/model",
            base_model="legacy/model",
            model_revision="abc123",
        )


def test_fit_transform_and_inverse(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=6, random_state=3, max_iter=500)

    scores = lens.transform(mixed_signals, layer=6)
    reconstructed = lens.inverse_transform(scores, layer=6)
    normalized = mixed_signals / np.maximum(
        np.linalg.norm(mixed_signals, axis=-1, keepdims=True), lens.norm_eps
    )

    assert lens.available_layers == (6,)
    assert lens.hidden_size == 3
    assert scores.shape == mixed_signals.shape
    np.testing.assert_allclose(reconstructed, normalized, atol=2e-5)


def test_geometric_median_preprocessing_round_trip(
    mixed_signals: np.ndarray, tmp_path: Path
) -> None:
    shifted = mixed_signals + np.array([8.0, -3.0, 5.0], dtype=np.float32)
    lens = ICALens(
        model_id="example/model",
        icalens_preprocessing="geometric-median-l2",
    ).fit(shifted, layer=1, random_state=3, max_iter=50)
    artifact = lens._get_layer(1)
    assert artifact.preprocessing_center is not None

    reconstructed = lens.inverse_transform(lens.transform(shifted, layer=1), layer=1)
    centered = shifted - artifact.preprocessing_center
    expected = centered / np.maximum(
        np.linalg.norm(centered, axis=-1, keepdims=True), lens.norm_eps
    )
    np.testing.assert_allclose(reconstructed, expected, atol=3e-5)

    restored = ICALens.from_pretrained(lens.save(tmp_path / "robust-lens"))
    assert restored.icalens_preprocessing == "geometric-median-l2"
    restored_artifact = restored._get_layer(1)
    assert restored_artifact.preprocessing_center is not None
    np.testing.assert_allclose(
        restored_artifact.preprocessing_center,
        artifact.preprocessing_center,
    )


def test_icalens_preprocessing_compatibility_arguments() -> None:
    assert ICALens(model_id="example/model").icalens_preprocessing == "l2"
    assert (
        ICALens(model_id="example/model", row_normalize=False).icalens_preprocessing
        == "none"
    )
    with pytest.raises(ValueError, match="conflicts"):
        ICALens(
            model_id="example/model",
            row_normalize=False,
            icalens_preprocessing="l2",
        )


def test_leading_dimensions_are_preserved(mixed_signals: np.ndarray) -> None:
    values = mixed_signals[:30].reshape(3, 10, 3)
    lens = make_lens().fit(mixed_signals, layer=2, n_components=2)
    assert lens.transform(values, layer=2).shape == (3, 10, 2)


def test_energy_is_per_position_fraction(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=2)
    scores = lens.transform(mixed_signals[:10], layer=2)
    energy = lens.energy(scores)
    np.testing.assert_allclose(energy.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(energy, scores**2 / np.sum(scores**2, axis=-1, keepdims=True))


def test_keep_and_ablate_topk_scores_per_vector() -> None:
    scores = np.array([[1.0, -4.0, 2.0], [-5.0, 2.0, 3.0]], dtype=np.float32)

    kept = ICALens.keep_topk(scores, k=2)
    ablated = ICALens.ablate_topk(scores, k=2)

    np.testing.assert_array_equal(kept, [[0.0, -4.0, 2.0], [-5.0, 0.0, 3.0]])
    np.testing.assert_array_equal(ablated, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_array_equal(kept + ablated, scores)
    assert not np.shares_memory(kept, scores)
    assert not np.shares_memory(ablated, scores)


def test_topk_score_operations_preserve_torch_properties() -> None:
    scores = torch.tensor([[1.0, -4.0, 2.0]], device="cpu", dtype=torch.float64)

    kept = ICALens.keep_topk(scores, k=1)
    ablated = ICALens.ablate_topk(scores, k=1)

    torch.testing.assert_close(kept, torch.tensor([[0.0, -4.0, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(ablated, torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64))
    assert kept.dtype == scores.dtype
    assert kept.device == scores.device


def test_topk_score_operations_validate_k() -> None:
    scores = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="between 1 and 3"):
        ICALens.keep_topk(scores, k=0)
    with pytest.raises(ValueError, match="between 1 and 3"):
        ICALens.ablate_topk(scores, k=4)


def test_restore_norm_uses_reference_norms() -> None:
    lens = make_lens()
    values = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    reference = np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    restored = lens.restore_norm(values, reference=reference)

    np.testing.assert_allclose(restored, [[1.2, 1.6], [0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(restored[0]), np.linalg.norm(reference[0]))
    assert restored.dtype == values.dtype


def test_restore_norm_preserves_torch_properties() -> None:
    lens = make_lens()
    values = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    reference = torch.tensor([[0.0, 2.0]], dtype=torch.float32)

    restored = lens.restore_norm(values, reference=reference)

    torch.testing.assert_close(restored, torch.tensor([[1.2, 1.6]], dtype=torch.float64))
    assert restored.dtype == values.dtype
    assert restored.device == values.device


def test_restore_norm_validates_inputs() -> None:
    lens = make_lens()
    values = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        lens.restore_norm(values, reference=np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="both be"):
        lens.restore_norm(values, reference=torch.ones((2, 3)))


def test_fit_records_detached_json_provenance(mixed_signals: np.ndarray) -> None:
    provenance = {"dataset": {"repo_id": "example/data", "revision": "abc"}}
    lens = make_lens().fit(mixed_signals, layer=2, provenance=provenance)
    provenance["dataset"]["repo_id"] = "changed"
    fitting = lens.metadata["layers"]["2"]["fitting"]
    assert fitting["source_scaling"] == "none"
    assert fitting["provenance"]["dataset"]["repo_id"] == "example/data"


def test_components_are_ranked_by_final_non_gaussianity(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=2, n_components=3, max_iter=20)
    fitting = lens.metadata["layers"]["2"]["fitting"]

    objectives = np.asarray(fitting["component_objectives"])
    strengths = np.asarray(fitting["component_strengths"])
    baseline = float(fitting["gaussian_objective"])

    assert fitting["implementation_version"] == "2"
    assert fitting["component_id_convention"].startswith(
        "descending absolute contrast deviation from Gaussian"
    )
    np.testing.assert_allclose(strengths, np.abs(objectives - baseline), atol=1e-7)
    assert np.all(strengths[:-1] >= strengths[1:])


def test_fit_records_objective_percentile_history(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(
        mixed_signals,
        layer=2,
        n_components=3,
        max_iter=8,
        batch_size=37,
        objective_every=3,
    )

    history = lens.metadata["layers"]["2"]["fitting"]["objective_history"]
    assert history["contrast"] == "logcosh"
    assert history["iterations"] == [3, 6, 8]
    assert history["percentiles"] == list(range(0, 101, 10))
    assert len(history["values"]) == 3
    assert all(len(row) == 11 for row in history["values"])
    assert all(row == sorted(row) for row in history["values"])
    assert any(row[0] < row[-1] for row in history["values"])


def test_plot_fitting_curve_returns_matplotlib_figure(mixed_signals: np.ndarray) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    lens = make_lens().fit(
        mixed_signals,
        layer=2,
        n_components=3,
        max_iter=4,
        objective_every=1,
    )

    figure = lens.plot_fitting_curve(layer=2)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_xlabel() == "Iteration"
    assert "layer 2" in figure.axes[0].get_title()
    assert not plt.fignum_exists(figure.number)
    plt.close(figure)


def test_plot_fitting_curve_requires_available_layer() -> None:
    with pytest.raises(NotFittedError, match=r"layers \[2\] are unavailable"):
        make_lens().plot_fitting_curve(layer=2)


def test_plot_fitting_curve_supports_multiple_and_all_layers(
    mixed_signals: np.ndarray,
) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    lens = make_lens()
    for layer in (1, 2, 3):
        lens.fit(mixed_signals, layer=layer, n_components=3, max_iter=2)

    selected = lens.plot_fitting_curve(layers=[1, 3], columns=2)
    all_layers = lens.plot_fitting_curve(layers="all", columns=3)

    assert [axis.get_title() for axis in selected.axes] == ["Layer 1", "Layer 3"]
    assert len(all_layers.axes) == 3
    assert not plt.fignum_exists(selected.number)
    assert not plt.fignum_exists(all_layers.number)


def test_render_fitting_summary_writes_png_and_pdf(
    mixed_signals: np.ndarray, tmp_path: Path
) -> None:
    from icalens.cli.plot_fitting import render_fitting_summary

    lens = make_lens()
    for layer in (1, 2):
        lens.fit(mixed_signals, layer=layer, n_components=3, max_iter=2)

    outputs = render_fitting_summary(
        [lens], titles=["Example"], output=tmp_path / "figures", force=False
    )

    assert [path.name for path in outputs] == ["fitting-curves.png", "fitting-curves.pdf"]
    assert all(path.is_file() for path in outputs)
    with pytest.raises(FileExistsError, match="--force"):
        render_fitting_summary(
            [lens], titles=["Example"], output=tmp_path / "figures", force=False
        )


@pytest.mark.parametrize("columns", [0, -1])
def test_plot_fitting_curve_rejects_nonpositive_columns(columns: int) -> None:
    with pytest.raises(ValueError, match="columns must be positive"):
        make_lens().plot_fitting_curve(layers="all", columns=columns)


def test_torch_transform_preserves_tensor_properties(mixed_signals: np.ndarray) -> None:
    values = torch.from_numpy(mixed_signals)
    lens = make_lens().fit(values, layer=2, n_components=2)
    scores = lens.transform(values[:10], layer=2)
    reconstructed = lens.inverse_transform(scores, layer=2)
    assert isinstance(scores, torch.Tensor)
    assert scores.dtype == values.dtype
    assert scores.device == values.device
    assert reconstructed.shape == (10, 3)


def test_deflation_algorithm(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(
        mixed_signals, layer=2, n_components=2, algorithm="deflation", max_iter=500
    )
    assert lens.transform(mixed_signals[:10], layer=2).shape == (10, 2)


def test_blockwise_fit_matches_single_batch(mixed_signals: np.ndarray) -> None:
    single_batch = make_lens().fit(
        mixed_signals,
        layer=2,
        n_components=3,
        max_iter=5,
        batch_size=len(mixed_signals),
    )
    blockwise = make_lens().fit(
        mixed_signals,
        layer=2,
        n_components=3,
        max_iter=5,
        batch_size=37,
    )
    np.testing.assert_allclose(
        blockwise.transform(mixed_signals, layer=2),
        single_batch.transform(mixed_signals, layer=2),
        atol=1e-5,
        rtol=1e-5,
    )
    fitting = blockwise.metadata["layers"]["2"]["fitting"]
    assert fitting["batch_size"] == 37
    assert fitting["memory_strategy"] == "blockwise_multi_pass"


def test_fit_accepts_compact_bfloat16_storage(mixed_signals: np.ndarray) -> None:
    values = torch.from_numpy(mixed_signals).to(torch.bfloat16)
    lens = make_lens().fit(values, layer=2, n_components=3, max_iter=2, batch_size=37)

    fitting = lens.metadata["layers"]["2"]["fitting"]
    assert values.dtype == torch.bfloat16
    assert fitting["input_dtype"] == "bfloat16"
    assert lens.transform(values[:10], layer=2).dtype == torch.bfloat16


def test_repeated_fit_builds_collection(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=1, n_components=2)
    returned = lens.fit(mixed_signals, layer=4, n_components=3)
    assert returned is lens
    assert lens.available_layers == (1, 4)


def test_unknown_layer_and_shape_are_clear(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=1)
    with pytest.raises(NotFittedError, match="available layers: 1"):
        lens.transform(mixed_signals, layer=2)
    with pytest.raises(ValueError, match="expected 3"):
        lens.transform(np.ones((4, 2), dtype=np.float32), layer=1)


def test_empty_lens_cannot_be_saved(tmp_path) -> None:
    with pytest.raises(NotFittedError):
        make_lens().save(tmp_path / "artifact")


def test_full_components_require_one_more_sample_than_features() -> None:
    values = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="centering limits the data rank"):
        make_lens().fit(values, layer=1, n_components=3)
