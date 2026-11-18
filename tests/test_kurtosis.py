from __future__ import annotations

import numpy as np
import pytest
import torch

from icalens import ICALens
from icalens.kurtosis import (
    excess_from_moments,
    excess_kurtosis,
    lens_excess_kurtosis,
    merge_moments,
)


def _reference(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    return (centered**4).mean(axis=0) / (centered**2).mean(axis=0) ** 2 - 3.0


def _laplace_mixture(seed: int, samples: int, width: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mixing = torch.randn(width, width, generator=torch.Generator().manual_seed(123))
    uniform = torch.rand(samples, width, generator=generator) - 0.5
    sources = -torch.sign(uniform) * torch.log1p(-2 * uniform.abs())
    return (sources @ mixing.T + 2.0).float(), mixing


def test_streaming_moments_match_reference_and_are_batch_invariant() -> None:
    values = torch.randn(5000, 4, dtype=torch.float64).pow(3)
    expected = _reference(values.numpy())
    one = excess_from_moments(merge_moments(None, values)).numpy()
    state = None
    for batch in values.split(333):
        state = merge_moments(state, batch)
    assert state is not None
    np.testing.assert_allclose(one, expected, rtol=1e-10)
    np.testing.assert_allclose(excess_from_moments(state).numpy(), expected, rtol=1e-10)


def test_excess_kurtosis_is_invariant_to_direction_scale() -> None:
    values = torch.randn(4000, 5).pow(3)
    reading = np.random.default_rng(0).normal(size=(5, 5))
    scaled = reading * np.array([[0.1], [2.0], [5.0], [1.0], [30.0]])
    np.testing.assert_allclose(
        excess_kurtosis(values, reading), excess_kurtosis(values, scaled), rtol=1e-5
    )


def test_fit_records_train_and_heldout_kurtosis_history() -> None:
    train, _ = _laplace_mixture(0, 6000)
    heldout, _ = _laplace_mixture(1, 3000)
    lens = ICALens(model_id="m", model_revision="r", icalens_preprocessing="none")
    lens.fit(
        train,
        layer=0,
        max_iter=25,
        batch_size=1000,
        heldout=heldout,
        heldout_every=10,
        heldout_provenance={"dataset": "synthetic"},
    )
    fitting = lens._get_layer(0).fitting
    history = fitting["kurtosis_history"]
    assert history["train"]["iterations"] == list(range(26))
    assert history["heldout"]["iterations"] == [0, 10, 20, 25]
    assert history["heldout"]["provenance"] == {"dataset": "synthetic"}
    assert history["heldout"]["n_samples"] == 3000

    final = fitting["component_excess_kurtosis"]
    np.testing.assert_allclose(
        final["train"], lens_excess_kurtosis(lens, train, layer=0), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        final["heldout"], lens_excess_kurtosis(lens, heldout, layer=0), rtol=1e-4, atol=1e-5
    )
    assert history["train"]["mean"][-1] == pytest.approx(np.mean(final["train"]), rel=1e-6)
    assert history["heldout"]["mean"][-1] == pytest.approx(np.mean(final["heldout"]), rel=1e-6)
    # Laplace sources have excess kurtosis 3; ICA should recover them.
    assert np.mean(final["train"]) == pytest.approx(3.0, abs=0.5)


def test_fit_without_heldout_records_only_train(tmp_path) -> None:
    train, _ = _laplace_mixture(2, 3000)
    lens = ICALens(model_id="m", model_revision="r", icalens_preprocessing="l2")
    lens.fit(train, layer=1, max_iter=4, batch_size=512, objective_every=2)
    fitting = lens._get_layer(1).fitting
    assert "heldout" not in fitting["kurtosis_history"]
    assert fitting["kurtosis_history"]["train"]["iterations"] == [0, 2, 4]
    # The standalone estimator applies the lens's own (L2) preprocessing.
    np.testing.assert_allclose(
        fitting["component_excess_kurtosis"]["train"],
        lens_excess_kurtosis(lens, train, layer=1),
        rtol=1e-4,
        atol=1e-5,
    )
    saved = ICALens.from_pretrained(lens.save(tmp_path / "lens"))
    assert saved._get_layer(1).fitting["kurtosis_history"] == fitting["kurtosis_history"]


def test_fit_rejects_heldout_with_wrong_width() -> None:
    train, _ = _laplace_mixture(3, 2000)
    lens = ICALens(model_id="m", model_revision="r", icalens_preprocessing="none")
    with pytest.raises(ValueError, match="hidden size"):
        lens.fit(train, layer=0, max_iter=2, heldout=torch.randn(100, 5))
