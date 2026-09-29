from __future__ import annotations

from typing import Any

import pytest
import torch

from icalens import ICALens
from icalens.erf import ERFAnalysis


def test_lens_exposes_cached_erf_namespace() -> None:
    lens = ICALens(model_id="example/model")

    assert isinstance(lens.erf, ERFAnalysis)
    assert lens.erf is lens.erf


def test_suffix_sweep_uses_stored_profile_and_returns_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import icalens.analysis as analysis
    import icalens.experiments.erf_suffix_sweep as sweep

    class FakeLens:
        model_id = "example/model"
        model_revision = "revision"
        _analysis_device = "cpu"

        def component_profile(self, *, layer: int, component: int) -> dict[str, Any]:
            assert (layer, component) == (6, 214)
            return {
                "component": component,
                "tail_direction": "positive",
                "examples": {
                    "positive": {
                        "occurrences": [
                            {"absolute_score_rank": index + 1} for index in range(3)
                        ]
                    }
                },
            }

    fake_lens = FakeLens()
    fake_model = torch.nn.Linear(1, 1)
    fake_tokenizer = object()
    monkeypatch.setattr(
        analysis,
        "_resolve_model_and_tokenizer",
        lambda lens, model, tokenizer, device, verbose=False: (
            fake_model,
            fake_tokenizer,
        ),
    )
    validated: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sweep,
        "_validate_full_context_ranks",
        lambda prepared: validated.append(prepared),
    )

    def fake_measure_layer(**kwargs: Any) -> None:
        assert kwargs["layer"] == 6
        assert kwargs["prepared_components"][214]["occurrences"] == [
            {"absolute_score_rank": 1},
            {"absolute_score_rank": 2},
        ]
        assert kwargs["rank_thresholds"] == (1, 3, 5)
        kwargs["checkpoint"](
            214,
            {
                "layer": 6,
                "component": 214,
                "threshold_results": {"1": {"suffix_erf_mean": 2.0}},
                "occurrences": [],
            },
        )

    monkeypatch.setattr(sweep, "_measure_layer", fake_measure_layer)

    result = ERFAnalysis(fake_lens).suffix_sweep(  # type: ignore[arg-type]
        layer=6,
        component=214,
        rank_thresholds=(5, 1, 3),
        occurrences=2,
        device="cpu",
    )

    assert validated
    assert result["component"] == 214
    assert result["method"] == sweep.METHOD
    assert result["model_id"] == "example/model"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"occurrences": 0}, "occurrences must be a positive integer"),
        ({"rank_thresholds": ()}, "rank_thresholds must not be empty"),
        (
            {"rank_thresholds": (1, 1)},
            "rank_thresholds must not contain duplicates",
        ),
    ],
)
def test_suffix_sweep_validates_public_arguments(
    kwargs: dict[str, Any], message: str
) -> None:
    class UnusedLens:
        pass

    with pytest.raises(ValueError, match=message):
        ERFAnalysis(UnusedLens()).suffix_sweep(  # type: ignore[arg-type]
            layer=0, component=0, **kwargs
        )
