from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tpp.plot_official import (
    aggregate_seed_curves,
    discover_seed_roots,
    validate_seed_configs,
)


def _payload(first: float, second: float) -> dict:
    values = {1: first, 2: second}
    metrics = {f"tpp_threshold_{budget}_total_metric": value for budget, value in values.items()}
    return {
        "eval_config": {"n_values": [1, 2]},
        "eval_result_metrics": {"tpp_metrics": metrics},
        "eval_result_unstructured": {
            "dataset": {
                "class": {
                    f"tpp_threshold_{budget}_total_metric": value
                    for budget, value in values.items()
                }
            }
        },
    }


def test_discover_seed_roots_supports_seed_layout_and_legacy_root(tmp_path: Path) -> None:
    (tmp_path / "seed-1").mkdir()
    (tmp_path / "seed-0").mkdir()
    assert [name for name, _ in discover_seed_roots(tmp_path)] == ["seed-0", "seed-1"]

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    assert discover_seed_roots(legacy) == [("single", legacy)]


def test_aggregate_seed_curves_averages_layers_before_seeds() -> None:
    budgets, seed_values = aggregate_seed_curves(
        {
            "seed-0": {6: _payload(1.0, 2.0), 10: _payload(3.0, 4.0)},
            "seed-1": {6: _payload(5.0, 6.0)},
        },
        suffix="total_metric",
    )

    assert budgets == [1, 2]
    np.testing.assert_allclose(seed_values, [[2.0, 3.0], [5.0, 6.0]])
    np.testing.assert_allclose(seed_values.mean(axis=0), [3.5, 4.5])


def test_seed_config_validation_ignores_only_random_seed(tmp_path: Path) -> None:
    roots = []
    for seed, train_size in ((0, 4000), (1, 4000)):
        root = tmp_path / f"seed-{seed}"
        model = root / "gpt2"
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({"method_definition_version": 4, "settings": {
                "random_seed": seed, "train_size": train_size
            }}),
            encoding="utf-8",
        )
        roots.append((root.name, root))
    validate_seed_configs(roots, "gpt2")

    path = roots[1][1] / "gpt2" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["settings"]["train_size"] = 2000
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="differ beyond random_seed"):
        validate_seed_configs(roots, "gpt2")
