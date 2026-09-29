from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from icalens.experiments.erf_gradient import (
    METHOD,
    _component_checkpoint_valid,
    _component_units,
    _gradient_statistics,
    _load_prepared_run,
    _stable_seed,
    _write_prepared_inputs,
)
from icalens.experiments.erf_gradient_figure import render


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gradient_statistics_use_geometric_distance() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.7])

    result, distances = _gradient_statistics(probabilities)

    assert distances.tolist() == [3.0, 2.0, 1.0]
    expected = math.exp(0.1 * math.log(3) + 0.2 * math.log(2))
    assert result["gradient_erf"] == pytest.approx(expected)
    assert result["arithmetic_gradient_erf"] == pytest.approx(1.4)
    assert result["influence_radius_50"] == 1
    assert result["influence_radius_90"] == 2


def test_stable_seed_depends_on_label_and_layer() -> None:
    assert _stable_seed(7, "gpt2", 3) == _stable_seed(7, "gpt2", 3)
    assert _stable_seed(7, "gpt2", 3) != _stable_seed(7, "gpt2", 4)
    assert _stable_seed(7, "gpt2", 3) != _stable_seed(7, "gemma2", 3)


def test_component_units_sort_serialized_layer_keys_numerically() -> None:
    selections = {"gemma2": {"0": [1], "1": [2], "10": [3], "2": [4]}}

    units = _component_units(selections)

    assert [(layer, component) for _, _, layer, component in units] == [
        (0, 1),
        (1, 2),
        (2, 4),
        (10, 3),
    ]


def test_component_checkpoint_validation(tmp_path: Path) -> None:
    path = tmp_path / "component.json"
    path.write_text(
        json.dumps(
            {
                "model_label": "gpt2",
                "layer": 2,
                "component": 5,
                "method": METHOD,
                "gradient_erf_median": 1.5,
                "n_occurrences": 1,
                "occurrences": [{}],
            }
        )
    )

    assert _component_checkpoint_valid(path, label="gpt2", layer=2, component=5, method=METHOD)
    assert not _component_checkpoint_valid(path, label="gpt2", layer=2, component=6, method=METHOD)


def test_prepared_run_cache_round_trip(tmp_path: Path) -> None:
    lens_path = tmp_path / "lens"
    lens_path.mkdir()
    profile_path = lens_path / "profile.json"
    profile_path.write_text("{}")
    manifest_path = lens_path / "icalens.json"
    manifest_path.write_text(
        json.dumps({"layers": {"0": {"component_profile": "profile.json"}}})
    )
    output = tmp_path / "run"
    output.mkdir()
    resolved = {
        "format": "icalens.erf_gradient",
        "schema_version": 1,
        "method": METHOD,
        "components_per_layer": 1,
        "occurrences_per_component": 2,
        "seed": 0,
        "lens_order": ["gpt2"],
        "lenses": {
            "gpt2": {
                "path": str(lens_path),
                "manifest_sha256": _digest(manifest_path),
                "profile_sha256": {"0": _digest(profile_path)},
            }
        },
        "selections": {"gpt2": {"0": [7]}},
    }
    (output / "run.json").write_text(json.dumps({"resolved": resolved}))
    prepared = {
        "gpt2": {
            "0": {
                "7": {
                    "component": 7,
                    "tail_direction": "positive",
                    "occurrences": [{"source_index": 3}],
                }
            }
        }
    }
    _write_prepared_inputs(output, prepared)
    args = SimpleNamespace(
        components_per_layer=1,
        occurrences_per_component=2,
        seed=0,
    )

    cached = _load_prepared_run(output, lens_specs={"gpt2": lens_path}, args=args)

    assert cached is not None
    cached_resolved, selections, cached_prepared = cached
    assert cached_resolved == resolved
    assert selections == resolved["selections"]
    assert cached_prepared == prepared


def test_prepared_run_cache_rejects_changed_profile(tmp_path: Path) -> None:
    lens_path = tmp_path / "lens"
    lens_path.mkdir()
    profile_path = lens_path / "profile.json"
    profile_path.write_text("old")
    manifest_path = lens_path / "icalens.json"
    manifest_path.write_text(
        json.dumps({"layers": {"0": {"component_profile": "profile.json"}}})
    )
    output = tmp_path / "run"
    output.mkdir()
    resolved = {
        "format": "icalens.erf_gradient",
        "schema_version": 1,
        "method": METHOD,
        "components_per_layer": 1,
        "occurrences_per_component": 2,
        "seed": 0,
        "lens_order": ["gpt2"],
        "lenses": {
            "gpt2": {
                "path": str(lens_path),
                "manifest_sha256": _digest(manifest_path),
                "profile_sha256": {"0": _digest(profile_path)},
            }
        },
        "selections": {"gpt2": {"0": [7]}},
    }
    (output / "run.json").write_text(json.dumps({"resolved": resolved}))
    _write_prepared_inputs(
        output,
        {"gpt2": {"0": {"7": {"component": 7, "occurrences": []}}}},
    )
    profile_path.write_text("new")
    args = SimpleNamespace(components_per_layer=1, occurrences_per_component=2, seed=0)

    with pytest.raises(ValueError, match="profile changed.*choose another output"):
        _load_prepared_run(output, lens_specs={"gpt2": lens_path}, args=args)


def test_prepared_run_cache_rejects_changed_seed(tmp_path: Path) -> None:
    lens_path = tmp_path / "lens"
    lens_path.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    resolved = {
        "format": "icalens.erf_gradient",
        "schema_version": 1,
        "method": METHOD,
        "components_per_layer": 1,
        "occurrences_per_component": 2,
        "seed": 0,
        "lens_order": ["gpt2"],
        "lenses": {"gpt2": {"path": str(lens_path)}},
        "selections": {"gpt2": {"0": [7]}},
    }
    (output / "run.json").write_text(json.dumps({"resolved": resolved}))
    args = SimpleNamespace(
        components_per_layer=1,
        occurrences_per_component=2,
        seed=1,
    )

    with pytest.raises(ValueError, match="seed: 0 -> 1"):
        _load_prepared_run(output, lens_specs={"gpt2": lens_path}, args=args)


def test_erf_gradient_figure_writes_all_outputs(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    run = {
        "status": "complete",
        "resolved": {
            "components_per_layer": 2,
            "seed": 3,
            "lens_order": ["gpt2"],
            "lenses": {
                "gpt2": {
                    "layers": [0],
                    "model": {"repo_id": "openai-community/gpt2", "revision": "r"},
                }
            },
        },
    }
    (experiment / "run.json").write_text(json.dumps(run))
    with (experiment / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "layer", "component", "gradient_erf_median"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"model": "gpt2", "layer": 0, "component": 1, "gradient_erf_median": 1.5},
                {"model": "gpt2", "layer": 0, "component": 2, "gradient_erf_median": 3.5},
            ]
        )

    outputs = render(
        experiment,
        output_prefix=tmp_path / "figure" / "erf-gradient",
        panel_titles=None,
        force=False,
    )

    assert outputs == [
        tmp_path / "figure" / "erf-gradient.png",
        tmp_path / "figure" / "erf-gradient.pdf",
        tmp_path / "figure" / "erf-gradient.txt",
    ]
    assert all(path.is_file() for path in outputs)
    caption = outputs[2].read_text()
    assert "[1,2), [2,3), [3,4), [4,8), [8,16), [16,32), and [32,infinity)" in caption


def test_erf_gradient_figure_reads_partial_component_checkpoints(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    component_dir = experiment / "components" / "gpt2" / "layer_00"
    component_dir.mkdir(parents=True)
    run = {
        "status": "measuring",
        "resolved": {
            "components_per_layer": 2,
            "seed": 0,
            "lens_order": ["gpt2"],
            "lenses": {
                "gpt2": {
                    "layers": [0, 1],
                    "model": {"repo_id": "openai-community/gpt2", "revision": "r"},
                }
            },
        },
    }
    (experiment / "run.json").write_text(json.dumps(run))
    (component_dir / "C1.json").write_text(
        json.dumps(
            {
                "model_label": "gpt2",
                "layer": 0,
                "component": 1,
                "gradient_erf_median": 2.5,
            }
        )
    )

    outputs = render(
        experiment,
        output_prefix=tmp_path / "partial" / "erf-gradient",
        panel_titles=None,
        force=False,
    )

    caption = outputs[2].read_text()
    assert "Partial visualization based on 1 of 4 planned components" in caption
    assert "normalized over the components completed in that layer" in caption
