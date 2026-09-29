from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from integrity.profile_occurrence_replay import compare_component_occurrences
from integrity.profile_readout_replay import readout_fragment
from integrity.profiling_replay import section_matrix
from integrity.reference_preflight import run as run_reference_preflight
from integrity.run import CANARIES, execute_check, repository_identity


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def reference_fixture(tmp_path: Path) -> tuple[Path, Path]:
    lens_root = tmp_path / "lens"
    activation_root = tmp_path / "activations"
    model = {"repo_id": "owner/model", "revision": "model-sha", "type": "base"}
    dataset = {"repo_id": "owner/data", "revision": "data-sha", "split": "train"}
    framing = {
        "strategy": "prepend-eos",
        "token": "<eos>",
        "token_id": 9,
        "included_in_fitting_samples": False,
    }
    provenance = {
        "candidate_tokens": 20,
        "context_length": 8,
        "dataset": dataset,
        "document_framing": framing,
        "fitting_tokens": 10,
        "sampling_seed": 0,
        "text_field": "text",
        "token_scope": "all",
    }
    write_json(
        lens_root / "icalens.json",
        {
            "model": model,
            "activation_site": "resid_post",
            "layer_indexing": "transformer_blocks_zero_based",
            "layers": {
                "6": {
                    "file": "layer.safetensors",
                    "component_profile": "profile.json.gz",
                    "n_components": 2,
                    "fitting": {"provenance": provenance},
                }
            },
        },
    )
    write_json(
        activation_root / "activations.json",
        {
            "model": model,
            "activation_site": "resid_post",
            "layer_indexing": "transformer_blocks_zero_based",
            "sample_count": 10,
            "hidden_size": 2,
            "provenance": provenance,
            "layers": {
                "6": {
                    "file": "layer.bin",
                    "shape": [10, 2],
                    "status": "complete",
                }
            },
        },
    )
    for path in (
        lens_root / "layer.safetensors",
        lens_root / "profile.json.gz",
        activation_root / "layer.bin",
        activation_root / "samples.safetensors",
    ):
        path.write_bytes(b"fixture")
    return lens_root, activation_root


def test_reference_preflight_detects_upstream_provenance_mismatch(tmp_path: Path) -> None:
    lens_root, activation_root = reference_fixture(tmp_path)
    assert (
        run_reference_preflight(lens_root=lens_root, activation_root=activation_root, layer=6)[
            "status"
        ]
        == "pass"
    )

    manifest_path = activation_root / "activations.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["dataset"]["revision"] = "changed"
    write_json(manifest_path, manifest)
    result = run_reference_preflight(lens_root=lens_root, activation_root=activation_root, layer=6)
    assert result["status"] == "fail"
    assert "provenance_dataset" in result["failed_checks"]


def test_repository_identity_hashes_untracked_file_contents(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "integrity@example.invalid"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Integrity Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    untracked = tmp_path / "new.py"
    untracked.write_text("first", encoding="utf-8")
    first = repository_identity(tmp_path)
    untracked.write_text("second", encoding="utf-8")
    second = repository_identity(tmp_path)

    assert first["dirty"] is True
    assert first["changed_paths"] == ["new.py"]
    assert first["fingerprint"] != second["fingerprint"]


def test_execute_check_keeps_runtime_output_in_log(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "check" / "runtime.log"

    def operation() -> dict:
        print("detailed production output")
        return {"check": "C10-test", "status": "pass", "scope": {}}

    result = execute_check("C10-test", operation, log_path=log_path, verbose=False)

    output = capsys.readouterr().out
    assert output.startswith("START C10-test (")
    assert "PASS C10-test []" in output
    assert log_path.read_text() == "detailed production output\n"
    assert result["artifacts"]["runtime_log"] == str(log_path)
    assert result["timing"]["duration_seconds"] >= 0
    assert result["timing"]["started_at"]
    assert result["timing"]["finished_at"]


def test_execute_check_records_exception_as_failure(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "check" / "runtime.log"

    def operation() -> dict:
        raise RuntimeError("broken replay")

    result = execute_check("C10-test", operation, log_path=log_path, verbose=False)

    assert result["status"] == "fail"
    assert result["failed_checks"] == ["exception"]
    assert result["error"] == "RuntimeError: broken replay"
    assert "Traceback" in log_path.read_text()
    assert result["timing"]["duration_seconds"] >= 0
    assert "FAIL C10-test []" in capsys.readouterr().out


def test_execute_check_does_not_swallow_keyboard_interrupt(tmp_path: Path) -> None:
    def operation() -> dict:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_check(
            "C10-test",
            operation,
            log_path=tmp_path / "runtime.log",
            verbose=False,
        )


def test_model_canaries_cover_three_released_architectures() -> None:
    assert {name: canary.layer for name, canary in CANARIES.items()} == {
        "gpt2": 6,
        "gemma": 13,
        "qwen": 16,
    }
    assert all(canary.lens.name.startswith("icalens-") for canary in CANARIES.values())


def test_profile_section_matrix_has_stable_key_order() -> None:
    profile = {
        "components": [
            {"score_statistics": {"variance": 2.0, "mean": 1.0}},
            {"score_statistics": {"mean": 3.0, "variance": 4.0}},
        ]
    }

    keys, values = section_matrix(profile, np.asarray([1, 0]), "score_statistics")

    assert keys == ["mean", "variance"]
    assert values.tolist() == [[3.0, 4.0], [1.0, 2.0]]


def test_profile_readout_fragment_has_stable_semantic_order() -> None:
    readouts = []
    for component in range(2):
        readouts.append(
            {
                sign: {
                    side: [
                        {
                            "token_id": component * 100 + sign_index * 10 + side_index,
                            "token": f"t-{component}-{sign}-{side}",
                            "text": f"x-{component}-{sign}-{side}",
                            "logit": float(component + sign_index + side_index),
                        }
                    ]
                    for side_index, side in enumerate(("top_tokens", "bottom_tokens"))
                }
                for sign_index, sign in enumerate(("positive", "negative"))
            }
        )

    ids, logits, identities = readout_fragment(readouts, np.asarray([1]))

    assert ids.tolist() == [100, 101, 110, 111]
    assert logits.tolist() == [1.0, 2.0, 2.0, 3.0]
    assert identities[0].startswith("t-1-positive-top_tokens\0")


def test_occurrence_comparison_checks_context_and_numeric_values() -> None:
    occurrence = {
        "token": "x",
        "text": "x",
        "token_id": 1,
        "position": 2,
        "context": "xyz",
        "context_target_start": 0,
        "context_target_end": 1,
        "source_index": 3,
        "absolute_score_rank": 1,
        "score": 4.0,
        "energy": 0.5,
    }
    component = {
        "tail_direction": "positive",
        "examples": {
            "positive": {"tokens": [{"text": "x", "count": 1}], "occurrences": [occurrence]},
            "negative": {"tokens": [], "occurrences": []},
        },
    }

    exact, error = compare_component_occurrences(component, component, rtol=0, atol=0)

    assert exact is True
    assert error == 0
