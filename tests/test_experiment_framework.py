from __future__ import annotations

import json
from pathlib import Path

import pytest

from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json


def test_resumable_run_validates_sections_and_updates_atomically(tmp_path: Path) -> None:
    run = ResumableRun.open(
        output=tmp_path,
        resolved={"layers": [5]},
        source={"dirty": False},
        status="preparing",
    )
    run.validate_section("evaluation", {"model": "Inkling"})
    run.set_status("complete", complete=True)

    stored = json.loads((tmp_path / "run.json").read_text())
    assert stored["status"] == "complete"
    assert stored["completed_at"] is not None
    assert not (tmp_path / "run.json.tmp").exists()

    with pytest.raises(ValueError, match=r"evaluation\.model"):
        run.validate_section("evaluation", {"model": "another-model"})


def test_atomic_write_json_preserves_parseable_result(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "checkpoint.json"
    atomic_write_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}


def test_experiment_display_logs_command_output_and_progress(tmp_path: Path) -> None:
    display = ExperimentDisplay(
        output=tmp_path,
        title="Test run",
        completed=2,
        total=4,
        command=["icalens", "experiment", "test"],
    )
    with display:
        display.phase("Evaluating", layer=5)
        print("captured output")
        display.advance()
        display.set_outcome("incomplete")

    detail = (tmp_path / "experiment-detail.log").read_text()
    assert "command: icalens experiment test" in detail
    assert "captured output" in detail
    assert "status: incomplete" in detail
    assert "completed: 3/4" in detail


def test_experiment_display_rejects_progress_overflow(tmp_path: Path) -> None:
    display = ExperimentDisplay(output=tmp_path, title="Test", completed=1, total=1)
    with pytest.raises(ValueError, match="exceed"):
        display.advance()


def test_experiment_display_tracks_named_hierarchical_units_once(tmp_path: Path) -> None:
    first = ("dataset-a", 5, "ica")
    display = ExperimentDisplay(
        output=tmp_path,
        title="Layered experiment",
        completed=1,
        total=3,
        completed_unit_ids=[first],
        unit_label="dataset × layer × method results",
        command="icalens experiment example --layers 5,11",
    )

    assert not display.complete_unit(first)
    assert display.complete_unit(("dataset-a", 5, "sae"))
    assert display.completed == 2
    assert display.unit_label == "dataset × layer × method results"
