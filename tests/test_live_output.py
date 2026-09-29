from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from icalens.cli._live_output import (
    _dirty_warning,
    _eta,
    _layer_progress,
    _OperationDisplay,
    _portable_command,
    _run_header,
)


def test_eta_uses_only_newly_completed_work() -> None:
    assert _eta(elapsed_seconds=120, newly_completed=2, remaining=3) == "03:00"
    assert _eta(elapsed_seconds=120, newly_completed=0, remaining=3) is None
    assert _eta(elapsed_seconds=120, newly_completed=2, remaining=0) is None


def test_fit_progress_starts_from_existing_output(tmp_path: Path) -> None:
    activations = tmp_path / "activations"
    output = tmp_path / "lens"
    activations.mkdir()
    output.mkdir()
    (activations / "activations.json").write_text(
        json.dumps({"layers": {str(layer): {} for layer in range(4)}})
    )
    (output / "icalens.json").write_text(
        json.dumps(
            {
                "layers": {
                    "0": {"file": "layer0.safetensors"},
                    "1": {"file": "layer1.safetensors"},
                }
            }
        )
    )

    requested, completed = _layer_progress(
        ["--input", str(activations), "--layers", "all", "--output", str(output)],
        title="ICA Lens · activation fitting",
    )

    assert requested == (0, 1, 2, 3)
    assert completed == {0, 1}


def test_capture_progress_reads_completed_layer_status(tmp_path: Path) -> None:
    output = tmp_path / "activations"
    output.mkdir()
    (output / "activations.json").write_text(
        json.dumps(
            {
                "layers": {
                    "0": {"status": "complete"},
                    "1": {"status": "pending"},
                }
            }
        )
    )

    requested, completed = _layer_progress(
        ["--layers", "all", "--output", str(output)],
        title="ICA Lens · text activation capture",
    )

    assert requested == (0, 1)
    assert completed == {0}


def test_profile_suboperations_do_not_treat_existing_profiles_as_completed(
    tmp_path: Path,
) -> None:
    lens = tmp_path / "lens"
    lens.mkdir()
    (lens / "icalens.json").write_text(
        json.dumps(
            {
                "layers": {
                    "0": {"component_profile": "profiles/layer0.json.gz"},
                    "1": {"component_profile": "profiles/layer1.json.gz"},
                }
            }
        )
    )

    for operation in ("refresh-statistics", "refresh-examples", "add-r-lens"):
        requested, completed = _layer_progress(
            [operation, "--lens", str(lens), "--layers", "all"],
            title="ICA Lens · component profiling",
        )

        assert requested == (0, 1)
        assert completed == set()


def test_refresh_messages_update_phase_and_durable_layer_progress(tmp_path: Path) -> None:
    display = _OperationDisplay(
        title="ICA Lens · component profiling",
        detail_path=tmp_path / "run.log",
        command="icalens profile refresh-statistics",
        source_dirty=False,
        requested_layers=(0, 1, 2),
        completed_layers=set(),
    )

    display._observe("Durable completed layers: 0")
    assert display.completed_layers == {0}
    assert display.initial_completed == 1

    display._observe("Refreshing score statistics for layer 1 from cached activations...")
    assert display.phase == "Refreshing statistics"
    assert display.current_layer == 1

    display._observe("Refreshed layer 1: 4096 components from 1000000 activation rows.")
    assert display.completed_layers == {0, 1}
    assert display.phase == "Checkpointing"
    display.detail.close()


def test_log_header_records_portable_command() -> None:
    command = _portable_command(
        "ICA Lens · component profiling",
        ["--lens", "a path/lens", "--layers", "all"],
    )
    header = _run_header(command, source_dirty=True)

    assert command == "icalens profile --lens 'a path/lens' --layers all"
    assert "# ICA Lens run\n" in header
    assert "working_directory:" in header
    assert "icalens_version:" in header
    assert "source_worktree_dirty: true\n" in header
    assert f"command: {command}\n" in header


def test_dirty_warning_is_non_blocking_and_only_shown_when_dirty() -> None:
    assert "WARNING" in (_dirty_warning(True) or "")
    assert _dirty_warning(False) is None
    assert _dirty_warning(None) is None


def test_dirty_warning_has_dedicated_panel(tmp_path: Path) -> None:
    display = _OperationDisplay(
        title="ICA Lens · activation fitting",
        detail_path=tmp_path / "run.log",
        command="icalens fit activations --input activations",
        source_dirty=True,
        requested_layers=(0,),
        completed_layers=set(),
    )
    console = Console(record=True, width=120)
    console.print(display.render())
    rendered = console.export_text()

    assert "Source warning" in rendered
    assert "uncommitted changes" in rendered
