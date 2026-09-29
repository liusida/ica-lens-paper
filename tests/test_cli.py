from __future__ import annotations

import pytest

from icalens import cli


def test_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--help"])
    output = capsys.readouterr().out
    assert "fit text" in output
    assert "publish" in output


def test_fit_help(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["fit", "--help"])
    output = capsys.readouterr().out
    assert "icalens fit {text,chat,activations}" in output


def test_plot_help(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["plot", "--help"])
    output = capsys.readouterr().out
    assert "fitting-summary" in output


def test_dispatches_text_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import fit_text

    received: list[str] = []
    monkeypatch.setattr(fit_text, "main", lambda args: received.extend(args))

    cli.main(["fit", "text", "--layers", "6"])

    assert received == ["--layers", "6"]


def test_dispatches_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import publish

    received: list[str] = []
    monkeypatch.setattr(publish, "main", lambda args: received.extend(args))

    cli.main(["publish", "owner/lens", "--lens", "artifact"])

    assert received == ["owner/lens", "--lens", "artifact"]


def test_dispatches_integrity_reproduction(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import integrity

    received: list[str] = []
    monkeypatch.setattr(integrity, "main", lambda args: received.extend(args))

    cli.main(["integrity", "reproduce", "--layer", "6"])

    assert received == ["--layer", "6"]


def test_dispatches_autointerpretability(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.experiments import autointerpretability

    received: list[str] = []
    monkeypatch.setattr(autointerpretability, "main", lambda args: received.extend(args))

    cli.main(["experiment", "autointerpretability", "summarize", "--input", "run"])

    assert received == ["summarize", "--input", "run"]


def test_dispatches_erf_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.experiments import erf_gradient

    received: list[str] = []
    monkeypatch.setattr(erf_gradient, "main", lambda args: received.extend(args))

    cli.main(["experiment", "erf-gradient", "--output", "run"])

    assert received == ["--output", "run"]


def test_dispatches_erf_suffix_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.experiments import erf_suffix_sweep

    received: list[str] = []
    monkeypatch.setattr(erf_suffix_sweep, "main", lambda args: received.extend(args))

    cli.main(["experiment", "erf-suffix-sweep", "--output", "run"])

    assert received == ["--output", "run"]


def test_dispatches_erf_gradient_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.experiments import erf_gradient_figure

    received: list[str] = []
    monkeypatch.setattr(erf_gradient_figure, "main", lambda args: received.extend(args))

    cli.main(["experiment", "figure", "erf-gradient", "run"])

    assert received == ["run"]


def test_dispatches_erf_suffix_sweep_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.experiments import erf_suffix_sweep_figure

    received: list[str] = []
    monkeypatch.setattr(erf_suffix_sweep_figure, "main", lambda args: received.extend(args))

    cli.main(["experiment", "figure", "erf-suffix-sweep", "run"])

    assert received == ["run"]


def test_dispatches_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import profile

    received: list[str] = []
    monkeypatch.setattr(profile, "main", lambda args: received.extend(args))

    cli.main(["profile", "--lens", "artifact", "--layer", "6"])

    assert received == ["--lens", "artifact", "--layer", "6"]


def test_dispatches_fitting_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import plot_fitting

    received: list[str] = []
    monkeypatch.setattr(plot_fitting, "main", lambda args: received.extend(args))

    cli.main(["plot", "fitting-summary", "owner/lens"])

    assert received == ["owner/lens"]


def test_profile_defaults_to_updating_local_lens(tmp_path) -> None:
    from icalens.cli.profile import _resolve_output

    artifact = tmp_path / "lens"
    artifact.mkdir()

    assert _resolve_output(str(artifact), None) == artifact.resolve()


def test_profile_hub_source_requires_output() -> None:
    from icalens.cli.profile import _resolve_output

    with pytest.raises(ValueError, match="--output is required"):
        _resolve_output("owner/lens", None)


def test_unknown_command_is_clear() -> None:
    with pytest.raises(SystemExit, match="unknown command"):
        cli.main(["unknown"])
