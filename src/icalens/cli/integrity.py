"""Reproduce one ICA Lens layer from recorded sources and compare references."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

SCHEMA_VERSION = 1
BASE_STAGES = ("capture", "compare_activations", "fit", "profile", "compare_results")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens integrity reproduce", description=__doc__)
    parser.add_argument("--reference-lens", type=Path, required=True)
    parser.add_argument("--reference-activations", type=Path, required=True)
    parser.add_argument(
        "--reference-experiments",
        type=Path,
        default=None,
        help="Audit matching official experiment results below this directory.",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-rows", type=int, default=1024)
    parser.add_argument("--verification-components", type=int, default=64)
    parser.add_argument("--verification-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reference_lens_path = args.reference_lens.expanduser().resolve()
    reference_activations_path = args.reference_activations.expanduser().resolve()
    reference_experiments_path = (
        args.reference_experiments.expanduser().resolve()
        if args.reference_experiments is not None
        else None
    )
    output = args.output.expanduser().resolve()
    if output == reference_lens_path or output == reference_activations_path:
        raise ValueError("--output must be separate from both reference artifacts")
    if args.verification_rows <= 0 or args.verification_components <= 0:
        raise ValueError("verification sample sizes must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("comparison tolerances must be non-negative")

    reference_activations = ActivationDataset(reference_activations_path)
    reference_lens = ICALens.from_pretrained(reference_lens_path)
    _validate_references(reference_lens, reference_activations, args.layer)
    fitting = reference_lens.metadata["layers"][str(args.layer)]["fitting"]
    _validate_sampling_provenance(fitting, reference_activations)
    profile = reference_lens._get_profile(reference_lens._get_layer(args.layer))
    experiment_artifacts = (
        _discover_official_experiments(
            reference_experiments_path,
            model_id=reference_lens.model_id,
            layer=args.layer,
        )
        if reference_experiments_path is not None
        else []
    )
    if reference_experiments_path is not None and not experiment_artifacts:
        raise ValueError(
            f"no official experiment results match {reference_lens.model_id}, layer {args.layer} "
            f"below {reference_experiments_path}"
        )
    stages = (*BASE_STAGES, "audit_experiments") if experiment_artifacts else BASE_STAGES
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "operation": "reproduce",
        "reference_lens": str(reference_lens_path),
        "reference_lens_manifest_sha256": _sha256(reference_lens_path / "icalens.json"),
        "reference_activations": str(reference_activations_path),
        "reference_activation_manifest_sha256": _sha256(
            reference_activations_path / "activations.json"
        ),
        "layer": args.layer,
        "verification_rows": min(args.verification_rows, reference_activations.sample_count),
        "verification_components": min(
            args.verification_components, reference_activations.hidden_size
        ),
        "verification_seed": args.verification_seed,
        "device": args.device,
        "rtol": args.rtol,
        "atol": args.atol,
        "capture": _capture_configuration(reference_lens, reference_activations, fitting),
        "fit": _fit_configuration(fitting),
        "profile": _profile_configuration(profile, reference_activations.sample_count),
        "reference_experiments": (
            {
                "root": str(reference_experiments_path),
                "artifacts": experiment_artifacts,
            }
            if reference_experiments_path is not None
            else None
        ),
    }
    source = source_provenance()
    warn_if_dirty(source)
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="running",
        filename="run.json",
    )
    completed = list(run.state.get("completed_stages", []))
    if any(stage not in stages for stage in completed):
        raise ValueError(f"{run.path} contains an unknown completed stage")
    reproduced_activations = output / "captured"
    reproduced_lens = output / "reproduced-lens"
    report_path = output / "report.json"
    completed = _durable_completed_stages(completed, output=output, layer=args.layer, stages=stages)
    run.update(completed_stages=completed)

    try:
        with ExperimentDisplay(
            output=output / "logs",
            title="ICA Lens · integrity reproduction",
            completed=len(completed),
            total=len(stages),
            completed_unit_ids=completed,
            unit_label="stages",
            source_dirty=source.get("dirty"),
        ) as display:
            for stage in stages:
                if stage in completed:
                    continue
                display.phase(stage.replace("_", " ").title(), layer=args.layer)
                if stage == "capture":
                    _run_command(
                        _capture_command(resolved, reproduced_activations, args.layer), display
                    )
                elif stage == "compare_activations":
                    comparison = _compare_activations(
                        reference_activations_path,
                        reproduced_activations,
                        layer=args.layer,
                        rows=resolved["verification_rows"],
                        seed=args.verification_seed,
                    )
                    _merge_report(report_path, "activations", comparison)
                    _require_pass(comparison)
                elif stage == "fit":
                    _run_command(
                        _fit_command(resolved, reproduced_activations, reproduced_lens), display
                    )
                elif stage == "profile":
                    for command in _profile_commands(
                        resolved, reproduced_activations, reproduced_lens, args.layer
                    ):
                        _run_command(command, display)
                elif stage == "compare_results":
                    comparison = _compare_results(
                        reference_lens_path,
                        reproduced_lens,
                        reference_activations_path,
                        reproduced_activations,
                        layer=args.layer,
                        rows=resolved["verification_rows"],
                        components=resolved["verification_components"],
                        seed=args.verification_seed,
                        rtol=args.rtol,
                        atol=args.atol,
                    )
                    _merge_report(report_path, "results", comparison)
                    _require_pass(comparison)
                else:
                    audit = _audit_official_experiments(experiment_artifacts)
                    _merge_report(report_path, "experiments", audit)
                    _require_pass(audit)
                completed.append(stage)
                run.update(completed_stages=completed)
                display.complete_unit(stage, refresh=True)
    except BaseException:
        run.set_status("failed")
        raise
    report = json.loads(report_path.read_text())
    report["status"] = "pass"
    atomic_write_json(report_path, report)
    _write_markdown_report(output / "report.md", report)
    run.set_status("complete", complete=True)
    print(f"Integrity reproduction passed: {output / 'report.md'}")


def _durable_completed_stages(
    recorded: list[str], *, output: Path, layer: int, stages: Sequence[str]
) -> list[str]:
    """Trust recorded stages only while their durable outputs still validate."""
    valid: list[str] = []
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    for stage in stages:
        if stage not in recorded:
            break
        try:
            if stage == "capture":
                dataset = ActivationDataset(output / "captured")
                dataset.layer(layer)
                dataset.samples()
            elif stage == "compare_activations":
                if report.get("activations", {}).get("status") != "pass":
                    break
            elif stage == "fit":
                lens = ICALens.from_pretrained(output / "reproduced-lens")
                lens._get_layer(layer)
            elif stage == "profile":
                lens = ICALens.from_pretrained(output / "reproduced-lens")
                profile = lens._get_profile(lens._get_layer(layer))
                if (
                    not all("score_statistics" in item for item in profile["components"])
                    or not isinstance(profile.get("score_statistics_provenance"), dict)
                    or not isinstance(profile.get("example_provenance"), dict)
                    or profile.get("selection", {}).get("example_absolute_score_rank")
                    != "competition_rank_by_absolute_score"
                    or not _profile_occurrence_ranks_complete(profile)
                ):
                    break
            elif stage == "compare_results":
                if report.get("results", {}).get("status") != "pass":
                    break
            elif report.get("experiments", {}).get("status") != "pass":
                break
        except Exception:
            break
        valid.append(stage)
    return valid


def _validate_references(lens: ICALens, dataset: ActivationDataset, layer: int) -> None:
    if layer not in lens.available_layers or layer not in dataset.available_layers:
        raise ValueError(f"layer {layer} must exist in both reference artifacts")
    if lens.model_id != dataset.model["repo_id"]:
        raise ValueError("reference Lens and activations use different models")
    if lens.model_revision != dataset.model.get("revision"):
        raise ValueError("reference Lens and activations use different model revisions")
    if lens._hidden_size != dataset.hidden_size:
        raise ValueError("reference Lens and activations have different hidden sizes")


def _validate_sampling_provenance(fitting: dict[str, Any], dataset: ActivationDataset) -> None:
    """Reject a cache that does not represent the Lens fitting population."""
    expected = fitting.get("provenance")
    if not isinstance(expected, dict):
        raise ValueError("reference Lens is missing fitting provenance")
    actual = dataset.manifest["provenance"]
    fields = {
        "candidate_tokens": (expected.get("candidate_tokens"), actual.get("candidate_tokens")),
        "fitting_tokens": (expected.get("fitting_tokens"), actual.get("fitting_tokens")),
        "sampling_seed": (expected.get("sampling_seed"), actual.get("sampling_seed")),
        "context_length": (expected.get("context_length"), actual.get("context_length")),
    }
    mismatches = [
        f"{name}: Lens={lens_value!r}, activations={cache_value!r}"
        for name, (lens_value, cache_value) in fields.items()
        if lens_value != cache_value
    ]
    if mismatches:
        raise ValueError(
            "reference activations do not contain the Lens fitting population ("
            + "; ".join(mismatches)
            + "); recapture the activation cache from the Lens fitting provenance"
        )


def _capture_configuration(
    lens: ICALens, dataset: ActivationDataset, fitting: dict[str, Any]
) -> dict[str, Any]:
    provenance = fitting["provenance"]
    model = dataset.model
    source = provenance["dataset"]
    kind = "chat" if model.get("type") == "instruct" else "text"
    if "repo_id" not in source:
        raise ValueError("integrity reproduction currently requires a Hub dataset reference")
    return {
        "kind": kind,
        "model": lens.model_id,
        "model_revision": lens.model_revision,
        "dataset": source["repo_id"],
        "dataset_revision": source["revision"],
        "split": source["split"],
        "context_length": provenance["context_length"],
        "candidate_tokens": provenance["candidate_tokens"],
        "token_budget": provenance["fitting_tokens"],
        "seed": provenance["sampling_seed"],
        "text_field": provenance.get("text_field"),
        "messages_field": provenance.get("messages_field"),
        "token_scope": provenance.get("token_scope"),
        "document_framing": (provenance.get("document_framing") or {}).get("strategy"),
    }


def _fit_configuration(fitting: dict[str, Any]) -> dict[str, Any]:
    return {
        "preprocessing": fitting.get("icalens_preprocessing", "none"),
        "max_iter": int(fitting["max_iter"]),
        "objective_every": int(fitting.get("objective_every", 1)),
        "fit_batch_size": int(fitting["batch_size"]),
        "seed": int(fitting["random_state"]),
    }


def _profile_configuration(profile: dict[str, Any], population: int) -> dict[str, Any]:
    selection = profile["selection"]
    if (
        selection.get("example_selection") != "top_absolute_score_on_selected_tail"
        or selection.get("example_absolute_score_rank") != "competition_rank_by_absolute_score"
        or "example_provenance" not in profile
        or not _profile_occurrence_ranks_complete(profile)
    ):
        raise ValueError(
            "reference Lens examples use an older or incomplete selection protocol; "
            "run `icalens profile refresh-examples-rank` for existing artifacts, or "
            "`icalens profile refresh-examples` to regenerate examples, before "
            "integrity reproduction"
        )
    sampling = profile["provenance"]["profile_sampling"]
    statistics = profile.get("score_statistics_provenance", {}).get("statistics_sampling", {})
    statistics_tokens = int(statistics.get("selected_tokens", population))
    return {
        "sample_seed": int(sampling["seed"]),
        "profile_tokens": max(int(sampling["selected_tokens"]), statistics_tokens, population),
        "statistics_tokens": statistics_tokens,
        "example_tokens": population,
        "top_k_examples": int(selection["top_k_examples_on_selected_tail"]),
        "logit_lens_top_k": int(selection["logit_lens_top_k"]),
        "logit_lens_batch_size": int(selection["logit_lens_batch_size"]),
    }


def _profile_occurrence_ranks_complete(profile: dict[str, Any]) -> bool:
    return all(
        isinstance(occurrence.get("absolute_score_rank"), int)
        and occurrence["absolute_score_rank"] >= 1
        for component in profile.get("components", [])
        for sign in ("positive", "negative")
        for occurrence in component.get("examples", {}).get(sign, {}).get("occurrences", [])
    )


def _capture_command(resolved: dict[str, Any], output: Path, layer: int) -> list[str]:
    config = resolved["capture"]
    command = _cli("capture", config["kind"], "--model", config["model"])
    command += [
        "--model-revision",
        config["model_revision"],
        "--dataset",
        config["dataset"],
        "--dataset-revision",
        config["dataset_revision"],
        "--split",
        config["split"],
        "--layers",
        str(layer),
        "--output",
        str(output),
        "--context-length",
        str(config["context_length"]),
        "--candidate-tokens",
        str(config["candidate_tokens"]),
        "--token-budget",
        str(config["token_budget"]),
        "--seed",
        str(config["seed"]),
    ]
    if config["kind"] == "text":
        command += ["--text-field", config["text_field"]]
        if config["document_framing"]:
            command += ["--document-framing", config["document_framing"]]
    else:
        command += ["--messages-field", config["messages_field"]]
        command += ["--token-scope", config["token_scope"]]
    return command


def _fit_command(resolved: dict[str, Any], activations: Path, output: Path) -> list[str]:
    config = resolved["fit"]
    return _cli(
        "fit",
        "activations",
        "--input",
        str(activations),
        "--layers",
        str(resolved["layer"]),
        "--output",
        str(output),
        "--icalens-preprocessing",
        config["preprocessing"],
        "--max-iter",
        str(config["max_iter"]),
        "--objective-every",
        str(config["objective_every"]),
        "--fit-batch-size",
        str(config["fit_batch_size"]),
        "--seed",
        str(config["seed"]),
    )


def _profile_commands(
    resolved: dict[str, Any], activations: Path, lens: Path, layer: int
) -> list[list[str]]:
    config = resolved["profile"]
    common = ["--lens", str(lens), "--layer", str(layer), "--activations", str(activations)]
    base = _cli(
        "profile",
        *common,
        "--sample-seed",
        str(config["sample_seed"]),
        "--max-tokens",
        str(config["profile_tokens"]),
        "--top-k-examples",
        str(config["top_k_examples"]),
        "--logit-lens-top-k",
        str(config["logit_lens_top_k"]),
        "--logit-lens-batch-size",
        str(config["logit_lens_batch_size"]),
        "--device",
        resolved["device"],
    )
    return [base]


def _cli(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "icalens.cli", *arguments]


def _run_command(command: list[str], display: ExperimentDisplay) -> None:
    print("$ " + " ".join(command))
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _sample_indices(size: int, count: int, seed: int) -> torch.Tensor:
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed))[:count].sort().values


def _compare_activations(
    reference: Path, reproduced: Path, *, layer: int, rows: int, seed: int
) -> dict[str, Any]:
    expected = ActivationDataset(reference)
    actual = ActivationDataset(reproduced)
    indices = _sample_indices(expected.sample_count, rows, seed)
    metadata_equal = all(
        torch.equal(
            expected.samples()[key].index_select(0, indices),
            actual.samples()[key].index_select(0, indices),
        )
        for key in expected.samples()
    )
    values_equal = torch.equal(
        expected.layer(layer).index_select(0, indices), actual.layer(layer).index_select(0, indices)
    )
    return {
        "status": "pass" if metadata_equal and values_equal else "fail",
        "sampled_rows": rows,
        "seed": seed,
        "metadata_exact": metadata_equal,
        "activation_values_exact": values_equal,
    }


def _compare_results(
    reference_lens_path: Path,
    reproduced_lens_path: Path,
    reference_activations_path: Path,
    reproduced_activations_path: Path,
    *,
    layer: int,
    rows: int,
    components: int,
    seed: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    expected_lens = ICALens.from_pretrained(reference_lens_path)
    actual_lens = ICALens.from_pretrained(reproduced_lens_path)
    expected_layer = expected_lens._get_layer(layer)
    actual_layer = actual_lens._get_layer(layer)
    assert expected_layer.center is not None and actual_layer.center is not None
    assert expected_layer.reading_matrix is not None and actual_layer.reading_matrix is not None
    assert expected_layer.writing_matrix is not None and actual_layer.writing_matrix is not None
    row_indices = _sample_indices(
        ActivationDataset(reference_activations_path).sample_count, rows, seed
    )
    component_indices = _sample_indices(expected_layer.n_components, components, seed)
    component_np = component_indices.numpy()
    matrices_close = (
        np.allclose(expected_layer.center, actual_layer.center, rtol=rtol, atol=atol)
        and np.allclose(
            expected_layer.reading_matrix[component_np],
            actual_layer.reading_matrix[component_np],
            rtol=rtol,
            atol=atol,
        )
        and np.allclose(
            expected_layer.writing_matrix[:, component_np],
            actual_layer.writing_matrix[:, component_np],
            rtol=rtol,
            atol=atol,
        )
    )
    expected_values = (
        ActivationDataset(reference_activations_path).layer(layer).index_select(0, row_indices)
    )
    actual_values = (
        ActivationDataset(reproduced_activations_path).layer(layer).index_select(0, row_indices)
    )
    expected_scores = np.asarray(
        expected_lens.transform(expected_values.float().numpy(), layer=layer)
    )[:, component_np]
    actual_scores = np.asarray(actual_lens.transform(actual_values.float().numpy(), layer=layer))[
        :, component_np
    ]
    scores_close = np.allclose(expected_scores, actual_scores, rtol=rtol, atol=atol)
    expected_profile = expected_lens._get_profile(expected_layer)
    actual_profile = actual_lens._get_profile(actual_layer)
    profiles_close = _profiles_close(expected_profile, actual_profile, component_np, rtol, atol)
    passed = bool(matrices_close and scores_close and profiles_close)
    return {
        "status": "pass" if passed else "fail",
        "sampled_rows": rows,
        "sampled_components": components,
        "seed": seed,
        "fitted_matrices_close": bool(matrices_close),
        "probe_scores_close": bool(scores_close),
        "profiles_close": bool(profiles_close),
        "maximum_absolute_score_error": float(np.max(np.abs(expected_scores - actual_scores))),
        "rtol": rtol,
        "atol": atol,
    }


def _profiles_close(
    expected: dict[str, Any],
    actual: dict[str, Any],
    components: np.ndarray,
    rtol: float,
    atol: float,
) -> bool:
    for index in components.tolist():
        left = expected["components"][index]
        right = actual["components"][index]
        if left["component"] != right["component"] or left.get("tail_direction") != right.get(
            "tail_direction"
        ):
            return False
        for section in ("score_statistics", "sign_statistics"):
            for key, value in left.get(section, {}).items():
                if key.endswith("_rank"):
                    if value != right.get(section, {}).get(key):
                        return False
                elif not np.isclose(value, right.get(section, {}).get(key), rtol=rtol, atol=atol):
                    return False
        if _example_identities(left) != _example_identities(right):
            return False
        if _token_identities(left.get("logit_lens")) != _token_identities(right.get("logit_lens")):
            return False
    return True


def _example_identities(component: dict[str, Any]) -> Any:
    result = []
    for sign in ("negative", "positive"):
        values = component.get("examples", {}).get(sign, {}).get("occurrences", [])
        result.append(
            [
                (
                    v.get("source_index"),
                    v.get("position"),
                    v.get("token_id"),
                    v.get("absolute_score_rank"),
                )
                for v in values
            ]
        )
    return result


def _token_identities(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _token_identities(item) for key, item in value.items()}
    if isinstance(value, list):
        return [
            item.get("token_id")
            if isinstance(item, dict) and "token_id" in item
            else _token_identities(item)
            for item in value
        ]
    return value


def _discover_official_experiments(
    root: Path, *, model_id: str, layer: int
) -> list[dict[str, Any]]:
    """Resolve completed official experiment summaries for one model layer."""
    if not root.is_dir():
        raise FileNotFoundError(f"reference experiment directory not found: {root}")
    artifacts: list[dict[str, Any]] = []
    for run_path in sorted(root.glob("*/official/results/*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        resolved = run.get("resolved", {})
        if resolved.get("model_id") != model_id or layer not in resolved.get("layers", []):
            continue
        results_path = run_path.with_name("results.json")
        if not results_path.is_file():
            raise FileNotFoundError(f"matching official experiment has no results: {results_path}")
        artifacts.append(
            {
                "experiment": resolved.get("experiment"),
                "directory": str(run_path.parent),
                "run_sha256": _sha256(run_path),
                "results_sha256": _sha256(results_path),
                "layer": layer,
                "model_id": model_id,
            }
        )
    return artifacts


def _audit_official_experiments(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    passed = True
    for artifact in artifacts:
        directory = Path(artifact["directory"])
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        resolved = run.get("resolved", {})
        experiment = payload.get("experiment", {})
        rows = payload.get("rows")
        layer = int(artifact["layer"])
        matching_rows = (
            [row for row in rows if isinstance(row, dict) and row.get("layer") == layer]
            if isinstance(rows, list)
            else []
        )
        metrics_finite = bool(matching_rows) and all(
            _row_metrics_finite(row) for row in matching_rows
        )
        identity_matches = all(
            resolved.get(key) == experiment.get(key) == artifact[key]
            for key in ("experiment", "model_id")
        )
        status_complete = run.get("status") == "complete"
        checksums_match = artifact["run_sha256"] == _sha256(directory / "run.json") and artifact[
            "results_sha256"
        ] == _sha256(directory / "results.json")
        item_passed = bool(
            status_complete and identity_matches and metrics_finite and checksums_match
        )
        passed = passed and item_passed
        checked.append(
            {
                "experiment": artifact["experiment"],
                "directory": str(directory),
                "layer": layer,
                "matching_rows": len(matching_rows),
                "run_complete": status_complete,
                "identity_matches": identity_matches,
                "metrics_finite": metrics_finite,
                "checksums_match": checksums_match,
                "status": "pass" if item_passed else "fail",
            }
        )
    return {
        "status": "pass" if passed and bool(checked) else "fail",
        "checked_artifacts": len(checked),
        "artifacts": checked,
    }


def _row_metrics_finite(row: dict[str, Any]) -> bool:
    numeric = [
        value
        for key, value in row.items()
        if key not in {"layer", "k", "effective_k"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    return bool(numeric) and all(np.isfinite(value) for value in numeric)


def _merge_report(path: Path, section: str, value: dict[str, Any]) -> None:
    report = json.loads(path.read_text()) if path.is_file() else {"schema_version": SCHEMA_VERSION}
    report[section] = value
    report["status"] = "running"
    atomic_write_json(path, report)


def _require_pass(comparison: dict[str, Any]) -> None:
    if comparison["status"] != "pass":
        raise RuntimeError("integrity comparison failed; inspect report.json")


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ICA Lens integrity reproduction",
        "",
        f"Overall: **{report['status'].upper()}**",
        "",
    ]
    for name in ("activations", "results", "experiments"):
        if name not in report:
            continue
        section = report.get(name, {})
        lines += [f"## {name.replace('_', ' ').title()}", ""]
        for key, value in section.items():
            lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
