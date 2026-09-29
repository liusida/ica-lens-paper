"""Numerical replays for retained downstream experiment artifacts."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from icalens import ICALens
from icalens.experiments._saebench_worker import ICAFeatureEncoder
from icalens.experiments.reconstruction import _aggregate_layer
from icalens.experiments.saebench_sparse_probing import (
    _prepare_layer_baselines,
    _resolve_baselines,
    collect_result_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _report(
    check: str, inputs: list[str], code: str, output_id: str, checks: dict[str, bool], **extra: Any
) -> dict[str, Any]:
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "check": check,
        "status": "pass" if not failed else "fail",
        "scope": {"input_data_ids": inputs, "code_ids": [code], "output_data_ids": [output_id]},
        "checks": checks,
        "failed_checks": failed,
        **extra,
    }


def c20_toy(output: Path) -> dict[str, Any]:
    reference = ROOT / "experiments/toy-example/results/results.json"
    with tempfile.TemporaryDirectory(prefix="icalens-integrity-toy-") as directory:
        generated = Path(directory)
        subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "scipy",
                "python",
                str(ROOT / "experiments/toy-example/scripts/analyze.py"),
                "--capture",
                str(ROOT / "experiments/toy-example/work/selected"),
                "--output",
                str(generated),
                "--figure-output",
                str(generated / "figures"),
                "--b-selection",
                "concept",
                "--b-concept-rank",
                "10",
                "--ica-lens",
                str(ROOT / "experiments/toy-example/work/source/ica-fit/lens"),
                "--force",
            ],
            cwd=ROOT,
            check=True,
        )
        expected, actual = _json(reference), _json(generated / "results.json")
    checks = {
        "selected_target_exact": actual["direction_b"] == expected["direction_b"],
        "statistics_close": all(
            np.isclose(
                actual["statistics"][d][s], expected["statistics"][d][s], rtol=1e-10, atol=1e-12
            )
            for d in expected["statistics"]
            for s in expected["statistics"][d]
        ),
        "coherence_close": all(
            np.isclose(actual["coherence"][k], v, rtol=1e-10, atol=1e-12)
            for k, v in expected["coherence"].items()
        ),
    }
    return _report(
        "C20-toy-example", ["D01", "D02"], "C20", "D20", checks, reference=str(reference)
    )


def c21_reconstruction(output: Path) -> dict[str, Any]:
    root = ROOT / "experiments/reconstruction/official/results/gpt2-context64-all-eval64"
    stored = _json(root / "results.json")
    rows: list[dict[str, Any]] = []
    layer_checks = []
    for path in sorted((root / "layers").glob("layer_*.json")):
        value = _json(path)
        recomputed = _aggregate_layer(int(value["layer"]), value["datasets"])
        layer_checks.append(recomputed == value["rows"])
        rows.extend(recomputed)
    return _report(
        "C21-reconstruction-aggregation-gpt2-context64",
        ["D14", "D11", "D04"],
        "C21",
        "D21",
        {
            "all_layer_metrics_recomputed": all(layer_checks),
            "global_rows_exact": rows == stored["rows"],
        },
        reference=str(root / "results.json"),
    )


def c22_sparse_probing(output: Path) -> dict[str, Any]:
    root = ROOT / "experiments/sparse-probing/official/results/gpt2"
    run = _json(root / "run.json")
    layers = [int(v) for v in run["resolved"]["layers"]]
    actual = collect_result_rows(root, layers)
    expected = _json(root / "results.json")["rows"]
    snapshot = _json(root / "checkpoints/layer_06/snapshot.json")
    tensors = load_file(snapshot["layer_file"], device="cpu")
    center = tensors["center"].to(torch.float32)
    reading = tensors["reading_matrix"].to(torch.float32)
    writing = tensors["writing_matrix"].to(torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(2206)
    activations = center + torch.randn((4, center.numel()), generator=generator)
    raw_scores = (activations - center) @ reading.T
    signed_scores = torch.cat((raw_scores.clamp_min(0), (-raw_scores).clamp_min(0)), dim=-1)
    norms = torch.linalg.vector_norm(writing.T, dim=-1)
    expected_features = signed_scores * torch.cat((norms, norms))
    encoder = ICAFeatureEncoder(snapshot, device="cpu", dtype=torch.float32)
    actual_features = encoder.encode(activations)
    return _report(
        "C22-sparse-probing-aggregation-gpt2",
        ["D01", "D02", "D03", "D11", "D04", "D05"],
        "C22",
        "D22",
        {
            "rows_exact": actual == expected,
            "ica_decoder_rows_unit_norm": torch.allclose(
                encoder.W_dec.norm(dim=-1), torch.ones(encoder.W_dec.shape[0]), atol=1e-6
            ),
            "ica_score_norm_compensation_exact": torch.allclose(
                actual_features, expected_features, rtol=1e-6, atol=1e-6
            ),
        },
        reference=str(root / "results.json"),
    )


def c23_autointerpretability(output: Path) -> dict[str, Any]:
    root = ROOT / "experiments/autointerpretability/runs/gpt2-tinker"
    stored = _json(root / "results.json")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in stored["features"]:
        groups.setdefault((int(row["layer"]), str(row["method"])), []).append(row)
    expected = {(int(row["layer"]), str(row["method"])): row for row in stored["conditions"]}
    checks = {}
    for key, rows in groups.items():
        scores = np.asarray([row["combined_score"] for row in rows], dtype=np.float64)
        condition = expected[key]
        checks[f"layer{key[0]}-{key[1]}"] = (
            len(rows) == int(condition["completed"])
            and int(np.isfinite(scores).sum()) == int(condition["defined_scores"])
            and np.isclose(scores.mean(), float(condition["mean_top_random_score"]), equal_nan=True)
        )
    return _report(
        "C23-autointerpretability-aggregation-gpt2",
        ["D01", "D02", "D03", "D11", "D12", "D04", "D05"],
        "C23",
        "D23",
        checks,
        reference=str(root / "results.json"),
        note=(
            "Recomputes deterministic aggregation; external explainer/simulator "
            "responses remain accepted inputs."
        ),
    )


def c24_ica_erf(output: Path) -> dict[str, Any]:
    root = ROOT / "experiments/effective-receptive-fields/runs/100-per-layer-suffix-sweep-v2"
    stored = _json(root / "summary.json")["rows"]
    index = {(r["model"], int(r["layer"]), int(r["component"]), int(r["top_k"])): r for r in stored}
    checks: dict[str, bool] = {}
    count = 0
    for path in sorted((root / "components/gpt2/layer_06").glob("C*.json")):
        value = _json(path)
        for threshold, summary in value["threshold_results"].items():
            row = index[("gpt2", 6, int(value["component"]), int(threshold))]
            checks[f"C{value['component']}-k{threshold}"] = all(
                np.isclose(float(row[name]), float(summary[name]))
                for name in (
                    "suffix_erf_mean",
                    "suffix_erf_median",
                    "suffix_erf_min",
                    "suffix_erf_max",
                    "recovered_fraction",
                )
            )
            count += 1
    checks["natural_slice_nonempty"] = count > 0
    return _report(
        "C24-ica-erf-aggregation-gpt2-layer6",
        ["D01", "D02", "D03", "D11", "D12"],
        "C24",
        "D24",
        checks,
        reference=str(root / "summary.json"),
    )


def c25_sae_erf(output: Path) -> dict[str, Any]:
    from icalens.experiments.erf_sae import validate_result

    path = (
        ROOT
        / "experiments/effective-receptive-fields/runs/sae-suffix-sweep-v2"
        / "gpt2/results/layer_06.json"
    )
    bundle = _json(path)
    checks: dict[str, bool] = {}
    for feature, value in bundle["results"].items():
        try:
            validate_result(
                value,
                int(value["n_occurrences"]),
                tuple(bundle.get("rank_thresholds", [1, 3, 5, 10, 15])),
            )
            checks[f"F{feature}"] = True
        except (KeyError, TypeError, ValueError):
            checks[f"F{feature}"] = False
    checks["feature_set_exact"] = set(bundle["results"]) == set(bundle["features"])
    return _report(
        "C25-sae-erf-aggregation-gpt2-layer6",
        ["D01", "D02", "D10", "D04"],
        "C25",
        "D24",
        checks,
        reference=str(path),
    )


def c26_seed_stability(output: Path) -> dict[str, Any]:
    path = ROOT / "experiments/fitting-seed-stability/results/iteration-matched-summary.json"
    value = _json(path)
    checks = {}
    names = (
        "mean",
        "median",
        "p05",
        "minimum",
        "fraction_ge_0.90",
        "fraction_ge_0.95",
        "fraction_ge_0.99",
    )
    for comparison in value["comparisons"]:
        key = f"{comparison['iterations_a']}-{comparison['iterations_b']}"
        checks[key] = all(
            np.isclose(comparison[name], np.mean([row[name] for row in comparison["per_seed"]]))
            for name in names
        )
    return _report(
        "C26-fitting-seed-stability-aggregation",
        ["D10", "D11"],
        "C26",
        "D25",
        checks,
        reference=str(path),
    )


def c27_directional_overlap(output: Path) -> dict[str, Any]:
    root = ROOT / "experiments/ica-sae-directional-overlap/results"
    if not torch.cuda.is_available():
        raise RuntimeError("C27 directional-overlap replay requires CUDA")
    script = ROOT / "experiments/ica-sae-directional-overlap/run.py"
    spec = importlib.util.spec_from_file_location("integrity_directional_overlap", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    lens = ICALens.from_pretrained(
        ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k"
    )
    baseline = _prepare_layer_baselines(_resolve_baselines(lens.model_id, "sae"), layer=6)["sae"]
    actual_values, actual_nearest, actual_random, actual_random_nearest = module.measure_layer(
        lens, "gpt2", 6, baseline, "cuda", 128, 0
    )
    reference = root / "gpt2/layer_06.npz"
    with np.load(reference, allow_pickle=False) as archive:
        expected_values = archive["nearest_absolute_cosine"]
        expected_nearest = archive["nearest_sae_feature"]
        expected_random = archive["random_nearest_absolute_cosine"]
        expected_random_nearest = archive["random_nearest_sae_feature"]
    checks = {
        "nearest_cosines_close": np.allclose(actual_values, expected_values, rtol=1e-5, atol=1e-6),
        "nearest_feature_ids_exact": np.array_equal(actual_nearest, expected_nearest),
        "random_cosines_close": np.allclose(actual_random, expected_random, rtol=1e-5, atol=1e-6),
        "random_feature_ids_exact": np.array_equal(actual_random_nearest, expected_random_nearest),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "fragment.npz",
        expected_values=expected_values,
        actual_values=actual_values,
        expected_nearest=expected_nearest,
        actual_nearest=actual_nearest,
        expected_random=expected_random,
        actual_random=actual_random,
        expected_random_nearest=expected_random_nearest,
        actual_random_nearest=actual_random_nearest,
    )
    return _report(
        "C27-directional-overlap-gpt2-layer6",
        ["D11", "D04"],
        "C27",
        "D26",
        checks,
        reference=str(reference),
        note="Recomputes full nearest-SAE and matched-random arrays for one layer.",
    )


def c40_twin_data(output: Path) -> dict[str, Any]:
    source = ROOT / "paper/overleaf-v2.1"
    with tempfile.TemporaryDirectory(prefix="icalens-integrity-paper-") as directory:
        paper = Path(directory) / "paper"
        shutil.copytree(source, paper)
        subprocess.run(
            [
                sys.executable,
                str(paper / "scripts/prepare_data.py"),
                "--figure",
                "ica-sae-directional-overlap",
                "--project-root",
                str(ROOT),
                "--force",
            ],
            cwd=paper,
            check=True,
        )
        names = ("ica-sae-overlap-distribution.npz", "ica-sae-overlap-layers.npz")
        checks = {}
        for name in names:
            with (
                np.load(source / "figures" / name, allow_pickle=False) as expected,
                np.load(paper / "figures" / name, allow_pickle=False) as actual,
            ):
                checks[name] = expected.files == actual.files and all(
                    np.array_equal(expected[k], actual[k]) for k in expected.files
                )
    return _report("C40-twin-data-directional-overlap", ["D26"], "C40", "D30", checks)


def c41_rendering(output: Path) -> dict[str, Any]:
    source = ROOT / "paper/overleaf-v2.1"
    with tempfile.TemporaryDirectory(prefix="icalens-integrity-render-") as directory:
        paper = Path(directory) / "paper"
        shutil.copytree(source, paper)
        script = paper / "scripts/figures/ica_sae_overlap_distribution.py"
        environment = dict(__import__("os").environ)
        environment["PYTHONPATH"] = str(paper / "scripts")
        subprocess.run(
            [sys.executable, str(script), "--force"],
            cwd=paper,
            env=environment,
            check=True,
        )
        expected_png = Path(directory) / "expected.png"
        actual_png = Path(directory) / "actual.png"
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                "150",
                str(source / "figures/ica-sae-overlap-distribution.pdf"),
                str(expected_png.with_suffix("")),
            ],
            check=True,
        )
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                "150",
                str(paper / "figures/ica-sae-overlap-distribution.pdf"),
                str(actual_png.with_suffix("")),
            ],
            check=True,
        )
        checks = {"raster_exact": expected_png.read_bytes() == actual_png.read_bytes()}
    return _report("C41-render-directional-overlap", ["D30"], "C41", "D31", checks)


CHECKS = (
    c20_toy,
    c21_reconstruction,
    c22_sparse_probing,
    c23_autointerpretability,
    c24_ica_erf,
    c25_sae_erf,
    c26_seed_stability,
    c27_directional_overlap,
    c40_twin_data,
    c41_rendering,
)
