"""Fit GPT-2 layer 6 repeatedly while varying only FastICA initialization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_ACTIVATIONS = Path(
    "/home/liusida/Expansion/research/ICA-data/icalens-activations/gpt2-pile10k-1m"
)
DEFAULT_REFERENCE_LENS = Path(
    "local-icalens-models/official/icalens-gpt2-small-pile10k"
)
DEFAULT_OUTPUT = Path("pilot-experiments/fitting-seed-stability/output")
EXPECTED_MODEL = "openai-community/gpt2"
EXPECTED_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXPECTED_DATASET = "NeelNanda/pile-10k"
EXPECTED_DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
LAYER = 6
TOKEN_COUNT = 1_000_000
DEFAULT_ITERATIONS = 50
FIT_BATCH_SIZE = 32_768
PREPROCESSING = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--reference-lens", type=Path, default=DEFAULT_REFERENCE_LENS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--iterations", default=str(DEFAULT_ITERATIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activations = args.activations.expanduser().resolve()
    reference_lens = args.reference_lens.expanduser().resolve()
    output = args.output.expanduser().resolve()
    seeds = _parse_seeds(args.seeds)
    iterations = _parse_iterations(args.iterations)
    _validate_completed_lens(
        _read_json(reference_lens / "icalens.json"),
        seed=0,
        iterations=DEFAULT_ITERATIONS,
    )
    activation_manifest = _read_json(activations / "activations.json")
    _validate_activation_cache(activation_manifest)

    executable = Path(sys.executable).with_name("icalens")
    for seed in seeds:
        for max_iter in iterations:
            lens_path = output / f"seed-{seed}-iter-{max_iter}"
            manifest_path = lens_path / "icalens.json"
            if manifest_path.is_file():
                _validate_completed_lens(
                    _read_json(manifest_path), seed=seed, iterations=max_iter
                )
                print(
                    f"Seed {seed}, {max_iter} iterations: "
                    "valid completed fit; skipping.",
                    flush=True,
                )
                continue
            _run(
                [
                    str(executable),
                    "fit",
                    "activations",
                    "--input",
                    str(activations),
                    "--layers",
                    str(LAYER),
                    "--output",
                    str(lens_path),
                    "--icalens-preprocessing",
                    PREPROCESSING,
                    "--max-iter",
                    str(max_iter),
                    "--fit-batch-size",
                    str(FIT_BATCH_SIZE),
                    "--seed",
                    str(seed),
                ]
            )
            _validate_completed_lens(
                _read_json(manifest_path), seed=seed, iterations=max_iter
            )


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain unique non-negative integers")
    return seeds


def _parse_iterations(value: str) -> tuple[int, ...]:
    try:
        iterations = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise ValueError("--iterations must be comma-separated integers") from error
    if (
        not iterations
        or len(set(iterations)) != len(iterations)
        or any(iteration <= 0 for iteration in iterations)
    ):
        raise ValueError("--iterations must contain unique positive integers")
    return iterations


def _validate_activation_cache(manifest: dict[str, Any]) -> None:
    provenance = manifest.get("provenance", {})
    dataset = provenance.get("dataset", {})
    model = manifest.get("model", {})
    layer = manifest.get("layers", {}).get(str(LAYER), {})
    expected = {
        "format": (manifest.get("format"), "icalens.activations"),
        "format_version": (manifest.get("format_version"), 1),
        "status": (manifest.get("status"), "complete"),
        "sample_count": (manifest.get("sample_count"), TOKEN_COUNT),
        "hidden_size": (manifest.get("hidden_size"), 768),
        "dtype": (manifest.get("dtype"), "bfloat16"),
        "activation_site": (manifest.get("activation_site"), "resid_post"),
        "layer_indexing": (
            manifest.get("layer_indexing"),
            "transformer_blocks_zero_based",
        ),
        "model.repo_id": (model.get("repo_id"), EXPECTED_MODEL),
        "model.revision": (model.get("revision"), EXPECTED_MODEL_REVISION),
        "model.type": (model.get("type"), "base"),
        "dataset.repo_id": (dataset.get("repo_id"), EXPECTED_DATASET),
        "dataset.revision": (dataset.get("revision"), EXPECTED_DATASET_REVISION),
        "dataset.split": (dataset.get("split"), "train"),
        "candidate_tokens": (provenance.get("candidate_tokens"), 5_465_620),
        "context_length": (provenance.get("context_length"), 1024),
        "fitting_tokens": (provenance.get("fitting_tokens"), TOKEN_COUNT),
        "sampling_seed": (provenance.get("sampling_seed"), 0),
        "text_field": (provenance.get("text_field"), "text"),
        "token_scope": (provenance.get("token_scope"), "all"),
        "layer.status": (layer.get("status"), "complete"),
        "layer.shape": (layer.get("shape"), [TOKEN_COUNT, 768]),
    }
    _require_matches("activation cache", expected)


def _validate_completed_lens(
    manifest: dict[str, Any], *, seed: int, iterations: int
) -> None:
    layer = manifest.get("layers", {}).get(str(LAYER), {})
    fitting = layer.get("fitting", {})
    model = manifest.get("model", {})
    expected = {
        "activation_site": (manifest.get("activation_site"), "resid_post"),
        "layer_indexing": (
            manifest.get("layer_indexing"),
            "transformer_blocks_zero_based",
        ),
        "hidden_size": (manifest.get("hidden_size"), 768),
        "model.repo_id": (model.get("repo_id"), EXPECTED_MODEL),
        "model.revision": (model.get("revision"), EXPECTED_MODEL_REVISION),
        "model.type": (model.get("type"), "base"),
        "input_preprocessing.icalens_preprocessing": (
            manifest.get("input_preprocessing", {}).get("icalens_preprocessing"),
            PREPROCESSING,
        ),
        "n_components": (layer.get("n_components"), 768),
        "algorithm": (fitting.get("ica_algorithm"), "parallel"),
        "fun": (fitting.get("fun"), "logcosh"),
        "whiten": (fitting.get("whiten"), "unit-variance"),
        "whiten_solver": (fitting.get("whiten_solver"), "eigh"),
        "source_scaling": (fitting.get("source_scaling"), "none"),
        "max_iter": (fitting.get("max_iter"), iterations),
        "stopping_criterion": (fitting.get("stopping_criterion"), "fixed_iterations"),
        "n_iter": (fitting.get("n_iter"), iterations),
        "objective_every": (fitting.get("objective_every"), 1),
        "batch_size": (fitting.get("batch_size"), FIT_BATCH_SIZE),
        "n_samples": (fitting.get("n_samples"), TOKEN_COUNT),
        "input_dtype": (fitting.get("input_dtype"), "bfloat16"),
        "memory_strategy": (fitting.get("memory_strategy"), "blockwise_multi_pass"),
        "stored_dtype": (fitting.get("stored_dtype"), "float32"),
        "random_state": (fitting.get("random_state"), seed),
    }
    _require_matches(f"seed {seed} Lens", expected)


def _require_matches(label: str, values: dict[str, tuple[Any, Any]]) -> None:
    mismatches = [
        f"{key}: {actual!r} != {expected!r}"
        for key, (actual, expected) in values.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(f"incompatible {label}: " + "; ".join(mismatches))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
