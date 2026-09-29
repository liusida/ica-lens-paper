"""Prepare and run the five-representation SAEBench TPP experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from icalens import ICALens
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._saebench_environment import prepare_backend, resolve_backend
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.saebench_sparse_probing import (
    _evaluation_input_protocol,
    _parse_layers,
    _resolve_baselines,
    _write_layer_snapshot,
)

METHODS = ("ica", "unfitted_ica", "sae", "untrained_sae_matched_l0", "pca")
METHOD_DEFINITION_VERSION = 4
DEFAULT_ACTIVATION_CACHE_ROOT = Path("~/Expansion/research/ICA-data/tpp").expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--n-values", default=None, help="Comma-separated feature budgets.")
    parser.add_argument(
        "--methods",
        default="ica,sae",
        help="Comma-separated representations to evaluate, or 'all' (default: ica,sae).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the preset random seed for dataset sampling and probe training.",
    )
    parser.add_argument("--saebench-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--activation-cache-root",
        type=Path,
        default=DEFAULT_ACTIVATION_CACHE_ROOT,
        help=(
            "Root for large transient SAEBench activation caches "
            f"(default: {DEFAULT_ACTIVATION_CACHE_ROOT})."
        ),
    )
    parser.add_argument(
        "--activation-cache-run-dir",
        type=Path,
        default=None,
        help="Reuse an explicit existing run cache directory instead of the derived cache key.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    started_at = time.time()
    args = parse_args()
    source = source_provenance()
    warn_if_dirty(source)
    lens = ICALens.from_pretrained(args.lens)
    layers = _parse_layers(args.layers, lens.available_layers)
    backend = resolve_backend(lens.model_id)
    baselines = _resolve_baselines(lens.model_id, "sae,pca")
    settings = settings_for(args.preset, args.n_values)
    if args.seed is not None:
        settings["random_seed"] = args.seed
    method_argument = ",".join(METHODS) if args.methods.strip().lower() == "all" else args.methods
    methods = tuple(
        dict.fromkeys(value.strip() for value in method_argument.split(",") if value.strip())
    )
    unknown_methods = sorted(set(methods).difference(METHODS))
    if not methods or unknown_methods:
        raise ValueError(
            f"--methods must select from {','.join(METHODS)}; unknown: {unknown_methods}"
        )
    output = args.output.expanduser().resolve()
    fitting_seeds = {
        str(layer): int(lens._get_layer(layer).fitting["random_state"]) for layer in layers
    }
    layer_fingerprints = {str(layer): layer_fingerprint(lens, layer) for layer in layers}
    input_protocol = _evaluation_input_protocol(lens._get_profile(lens._get_layer(layers[0])))
    config = {
        "schema_version": 3,
        "experiment": "tpp-five-representation",
        "lens": str(args.lens),
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "layers": layers,
        "methods": list(methods),
        "method_definition_version": METHOD_DEFINITION_VERSION,
        "ica_definition": "profile-oriented one-sided ReLU feature per ICA component",
        "evaluation_input_protocol": input_protocol,
        "settings": settings,
        "saebench_backend": asdict(backend),
        "baseline_definitions": baselines,
        "fitting_seed_by_layer": fitting_seeds,
        "lens_layer_sha256": layer_fingerprints,
        "untrained_sae_definition": "tied-random-decoder-matched-l0",
        "unfitted_ica_definition": "seeded-fastica-symmetric-decorrelation-after-whitening",
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    output.mkdir(parents=True, exist_ok=True)
    activation_cache_root = args.activation_cache_root.expanduser().resolve()
    cache_key = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    run_cache_dir = (
        args.activation_cache_run_dir.expanduser().resolve()
        if args.activation_cache_run_dir is not None
        else activation_cache_root / _path_slug(lens.model_id) / cache_key
    )
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(run_cache_dir).free
    print(
        f"TPP activation cache: {run_cache_dir} ({free_bytes / 2**30:.1f} GiB free)",
        flush=True,
    )
    atomic_write_json(
        output / "storage.json",
        {
            "schema_version": 1,
            "activation_cache_root": str(activation_cache_root),
            "run_cache_key": cache_key,
            "run_cache_dir": str(run_cache_dir),
            "cache_directory_source": (
                "explicit" if args.activation_cache_run_dir is not None else "derived"
            ),
            "layer_cache_dirs": {
                str(layer): str(run_cache_dir / f"layer_{layer:02d}") for layer in layers
            },
        },
    )
    config_path = output / "config.json"
    validate_or_write(config_path, config)
    run = ResumableRun.open(
        output=output,
        resolved=config,
        source=source,
        status="running",
    )
    completed_layers = {
        layer
        for layer in layers
        if valid_layer_result(
            output / "layers" / f"layer_{layer:02d}" / "result.json",
            settings=settings,
            methods=methods,
        )
    }
    if len(completed_layers) == len(layers):
        run.set_status("complete", complete=True)
        print(f"All {len(layers)} layer(s) are already complete.")
        return
    prepared = prepare_backend(
        backend,
        cache_dir=args.cache_dir,
        saebench_path=args.saebench_path,
        refresh=False,
    )
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · targeted probe perturbation",
        completed=len(completed_layers),
        total=len(layers),
        source_dirty=bool(source.get("dirty")),
        unit_label="layers",
        started_at=started_at,
        completed_unit_ids=completed_layers,
    )
    try:
        with display:
            for layer in layers:
                if layer in completed_layers:
                    continue
                display.phase("Evaluating five representations", Layer=layer)
                layer_dir = output / "layers" / f"layer_{layer:02d}"
                snapshot = _write_layer_snapshot(
                    lens,
                    layer=layer,
                    output=output / "checkpoints" / f"layer_{layer:02d}",
                    saebench_model_name=backend.saebench_model_name,
                    baselines=baselines,
                )
                snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
                snapshot_payload["fitting_seed"] = fitting_seeds[str(layer)]
                snapshot_payload["ica_feature_sides"] = "profile_oriented_positive_half_wave"
                snapshot_payload["ica_activation"] = "relu"
                atomic_write_json(snapshot, snapshot_payload)
                layer_cache_dir = run_cache_dir / f"layer_{layer:02d}"
                command = [
                    str(prepared.python),
                    str(Path(__file__).with_name("worker.py")),
                    "--saebench-root",
                    str(prepared.root),
                    "--snapshot",
                    str(snapshot),
                    "--config",
                    str(config_path),
                    "--output",
                    str(layer_dir),
                    "--activation-cache",
                    str(layer_cache_dir),
                ]
                print("RUN " + " ".join(command), flush=True)
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
                display.complete_unit(layer, refresh=True)
    except BaseException:
        run.set_status("failed")
        raise
    else:
        run.set_status("complete", complete=True)


def valid_layer_result(
    path: Path, *, settings: dict[str, object], methods: tuple[str, ...]
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("method_definition_version") != METHOD_DEFINITION_VERSION
        ):
            return False
        result_methods = payload["methods"]
        expected = {f"{name}_custom_sae" for name in methods}
        if set(result_methods) != expected:
            return False
        for result in result_methods.values():
            config = result["eval_config"]
            if list(config["dataset_names"]) != list(settings["datasets"]):
                return False
            if list(config["n_values"]) != list(settings["n_values"]):
                return False
            if int(config["train_set_size"]) != int(settings["train_size"]):
                return False
            if int(config["test_set_size"]) != int(settings["test_size"]):
                return False
            metrics = result["eval_result_metrics"]["tpp_metrics"]
            for budget in settings["n_values"]:
                for suffix in ("total_metric", "intended_diff_only", "unintended_diff_only"):
                    value = metrics[f"tpp_threshold_{int(budget)}_{suffix}"]
                    if value is None or not isinstance(value, (int, float)):
                        return False
        return set(payload["feature_configs"]) == set(methods)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def layer_fingerprint(lens: ICALens, layer: int) -> str:
    artifact = lens._get_layer(layer)
    digest = hashlib.sha256()
    for value in (artifact.center, artifact.reading_matrix, artifact.writing_matrix):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    profile = lens._get_profile(artifact)
    directions = {
        int(component["component"]): component.get("tail_direction")
        for component in profile.get("components", [])
    }
    digest.update(json.dumps(directions, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _path_slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip(
        "-"
    )


def settings_for(preset: str, n_values: str | None) -> dict[str, object]:
    if preset == "smoke":
        settings: dict[str, object] = {
            "datasets": ["canrager/amazon_reviews_mcauley_1and5"],
            "n_values": [1, 2, 5, 10],
            "train_size": 200,
            "test_size": 100,
            "context_length": 128,
            "probe_epochs": 4,
            "llm_batch_size": 1,
            "sae_batch_size": 64,
            "llm_dtype": "float32",
            "random_seed": 42,
        }
    else:
        settings = {
            "datasets": [
                "LabHC/bias_in_bios_class_set1",
                "canrager/amazon_reviews_mcauley_1and5",
            ],
            "n_values": [1, 2, 5, 10, 20, 50, 100],
            "train_size": 4000,
            "test_size": 1000,
            "context_length": 128,
            "probe_epochs": 20,
            "llm_batch_size": 1,
            "sae_batch_size": 125,
            "llm_dtype": "float32",
            "random_seed": 42,
        }
    if n_values:
        settings["n_values"] = [int(value) for value in n_values.split(",")]
    return settings


def validate_or_write(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"incompatible existing run configuration: {path}")
        return
    atomic_write_json(path, payload)


if __name__ == "__main__":
    main()
