#!/usr/bin/env python3
"""Build the complete four-layer fixed-panel experiment in one run directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from icalens.experiments._run import atomic_write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "runs/fixed-panel"
DEFAULT_ARCHIVE = Path(
    "/media/liusida/Expansion/research/ICA-data/"
    "fitting-autointerpretability-convergence/fixed-panel"
)
DEFAULT_CACHES = (
    ROOT / "runs/qwen-layers-07-23/trajectory",
    ROOT / "runs/qwen-layers-15-31/trajectory",
)
LAYERS = (7, 15, 23, 31)
STAGES = ("trajectory", "cohort", "activations", "fixed-panel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--trajectory-cache",
        type=Path,
        action="append",
        help="Existing trajectory root to import; repeat for multiple layer groups",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--statistics-batch-size", type=int, default=16_384)
    parser.add_argument(
        "--n-components",
        type=int,
        default=100,
        help="Prefix-stable cohort capacity to prepare per layer (default: 100).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", choices=STAGES, default="fixed-panel")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool = False) -> None:
    print("RUN", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def read_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"trajectory cache is incomplete: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format") != "icalens.fastica_trajectory":
        raise ValueError(f"not a FastICA trajectory: {summary_path}")
    return summary


def common_configuration(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key not in {"layers", "layer_results"}}


def import_file(source: Path, destination: Path) -> str:
    """Atomically hard-link an immutable artifact, falling back to a copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"existing imported file has the wrong size: {destination}")
        return "reused"
    temporary = destination.with_name(destination.name + ".importing")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, temporary)
        method = "copy"
    os.replace(temporary, destination)
    return method


def merge_trajectories(caches: tuple[Path, ...], destination: Path) -> dict[str, Any]:
    summaries = [(cache, read_summary(cache)) for cache in caches]
    baseline = common_configuration(summaries[0][1])
    layers_seen: dict[int, Path] = {}
    layer_results: dict[int, dict[str, Any]] = {}
    import_methods: set[str] = set()
    for cache, summary in summaries:
        if common_configuration(summary) != baseline:
            raise ValueError(f"trajectory configuration differs: {cache}")
        results = {int(row["layer"]): row for row in summary["layer_results"]}
        if set(results) != set(map(int, summary["layers"])):
            raise ValueError(f"trajectory layer summary is inconsistent: {cache}")
        for layer in map(int, summary["layers"]):
            if layer in layers_seen:
                raise ValueError(f"layer {layer} appears in multiple trajectory caches")
            layers_seen[layer] = cache
            layer_results[layer] = results[layer]
            shared = cache / "shared" / f"layer-{layer:02d}.safetensors"
            layer_record = cache / "layers" / f"layer-{layer:02d}.json"
            if not shared.is_file() or not layer_record.is_file():
                raise FileNotFoundError(f"trajectory cache is missing layer {layer} artifacts")
            import_methods.add(import_file(shared, destination / "shared" / shared.name))
            import_methods.add(
                import_file(layer_record, destination / "layers" / layer_record.name)
            )
            checkpoints = sorted(
                (cache / "checkpoints" / f"layer-{layer:02d}").glob("*.safetensors")
            )
            if len(checkpoints) != len(summary["checkpoints"]):
                raise ValueError(f"trajectory cache has incomplete checkpoints for layer {layer}")
            for checkpoint in checkpoints:
                import_methods.add(
                    import_file(
                        checkpoint,
                        destination / "checkpoints" / f"layer-{layer:02d}" / checkpoint.name,
                    )
                )
    if set(layers_seen) != set(LAYERS):
        raise ValueError(f"trajectory caches cover {sorted(layers_seen)}, expected {list(LAYERS)}")
    merged = {
        **baseline,
        "layers": list(LAYERS),
        "layer_results": [layer_results[layer] for layer in LAYERS],
    }
    existing_summary = destination / "summary.json"
    if existing_summary.is_file():
        existing = json.loads(existing_summary.read_text(encoding="utf-8"))
        if existing != merged:
            raise ValueError(f"existing merged trajectory is incompatible: {existing_summary}")
    else:
        atomic_write_json(existing_summary, merged)
    record = {
        "status": "complete",
        "format": "icalens.merged-fastica-trajectory",
        "layers": list(LAYERS),
        "sources": [str(path) for path, _ in summaries],
        "import_methods": sorted(import_methods),
    }
    atomic_write_json(destination / "import.json", record)
    return record


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    trajectory = output / "trajectory"
    cohort = output / "cohort.json"
    checkpoint_prepared = output / "checkpoint-prepared"
    fixed_prepared = output / "prepared"
    if args.trajectory_cache is not None:
        cache_arguments = tuple(args.trajectory_cache)
    elif (trajectory / "summary.json").is_file():
        cache_arguments = ()
    else:
        cache_arguments = tuple(path for path in DEFAULT_CACHES if path.is_dir())
        if len(cache_arguments) == 1:
            raise ValueError(
                "only one default trajectory cache exists; provide both with repeated "
                "--trajectory-cache, or remove the partial cache set to fit from scratch"
            )
    caches = tuple(path.expanduser().resolve() for path in cache_arguments)
    configuration = {
        "format": "icalens.fitting-autointerpretability-fixed-panel-run",
        "schema_version": 1,
        "layers": list(LAYERS),
        "output": str(output),
        "archive": str(archive),
        "batch_size": args.batch_size,
        "statistics_batch_size": args.statistics_batch_size,
        "device": args.device,
        "n_components_per_layer": args.n_components,
        "selection_protocol": "checkpoint-specific-intrinsic-5-train-20-valid",
    }
    if args.dry_run:
        print(json.dumps(configuration, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "fixed-panel-run.json"
    if run_path.is_file():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        existing_resolved = dict(existing.get("resolved", {}))
        # Cache locations affect only how immutable trajectory files arrive, not
        # the numerical run. Accept and migrate manifests from the first version.
        existing_resolved.pop("trajectory_caches", None)
        if existing_resolved != configuration:
            raise ValueError(f"incompatible existing fixed-panel run: {run_path}")
    atomic_write_json(
        output / "fixed-panel-run.json", {"status": "running", "resolved": configuration}
    )
    try:
        if (trajectory / "summary.json").is_file():
            summary = read_summary(trajectory)
            if summary.get("layers") != list(LAYERS):
                raise ValueError(
                    f"existing trajectory does not contain all four layers: {trajectory}"
                )
            print(f"PASS unified trajectory already exists: {trajectory}")
        elif caches:
            print("START importing four-layer trajectory", flush=True)
            merge_trajectories(caches, trajectory)
            print(f"PASS unified trajectory: {trajectory}")
        else:
            run(
                [
                    sys.executable,
                    str(ROOT / "trajectory.py"),
                    "--layers",
                    ",".join(map(str, LAYERS)),
                    "--output",
                    str(trajectory),
                ]
            )
        if args.stop_after == "trajectory":
            atomic_write_json(
                run_path, {"status": "trajectory-complete", "resolved": configuration}
            )
            return

        if cohort.is_file():
            existing_cohort = json.loads(cohort.read_text(encoding="utf-8"))
            existing_count = existing_cohort.get("n_components_per_layer")
            if existing_count != args.n_components:
                raise ValueError(
                    f"existing cohort has {existing_count} components, requested "
                    f"{args.n_components}; remove stale preparation/cohort artifacts "
                    "before changing prepared capacity"
                )
            print(f"PASS unified cohort already exists: {cohort}")
        else:
            run(
                [
                    sys.executable,
                    str(ROOT / "select_cohort.py"),
                    "--layers",
                    ",".join(map(str, LAYERS)),
                    "--trajectory",
                    str(trajectory),
                    "--output",
                    str(cohort),
                    "--n-components",
                    str(args.n_components),
                    "--device",
                    args.device,
                ]
            )
        if args.stop_after == "cohort":
            atomic_write_json(run_path, {"status": "cohort-complete", "resolved": configuration})
            return

        run(
            [
                sys.executable,
                str(ROOT / "prepare.py"),
                "--layers",
                ",".join(map(str, LAYERS)),
                "--trajectory",
                str(trajectory),
                "--cohort",
                str(cohort),
                "--output",
                str(checkpoint_prepared),
                "--archive",
                str(archive),
                "--batch-size",
                str(args.batch_size),
                "--statistics-batch-size",
                str(args.statistics_batch_size),
                "--device",
                args.device,
            ]
        )
        if args.stop_after == "activations":
            atomic_write_json(
                run_path, {"status": "activations-complete", "resolved": configuration}
            )
            return

        run(
            [
                sys.executable,
                str(ROOT / "prepare_intrinsic_panel.py"),
                "--layers",
                ",".join(map(str, LAYERS)),
                "--input",
                str(checkpoint_prepared),
                "--output",
                str(fixed_prepared),
            ]
        )
        atomic_write_json(
            output / "fixed-panel-run.json", {"status": "complete", "resolved": configuration}
        )
        print(f"PASS four-layer intrinsic preparation: {fixed_prepared}")
    except BaseException:
        atomic_write_json(
            output / "fixed-panel-run.json", {"status": "interrupted", "resolved": configuration}
        )
        raise


if __name__ == "__main__":
    main()
