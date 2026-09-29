#!/usr/bin/env python3
"""Measure representative ICA fitting wall time with the production CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from icalens import ICALens
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ACTIVATION_ROOT = Path("/home/liusida/Expansion/research/ICA-data/icalens-activations")
SPECS = {
    "gpt2": {
        "input": ACTIVATION_ROOT / "gpt2-pile10k-1m",
        "layers": (5, 6),
        "fit_batch_size": 32768,
    },
    "gemma2": {
        "input": ACTIVATION_ROOT / "gemma-2-2b-pile10k-1m",
        "layers": (12, 13),
        "fit_batch_size": 65536,
    },
    "qwen9b": {
        "input": ACTIVATION_ROOT / "qwen3.5-9b-base-pile10k-1m",
        "layers": (15, 16),
        "fit_batch_size": 65536,
    },
}


def valid_result(path: Path, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return (
            result["identity"] == identity
            and result["status"] == "complete"
            and float(result["elapsed_seconds"]) > 0
            and Path(result["lens_output"]).is_dir()
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS), default=tuple(SPECS))
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "fitting-time-results")
    args = parser.parse_args()
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    executable = shutil.which("icalens")
    if executable is None:
        raise RuntimeError("icalens console command is unavailable; run through `uv run python`")

    units: list[tuple[str, int]] = []
    resolved_models: dict[str, Any] = {}
    for model in args.models:
        spec = SPECS[model]
        manifest = spec["input"] / "activations.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or int(metadata["sample_count"]) != 1_000_000:
            raise ValueError(f"expected a complete 1M-token cache: {manifest}")
        resolved_models[model] = {
            "input": str(spec["input"].resolve()),
            "layers": list(spec["layers"]),
            "hidden_size": int(metadata["hidden_size"]),
            "sample_count": int(metadata["sample_count"]),
            "fit_batch_size": int(spec["fit_batch_size"]),
        }
        units.extend((model, layer) for layer in spec["layers"])

    resolved = {
        "format": "icalens.efficiency_fitting_time",
        "schema_version": 1,
        "protocol": "production CLI wall time for one full ICA layer fit",
        "preprocessing": "none",
        "max_iter": args.max_iter,
        "seed": args.seed,
        "models": resolved_models,
    }
    output = args.output.expanduser().resolve()
    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="measuring")
    completed: set[str] = set()
    for model, layer in units:
        unit = f"{model}-layer{layer}"
        identity = {
            "model": model,
            "layer": layer,
            **resolved_models[model],
            "preprocessing": "none",
            "max_iter": args.max_iter,
            "seed": args.seed,
        }
        if valid_result(output / f"{unit}.json", identity):
            completed.add(unit)

    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · fitting-time measurement",
        completed=len(completed),
        total=len(units),
        completed_unit_ids=completed,
        unit_label="layer fits",
        detail_filename="fitting-time-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        for model, layer in units:
            unit = f"{model}-layer{layer}"
            if unit in completed:
                log(f"{unit}: reusing validated timing.")
                continue
            spec = resolved_models[model]
            lens_output = output / "lenses" / unit
            command = [
                executable,
                "fit",
                "activations",
                "--input",
                spec["input"],
                "--layers",
                str(layer),
                "--output",
                str(lens_output),
                "--icalens-preprocessing",
                "none",
                "--max-iter",
                str(args.max_iter),
                "--fit-batch-size",
                str(spec["fit_batch_size"]),
                "--seed",
                str(args.seed),
            ]
            display.phase("Fitting one complete ICA layer", model=model, layer=layer)
            unit_log = output / "logs" / f"{unit}.log"
            started = time.perf_counter()
            with unit_log.open("w", encoding="utf-8") as handle:
                process = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, text=True
                )
            elapsed = time.perf_counter() - started
            if process.returncode != 0:
                raise RuntimeError(f"{unit} failed; see {unit_log}")
            lens = ICALens.from_pretrained(lens_output)
            if tuple(lens.available_layers) != (layer,):
                raise ValueError(
                    f"unexpected fitted layers in {lens_output}: {lens.available_layers}"
                )
            identity = {
                "model": model,
                "layer": layer,
                **spec,
                "preprocessing": "none",
                "max_iter": args.max_iter,
                "seed": args.seed,
            }
            result_path = output / f"{unit}.json"
            atomic_write_json(
                result_path,
                {
                    "identity": identity,
                    "status": "complete",
                    "elapsed_seconds": elapsed,
                    "lens_output": str(lens_output),
                    "command": command,
                    "log": str(unit_log),
                },
            )
            if not valid_result(result_path, identity):
                raise ValueError(f"new timing failed validation: {result_path}")
            log(f"{unit}: {elapsed / 60:.2f} minutes.")
            display.complete_unit(unit, refresh=True)
        run.set_status("complete", complete=True)
        log(f"Fitting-time measurement complete: {output}")


if __name__ == "__main__":
    main()
