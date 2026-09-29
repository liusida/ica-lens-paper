"""Run official SAE suffix ERF for GPT-2, Gemma 2, and Qwen in one display."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from icalens._activation_dataset import ActivationDataset
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.erf_suffix_sweep import _parse_rank_thresholds

ROOT = Path(__file__).resolve().parent
MODELS = {
    "gpt2": ("openai-community/gpt2", "gpt2-pile10k-1m", 12),
    "gemma2": ("google/gemma-2-2b", "gemma-2-2b-pile10k-1m", 26),
    "qwen9b": ("Qwen/Qwen3.5-9B-Base", "qwen3.5-9b-base-pile10k-1m", 32),
}


def _completed_layer(path, model, layer, args):
    run_path = path / "run.json"
    result_path = path / "results" / f"layer_{layer:02d}.json"
    if not run_path.is_file() or not result_path.is_file():
        return False
    run = json.loads(run_path.read_text())
    resolved = run.get("resolved", {})
    result = json.loads(result_path.read_text())
    identity_sha256 = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        result.get("kind") == "result"
        and result.get("identity_sha256") == identity_sha256
        and result.get("layer") == layer
        and resolved.get("model", {}).get("repo_id") == model
        and resolved.get("rank_thresholds") == list(args.rank_thresholds)
        and resolved.get("features_per_layer") == args.features
        and resolved.get("occurrences_per_feature") == args.occurrences
        and resolved.get("seed") == args.seed
        and resolved.get("max_batch_size") == args.max_batch_size
        and resolved.get("batch_token_budget") == args.batch_token_budget
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/home/liusida/Expansion/research/ICA-data/icalens-activations"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sae-suffix-sweep-v2")
    parser.add_argument("--rank-thresholds", type=_parse_rank_thresholds, default=(1, 3, 5, 10, 15))
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--occurrences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--batch-token-budget", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs, units = [], []
    for label in dict.fromkeys(args.models):
        model, cache_name, n_layers = MODELS[label]
        cache = ActivationDataset(args.cache_root / cache_name)
        layers = (
            list(range(n_layers))
            if args.layers == "all"
            else sorted({int(item) for item in args.layers.split(",") if item.strip()})
        )
        if (
            cache.model["repo_id"] != model
            or cache.sample_count != 1_000_000
            or not layers
            or set(layers).difference(cache.available_layers)
        ):
            raise ValueError(f"{label}: incompatible cache or layers")
        output = args.output.resolve() / label
        command = [
            sys.executable,
            "-m",
            "icalens.experiments.erf_sae",
            "--label",
            label,
            "--cache",
            str(cache.path),
            "--layers",
            ",".join(map(str, layers)),
            "--output",
            str(output),
            "--rank-thresholds",
            ",".join(map(str, args.rank_thresholds)),
            "--features",
            str(args.features),
            "--occurrences",
            str(args.occurrences),
            "--seed",
            str(args.seed),
            "--max-batch-size",
            str(args.max_batch_size),
            "--batch-token-budget",
            str(args.batch_token_budget),
        ]
        jobs.append(
            {"label": label, "model": model, "layers": layers, "output": output, "command": command}
        )
        units.extend(f"{label}/layer_{layer:02d}" for layer in layers)
    if args.dry_run:
        print(
            json.dumps(
                {"layer_count": len(units), "model_processes": [job["command"] for job in jobs]},
                indent=2,
            )
        )
        return
    output = args.output.resolve()
    resolved = {
        "protocol": "sae-suffix-sweep-launcher-v2",
        "models": list(dict.fromkeys(args.models)),
        "layers": args.layers,
        "rank_thresholds": list(args.rank_thresholds),
        "features": args.features,
        "occurrences": args.occurrences,
        "seed": args.seed,
        "max_batch_size": args.max_batch_size,
        "batch_token_budget": args.batch_token_budget,
        "units": units,
    }
    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="running")
    completed = {
        f"{job['label']}/layer_{layer:02d}"
        for job in jobs
        for layer in job["layers"]
        if _completed_layer(job["output"], job["model"], layer, args)
    }
    try:
        with ExperimentDisplay(
            output=output / "logs",
            title="SAE suffix ERF",
            total=len(units),
            completed=len(completed),
            completed_unit_ids=completed,
            source_dirty=source.get("dirty"),
            unit_label="layers",
        ) as display:
            warn_if_dirty(source)
            for job in jobs:
                pending = [
                    layer
                    for layer in job["layers"]
                    if f"{job['label']}/layer_{layer:02d}" not in completed
                ]
                if not pending:
                    log(f"{job['label']}: all requested layers complete; skipping model load")
                    continue
                display.phase("Running model", model=job["label"], layer=pending[0])
                child = subprocess.Popen(
                    job["command"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert child.stdout is not None
                for line in child.stdout:
                    print(line, end="")
                    if not line.startswith("ERF_LAYER_COMPLETE "):
                        continue
                    _, label, layer_text, *_ = line.split()
                    layer = int(layer_text)
                    unit = f"{label}/layer_{layer:02d}"
                    if unit not in units or not _completed_layer(
                        job["output"], job["model"], layer, args
                    ):
                        raise ValueError(f"invalid child completion marker: {line.strip()}")
                    completed.add(unit)
                    display.complete_unit(unit, refresh=True)
                    display.phase("Running model", model=label, layer=layer)
                status = child.wait()
                if status:
                    raise subprocess.CalledProcessError(status, job["command"])
            if completed != set(units):
                raise ValueError(f"missing completed layers: {sorted(set(units) - completed)}")
            atomic_write_json(
                output / "launcher-summary.json",
                {"resolved": resolved, "completed": sorted(completed)},
            )
            run.set_status("complete", complete=True)
            log(f"Complete: {len(completed)}/{len(units)} layers")
    except BaseException:
        run.set_status("interrupted")
        raise


if __name__ == "__main__":
    main()
