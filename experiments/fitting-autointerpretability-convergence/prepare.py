#!/usr/bin/env python3
"""Prepare matched trajectory rows for the existing autointerp evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from safetensors.torch import load_file
from select_cohort import LAYERS, checkpoint_path, validate_trajectory
from trajectory import parse_layers
from transformers import AutoModelForCausalLM

from icalens._activation_dataset import ActivationDataset
from icalens._capture import transformer_blocks
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.autointerpretability_protocol import FRAGMENT_LENGTH, select_record_indices

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = ROOT / "runs/trajectory"
DEFAULT_COHORT = ROOT / "runs/cohort.json"
DEFAULT_FRAGMENTS = ROOT.parent / "autointerpretability/runs/qwen3.5-9b/fragments.jsonl"
DEFAULT_OUTPUT = ROOT / "runs/prepared"
DEFAULT_ARCHIVE = Path(
    "/home/liusida/Expansion/research/ICA-data/"
    "fitting-autointerpretability-convergence"
)
EVALUATED_CHECKPOINTS = (0, 1, 2, 3, 5, 7, 10, 20, 50, 100, 200)
MODEL_ID = "Qwen/Qwen3.5-9B-Base"
MODEL_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
N_FRAGMENTS, PROGRESS_INTERVAL = 50_000, 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--layers", default=",".join(map(str, LAYERS)))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--statistics-batch-size", type=int, default=16_384)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read_fragments(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    invalid_length = any(
        len(row.get("token_ids", ())) != FRAGMENT_LENGTH for row in rows
    )
    if len(rows) != N_FRAGMENTS or invalid_length:
        raise ValueError(f"expected {N_FRAGMENTS} fragments of length {FRAGMENT_LENGTH}: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_directions(trajectory: Path, layer: int, rows: list[int], device: str):
    shared = load_file(trajectory / "shared" / f"layer-{layer:02d}.safetensors")
    whitening = shared["whitening"].to(device=device, dtype=torch.float32)
    center = shared["center"].to(device=device, dtype=torch.float32)
    previous = None
    chosen = {}
    indices = torch.tensor(rows)
    for iteration in validate_trajectory(trajectory)["checkpoints"]:
        unmixing = load_file(checkpoint_path(trajectory, layer, iteration))["unmixing"]
        reading = unmixing.index_select(0, indices).to(device) @ whitening
        normalized = torch.nn.functional.normalize(reading, dim=1)
        if previous is not None:
            sign = torch.where((normalized * previous).sum(1) < 0, -1.0, 1.0)
            reading, normalized = reading * sign[:, None], normalized * sign[:, None]
        if iteration in EVALUATED_CHECKPOINTS:
            chosen[iteration] = reading
        previous = normalized
    return chosen, center


def tail_signs(dataset, layer, reading, center, batch_size, device):
    raw = dataset.layer(layer)
    moments = torch.zeros((3, reading.shape[0]), dtype=torch.float64)
    positive_squared = torch.zeros(reading.shape[0], dtype=torch.float64)
    for start in range(0, dataset.sample_count, batch_size):
        x = raw[start : start + batch_size].to(device=device, dtype=torch.float32)
        scores = ((x - center) @ reading.T).to(torch.float64)
        moments[0] += scores.sum(0).cpu()
        moments[1] += scores.square().sum(0).cpu()
        moments[2] += scores.pow(3).sum(0).cpu()
        positive_squared += scores.clamp_min(0).square().sum(0).cpu()
    mean, second, third = moments / dataset.sample_count
    variance = (second - mean.square()).clamp_min(0)
    central3 = third - 3 * mean * second + 2 * mean.pow(3)
    skew = torch.where(variance > 0, central3 / variance.pow(1.5), torch.zeros_like(variance))
    positive_fraction = positive_squared / moments[1].clamp_min(
        torch.finfo(torch.float64).tiny
    )
    signs = torch.where(
        skew > 0,
        1.0,
        torch.where(
            skew < 0,
            -1.0,
            torch.where(positive_fraction >= 0.5, 1.0, -1.0),
        ),
    )
    return signs.to(device=device, dtype=torch.float32), skew.tolist()


def cache_path(archive, iteration, layer):
    return archive / f"iter-{iteration:03d}" / f"layer-{layer:02d}" / "candidate_activations.npy"


def open_stores(archive, start, n_features):
    result = {}
    for iteration in EVALUATED_CHECKPOINTS:
        for layer in LAYERS:
            partial = cache_path(archive, iteration, layer).with_suffix(".npy.partial")
            partial.parent.mkdir(parents=True, exist_ok=True)
            if start:
                if not partial.is_file():
                    raise FileNotFoundError(f"missing resume cache: {partial}")
                result[(iteration, layer)] = np.load(partial, mmap_mode="r+")
            else:
                result[(iteration, layer)] = np.lib.format.open_memmap(
                    partial, mode="w+", dtype=np.float16,
                    shape=(N_FRAGMENTS, FRAGMENT_LENGTH, n_features),
                )
    return result


def write_metadata(output, archive, fragments, cohort_path, cohort, n_features):
    for iteration in EVALUATED_CHECKPOINTS:
        root = output / f"iter-{iteration:03d}"
        root.mkdir(parents=True, exist_ok=True)
        fragment_link = root / "fragments.jsonl"
        if not fragment_link.exists():
            fragment_link.symlink_to(fragments)
        for layer in LAYERS:
            directory = root / f"layer_{layer:02d}" / "ica"
            directory.mkdir(parents=True, exist_ok=True)
            source = cache_path(archive, iteration, layer)
            link = directory / "candidate_activations.npy"
            if not link.exists():
                link.symlink_to(source)
            values = np.load(source, mmap_mode="r")
            accepted = []
            ids = cohort["layers"][str(layer)]["row_ids"]
            for position, feature in enumerate(ids):
                train, top, random = select_record_indices(
                    np.asarray(values[:, :, position], dtype=np.float32), seed=int(feature)
                )
                accepted.append({"feature": int(feature), "candidate_position": position,
                                 "train_top": train, "valid_top": top, "valid_random": random})
            atomic_write_json(directory / "selection.json",
                              {"candidate_ids": ids, "accepted": accepted, "rejected": []})
            atomic_write_json(root / f"layer_{layer:02d}" / "prepared.json",
                              {"format": "icalens.autointerpretability-prepared-layer",
                               "schema_version": 1, "layer": layer, "methods": ["ica"],
                               "n_fragments": N_FRAGMENTS, "n_features": n_features})
        atomic_write_json(root / "run.json", {"status": "prepared", "resolved": {
            "format": "icalens.autointerpretability-trajectory", "schema_version": 1,
            "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
            "activation_site": "resid_post", "layer_indexing": "transformer_blocks_zero_based",
            "layers": list(LAYERS), "n_fragments": N_FRAGMENTS,
            "fragment_length": FRAGMENT_LENGTH, "n_features": n_features,
            "iteration": iteration, "cohort": str(cohort_path)}})


def main() -> None:
    global LAYERS
    args = parse_args()
    LAYERS = parse_layers(args.layers)
    if min(args.batch_size, args.statistics_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    trajectory, cohort_path = args.trajectory.resolve(), args.cohort.resolve()
    fragments_path = args.fragments.resolve()
    output = args.output.resolve()
    archive = args.archive.resolve()
    summary = validate_trajectory(trajectory, LAYERS)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    counts = {len(cohort["layers"][str(layer)]["row_ids"]) for layer in LAYERS}
    if len(counts) != 1:
        raise ValueError("cohort layers must contain the same number of rows")
    n_features = counts.pop()
    if n_features < 1:
        raise ValueError("cohort must contain at least one component per layer")
    resolved = {
        "format": "icalens.fastica_autointerpretability_preparation",
        "format_version": 1,
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "trajectory": str(trajectory),
        "trajectory_summary_sha256": sha256(trajectory / "summary.json"),
        "cohort": str(cohort_path),
        "cohort_sha256": sha256(cohort_path) if cohort_path.is_file() else None,
        "layers": list(LAYERS),
        "checkpoints": list(EVALUATED_CHECKPOINTS),
        "features_per_layer": n_features,
        "fragments": str(fragments_path),
        "fragments_sha256": sha256(fragments_path),
        "n_fragments": N_FRAGMENTS,
        "fragment_length": FRAGMENT_LENGTH,
        "statistics_batch_size": args.statistics_batch_size,
        "archive": str(archive),
        "estimated_cache_bytes": (
            len(LAYERS)
            * len(EVALUATED_CHECKPOINTS)
            * N_FRAGMENTS
            * FRAGMENT_LENGTH
            * n_features
            * 2
        ),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    fragments = read_fragments(fragments_path)
    source = source_provenance()
    run = ResumableRun.open(
        output=output,
        filename="preparation-run.json",
        resolved=resolved,
        source=source,
        status="preparing",
    )
    progress_path = archive / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.is_file()
        else {"completed_fragments": 0, "resolved": resolved}
    )
    if progress["resolved"] != resolved:
        raise ValueError(f"incompatible progress: {progress_path}")
    start_at = int(progress["completed_fragments"])
    tails_path = archive / "tails.json"
    stored_tails = json.loads(tails_path.read_text()) if tails_path.is_file() else None
    tail_layers = set(stored_tails.get("layers", {})) if stored_tails else set()
    final_caches_exist = all(
        cache_path(archive, iteration, layer).is_file()
        for iteration in EVALUATED_CHECKPOINTS
        for layer in LAYERS
    )
    summary_path = archive / "summary.json"
    stored_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    metadata_complete = stored_summary == {**resolved, "status": "complete"} and final_caches_exist
    chunk_count = math.ceil(N_FRAGMENTS / PROGRESS_INTERVAL)
    completed_chunks = math.ceil(start_at / PROGRESS_INTERVAL) if start_at else 0
    completed_units = len(tail_layers) + completed_chunks + int(metadata_complete)
    total_units = len(LAYERS) + chunk_count + 1
    try:
        with ExperimentDisplay(
            output=output / "logs",
            title="ICA Lens · fitting–autointerpretability preparation",
            completed=completed_units,
            total=total_units,
            source_dirty=source.get("dirty"),
            unit_label="durable units",
            detail_filename="preparation-detail.log",
        ) as display:
            warn_if_dirty(source)
            if metadata_complete:
                display.set_outcome("reused")
                log("All matched activation caches are complete; reused existing preparation.")
                run.set_status("complete", complete=True)
                return
            dataset = ActivationDataset(Path(summary["activations"]))
            directions, centers = {}, {}
            tail_data = stored_tails or {"resolved": resolved, "layers": {}}
            if tail_data.get("resolved") != resolved:
                raise ValueError(f"incompatible tail checkpoint: {tails_path}")
            for layer in LAYERS:
                display.phase("Orienting selected rows", layer=layer)
                ids = [int(x) for x in cohort["layers"][str(layer)]["row_ids"]]
                selected, center = aligned_directions(trajectory, layer, ids, args.device)
                record = tail_data["layers"].get(str(layer))
                if record is None:
                    log(f"Computing iteration-200 population tails for layer {layer}.")
                    tails, skew = tail_signs(
                        dataset, layer, selected[200], center,
                        args.statistics_batch_size, args.device,
                    )
                    record = {
                        "row_ids": ids,
                        "iteration_200_skewness": skew,
                        "tail_sign": tails.to(torch.int8).cpu().tolist(),
                    }
                    tail_data["layers"][str(layer)] = record
                    atomic_write_json(tails_path, tail_data)
                    display.advance(refresh=True)
                    log(f"Checkpointed population tails for layer {layer}.")
                tails = torch.tensor(record["tail_sign"], device=args.device)
                directions[layer] = {
                    iteration: value * tails[:, None]
                    for iteration, value in selected.items()
                }
                centers[layer] = center
            if start_at == N_FRAGMENTS and final_caches_exist:
                display.phase("Recovering finalized caches")
                write_metadata(output, archive, fragments_path, cohort_path, cohort, n_features)
                atomic_write_json(
                    archive / "summary.json", {**resolved, "status": "complete"}
                )
                display.advance(refresh=True)
                run.set_status("complete", complete=True)
                log("Recovered caches finalized before the previous interruption.")
                return
            if start_at == N_FRAGMENTS:
                display.phase("Finalizing encoded caches")
                for iteration in EVALUATED_CHECKPOINTS:
                    for layer in LAYERS:
                        final = cache_path(archive, iteration, layer)
                        partial = final.with_suffix(".npy.partial")
                        if not final.is_file() and partial.is_file():
                            os.replace(partial, final)
                if not all(
                    cache_path(archive, iteration, layer).is_file()
                    for iteration in EVALUATED_CHECKPOINTS
                    for layer in LAYERS
                ):
                    raise ValueError("encoded cache finalization is incomplete")
                write_metadata(output, archive, fragments_path, cohort_path, cohort, n_features)
                atomic_write_json(
                    archive / "summary.json", {**resolved, "status": "complete"}
                )
                display.advance(refresh=True)
                run.set_status("complete", complete=True)
                log("Recovered and finalized all encoded caches.")
                return
            stores = open_stores(archive, start_at, n_features)
            display.phase("Loading Qwen", model=MODEL_ID)
            model = cast(torch.nn.Module, load_model_to_cuda(
                AutoModelForCausalLM, MODEL_ID, revision=MODEL_REVISION,
                device=args.device, dtype=torch.bfloat16, touch="auto",
                low_cpu_mem_usage=True,
            ))
            model.eval()
            captured, handles = {}, []
            stacked = {
                layer: torch.cat([
                    directions[layer][iteration] for iteration in EVALUATED_CHECKPOINTS
                ])
                for layer in LAYERS
            }
            blocks = transformer_blocks(model)
            for layer in LAYERS:
                def hook(_m: Any, _a: Any, value: Any, layer: int = layer) -> None:
                    captured[layer] = (
                        value[0] if isinstance(value, tuple) else value
                    ).detach()
                handles.append(blocks[layer].register_forward_hook(hook))
            try:
                display.phase("Encoding fragments", fragments=f"{start_at}/{N_FRAGMENTS}")
                log(f"Starting or resuming fragment encoding at {start_at}/{N_FRAGMENTS}.")
                checkpoint_at = start_at
                for start in range(start_at, N_FRAGMENTS, args.batch_size):
                    stop = min(start + args.batch_size, N_FRAGMENTS)
                    ids = torch.tensor(
                        [x["token_ids"] for x in fragments[start:stop]],
                        dtype=torch.long, device=args.device,
                    )
                    captured.clear()
                    with torch.inference_mode():
                        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    for layer in LAYERS:
                        hidden = captured[layer].to(torch.float32) - centers[layer]
                        scores = (hidden @ stacked[layer].T).clamp_min(0).cpu().numpy()
                        for position, iteration in enumerate(EVALUATED_CHECKPOINTS):
                            left = position * n_features
                            right = (position + 1) * n_features
                            stores[(iteration, layer)][start:stop] = scores[:, :, left:right]
                    if stop - checkpoint_at >= PROGRESS_INTERVAL or stop == N_FRAGMENTS:
                        for store in stores.values():
                            store.flush()
                        atomic_write_json(
                            progress_path,
                            {"completed_fragments": stop, "resolved": resolved},
                        )
                        checkpoint_at = stop
                        display.advance(refresh=True)
                        display.phase("Encoding fragments", fragments=f"{stop}/{N_FRAGMENTS}")
                        log(f"Checkpointed encoded fragments {stop}/{N_FRAGMENTS}.")
            finally:
                for handle in handles:
                    handle.remove()
            for iteration in EVALUATED_CHECKPOINTS:
                for layer in LAYERS:
                    final = cache_path(archive, iteration, layer)
                    os.replace(final.with_suffix(".npy.partial"), final)
            display.phase("Writing evaluator metadata")
            write_metadata(output, archive, fragments_path, cohort_path, cohort, n_features)
            atomic_write_json(archive / "summary.json", {**resolved, "status": "complete"})
            display.advance(refresh=True)
            run.set_status("complete", complete=True)
            log(f"Prepared {len(EVALUATED_CHECKPOINTS)} matched checkpoints.")
    except BaseException:
        run.set_status("interrupted")
        raise


if __name__ == "__main__":
    main()
