#!/usr/bin/env python3
"""Prepare checkpoint-specific intrinsic panels with 20 top validation fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from prepare import EVALUATED_CHECKPOINTS, N_FRAGMENTS
from trajectory import parse_layers

from icalens.experiments._run import atomic_write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "runs/fixed-panel/checkpoint-prepared"
DEFAULT_OUTPUT = ROOT / "runs/fixed-panel/prepared"
EXPLANATION_COUNT = 5
TOP_VALIDATION_COUNT = 20
RANDOM_VALIDATION_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, destination.parent)
    if destination.is_symlink() and os.readlink(destination) == target:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(target)


def intrinsic_indices(
    maxima: np.ndarray,
    document_ids: np.ndarray,
    *,
    seed: np.random.SeedSequence,
) -> tuple[list[int], list[int], list[int]]:
    needed = EXPLANATION_COUNT + TOP_VALIDATION_COUNT
    ranked = np.argsort(-maxima, kind="stable")
    top: list[int] = []
    used_documents: set[int] = set()
    for index in ranked:
        document = int(document_ids[index])
        if document in used_documents:
            continue
        top.append(int(index))
        used_documents.add(document)
        if len(top) == needed:
            break
    if len(top) != needed:
        raise ValueError(f"fewer than {needed} unique-document top fragments")

    rng = np.random.default_rng(seed)
    split = rng.permutation(needed)
    train = [top[int(position)] for position in split[:EXPLANATION_COUNT]]
    validation = [top[int(position)] for position in split[EXPLANATION_COUNT:]]
    random_records: list[int] = []
    for index in rng.permutation(len(maxima)):
        document = int(document_ids[index])
        if maxima[index] <= 0 or document in used_documents:
            continue
        random_records.append(int(index))
        used_documents.add(document)
        if len(random_records) == RANDOM_VALIDATION_COUNT:
            break
    if len(random_records) != RANDOM_VALIDATION_COUNT:
        raise ValueError("fewer than five eligible random-validation fragments")
    return train, validation, random_records


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fragments_path = (source / "iter-000/fragments.jsonl").resolve()
    fragments = [
        json.loads(line) for line in fragments_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(fragments) != N_FRAGMENTS:
        raise ValueError(f"expected {N_FRAGMENTS} fragments: {fragments_path}")
    document_ids = np.asarray([int(row["document_index"]) for row in fragments])
    resolved = {
        "format": "icalens.autointerpretability-intrinsic-validation",
        "schema_version": 1,
        "source_preparation": str(source),
        "source_preparation_run_sha256": sha256(source / "preparation-run.json"),
        "fragments": str(fragments_path),
        "fragments_sha256": sha256(fragments_path),
        "layers": list(layers),
        "checkpoints": list(EVALUATED_CHECKPOINTS),
        "selection": "checkpoint-specific top 25 unique-document fragments",
        "split_sizes": {
            "train_top": EXPLANATION_COUNT,
            "valid_top": TOP_VALIDATION_COUNT,
            "valid_random": RANDOM_VALIDATION_COUNT,
        },
        "seed_rule": "numpy SeedSequence([layer, persistent_component_id, iteration])",
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    manifest = output / "intrinsic-panel-run.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("resolved") != resolved:
            raise ValueError(f"incompatible intrinsic preparation: {manifest}")
        if previous.get("status") == "complete":
            print(f"PASS intrinsic panel already complete: {output}")
            return

    output.mkdir(parents=True, exist_ok=True)
    for iteration in EVALUATED_CHECKPOINTS:
        source_iteration = source / f"iter-{iteration:03d}"
        destination_iteration = output / f"iter-{iteration:03d}"
        relative_symlink(fragments_path, destination_iteration / "fragments.jsonl")
        for layer in layers:
            source_layer = source_iteration / f"layer_{layer:02d}" / "ica"
            destination_layer = destination_iteration / f"layer_{layer:02d}" / "ica"
            original = json.loads((source_layer / "selection.json").read_text(encoding="utf-8"))
            n_features = len(original["accepted"])
            values_path = (source_layer / "candidate_activations.npy").resolve()
            relative_symlink(values_path, destination_layer / "candidate_activations.npy")
            values = np.load(values_path, mmap_mode="r")
            maxima = np.asarray(values).max(axis=1)
            accepted: list[dict[str, Any]] = []
            for row in original["accepted"]:
                feature = int(row["feature"])
                position = int(row["candidate_position"])
                train, top, random = intrinsic_indices(
                    maxima[:, position],
                    document_ids,
                    seed=np.random.SeedSequence([layer, feature, iteration]),
                )
                accepted.append(
                    {
                        "feature": feature,
                        "candidate_position": position,
                        "train_top": train,
                        "valid_top": top,
                        "valid_random": random,
                    }
                )
            atomic_write_json(
                destination_layer / "selection.json",
                {
                    "candidate_ids": original["candidate_ids"],
                    "accepted": accepted,
                    "rejected": [],
                },
            )
            atomic_write_json(
                destination_iteration / f"layer_{layer:02d}" / "prepared.json",
                {
                    "format": "icalens.autointerpretability-prepared-layer",
                    "schema_version": 1,
                    "layer": layer,
                    "methods": ["ica"],
                    "n_fragments": N_FRAGMENTS,
                    "n_features": n_features,
                    "selection_protocol": "checkpoint-specific-intrinsic-5-train-20-valid",
                },
            )
        atomic_write_json(
            destination_iteration / "run.json",
            {"status": "prepared", "resolved": {**resolved, "iteration": iteration}},
        )
        print(f"PASS intrinsic selections at iteration {iteration}")
    atomic_write_json(manifest, {"status": "complete", "resolved": resolved})
    print(f"PASS four-layer intrinsic preparation: {output}")


if __name__ == "__main__":
    main()
