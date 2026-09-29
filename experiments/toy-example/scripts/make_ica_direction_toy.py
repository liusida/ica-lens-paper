#!/usr/bin/env python3
"""Build the toy set from the selected ICA direction's strongest tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from icalens import ICALens


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "work/source")
    parser.add_argument("--output", type=Path, default=root / "work/selected")
    parser.add_argument("--component", type=int, default=65)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    paths = [output / name for name in ("activations.safetensors", "samples.json", "capture.json")]
    if any(path.exists() for path in paths) and not args.force:
        raise FileExistsError(f"output exists under {output}; pass --force to replace it")

    source_samples = json.loads((source / "output/samples.json").read_text(encoding="utf-8"))
    x = load_file(source / "output/activations.safetensors")["layer_00"].float()
    lens = ICALens.from_pretrained(source / "ica-fit/lens")
    scores = torch.as_tensor(lens.transform(x, layer=0)).float()[:, args.component]
    group_indices = torch.topk(scores.abs(), args.group_size).indices.tolist()
    group_set = set(group_indices)
    background_indices = [index for index in range(len(source_samples)) if index not in group_set]
    random.Random(args.seed).shuffle(background_indices)
    background_count = len(background_indices)
    selected = background_indices + group_indices
    samples = []
    for index, source_index in enumerate(selected):
        sample = dict(source_samples[source_index])
        sample.update({"index": index, "source_index": source_index,
                       "role": "background" if index < background_count else "concept",
                       "label": None if index < background_count
                       else f"C{index - background_count + 1}"})
        samples.append(sample)

    output.mkdir(parents=True, exist_ok=True)
    save_file({"layer_00": x[selected].contiguous()}, paths[0])
    paths[1].write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    paths[2].write_text(json.dumps({
        "format": "icalens-toy-example-ica-direction-v1", "source_component": args.component,
        "selection": "largest absolute ICA scores plus every remaining vocabulary token",
        "seed": args.seed,
        "counts": {"background": background_count, "concept": args.group_size,
                   "total": len(selected)},
        "samples_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
