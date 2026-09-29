#!/usr/bin/env python3
"""Estimate SAE dead-feature prevalence in the original ERF candidate sample."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from icalens._activation_dataset import ActivationDataset
from icalens.experiments._run import atomic_write_json
from icalens.experiments._sae import SAEFeatureEncoder
from icalens.experiments.erf_gradient import _stable_seed
from icalens.experiments.saebench_sparse_probing import (
    _prepare_layer_baselines,
    _resolve_baselines,
)

ROOT = Path(__file__).resolve().parent
ERF_RUN = ROOT / "runs/sae-suffix-sweep-v2"
MODELS = {
    "gpt2": ("openai-community/gpt2", "gpt2-pile10k-1m", 12),
    "gemma2": ("google/gemma-2-2b", "gemma-2-2b-pile10k-1m", 26),
    "qwen9b": ("Qwen/Qwen3.5-9B-Base", "qwen3.5-9b-base-pile10k-1m", 32),
}


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--layers",
        nargs="*",
        metavar="MODEL:LAYER,LAYER",
        help="Per-model layer overrides; default is first,middle,final.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/home/liusida/Expansion/research/ICA-data/icalens-activations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/sae-dead-candidate-scan-pilot",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_layers(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    overrides: dict[str, tuple[int, ...]] = {}
    for specification in args.layers or ():
        label, separator, values = specification.partition(":")
        if not separator or label not in MODELS:
            raise ValueError(f"invalid --layers specification: {specification!r}")
        layers = tuple(sorted({int(value) for value in values.split(",") if value}))
        if not layers:
            raise ValueError(f"empty --layers specification: {specification!r}")
        overrides[label] = layers
    resolved = {}
    for label in args.models:
        n_layers = MODELS[label][2]
        layers = overrides.get(label, (0, n_layers // 2, n_layers - 1))
        if min(layers) < 0 or max(layers) >= n_layers:
            raise ValueError(f"{label}: layer outside 0..{n_layers - 1}: {layers}")
        resolved[label] = layers
    unknown = set(overrides).difference(args.models)
    if unknown:
        raise ValueError(f"--layers supplied for unrequested models: {sorted(unknown)}")
    return resolved


def scan_layer(
    *, label: str, model: str, cache: ActivationDataset, layer: int, chunk_size: int, seed: int
) -> dict[str, object]:
    prepared_path = ERF_RUN / label / "prepared" / f"layer_{layer:02d}.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    candidate_count = int(prepared["candidate_features_profiled"])
    selected = {int(feature) for feature in prepared["features"]}
    if candidate_count != 200 or len(selected) != 100:
        raise ValueError(
            f"{label} L{layer}: expected 200 candidates and 100 selected features"
        )

    registry = _resolve_baselines(model, "sae")
    baseline = _prepare_layer_baselines(registry, layer=layer)
    width = int(baseline["sae"]["width"])
    permutation = np.random.default_rng(_stable_seed(seed, label, layer)).permutation(width)
    candidates = [int(value) for value in permutation[:candidate_count]]
    if not selected.issubset(candidates):
        raise ValueError(f"{label} L{layer}: saved features do not match candidate permutation")
    unselected = [feature for feature in candidates if feature not in selected]
    if len(unselected) != 100:
        raise ValueError(f"{label} L{layer}: expected 100 unselected candidates")

    encoder = SAEFeatureEncoder(
        {
            "hidden_size": cache.hidden_size,
            "layer": layer,
            "saebench_model_name": model,
            "baselines": baseline,
        },
        device="cuda",
        dtype=torch.float32,
    ).eval()
    ids = torch.tensor(unselected, device="cuda")
    maxima = torch.zeros(len(unselected), dtype=torch.float32, device="cuda")
    hidden = cache.layer(layer)
    with torch.inference_mode():
        for start in range(0, cache.sample_count, chunk_size):
            stop = min(start + chunk_size, cache.sample_count)
            codes = encoder.encode(hidden[start:stop].to("cuda", dtype=torch.float32))
            maxima = torch.maximum(maxima, codes.index_select(1, ids).amax(dim=0))
            if stop == cache.sample_count or stop % 262_144 == 0:
                print(
                    f"[{timestamp()}] {label} L{layer}: {stop}/{cache.sample_count}",
                    flush=True,
                )
    maxima_cpu = maxima.cpu().numpy()
    unselected_dead = int(np.count_nonzero(maxima_cpu <= 0))
    return {
        "model": label,
        "layer": layer,
        "dictionary_width": width,
        "candidate_count": candidate_count,
        "known_active_selected": len(selected),
        "unselected_count": len(unselected),
        "unselected_dead": unselected_dead,
        "unselected_active": len(unselected) - unselected_dead,
        "estimated_dead_fraction_in_200": unselected_dead / candidate_count,
        "dead_definition": "maximum encoder activation <= 0 on the 1M-token cache",
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    layers_by_model = resolve_layers(args)
    for label in args.models:
        model, cache_name, _ = MODELS[label]
        cache = ActivationDataset(args.cache_root / cache_name)
        for layer in layers_by_model[label]:
            path = output / f"{label}-layer-{layer:02d}.json"
            if path.is_file():
                print(f"[{timestamp()}] REUSE {path}", flush=True)
                records.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            print(f"[{timestamp()}] START {label} L{layer}", flush=True)
            record = scan_layer(
                label=label,
                model=model,
                cache=cache,
                layer=layer,
                chunk_size=args.chunk_size,
                seed=args.seed,
            )
            atomic_write_json(path, record)
            records.append(record)
            print(
                f"[{timestamp()}] PASS {label} L{layer}: "
                f"dead={record['unselected_dead']}/200",
                flush=True,
            )
            torch.cuda.empty_cache()
    atomic_write_json(
        output / "summary.json",
        {
            "status": "complete",
            "seed": args.seed,
            "layers": records,
            "note": "Each denominator is the original 200 uniformly sampled candidates; "
            "the saved 100 selected candidates are known active.",
        },
    )
    print(f"[{timestamp()}] COMPLETE {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
