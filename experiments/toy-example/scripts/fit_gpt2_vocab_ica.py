#!/usr/bin/env python3
"""Fit full-dimensional ICA on the GPT-2 vocabulary and report extreme tokens."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import kurtosis

from icalens import ICALens


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=root / "work/source/output")
    parser.add_argument("--output", type=Path, default=root / "work/source/ica-fit")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    capture = args.capture.expanduser().resolve()
    output = args.output.expanduser().resolve()
    result_path = output / "top-components.json"
    lens_path = output / "lens"
    if (result_path.exists() or lens_path.exists()) and not args.force:
        raise FileExistsError(f"output exists under {output}; pass --force to replace it")
    samples = json.loads((capture / "samples.json").read_text(encoding="utf-8"))
    provenance = json.loads((capture / "capture.json").read_text(encoding="utf-8"))
    activations = load_file(capture / "activations.safetensors")
    lens = ICALens(
        model_id=provenance["model"]["repo_id"],
        model_revision=provenance["model"]["revision"],
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        icalens_preprocessing="none",
    )
    results = {}
    for layer in provenance["layers"]:
        x = activations[f"layer_{layer:02d}"].float()
        lens.fit(
            x, layer=layer, n_components=x.shape[1], max_iter=args.max_iter,
            random_state=args.seed, progress=True, device="cuda", batch_size=8192,
        )
        scores = torch.as_tensor(lens.transform(x, layer=layer)).float()
        component_kurtosis = np.asarray(kurtosis(scores.numpy(), axis=0))
        order = np.argsort(-component_kurtosis)
        results[str(layer)] = {
            "samples": len(x), "dimensions": x.shape[1], "components": scores.shape[1],
            "top_components": [
                {
                    "component": int(component),
                    "excess_kurtosis": float(component_kurtosis[component]),
                    "top_absolute_samples": [
                        {**samples[index], "score": float(scores[index, component])}
                        for index in torch.topk(scores[:, component].abs(), 20).indices.tolist()
                    ],
                }
                for component in order[:10]
            ],
        }
    output.mkdir(parents=True, exist_ok=True)
    lens.save(lens_path)
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(result_path)
    print(lens_path)


if __name__ == "__main__":
    main()
