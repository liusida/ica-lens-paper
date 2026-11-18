#!/usr/bin/env python3
"""Does an upstream neuron predict a downstream one? (default L0 N996 -> L1 N586)

Exploration script (not core code). Records the upstream neuron's post-GELU activation at
the same 1M profiling tokens as ``neuron_activations.py`` (rows align with the activation
dataset), then compares it with the saved downstream activations: correlation, how often
one is strong when the other is, and the downstream distribution per upstream band.
Writes ``pair_L<a>N<i>_L<b>N<j>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from icalens._activation_dataset import ActivationDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_activations import ACTIVATIONS, replay_sample  # noqa: E402

HERE = Path(__file__).resolve().parent


class _Stop(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--up-layer", type=int, default=0)
    parser.add_argument("--up-neuron", type=int, default=996)
    parser.add_argument("--down-neuron", type=int, default=586)
    parser.add_argument("--activations", type=Path, default=ACTIVATIONS)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    result = json.loads(args.result.read_text())
    down_layer = int(result["layer"])
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    folder = args.result.parent / "neuron-activations"
    down = np.load(folder / "activations.npy")[:, neurons.index(args.down_neuron)].astype(
        np.float64
    )

    dataset = ActivationDataset(args.activations)
    model_id, revision = dataset.model["repo_id"], dataset.model["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    documents, selected = replay_sample(dataset, tokenizer)
    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    holder: dict[str, torch.Tensor] = {}

    def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        holder["v"] = inputs[0][0, :, args.up_neuron].index_select(0, holder["p"])
        raise _Stop

    handle = model.transformer.h[args.up_layer].mlp.c_proj.register_forward_pre_hook(hook)
    values = []
    try:
        for document_index, positions in tqdm(selected.items(), desc="Upstream", unit="doc"):
            holder["p"] = positions.to("cuda")
            try:
                model(input_ids=documents[document_index].input_ids[None].to("cuda"))
            except _Stop:
                pass
            values.append(holder["v"].float().cpu().numpy())
    finally:
        handle.remove()
    up = np.concatenate(values).astype(np.float64)
    if len(up) != len(down):
        raise RuntimeError("upstream activations do not align with the saved downstream ones")

    def q(values: np.ndarray, p: float) -> float:
        return float(np.quantile(values, p / 100))

    up_t = {p: q(up, p) for p in (90, 99, 99.9)}
    down_t = {p: q(down, p) for p in (90, 99, 99.9)}
    stats = {}
    for p in (90, 99, 99.9):
        up_strong, down_strong = up >= up_t[p], down >= down_t[p]
        both = (up_strong & down_strong).sum()
        stats[str(p)] = {
            "up_threshold": up_t[p],
            "down_threshold": down_t[p],
            "P(down strong | up strong)": float(both / up_strong.sum()),
            "P(up strong | down strong)": float(both / down_strong.sum()),
            "base rate": float(down_strong.mean()),
        }
    edges = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 100.0]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (up >= lo) & (up < hi)
        if mask.any():
            bands.append(
                {
                    "up_band": [lo, hi],
                    "tokens": int(mask.sum()),
                    "down_mean": float(down[mask].mean()),
                    "down_median": float(np.median(down[mask])),
                    "P(down >= 1)": float((down[mask] >= 1).mean()),
                    "P(down >= 2)": float((down[mask] >= 2).mean()),
                }
            )
    positive = (up > 0) & (down > 0)
    data = {
        "up": {"layer": args.up_layer, "neuron": args.up_neuron},
        "down": {"layer": down_layer, "neuron": args.down_neuron},
        "tokens": int(len(up)),
        "pearson": float(np.corrcoef(up, down)[0, 1]),
        "pearson_both_positive": float(np.corrcoef(up[positive], down[positive])[0, 1]),
        "up_fraction_positive": float((up > 0).mean()),
        "thresholds": stats,
        "down_by_up_band": bands,
    }
    stem = f"pair_L{args.up_layer}N{args.up_neuron}_L{down_layer}N{args.down_neuron}"
    np.save(folder / f"{stem}_up_activations.npy", up.astype(np.float32))
    (folder / f"{stem}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
