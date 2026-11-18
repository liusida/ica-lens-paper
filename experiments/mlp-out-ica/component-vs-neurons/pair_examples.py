#!/usr/bin/env python3
"""Examples where two neurons both fire, only one fires, or neither (default L0 N996, L1 N586).

Exploration script (not core code). "Fires" = activation >= that neuron's q-th percentile
over the 1M profiling tokens (default q99). Uses the saved activations
(``activations.npy`` and ``pair_...up_activations.npy`` from ``neuron_pair.py``), samples
``--per-group`` random tokens per group, and stores a context window with BOTH neurons'
per-token activations. Writes ``pair_examples_L<a>N<i>_L<b>N<j>.json`` and its twin ``.html``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_torch_file
from transformers import AutoTokenizer, GPT2LMHeadModel

from icalens._activation_dataset import ActivationDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_activations import ACTIVATIONS, replay_sample  # noqa: E402
from render_report import render_any  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--up-layer", type=int, default=0)
    parser.add_argument("--up-neuron", type=int, default=996)
    parser.add_argument("--down-neuron", type=int, default=586)
    parser.add_argument("--quantile", type=float, default=99.0)
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activations", type=Path, default=ACTIVATIONS)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    result = json.loads(args.result.read_text())
    down_layer = int(result["layer"])
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    folder = args.result.parent / "neuron-activations"
    stem = f"L{args.up_layer}N{args.up_neuron}_L{down_layer}N{args.down_neuron}"
    down = np.load(folder / "activations.npy")[:, neurons.index(args.down_neuron)]
    up = np.load(folder / f"pair_{stem}_up_activations.npy")
    up_t = float(np.quantile(up, args.quantile / 100))
    down_t = float(np.quantile(down, args.quantile / 100))
    up_on, down_on = up >= up_t, down >= down_t
    groups = {
        "both": up_on & down_on,
        f"only L{args.up_layer} N{args.up_neuron}": up_on & ~down_on,
        f"only L{down_layer} N{args.down_neuron}": ~up_on & down_on,
        "neither": ~up_on & ~down_on,
    }

    dataset = ActivationDataset(args.activations)
    samples = load_torch_file(dataset.path / dataset.manifest["samples_file"])
    model_id, revision = dataset.model["repo_id"], dataset.model["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    documents, _ = replay_sample(dataset, tokenizer)
    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        model.transformer.h[args.up_layer].mlp.c_proj.register_forward_pre_hook(
            lambda _m, i: captured.__setitem__("up", i[0][0, :, args.up_neuron].float().cpu())
        ),
        model.transformer.h[down_layer].mlp.c_proj.register_forward_pre_hook(
            lambda _m, i: captured.__setitem__("down", i[0][0, :, args.down_neuron].float().cpu())
        ),
    ]

    rng = np.random.default_rng(args.seed)
    out_groups = []
    try:
        for name, mask in groups.items():
            rows = np.flatnonzero(mask)
            chosen = rng.choice(rows, size=min(args.per_group, len(rows)), replace=False)
            examples = []
            for row in chosen:
                doc = int(samples["document_index"][row])
                position = int(samples["position"][row])
                ids = documents[doc].input_ids
                stop = min(len(ids), position + 9)
                model(input_ids=ids[:stop][None].to("cuda"), use_cache=False)
                start = max(1, position - 24)
                window = [
                    {
                        "text": tokenizer.decode([int(ids[p])]),
                        "up": float(captured["up"][p]),
                        "down": float(captured["down"][p]),
                        "target": p == position,
                    }
                    for p in range(start, stop)
                ]
                examples.append(
                    {
                        "row": int(row),
                        "token": tokenizer.decode([int(ids[position])]),
                        "up": float(up[row]),
                        "down": float(down[row]),
                        "window": window,
                    }
                )
            examples.sort(key=lambda e: -(e["up"] + e["down"]))
            out_groups.append(
                {"name": name, "tokens_in_group": int(mask.sum()), "examples": examples}
            )
    finally:
        for hook in hooks:
            hook.remove()

    data = {
        "kind": "neuron_pair_examples",
        "up": {"layer": args.up_layer, "neuron": args.up_neuron, "threshold": up_t},
        "down": {"layer": down_layer, "neuron": args.down_neuron, "threshold": down_t},
        "quantile": args.quantile,
        "tokens": int(len(up)),
        "color_scale": {"up": float(up.max()), "down": float(down.max())},
        "groups": out_groups,
    }
    data_path = folder / f"pair_examples_{stem}_q{args.quantile:g}.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    data_path.with_suffix(".html").write_text(render_any(data))
    print(f"Saved {data_path} and its .html")
    for group in out_groups:
        print(
            f"{group['name']:>16} ({group['tokens_in_group']:,}): "
            f"{[(e['token'], round(e['up'], 1), round(e['down'], 1)) for e in group['examples']]}"
        )


if __name__ == "__main__":
    main()
