#!/usr/bin/env python3
"""Randomly sample a neuron's examples from activation bands, as a twin .json/.html report.

Exploration script (not core code). Uses the 1M-token activations already saved by
``neuron_activations.py`` (rows align with the activation dataset's sample
metadata), draws ``--per-band`` random tokens from each activation band
``[lo, hi)``, and stores a per-token context window (neuron activation and
component score for every context token) for each. Writes
``band_examples_N<neuron>.json`` and renders its twin ``.html`` with
``render_report.py``; re-render the HTML alone with
``render_report.py band_examples_N<neuron>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from safetensors.torch import load_file as load_torch_file
from transformers import AutoTokenizer, GPT2LMHeadModel

from icalens._activation_dataset import ActivationDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_activations import ACTIVATIONS, LENS, replay_sample, token_window  # noqa: E402
from render_report import render_any  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--neuron", type=int, default=586)
    parser.add_argument(
        "--edges", default="0,1,2,3,4,5", help="Band edges; bands are [e_i, e_i+1)."
    )
    parser.add_argument("--per-band", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lens", type=Path, default=LENS)
    parser.add_argument("--activations", type=Path, default=ACTIVATIONS)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    layer, component = int(result["layer"]), int(result["component"])
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    if args.neuron not in neurons:
        raise ValueError(f"neuron {args.neuron} has no saved activations; have {neurons}")
    folder = args.result.parent / "neuron-activations"
    activation = np.load(folder / "activations.npy")[:, neurons.index(args.neuron)]
    score = np.load(folder / f"component_C{component}_score.npy")

    dataset = ActivationDataset(args.activations)
    samples = load_torch_file(dataset.path / dataset.manifest["samples_file"])
    document_index = samples["document_index"].numpy()
    position = samples["position"].numpy()
    if len(activation) != len(document_index):
        raise RuntimeError("saved activations do not align with the activation dataset")

    edges = [float(value) for value in args.edges.split(",")]
    rng = np.random.default_rng(args.seed)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        rows = np.flatnonzero((activation >= lo) & (activation < hi))
        chosen = rng.choice(rows, size=min(args.per_band, len(rows)), replace=False)
        chosen = chosen[np.argsort(-activation[chosen])]
        bands.append({"lo": lo, "hi": hi, "count_in_band": int(len(rows)), "rows": chosen})

    model_id, revision = dataset.model["repo_id"], dataset.model["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    documents, _ = replay_sample(dataset, tokenizer)
    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    tensors = load_file(args.lens / f"artifacts/mlp_out/layer_{layer:02d}.safetensors")
    sign = 1.0 if result["tail_direction"] == "positive" else -1.0
    reading = torch.tensor(tensors["reading_matrix"][component], dtype=torch.float32) * sign
    center = torch.tensor(tensors["center"], dtype=torch.float32)

    for band in bands:
        examples = []
        for row in band.pop("rows"):
            ids = documents[int(document_index[row])].input_ids
            if int(ids[int(position[row])]) != int(samples["token_id"][row]):
                raise RuntimeError(f"replayed token mismatch at row {row}")
            window = token_window(
                model,
                tokenizer,
                ids,
                int(position[row]),
                layer=layer,
                neuron=args.neuron,
                reading=reading,
                center=center,
            )
            examples.append(
                {
                    "row": int(row),
                    "activation": float(activation[row]),
                    "component_score": float(score[row]),
                    "document_index": int(document_index[row]),
                    "position": int(position[row]),
                    "token": next(token["text"] for token in window if token["target"]),
                    "window": window,
                }
            )
        band["examples"] = examples

    data = {
        "kind": "neuron_bands",
        "layer": layer,
        "component": component,
        "neuron": args.neuron,
        "tokens": int(len(activation)),
        "activation": "post-GELU (input to c_proj)",
        "sampling": {"per_band": args.per_band, "seed": args.seed, "edges": edges},
        "color_scale": float(activation.max()),
        "bands": bands,
    }
    data_path = folder / f"band_examples_N{args.neuron}.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    html_path = data_path.with_suffix(".html")
    html_path.write_text(render_any(data))
    print(f"Saved {data_path}\nSaved {html_path}")
    for band in bands:
        print(
            f"[{band['lo']:g}, {band['hi']:g}): {band['count_in_band']:,} tokens in band; "
            f"sampled {[example['token'] for example in band['examples']]}"
        )


if __name__ == "__main__":
    main()
