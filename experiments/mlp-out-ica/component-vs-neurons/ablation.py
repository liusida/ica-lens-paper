#!/usr/bin/env python3
"""Threshold-ablation sweep for ONE GPT-2 MLP neuron (default L1 N586).

Exploration script (not core code). Question: below which activation level can this
neuron's activations be removed without changing the model's outputs much?

  Calibration: the neuron's post-GELU activations on the 1M Pile-10k profiling tokens
               (``activations.npy`` from ``neuron_activations.py``). For each percentile q,
               t(q) is the q-th percentile, m_below(q) = E[a | a < t(q)] and
               m_above(q) = E[a | a >= t(q)].
  Evaluation:  held-out OpenWebText (first ``--eval-tokens`` usable tokens, same framing
               and context length as the lens), run in float32.
  Variants (only this neuron, at every token of its layer, edited before c_proj):
    threshold q  a < t(q)  -> m_below(q); keep a >= t(q)           ("strong regime kept")
    inverse q    a >= t(q) -> m_above(q); keep a < t(q)            ("only weak kept")
    random q     same number of tokens as `threshold q` in each document, chosen at
                 random, -> m_below(q)
    mean         every token -> overall calibration mean
    zero         every token -> 0
  Metrics per variant, over every content position with a next token:
    mean KL(P || P'), mean change in next-token cross-entropy, top-1 flip rate, and KL
    split by the neuron's clean activation band at that position.
  For full mean ablation it also records the average logit change at positions where the
  neuron is strongly active (the tokens it pushes up or down).

Writes ``ablation_N<neuron>.json`` and its twin ``.html`` (render with render_report.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from icalens.cli.fit_text import load_pile_documents

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_report import render_any  # noqa: E402

HERE = Path(__file__).resolve().parent
EVAL_DATASET = "Skylion007/openwebtext"
EVAL_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
QUANTILES = (50, 90, 95, 97, 98, 99, 99.5, 99.9)
BANDS = (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 100.0)  # clean-activation bands for the KL split


def calibrate(activation: np.ndarray) -> dict[str, Any]:
    levels = {}
    for q in QUANTILES:
        t = float(np.quantile(activation, q / 100))
        below = activation < t
        levels[str(q)] = {
            "q": q,
            "threshold": t,
            "mean_below": float(activation[below].mean()) if below.any() else t,
            "mean_above": float(activation[~below].mean()),
            "fraction_below": float(below.mean()),
        }
    return {"mean": float(activation.mean()), "levels": levels}


def build_variants(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {"name": "mean", "kind": "constant", "value": calibration["mean"]},
        {"name": "zero", "kind": "constant", "value": 0.0},
    ]
    for level in calibration["levels"].values():
        q = level["q"]
        variants += [
            {"name": f"threshold q{q:g}", "kind": "threshold", "q": q, **level},
            {"name": f"inverse q{q:g}", "kind": "inverse", "q": q, **level},
            {"name": f"random q{q:g}", "kind": "random", "q": q, **level},
        ]
    return variants


def edit(values: torch.Tensor, variant: dict[str, Any], generator: torch.Generator) -> torch.Tensor:
    """Return the edited activations of one neuron for one sequence ([T])."""
    kind = variant["kind"]
    if kind == "constant":
        return torch.full_like(values, variant["value"])
    below = values < variant["threshold"]
    if kind == "threshold":
        return torch.where(below, torch.full_like(values, variant["mean_below"]), values)
    if kind == "inverse":
        return torch.where(below, values, torch.full_like(values, variant["mean_above"]))
    count = int(below.sum())
    chosen = torch.zeros_like(below)
    if count:
        order = torch.randperm(len(values), generator=generator)[:count].to(values.device)
        chosen[order] = True
    return torch.where(chosen, torch.full_like(values, variant["mean_below"]), values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--neuron", type=int, default=586)
    parser.add_argument("--eval-tokens", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-logit-tokens", type=int, default=20)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    layer = int(result["layer"])
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    folder = args.result.parent / "neuron-activations"
    calibration_values = np.load(folder / "activations.npy")[:, neurons.index(args.neuron)]
    calibration = calibrate(calibration_values.astype(np.float64))
    variants = build_variants(calibration)

    manifest = json.loads(Path(result["lens"], "icalens.json").read_text())
    model_id, revision = manifest["model"]["repo_id"], manifest["model"]["revision"]
    provenance = manifest["layers"][str(layer)]["fitting"]["provenance"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    documents = load_pile_documents(
        tokenizer,
        dataset_id=EVAL_DATASET,
        dataset_revision=EVAL_REVISION,
        split="train",
        text_field="text",
        candidate_token_budget=args.eval_tokens,
        context_length=int(provenance["context_length"]),
        document_framing=provenance["document_framing"],
    )
    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision).to("cuda").eval()
    c_proj = model.transformer.h[layer].mlp.c_proj
    unembed = model.lm_head.weight  # [vocab, d]

    state: dict[str, Any] = {"variant": None, "clean": None}
    generator = torch.Generator().manual_seed(args.seed)

    def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        hidden = inputs[0]
        state["clean_activation"] = hidden[0, :, args.neuron].detach().clone()
        if state["variant"] is None:
            return inputs
        hidden = hidden.clone()
        hidden[0, :, args.neuron] = edit(hidden[0, :, args.neuron], state["variant"], generator)
        return (hidden, *inputs[1:])

    handle = c_proj.register_forward_pre_hook(hook)
    n_bands = len(BANDS) - 1
    totals = {
        v["name"]: {
            "kl": 0.0,
            "dloss": 0.0,
            "flips": 0,
            "band_kl": [0.0] * n_bands,
            "replaced": 0,
        }
        for v in variants
    }
    band_counts = [0] * n_bands
    positions = 0
    strong_logit_delta = torch.zeros(unembed.shape[0], dtype=torch.float64, device="cuda")
    strong_count = 0
    try:
        with torch.inference_mode():
            for document in tqdm(documents, desc="Ablation sweep", unit="doc", dynamic_ncols=True):
                ids = document.input_ids.unsqueeze(0).to("cuda")
                content = document.candidate_positions.to("cuda")
                content = content[content < ids.shape[1] - 1]  # positions with a next token
                if len(content) == 0:
                    continue
                targets = ids[0, content + 1]
                state["variant"] = None
                clean_logits = model(input_ids=ids, use_cache=False).logits[0, content].float()
                clean_logp = clean_logits.log_softmax(-1)
                clean_top = clean_logp.argmax(-1)
                clean_loss = -clean_logp.gather(1, targets[:, None])[:, 0]
                activation = state["clean_activation"][content]
                band = torch.bucketize(activation, torch.tensor(BANDS[1:-1], device="cuda"))
                for index in range(n_bands):
                    band_counts[index] += int((band == index).sum())
                positions += len(content)
                for variant in variants:
                    state["variant"] = variant
                    logp = model(input_ids=ids, use_cache=False).logits[0, content]
                    logp = logp.float().log_softmax(-1)
                    kl = (clean_logp.exp() * (clean_logp - logp)).sum(-1)
                    total = totals[variant["name"]]
                    total["kl"] += float(kl.sum())
                    total["dloss"] += float(
                        (-logp.gather(1, targets[:, None])[:, 0] - clean_loss).sum()
                    )
                    total["flips"] += int((logp.argmax(-1) != clean_top).sum())
                    for index in range(n_bands):
                        total["band_kl"][index] += float(kl[band == index].sum())
                    if variant["kind"] in ("threshold", "random"):
                        total["replaced"] += int((activation < variant["threshold"]).sum())
                    elif variant["kind"] == "inverse":
                        total["replaced"] += int((activation >= variant["threshold"]).sum())
                    else:
                        total["replaced"] += len(content)
                    if variant["name"] == "mean":
                        strong = activation >= 2.0
                        if strong.any():
                            delta = (logp[strong] - clean_logp[strong]).double()
                            strong_logit_delta += delta.sum(0)
                            strong_count += int(strong.sum())
    finally:
        handle.remove()

    rows = []
    for variant in variants:
        total = totals[variant["name"]]
        rows.append(
            {
                "name": variant["name"],
                "kind": variant["kind"],
                "q": variant.get("q"),
                "threshold": variant.get("threshold"),
                "replacement": variant.get("value", variant.get("mean_below")),
                "fraction_replaced": total["replaced"] / positions,
                "mean_kl": total["kl"] / positions,
                "mean_delta_loss": total["dloss"] / positions,
                "top1_flip_rate": total["flips"] / positions,
                "kl_by_clean_band": [
                    total["band_kl"][i] / band_counts[i] if band_counts[i] else None
                    for i in range(n_bands)
                ],
            }
        )
    average = (strong_logit_delta / max(strong_count, 1)).float()

    def vocab(values: torch.Tensor, largest: bool) -> list[dict[str, Any]]:
        top = torch.topk(values, args.top_logit_tokens, largest=largest)
        return [
            {"token": tokenizer.decode([int(i)]), "delta_logprob": float(v)}
            for v, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)
        ]

    data = {
        "kind": "neuron_ablation",
        "layer": layer,
        "neuron": args.neuron,
        "calibration": {
            "tokens": int(len(calibration_values)),
            "dataset": "NeelNanda/pile-10k profiling sample (mlp_out lens fitting tokens)",
            **calibration,
        },
        "evaluation": {
            "dataset": EVAL_DATASET,
            "revision": EVAL_REVISION,
            "positions": positions,
            "documents": len(documents),
            "precision": "float32",
        },
        "bands": [[BANDS[i], BANDS[i + 1]] for i in range(n_bands)],
        "band_counts": band_counts,
        "variants": rows,
        "mean_ablation_strong_positions": {
            "condition": "clean activation >= 2.0",
            "positions": strong_count,
            "most_decreased": vocab(average, largest=False),
            "most_increased": vocab(average, largest=True),
        },
    }
    data_path = folder / f"ablation_N{args.neuron}.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    data_path.with_suffix(".html").write_text(render_any(data))
    print(f"Saved {data_path} and its .html")
    for row in rows:
        print(
            f"{row['name']:>18}: KL {row['mean_kl']:.2e}  dloss {row['mean_delta_loss']:+.2e}  "
            f"flip {row['top1_flip_rate']:.2e}  replaced {row['fraction_replaced']:.3f}"
        )


if __name__ == "__main__":
    main()
