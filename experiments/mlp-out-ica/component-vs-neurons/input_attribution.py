#!/usr/bin/env python3
"""Attribute a layer-1 MLP neuron's pre-activation to layer-0 MLP neurons (and everything else).

Exploration script (not core code). GPT-2, target neuron N (default L1 N586), at the tokens
where its calibration activation is >= its q-th percentile (default q99, a >~ 1).

Math (per target token t; all LayerNorm scales sigma and the layer-1 attention pattern A
are frozen at their actual values, which makes the decomposition EXACT):

  r0_s     = emb_s + attn0_s + mlp0_s,       mlp0_s = sum_j a_j(s) W0_out[j] + b0_out
  r1mid_t  = r0_t + sum_h sum_s A_h[t,s] (LN1(r0_s) W_V,h + b_V,h) W_O,h + b_O
  z_t      = w_in . LN2(r1mid_t) + b_in  =  u . r1mid_t / sigma2_t + (w_in . beta2 + b_in)
  with u = P(gamma2 * w_in), P = centering, LN1(x) = gamma1 * P x / sigma1 + beta1.

  So every additive piece c of the residual contributes
     direct:  u . c_t / sigma2_t
     value :  sum_h sum_s A_h[t,s] / sigma1_s * (c_s . k_h) / sigma2_t,
              k_h = P(gamma1 * (W_V,h W_O,h u))
  and layer-0 neuron j contributes  D_j(t) = a_j(t) (u . W0_out[j]) / sigma2_t
                                     V_j(t) = sum_h (k_h . W0_out[j])
                                              * sum_s A_h[t,s] a_j(s) / sigma1_s / sigma2_t.
  The pieces sum to z_t to float precision (reported as max |error|).

Attribution is not the same as a counterfactual: removing a piece also changes sigma and A.
A validation step zero-ablates each sampled token's top-5 layer-0 neurons (at all positions)
and compares the model's actual change in z with the frozen-linear prediction.

Writes ``input_attribution_N<neuron>.json`` and its twin ``.html`` (render_report.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_torch_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from icalens._activation_dataset import ActivationDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_activations import ACTIVATIONS, replay_sample  # noqa: E402
from render_report import render_any  # noqa: E402

HERE = Path(__file__).resolve().parent
PIECES = ("emb", "attn0", "mlp0_neurons", "mlp0_bias")
EXAMPLE_BANDS = ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 100.0))


class Weights:
    """Frozen-linear weights for one target neuron in layer `layer` (1)."""

    def __init__(self, model: GPT2LMHeadModel, layer: int, neuron: int) -> None:
        if layer != 1:
            raise ValueError("this script decomposes layer-1 neurons into layer-0 pieces")
        blocks = model.transformer.h
        self.n_heads = model.config.n_head
        self.head_dim = model.config.n_embd // self.n_heads
        d = model.config.n_embd
        self.eps1 = blocks[1].ln_1.eps
        self.eps2 = blocks[1].ln_2.eps
        gamma1 = blocks[1].ln_1.weight.double()
        beta1 = blocks[1].ln_1.bias.double()
        gamma2 = blocks[1].ln_2.weight.double()
        beta2 = blocks[1].ln_2.bias.double()
        w_in = blocks[1].mlp.c_fc.weight[:, neuron].double()
        b_in = blocks[1].mlp.c_fc.bias[neuron].double()
        center = lambda v: v - v.mean(-1, keepdim=True)  # noqa: E731
        self.u = center(gamma2 * w_in)  # [d]
        self.constant_ln2 = float(w_in @ beta2 + b_in)
        c_attn_w = blocks[1].attn.c_attn.weight.double()  # [d, 3d]
        c_attn_b = blocks[1].attn.c_attn.bias.double()
        w_v, b_v = c_attn_w[:, 2 * d :], c_attn_b[2 * d :]
        w_o, b_o = blocks[1].attn.c_proj.weight.double(), blocks[1].attn.c_proj.bias.double()
        k = []
        head_constant = []
        for h in range(self.n_heads):
            cols = slice(h * self.head_dim, (h + 1) * self.head_dim)
            ou = w_o[cols] @ self.u  # [head_dim]
            q = w_v[:, cols] @ ou  # [d]
            k.append(center(gamma1 * q))
            head_constant.append(float(beta1 @ q + b_v[cols] @ ou))
        self.k = torch.stack(k)  # [H, d]
        self.head_constant = torch.tensor(head_constant, dtype=torch.float64)  # [H]
        self.constant_bo = float(b_o @ self.u)
        w0_out = blocks[0].mlp.c_proj.weight.double()  # [3072, d]
        self.b0_out = blocks[0].mlp.c_proj.bias.double()
        self.direct_neuron = w0_out @ self.u  # [3072]
        self.value_neuron = w0_out @ self.k.T  # [3072, H]
        self.w_in, self.b_in = w_in, b_in


def _first(output: Any) -> torch.Tensor:
    """Module outputs may be a tensor or a tuple whose first item is the tensor."""
    return output[0] if isinstance(output, tuple) else output


def sigma(x: torch.Tensor, eps: float) -> torch.Tensor:
    return (x.var(-1, unbiased=False) + eps).sqrt()


@torch.inference_mode()
def forward_capture(
    model: GPT2LMHeadModel,
    ids: torch.Tensor,
    neuron: int,
    edit0: tuple[list[int], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """One float32 forward capturing everything the decomposition needs."""
    store: dict[str, torch.Tensor] = {}
    blocks = model.transformer.h
    hooks = []

    def pre_c_proj0(_m: Any, inputs: tuple[torch.Tensor, ...]) -> Any:
        act = inputs[0]
        if edit0 is not None:
            act = act.clone()
            act[0, :, edit0[0]] = edit0[1]
        store["a0"] = act[0].double()
        return (act, *inputs[1:]) if edit0 is not None else None

    hooks.append(blocks[0].mlp.c_proj.register_forward_pre_hook(pre_c_proj0))
    hooks.append(
        blocks[0].attn.register_forward_hook(
            lambda _m, _i, out: store.__setitem__("attn0", _first(out)[0].double())
        )
    )
    hooks.append(
        blocks[0].register_forward_hook(
            lambda _m, _i, out: store.__setitem__("r0", _first(out)[0].double())
        )
    )
    hooks.append(
        blocks[1].attn.register_forward_hook(
            lambda _m, _i, out: store.__setitem__("attn1", _first(out)[0].double())
        )
    )
    hooks.append(
        blocks[1].mlp.c_fc.register_forward_hook(
            lambda _m, _i, out: store.__setitem__("z", out[0, :, neuron].double())
        )
    )
    try:
        out = model(
            input_ids=ids, use_cache=False, output_attentions=True, output_hidden_states=True
        )
    finally:
        for hook in hooks:
            hook.remove()
    store["emb"] = out.hidden_states[0][0].double()
    store["A"] = out.attentions[1][0].double()  # [H, T, T]
    return store


def decompose(
    store: dict[str, torch.Tensor], weights: Weights, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Exact decomposition of z at `targets` (token indices)."""
    r0 = store["r0"]
    sigma1 = sigma(r0, weights.eps1)  # [T]
    r1mid = r0 + store["attn1"]
    sigma2 = sigma(r1mid[targets], weights.eps2)  # [N]
    a0 = store["a0"]
    pieces = {
        "emb": store["emb"],
        "attn0": store["attn0"],
        "mlp0_neurons": a0 @ store["w0_out"],
        "mlp0_bias": weights.b0_out.expand_as(r0),
    }
    u, k = weights.u.to(r0.device), weights.k.to(r0.device)
    B = store["A"][:, targets, :] / sigma1[None, None, :]  # [H, N, T]
    result: dict[str, torch.Tensor] = {}
    for name, value in pieces.items():
        result[f"direct.{name}"] = (value[targets] @ u) / sigma2
        projected = value @ k.T  # [T, H]: c_s . k_h
        result[f"value.{name}"] = torch.einsum("hnt,th->n", B, projected) / sigma2
        result[f"value_source.{name}"] = torch.einsum("hnt,th->nt", B, projected) / sigma2[:, None]
    result["value.biases"] = (
        weights.head_constant.to(r0.device).sum() + weights.constant_bo
    ) / sigma2
    result["constant"] = torch.full_like(sigma2, weights.constant_ln2)
    # Per layer-0 neuron.
    result["neuron_direct"] = a0[targets] * weights.direct_neuron.to(r0.device) / sigma2[:, None]
    pooled = torch.einsum("hnt,tj->nhj", B, a0)  # [N, H, 3072]
    result["neuron_value"] = (
        torch.einsum("nhj,jh->nj", pooled, weights.value_neuron.to(r0.device)) / sigma2[:, None]
    )
    total = sum(result[f"direct.{n}"] + result[f"value.{n}"] for n in PIECES)
    total = total + result["value.biases"] + result["constant"]
    result["reconstructed"] = total
    result["z"] = store["z"][targets]
    result["sigma2"] = sigma2
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--neuron", type=int, default=586)
    parser.add_argument("--quantile", type=float, default=99.0)
    parser.add_argument("--examples-per-band", type=int, default=8)
    parser.add_argument("--validation-tokens", type=int, default=200)
    parser.add_argument("--max-docs", type=int, default=None, help="Smoke-test limit.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activations", type=Path, default=ACTIVATIONS)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    result = json.loads(args.result.read_text())
    layer = int(result["layer"])
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    folder = args.result.parent / "neuron-activations"
    calibration = np.load(folder / "activations.npy")[:, neurons.index(args.neuron)]
    threshold = float(np.quantile(calibration, args.quantile / 100))

    dataset = ActivationDataset(args.activations)
    samples = load_torch_file(dataset.path / dataset.manifest["samples_file"])
    rows = np.flatnonzero(calibration >= threshold)
    by_doc: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_doc[int(samples["document_index"][row])].append(int(row))
    doc_order = sorted(by_doc)[: args.max_docs] if args.max_docs else sorted(by_doc)

    rng = np.random.default_rng(args.seed)
    eligible = [row for doc in doc_order for row in by_doc[doc]]
    example_rows: set[int] = set()
    for lo, hi in EXAMPLE_BANDS:
        band = [r for r in eligible if lo <= calibration[r] < hi]
        if band:
            pick = rng.choice(band, size=min(args.examples_per_band, len(band)), replace=False)
            example_rows.update(int(r) for r in pick)
    validation_rows = set(
        int(r)
        for r in rng.choice(
            eligible, size=min(args.validation_tokens, len(eligible)), replace=False
        )
    )

    model_id, revision = dataset.model["repo_id"], dataset.model["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    documents, _ = replay_sample(dataset, tokenizer)
    model = (
        GPT2LMHeadModel.from_pretrained(model_id, revision=revision, attn_implementation="eager")
        .to("cuda")
        .eval()
    )
    weights = Weights(model, layer, args.neuron)
    w0_out = model.transformer.h[0].mlp.c_proj.weight.double()

    n0 = int(weights.direct_neuron.shape[0])
    sum_direct = torch.zeros(n0, dtype=torch.float64, device="cuda")
    sum_value = torch.zeros(n0, dtype=torch.float64, device="cuda")
    sum_pieces: Counter[str] = Counter()
    top5_counts: Counter[int] = Counter()
    top5_tokens: dict[int, Counter[str]] = defaultdict(Counter)
    concentration = []
    max_error = 0.0
    examples, validation = [], []
    count = 0
    for doc in tqdm(doc_order, desc="Input attribution", unit="doc", dynamic_ncols=True):
        doc_rows = by_doc[doc]
        positions = torch.tensor([int(samples["position"][r]) for r in doc_rows], device="cuda")
        ids_full = documents[doc].input_ids
        ids = ids_full[: int(positions.max()) + 1].unsqueeze(0).to("cuda")
        store = forward_capture(model, ids, args.neuron)
        store["w0_out"] = w0_out
        parts = decompose(store, weights, positions)
        max_error = max(max_error, float((parts["reconstructed"] - parts["z"]).abs().max()))
        neuron_total = parts["neuron_direct"] + parts["neuron_value"]  # [N, 3072]
        sum_direct += parts["neuron_direct"].sum(0)
        sum_value += parts["neuron_value"].sum(0)
        for name in PIECES:
            sum_pieces[f"direct.{name}"] += float(parts[f"direct.{name}"].sum())
            sum_pieces[f"value.{name}"] += float(parts[f"value.{name}"].sum())
        sum_pieces["value.biases"] += float(parts["value.biases"].sum())
        sum_pieces["constant"] += float(parts["constant"].sum())
        sum_pieces["z"] += float(parts["z"].sum())
        count += len(doc_rows)
        magnitude = neuron_total.abs()
        ordered = magnitude.sort(-1, descending=True).values.cumsum(-1)
        share = ordered / ordered[:, -1:].clamp_min(1e-12)
        k50 = (share < 0.5).sum(-1) + 1
        k90 = (share < 0.9).sum(-1) + 1
        concentration += list(zip(k50.tolist(), k90.tolist(), strict=True))
        top5 = neuron_total.topk(5, dim=-1).indices.cpu()

        for index, row in enumerate(doc_rows):
            position = int(positions[index])
            token_text = tokenizer.decode([int(ids_full[position])])
            for j in top5[index].tolist():
                top5_counts[j] += 1
                top5_tokens[j][token_text] += 1
            if row in example_rows:
                order = neuron_total[index].abs().topk(10).indices
                start = max(1, position - 24)
                z_window = store["z"][start : position + 1]
                gelu = torch.nn.functional.gelu(z_window)
                source = sum(parts[f"value_source.{n}"][index] for n in PIECES)  # [T]
                source_top = source[1:].abs().topk(min(5, len(source) - 1)).indices + 1
                examples.append(
                    {
                        "row": row,
                        "document_index": doc,
                        "position": position,
                        "token": token_text,
                        "z": float(parts["z"][index]),
                        "activation": float(torch.nn.functional.gelu(parts["z"][index])),
                        "pieces": {
                            key: float(parts[key][index])
                            for key in [f"{p}.{n}" for p in ("direct", "value") for n in PIECES]
                            + ["value.biases", "constant"]
                        },
                        "top_neurons": [
                            {
                                "neuron": int(j),
                                "a0": float(store["a0"][position, j]),
                                "direct": float(parts["neuron_direct"][index, j]),
                                "value": float(parts["neuron_value"][index, j]),
                            }
                            for j in order.tolist()
                        ],
                        "value_sources": [
                            {
                                "offset": int(s) - position,
                                "token": tokenizer.decode([int(ids_full[int(s)])]),
                                "value": float(source[s]),
                            }
                            for s in source_top.tolist()
                        ],
                        "window": [
                            {
                                "text": tokenizer.decode([int(ids_full[p])]),
                                "activation": float(gelu[p - start]),
                                "component_score": 0.0,
                                "target": p == position,
                            }
                            for p in range(start, position + 1)
                        ],
                    }
                )
            if row in validation_rows:
                chosen = neuron_total[index].topk(5).indices.tolist()
                edited = forward_capture(
                    model, ids, args.neuron, edit0=(chosen, torch.tensor(0.0, device="cuda"))
                )
                actual = float(edited["z"][position] - store["z"][position])
                predicted = -float(
                    (
                        parts["neuron_direct"][index, chosen] + parts["neuron_value"][index, chosen]
                    ).sum()
                )
                validation.append({"row": row, "predicted": predicted, "actual": actual})

    mean_direct = (sum_direct / count).cpu()
    mean_value = (sum_value / count).cpu()
    mean_total = mean_direct + mean_value
    top = mean_total.abs().topk(25).indices.tolist()
    k50s, k90s = np.array([c[0] for c in concentration]), np.array([c[1] for c in concentration])
    predicted = np.array([v["predicted"] for v in validation])
    actual = np.array([v["actual"] for v in validation])
    data = {
        "kind": "neuron_input_attribution",
        "layer": layer,
        "neuron": args.neuron,
        "source_layer": 0,
        "selection": {"quantile": args.quantile, "threshold": threshold, "tokens": count},
        "max_reconstruction_error": max_error,
        "mean_z": sum_pieces["z"] / count,
        "mean_pieces": {k: v / count for k, v in sum_pieces.items() if k != "z"},
        "neuron_totals": {
            "mean_direct_sum": float(mean_direct.sum()),
            "mean_value_sum": float(mean_value.sum()),
        },
        "top_neurons": [
            {
                "neuron": int(j),
                "mean_direct": float(mean_direct[j]),
                "mean_value": float(mean_value[j]),
                "mean_total": float(mean_total[j]),
                "top5_fraction": top5_counts[j] / count,
                "top5_tokens": [t for t, _ in top5_tokens[j].most_common(8)],
            }
            for j in top
        ],
        "concentration": {
            "k50_median": float(np.median(k50s)),
            "k90_median": float(np.median(k90s)),
            "k50_p90": float(np.quantile(k50s, 0.9)),
            "k90_p90": float(np.quantile(k90s, 0.9)),
        },
        "validation": {
            "ablation": "zero the token's top-5 layer-0 neurons at all positions",
            "tokens": len(validation),
            "pearson": float(np.corrcoef(predicted, actual)[0, 1]) if len(validation) > 2 else None,
            "mean_predicted": float(predicted.mean()) if len(validation) else None,
            "mean_actual": float(actual.mean()) if len(validation) else None,
            "points": validation,
        },
        "examples": sorted(examples, key=lambda e: -e["z"]),
    }
    data_path = folder / f"input_attribution_N{args.neuron}.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    data_path.with_suffix(".html").write_text(render_any(data))
    print(f"Saved {data_path} and its .html")
    print(f"tokens {count}, max |reconstructed - z| = {max_error:.2e}, mean z {data['mean_z']:.3f}")
    for key, value in data["mean_pieces"].items():
        print(f"  {key:>22}: {value:+.3f}")
    print(
        "top neurons:", [(t["neuron"], round(t["mean_total"], 3)) for t in data["top_neurons"][:10]]
    )
    print("concentration:", data["concentration"])
    print("validation:", {k: v for k, v in data["validation"].items() if k != "points"})


if __name__ == "__main__":
    main()
