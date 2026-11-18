#!/usr/bin/env python3
"""Decompose an mlp_out ICA component's score into per-neuron contributions.

Exploration script (not core code). In GPT-2 one MLP block computes

    mlp_out = sum_i act_i * W_out[i] + b_out          (act_i = post-GELU, W_out = c_proj.weight)

and the ICA component's score is ``r . (mlp_out - center)`` with ``r`` its reading
row (oriented so the selected tail is positive). Therefore, exactly,

    score = sum_i act_i * (r . W_out[i])  +  r . (b_out - center)
          = sum_i projected_i             +  constant

``projected_i = act_i * (r . W_out[i])`` is neuron i's activation projected from
its own output direction onto the component's reading direction: how many units of
component score neuron i contributes at this token.

Two modes:
  * dataset (default): uses the 1M-token activations saved by
    ``neuron_activations.py`` and records the projected values of the top-5
    neurons, plus the remainder (all other neurons + bf16 rounding).
  * ``--prompt TEXT``: runs one prompt in float32 and prints, per token, the
    activations, their projected values, the other 3067 neurons' total, the
    constant, and the actual component score (which they sum to exactly).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = Path(__file__).resolve().parent
LENS = Path(
    "/home/liusida/research/ICA-paper/icalens_pypi/local-icalens-models/experimental/"
    "icalens-gpt2-small-mlp-out-pile10k"
)


def component_terms(
    lens: Path, result: dict, model: GPT2LMHeadModel
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Reading row r (oriented), per-neuron weights r.W_out[i], and the constant term."""
    layer, component = int(result["layer"]), int(result["component"])
    tensors = load_file(lens / f"artifacts/mlp_out/layer_{layer:02d}.safetensors")
    sign = 1.0 if result["tail_direction"] == "positive" else -1.0
    reading = torch.tensor(tensors["reading_matrix"][component], dtype=torch.float64) * sign
    center = torch.tensor(tensors["center"], dtype=torch.float64)
    c_proj = model.transformer.h[layer].mlp.c_proj
    w_out = c_proj.weight.detach().double().cpu()  # [3072, 768]
    weights = w_out @ reading  # [3072]
    constant = float(reading @ (c_proj.bias.detach().double().cpu() - center))
    return reading, weights, constant


def dataset_mode(args: argparse.Namespace, result: dict, model: GPT2LMHeadModel) -> None:
    folder = args.result.parent / "neuron-activations"
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    component = int(result["component"])
    activations = np.load(folder / "activations.npy").astype(np.float64)  # [N, 5]
    score = np.load(folder / f"component_C{component}_score.npy").astype(np.float64)
    _, weights, constant = component_terms(args.lens, result, model)
    neuron_weights = weights[neurons].numpy()
    projected = activations * neuron_weights  # [N, 5]
    remainder = score - projected.sum(axis=1) - constant
    np.save(folder / "projected.npy", projected.astype(np.float32))
    np.save(folder / "remainder_other_neurons.npy", remainder.astype(np.float32))

    tail = score > np.quantile(score, 0.99)
    summary = {
        "formula": "C score = sum_i act_i * (r . W_out[i]) + r . (b_out - center)",
        "neurons": neurons,
        "projection_weight_r_dot_w_out": dict(
            zip(map(str, neurons), map(float, neuron_weights), strict=False)
        ),
        "constant_r_dot_bias_minus_center": constant,
        "remainder": "other 3067 neurons plus bf16 rounding of the cached mlp_out",
        "all_tokens": {
            "mean_score": float(score.mean()),
            "mean_projected": dict(
                zip(map(str, neurons), map(float, projected.mean(axis=0)), strict=False)
            ),
            "mean_remainder": float(remainder.mean()),
        },
        "top_1pct_score_tokens": {
            "count": int(tail.sum()),
            "mean_score": float(score[tail].mean()),
            "mean_projected": dict(
                zip(map(str, neurons), map(float, projected[tail].mean(axis=0)), strict=False)
            ),
            "mean_remainder": float(remainder[tail].mean()),
            "constant": constant,
        },
    }
    (folder / "projection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"constant r.(b_out - center) = {constant:+.3f}")
    print(
        "projection weights r.W_out[i]:",
        {n: round(float(w), 3) for n, w in zip(neurons, neuron_weights, strict=False)},
    )
    print(f"Top-1% C{component} tokens (n={tail.sum()}), mean score {score[tail].mean():.2f}:")
    for neuron, value in zip(neurons, projected[tail].mean(axis=0), strict=False):
        print(f"  neuron {neuron:4d} projected {value:+.2f}")
    print(f"  other neurons    {remainder[tail].mean():+.2f}")
    print(f"  constant         {constant:+.2f}")
    print(f"Saved projected.npy, remainder_other_neurons.npy, projection_summary.json to {folder}")


@torch.inference_mode()
def prompt_mode(
    args: argparse.Namespace, result: dict, model: GPT2LMHeadModel, tokenizer: object
) -> None:
    layer, component = int(result["layer"]), int(result["component"])
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    reading, weights, constant = component_terms(args.lens, result, model)
    tensors = load_file(args.lens / f"artifacts/mlp_out/layer_{layer:02d}.safetensors")
    center = torch.tensor(tensors["center"], dtype=torch.float64)

    framing = json.loads((args.lens / "icalens.json").read_text())
    framing = framing["layers"][str(layer)]["fitting"]["provenance"]["document_framing"]
    ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    prefix = [] if framing["strategy"] == "none" else [int(framing["token_id"])]
    input_ids = torch.tensor([prefix + ids])

    captured: dict[str, torch.Tensor] = {}
    mlp = model.transformer.h[layer].mlp
    handles = [
        mlp.c_proj.register_forward_pre_hook(
            lambda _m, inputs: captured.__setitem__("act", inputs[0][0].double())
        ),
        mlp.register_forward_hook(
            lambda _m, _i, output: captured.__setitem__("out", output[0].double())
        ),
    ]
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    act = captured["act"][len(prefix) :]  # [T, 3072]
    actual = (captured["out"][len(prefix) :] - center) @ reading  # [T]
    contributions = act * weights  # [T, 3072]
    selected = contributions[:, neurons]
    others = contributions.sum(dim=1) - selected.sum(dim=1)
    reconstructed = contributions.sum(dim=1) + constant
    max_error = float((reconstructed - actual).abs().max())

    tokens = [tokenizer.decode([token]) for token in ids]
    header = (
        ["token"]
        + [f"act {n}" for n in neurons]
        + [f"proj {n}" for n in neurons]
        + ["other neurons", "constant", f"C{component} actual"]
    )
    lines = [
        f"# L{layer} C{component} decomposition for one prompt\n",
        f"Prompt: {args.prompt!r}\n",
        "proj i = act i x (r . W_out[i]); "
        f"C{component} = sum of proj over all 3072 neurons + constant.",
        "Projection weights r.W_out[i]: "
        + ", ".join(f"{n}: {float(weights[n]):+.3f}" for n in neurons)
        + f"; constant r.(b_out - center) = {constant:+.3f}. "
        f"Max |reconstructed - actual| = {max_error:.2e}.\n",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    rows = []
    for index, token in enumerate(tokens):
        row = {
            "token": token,
            "activation": {str(n): float(act[index, n]) for n in neurons},
            "projected": {str(n): float(selected[index, k]) for k, n in enumerate(neurons)},
            "other_neurons": float(others[index]),
            "constant": constant,
            "component_actual": float(actual[index]),
        }
        rows.append(row)
        cells = (
            [repr(token).replace("|", "\\|")]
            + [f"{act[index, n]:.2f}" for n in neurons]
            + [f"{selected[index, k]:+.2f}" for k in range(len(neurons))]
            + [f"{others[index]:+.2f}", f"{constant:+.2f}", f"**{actual[index]:.2f}**"]
        )
        lines.append("| " + " | ".join(cells) + " |")
    folder = args.result.parent / "prompts"
    folder.mkdir(exist_ok=True)
    stem = args.name or "prompt"
    (folder / f"{stem}.md").write_text("\n".join(lines) + "\n")
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "layer": layer,
                "component": component,
                "neurons": neurons,
                "projection_weights": {str(n): float(weights[n]) for n in neurons},
                "constant": constant,
                "max_reconstruction_error": max_error,
                "tokens": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print("\n".join(lines))
    print(f"\nSaved {folder / (stem + '.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--lens", type=Path, default=LENS)
    parser.add_argument("--prompt", default=None, help="Decompose one prompt instead.")
    parser.add_argument("--name", default=None, help="Output file stem for --prompt.")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    manifest = json.loads((args.lens / "icalens.json").read_text())
    model_id, revision = manifest["model"]["repo_id"], manifest["model"]["revision"]
    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision).eval()
    if args.prompt is None:
        dataset_mode(args, result, model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        prompt_mode(args, result, model, tokenizer)


if __name__ == "__main__":
    main()
