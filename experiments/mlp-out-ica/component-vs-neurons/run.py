#!/usr/bin/env python3
"""Compare one mlp_out ICA component with GPT-2 MLP neurons by cosine similarity.

Question: is an ICA component that looks clean already aligned with a single MLP
neuron, or is it spread across many neurons?

Geometry (GPT-2, one block):
    mlp_out = sum_i gelu(pre_i) * W_out[i]        W_out = c_proj.weight, [3072, 768]

so neuron i writes along W_out[i] in the 768-d mlp_out space, which is where the
ICA component's writing direction (a column of the writing matrix) also lives.

Two comparisons:
  * writing: cos(component writing direction, W_out[i])  -- "does neuron i write
    the same thing?"  This is the primary comparison.
  * reading: cos(component reading direction, W_out[i])  -- how much neuron i's
    output moves this component's score.

A baseline reports the max |cos| a random direction reaches against the 3072
neurons, so a top match can be judged against chance in 768 dimensions.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import GPT2LMHeadModel

HERE = Path(__file__).resolve().parent
LENS = Path(
    "/home/liusida/research/ICA-paper/icalens_pypi/local-icalens-models/experimental/"
    "icalens-gpt2-small-mlp-out-pile10k"
)


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lens", type=Path, default=LENS)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--component", type=int, default=79)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--baseline-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or HERE / f"L{args.layer}-C{args.component}"
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.lens / "icalens.json").read_text())
    if manifest["activation_site"] != "mlp_out":
        raise ValueError("this experiment expects an mlp_out lens")
    model_id = manifest["model"]["repo_id"]
    revision = manifest["model"]["revision"]
    tensors = load_file(args.lens / f"artifacts/mlp_out/layer_{args.layer:02d}.safetensors")
    writing = tensors["writing_matrix"][:, args.component].astype(np.float64)
    reading = tensors["reading_matrix"][args.component].astype(np.float64)

    profile_path = args.lens / f"component_profiles/mlp_out/layer_{args.layer:02d}.json.gz"
    profile = json.loads(gzip.open(profile_path).read())["components"][args.component]
    tail = profile["tail_direction"]
    # Orient the component so its selected (dominant) tail is positive.
    sign = 1.0 if tail == "positive" else -1.0
    writing *= sign
    reading *= sign

    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision)
    w_out = model.transformer.h[args.layer].mlp.c_proj.weight.detach().double().numpy()
    neurons = unit(w_out)  # [3072, 768], neuron i's output direction

    write_cos = neurons @ unit(writing)
    read_cos = neurons @ unit(reading)

    rng = np.random.default_rng(args.seed)
    random_dirs = unit(rng.standard_normal((args.baseline_samples, neurons.shape[1])))
    random_max = np.abs(random_dirs @ neurons.T).max(axis=1)
    # Baseline among the lens's own components: how close do other components get?
    all_writing = unit(tensors["writing_matrix"].T.astype(np.float64))
    component_max = np.abs(all_writing @ neurons.T).max(axis=1)

    def ranked(values: np.ndarray) -> list[dict[str, float | int]]:
        order = np.argsort(-np.abs(values))[: args.top_k]
        return [
            {
                "neuron": int(index),
                "cos": float(values[index]),
                "abs_cos": float(abs(values[index])),
                "w_out_norm": float(np.linalg.norm(w_out[index])),
            }
            for index in order
        ]

    result = {
        "model": f"{model_id}@{revision}",
        "lens": str(args.lens),
        "layer": args.layer,
        "component": args.component,
        "tail_direction": tail,
        "orientation": "component sign flipped so its selected tail is positive",
        "profile_summary": {
            "excess_kurtosis": profile["score_statistics"]["excess_kurtosis"],
            "top_tail_tokens": [item["text"] for item in profile["examples"][tail]["tokens"][:10]],
            "logit_lens": [t["text"] for t in profile["logit_lens"]["dominant"]["top_tokens"][:10]],
            "r_lens": [t["text"] for t in profile["r_lens"]["dominant"]["top_tokens"][:10]]
            if "r_lens" in profile
            else None,
        },
        "nearest_by_writing_direction": ranked(write_cos),
        "nearest_by_reading_direction": ranked(read_cos),
        "baselines": {
            "random_direction_max_abs_cos": {
                "samples": args.baseline_samples,
                "mean": float(random_max.mean()),
                "p99": float(np.quantile(random_max, 0.99)),
                "max": float(random_max.max()),
            },
            "this_layer_components_max_abs_cos": {
                "median": float(np.median(component_max)),
                "p90": float(np.quantile(component_max, 0.9)),
                "max": float(component_max.max()),
                "this_component": float(component_max[args.component]),
                "this_component_rank_of_768": int(
                    (component_max > component_max[args.component]).sum() + 1
                ),
            },
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(output / "writing_cos_all_neurons.npy", write_cos)
    np.save(output / "reading_cos_all_neurons.npy", read_cos)

    print(f"L{args.layer} C{args.component} ({tail} tail)")
    print("Nearest neurons by writing direction:")
    for row in result["nearest_by_writing_direction"]:
        print(f"  neuron {row['neuron']:4d}  cos={row['cos']:+.3f}")
    print("Nearest neurons by reading direction:")
    for row in result["nearest_by_reading_direction"]:
        print(f"  neuron {row['neuron']:4d}  cos={row['cos']:+.3f}")
    base = result["baselines"]
    print(
        "Random-direction max|cos| vs 3072 neurons: "
        f"mean {base['random_direction_max_abs_cos']['mean']:.3f}, "
        f"p99 {base['random_direction_max_abs_cos']['p99']:.3f}"
    )
    print(f"Saved {output / 'result.json'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
