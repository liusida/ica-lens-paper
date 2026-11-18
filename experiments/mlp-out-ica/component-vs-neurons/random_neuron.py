#!/usr/bin/env python3
"""Is L1 N586 cleaner than a random L1 neuron? Compare it with a random neuron and a population.

Exploration script (not core code). N586 was found as the neuron nearest to the clean ICA
component C79, so it may be unusually clean (a selection effect). This script:

  * picks one random L1 neuron R (seeded; excluding C79's 5 nearest neurons) for a
    side-by-side look, plus a population of ``--population`` random L1 neurons (R included)
    for a reference distribution;
  * records their post-GELU activations (and N586's) on the same 1M profiling tokens;
  * computes per-neuron "cleanliness" metrics:
      - excess kurtosis and fraction active (> 0),
      - best |Pearson r| with any of the 768 mlp_out ICA component scores (how well a single
        component explains the neuron),
      - best |cos| between the neuron's writing direction (W_out row) and any ICA writing
        direction,
      - token concentration of the neuron's top-1% tokens: effective number of token types
        exp(H) and the share of those tokens covered by its 10 most common token types;
  * stores top-20 examples and random examples from percentile bands of the upper tail
    (same percentile bands for both neurons, so they are comparable).

Writes ``random_neuron_N<R>_vs_N586.json`` and its twin ``.html`` (render_report.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file
from safetensors.torch import load_file as load_torch_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from icalens._activation_dataset import ActivationDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_activations import (  # noqa: E402
    ACTIVATIONS,
    LENS,
    collect,
    describe,
    replay_sample,
    token_window,
)
from render_report import render_any  # noqa: E402

HERE = Path(__file__).resolve().parent
TAIL_BANDS = (99.0, 99.5, 99.9, 99.99, 100.0)  # percentile bands of the upper tail


@torch.inference_mode()
def component_correlations(
    dataset: ActivationDataset,
    layer: int,
    reading: torch.Tensor,
    center: torch.Tensor,
    activations: np.ndarray,
    chunk: int = 65536,
) -> np.ndarray:
    """Pearson r between every neuron column and every ICA component score -> [n, 768]."""
    values = dataset.layer(layer)
    reading, center = reading.cuda(), center.cuda()
    n = activations.shape[1]
    k = reading.shape[0]
    sa = torch.zeros(n, dtype=torch.float64, device="cuda")
    saa = torch.zeros_like(sa)
    ss = torch.zeros(k, dtype=torch.float64, device="cuda")
    sss = torch.zeros_like(ss)
    sas = torch.zeros(n, k, dtype=torch.float64, device="cuda")
    total = dataset.sample_count
    for start in tqdm(range(0, total, chunk), desc="Component correlations", dynamic_ncols=True):
        x = values[start : start + chunk].float().cuda()
        s = ((x - center) @ reading.T).double()
        a = torch.from_numpy(activations[start : start + chunk]).cuda().double()
        sa += a.sum(0)
        saa += (a * a).sum(0)
        ss += s.sum(0)
        sss += (s * s).sum(0)
        sas += a.T @ s
    cov = sas / total - torch.outer(sa / total, ss / total)
    var_a = saa / total - (sa / total) ** 2
    var_s = sss / total - (ss / total) ** 2
    return (cov / torch.sqrt(torch.outer(var_a, var_s))).cpu().numpy()


def token_concentration(activation: np.ndarray, token_ids: np.ndarray) -> dict[str, Any]:
    threshold = float(np.quantile(activation, 0.99))
    ids = token_ids[activation >= threshold]
    _, counts = np.unique(ids, return_counts=True)
    p = np.sort(counts)[::-1] / counts.sum()
    return {
        "q99_threshold": threshold,
        "q99_tokens": int(len(ids)),
        "distinct_types": int(len(counts)),
        "effective_types": float(np.exp(-(p * np.log(p)).sum())),
        "top10_type_share": float(p[:10].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--reference", type=int, default=586)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument("--per-band", type=int, default=8)
    parser.add_argument("--lens", type=Path, default=LENS)
    parser.add_argument("--activations", type=Path, default=ACTIVATIONS)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    layer = int(result["layer"])
    excluded = {int(row["neuron"]) for row in result["nearest_by_writing_direction"]}
    rng = np.random.default_rng(args.seed)
    pool = np.array([n for n in range(3072) if n not in excluded and n != args.reference])
    population = [int(n) for n in rng.choice(pool, size=args.population, replace=False)]
    neurons = [args.reference, *population]

    dataset = ActivationDataset(args.activations)
    model_id, revision = dataset.model["repo_id"], dataset.model["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    documents, selected = replay_sample(dataset, tokenizer)
    samples = load_torch_file(dataset.path / dataset.manifest["samples_file"])
    token_ids = samples["token_id"].numpy()
    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    values, doc_ids, pos_ids = collect(model, documents, selected, layer=layer, neurons=neurons)
    if not np.array_equal(doc_ids, samples["document_index"].numpy()) or not np.array_equal(
        pos_ids, samples["position"].numpy()
    ):
        raise RuntimeError("collected rows do not align with the activation dataset")

    tensors = load_file(args.lens / f"artifacts/mlp_out/layer_{layer:02d}.safetensors")
    reading = torch.tensor(tensors["reading_matrix"], dtype=torch.float32)  # [768, 768]
    writing = torch.tensor(tensors["writing_matrix"], dtype=torch.float32)  # [768, 768], cols
    center = torch.tensor(tensors["center"], dtype=torch.float32)
    corr = component_correlations(dataset, layer, reading, center, values)

    w_out = model.transformer.h[layer].mlp.c_proj.weight.detach().float().cpu()[neurons]  # [n, 768]
    w_out = w_out / w_out.norm(dim=1, keepdim=True)
    w_dirs = writing / writing.norm(dim=0, keepdim=True)
    cos = (w_out @ w_dirs).numpy()  # [n, 768]

    metrics = []
    for column, neuron in enumerate(neurons):
        activation = values[:, column]
        stats = describe(activation)
        best_r = int(np.argmax(np.abs(corr[column])))
        best_cos = int(np.argmax(np.abs(cos[column])))
        metrics.append(
            {
                "neuron": neuron,
                "excess_kurtosis": stats["excess_kurtosis"],
                "fraction_positive": stats["fraction_positive"],
                "max": stats["max"],
                "best_component_by_r": best_r,
                "best_abs_r": float(abs(corr[column, best_r])),
                "best_r_sign": float(np.sign(corr[column, best_r])),
                "second_abs_r": float(np.sort(np.abs(corr[column]))[-2]),
                "best_component_by_cos": best_cos,
                "best_abs_cos": float(abs(cos[column, best_cos])),
                **token_concentration(activation, token_ids),
            }
        )

    # R = first population neuron that is alive (some neurons never fire on this sample).
    r_column = next(i for i, m in enumerate(metrics) if i > 0 and m["max"] >= 1.0)
    random_neuron = neurons[r_column]
    dead = [m["neuron"] for m in metrics[1:r_column]]

    def example_entry(column: int) -> dict[str, Any]:
        neuron = neurons[column]
        row_metrics = metrics[column]
        component = row_metrics["best_component_by_r"]
        comp_reading = reading[component] * row_metrics["best_r_sign"]
        activation = values[:, column]

        def make(row: int) -> dict[str, Any]:
            ids = documents[int(doc_ids[row])].input_ids
            window = token_window(
                model,
                tokenizer,
                ids,
                int(pos_ids[row]),
                layer=layer,
                neuron=neuron,
                reading=comp_reading,
                center=center,
            )
            return {
                "activation": float(activation[row]),
                "component_score": next(t["component_score"] for t in window if t["target"]),
                "token": next(t["text"] for t in window if t["target"]),
                "window": window,
            }

        top = [make(int(r)) for r in np.argsort(-activation)[: args.top_examples]]
        edges = [float(np.quantile(activation, q / 100)) for q in TAIL_BANDS]
        edges[-1] = float(activation.max()) + 1e-6
        bands = []
        for (q_lo, q_hi), lo, hi in zip(
            zip(TAIL_BANDS[:-1], TAIL_BANDS[1:], strict=True), edges[:-1], edges[1:], strict=True
        ):
            rows = np.flatnonzero((activation >= lo) & (activation < hi))
            chosen = rng.choice(rows, size=min(args.per_band, len(rows)), replace=False)
            chosen = chosen[np.argsort(-activation[chosen])]
            bands.append(
                {
                    "q_lo": q_lo,
                    "q_hi": q_hi,
                    "lo": lo,
                    "hi": hi,
                    "count": int(len(rows)),
                    "examples": [make(int(r)) for r in chosen],
                }
            )
        return {
            "neuron": neuron,
            "score_component": component,
            "color_scale": float(activation.max()),
            "top_examples": top,
            "bands": bands,
        }

    data = {
        "kind": "random_neuron_comparison",
        "layer": layer,
        "tokens": int(values.shape[0]),
        "reference": args.reference,
        "random_neuron": random_neuron,
        "population": {
            "size": args.population,
            "seed": args.seed,
            "excluded": sorted(excluded | {args.reference}),
            "skipped_dead_before_random": dead,
            "dead_fraction": float(np.mean([m["max"] < 0.05 for m in metrics[1:]])),
        },
        "metrics": metrics,  # metrics[0] = reference, metrics[1] = random neuron R
        "random_index": r_column,
        "examples": [example_entry(0), example_entry(r_column)],
    }
    folder = args.result.parent / "neuron-activations"
    data_path = folder / f"random_neuron_N{random_neuron}_vs_N{args.reference}.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    data_path.with_suffix(".html").write_text(render_any(data))
    print(f"Saved {data_path} and its .html")
    keys = (
        "excess_kurtosis",
        "fraction_positive",
        "best_abs_r",
        "best_abs_cos",
        "effective_types",
        "top10_type_share",
    )
    pop = metrics[1:]
    for key in keys:
        ref, rnd = metrics[0][key], metrics[r_column][key]
        rank = np.mean([m[key] < ref for m in pop])
        med = float(np.median([m[key] for m in pop]))
        print(
            f"{key:>18}: N{args.reference} {ref:.3f} (pop pct {100 * rank:.0f})  "
            f"N{random_neuron} {rnd:.3f}  pop median {med:.3f}"
        )


if __name__ == "__main__":
    main()
