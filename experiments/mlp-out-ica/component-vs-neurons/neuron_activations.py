#!/usr/bin/env python3
"""Profile GPT-2 MLP neuron activations on the mlp_out lens's 1M-token profiling sample.

Exploration script (not core code). For each neuron nearest to an ICA component
(from ``run.py``'s ``result.json``) it records the post-GELU activation
(the input to ``c_proj``) at exactly the 1M tokens used to fit and profile the
mlp_out lens, then reports:

  * the top-20 activating tokens with context,
  * the full activation distribution (histogram + summary statistics),
  * the Pearson correlation with the ICA component's score at the same tokens.

The token sample is replayed from the activation dataset's provenance (same
dataset revision, framing, context length, candidate pool and seed), and every
replayed token is checked against the dataset's stored token IDs.
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
from icalens.cli.fit_text import load_pile_documents, sample_positions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_report import REPORT_STEM, render  # noqa: E402

HERE = Path(__file__).resolve().parent
LENS = Path(
    "/home/liusida/research/ICA-paper/icalens_pypi/local-icalens-models/experimental/"
    "icalens-gpt2-small-mlp-out-pile10k"
)
ACTIVATIONS = Path(
    "/home/liusida/Expansion/research/ICA-data/icalens-activations/gpt2-mlp-out-pile10k-1m"
)


class _Stop(Exception):
    pass


def replay_sample(dataset: ActivationDataset, tokenizer: Any) -> tuple[list[Any], dict]:
    provenance = dataset.provenance
    source = provenance["dataset"]
    documents = load_pile_documents(
        tokenizer,
        dataset_id=source["repo_id"],
        dataset_revision=source["revision"],
        split=source["split"],
        text_field=provenance["text_field"],
        candidate_token_budget=int(provenance["candidate_tokens"]),
        context_length=int(provenance["context_length"]),
        document_framing=provenance["document_framing"],
    )
    selected = sample_positions(
        documents,
        token_budget=int(provenance["fitting_tokens"]),
        seed=int(provenance["sampling_seed"]),
    )
    return documents, selected


@torch.inference_mode()
def collect(
    model: GPT2LMHeadModel,
    documents: list[Any],
    selected: dict[int, torch.Tensor],
    *,
    layer: int,
    neurons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = torch.tensor(neurons, device="cuda")
    store: dict[str, torch.Tensor] = {}
    positions_holder: dict[str, torch.Tensor] = {}

    def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        hidden = inputs[0][0]  # [seq, 3072] post-GELU activations
        store["values"] = hidden.index_select(0, positions_holder["p"]).index_select(1, index)
        raise _Stop

    handle = model.transformer.h[layer].mlp.c_proj.register_forward_pre_hook(hook)
    rows: list[np.ndarray] = []
    doc_ids: list[np.ndarray] = []
    pos_ids: list[np.ndarray] = []
    try:
        for document_index, positions in tqdm(
            selected.items(), desc="Neuron activations", unit="doc", dynamic_ncols=True
        ):
            input_ids = documents[document_index].input_ids.unsqueeze(0).to("cuda")
            positions_holder["p"] = positions.to("cuda")
            try:
                model(input_ids=input_ids, use_cache=False)
            except _Stop:
                pass
            rows.append(store["values"].float().cpu().numpy())
            doc_ids.append(np.full(len(positions), document_index))
            pos_ids.append(positions.numpy())
    finally:
        handle.remove()
    return np.concatenate(rows), np.concatenate(doc_ids), np.concatenate(pos_ids)


def describe(values: np.ndarray) -> dict[str, float]:
    centered = values - values.mean()
    variance = float((centered**2).mean())
    return {
        "mean": float(values.mean()),
        "std": float(np.sqrt(variance)),
        "min": float(values.min()),
        "max": float(values.max()),
        "skewness": float((centered**3).mean() / variance**1.5),
        "excess_kurtosis": float((centered**4).mean() / variance**2 - 3.0),
        "fraction_positive": float((values > 0).mean()),
        "fraction_above_1": float((values > 1).mean()),
        **{f"p{q}": float(np.quantile(values, q / 100)) for q in (1, 10, 50, 90, 99, 99.9)},
    }


def context(tokenizer: Any, ids: torch.Tensor, position: int, left: int = 24, right: int = 6):
    start = max(1, position - left)  # skip the framing token at position 0
    before = tokenizer.decode(ids[start:position].tolist())
    target = tokenizer.decode([int(ids[position])])
    after = tokenizer.decode(ids[position + 1 : position + 1 + right].tolist())
    return before, target, after


@torch.inference_mode()
def token_window(
    model: GPT2LMHeadModel,
    tokenizer: Any,
    ids: torch.Tensor,
    position: int,
    *,
    layer: int,
    neuron: int,
    reading: torch.Tensor,
    center: torch.Tensor,
    left: int = 24,
    right: int = 8,
) -> list[dict[str, Any]]:
    """Per-token neuron activation and component score around one example."""
    captured: dict[str, torch.Tensor] = {}
    mlp = model.transformer.h[layer].mlp
    handles = [
        mlp.c_proj.register_forward_pre_hook(
            lambda _m, inputs: captured.__setitem__("act", inputs[0][0, :, neuron].float())
        ),
        mlp.register_forward_hook(
            lambda _m, _i, output: captured.__setitem__("out", output[0].float())
        ),
    ]
    stop = min(len(ids), position + 1 + right)
    try:
        model(input_ids=ids[:stop].unsqueeze(0).to("cuda"), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    scores = (captured["out"].cpu() - center) @ reading
    start = max(1, position - left)  # position 0 is the framing token
    return [
        {
            "text": tokenizer.decode([int(ids[index])]),
            "activation": float(captured["act"][index]),
            "component_score": float(scores[index]),
            "target": index == position,
        }
        for index in range(start, stop)
    ]


def add_context_windows(
    output: Path,
    summary: dict[str, Any],
    documents: list[Any],
    tokenizer: Any,
    model: GPT2LMHeadModel,
    reading: torch.Tensor,
    center: torch.Tensor,
) -> None:
    """Store per-token activations around every top example, then render the HTML."""
    layer = int(summary["layer"])
    for neuron, entry in summary["neurons"].items():
        entry["top_example_windows"] = [
            token_window(
                model,
                tokenizer,
                documents[example["document_index"]].input_ids,
                example["position"],
                layer=layer,
                neuron=int(neuron),
                reading=reading,
                center=center,
            )
            for example in entry["top_examples"]
        ]
    data_path = output / f"{REPORT_STEM}.json"
    data_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {data_path}")
    html_path = data_path.with_suffix(".html")
    html_path.write_text(render(summary))
    print(f"Saved {html_path} (re-render from the twin .json with render_report.py)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, default=HERE / "L1-C79" / "result.json")
    parser.add_argument("--lens", type=Path, default=LENS)
    parser.add_argument("--activations", type=Path, default=ACTIVATIONS)
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument(
        "--windows-only",
        action="store_true",
        help=(
            "Recompute only the per-token context windows into an existing top_examples.json "
            "(no 1M-token pass). For UI-only changes run render_report.py instead."
        ),
    )
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    layer, component = int(result["layer"]), int(result["component"])
    neurons = [int(row["neuron"]) for row in result["nearest_by_writing_direction"]]
    output = args.result.parent / "neuron-activations"
    output.mkdir(exist_ok=True)

    dataset = ActivationDataset(args.activations)
    model_id = dataset.model["repo_id"]
    revision = dataset.model["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    documents, selected = replay_sample(dataset, tokenizer)

    samples = load_torch_file(dataset.path / dataset.manifest["samples_file"])
    replayed_tokens = torch.cat(
        [documents[d].input_ids.index_select(0, p) for d, p in selected.items()]
    )
    if not torch.equal(replayed_tokens, samples["token_id"].long()):
        raise RuntimeError("replayed tokens do not match the activation dataset's sample")

    model = GPT2LMHeadModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    tensors = load_file(args.lens / f"artifacts/mlp_out/layer_{layer:02d}.safetensors")
    sign = 1.0 if result["tail_direction"] == "positive" else -1.0
    reading = torch.tensor(tensors["reading_matrix"][component], dtype=torch.float32) * sign
    center = torch.tensor(tensors["center"], dtype=torch.float32)
    if args.windows_only:
        summary = json.loads((output / f"{REPORT_STEM}.json").read_text())
        add_context_windows(output, summary, documents, tokenizer, model, reading, center)
        return
    values, doc_ids, pos_ids = collect(model, documents, selected, layer=layer, neurons=neurons)
    np.save(output / "activations.npy", values)  # [1M, n_neurons], columns follow `neurons`

    # ICA component score at the same rows (activation-dataset row order).
    layer_values = dataset.layer(layer)
    score = torch.cat(
        [
            (layer_values[start : start + 65536].float() - center) @ reading
            for start in range(0, dataset.sample_count, 65536)
        ]
    ).numpy()
    np.save(output / f"component_C{component}_score.npy", score)

    summary: dict[str, Any] = {
        "layer": layer,
        "component": component,
        "tokens": int(values.shape[0]),
        "activation": "post-GELU (input to c_proj)",
        "neurons": {},
        "component_score": describe(score),
    }
    for column, neuron in enumerate(neurons):
        activation = values[:, column]
        order = np.argsort(-activation)[: args.top_examples]
        examples = []
        for row in order:
            ids = documents[int(doc_ids[row])].input_ids
            before, target, after = context(tokenizer, ids, int(pos_ids[row]))
            examples.append(
                {
                    "activation": float(activation[row]),
                    "component_score": float(score[row]),
                    "document_index": int(doc_ids[row]),
                    "position": int(pos_ids[row]),
                    "token": target,
                    "context": before + "[[" + target + "]]" + after,
                }
            )
        summary["neurons"][str(neuron)] = {
            "writing_cos": result["nearest_by_writing_direction"][column]["cos"],
            "statistics": describe(activation),
            "pearson_with_component_score": float(np.corrcoef(activation, score)[0, 1]),
            "top_examples": examples,
        }
    add_context_windows(output, summary, documents, tokenizer, model, reading, center)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = len(neurons) + 1
    figure, axes = plt.subplots(2, columns, figsize=(3.0 * columns, 5.6))
    for column, neuron in enumerate(neurons):
        activation = values[:, column]
        axis = axes[0, column]
        axis.hist(activation, bins=200, color="#1d4ed8", log=True)
        stats = summary["neurons"][str(neuron)]["statistics"]
        axis.set_title(f"neuron {neuron}\nkurt {stats['excess_kurtosis']:.1f}", fontsize=8)
        axis.tick_params(labelsize=6)
        axis = axes[1, column]
        axis.scatter(activation, score, s=0.3, alpha=0.15, color="#0f172a", rasterized=True)
        corr = summary["neurons"][str(neuron)]["pearson_with_component_score"]
        axis.set_title(f"vs C{component} score, r={corr:+.2f}", fontsize=8)
        axis.set_xlabel("activation", fontsize=7)
        axis.tick_params(labelsize=6)
    axes[0, -1].hist(score, bins=200, color="#c2410c", log=True)
    axes[0, -1].set_title(
        f"C{component} score\nkurt {summary['component_score']['excess_kurtosis']:.1f}", fontsize=8
    )
    axes[0, -1].tick_params(labelsize=6)
    axes[1, -1].axis("off")
    axes[0, 0].set_ylabel("tokens (log)", fontsize=7)
    axes[1, 0].set_ylabel(f"C{component} score", fontsize=7)
    figure.suptitle(
        f"GPT-2 L{layer} MLP neurons nearest to mlp_out C{component} "
        f"({values.shape[0]:,} profiling tokens)",
        fontsize=9,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"distributions.{suffix}", dpi=200)
    print(f"Saved results to {output}")
    for neuron in neurons:
        entry = summary["neurons"][str(neuron)]
        print(
            f"neuron {neuron}: kurt {entry['statistics']['excess_kurtosis']:.1f}, "
            f"r={entry['pearson_with_component_score']:+.3f}, top tokens "
            f"{[e['token'] for e in entry['top_examples'][:8]]}"
        )


if __name__ == "__main__":
    main()
