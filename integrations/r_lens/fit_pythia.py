"""Fit the official Pythia-70M R-lens used by ICA Lens profiling.

Pythia uses GPT-NeoX LayerNorm and GELU blocks, so this applies the same
LayerNorm-only RelP rule as the GPT-2 fitter. The forward pass stays exact;
only the normalization-scale term is detached in the backward pass.

Run from the ICALens project root:

    uv run python integrations/r_lens/fit_pythia.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from fit_gpt2 import install_relp_layernorm_rule, load_prompts, resolve_layers
from gb10_load_llm import load_model_to_cuda
from progress import configure_jlens_progress
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-70m"
MODEL_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
DATASET_ID = "NeelNanda/pile-10k"
EXPECTED_LAYER_NORMS = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local-r-lens-models/official/pythia-70m/lens.pt"),
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=25,
        help="Pile-10k documents to average, matching the existing official lenses.",
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--skip-first",
        type=int,
        default=4,
        help="Leading token positions excluded from the Jacobian average.",
    )
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=32,
        help="Jacobian output dimensions computed per backward pass.",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=5,
        help="Target residual block; Pythia-70M defaults to its final block (5).",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated source layers below target, or 'all'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Resumable fitting checkpoint; defaults beside --output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import jlens
    except ImportError as error:
        raise SystemExit(
            "Install the official fitter first:\n"
            "  uv pip install "
            '"git+https://github.com/anthropics/jacobian-lens.git"'
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.output.with_name(
        f"fit-checkpoint-layer-{args.target_layer:02d}.pt"
    )
    source_layers = resolve_layers(args.layers, args.target_layer)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    configure_jlens_progress()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda",
        dtype=torch.float32,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    hf_model.requires_grad_(False)

    parity_ids = tokenizer("R-lens forward parity check.", return_tensors="pt")[
        "input_ids"
    ].cuda()
    with torch.inference_mode():
        ordinary_logits = hf_model(parity_ids).logits
    layer_norm_count = install_relp_layernorm_rule(hf_model)
    if layer_norm_count != EXPECTED_LAYER_NORMS:
        raise RuntimeError(
            f"expected {EXPECTED_LAYER_NORMS} Pythia LayerNorm modules, found "
            f"{layer_norm_count}; the architecture may have changed"
        )
    with torch.inference_mode():
        relp_logits = hf_model(parity_ids).logits
    if not torch.equal(ordinary_logits, relp_logits):
        maximum_difference = (ordinary_logits - relp_logits).abs().max().item()
        raise RuntimeError(
            "RelP patch changed Pythia forward values; "
            f"max difference={maximum_difference:.3e}"
        )
    logging.info("Verified bit-identical ordinary and RelP forward logits")

    # Current Transformers exposes the GPT-NeoX unembedding as the top-level
    # ``lm_head``. Pass the layout explicitly because jlens' built-in Pythia
    # layout still expects the older ``embed_out`` name.
    layout = jlens.hf.Layout(
        "gpt_neox",
        norm="final_layer_norm",
        embed="embed_in",
        lm_head="lm_head",
    )
    model = jlens.from_hf(hf_model, tokenizer, layout=layout)
    prompts = load_prompts(args.prompts)
    logging.info(
        "Fitting Pythia-70M R-lens: prompts=%d layers=%s target=%d dim_batch=%d",
        len(prompts),
        source_layers,
        args.target_layer,
        args.dim_batch,
    )
    fitted = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=1,
        resume=True,
    )

    fitted.jacobians[args.target_layer] = torch.eye(fitted.d_model)
    fitted.source_layers = sorted(fitted.jacobians)
    fitted.save(str(args.output), dtype=torch.float16)

    artifact = torch.load(args.output, map_location="cpu", weights_only=True)
    artifact["provenance"] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "target_layer": args.target_layer,
        "max_seq_len": args.max_seq_len,
        "n_prompts": fitted.n_prompts,
        "skip_first": args.skip_first,
        "estimator": "relp",
        "rules": {
            "layer_norm_rule": True,
            "identity_rule": False,
            "half_rule": False,
        },
        "note": (
            "Pythia/GPT-NeoX-specific R-lens: exact LayerNorm forward with "
            "detached normalization-scale backward; Pythia has no SiLU/SwiGLU rules."
        ),
    }
    torch.save(artifact, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(artifact["provenance"], indent=2) + "\n"
    )
    logging.info("Saved R-lens to %s", args.output.resolve())


if __name__ == "__main__":
    main()
