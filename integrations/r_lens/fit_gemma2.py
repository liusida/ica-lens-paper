"""Fit a dense RelP R-lens for ``google/gemma-2-2b``.

The forward pass is unchanged. The backward graph uses the RelP LN rule for
Gemma RMSNorm, the identity rule for GELU, and the half rule for the gated MLP
product. Anthropic's ``jlens.fit`` then estimates the averaged transport maps.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda
from progress import configure_jlens_progress
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP, Gemma2RMSNorm

MODEL_ID = "google/gemma-2-2b"
DATASET_ID = "NeelNanda/pile-10k"


class _IdentityActivation(torch.autograd.Function):
    """Exact activation forward with the RelP identity-rule derivative."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        ratio = torch.where(
            values != 0,
            activated / values,
            torch.full_like(values, 0.5),
        )
        ctx.save_for_backward(ratio)
        return activated

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (ratio,) = ctx.saved_tensors
        return grad_output * ratio, None


class _HalfProduct(torch.autograd.Function):
    """Exact product forward with half of the ordinary gradient per input."""

    @staticmethod
    def forward(
        ctx: Any, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(left, right)
        return left * right

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        return 0.5 * grad_output * right, 0.5 * grad_output * left


def install_relp_rules(model: torch.nn.Module) -> tuple[int, int]:
    """Install Gemma 2 dense RelP rules without changing forward values."""

    norm_count = 0
    mlp_count = 0
    for module in model.modules():
        if isinstance(module, Gemma2RMSNorm):

            def relp_norm(current: Gemma2RMSNorm, values: torch.Tensor) -> torch.Tensor:
                float_values = values.float()
                inverse_rms = torch.rsqrt(
                    float_values.pow(2).mean(-1, keepdim=True) + current.eps
                ).detach()
                output = float_values * inverse_rms
                output = output * (1.0 + current.weight.float())
                return output.type_as(values)

            module.forward = MethodType(relp_norm, module)
            norm_count += 1
        elif isinstance(module, Gemma2MLP):

            def relp_mlp(current: Gemma2MLP, values: torch.Tensor) -> torch.Tensor:
                gate_values = current.gate_proj(values)
                gate = _IdentityActivation.apply(
                    gate_values,
                    current.act_fn(gate_values),
                )
                product = _HalfProduct.apply(gate, current.up_proj(values))
                return current.down_proj(product)

            module.forward = MethodType(relp_mlp, module)
            mlp_count += 1
    return norm_count, mlp_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local-r-lens-models/official/gemma-2-2b/lens.pt"),
    )
    parser.add_argument("--prompts", type=int, default=25)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument(
        "--target-layer",
        type=int,
        default=25,
        help="Target residual block; Gemma 2 2B defaults to its final block (25).",
    )
    parser.add_argument(
        "--layers", default="all", help="Comma-separated source layers, or 'all'."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def load_prompts(count: int) -> list[str]:
    if count < 1:
        raise ValueError("--prompts must be positive")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    prompts: list[str] = []
    for row in dataset:
        text = str(row["text"])
        if text.strip():
            prompts.append(text)
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise RuntimeError(f"dataset yielded only {len(prompts)} non-empty documents")
    return prompts


def resolve_layers(value: str, target_layer: int) -> list[int]:
    if value == "all":
        return list(range(target_layer))
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise ValueError("--layers must contain at least one layer")
    if layers[0] < 0 or layers[-1] >= target_layer:
        raise ValueError(
            f"source layers must be between 0 and {target_layer - 1}, got {layers}"
        )
    return layers


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
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    configure_jlens_progress()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        device="cuda",
        dtype=torch.float32,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    hf_model.requires_grad_(False)

    parity_ids = tokenizer(
        "R-lens forward parity check.", return_tensors="pt"
    )["input_ids"].cuda()
    with torch.inference_mode():
        ordinary_logits = hf_model(parity_ids).logits
    norm_count, mlp_count = install_relp_rules(hf_model)
    expected_layers = int(hf_model.config.num_hidden_layers)
    expected_norms = 4 * expected_layers + 1
    if (norm_count, mlp_count) != (expected_norms, expected_layers):
        raise RuntimeError(
            "unexpected Gemma 2 module counts: "
            f"norms={norm_count}/{expected_norms}, MLPs={mlp_count}/{expected_layers}"
        )
    with torch.inference_mode():
        relp_logits = hf_model(parity_ids).logits
    if not torch.equal(ordinary_logits, relp_logits):
        maximum_difference = (ordinary_logits - relp_logits).abs().max().item()
        raise RuntimeError(
            "RelP patch changed Gemma 2 forward values; "
            f"max difference={maximum_difference:.3e}"
        )
    logging.info("Verified bit-identical ordinary and RelP forward logits")

    model = jlens.from_hf(hf_model, tokenizer)
    prompts = load_prompts(args.prompts)
    logging.info(
        "Fitting Gemma 2 2B R-lens: prompts=%d layers=%s target=%d dim_batch=%d",
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
        "model_revision": getattr(hf_model.config, "_commit_hash", None),
        "dataset_id": DATASET_ID,
        "target_layer": args.target_layer,
        "max_seq_len": args.max_seq_len,
        "n_prompts": fitted.n_prompts,
        "skip_first": args.skip_first,
        "estimator": "relp",
        "rules": {
            "ln_rule": True,
            "identity_rule": True,
            "half_rule": True,
        },
        "note": "Dense Gemma 2 RelP with LN, identity, and half rules.",
    }
    torch.save(artifact, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(artifact["provenance"], indent=2) + "\n"
    )
    logging.info("Saved R-lens to %s", args.output.resolve())


if __name__ == "__main__":
    main()
