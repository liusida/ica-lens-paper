"""Fit a GPT-2-small R-lens compatible with GPT-2 ICA Lens activations.

This is an experimental GPT-2-specific RelP wrapper around Anthropic's
``jacobian-lens`` reference fitter. GPT-2 uses LayerNorm and GELU rather than
RMSNorm/SiLU/SwiGLU, so the applicable R-lens modification here is the
LayerNorm rule: the ordinary forward pass is preserved exactly, while the
normalization scale is treated as constant during the backward pass.

Install the reference fitter before running this script:

    uv pip install "git+https://github.com/anthropics/jacobian-lens.git"

Run from the ICALens project root:

    uv run python integrations/r_lens/fit_gpt2.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda
from progress import configure_jlens_progress
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai-community/gpt2"
DATASET_ID = "NeelNanda/pile-10k"


class _RelPLayerNorm(torch.autograd.Function):
    """Exact LayerNorm forward with the RelP normalization backward rule."""

    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        normalized_shape: tuple[int, ...],
        eps: float,
    ) -> torch.Tensor:
        output = F.layer_norm(values, normalized_shape, weight, bias, eps)
        dimensions = tuple(range(values.ndim - len(normalized_shape), values.ndim))
        variance = values.float().var(dim=dimensions, unbiased=False, keepdim=True)
        inverse_std = torch.rsqrt(variance + eps).to(values.dtype)
        ctx.dimensions = dimensions
        ctx.save_for_backward(inverse_std, weight)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        inverse_std, weight = ctx.saved_tensors
        weighted_gradient = grad_output * weight
        centered_gradient = weighted_gradient - weighted_gradient.mean(
            dim=ctx.dimensions,
            keepdim=True,
        )
        grad_values = centered_gradient * inverse_std
        return grad_values, None, None, None, None


def install_relp_layernorm_rule(model: torch.nn.Module) -> int:
    """Replace LayerNorm backward behavior without changing forward values."""

    count = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.LayerNorm):
            continue

        def relp_forward(
            current: torch.nn.LayerNorm,
            values: torch.Tensor,
        ) -> torch.Tensor:
            if current.weight is None or current.bias is None:
                raise ValueError("GPT-2 R-lens expects affine LayerNorm modules")
            return _RelPLayerNorm.apply(
                values,
                current.weight,
                current.bias,
                tuple(current.normalized_shape),
                float(current.eps),
            )

        module.forward = MethodType(relp_forward, module)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local-r-lens-models/official/gpt2-small/lens.pt"),
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=25,
        help="Pile-10k documents to average; 25 is a pilot, ~100 is stronger.",
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
        default=11,
        help="Target residual block; GPT-2 defaults to its final block (11).",
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


def load_prompts(count: int) -> list[str]:
    if count < 1:
        raise ValueError("--prompts must be positive")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    prompts = []
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
            "\"git+https://github.com/anthropics/jacobian-lens.git\""
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
        "R-lens forward parity check.",
        return_tensors="pt",
    )["input_ids"].cuda()
    with torch.inference_mode():
        ordinary_logits = hf_model(parity_ids).logits
    layer_norm_count = install_relp_layernorm_rule(hf_model)
    if layer_norm_count != 25:
        raise RuntimeError(
            f"expected 25 GPT-2 LayerNorm modules, found {layer_norm_count}; "
            "the architecture may have changed"
        )
    with torch.inference_mode():
        relp_logits = hf_model(parity_ids).logits
    if not torch.equal(ordinary_logits, relp_logits):
        maximum_difference = (ordinary_logits - relp_logits).abs().max().item()
        raise RuntimeError(
            "RelP patch changed GPT-2 forward values; "
            f"max difference={maximum_difference:.3e}"
        )
    logging.info("Verified bit-identical ordinary and RelP forward logits")

    model = jlens.from_hf(hf_model, tokenizer)
    prompts = load_prompts(args.prompts)
    logging.info(
        "Fitting GPT-2 R-lens: prompts=%d layers=%s target=%d dim_batch=%d",
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

    # The target layer maps to itself. Include this identity row so the saved
    # artifact follows the layout used by published J/R-lens files.
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
            "layer_norm_rule": True,
            "identity_rule": False,
            "half_rule": False,
        },
        "note": (
            "GPT-2-specific R-lens: exact LayerNorm forward with detached "
            "normalization-scale backward; GPT-2 has no SiLU/SwiGLU rules."
        ),
    }
    torch.save(artifact, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(artifact["provenance"], indent=2) + "\n"
    )
    logging.info("Saved R-lens to %s", args.output.resolve())


if __name__ == "__main__":
    main()
