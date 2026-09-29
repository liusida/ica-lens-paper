"""Fit a dense RelP R-lens for a supported Qwen 3.5 Base checkpoint."""

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
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

DATASET_ID = "NeelNanda/pile-10k"
PRESETS = {
    "Qwen/Qwen3.5-2B-Base": {
        "target_layer": 23,
        "output_name": "qwen3.5-2b-base",
    },
    "Qwen/Qwen3.5-9B-Base": {
        "target_layer": 30,
        "output_name": "qwen3.5-9b-base",
    },
}


class _IdentityActivation(torch.autograd.Function):
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
    """Install the dense Qwen RelP rules used by the published R-lens arm."""

    norm_count = 0
    mlp_count = 0
    for name, module in model.named_modules():
        is_residual_norm = (
            name.endswith(".input_layernorm")
            or name.endswith(".post_attention_layernorm")
            or name == "model.norm"
        )
        if isinstance(module, Qwen3_5RMSNorm) and is_residual_norm:

            def relp_norm(current: Qwen3_5RMSNorm, values: torch.Tensor) -> torch.Tensor:
                float_values = values.float()
                inverse_rms = torch.rsqrt(
                    float_values.pow(2).mean(-1, keepdim=True) + current.eps
                ).detach()
                output = float_values * inverse_rms
                output = output * (1.0 + current.weight.float())
                return output.type_as(values)

            module.forward = MethodType(relp_norm, module)
            norm_count += 1
        elif isinstance(module, Qwen3_5MLP):

            def relp_mlp(current: Qwen3_5MLP, values: torch.Tensor) -> torch.Tensor:
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
    parser.add_argument("--model", required=True, choices=tuple(PRESETS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompts", type=int, default=25)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument("--target-layer", type=int, default=None)
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

    preset = PRESETS[args.model]
    target_layer = (
        int(preset["target_layer"]) if args.target_layer is None else args.target_layer
    )
    output = args.output or Path(
        f"local-r-lens-models/official/{preset['output_name']}/lens.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or output.with_name(
        f"fit-checkpoint-layer-{target_layer:02d}.pt"
    )
    source_layers = resolve_layers(args.layers, target_layer)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    configure_jlens_progress()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
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
    # ``gb10-load-llm`` may expose either the multimodal wrapper config or the
    # already-resolved text config, depending on the checkpoint/Transformers
    # version.  Qwen 3.5 Base currently takes the latter path.
    model_config = hf_model.config
    if not hasattr(model_config, "num_hidden_layers"):
        model_config = model_config.get_text_config()
    expected_layers = int(model_config.num_hidden_layers)
    expected_norms = 2 * expected_layers + 1
    if (norm_count, mlp_count) != (expected_norms, expected_layers):
        raise RuntimeError(
            "unexpected Qwen 3.5 module counts: "
            f"norms={norm_count}/{expected_norms}, MLPs={mlp_count}/{expected_layers}"
        )
    with torch.inference_mode():
        relp_logits = hf_model(parity_ids).logits
    if not torch.equal(ordinary_logits, relp_logits):
        maximum_difference = (ordinary_logits - relp_logits).abs().max().item()
        raise RuntimeError(
            "RelP patch changed Qwen 3.5 forward values; "
            f"max difference={maximum_difference:.3e}"
        )
    logging.info("Verified bit-identical ordinary and RelP forward logits")

    model = jlens.from_hf(hf_model, tokenizer)
    if model.n_layers != expected_layers:
        raise RuntimeError(
            f"jlens discovered {model.n_layers} layers, expected {expected_layers}"
        )
    if not 0 <= target_layer < model.n_layers:
        raise ValueError(
            f"target layer {target_layer} is invalid for {model.n_layers} layers"
        )
    prompts = load_prompts(args.prompts)
    logging.info(
        "Fitting %s R-lens: prompts=%d layers=%s target=%d dim_batch=%d",
        args.model,
        len(prompts),
        source_layers,
        target_layer,
        args.dim_batch,
    )
    fitted = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=1,
        resume=True,
    )
    fitted.jacobians[target_layer] = torch.eye(fitted.d_model)
    fitted.source_layers = sorted(fitted.jacobians)
    fitted.save(str(output), dtype=torch.float16)

    artifact = torch.load(output, map_location="cpu", weights_only=True)
    artifact["provenance"] = {
        "model_id": args.model,
        "model_revision": getattr(hf_model.config, "_commit_hash", None),
        "dataset_id": DATASET_ID,
        "target_layer": target_layer,
        "max_seq_len": args.max_seq_len,
        "n_prompts": fitted.n_prompts,
        "skip_first": args.skip_first,
        "estimator": "relp",
        "rules": {
            "ln_rule": True,
            "identity_rule": True,
            "half_rule": True,
            "include_qk_norms": False,
            "gated_norms": False,
        },
        "note": "Dense Qwen 3.5 RelP with LN, identity, and half rules.",
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(artifact["provenance"], indent=2) + "\n"
    )
    logging.info("Saved R-lens to %s", output.resolve())


if __name__ == "__main__":
    main()
