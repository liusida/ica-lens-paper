#!/usr/bin/env python3
"""Measure two Qwen ICA components on an observed sentence and a counterfactual."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens.analysis import capture
from icalens.experiments._run import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LENS_PATH = ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k"
LAYER = 16
COMPONENTS = (779, 2186)
OBSERVED = (
    "Maya stopped at the bank before the trip, waiting in line to deposit a check "
    "and withdraw enough cash for the weekend."
)
RELATED = (
    "Maya sat on a bench outside the bank, waiting to repay a loan and collect "
    "some cash."
)
COUNTERFACTUAL = (
    "Maya wandered outdoors before traveling, then photographed birds, gathered "
    "flowers, and hurried home before the weekend."
)
LABELS = {
    779: "Payments / money owed",
    2186: "Sitting / waiting",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: str,
) -> dict[str, np.ndarray]:
    captured = capture(
        lens,
        text,
        layer=LAYER,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    scores = lens.transform(captured.activations.float(), layer=LAYER)
    scores = torch.as_tensor(scores).detach().cpu().float().numpy()
    return {
        "token_ids": captured.token_ids.cpu().numpy().astype(np.int64),
        "tokens": np.asarray(captured.tokens),
        "token_texts": np.asarray(captured.token_texts),
        "token_labels": np.asarray(captured.token_labels),
        "scores": scores[:, COMPONENTS].T.astype(np.float32),
    }


def profile_summary(lens: ICALens, component: int) -> dict[str, Any]:
    profile = lens.component_profile(layer=LAYER, component=component)
    sign = profile.get("tail_direction") or profile.get("dominant_sign")
    occurrences = profile.get("examples", {}).get(sign, {}).get("occurrences", [])
    return {
        "component": component,
        "label": LABELS[component],
        "tail_direction": sign,
        "evidence": [
            {
                "score": occurrence.get("score"),
                "position": occurrence.get("position"),
                "context": occurrence.get("context"),
                "text": occurrence.get("text"),
            }
            for occurrence in occurrences[:16]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=HERE / "results-counterfactual")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("this GB10 measurement requires CUDA")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lens = ICALens.from_pretrained(LENS_PATH)
    artifact = LENS_PATH / "artifacts/resid_post/layer_16.safetensors"
    profiles = [profile_summary(lens, component) for component in COMPONENTS]

    print("START Qwen L16 counterfactual measurement", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        lens.model_id,
        revision=lens.model_revision,
        use_fast=True,
    )
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        lens.model_id,
        revision=lens.model_revision,
        device=args.device,
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    try:
        measured = {
            "observed": measure(lens, model, tokenizer, OBSERVED, args.device),
            "related": measure(lens, model, tokenizer, RELATED, args.device),
            "counterfactual": measure(
                lens, model, tokenizer, COUNTERFACTUAL, args.device
            ),
        }
    finally:
        del model
        torch.cuda.empty_cache()

    arrays: dict[str, np.ndarray] = {
        "component_ids": np.asarray(COMPONENTS, dtype=np.int64)
    }
    for name, result in measured.items():
        for key, value in result.items():
            arrays[f"{name}_{key}"] = value
    np.savez_compressed(output / "qwen9b-layer16.npz", **arrays)
    atomic_write_json(
        output / "summary.json",
        {
            "format": "icalens.activation_pattern.counterfactual",
            "schema_version": 1,
            "model": "Qwen 3.5 9B Base",
            "model_id": lens.model_id,
            "model_revision": lens.model_revision,
            "lens": str(LENS_PATH.resolve()),
            "lens_artifact_sha256": sha256(artifact),
            "layer": LAYER,
            "components": profiles,
            "quantity": "raw signed ICA score",
            "sentences": {
                "observed": OBSERVED,
                "related": RELATED,
                "counterfactual": COUNTERFACTUAL,
            },
            "data": "qwen9b-layer16.npz",
        },
    )
    print(f"PASS Qwen L16 counterfactual measurement: {output}", flush=True)


if __name__ == "__main__":
    main()
