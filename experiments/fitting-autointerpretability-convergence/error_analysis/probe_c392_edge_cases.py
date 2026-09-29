#!/usr/bin/env python3
"""One-time token-level edge-case probe for Qwen iteration-0 L31/C392."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens._capture import transformer_blocks

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/fixed-panel"
TRAJECTORY = RUN / "trajectory"
TAILS = Path(
    "/media/liusida/Expansion/research/ICA-data/"
    "fitting-autointerpretability-convergence/fixed-panel/tails.json"
)
MODEL_ID = "Qwen/Qwen3.5-9B-Base"
MODEL_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
LAYER = 31
COMPONENT = 392
ITERATION = 0

PROMPTS = [
    "In the box, we found an old photograph.",
    "In chemistry, indium has the symbol In.",
    "in the box, we found an old photograph.",
    "The photograph was in the box.",
    "She replied: In the box, we found an old photograph.",
    "After waiting for several minutes, she replied: In the box, we found an old photograph.",
    '"In the box," she replied, "we found an old photograph."',
    "Inside the box, we found an old photograph.",
    "IN THE BOX, WE FOUND AN OLD PHOTOGRAPH.",
    "In the first box we found nothing, but she later said: In the second box was a photograph.",
]


def main() -> None:
    shared = load_file(TRAJECTORY / "shared/layer-31.safetensors")
    checkpoint = load_file(TRAJECTORY / "checkpoints/layer-31/iter-000.safetensors")
    center = shared["center"].to(device="cuda", dtype=torch.float32)
    whitening = shared["whitening"].to(device="cuda", dtype=torch.float32)
    unmixing = checkpoint["unmixing"][COMPONENT].to(device="cuda", dtype=torch.float32)
    reading = unmixing @ whitening

    cohort = json.loads((RUN / "cohort.json").read_text(encoding="utf-8"))
    row_ids = list(map(int, cohort["layers"][str(LAYER)]["row_ids"]))
    cohort_position = row_ids.index(COMPONENT)
    tails = json.loads(TAILS.read_text(encoding="utf-8"))
    tail_sign = int(tails["layers"][str(LAYER)]["tail_sign"][cohort_position])
    reading *= tail_sign

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = cast(
        torch.nn.Module,
        load_model_to_cuda(
            AutoModelForCausalLM,
            MODEL_ID,
            revision=MODEL_REVISION,
            device="cuda",
            dtype=torch.bfloat16,
            touch="auto",
            low_cpu_mem_usage=True,
        ),
    )
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _args: Any, output: Any) -> None:
        captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = transformer_blocks(model)[LAYER].register_forward_hook(hook)
    records = []
    try:
        for case, prompt in enumerate(PROMPTS, start=1):
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded["attention_mask"].to("cuda")
            captured.clear()
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            hidden = captured["hidden"][0].to(torch.float32) - center
            signed = (hidden @ reading).cpu()
            activation = signed.clamp_min(0)
            token_ids = input_ids[0].cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            decoded = [tokenizer.decode([token_id]) for token_id in token_ids]
            token_rows = [
                {
                    "position": position,
                    "token_id": token_id,
                    "token": token,
                    "decoded": text,
                    "signed_score": float(raw),
                    "activation": float(active),
                }
                for position, (token_id, token, text, raw, active) in enumerate(
                    zip(
                        token_ids,
                        tokens,
                        decoded,
                        signed.tolist(),
                        activation.tolist(),
                        strict=True,
                    )
                )
            ]
            records.append({"case": case, "prompt": prompt, "tokens": token_rows})
    finally:
        handle.remove()

    output = ROOT / "results/c392-edge-case-results.json"
    output.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "layer": LAYER,
                "component": COMPONENT,
                "iteration": ITERATION,
                "tail_sign": tail_sign,
                "score": "(resid_post - center) @ oriented_reading_direction",
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "case  pos  token_id  token              signed   activation  role",
        flush=True,
    )
    print("-" * 78)
    for record in records:
        rows = record["tokens"]
        maximum = max(rows, key=lambda row: row["activation"])
        related = [
            row
            for row in rows
            if re.match(r'^["\']?in', row["decoded"].strip().lower())
            or row["position"] == maximum["position"]
        ]
        for row in related:
            roles = []
            if re.match(r'^["\']?in', row["decoded"].strip().lower()):
                roles.append("in-related")
            if row["position"] == maximum["position"]:
                roles.append("maximum")
            print(
                f"{record['case']:>4} {row['position']:>4} {row['token_id']:>9} "
                f"{row['decoded']!r:<18} {row['signed_score']:>8.3f} "
                f"{row['activation']:>11.3f}  {','.join(roles)}"
            )
        print(f"     prompt: {record['prompt']}")
    print(f"\nFull token-level results: {output}")


if __name__ == "__main__":
    main()
