#!/usr/bin/env python3
"""Evaluator evidence plus novel edge cases for Qwen iteration-50 L15/C1503."""

from __future__ import annotations

import json
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
LAYER, COMPONENT, ITERATION = 15, 1503, 50

# Each tuple is (group, focus text, prompt). The focus is matched after stripping
# surrounding punctuation from each independently decoded token.
CASES = [
    ("observed initial", "Human", "Human rights matter in every country."),
    ("observed initial", "What", "What happened at the meeting?"),
    ("observed initial", "How", "How does this mechanism work?"),
    ("observed initial", "Question", "Question 2: Did you read the report?"),
    ("observed initial", "In", "In the box, we found an old photograph."),
    ("observed initial", "#", "# A short heading about science"),
    ("internal/case", "human", "They discussed human rights at the meeting."),
    ("internal/case", "Human", "The label Human appears in the document."),
    ("internal/case", "What", "I stopped and asked: What happened?"),
    ("internal/case", "how", "She explained how the mechanism works."),
    ("internal/case", "How", "He asked: How does this mechanism work?"),
    ("internal/case", "question", "Nobody answered the question clearly."),
    ("internal/case", "Question", "The form begins with Question 2 about age."),
    ("internal/case", "In", "She replied: In the box, we found a photograph."),
    ("morphology", "Humans", "Humans are a curious and exploratory species."),
    ("morphology", "Humanity", "Humanity depends on cooperation."),
    ("other opener", "Why", "Why did the experiment fail?"),
    ("other opener", "Who", "Who approved the proposal?"),
    ("other opener", "When", "When will the results be available?"),
    ("other opener", "Where", "Where was the artifact stored?"),
]


def normalized_token(text: str) -> str:
    return text.strip().strip("\"'#:,.?!").lower()


def main() -> None:
    shared = load_file(TRAJECTORY / "shared/layer-15.safetensors")
    checkpoint = load_file(TRAJECTORY / "checkpoints/layer-15/iter-050.safetensors")
    center = shared["center"].to(device="cuda", dtype=torch.float32)
    whitening = shared["whitening"].to(device="cuda", dtype=torch.float32)
    reading = checkpoint["unmixing"][COMPONENT].to(device="cuda", dtype=torch.float32) @ whitening

    cohort = json.loads((RUN / "cohort.json").read_text(encoding="utf-8"))
    row_ids = list(map(int, cohort["layers"][str(LAYER)]["row_ids"]))
    cohort_position = row_ids.index(COMPONENT)
    tails = json.loads(TAILS.read_text(encoding="utf-8"))
    tail_sign = int(tails["layers"][str(LAYER)]["tail_sign"][cohort_position])
    reading *= tail_sign

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    prepared = RUN / f"prepared/iter-{ITERATION:03d}"
    selection = json.loads(
        (prepared / f"layer_{LAYER:02d}/ica/selection.json").read_text(encoding="utf-8")
    )
    selected = next(row for row in selection["accepted"] if int(row["feature"]) == COMPONENT)
    fragment_ids = {
        int(fragment)
        for key in ("train_top", "valid_top", "valid_random")
        for fragment in selected[key]
    }
    fragments = {}
    with (prepared / "fragments.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["fragment"]) in fragment_ids:
                fragments[int(row["fragment"])] = row

    specifications = []
    for key, group in (
        ("train_top", "explanation example"),
        ("valid_top", "held-out top"),
        ("valid_random", "held-out random"),
    ):
        for fragment in selected[key]:
            row = fragments[int(fragment)]
            specifications.append(
                {
                    "group": group,
                    "focus": None,
                    "prompt": tokenizer.decode(row["token_ids"]),
                    "input_ids": list(map(int, row["token_ids"])),
                    "fragment": int(fragment),
                }
            )
    specifications.extend(
        {"group": group, "focus": focus, "prompt": prompt, "input_ids": None, "fragment": None}
        for group, focus, prompt in CASES
    )

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
        for case, specification in enumerate(specifications, start=1):
            group = specification["group"]
            focus = specification["focus"]
            prompt = specification["prompt"]
            if specification["input_ids"] is None:
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                ids = torch.tensor([specification["input_ids"]], dtype=torch.long)
                encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            input_ids = encoded["input_ids"].to("cuda")
            captured.clear()
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=encoded["attention_mask"].to("cuda"),
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
            maximum = max(token_rows, key=lambda row: row["activation"])
            if focus is None:
                focus = maximum["decoded"].strip() or maximum["token"]
                focus_rows = [maximum]
            else:
                target = normalized_token(focus)
                focus_rows = [
                    row for row in token_rows if normalized_token(row["decoded"]) == target
                ]
            records.append(
                {
                    "case": case,
                    "group": group,
                    "focus": focus,
                    "prompt": prompt,
                    "fragment": specification["fragment"],
                    "focus_tokens": focus_rows,
                    "maximum": maximum,
                    "tokens": token_rows,
                }
            )
    finally:
        handle.remove()

    output = ROOT / "results/c1503-edge-case-results.json"
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

    print("case  group             focus       pos   focus score   maximum token/score")
    print("-" * 88)
    for record in records:
        focus_rows = record["focus_tokens"]
        focus_summary = (
            ", ".join(f"{row['position']}:{row['activation']:.3f}" for row in focus_rows)
            or "not-single-token"
        )
        maximum = record["maximum"]
        print(
            f"{record['case']:>4}  {record['group']:<16} "
            f"{record['focus']:<10} {focus_summary:<17} "
            f"{maximum['decoded']!r}@{maximum['position']}={maximum['activation']:.3f}"
        )
        print(f"      {record['prompt']}")
    print(f"\nFull token-level results: {output}")


if __name__ == "__main__":
    main()
