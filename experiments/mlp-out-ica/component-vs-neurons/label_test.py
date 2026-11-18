#!/usr/bin/env python3
"""Test proposed labels for L0 N996 and L1 N586 on hand-written sentences.

Exploration script (not core code). Labels under test:
  L0 N996  singular entity/agent noun (person, animal, role, device)
  L1 N586  agentive-suffix noun (-er/-or/-ist/-ant/-ator), singular or plural

Each case has several sentences with the target word in [brackets]; both neurons are
read (post-GELU, input to c_proj) at the target's LAST token. "Fires" uses the same
q99 thresholds as pair_examples_L0N996_L1N586_q99.json. Writes ``label_test.json``
and its twin ``.html`` (render_report.py).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_report import render_any  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "L1-C79" / "neuron-activations"
UP, DOWN = (0, 996), (1, 586)

# (title, what it tests, prediction N996, prediction N586, sentences)
# predictions: "high" (should fire), "low" (should not), "?" (open question, not scored)
CASES = [
    ("1 singular person, no suffix", "Plain singular person nouns", "high", "low", [
        "The [patient] waited in the hallway for an hour.",
        "A [woman] knocked on the door last night.",
        "Every [citizen] has the right to vote.",
        "The [doctor] said the results were normal.",
    ]),
    ("2 plural -er agents", "Plurality should switch N996 off, keep N586 on", "low", "high", [
        "The [farmers] blocked the highway with tractors.",
        "Local [workers] went on strike yesterday.",
        "The [teachers] handed out the new books.",
        "Thousands of [voters] stayed home.",
    ]),
    ("3 singular -er/-or agents", "Overlap case: both should fire", "high", "high", [
        "The [teacher] handed back the exams.",
        "Our [manager] approved the budget.",
        "The [actor] forgot his lines.",
        "A local [farmer] found the bones.",
    ]),
    ("4 -er instruments", "Device, not person", "?", "high", [
        "Plug the [charger] into the wall.",
        "The [printer] ran out of ink again.",
        "She bought a new [blender] for the kitchen.",
        "The [computer] crashed during the update.",
    ]),
    ("5 -er ending, no agent", "Surface -er without agent meaning", "low", "low", [
        "She stood in the [corner] of the room.",
        "He drank a glass of [water] after the run.",
        "We will meet again in [December] next year.",
        "The cat slept under the [cover] all day.",
    ]),
    ("6 comparatives", "Adjective -er", "low", "low", [
        "He bought a [bigger] car than last year.",
        "The new model is much [faster] than the old one.",
        "It was [colder] than we expected.",
        "Prices are [higher] this month.",
    ]),
    ("7 plural people, no suffix", "Group of people without suffix", "low", "?", [
        "The [children] ran across the field.",
        "Many [people] attended the meeting.",
        "The [women] signed the petition.",
        "Several [students] complained about the exam.",
    ]),
    ("8 singular animals", "Non-human entity, no suffix", "high", "low", [
        "A [fox] crossed the road at night.",
        "The [dog] barked at the mailman.",
        "A [horse] stood alone in the field.",
        "The [snake] hid under a rock.",
    ]),
    ("9 abstract singular nouns", "Singular noun that is not an entity", "low", "low", [
        "The [idea] seemed simple at first.",
        "Her [decision] surprised everyone.",
        "The [problem] was never solved.",
        "There is no [evidence] for that claim.",
    ]),
    ("10 other agent suffixes", "-ist / -ian / -ant", "high", "high", [
        "The [pianist] bowed to the audience.",
        "The [librarian] smiled and nodded.",
        "The [assistant] booked the flights.",
        "A [scientist] explained the results.",
    ]),
]


def main() -> None:
    pair = json.loads((FOLDER / "pair_examples_L0N996_L1N586_q99.json").read_text())
    thresholds = {"up": pair["up"]["threshold"], "down": pair["down"]["threshold"]}
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to("cuda").eval()
    captured: dict[str, torch.Tensor] = {}
    handles = [
        model.transformer.h[layer].mlp.c_proj.register_forward_pre_hook(
            lambda _m, inputs, key=key, n=n: captured.__setitem__(key, inputs[0][0, :, n].float())
        )
        for key, (layer, n) in (("up", UP), ("down", DOWN))
    ]
    cases = []
    with torch.inference_mode():
        for title, tests, pred_up, pred_down, sentences in CASES:
            examples = []
            for marked in sentences:
                match = re.search(r"\[(.+?)\]", marked)
                text = marked.replace("[", "").replace("]", "")
                start, end = match.start(), match.end() - 2
                encoded = tokenizer(text, return_offsets_mapping=True)
                ids = [tokenizer.eos_token_id] + encoded["input_ids"]
                offsets = [(0, 0)] + encoded["offset_mapping"]
                pieces = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]
                target = pieces[-1]
                model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
                window = [
                    {
                        "text": tokenizer.decode([ids[i]]),
                        "up": float(captured["up"][i]),
                        "down": float(captured["down"][i]),
                        "target": i == target,
                    }
                    for i in range(1, len(ids))
                ]
                examples.append({
                    "sentence": text,
                    "word": match.group(1),
                    "pieces": [tokenizer.decode([ids[i]]) for i in pieces],
                    "up": float(captured["up"][target]),
                    "down": float(captured["down"][target]),
                    "window": window,
                })
            cases.append({
                "title": title, "tests": tests,
                "prediction": {"up": pred_up, "down": pred_down}, "examples": examples,
            })
    for handle in handles:
        handle.remove()

    def verdict(pred: str, values: list[float], threshold: float) -> dict:
        fired = sum(v >= threshold for v in values)
        if pred == "?":
            status = "open"
        elif pred == "high":
            status = "pass" if fired == len(values) else ("partial" if fired else "fail")
        else:
            status = "pass" if fired == 0 else ("partial" if fired < len(values) else "fail")
        return {"fired": fired, "n": len(values), "status": status}

    for case in cases:
        case["verdict"] = {
            key: verdict(case["prediction"][key], [e[key] for e in case["examples"]], thresholds[key])
            for key in ("up", "down")
        }
    data = {
        "kind": "neuron_label_test",
        "up": {**pair["up"], "label": "singular entity/agent noun"},
        "down": {**pair["down"], "label": "agentive-suffix noun (-er/-or/-ist/-ant)"},
        "color_scale": pair["color_scale"],
        "cases": cases,
    }
    path = FOLDER / "label_test.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    path.with_suffix(".html").write_text(render_any(data))
    print(f"Saved {path} and its .html")
    for case in cases:
        v = case["verdict"]
        print(f"{case['title']:<32} N996 {case['prediction']['up']:>4} {v['up']['fired']}/{v['up']['n']} "
              f"{v['up']['status']:<8} N586 {case['prediction']['down']:>4} "
              f"{v['down']['fired']}/{v['down']['n']} {v['down']['status']}")
        for e in case["examples"]:
            print(f"    {e['word']:<12} {'|'.join(e['pieces']):<18} {e['up']:+.2f} {e['down']:+.2f}")


if __name__ == "__main__":
    main()
