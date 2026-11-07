#!/usr/bin/env python3
"""Build an arbitrary-sign (+1) ICA preparation for the Qwen components that the
skewness rule flipped, reusing the original fragments and protocol unchanged.

For each layer:
- take the original 150-feature ICA cohort from `selection.json`;
- keep the components whose profiled `tail_direction` is "negative" (flipped);
- add two positive-skew controls, whose +1 activations must reproduce the stored
  ones (a sanity check on the re-encoding);
- encode the same 50k fragments with sign +1 and `clamp_min(0)`, exactly as
  `prepare` does but without the orientation;
- choose records with the unchanged `select_record_indices(seed=seed + feature)`.

The output is a preparation folder readable by
`icalens experiment autointerpretability evaluate --methods ica`.

Usage (from the repo root):
  uv run python experiments/autointerpretability/qwen-arbitrary-sign/prepare_arbitrary_sign.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from icalens import ICALens
from icalens._capture import transformer_blocks
from icalens.experiments.autointerpretability import FRAGMENT_LENGTH, _load_model
from icalens.experiments.autointerpretability_protocol import select_record_indices

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "icalens_pypi/experiments/autointerpretability/runs/qwen3.5-9b"
N_CONTROLS = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=HERE / "runs/qwen3.5-9b-arbitrary-sign")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()

    run = json.loads((source / "run.json").read_text())
    resolved = run["resolved"]
    lens = ICALens.from_pretrained(resolved["lens"])
    layers = [int(x) for x in resolved["layers"]]
    seed = int(resolved["seed"])
    fragments = [json.loads(line) for line in (source / "fragments.jsonl").open()]
    assert len(fragments) == resolved["n_fragments"]

    # Pick features per layer: flipped components plus positive controls.
    plan: dict[int, dict] = {}
    for layer in layers:
        profile = lens._get_profile(lens._get_layer(layer))
        tail = {int(c["component"]): c["tail_direction"] for c in profile["components"]}
        cohort = json.loads((source / f"layer_{layer:02d}/ica/selection.json").read_text())
        accepted = cohort["accepted"]
        flipped = [r["feature"] for r in accepted if tail[r["feature"]] == "negative"]
        controls = [r for r in accepted if tail[r["feature"]] == "positive"][:N_CONTROLS]
        plan[layer] = {"flipped": flipped, "controls": controls,
                       "ids": flipped + [r["feature"] for r in controls]}
        print(f"layer {layer}: {len(flipped)} flipped + {len(controls)} controls")

    output.mkdir(parents=True, exist_ok=True)
    stores = {}
    for layer in layers:
        d = output / f"layer_{layer:02d}/ica"
        d.mkdir(parents=True, exist_ok=True)
        stores[layer] = np.lib.format.open_memmap(
            d / "candidate_activations.npy.tmp", mode="w+", dtype=np.float16,
            shape=(len(fragments), FRAGMENT_LENGTH, len(plan[layer]["ids"])))

    model = _load_model(lens, device=args.device)
    blocks = transformer_blocks(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_m, _i, value):
            captured[layer] = (value[0] if isinstance(value, tuple) else value).detach()
        return hook

    handles = [blocks[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    ids_t = {layer: torch.as_tensor(plan[layer]["ids"], device=args.device) for layer in layers}
    try:
        for start in tqdm(range(0, len(fragments), args.batch_size), desc="Encoding"):
            batch = fragments[start:start + args.batch_size]
            input_ids = torch.tensor([r["token_ids"] for r in batch], dtype=torch.long,
                                     device=args.device)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                      use_cache=False)
            for layer in layers:
                hidden = captured[layer].to(dtype=torch.float32)
                scores = lens.transform(hidden, layer=layer)  # raw fitted sign (+1)
                values = scores.clamp_min(0).index_select(-1, ids_t[layer])
                stores[layer][start:start + len(batch)] = values.cpu().numpy()
    finally:
        for h in handles:
            h.remove()

    summary = {}
    for layer in layers:
        d = output / f"layer_{layer:02d}/ica"
        stores[layer].flush()
        (d / "candidate_activations.npy.tmp").replace(d / "candidate_activations.npy")
        values = np.load(d / "candidate_activations.npy", mmap_mode="r")
        n_flip = len(plan[layer]["flipped"])

        # Sanity check: +1 controls must reproduce the stored (positive-oriented) activations.
        stored = np.load(source / f"layer_{layer:02d}/ica/candidate_activations.npy", mmap_mode="r")
        max_err = 0.0
        for k, rec in enumerate(plan[layer]["controls"]):
            a = np.asarray(values[:2000, :, n_flip + k], dtype=np.float32)
            b = np.asarray(stored[:2000, :, int(rec["candidate_position"])], dtype=np.float32)
            max_err = max(max_err, float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6)))

        accepted, undefined = [], []
        for position, feature in enumerate(plan[layer]["flipped"]):
            try:
                train, top, rnd = select_record_indices(
                    np.asarray(values[:, :, position], dtype=np.float32), seed=seed + feature)
            except ValueError as error:  # reported as undefined, never replaced
                undefined.append({"feature": feature, "reason": str(error)})
                continue
            accepted.append({"feature": feature, "candidate_position": position,
                             "train_top": train, "valid_top": top, "valid_random": rnd})
        (d / "selection.json").write_text(json.dumps(
            {"candidate_ids": plan[layer]["ids"], "accepted": accepted,
             "rejected": undefined, "orientation": "arbitrary (+1, raw fitted sign)",
             "controls": [r["feature"] for r in plan[layer]["controls"]]}, indent=1))
        summary[layer] = {"flipped": n_flip, "accepted": len(accepted),
                          "undefined": undefined, "control_max_rel_err": max_err}
        print(f"layer {layer}: accepted {len(accepted)}/{n_flip}, "
              f"control max rel err {max_err:.2e}")

    shutil.copy(source / "fragment_sampling.json", output / "fragment_sampling.json")
    frag = output / "fragments.jsonl"
    if not frag.exists():
        frag.symlink_to(source / "fragments.jsonl")
    out_run = dict(run)
    out_run["resolved"] = dict(resolved, n_features=None)
    out_run["arbitrary_sign"] = {"source_preparation": str(source), "summary": summary}
    (output / "run.json").write_text(json.dumps(out_run, indent=2))


if __name__ == "__main__":
    main()
