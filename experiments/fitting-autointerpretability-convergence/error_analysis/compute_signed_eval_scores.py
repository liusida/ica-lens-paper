#!/usr/bin/env python3
"""Recompute signed iteration-50 component scores on all evaluation tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM

from icalens._capture import transformer_blocks
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare import MODEL_ID, MODEL_REVISION, aligned_directions, read_fragments

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/fixed-panel"
FRAGMENTS = ROOT.parent / "autointerpretability/runs/qwen3.5-9b/fragments.jsonl"
OUTPUT = ROOT / "results/signed-eval-token-scores-100k.npy"
PROGRESS = ROOT / "results/signed-eval-token-scores-100k-progress.json"
CASES = ((7, 338), (7, 2035), (15, 1503))
ITERATION = 50
CHECKPOINT_INTERVAL = 256
TARGET_TOKENS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_fragments = read_fragments(FRAGMENTS)
    sample_count = int(np.ceil(TARGET_TOKENS / 64))
    sample_indices = np.floor(
        np.arange(sample_count, dtype=np.float64) * len(all_fragments) / sample_count
    ).astype(np.int64)
    fragments = [all_fragments[int(index)] for index in sample_indices]
    cohort = json.loads((RUN / "cohort.json").read_text(encoding="utf-8"))
    tails = json.loads(
        Path(
            "/media/liusida/Expansion/research/ICA-data/"
            "fitting-autointerpretability-convergence/fixed-panel/tails.json"
        ).read_text(encoding="utf-8")
    )
    directions: dict[int, torch.Tensor] = {}
    centers: dict[int, torch.Tensor] = {}
    for layer in sorted({layer for layer, _ in CASES}):
        row_ids = list(map(int, cohort["layers"][str(layer)]["row_ids"]))
        aligned, center = aligned_directions(RUN / "trajectory", layer, row_ids, args.device)
        signs = torch.tensor(tails["layers"][str(layer)]["tail_sign"], device=args.device)
        requested = [component for candidate_layer, component in CASES if candidate_layer == layer]
        positions = torch.tensor(
            [row_ids.index(component) for component in requested], device=args.device
        )
        directions[layer] = (aligned[ITERATION] * signs[:, None]).index_select(0, positions)
        centers[layer] = center

    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    resolved = {
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "fragments": str(FRAGMENTS.resolve()),
        "iteration": ITERATION,
        "cases": [list(case) for case in CASES],
        "shape": [len(fragments), 64, len(CASES)],
        "sampling": "systematic whole-fragment sample from 50,000 OpenWebText fragments",
        "sample_indices": sample_indices.tolist(),
    }
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.is_file() else None
    if OUTPUT.is_file():
        print(f"PASS signed evaluation scores already complete: {OUTPUT}")
        return
    if progress is not None and progress.get("resolved") != resolved:
        raise ValueError(f"incompatible progress file: {PROGRESS}")
    completed = int(progress["completed_fragments"]) if progress else 0
    if completed:
        scores = np.load(temporary, mmap_mode="r+")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        scores = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=tuple(resolved["shape"])
        )

    source = source_provenance()
    with ExperimentDisplay(
        output=ROOT / "results/logs/signed-eval-token-scores",
        title="ICA Lens · signed evaluation-token scores",
        completed=completed,
        total=len(fragments),
        source_dirty=source.get("dirty"),
        unit_label="fragments",
        detail_filename="run-detail.log",
    ) as display:
        warn_if_dirty(source)
        display.phase("Loading Qwen", model=MODEL_ID)
        model = cast(
            torch.nn.Module,
            load_model_to_cuda(
                AutoModelForCausalLM,
                MODEL_ID,
                revision=MODEL_REVISION,
                device=args.device,
                dtype=torch.bfloat16,
                touch="auto",
                low_cpu_mem_usage=True,
            ),
        )
        model.eval()
        captured: dict[int, torch.Tensor] = {}
        handles: list[Any] = []
        blocks = transformer_blocks(model)
        for layer in directions:

            def hook(_module: Any, _inputs: Any, value: Any, layer: int = layer) -> None:
                captured[layer] = (value[0] if isinstance(value, tuple) else value).detach()

            handles.append(blocks[layer].register_forward_hook(hook))
        try:
            checkpoint_at = completed
            for start in range(completed, len(fragments), args.batch_size):
                stop = min(start + args.batch_size, len(fragments))
                input_ids = torch.tensor(
                    [row["token_ids"] for row in fragments[start:stop]],
                    dtype=torch.long,
                    device=args.device,
                )
                captured.clear()
                with torch.inference_mode():
                    model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                    )
                column = 0
                for layer in sorted(directions):
                    hidden = captured[layer].to(torch.float32) - centers[layer]
                    layer_scores = hidden @ directions[layer].T
                    width = layer_scores.shape[-1]
                    scores[start:stop, :, column : column + width] = layer_scores.cpu().numpy()
                    column += width
                display.advance(stop - start, refresh=True)
                display.phase("Scoring evaluation fragments", fragments=f"{stop}/{len(fragments)}")
                if stop - checkpoint_at >= CHECKPOINT_INTERVAL or stop == len(fragments):
                    scores.flush()
                    checkpoint_tmp = PROGRESS.with_suffix(PROGRESS.suffix + ".tmp")
                    checkpoint_tmp.write_text(
                        json.dumps({"resolved": resolved, "completed_fragments": stop}, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                    os.replace(checkpoint_tmp, PROGRESS)
                    checkpoint_at = stop
                    log(f"Checkpointed {stop}/{len(fragments)} evaluation fragments.")
        finally:
            for handle in handles:
                handle.remove()
        os.replace(temporary, OUTPUT)
        PROGRESS.unlink(missing_ok=True)
        display.set_outcome("complete")
        log(f"Saved signed evaluation-token scores: {OUTPUT}")


if __name__ == "__main__":
    main()
