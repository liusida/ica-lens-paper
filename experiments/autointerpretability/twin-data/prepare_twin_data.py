#!/usr/bin/env python3
"""Freeze GPT-2 Small and Gemma 2 2B autointerpretability twin data.

Uses the same extractor as the paper's Qwen panel of Figure 4
(`paper/overleaf-v2.1/scripts/prepare_data.py::_autointerpretability_panel`,
copied verbatim below because that paper folder is not version-controlled):
per-layer mean `combined_score` and a deterministic 10,000-resample bootstrap
95% interval, seeded by sha256("0:<layer>:<method>").

As a check, the same extractor is run on the Qwen run and must reproduce the
`qwen_*` arrays of the paper's `figures/ica-vs-sae.npz` exactly.

Usage (from the repo root):
  uv run python experiments/autointerpretability/twin-data/prepare_twin_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PYPI = HERE.parents[3] / "icalens_pypi"
RUNS = PYPI / "experiments/autointerpretability/runs"
PAPER_NPZ = PYPI / "paper/overleaf-v2.1/figures/ica-vs-sae.npz"
MODELS = (
    ("gpt2_small", "GPT-2 Small", "gpt2-tinker"),
    ("gemma2_2b", "Gemma 2 2B", "gemma-2-2b-tinker"),
)
METHODS = ("sae", "ica")


def _autointerpretability_panel(
    run_directory: Path, *, methods: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Extract exact deterministic bootstrap summaries from a saved run."""
    run = json.loads((run_directory / "run.json").read_text())
    results = json.loads((run_directory / "results.json").read_text())
    layers = np.asarray(run["resolved"]["layers"], dtype=np.int64)
    features = results["features"]
    means = np.full((len(methods), len(layers)), np.nan, dtype=np.float64)
    lows = np.full_like(means, np.nan)
    highs = np.full_like(means, np.nan)
    counts = np.zeros_like(means, dtype=np.int64)
    for method_index, method in enumerate(methods):
        for layer_index, layer in enumerate(layers):
            scores = np.asarray(
                [
                    float(row["combined_score"])
                    for row in features
                    if int(row["layer"]) == int(layer)
                    and str(row["method"]) == method
                    and np.isfinite(row.get("combined_score", np.nan))
                ],
                dtype=np.float64,
            )
            counts[method_index, layer_index] = len(scores)
            if not len(scores):
                continue
            means[method_index, layer_index] = float(scores.mean())
            seed_text = f"0:{int(layer)}:{method}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            samples = rng.choice(scores, size=(10_000, len(scores)), replace=True).mean(axis=1)
            lows[method_index, layer_index], highs[method_index, layer_index] = np.quantile(
                samples, (0.025, 0.975)
            )
    return {"layers": layers, "means": means, "low": lows, "high": highs, "counts": counts}


def check_against_paper() -> None:
    panel = _autointerpretability_panel(RUNS / "qwen3.5-9b-tinker", methods=METHODS)
    with np.load(PAPER_NPZ, allow_pickle=False) as paper:
        pairs = {"layers": "qwen_layers", "means": "qwen_mean_scores", "low": "qwen_ci95_low",
                 "high": "qwen_ci95_high", "counts": "qwen_finite_score_counts"}
        for key, field in pairs.items():
            if not np.array_equal(panel[key], paper[field]):
                raise SystemExit(f"extractor does not reproduce paper field {field}")
    print(f"check: Qwen extraction reproduces {PAPER_NPZ.name} exactly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    check_against_paper()
    for stem, title, run_name in MODELS:
        output = HERE / f"autointerpretability-{stem}.npz"
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} exists; pass --force to refresh it")
        run_dir = RUNS / run_name
        run = json.loads((run_dir / "run.json").read_text())
        prep, evaluation = run["resolved"]["preparation_resolved"], run["evaluation"]
        panel = _autointerpretability_panel(run_dir, methods=METHODS)
        np.savez_compressed(
            output,
            panel_id=np.asarray(stem),
            panel_title=np.asarray(title),
            method_ids=np.asarray(METHODS),
            method_labels=np.asarray(("SAE", "ICA (stronger-tail sign)")),
            layers=panel["layers"],
            mean_scores=panel["means"],
            ci95_low=panel["low"],
            ci95_high=panel["high"],
            finite_score_counts=panel["counts"],
            feature_budget=np.asarray(150),
            lens=np.asarray(prep["lens"]),
            sae_name=np.asarray(prep["sae"]["name"]),
            explainer_model=np.asarray(evaluation["explainer_model"]),
            simulator_model=np.asarray(evaluation["simulator_model"]),
            prompt_hash=np.asarray(evaluation["prompt_hash"]),
            source_run=np.asarray(f"icalens_pypi/experiments/autointerpretability/runs/{run_name}"),
            evaluation_dataset=np.asarray(prep["dataset"]["repo_id"]),
            score_name=np.asarray("top-and-random automated interpretability score"),
        )
        print(output)
        for m, method in enumerate(METHODS):
            row = "  ".join(f"L{layer}: {panel['means'][m, i]:.3f} "
                            f"[{panel['low'][m, i]:.3f}, {panel['high'][m, i]:.3f}] "
                            f"n={panel['counts'][m, i]}"
                            for i, layer in enumerate(panel["layers"]))
            print(f"  {method}: {row}")


if __name__ == "__main__":
    main()
