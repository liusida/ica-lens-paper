#!/usr/bin/env python3
"""Inter-annotator agreement on manual confidence (0-5) for the random 150 sample.

Reads annotations.json (annotator 1) and annotations-annotator2.json (annotator 2)
and writes agreement/agreement.json and agreement/agreement.md. Labels are free
text in different styles, so only confidence agreement is computed here; label
agreement needs a by-meaning judgment.

Usage (from the repo root):
  uv run python experiments/manual-annotation-random-150/agreement.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "agreement"
SCALE = np.arange(6)
NAMES = {"gpt2": "GPT-2 Small", "gemma-2-2b": "Gemma 2 2B", "qwen3.5-9b": "Qwen 3.5 9B Base"}


def weighted_kappa(a: np.ndarray, b: np.ndarray, power: int) -> float:
    """Cohen's kappa on the 0-5 scale with |i-j|**power disagreement weights."""
    observed = np.zeros((len(SCALE), len(SCALE)))
    np.add.at(observed, (a, b), 1)
    observed /= observed.sum()
    expected = np.outer(observed.sum(1), observed.sum(0))
    weights = np.abs(SCALE[:, None] - SCALE[None, :]) ** power
    return float(1 - (weights * observed).sum() / (weights * expected).sum())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks for ties."""
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
        for value in np.unique(x):
            ranks[x == value] = ranks[x == value].mean()
        return ranks
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def binary_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def summarize(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(a)),
        "exact": float(np.mean(a == b)),
        "within_1": float(np.mean(np.abs(a - b) <= 1)),
        "kappa_quadratic": weighted_kappa(a, b, 2),
        "kappa_linear": weighted_kappa(a, b, 1),
        "spearman": spearman(a, b),
        "mean_annotator1": float(a.mean()),
        "mean_annotator2": float(b.mean()),
        "high_annotator1": float(np.mean(a >= 4)),
        "high_annotator2": float(np.mean(b >= 4)),
        "high_agreement": float(np.mean((a >= 4) == (b >= 4))),
        "high_kappa": binary_kappa(a >= 4, b >= 4),
    }


def load(name: str) -> dict[tuple[str, int, int], int]:
    data = json.loads((HERE / name).read_text(encoding="utf-8"))
    return {(m, e["layer"], e["component"]): int(e["confidence"]) for m, v in data.items() for e in v}


def main() -> None:
    one, two = load("annotations.json"), load("annotations-annotator2.json")
    if one.keys() != two.keys():
        raise SystemExit("annotators cover different components")
    keys = sorted(one)
    groups = {"all": keys, **{m: [k for k in keys if k[0] == m] for m in NAMES}}
    result = {g: summarize(np.array([one[k] for k in ks]), np.array([two[k] for k in ks]))
              for g, ks in groups.items()}

    OUT.mkdir(exist_ok=True)
    (OUT / "agreement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["| Group | n | Exact | Within 1 | Weighted κ (quad.) | Weighted κ (lin.) | Spearman ρ | "
             "Mean A1 / A2 | ≥4 agreement (κ) |", "|---|---|---|---|---|---|---|---|---|"]
    for g, s in result.items():
        lines.append(f"| {NAMES.get(g, 'All')} | {s['n']} | {s['exact']:.0%} | {s['within_1']:.0%} | "
                     f"{s['kappa_quadratic']:.2f} | {s['kappa_linear']:.2f} | {s['spearman']:.2f} | "
                     f"{s['mean_annotator1']:.2f} / {s['mean_annotator2']:.2f} | "
                     f"{s['high_agreement']:.0%} ({s['high_kappa']:.2f}) |")
    (OUT / "agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
