#!/usr/bin/env python3
"""Qwen 3.5 9B Base autointerpretability: SAE vs ICA (arbitrary sign) vs ICA
(stronger-tail sign), in the style of the Pythia panel of Figure 4A.

Combines the stored stronger-tail evaluation with the arbitrary-sign recompute:
positive-skew components keep their stored scores, flipped components use the
new scores. Writes figures/qwen-arbitrary-sign.{png,pdf,txt}.

Usage (from the repo root):
  uv run python experiments/autointerpretability/qwen-arbitrary-sign/plot_arbitrary_sign.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from icalens import ICALens  # noqa: E402
from icalens.experiments.autointerpretability_figure import _bootstrap_mean  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[3] / "icalens_pypi/experiments/autointerpretability/runs"
PREP, STORED = RUNS / "qwen3.5-9b", RUNS / "qwen3.5-9b-tinker"
NEW = HERE / "runs/qwen3.5-9b-arbitrary-sign-tinker"
OUT = HERE / "figures"
LAYERS = (7, 15, 23, 31)
# Same colors/markers as the Pythia panel (plot_website_panels.py).
SERIES = (("sae", "SAE", "#B45F4D", "o"),
          ("arbitrary", "ICA (arbitrary sign)", "#93A9CC", "D"),
          ("strong", "ICA (stronger-tail sign)", "#3D5F99", "^"))
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
         "font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.0,
         "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.5,
         "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42}


def score(path: Path) -> float:
    value = json.loads(path.read_text()).get("combined_score")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else math.nan


def accepted(layer: int, method: str) -> list[int]:
    sel = json.loads((PREP / f"layer_{layer:02d}/{method}/selection.json").read_text())
    return [int(r["feature"]) for r in sel["accepted"]]


def main() -> None:
    lens = ICALens.from_pretrained("sida/icalens-qwen3.5-9b-base-pile10k")
    stats: dict[str, list[tuple[float, float, float]]] = {k: [] for k, *_ in SERIES}
    lines = ["Qwen 3.5 9B Base: mean automated-interpretability score with feature-bootstrap "
             "95% intervals (Tinker: Inkling explainer, Qwen3.8-27B simulator).",
             "layer\tmethod\tn\tmean\tbootstrap_95_low\tbootstrap_95_high"]
    for i, layer in enumerate(LAYERS):
        profile = lens._get_profile(lens._get_layer(layer))
        tail = {int(c["component"]): c["tail_direction"] for c in profile["components"]}
        d = f"layer_{layer:02d}"
        strong, arbitrary = [], []
        for f in accepted(layer, "ica"):
            s = score(STORED / d / f"ica/results/feature_{f}.json")
            strong.append(s)
            arbitrary.append(score(NEW / d / f"ica/results/feature_{f}.json")
                             if tail[f] == "negative" else s)
        sae = [score(STORED / d / f"sae/results/feature_{f}.json") for f in accepted(layer, "sae")]
        for j, (key, values) in enumerate((("sae", sae), ("arbitrary", arbitrary),
                                           ("strong", strong))):
            arr = np.asarray([v for v in values if math.isfinite(v)])
            m = _bootstrap_mean(arr, 1000 * i + j)
            stats[key].append(m)
            label = dict((k, lab) for k, lab, *_ in SERIES)[key]
            lines.append(f"{layer}\t{label}\t{len(arr)}\t{m[0]:.6f}\t{m[1]:.6f}\t{m[2]:.6f}")

    OUT.mkdir(exist_ok=True)
    x = np.asarray(LAYERS)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.75, 2.55))
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.24, top=0.72)
        ax.set_title("Qwen 3.5 9B Base", loc="left", fontweight="bold", pad=4)
        for key, label, color, marker in SERIES:
            m = np.asarray(stats[key])
            ax.errorbar(x, m[:, 0], yerr=(m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]), label=label,
                        color=color, marker=marker, linewidth=1.5, markersize=3.5,
                        elinewidth=0.8, capsize=2.0)
        ax.set_xticks(x)
        ax.set_ylim(0.0, 0.8)
        ax.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Interpretability score")
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.56, 0.98),
                   columnspacing=1.1, handlelength=1.8)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"qwen-arbitrary-sign.{ext}", dpi=300)
        plt.close(fig)
    lines.append("Arbitrary sign scores the raw fitted sign's positive tail; components the "
                 "stronger-tail rule left unchanged reuse their stored scores.")
    (OUT / "qwen-arbitrary-sign.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "qwen-arbitrary-sign.png")


if __name__ == "__main__":
    main()
