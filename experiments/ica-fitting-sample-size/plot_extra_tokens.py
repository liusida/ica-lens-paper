#!/usr/bin/env python3
"""Compare fitting and additional-document evaluation curves for all budgets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/icalens-extra-token-mpl")
import matplotlib.pyplot as plt
import numpy as np

from plot import COLORS, LABELS, MARKERS, ORDER

HERE = Path(__file__).resolve().parent


def load():
    data, metadata = {}, {}
    for budget in ORDER:
        root = HERE / "work/extra-evaluation" / budget
        complete = json.loads((root / "complete.json").read_text())
        assert complete["evaluation_tokens"] == 1_000_000 and complete["states"] == 51
        metadata[budget] = json.loads((root / "configuration.json").read_text())
        with np.load(root / "trajectory.npz") as archive:
            evaluation = {key: archive[key].copy() for key in archive.files}
        with np.load(HERE / "work/runs" / budget / "trajectory.npz") as archive:
            training = {key: archive[key].copy() for key in archive.files}
        assert np.array_equal(evaluation["step"], np.arange(51))
        assert np.array_equal(training["steps"], evaluation["step"])
        for name, values in (("train_kurtosis", training["component_excess_kurtosis"]),
                             ("eval_kurtosis", evaluation["component_excess_kurtosis"]),
                             ("eval_logcosh", evaluation["component_logcosh_deviation"]),
                             ("eval_skewness", np.abs(evaluation["component_skewness"]))):
            assert values.shape == (51, 768) and np.isfinite(values).all()
            data[f"{budget}_{name}"] = np.quantile(values, [.25, .5, .75], axis=1)
    assert len({v["evaluation_manifest_sha256"] for v in metadata.values()}) == 1
    data["steps"] = np.arange(51)
    data["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    return data


def render(data, output, kind):
    settings = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": .7,
        "pdf.fonttype": 42, "ps.fonttype": 42}
    main = kind == "kurtosis"
    keys = ("train_kurtosis", "eval_kurtosis") if main else ("eval_logcosh", "eval_skewness")
    titles = ("A · Fitting tokens", "B · New 1M tokens") if main else (
        "A · Standardized logcosh deviation", "B · Absolute skewness")
    twin = {key: value for key, value in data.items() if key in {"steps", "metadata_json"} or any(key.endswith(s) for s in keys)}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=main)
        for ax, key, title in zip(axes, keys, titles):
            for zorder, budget in enumerate(ORDER, 2):
                low, middle, high = data[f"{budget}_{key}"]
                ax.fill_between(data["steps"], low, high, color=COLORS[budget], alpha=.09, linewidth=0)
                ax.plot(data["steps"], middle, color=COLORS[budget], marker=MARKERS[budget],
                    markevery=10, markersize=3, linewidth=1.3, label=LABELS[budget], zorder=zorder)
            ax.set_title(title, loc="left", fontweight="bold", pad=5)
            ax.set_xlim(0, 50)
            ax.set_xticks([0, 10, 20, 30, 40, 50])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=".89", linewidth=.5)
            ax.set_axisbelow(True)
            if main:
                ax.set_yscale("symlog", linthresh=1)
                ax.axhline(0, color=".6", linestyle="--", linewidth=.7, zorder=0)
            else:
                ax.set_ylim(bottom=0)
        if main:
            lower = min(0., *(float(data[f"{budget}_{key}"][0].min()) for budget in ORDER for key in keys))
            upper = max(float(data[f"{budget}_{key}"][2].max()) for budget in ORDER for key in keys)
            axes[0].set_ylim(lower-.08, upper*1.2)
            axes[0].set_ylabel("Median component excess kurtosis")
        else:
            axes[0].set_ylabel("Median |logcosh − Gaussian|")
            axes[1].set_ylabel("Median |skewness|")
        fig.legend(*axes[0].get_legend_handles_labels(), title="Fit budget (tokens)",
            loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.53, 1.01), title_fontsize=7.5)
        fig.supxlabel("FastICA iteration", fontsize=8, y=.02)
        fig.subplots_adjust(left=.13, right=.98, bottom=.18, top=.73, wspace=.16 if main else .38)
        fig.savefig(output.with_suffix(".pdf"))
        fig.savefig(output.with_suffix(".png"), dpi=300)
        plt.close(fig)
    np.savez_compressed(output.with_suffix(".npz"), **twin)
    output.with_suffix(".txt").write_text(
        "GPT-2 Small residual stream after block 11; 768 ICA components. Five fits used nested "
        "1K/5K/10K/100K/1M-token subsets and 50 fixed FastICA updates. All 51 saved states are "
        "evaluated on the same additional 1,000,000 token positions from 1,852 document-disjoint "
        "Pile documents, excluding exact-text duplicates of the 1,877 training documents. This "
        "is held-out same-corpus generalization, not cross-domain OOD. No direction or whitening "
        "is refitted. Curves show medians across components; bands show their 25th–75th percentiles, "
        "not uncertainty across seeds. Excess kurtosis and skewness use central moments and "
        "evaluation variance. Logcosh contrast is computed after per-component evaluation mean/variance "
        "standardization, with absolute deviation from the standard Gaussian logcosh expectation. "
        "Raw-score logcosh and all per-component statistics are also retained. The kurtosis figure "
        "uses symmetric-log axes linear from -1 to 1 and a Gaussian reference line at zero.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    roots = [HERE / "figures/ica-fitting-extra-token-kurtosis", HERE / "figures/ica-fitting-extra-token-nongaussianity"]
    if not args.force and any(path.with_suffix(".pdf").exists() for path in roots):
        raise SystemExit("pass --force to replace existing figures")
    data = load()
    render(data, roots[0], "kurtosis")
    render(data, roots[1], "diagnostics")


if __name__ == "__main__":
    main()
