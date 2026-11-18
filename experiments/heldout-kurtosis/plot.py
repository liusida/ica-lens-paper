#!/usr/bin/env python3
"""Render Figure 3 with ICA measured on OpenWebText and Pile baselines."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

PAPER = Path(__file__).resolve().parents[3] / "icalens_pypi/paper/overleaf-v2.1"
MODELS = ("gpt2", "gemma2", "qwen9b")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("reference_or_output", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--paper", type=Path, default=PAPER)
    args = parser.parse_args()
    output = args.output or args.reference_or_output
    sys.path.insert(0, str(args.paper.resolve() / "scripts"))

    sys.path.insert(0, str(args.paper.resolve() / "scripts/figures"))
    from direction_excess_kurtosis import STYLES, _render

    with args.heldout.open(newline="") as handle:
        heldout = {
            (row["model"], int(row["layer"])): float(row["heldout_mean_excess_kurtosis"])
            for row in csv.DictReader(handle)
        }
    with np.load(args.paper / "figures/direction-excess-kurtosis.npz") as reference:
        data = dict(reference)

    values = data["mean_excess_kurtosis"]
    ica = list(data["method_ids"]).index("ica")
    for model_index, model in enumerate(MODELS):
        for layer_index in np.flatnonzero(data["valid_layers"][model_index]):
            layer = int(data["layers"][model_index, layer_index])
            key = model, layer
            if key in heldout:
                values[model_index, ica, layer_index] = heldout.pop(key)
            else:
                values[model_index, :, layer_index] = np.nan
    if heldout:
        raise ValueError(f"Layers absent from Figure 3: {sorted(heldout)}")

    STYLES["ica"] = ("ICA (OpenWebText)", *STYLES["ica"][1:])
    for suffix in (".png", ".pdf"):
        with mpl.rc_context({"savefig.dpi": 300}):
            _render(data, output.with_suffix(suffix), debug_box=False)


if __name__ == "__main__":
    main()
