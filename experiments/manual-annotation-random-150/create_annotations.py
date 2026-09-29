"""Initialize the manual-annotation JSON from the sampled component lists."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("experiments/manual-annotation-random-150")
RESULTS = ROOT / "results"
OUTPUT = ROOT / "annotations.json"
MODELS = ("gpt2", "gemma-2-2b", "qwen3.5-9b")


def main() -> None:
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if existing != {}:
            raise FileExistsError(f"refusing to overwrite existing annotations: {OUTPUT}")
    annotations = {}
    for model in MODELS:
        with (RESULTS / model / "components.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 50:
            raise ValueError(f"expected 50 sampled components for {model}, found {len(rows)}")
        annotations[model] = [
            {
                "layer": int(row["layer"]),
                "component": int(row["component_id"]),
                "label": "",
                "confidence": None,
            }
            for row in rows
        ]
    OUTPUT.write_text(json.dumps(annotations, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sum(map(len, annotations.values()))} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
