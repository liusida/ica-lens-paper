#!/usr/bin/env python3
"""Extract annotator 2 from the uploaded two-annotator CSV into annotations-annotator2.json.

The CSV (raw/annotator2-upload-2026-11-09.csv) is annotator 2's copy of the
original sheet with `new_annotation` / `new_confidence` columns added. The output
uses the same schema as annotations.json (annotator 1). The script fails if the
CSV's `original_*` columns differ from annotations.json in any row.

Usage (from the repo root):
  uv run python experiments/manual-annotation-random-150/extract_annotator2.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw/annotator2-upload-2026-11-09.csv"
ANNOTATOR1 = HERE / "annotations.json"
OUTPUT = HERE / "annotations-annotator2.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.force:
        raise FileExistsError(f"{OUTPUT} exists; pass --force to replace it")

    annotator1 = json.loads(ANNOTATOR1.read_text(encoding="utf-8"))
    with RAW.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output: dict[str, list[dict[str, object]]] = {}
    for model, entries in annotator1.items():
        by_key = {(int(r["layer"]), int(r["component"])): r
                  for r in rows if r["model_key"] == model}
        if len(by_key) != len(entries):
            raise SystemExit(f"{model}: CSV has {len(by_key)} rows, annotations.json has {len(entries)}")
        output[model] = []
        for entry in entries:
            row = by_key.get((entry["layer"], entry["component"]))
            if row is None:
                raise SystemExit(f"{model} L{entry['layer']} C{entry['component']} missing from CSV")
            if (row["original_annotation"], int(row["original_confidence"])) != (entry["label"], entry["confidence"]):
                raise SystemExit(f"{row['component_id']}: CSV original_* columns differ from annotations.json")
            label, confidence = row["new_annotation"].strip(), row["new_confidence"].strip()
            if not label or not confidence:
                raise SystemExit(f"{row['component_id']}: blank annotator-2 entry")
            output[model].append({"layer": entry["layer"], "component": entry["component"],
                                  "label": label, "confidence": int(confidence)})

    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUTPUT.name}: {sum(map(len, output.values()))} components; "
          "annotator-1 columns match annotations.json")


if __name__ == "__main__":
    main()
