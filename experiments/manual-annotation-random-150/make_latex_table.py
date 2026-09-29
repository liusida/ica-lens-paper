#!/usr/bin/env python3
"""Render the completed random-component annotations as LaTeX tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_TITLES = {
    "gpt2": "GPT-2 Small",
    "gemma-2-2b": "Gemma 2 2B",
    "qwen3.5-9b": "Qwen 3.5 9B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "annotations.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tables")
    parser.add_argument("--force", action="store_true", help="Replace an existing table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.input.read_text(encoding="utf-8"))
    validate(annotations)
    destinations = {
        model: args.output / f"manual-annotations-{model}.tex" for model in MODEL_TITLES
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"output exists: {existing[0]}; pass --force to replace it")
    args.output.mkdir(parents=True, exist_ok=True)
    for model, destination in destinations.items():
        destination.write_text(render(annotations[model]), encoding="utf-8")
        print(destination)


def validate(annotations: object) -> None:
    if not isinstance(annotations, dict) or set(annotations) != set(MODEL_TITLES):
        raise ValueError("annotations must contain exactly the three expected models")
    for model, rows in annotations.items():
        if not isinstance(rows, list) or len(rows) != 50:
            raise ValueError(f"{model} must contain exactly 50 annotations")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{model} contains a malformed annotation")
            if not isinstance(row.get("layer"), int) or not isinstance(row.get("component"), int):
                raise ValueError(f"{model} annotation has an invalid layer or component")
            if not isinstance(row.get("label"), str) or not row["label"].strip():
                raise ValueError(f"{model} annotation has an empty label")
            confidence = row.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, int)
                or confidence not in range(6)
            ):
                raise ValueError(f"{model} annotation has an invalid confidence")


def render(annotations: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{@{}rrp{0.62\textwidth}r@{}}",
        r"  \toprule",
        r"  Layer & Component & Manual label & Confidence \\",
        r"  \midrule",
    ]
    for row in annotations:
        label = "Unknown" if int(row["confidence"]) == 0 else str(row["label"])
        lines.append(
            f"  {int(row['layer'])} & C{int(row['component'])} & "
            f"{latex_escape(label)} & {int(row['confidence'])} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def latex_escape(value: str) -> str:
    value = value.translate(str.maketrans({"’": "'", "“": "``", "”": "''", "–": "--"}))
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


if __name__ == "__main__":
    main()
