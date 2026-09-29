"""Fill blank ERF=1 annotations from each component's top occurrence token."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from icalens import ICALens

ROOT = Path("experiments/manual-annotation-random-150")
ANNOTATIONS = ROOT / "annotations.json"
RESULTS = ROOT / "results"
LENSES = {
    "gpt2": Path("local-icalens-models/official/icalens-gpt2-small-pile10k"),
    "gemma-2-2b": Path("local-icalens-models/official/icalens-gemma-2-2b-pile10k"),
    "qwen3.5-9b": Path(
        "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k"
    ),
}
HEADLINE_THRESHOLD = "15"


def main() -> None:
    annotations = _read_json(ANNOTATIONS)
    filled = 0
    preserved = 0
    for model, lens_path in LENSES.items():
        entries = annotations.get(model)
        if not isinstance(entries, list) or len(entries) != 50:
            raise ValueError(f"expected 50 annotation entries for {model}")
        by_component = {
            (int(entry["layer"]), int(entry["component"])): entry
            for entry in entries
        }
        lens = ICALens.from_pretrained(lens_path)
        for erf_path in sorted((RESULTS / model / "erf").glob("*.json")):
            erf = _read_json(erf_path)
            mean = float(
                erf["threshold_results"][HEADLINE_THRESHOLD]["suffix_erf_mean"]
            )
            if mean != 1.0:
                continue
            key = (int(erf["layer"]), int(erf["component"]))
            entry = by_component.get(key)
            if entry is None:
                raise ValueError(f"ERF cache {erf_path} has no matching annotation")
            if str(entry.get("label", "")):
                preserved += 1
                continue
            profile = lens.component_profile(layer=key[0], component=key[1])
            direction = profile.get("tail_direction")
            occurrences = (
                profile.get("examples", {}).get(direction, {}).get("occurrences", [])
            )
            if not occurrences or not isinstance(occurrences[0].get("text"), str):
                raise ValueError(
                    f"{model} layer {key[0]} C{key[1]} has no selected-tail occurrence token"
                )
            label = occurrences[0]["text"].strip()
            if not label:
                raise ValueError(
                    f"{model} layer {key[0]} C{key[1]} has an empty top occurrence token"
                )
            entry["label"] = label
            filled += 1

    temporary = ANNOTATIONS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(annotations, indent=2) + "\n", encoding="utf-8")
    temporary.replace(ANNOTATIONS)
    print(f"Filled {filled} labels; preserved {preserved} existing nonblank labels.")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


if __name__ == "__main__":
    main()
