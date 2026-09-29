"""Reproducibly sample 50 ICA components from each of three base models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("experiments/manual-annotation-random-150")
RESULTS = ROOT / "results"
LENSES = {
    "gpt2": Path("local-icalens-models/official/icalens-gpt2-small-pile10k"),
    "gemma-2-2b": Path("local-icalens-models/official/icalens-gemma-2-2b-pile10k"),
    "qwen3.5-9b": Path(
        "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k"
    ),
}
SAMPLE_SIZE = 50
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = RESULTS / "sampling.json"
    csv_paths = {label: RESULTS / label / "components.csv" for label in LENSES}
    existing = [
        path for path in (metadata_path, *csv_paths.values()) if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError("sample outputs already exist; pass --force to replace them")

    total_samples = 0
    models: list[dict[str, Any]] = []
    for label, lens_path in LENSES.items():
        manifest_path = lens_path / "icalens.json"
        manifest = _read_json(manifest_path)
        population = _component_population(manifest)
        derived_seed = _stable_seed(SEED, label)
        rng = np.random.default_rng(derived_seed)
        selected_indices = sorted(
            int(value)
            for value in rng.choice(len(population), size=SAMPLE_SIZE, replace=False)
        )
        rows: list[dict[str, int]] = []
        for sample_index, population_index in enumerate(selected_indices, start=1):
            layer, component = population[population_index]
            rows.append(
                {
                    "sample_index": sample_index,
                    "layer": layer,
                    "component_id": component,
                }
            )
        csv_path = csv_paths[label]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        total_samples += len(rows)
        models.append(
            {
                "label": label,
                "lens": str(lens_path),
                "model": manifest.get("model"),
                "available_layers": sorted(int(layer) for layer in manifest["layers"]),
                "population_size": len(population),
                "derived_seed": derived_seed,
                "output": str(csv_path),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "sampling_method": (
                    "uniform without replacement over all available "
                    "(layer, component) pairs within each model"
                ),
                "rng": "numpy.random.default_rng",
                "base_seed": SEED,
                "samples_per_model": SAMPLE_SIZE,
                "total_samples": total_samples,
                "models": models,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {total_samples} sampled components under {RESULTS}")


def _component_population(manifest: dict[str, Any]) -> list[tuple[int, int]]:
    population: list[tuple[int, int]] = []
    layers = manifest.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Lens manifest has no layers")
    for layer_text, layer_manifest in sorted(
        layers.items(), key=lambda item: int(item[0])
    ):
        count = int(layer_manifest["n_components"])
        population.extend((int(layer_text), component) for component in range(count))
    return population


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


if __name__ == "__main__":
    main()
