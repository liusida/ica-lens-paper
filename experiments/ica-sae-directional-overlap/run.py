#!/usr/bin/env python3
"""Measure nearest SAE-decoder overlap for every ICA writing direction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from icalens import ICALens
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._sae import _load_sae_tensors, _orient_decoder
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.saebench_sparse_probing import _prepare_layer_baselines, _resolve_baselines

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORMAT, SCHEMA_VERSION = "icalens.ica_sae_directional_overlap", 3
MODEL_SPECS = {
    "gpt2": (
        "GPT-2 Small",
        "#4C78A8",
        ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k",
    ),
    "gemma2": (
        "Gemma 2 2B",
        "#59A14F",
        ROOT / "local-icalens-models/official/icalens-gemma-2-2b-pile10k",
    ),
    "qwen9b": (
        "Qwen 3.5 9B Base",
        "#B279A2",
        ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_layers(value: str, available: tuple[int, ...]) -> list[int]:
    if value == "all":
        return list(available)
    layers = sorted({int(x) for x in value.split(",") if x.strip()})
    if not layers or set(layers).difference(available):
        raise ValueError(f"invalid layers {layers}; available: {list(available)}")
    return layers


def resolved_configuration(labels, lenses, layers, random_seed) -> dict[str, Any]:
    models = {}
    for label in labels:
        title, _, lens_path = MODEL_SPECS[label]
        lens, lens_path = lenses[label], lens_path.resolve()
        manifest_path = lens_path / "icalens.json"
        manifest = json.loads(manifest_path.read_text())
        sae = _resolve_baselines(lens.model_id, "sae")["sae"]
        dependencies = {}
        for layer in layers[label]:
            entry = manifest["layers"][str(layer)]
            artifact = lens_path / entry["file"]
            checkpoints = sae.get("layer_checkpoints")
            checkpoint = (
                str(checkpoints[str(layer)])
                if isinstance(checkpoints, dict)
                else str(sae["checkpoint_template"]).format(layer=layer)
            )
            dependencies[str(layer)] = {
                "ica_artifact": str(artifact),
                "ica_artifact_sha256": sha256(artifact),
                "ica_components": int(entry["n_components"]),
                "sae_checkpoint": checkpoint,
            }
        models[label] = {
            "title": title,
            "lens": str(lens_path),
            "lens_manifest_sha256": sha256(manifest_path),
            "model": manifest["model"],
            "layers": layers[label],
            "layer_dependencies": dependencies,
            "sae": {
                key: sae[key]
                for key in ("name", "repo_id", "revision", "checkpoint_format", "width")
            },
        }
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "method": "maximum absolute cosine over all same-layer SAE decoder rows",
        "ica_direction": "column of writing_matrix normalized to unit L2 norm",
        "sae_direction": "checkpoint W_dec row normalized to unit L2 norm",
        "random_reference": (
            "isotropic Gaussian directions matched to the ICA component count and "
            "compared with the same SAE decoder dictionary"
        ),
        "random_seed": random_seed,
        "compute_dtype": "float32",
        "models": models,
    }


def unit_identity(resolved, label, layer):
    model = resolved["models"][label]
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "method": resolved["method"],
        "ica_direction": resolved["ica_direction"],
        "sae_direction": resolved["sae_direction"],
        "random_reference": resolved["random_reference"],
        "random_seed": resolved["random_seed"],
        "compute_dtype": resolved["compute_dtype"],
        "model": model["model"],
        "lens_manifest_sha256": model["lens_manifest_sha256"],
        "sae": model["sae"],
        "layer": layer,
        "dependencies": model["layer_dependencies"][str(layer)],
    }


@torch.inference_mode()
def nearest_cosines(ica, sae, chunk_size):
    ica, sae = (
        torch.nn.functional.normalize(ica.float(), dim=1),
        torch.nn.functional.normalize(sae.float(), dim=1),
    )
    values, indices = [], []
    for start in range(0, len(ica), chunk_size):
        maximum, nearest = (ica[start : start + chunk_size] @ sae.T).abs().max(dim=1)
        values.append(maximum.cpu())
        indices.append(nearest.cpu())
    return torch.cat(values).numpy(), torch.cat(indices).numpy()


def layer_random_seed(base_seed, label, layer):
    digest = hashlib.sha256(f"{base_seed}:{label}:{layer}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def validate_checkpoint_arrays(identity, recorded, values, nearest, random_values, random_nearest):
    count = int(identity["dependencies"]["ica_components"])
    width = int(identity["sae"]["width"])
    return not (
        recorded != identity_sha256(identity)
        or values.shape != (count,)
        or nearest.shape != values.shape
        or random_values.shape != values.shape
        or random_nearest.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(random_values).all()
        or np.any((values < 0) | (values > 1.0001))
        or np.any((random_values < 0) | (random_values > 1.0001))
        or np.any((nearest < 0) | (nearest >= width))
        or np.any((random_nearest < 0) | (random_nearest >= width))
    )


def atomic_write_checkpoint(path, identity, values, nearest, random_values, random_nearest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    identity_hash = identity_sha256(identity)
    values = values.astype(np.float32)
    nearest = nearest.astype(np.int64)
    random_values = random_values.astype(np.float32)
    random_nearest = random_nearest.astype(np.int64)
    if not validate_checkpoint_arrays(
        identity, identity_hash, values, nearest, random_values, random_nearest
    ):
        raise ValueError("refusing to write invalid checkpoint arrays")
    np.savez_compressed(
        temporary,
        identity_sha256=np.asarray(identity_hash),
        nearest_absolute_cosine=values,
        nearest_sae_feature=nearest,
        random_nearest_absolute_cosine=random_values,
        random_nearest_sae_feature=random_nearest,
    )
    with np.load(temporary, allow_pickle=False) as data:
        if not validate_checkpoint_arrays(
            identity,
            str(data["identity_sha256"]),
            data["nearest_absolute_cosine"],
            data["nearest_sae_feature"],
            data["random_nearest_absolute_cosine"],
            data["random_nearest_sae_feature"],
        ):
            raise ValueError("temporary checkpoint validation failed")
    temporary.replace(path)


def load_checkpoint(path, identity):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            recorded = str(data["identity_sha256"])
            values = data["nearest_absolute_cosine"].astype(np.float32)
            nearest = data["nearest_sae_feature"].astype(np.int64)
            random_values = data["random_nearest_absolute_cosine"].astype(np.float32)
            random_nearest = data["random_nearest_sae_feature"].astype(np.int64)
    except (OSError, KeyError, ValueError):
        return None
    if not validate_checkpoint_arrays(
        identity, recorded, values, nearest, random_values, random_nearest
    ):
        return None
    return values, nearest, random_values, random_nearest


def measure_layer(lens, label, layer, baseline, device, chunk_size, random_seed):
    artifact = lens._get_layer(layer)
    if artifact.writing_matrix is None:
        raise ValueError(f"layer {layer} has no writing matrix")
    writing = torch.from_numpy(np.asarray(artifact.writing_matrix)).T.contiguous()
    hidden_size, width = int(writing.shape[1]), int(baseline["width"])
    tensors = _load_sae_tensors(
        Path(baseline["weights_file"]), checkpoint_format=str(baseline["checkpoint_format"])
    )
    decoder = _orient_decoder(tensors["W_dec"], hidden_size=hidden_size, width=width)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(layer_random_seed(random_seed, label, layer))
    random_directions = torch.randn(writing.shape, generator=generator, dtype=torch.float32)
    decoder = decoder.to(device)
    values, nearest = nearest_cosines(writing.to(device), decoder, chunk_size)
    random_values, random_nearest = nearest_cosines(
        random_directions.to(device), decoder, chunk_size
    )
    return values, nearest, random_values, random_nearest


def summarize(label, layer, values, random_values, identity, path):
    return {
        "model": label,
        "layer": layer,
        "ica_components": len(values),
        "sae_features": int(identity["sae"]["width"]),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "random_median": float(np.median(random_values)),
        "random_q25": float(np.quantile(random_values, 0.25)),
        "random_q75": float(np.quantile(random_values, 0.75)),
        "random_mean": float(np.mean(random_values)),
        "random_minimum": float(np.min(random_values)),
        "random_maximum": float(np.max(random_values)),
        "median_excess_over_random": float(np.median(values) - np.median(random_values)),
        "checkpoint": str(path),
        "sae_checkpoint": identity["dependencies"]["sae_checkpoint"],
        "sae_repo_id": identity["sae"]["repo_id"],
        "sae_revision": identity["sae"]["revision"],
    }


def write_summary(output, resolved, rows):
    atomic_write_json(
        output / "summary.json",
        {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "comparison": resolved["method"],
            "ica_direction": resolved["ica_direction"],
            "sae_direction": resolved["sae_direction"],
            "random_reference": resolved["random_reference"],
            "random_seed": resolved["random_seed"],
            "nearest_neighbor_width_caveat": "nearest cosine depends on SAE dictionary width",
            "models": list(resolved["models"]),
            "rows": rows,
        },
    )
    path, temporary = output / "summary.csv", output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render(rows, output, resolved):
    labels = list(resolved["models"])
    figure, axes = plt.subplots(1, len(labels), figsize=(3.15 * len(labels), 2.3), squeeze=False)
    for column, label in enumerate(labels):
        selected = [row for row in rows if row["model"] == label]
        x = np.asarray([row["layer"] for row in selected])
        median, q25, q75, random_median, random_q25, random_q75 = (
            np.asarray([row[key] for row in selected])
            for key in (
                "median",
                "q25",
                "q75",
                "random_median",
                "random_q25",
                "random_q75",
            )
        )
        axis, color = axes[0, column], MODEL_SPECS[label][1]
        axis.fill_between(x, random_q25, random_q75, color="0.65", alpha=0.18, linewidth=0)
        axis.plot(
            x,
            random_median,
            color="0.55",
            linestyle="--",
            linewidth=1.0,
            label="Random",
        )
        axis.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
        axis.plot(
            x,
            median,
            color=color,
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            label="ICA",
        )
        axis.set(title=MODEL_SPECS[label][0], xlabel="Layer", ylim=(0, 1))
        axis.grid(axis="y", color=".88", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Nearest absolute SAE cosine")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "directional-overlap.pdf")
    figure.savefig(output / "directional-overlap.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=tuple(MODEL_SPECS)
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk size must be positive")
    labels = list(args.models)
    lenses = {label: ICALens.from_pretrained(MODEL_SPECS[label][2]) for label in labels}
    layers = {label: parse_layers(args.layers, lenses[label].available_layers) for label in labels}
    resolved = resolved_configuration(labels, lenses, layers, args.random_seed)
    output = args.output.expanduser().resolve()
    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="measuring")
    units = [(label, layer) for label in labels for layer in layers[label]]
    completed, cached = set(), {}
    for label, layer in units:
        identity = unit_identity(resolved, label, layer)
        path = output / label / f"layer_{layer:02d}.npz"
        value = None if args.force else load_checkpoint(path, identity)
        if value is not None:
            completed.add((label, layer))
            cached[label, layer] = value
    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · ICA–SAE directional overlap",
        completed=len(completed),
        total=len(units),
        completed_unit_ids=completed,
        unit_label="model-layers",
        recent_label="Recent overlap output",
        detail_filename="directional-overlap-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        log(f"Validated configuration: {len(labels)} models, {len(units)} layers.")
        for label in labels:
            pending = [layer for layer in layers[label] if (label, layer) not in completed]
            if not pending:
                log(f"{label}: all requested layers complete; reusing checkpoints.")
                continue
            registry = _resolve_baselines(lenses[label].model_id, "sae")
            for layer in pending:
                display.phase("Comparing directions", model=label, layer=layer)
                log(f"{label} layer {layer}: loading pinned SAE checkpoint.")
                baseline = _prepare_layer_baselines(registry, layer=layer)["sae"]
                values, nearest, random_values, random_nearest = measure_layer(
                    lenses[label],
                    label,
                    layer,
                    baseline,
                    args.device,
                    args.chunk_size,
                    args.random_seed,
                )
                identity, path = (
                    unit_identity(resolved, label, layer),
                    output / label / f"layer_{layer:02d}.npz",
                )
                atomic_write_checkpoint(
                    path,
                    identity,
                    values,
                    nearest,
                    random_values,
                    random_nearest,
                )
                cached[label, layer] = load_checkpoint(path, identity)
                if cached[label, layer] is None:
                    raise ValueError(f"new checkpoint failed validation: {path}")
                display.complete_unit((label, layer), refresh=True)
                log(f"{label} layer {layer}: checkpointed median {np.median(values):.4f}.")
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        rows = [
            summarize(
                label,
                layer,
                cached[label, layer][0],
                cached[label, layer][2],
                unit_identity(resolved, label, layer),
                output / label / f"layer_{layer:02d}.npz",
            )
            for label, layer in units
        ]
        display.phase("Writing summaries")
        write_summary(output, resolved, rows)
        render(rows, output, resolved)
        run.set_status("complete", complete=True)
        log(f"Experiment complete: {output}")


if __name__ == "__main__":
    main()
