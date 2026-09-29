#!/usr/bin/env python3
"""Measure projection variance and excess kurtosis across representation methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._sae import _load_sae_tensors, _orient_decoder
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.saebench_sparse_probing import (
    _prepare_layer_baselines,
    _resolve_baselines,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORMAT = "icalens.kurtosis_variance_comparison"
SCHEMA_VERSION = 4
METHODS = ("ica", "sae", "pca", "random")
MODEL_SPECS = {
    "gpt2": {
        "title": "GPT-2 Small",
        "lens": ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k",
        "activations": Path(
            "/home/liusida/Expansion/research/ICA-data/icalens-activations/gpt2-pile10k-1m"
        ),
    },
    "gemma2": {
        "title": "Gemma 2 2B",
        "lens": ROOT / "local-icalens-models/official/icalens-gemma-2-2b-pile10k",
        "activations": Path(
            "/home/liusida/Expansion/research/ICA-data/icalens-activations/"
            "gemma-2-2b-pile10k-1m"
        ),
    },
    "qwen9b": {
        "title": "Qwen 3.5 9B Base",
        "lens": ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k",
        "activations": Path(
            "/home/liusida/Expansion/research/ICA-data/icalens-activations/"
            "qwen3.5-9b-base-pile10k-1m"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    value = ":".join((str(base), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % (2**63)


def identity_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_models(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(MODEL_SPECS)
    invalid = sorted(set(values).difference(MODEL_SPECS))
    if invalid:
        raise ValueError(f"unknown models: {invalid}")
    return list(dict.fromkeys(values))


def parse_layers(value: str, available: tuple[int, ...]) -> list[int]:
    if value == "all":
        return list(available)
    selected = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    invalid = sorted(set(selected).difference(available))
    if not selected or invalid:
        raise ValueError(f"invalid layers {invalid or selected}; available: {list(available)}")
    return selected


def resolved_configuration(
    labels: list[str],
    lenses: dict[str, ICALens],
    datasets: dict[str, ActivationDataset],
    layers: dict[str, list[int]],
    *,
    activation_samples: int,
    directions: int | None,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label in labels:
        lens, dataset = lenses[label], datasets[label]
        baseline = _resolve_baselines(lens.model_id, "sae")["sae"]
        models[label] = {
            "title": MODEL_SPECS[label]["title"],
            "lens": str(Path(MODEL_SPECS[label]["lens"]).resolve()),
            "lens_manifest_sha256": sha256(Path(MODEL_SPECS[label]["lens"]) / "icalens.json"),
            "activation_dataset": str(dataset.path),
            "activation_manifest_sha256": sha256(dataset.path / "activations.json"),
            "model": dataset.model,
            "layers": layers[label],
            "directions_per_method": (
                dataset.hidden_size if directions is None else min(directions, dataset.hidden_size)
            ),
            "sae": {
                key: baseline[key]
                for key in ("name", "repo_id", "revision", "checkpoint_format", "width")
            },
        }
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "models": models,
        "methods": list(METHODS),
        "activation_samples": activation_samples,
        "activation_sampling": (
            "all rows when N equals dataset size; otherwise seeded without replacement"
        ),
        "requested_directions_per_method": (
            "model_hidden_size" if directions is None else directions
        ),
        "direction_count_rule": (
            "model hidden size by default; an explicit smaller cap is allowed for smoke runs"
        ),
        "direction_sampling": "seeded uniform sampling without replacement",
        "sampling_seed": seed,
        "maximum_activation_batch_size": batch_size,
        "maximum_projection_elements_per_batch": 4_000_000,
        "projection": "raw centered residual activation projected onto unit-L2 direction",
        "ica_direction": "normalized row of effective ICA reading matrix",
        "sae_direction": "normalized SAE decoder row; encoder nonlinearity is not applied",
        "pca_direction": "orthonormal covariance eigenvector recovered from ICA writing matrix",
        "random_direction": "seeded Haar-distributed orthonormal direction",
        "variance": "population second central moment",
        "mean": "population mean of the raw unit-direction projection",
        "skewness": "population standardized third central moment m3/m2^(3/2)",
        "kurtosis": "population excess kurtosis m4/m2^2 - 3 (Gaussian = 0)",
    }


def unit_identity(
    resolved: dict[str, Any], label: str, layer: int, dependency: dict[str, str]
) -> dict[str, Any]:
    return {
        "run": resolved,
        "model": label,
        "layer": layer,
        "dependency": dependency,
    }


def sample_rows(values: torch.Tensor, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count > len(values):
        raise ValueError(f"cannot sample {count} directions from {len(values)} rows")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(values), generator=generator)[:count]
    selected = torch.nn.functional.normalize(values[indices].float(), dim=1)
    return indices, selected


def pca_directions(
    writing: np.ndarray, count: int, seed: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the project PCA baseline and retain a seeded subset of its rows."""
    work = torch.from_numpy(np.asarray(writing)).to(device=device, dtype=torch.float32)
    covariance = work @ work.T
    del work
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors.flip(1).T.contiguous()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(components), generator=generator)[:count]
    selected = components[indices.to(components.device)].cpu()
    del covariance, components, eigenvectors
    return indices, selected


def random_directions(hidden_size: int, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the first count rows of a Haar orthogonal basis without forming a d x d matrix."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn((hidden_size, count), generator=generator, dtype=torch.float32)
    basis, triangular = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(
        torch.diagonal(triangular) < 0,
        -torch.ones(count, dtype=torch.float32),
        torch.ones(count, dtype=torch.float32),
    )
    return torch.arange(count, dtype=torch.int64), (basis * signs).T.contiguous()


def _merge_moments(
    state: tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    values: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge one batch into stable parallel central moments through order four."""
    values = values.to(torch.float64)
    nb = int(values.shape[0])
    mean_b = values.mean(dim=0)
    centered = values - mean_b
    m2_b = centered.square().sum(dim=0)
    m3_b = centered.pow(3).sum(dim=0)
    m4_b = centered.pow(4).sum(dim=0)
    if state is None:
        return nb, mean_b, m2_b, m3_b, m4_b
    na, mean_a, m2_a, m3_a, m4_a = state
    n = na + nb
    delta = mean_b - mean_a
    mean = mean_a + delta * (nb / n)
    m2 = m2_a + m2_b + delta.square() * na * nb / n
    m3 = (
        m3_a
        + m3_b
        + delta.pow(3) * na * nb * (na - nb) / (n * n)
        + 3 * delta * (na * m2_b - nb * m2_a) / n
    )
    m4 = (
        m4_a
        + m4_b
        + delta.pow(4) * na * nb * (na * na - na * nb + nb * nb) / (n**3)
        + 6 * delta.square() * (na * na * m2_b + nb * nb * m2_a) / (n * n)
        + 4 * delta * (na * m3_b - nb * m3_a) / n
    )
    return n, mean, m2, m3, m4


def measure(
    activations: torch.Tensor,
    indices: torch.Tensor | None,
    directions: dict[str, torch.Tensor],
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    combined = torch.cat([directions[method] for method in METHODS], dim=0).to(
        device=device, dtype=torch.float32
    )
    effective_batch_size = min(batch_size, max(16, 4_000_000 // len(combined)))
    state = None
    with torch.inference_mode():
        total = len(activations) if indices is None else len(indices)
        for start in range(0, total, effective_batch_size):
            if indices is None:
                batch = activations[start : start + effective_batch_size]
            else:
                batch = activations[indices[start : start + effective_batch_size]]
            projections = batch.to(device=device, dtype=torch.float32) @ combined.T
            state = _merge_moments(state, projections)
    assert state is not None
    n, means, m2, m3, m4 = state
    variance = m2 / n
    skewness = m3 / n / variance.pow(1.5)
    excess = m4 / n / variance.square() - 3.0
    result: dict[str, np.ndarray] = {}
    width = next(iter(directions.values())).shape[0]
    for method_index, method in enumerate(METHODS):
        selection = slice(method_index * width, (method_index + 1) * width)
        result[f"{method}_mean"] = means[selection].cpu().numpy()
        result[f"{method}_variance"] = variance[selection].cpu().numpy()
        result[f"{method}_skewness"] = skewness[selection].cpu().numpy()
        result[f"{method}_excess_kurtosis"] = excess[selection].cpu().numpy()
    return result


def load_checkpoint(
    path: Path, identity: dict[str, Any], count: int
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
    except (OSError, ValueError):
        return None
    if str(values.get("identity_sha256", "")) != identity_sha256(identity):
        return None
    for method in METHODS:
        for suffix in ("direction_ids", "mean", "variance", "skewness", "excess_kurtosis"):
            key = f"{method}_{suffix}"
            if key not in values or values[key].shape != (count,):
                return None
            if suffix != "direction_ids" and not np.isfinite(values[key]).all():
                return None
        if np.any(values[f"{method}_variance"] <= 0):
            return None
    return values


def atomic_write_checkpoint(
    path: Path, identity: dict[str, Any], values: dict[str, np.ndarray], count: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary, identity_sha256=np.asarray(identity_sha256(identity)), **values
    )
    if load_checkpoint(temporary, identity, count) is None:
        raise ValueError(f"temporary checkpoint failed validation: {temporary}")
    temporary.replace(path)


def summary_row(
    label: str, layer: int, path: Path, values: dict[str, np.ndarray]
) -> dict[str, Any]:
    row: dict[str, Any] = {"model": label, "layer": layer, "checkpoint": str(path)}
    for method in METHODS:
        for metric in ("mean", "variance", "skewness", "excess_kurtosis"):
            samples = values[f"{method}_{metric}"]
            row[f"{method}_{metric}_mean"] = float(np.mean(samples))
            row[f"{method}_{metric}_median"] = float(np.median(samples))
            row[f"{method}_{metric}_q25"] = float(np.quantile(samples, 0.25))
            row[f"{method}_{metric}_q75"] = float(np.quantile(samples, 0.75))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--layers", default="all", help="all or comma-separated layer IDs")
    parser.add_argument("--activation-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--directions",
        type=int,
        default=None,
        help="optional per-model cap; default uses the model hidden size",
    )
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=HERE / "runs/main")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (
        args.activation_samples < 100
        or (args.directions is not None and args.directions < 1)
        or args.batch_size < 1
    ):
        parser.error("activation samples >= 100, direction cap, and batch size must be positive")

    labels = parse_models(args.models)
    lenses = {label: ICALens.from_pretrained(MODEL_SPECS[label]["lens"]) for label in labels}
    datasets = {label: ActivationDataset(MODEL_SPECS[label]["activations"]) for label in labels}
    layers = {
        label: parse_layers(args.layers, datasets[label].available_layers) for label in labels
    }
    for label in labels:
        lens, dataset = lenses[label], datasets[label]
        if (
            dataset.model["repo_id"] != lens.model_id
            or dataset.model["revision"] != lens.model_revision
        ):
            raise ValueError(f"activation/model identity mismatch for {label}")
        if args.activation_samples > dataset.sample_count:
            raise ValueError(f"{label} only contains {dataset.sample_count} activations")
        if set(layers[label]).difference(lens.available_layers):
            raise ValueError(f"lens has no requested layer for {label}")

    output = args.output.expanduser().resolve()
    source = source_provenance()
    units = [(label, layer) for label in labels for layer in layers[label]]
    resolved = resolved_configuration(
        labels,
        lenses,
        datasets,
        layers,
        activation_samples=args.activation_samples,
        directions=args.directions,
        seed=args.sampling_seed,
        batch_size=args.batch_size,
    )
    run = ResumableRun.open(
        output=output, resolved=resolved, source=source, status="measuring"
    )
    identities: dict[tuple[str, int], dict[str, Any]] = {}
    cached: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    completed: set[tuple[str, int]] = set()
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for label, layer in units:
        lens = lenses[label]
        baseline = _prepare_layer_baselines(
            _resolve_baselines(lens.model_id, "sae"), layer=layer
        )["sae"]
        baselines[label, layer] = baseline
        artifact = lens._get_layer(layer)
        ica_path = Path(MODEL_SPECS[label]["lens"]) / artifact.file
        weights_path = Path(baseline["weights_file"])
        dependency = {
            "ica_layer_file": artifact.file,
            "ica_layer_bytes": ica_path.stat().st_size,
            "sae_repo_id": str(baseline["repo_id"]),
            "sae_revision": str(baseline["revision"]),
            "sae_checkpoint": str(baseline["checkpoint"]),
            "sae_checkpoint_format": str(baseline["checkpoint_format"]),
            "sae_weights_bytes": weights_path.stat().st_size,
        }
        identity = unit_identity(resolved, label, layer, dependency)
        identities[label, layer] = identity
        path = output / label / f"layer_{layer:02d}.npz"
        direction_count = int(resolved["models"][label]["directions_per_method"])
        values = None if args.force else load_checkpoint(path, identity, direction_count)
        if values is not None:
            cached[label, layer] = values
            completed.add((label, layer))

    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · kurtosis and variance comparison",
        completed=len(completed),
        total=len(units),
        completed_unit_ids=completed,
        unit_label="model-layers",
        recent_label="Recent measurement output",
        detail_filename="measurement-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        counts = ", ".join(
            f"{label}={resolved['models'][label]['directions_per_method']}"
            for label in labels
        )
        log(f"Validated {len(units)} model-layer units; directions per method: {counts}.")
        for label, layer in units:
            if (label, layer) in completed:
                continue
            display.phase("Preparing direction dictionaries", model=label, layer=layer)
            lens, dataset = lenses[label], datasets[label]
            direction_count = int(resolved["models"][label]["directions_per_method"])
            artifact = lens._get_layer(layer)
            assert artifact.reading_matrix is not None and artifact.writing_matrix is not None
            baseline = baselines[label, layer]
            tensors = _load_sae_tensors(
                Path(baseline["weights_file"]),
                checkpoint_format=str(baseline["checkpoint_format"]),
            )
            sae = _orient_decoder(
                tensors["W_dec"], hidden_size=dataset.hidden_size, width=int(baseline["width"])
            )
            direction_ids: dict[str, torch.Tensor] = {}
            directions: dict[str, torch.Tensor] = {}
            direction_ids["ica"], directions["ica"] = sample_rows(
                torch.from_numpy(np.asarray(artifact.reading_matrix)),
                direction_count,
                stable_seed(args.sampling_seed, label, layer, "ica"),
            )
            direction_ids["sae"], directions["sae"] = sample_rows(
                sae,
                direction_count,
                stable_seed(args.sampling_seed, label, layer, "sae"),
            )
            direction_ids["pca"], directions["pca"] = pca_directions(
                artifact.writing_matrix,
                direction_count,
                stable_seed(args.sampling_seed, label, layer, "pca"),
                args.device,
            )
            direction_ids["random"], directions["random"] = random_directions(
                dataset.hidden_size,
                direction_count,
                stable_seed(args.sampling_seed, label, layer, "random"),
            )
            activation_ids = None
            if args.activation_samples < dataset.sample_count:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(stable_seed(args.sampling_seed, label, layer, "activations"))
                activation_ids = torch.randperm(dataset.sample_count, generator=generator)[
                    : args.activation_samples
                ].sort().values
            display.phase("Streaming stored activations", model=label, layer=layer)
            values = measure(
                dataset.layer(layer),
                activation_ids,
                directions,
                device=args.device,
                batch_size=args.batch_size,
            )
            for method in METHODS:
                values[f"{method}_direction_ids"] = direction_ids[method].numpy().astype(np.int64)
            path = output / label / f"layer_{layer:02d}.npz"
            atomic_write_checkpoint(path, identities[label, layer], values, direction_count)
            checked = load_checkpoint(path, identities[label, layer], direction_count)
            if checked is None:
                raise ValueError(f"new checkpoint failed validation: {path}")
            cached[label, layer] = checked
            display.complete_unit((label, layer), refresh=True)
            log(
                f"{label} L{layer}: median excess kurtosis "
                + ", ".join(
                    f"{method.upper()}={np.median(values[f'{method}_excess_kurtosis']):.3g}"
                    for method in METHODS
                )
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        rows = [
            summary_row(
                label,
                layer,
                output / label / f"layer_{layer:02d}.npz",
                cached[label, layer],
            )
            for label, layer in units
        ]
        display.phase("Writing summaries")
        atomic_write_json(
            output / "summary.json",
            {"format": FORMAT, "schema_version": SCHEMA_VERSION, "rows": rows},
        )
        temporary = output / "summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "summary.csv")
        run.set_status("complete", complete=True)
        log(f"Experiment complete: {output}")


if __name__ == "__main__":
    main()
