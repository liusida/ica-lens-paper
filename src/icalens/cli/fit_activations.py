"""Fit an ICA Lens from a reusable disk-backed activation dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import torch

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset

from ._status import log
from .fit_text import peak_rss_gib, set_cuda_memory_limit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens fit activations", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Captured activation directory.")
    parser.add_argument("--layers", default="all", help="Captured layers to fit (default: all).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--icalens-preprocessing",
        choices=("none", "l2", "geometric-median-l2"),
        default="none",
    )
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--objective-every", type=int, default=1)
    parser.add_argument("--fit-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-vram-gb", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("'icalens fit activations' requires a CUDA device")
    if args.max_vram_gb is not None:
        set_cuda_memory_limit(args.max_vram_gb)
    if args.fit_batch_size < 0:
        raise ValueError("--fit-batch-size must be non-negative")
    if args.max_iter <= 0 or args.objective_every <= 0:
        raise ValueError("--max-iter and --objective-every must be positive")
    torch.cuda.reset_peak_memory_stats()

    dataset = ActivationDataset(args.input)
    requested_layers = _parse_layers(args.layers, dataset.available_layers)
    log(f"Requested layers: {','.join(map(str, requested_layers))}")
    model = dataset.model
    model_type = str(model.get("type") or "base")
    if model_type not in {"base", "instruct"}:
        raise ValueError(f"unsupported captured model type: {model_type!r}")
    output = args.output.expanduser().resolve()
    manifest = output / "icalens.json"
    if manifest.is_file():
        lens = ICALens.from_pretrained(output)
        _validate_resume(lens, dataset, args.icalens_preprocessing)
        # ``fit`` detaches a lens from its source artifact. Materialize every
        # existing layer first so the next checkpoint can preserve their tensors.
        for completed_layer in lens.available_layers:
            lens._get_layer(completed_layer)
        layers = tuple(layer for layer in requested_layers if layer not in lens.available_layers)
        log(
            f"Resuming {output}; preserving completed layers {lens.available_layers}. "
            f"Remaining requested layers: {layers or 'none'}."
        )
    else:
        if output.is_dir() and any(output.iterdir()):
            raise FileExistsError(
                f"output directory is non-empty but has no ICA Lens manifest: {output}"
            )
        lens = ICALens(
            model_id=str(model["repo_id"]),
            model_revision=str(model.get("revision") or "unknown"),
            model_type=cast(Literal["base", "instruct"], model_type),
            activation_site=str(dataset.manifest["activation_site"]),
            layer_indexing=str(dataset.manifest["layer_indexing"]),
            icalens_preprocessing=args.icalens_preprocessing,
        )
        layers = requested_layers
    for layer in layers:
        values = dataset.layer(layer)
        minimum = dataset.hidden_size + 1
        if dataset.sample_count < minimum:
            raise ValueError(
                f"Cannot fit {dataset.hidden_size} components from {dataset.sample_count} rows; "
                f"capture at least {minimum} token activations."
            )
        batch_size = dataset.sample_count if args.fit_batch_size == 0 else args.fit_batch_size
        log(
            f"Fitting layer {layer} from disk-backed {dataset.dtype} activations: "
            f"{dataset.sample_count} tokens, {dataset.hidden_size} components, "
            f"max_iter={args.max_iter}, fit_batch_size={batch_size}..."
        )
        lens.fit(
            values,
            layer=layer,
            n_components=dataset.hidden_size,
            algorithm="parallel",
            fun="logcosh",
            max_iter=args.max_iter,
            random_state=args.seed,
            progress=True,
            device="cuda",
            batch_size=batch_size,
            objective_every=args.objective_every,
            provenance=dataset.provenance,
        )
        lens.save(output)
        log(f"Checkpointed layer {layer} to {output}")
    log(f"Available layers: {lens.available_layers}")
    log(f"Peak PyTorch CUDA memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")
    log(f"Peak process resident memory (RSS): {peak_rss_gib():.2f} GiB")


def _validate_resume(lens: ICALens, dataset: ActivationDataset, preprocessing: str) -> None:
    model = dataset.model
    expected = {
        "model_id": str(model["repo_id"]),
        "model_revision": str(model.get("revision") or "unknown"),
        "model_type": str(model.get("type") or "base"),
        "activation_site": str(dataset.manifest["activation_site"]),
        "layer_indexing": str(dataset.manifest["layer_indexing"]),
        "icalens_preprocessing": preprocessing,
    }
    mismatches = [
        f"{name}: {getattr(lens, name)!r} != {value!r}"
        for name, value in expected.items()
        if getattr(lens, name) != value
    ]
    if lens._hidden_size is not None and lens._hidden_size != dataset.hidden_size:
        mismatches.append(f"hidden_size: {lens._hidden_size!r} != {dataset.hidden_size!r}")
    if mismatches:
        raise ValueError("cannot resume from incompatible artifact: " + "; ".join(mismatches))


def _parse_layers(value: str, available: tuple[int, ...]) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return available
    try:
        layers = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("--layers must be comma-separated integers or 'all'") from error
    missing = tuple(layer for layer in layers if layer not in available)
    if not layers or missing:
        raise ValueError(f"requested unavailable layers {missing}; available: {available}")
    return layers


if __name__ == "__main__":
    main()
