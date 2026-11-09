#!/usr/bin/env python3
"""Fit GPT-2 FastICA at several activation budgets and retain iteration diagnostics.

Each budget is an independent SLURM task, so run.py --budget 1k can be launched
on a separate GPU for every budget. The requested N is the number of GPT-2 token
activation rows used by both FastICA and each iteration's kurtosis diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens._artifact import LayerArtifact
from icalens._fastica import OBJECTIVE_PERCENTILES, fit_fastica

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTIVATIONS = ROOT / "experiments/ica-fitting-sample-size/work/activations/gpt2-pile10k-layer11-1m"
BUDGETS = {"1k": 1_000, "5k": 5_000, "10k": 10_000, "100k": 100_000, "1m": 1_000_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/ica-fitting-sample-size/runs")
    parser.add_argument("--budget", choices=[*BUDGETS, "all"], required=True)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def component_moments(data: torch.Tensor, unmixing: torch.Tensor, whitening: torch.Tensor,
                      center: torch.Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population skewness and excess kurtosis of current whitened ICA sources."""
    count = 0
    sums = [torch.zeros(unmixing.shape[0], dtype=torch.float64, device=unmixing.device) for _ in range(4)]
    logcosh_sum = torch.zeros_like(sums[0])
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size].to(device=unmixing.device, dtype=unmixing.dtype)
            source = unmixing @ (whitening @ (batch - center).T)
            absolute = source.abs()
            logcosh_sum += (absolute + torch.nn.functional.softplus(-2 * absolute) - np.log(2)).double().sum(dim=1)
            for order, total in enumerate(sums, start=1):
                total.add_(source.to(torch.float64).pow(order).sum(dim=1))
            count += source.shape[1]
    mean = sums[0] / count
    m2 = sums[1] / count - mean.square()
    m3 = sums[2] / count - 3 * mean * (sums[1] / count) + 2 * mean.pow(3)
    m4 = sums[3] / count - 4 * mean * (sums[2] / count) + 6 * mean.square() * (sums[1] / count) - 3 * mean.pow(4)
    variance = m2.clamp_min(1e-30)
    skew = m3 / variance.pow(1.5)
    kurtosis = m4 / variance.square() - 3
    return skew.cpu().numpy(), kurtosis.cpu().numpy(), (logcosh_sum / count).cpu().numpy()


def run_one(args: argparse.Namespace, dataset: ActivationDataset, name: str, budget: int) -> None:
    if budget > dataset.sample_count:
        raise ValueError(f"requested {budget:,} rows but dataset has only {dataset.sample_count:,}; capture a larger GPT-2 activation dataset first")
    if args.layer not in dataset.available_layers:
        raise ValueError(f"layer {args.layer} unavailable; available layers: {dataset.available_layers}")
    target = args.output / name
    target.mkdir(parents=True, exist_ok=True)
    trace_path = target / "trajectory.npz"
    if (target / "complete.json").exists():
        print(f"{name}: complete; skipping", flush=True)
        return
    # Use nested, uniformly shuffled token subsets so increasing N does not
    # merely add later documents from a differently distributed corpus prefix.
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(dataset.sample_count, generator=generator)[:budget]
    np.save(target / "fit-row-indices.npy", indices.numpy())
    values = dataset.layer(args.layer)[indices]
    # The experimental budget is a count of token-position activation rows.
    # Measure every intermediate direction on that same fit input, not on a
    # smaller auxiliary sample, so both the fit and plotted statistic match N.
    diagnostic = values
    rows: list[dict[str, object]] = []
    previous_unmixing: torch.Tensor | None = None
    checkpoints = target / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    (target / "run.json").write_text(json.dumps({
        "model": dataset.model, "activation_manifest_sha256": dataset.provenance["activation_dataset"]["manifest_sha256"],
        "fitting_tokens": budget, "statistic_tokens": budget, "layer": args.layer,
        "components": args.components or dataset.hidden_size, "steps": args.steps,
        "seed": args.seed, "sampling": "nested seeded permutation", "preprocessing": "none",
        "input_dtype": str(values.dtype), "fit_dtype": "float32", "tf32": False,
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "checkpoint_component_order": "persistent optimization row before final contrast sorting",
    }, indent=2) + "\n")

    def record(step: int, unmixing: torch.Tensor, whitening: torch.Tensor, center: torch.Tensor) -> None:
        nonlocal previous_unmixing
        skew, kurt, contrast = component_moments(diagnostic, unmixing, whitening, center, args.batch_size)
        limit = 0.0 if previous_unmixing is None else float((1 - (unmixing * previous_unmixing).sum(dim=1).abs()).abs().max())
        previous_unmixing = unmixing.detach().clone()
        orthogonality_error = float((unmixing @ unmixing.T - torch.eye(len(unmixing), device=unmixing.device)).abs().max())
        save_file({"unmixing": unmixing.detach().cpu().contiguous()}, str(checkpoints / f"step-{step:03d}.safetensors"))
        if step == 0:
            save_file({"whitening": whitening.detach().cpu().contiguous(), "center": center.detach().cpu().contiguous()}, str(target / "preprocessing.safetensors"))
        rows.append({"step": step, "median_abs_skewness": float(np.median(np.abs(skew))),
                     "median_excess_kurtosis": float(np.median(kurt)),
                     "mean_excess_kurtosis": float(np.mean(kurt)),
                     "q25_excess_kurtosis": float(np.quantile(kurt, .25)),
                     "q75_excess_kurtosis": float(np.quantile(kurt, .75)),
                     "component_skewness": skew, "component_excess_kurtosis": kurt,
                     "component_logcosh": contrast, "elapsed_seconds": time.time() - started,
                     "update_limit": limit, "orthogonality_error": orthogonality_error})
        temporary = trace_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, steps=np.array([r["step"] for r in rows]),
                            median_abs_skewness=np.array([r["median_abs_skewness"] for r in rows]),
                            median_excess_kurtosis=np.array([r["median_excess_kurtosis"] for r in rows]),
                            mean_excess_kurtosis=np.array([r["mean_excess_kurtosis"] for r in rows]),
                            q25_excess_kurtosis=np.array([r["q25_excess_kurtosis"] for r in rows]),
                            q75_excess_kurtosis=np.array([r["q75_excess_kurtosis"] for r in rows]),
                            component_skewness=np.stack([r["component_skewness"] for r in rows]),
                            component_logcosh=np.stack([r["component_logcosh"] for r in rows]),
                            elapsed_seconds=np.array([r["elapsed_seconds"] for r in rows]),
                            update_limit=np.array([r["update_limit"] for r in rows]),
                            orthogonality_error=np.array([r["orthogonality_error"] for r in rows]),
                            component_excess_kurtosis=np.stack([r["component_excess_kurtosis"] for r in rows]))
        os.replace(temporary, trace_path)
        print(f"{name}: step {step}/{args.steps}, median excess kurtosis={np.median(kurt):.4g}", flush=True)

    started = time.time()
    result = fit_fastica(values, n_components=args.components or dataset.hidden_size,
                         algorithm="parallel", fun="logcosh", max_iter=args.steps,
                         random_state=args.seed, progress=True, device="cuda",
                         batch_size=args.batch_size, row_normalize=False,
                         objective_every=1, checkpoint_callback=record)
    # Persist a native ICA Lens, using the same artifact conventions as ICALens.fit.
    lens = ICALens(model_id=dataset.model["repo_id"], model_revision=dataset.model["revision"],
                   model_type=dataset.model.get("type", "base"), activation_site="resid_post",
                   layer_indexing="transformer_blocks_zero_based", icalens_preprocessing="none")
    lens._hidden_size = dataset.hidden_size
    lens._layers[args.layer] = LayerArtifact(
        layer=args.layer, file=f"artifacts/resid_post/layer_{args.layer:02d}.safetensors",
        n_components=result.components.shape[0],
        center=result.center.detach().float().cpu().numpy(),
        reading_matrix=result.components.detach().float().cpu().numpy(),
        writing_matrix=result.mixing.detach().float().cpu().numpy(),
        fitting={"algorithm": "fastica", "implementation": "icalens.torch", "fun": "logcosh",
                 "n_samples": budget, "max_iter": args.steps, "n_iter": result.n_iter,
                 "random_state": args.seed, "whiten": "unit-variance", "source_scaling": "none",
                 "stopping_criterion": "fixed_iterations", "icalens_preprocessing": "none",
                 "component_objectives": result.component_objectives,
                 "component_strengths": result.component_strengths,
                 "gaussian_objective": result.gaussian_objective,
                 "objective_history": {"contrast": "logcosh", "iterations": result.objective_iterations,
                     "percentiles": list(OBJECTIVE_PERCENTILES), "values": result.objective_history},
                 "provenance": {"activation_dataset": dataset.provenance, "fitting_tokens": budget}})
    lens.save(target / "lens")
    (target / "complete.json").write_text(json.dumps({"budget": budget, "layer": args.layer,
        "steps_requested": args.steps, "steps_completed": result.n_iter,
        "diagnostic_samples": len(diagnostic), "activation_manifest": dataset.provenance["activation_dataset"]["manifest_sha256"],
        "seed": args.seed, "sampling": "nested seeded permutation of the common activation pool",
        "elapsed_seconds": time.time()-started}, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("run on a GPU allocation")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = ActivationDataset(args.activations)
    components = args.components or dataset.hidden_size
    names = list(BUDGETS) if args.budget == "all" else [args.budget]
    if any(BUDGETS[name] <= components for name in names):
        raise ValueError(f"{components} components require at least {components + 1} token samples after centering")
    for name in names:
        run_one(args, dataset, name, BUDGETS[name])


if __name__ == "__main__":
    main()
