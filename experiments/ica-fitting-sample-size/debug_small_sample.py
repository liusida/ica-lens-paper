#!/usr/bin/env python3
"""Independent metric, precision, dimensionality, and Gaussian controls for 1K."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
from safetensors.torch import load_file

from icalens._activation_dataset import ActivationDataset
from icalens._fastica import fit_fastica

HERE = Path(__file__).resolve().parent
OUT = HERE / "work/debug-small-sample"


def statistics(z):
    s = z.double() - z.double().mean(0)
    square = s.square()
    energy = square.sum(0)
    fourth = square.square().sum(0)
    k = len(z) * fourth / energy.square() - 3
    return {
        "median_excess_kurtosis": float(torch.quantile(k, .5)),
        "median_top1_energy_share": float(torch.quantile(square.max(0).values / energy, .5)),
        "median_top1_fourth_moment_share": float(torch.quantile(square.max(0).values.square() / fourth, .5)),
        "median_variance": float(torch.quantile(square.mean(0), .5)),
    }, k


def evaluate(values, heldout, result):
    train, _ = statistics((values.to(result.center.dtype) - result.center) @ result.components.T)
    test, _ = statistics((heldout.to(result.center.dtype) - result.center) @ result.components.T)
    return {"training": train, "unseen_tokens": test, "n_iter": result.n_iter}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = ActivationDataset(HERE / "work/activations/gpt2-pile10k-layer11-1m").layer(11)
    rows = np.load(HERE / "work/runs/1k/fit-row-indices.npy")
    perm = np.load(HERE / "work/runs/1m/fit-row-indices.npy")
    x = data[torch.from_numpy(rows)].cuda().float()
    # 10K is enough for controls; the prior full final-state evaluation used
    # 100K disjoint positions and is retained separately.
    heldout = data[torch.from_numpy(perm[100_000:110_000].copy())].cuda().float()
    saved = load_file(HERE / "work/runs/1k/preprocessing.safetensors", device="cuda")
    w = load_file(HERE / "work/runs/1k/checkpoints/step-050.safetensors", device="cuda")["unmixing"]
    white = (x - saved["center"]) @ saved["whitening"].T
    z = white @ w.T
    summary, k = statistics(z)
    cpu = z.cpu().double().numpy()
    direct = np.mean((cpu - cpu.mean(0))**4, axis=0) / np.mean((cpu - cpu.mean(0))**2, axis=0)**2 - 3
    scipy_k = scipy_kurtosis(cpu, axis=0, fisher=True, bias=True)
    reading64 = w.double() @ saved["whitening"].double()
    z64 = (x.double() - saved["center"].double()) @ reading64.T
    _, k64 = statistics(z64)
    record = {
        "saved_1k": summary,
        "metric_checks": {
            "max_abs_torch_vs_numpy": float(np.max(np.abs(k.cpu().numpy() - direct))),
            "max_abs_torch_vs_scipy": float(np.max(np.abs(k.cpu().numpy() - scipy_k))),
            "median_kurtosis_fp64_projection": float(torch.quantile(k64, .5)),
            "max_abs_kurtosis_fp32_vs_fp64_projection": float((k - k64).abs().max()),
            "max_abs_whitened_covariance_minus_identity": float((white.T @ white / len(x) - torch.eye(768, device="cuda")).abs().max()),
            "max_abs_unmixing_orthogonality_error": float((w @ w.T - torch.eye(768, device="cuda")).abs().max()),
            "max_abs_source_variance_minus_one": float((z.double().var(0, correction=0) - 1).abs().max()),
            "max_abs_source_mean": float(z.double().mean(0).abs().max()),
        },
        "whitening_without_ica": statistics(white)[0],
        "gpt2_component_sweep": {}, "gaussian_controls": {},
        "control_heldout_tokens": len(heldout),
        "control_heldout_definition": "token-disjoint, not document-disjoint",
    }
    print("metric_checks", record["metric_checks"], flush=True)
    selected = int((k - torch.quantile(k, .5)).abs().argmin())
    train_s = z[:, selected].double()
    train_s = (train_s - train_s.mean()) / train_s.std(correction=0)
    eval_z = ((heldout - saved["center"]) @ reading64.float().T)[:, selected].double()
    eval_z = (eval_z - eval_z.mean()) / eval_z.std(correction=0)
    np.savez_compressed(OUT / "example-component.npz", training=train_s.cpu().numpy(),
                        unseen_tokens=eval_z[:1000].cpu().numpy(),
                        component_index=selected, excess_kurtosis=float(k[selected]))

    for components in (64, 128, 256, 512, 768):
        result = fit_fastica(x, n_components=components, max_iter=50,
            random_state=0, row_normalize=False, device="cuda", batch_size=8192,
            objective_every=50)
        record["gpt2_component_sweep"][str(components)] = evaluate(x, heldout, result)
        print("GPT2", components, record["gpt2_component_sweep"][str(components)], flush=True)

    # Pure Gaussian data have no population non-Gaussian signal to recover.
    # Independent draws and the same N/d isolate finite-sample optimization.
    for seed in (0, 1, 2):
        g = torch.Generator(device="cuda").manual_seed(100 + seed)
        gx = torch.randn((1000, 768), generator=g, device="cuda")
        gh = torch.randn((10_000, 768), generator=g, device="cuda")
        result = fit_fastica(gx, n_components=768, max_iter=50,
            random_state=seed, row_normalize=False, device="cuda", batch_size=8192,
            objective_every=50)
        record["gaussian_controls"][str(seed)] = evaluate(gx, gh, result)
        print("Gaussian", seed, record["gaussian_controls"][str(seed)], flush=True)

    result = fit_fastica(x.double(), n_components=768, max_iter=50,
        random_state=0, row_normalize=False, device="cuda", batch_size=8192,
        objective_every=50)
    record["gpt2_float64_fit"] = evaluate(x, heldout, result)
    print("FP64", record["gpt2_float64_fit"], flush=True)

    # An independent implementation and SVD whitening rule out a bug specific
    # to the project FastICA loop or its covariance-eigendecomposition path.
    with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
        estimator = FastICA(n_components=768, algorithm="parallel", fun="logcosh",
            max_iter=50, tol=1e-12, whiten="unit-variance", whiten_solver="svd", random_state=0)
        train = estimator.fit_transform(x.cpu().double().numpy())
        test = estimator.transform(heldout.cpu().double().numpy())
    record["sklearn_svd_reference"] = {
        "training_median_excess_kurtosis": float(np.median(scipy_kurtosis(train, axis=0, fisher=True, bias=True))),
        "unseen_median_excess_kurtosis": float(np.median(scipy_kurtosis(test, axis=0, fisher=True, bias=True))),
        "n_iter": int(estimator.n_iter_), "warnings": [str(item.message) for item in caught]}
    print("sklearn", record["sklearn_svd_reference"], flush=True)
    (OUT / "report.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
