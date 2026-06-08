from __future__ import annotations

import inspect
import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from .paths import V6_ROOT


VENDOR_FASTICA_SRC = V6_ROOT / "vendor" / "FastICA_torch" / "src"
if str(VENDOR_FASTICA_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_FASTICA_SRC))

from fastica_torch import FastICA
import fastica_torch.fastica as fastica_module


ProgressStatsCallback = Callable[[int, dict[str, float], float], None]
LimVecCallback = Callable[[int, torch.Tensor], None]


def normalize_rows_available() -> bool:
    return True


@contextmanager
def _fastica_progress_callback(
    callback: ProgressStatsCallback | None,
    started_at: float,
    lim_vec_callback: LimVecCallback | None = None,
) -> Iterator[None]:
    if callback is None:
        yield
        return

    original_ica_par = fastica_module._ica_par
    original_ica_def = fastica_module._ica_def

    def wrapped_ica_par(*args: Any, **kwargs: Any) -> Any:
        return _ica_par_with_progress_stats(
            *args,
            progress_callback=callback,
            started_at=started_at,
            lim_vec_callback=lim_vec_callback,
            **kwargs,
        )

    def wrapped_ica_def(*args: Any, **kwargs: Any) -> Any:
        return _ica_def_with_progress_stats(
            *args,
            progress_callback=callback,
            started_at=started_at,
            lim_vec_callback=lim_vec_callback,
            **kwargs,
        )

    fastica_module._ica_par = wrapped_ica_par
    fastica_module._ica_def = wrapped_ica_def
    try:
        yield
    finally:
        fastica_module._ica_par = original_ica_par
        fastica_module._ica_def = original_ica_def


def _ica_par_with_progress_stats(
    X: torch.Tensor,
    tol: float,
    g: Callable[..., Any],
    fun_args: dict[str, Any],
    max_iter: int,
    w_init: torch.Tensor,
    progress: bool = False,
    lim_history: list[float] | None = None,
    *,
    progress_callback: ProgressStatsCallback,
    started_at: float,
    lim_vec_callback: LimVecCallback | None = None,
) -> tuple[torch.Tensor, int]:
    W = fastica_module._sym_decorrelation(w_init)
    ii = 0
    iterator = range(max_iter)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, total=max_iter, desc="FastICA parallel", unit="iter", dynamic_ncols=True)

    for ii in iterator:
        wtx = W @ X
        gwtx, g_wtx = g(wtx, fun_args)
        term1 = (gwtx @ X.T) / X.shape[1]
        term2 = g_wtx.unsqueeze(1) * W
        W1 = fastica_module._sym_decorrelation(term1 - term2)

        dot_products = torch.sum(W1 * W, dim=1)
        lim_vec = torch.abs(torch.abs(dot_products) - 1)
        lim = torch.max(lim_vec)
        if lim_history is not None:
            lim_history.append(float(lim.detach().cpu()))
        if lim_vec_callback is not None:
            lim_vec_callback(ii + 1, lim_vec)
        progress_callback(ii + 1, _lim_vec_stats(lim_vec), time.time() - started_at)
        if progress:
            iterator.set_postfix(max=f"{lim.item():.2e}", p95=f"{torch.quantile(lim_vec.float(), 0.95).item():.2e}")
        if not torch.isfinite(lim) or lim.item() > 1e20:
            raise RuntimeError(f"FastICA diverged at iter {ii}: lim={lim.item():.3e}")

        W = W1
        if lim < tol:
            break

    return W, ii + 1


def _ica_def_with_progress_stats(
    X: torch.Tensor,
    tol: float,
    g: Callable[..., Any],
    fun_args: dict[str, Any],
    max_iter: int,
    w_init: torch.Tensor,
    progress: bool = False,
    *,
    progress_callback: ProgressStatsCallback,
    started_at: float,
    lim_vec_callback: LimVecCallback | None = None,
) -> tuple[torch.Tensor, int]:
    n_components = int(w_init.shape[0])
    W = torch.zeros((n_components, n_components), dtype=X.dtype, device=X.device)
    n_iter: list[int] = []
    final_lims: list[torch.Tensor] = []
    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(total=n_components * max_iter, desc="FastICA deflation", unit="iter", dynamic_ncols=True)

    try:
        for component_index in range(n_components):
            w = w_init[component_index, :].clone()
            w /= torch.sqrt((w**2).sum())
            lim = torch.tensor(float("inf"), dtype=X.dtype, device=X.device)
            ii = 0
            for ii in range(max_iter):
                wtx = w @ X
                gx, g_wtx = g(wtx, fun_args)
                w1 = (X * gx.unsqueeze(0)).mean(dim=1) - g_wtx.mean() * w
                w1 = fastica_module._gs_decorrelation(w1, W, component_index)
                w1 /= torch.sqrt((w1**2).sum())
                lim = torch.abs(torch.abs((w1 * w).sum()) - 1)
                w = w1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(component=component_index, lim=f"{lim.item():.2e}")
                if not torch.isfinite(lim) or lim.item() > 1e20:
                    raise RuntimeError(
                        f"FastICA deflation diverged at component {component_index}, iter {ii}: lim={lim.item():.3e}"
                    )
                if lim < tol:
                    if progress_bar is not None:
                        progress_bar.update(max_iter - ii - 1)
                    break

            n_iter.append(ii + 1)
            W[component_index, :] = w
            final_lims.append(lim.detach())
            lim_vec = torch.stack(final_lims)
            completed = component_index + 1
            if lim_vec_callback is not None:
                lim_vec_callback(completed, lim_vec)
            lim_stats = _lim_vec_stats(lim_vec)
            lim_stats.update(
                {
                    "component_index": int(component_index),
                    "completed_components": int(completed),
                    "component_iterations": int(ii + 1),
                    "component_lim": float(lim.detach().cpu().item()),
                }
            )
            progress_callback(completed, lim_stats, time.time() - started_at)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return W, max(n_iter) if n_iter else 0


def _lim_vec_stats(lim_vec: torch.Tensor) -> dict[str, float]:
    values = lim_vec.detach().to(device="cpu", dtype=torch.float64)
    return {
        "max": float(torch.max(values).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "min": float(torch.min(values).item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "median": float(torch.median(values).item()),
        "mean": float(torch.mean(values).item()),
        "std": float(torch.std(values, unbiased=False).item()),
    }


def normalize_rows(values: torch.Tensor, *, norm_eps: float) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(norm_eps)


def fit_fastica(
    activations: torch.Tensor,
    *,
    n_components: int,
    seed: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    fun: str,
    norm_eps: float,
    progress: bool,
    progress_callback: ProgressStatsCallback | None = None,
    lim_vec_callback: LimVecCallback | None = None,
) -> dict[str, Any]:
    return _fit_fastica_preprocessed(
        activations,
        preprocess="with_normalization",
        n_components=n_components,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        fun=fun,
        norm_eps=norm_eps,
        progress=progress,
        progress_callback=progress_callback,
        lim_vec_callback=lim_vec_callback,
    )


def fit_fastica_without_normalization(
    activations: torch.Tensor,
    *,
    n_components: int,
    seed: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    fun: str,
    norm_eps: float,
    progress: bool,
    progress_callback: ProgressStatsCallback | None = None,
    lim_vec_callback: LimVecCallback | None = None,
) -> dict[str, Any]:
    return _fit_fastica_preprocessed(
        activations,
        preprocess="without_normalization",
        n_components=n_components,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        fun=fun,
        norm_eps=norm_eps,
        progress=progress,
        progress_callback=progress_callback,
        lim_vec_callback=lim_vec_callback,
    )


def _fit_fastica_preprocessed(
    activations: torch.Tensor,
    *,
    preprocess: str,
    n_components: int,
    seed: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    fun: str,
    norm_eps: float,
    progress: bool,
    progress_callback: ProgressStatsCallback | None,
    lim_vec_callback: LimVecCallback | None,
) -> dict[str, Any]:
    started_at = time.time()
    lim_stats_history: list[dict[str, float | int]] = []

    def record_progress(iteration: int, lim_stats: dict[str, float], elapsed_seconds: float) -> None:
        lim_stats_history.append({"iteration": int(iteration), "elapsed_seconds": round(float(elapsed_seconds), 6), **lim_stats})
        if progress_callback is not None:
            progress_callback(iteration, lim_stats, elapsed_seconds)

    def record_lim_vec(iteration: int, lim_vec: torch.Tensor) -> None:
        if lim_vec_callback is not None:
            lim_vec_callback(iteration, lim_vec)

    if preprocess == "with_normalization":
        x = normalize_rows(activations, norm_eps=norm_eps)
    elif preprocess == "without_normalization":
        x = activations
    else:
        raise ValueError(f"Unsupported FastICA preprocess mode: {preprocess!r}")

    init_kwargs: dict[str, Any] = {
        "n_components": n_components,
        "algorithm": algorithm,
        "whiten": "unit-variance",
        "whiten_solver": "eigh",
        "fun": fun,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": seed,
        "float64_covariance": True,
    }
    if "progress" in inspect.signature(FastICA).parameters:
        init_kwargs["progress"] = progress
    ica = FastICA(**init_kwargs)
    with _fastica_progress_callback(record_progress, started_at, lim_vec_callback=record_lim_vec):
        _sources = ica.fit_transform(x)

    components = ica.components_.detach().clone()
    mean = ica.mean_.detach().clone().reshape(1, -1)
    whitening = ica.whitening_.detach().clone()
    unmixing = ica._unmixing.detach().clone()
    directions = unmixing / torch.linalg.vector_norm(unmixing, dim=1, keepdim=True).clamp_min(norm_eps)
    iterations = int(ica.n_iter_)
    lim_history = list(getattr(ica, "lim_history_", []))
    final_lim = float(lim_history[-1]) if lim_history else None
    if final_lim is None and lim_stats_history:
        final_lim = float(lim_stats_history[-1]["max"])
    return {
        "method": "fastica",
        "algorithm": algorithm,
        "preprocess": preprocess,
        "mean": mean.cpu(),
        "whitening": whitening.cpu(),
        "components": components.cpu(),
        "unmixing": unmixing.cpu(),
        "directions": directions.cpu(),
        "rows": int(activations.shape[0]),
        "hidden_size": int(activations.shape[1]),
        "n_components": int(n_components),
        "iterations": iterations,
        "converged": bool(iterations < max_iter),
        "lim_history": lim_history,
        "lim_stats_history": lim_stats_history,
        "final_lim": final_lim,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "fun": fun,
        "seed": int(seed),
        "norm_eps": float(norm_eps),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }

def save_ica_artifact(path: Path, artifact: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: value for key, value in artifact.items() if isinstance(value, torch.Tensor)}
    summary = {key: value for key, value in artifact.items() if not isinstance(value, torch.Tensor)}
    full_metadata = {**metadata, **summary}
    torch.save({"tensors": tensors, "metadata": full_metadata}, path)
    path.with_suffix(".json").write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
