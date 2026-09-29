"""Blockwise PyTorch implementation of the FastICA algorithm.

The implementation follows Hyvarinen's fixed-point algorithm and is adapted
from FastICA_torch by Richard Hakim, which is distributed under the MIT
License. ICA Lens keeps the implementation private so its public artifact
format is independent of the fitting backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import cast

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class FastICAResult:
    center: torch.Tensor
    components: torch.Tensor
    mixing: torch.Tensor
    n_iter: int
    objective_history: list[list[float]] | None
    objective_iterations: list[int] | None
    component_objectives: list[float]
    component_strengths: list[float]
    gaussian_objective: float


def geometric_median(
    values: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 8192,
    max_iter: int = 50,
    tol: float = 1e-5,
    eps: float = 1e-12,
    progress: bool = False,
) -> torch.Tensor:
    """Estimate the geometric median with blockwise Weiszfeld iterations."""
    if values.ndim != 2 or not values.is_floating_point():
        raise TypeError("geometric median input must be a two-dimensional floating tensor")
    if batch_size <= 0 or max_iter <= 0 or tol <= 0 or eps <= 0:
        raise ValueError("geometric median controls must be positive")
    target = values.device if device is None else torch.device(device)
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    total = torch.zeros(values.shape[1], device=target, dtype=torch.float64)
    count = 0
    for start in range(0, int(values.shape[0]), batch_size):
        batch = values[start : start + batch_size].to(device=target, dtype=dtype)
        if not bool(torch.isfinite(batch).all()):
            raise ValueError("geometric median input contains non-finite values")
        total += batch.to(torch.float64).sum(dim=0)
        count += int(batch.shape[0])
    estimate = (total / count).to(dtype)
    iterations = tqdm(
        range(max_iter),
        desc="Geometric median",
        unit="iter",
        dynamic_ncols=True,
        disable=not progress,
    )
    for _ in iterations:
        weighted = torch.zeros_like(total)
        weight_sum = torch.zeros((), device=target, dtype=torch.float64)
        for start in range(0, int(values.shape[0]), batch_size):
            batch = values[start : start + batch_size].to(device=target, dtype=dtype)
            distances = torch.linalg.vector_norm(batch - estimate, dim=1).clamp_min(eps)
            weights = distances.reciprocal().to(torch.float64)
            weighted += (batch.to(torch.float64) * weights[:, None]).sum(dim=0)
            weight_sum += weights.sum()
        updated = (weighted / weight_sum).to(dtype)
        change = torch.linalg.vector_norm(updated - estimate)
        scale = torch.linalg.vector_norm(estimate).clamp_min(1.0)
        estimate = updated
        if progress:
            iterations.set_postfix(change=f"{float(change):.2e}")
        if float(change / scale) <= tol:
            break
    return estimate


OBJECTIVE_PERCENTILES = tuple(range(0, 101, 10))
GAUSSIAN_OBJECTIVES = {
    "logcosh": 0.374567207491457,
    "exp": -0.7071067811865476,
    "cube": 0.75,
}


def fit_fastica(
    values: torch.Tensor,
    *,
    n_components: int,
    algorithm: str = "parallel",
    fun: str = "logcosh",
    max_iter: int = 200,
    random_state: int | None = 0,
    progress: bool = False,
    device: str | torch.device | None = None,
    batch_size: int = 8192,
    row_normalize: bool = True,
    preprocessing_center: torch.Tensor | None = None,
    norm_eps: float = 1e-12,
    objective_every: int = 1,
    checkpoint_callback: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], None]
    | None = None,
    initial_unmixing: torch.Tensor | None = None,
    start_iteration: int = 0,
) -> FastICAResult:
    """Fit unit-variance-whitened FastICA without moving all samples to CUDA."""
    if values.ndim != 2:
        raise ValueError("FastICA input must be two-dimensional")
    n_samples, n_features = map(int, values.shape)
    if not 0 < n_components <= min(n_samples, n_features):
        raise ValueError("n_components exceeds the input dimensions")
    if algorithm not in {"parallel", "deflation"}:
        raise ValueError("algorithm must be 'parallel' or 'deflation'")
    if fun not in {"logcosh", "exp", "cube"}:
        raise ValueError("fun must be 'logcosh', 'exp', or 'cube'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if objective_every <= 0:
        raise ValueError("objective_every must be positive")
    if start_iteration < 0 or start_iteration >= max_iter:
        raise ValueError("start_iteration must be non-negative and smaller than max_iter")
    if start_iteration > 0 and initial_unmixing is None:
        raise ValueError("initial_unmixing is required when start_iteration is positive")
    if checkpoint_callback is not None and algorithm != "parallel":
        raise ValueError("checkpoint_callback is supported only by parallel FastICA")
    if initial_unmixing is not None and algorithm != "parallel":
        raise ValueError("initial_unmixing is supported only by parallel FastICA")

    fit_device = values.device if device is None else torch.device(device)
    fit_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    center = _mean(
        values,
        device=fit_device,
        dtype=fit_dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        preprocessing_center=preprocessing_center,
        norm_eps=norm_eps,
        progress=progress,
    )
    covariance = _covariance(
        values,
        center=center,
        device=fit_device,
        dtype=fit_dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        preprocessing_center=preprocessing_center,
        norm_eps=norm_eps,
        progress=progress,
    )
    if progress:
        tqdm.write(f"FastICA whitening: eigh on {n_features}x{n_features} covariance...")
    whitening_start = perf_counter()
    whitening = _whitening_matrix(
        covariance,
        n_samples=n_samples,
        n_components=n_components,
        dtype=fit_dtype,
    )
    if progress:
        tqdm.write(f"FastICA whitening: eigh completed in {perf_counter() - whitening_start:.1f}s")

    generator = torch.Generator(device=fit_device)
    if random_state is not None:
        generator.manual_seed(random_state)
    initial = torch.randn(
        (n_components, n_components),
        dtype=fit_dtype,
        device=fit_device,
        generator=generator,
    )
    nonlinearity = _nonlinearity(fun)
    objective = _contrast(fun)
    batch_kwargs = {
        "center": center,
        "whitening": whitening,
        "device": fit_device,
        "dtype": fit_dtype,
        "batch_size": batch_size,
        "row_normalize": row_normalize,
        "preprocessing_center": preprocessing_center,
        "norm_eps": norm_eps,
    }
    if algorithm == "parallel":

        def emit_checkpoint(iteration: int, unmixing: torch.Tensor) -> None:
            if checkpoint_callback is not None:
                checkpoint_callback(iteration, unmixing, whitening, center)

        unmixing, n_iter, objective_iterations, objective_history = _fit_parallel(
            values,
            initial,
            nonlinearity,
            objective,
            max_iter=max_iter,
            objective_every=objective_every,
            progress=progress,
            batch_kwargs=batch_kwargs,
            checkpoint_callback=emit_checkpoint if checkpoint_callback is not None else None,
            initial_unmixing=initial_unmixing,
            start_iteration=start_iteration,
        )
    else:
        unmixing, n_iter = _fit_deflation(
            values,
            initial,
            nonlinearity,
            objective,
            max_iter=max_iter,
            progress=progress,
            batch_kwargs=batch_kwargs,
        )
        objective_history = None
        objective_iterations = None

    final_objectives = _component_objectives(
        values,
        unmixing=unmixing,
        objective=objective,
        progress=progress,
        batch_kwargs=batch_kwargs,
    )
    gaussian_objective = GAUSSIAN_OBJECTIVES[fun]
    strengths = torch.abs(final_objectives - gaussian_objective)
    order = torch.argsort(strengths, descending=True, stable=True)
    unmixing = unmixing.index_select(0, order)
    final_objectives = final_objectives.index_select(0, order)
    strengths = strengths.index_select(0, order)

    if progress:
        tqdm.write("FastICA finalization: ordering components and computing matrices...")
    finalization_start = perf_counter()
    components = unmixing @ whitening
    mixing = torch.linalg.pinv(components)
    if progress:
        tqdm.write(f"FastICA finalization: completed in {perf_counter() - finalization_start:.1f}s")
    return FastICAResult(
        center=center,
        components=components,
        mixing=mixing,
        n_iter=n_iter,
        objective_history=objective_history,
        objective_iterations=objective_iterations,
        component_objectives=[float(value) for value in final_objectives.detach().cpu().tolist()],
        component_strengths=[float(value) for value in strengths.detach().cpu().tolist()],
        gaussian_objective=gaussian_objective,
    )


def _component_objectives(
    source: torch.Tensor,
    *,
    unmixing: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    progress: bool,
    batch_kwargs: dict[str, object],
) -> torch.Tensor:
    """Evaluate the final contrast of every component in a blockwise pass."""
    objective_sum = torch.zeros(unmixing.shape[0], dtype=unmixing.dtype, device=unmixing.device)
    progress_bar = tqdm(
        total=int(source.shape[0]),
        desc="FastICA component objectives",
        unit="sample",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        for whitened in _whitened_batches(source, **batch_kwargs):
            batch_count = int(whitened.shape[1])
            objective_sum += objective(unmixing @ whitened) * batch_count
            progress_bar.update(batch_count)
    finally:
        progress_bar.close()
    return objective_sum / int(source.shape[0])


def _mean(
    values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    preprocessing_center: torch.Tensor | None,
    norm_eps: float,
    progress: bool,
) -> torch.Tensor:
    total = torch.zeros(values.shape[1], dtype=torch.float64, device=device)
    count = 0
    progress_bar = tqdm(
        total=int(values.shape[0]),
        desc="FastICA mean",
        unit="sample",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        for batch in _preprocessed_batches(
            values,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            row_normalize=row_normalize,
            preprocessing_center=preprocessing_center,
            norm_eps=norm_eps,
        ):
            total += batch.to(torch.float64).sum(dim=0)
            batch_count = int(batch.shape[0])
            count += batch_count
            progress_bar.update(batch_count)
    finally:
        progress_bar.close()
    return (total / count).to(dtype)


def _covariance(
    values: torch.Tensor,
    *,
    center: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    preprocessing_center: torch.Tensor | None,
    norm_eps: float,
    progress: bool,
) -> torch.Tensor:
    feature_count = int(values.shape[1])
    covariance = torch.zeros((feature_count, feature_count), dtype=torch.float64, device=device)
    progress_bar = tqdm(
        total=int(values.shape[0]),
        desc="FastICA covariance",
        unit="sample",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        for batch in _preprocessed_batches(
            values,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            row_normalize=row_normalize,
            preprocessing_center=preprocessing_center,
            norm_eps=norm_eps,
        ):
            centered64 = (batch - center).to(torch.float64)
            covariance.addmm_(centered64.T, centered64)
            progress_bar.update(int(batch.shape[0]))
    finally:
        progress_bar.close()
    return covariance


def _whitening_matrix(
    covariance: torch.Tensor,
    *,
    n_samples: int,
    n_components: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)[:n_components]
    eigenvectors = eigenvectors.flip(1)[:, :n_components]
    threshold = torch.finfo(torch.float64).eps * max(covariance.shape) * eigenvalues[0]
    if bool((eigenvalues <= threshold).any()):
        raise ValueError("input rank is smaller than n_components")
    signs = torch.where(eigenvectors[0] < 0, -1.0, 1.0)
    eigenvectors = eigenvectors * signs
    whitening = (eigenvectors / eigenvalues.sqrt()).T
    return cast(torch.Tensor, whitening.to(dtype=dtype) * n_samples**0.5)


def _fit_parallel(
    source: torch.Tensor,
    initial: torch.Tensor,
    nonlinearity: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    objective: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_iter: int,
    objective_every: int,
    progress: bool,
    batch_kwargs: dict[str, object],
    checkpoint_callback: Callable[[int, torch.Tensor], None] | None = None,
    initial_unmixing: torch.Tensor | None = None,
    start_iteration: int = 0,
) -> tuple[torch.Tensor, int, list[int], list[list[float]]]:
    if initial_unmixing is None:
        weights = _symmetric_decorrelation(initial)
        if checkpoint_callback is not None:
            checkpoint_callback(0, weights)
    else:
        if initial_unmixing.shape != initial.shape:
            raise ValueError(
                f"initial_unmixing has shape {tuple(initial_unmixing.shape)}, "
                f"expected {tuple(initial.shape)}"
            )
        weights = initial_unmixing.to(device=initial.device, dtype=initial.dtype)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("initial_unmixing contains non-finite values")
    n_samples = int(source.shape[0])
    iterations = tqdm(
        range(start_iteration, max_iter),
        desc="FastICA parallel",
        unit="iter",
        dynamic_ncols=True,
        disable=not progress,
    )
    objective_iterations: list[int] = []
    objective_history: list[list[float]] = []
    for iteration_index in iterations:
        iteration = iteration_index + 1
        record_objective = iteration % objective_every == 0 or iteration == max_iter
        term_sum = torch.zeros_like(weights)
        derivative_sum = torch.zeros(weights.shape[0], dtype=weights.dtype, device=weights.device)
        objective_sum = (
            torch.zeros(weights.shape[0], dtype=weights.dtype, device=weights.device)
            if record_objective
            else None
        )
        for whitened in _whitened_batches(source, **batch_kwargs):
            projected = weights @ whitened
            transformed, derivative_mean = nonlinearity(projected)
            batch_count = int(whitened.shape[1])
            term_sum.addmm_(transformed, whitened.T)
            derivative_sum += derivative_mean * batch_count
            if objective_sum is not None:
                objective_sum += objective(projected) * batch_count
        updated = term_sum / n_samples - (derivative_sum / n_samples)[:, None] * weights
        updated = _symmetric_decorrelation(updated)
        limit = torch.max(torch.abs(torch.abs(torch.sum(updated * weights, dim=1)) - 1.0))
        postfix = {"limit": f"{float(limit):.2e}"}
        if objective_sum is not None:
            component_objectives = objective_sum / n_samples
            quantiles = torch.quantile(
                component_objectives,
                torch.linspace(
                    0,
                    1,
                    len(OBJECTIVE_PERCENTILES),
                    dtype=component_objectives.dtype,
                    device=weights.device,
                ),
            )
            objective_iterations.append(iteration)
            objective_history.append([float(value) for value in quantiles])
            postfix["obj"] = f"{float(component_objectives.mean()):.2f}"
        iterations.set_postfix(postfix)
        weights = updated
        if checkpoint_callback is not None:
            checkpoint_callback(iteration, weights)
    return weights, max_iter, objective_iterations, objective_history


def _fit_deflation(
    source: torch.Tensor,
    initial: torch.Tensor,
    nonlinearity: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    objective: Callable[[torch.Tensor], torch.Tensor] | None,
    *,
    max_iter: int,
    progress: bool,
    batch_kwargs: dict[str, object],
) -> tuple[torch.Tensor, int]:
    count = int(initial.shape[0])
    n_samples = int(source.shape[0])
    weights = torch.zeros_like(initial)
    components = tqdm(
        range(count),
        desc="FastICA deflation",
        unit="component",
        dynamic_ncols=True,
        disable=not progress,
    )
    for component in components:
        weight = initial[component] / torch.linalg.vector_norm(initial[component])
        objective_value = torch.zeros((), dtype=weight.dtype, device=weight.device)
        for _ in range(max_iter):
            term_sum = torch.zeros_like(weight)
            derivative_sum = torch.zeros((), dtype=weight.dtype, device=weight.device)
            objective_value.zero_()
            for whitened in _whitened_batches(source, **batch_kwargs):
                projected = weight @ whitened
                transformed, derivative_mean = nonlinearity(projected[None, :])
                batch_count = int(whitened.shape[1])
                term_sum += whitened @ transformed[0]
                derivative_sum += derivative_mean[0] * batch_count
                if objective is not None:
                    objective_value += objective(projected[None, :])[0] * batch_count
            updated = term_sum / n_samples - (derivative_sum / n_samples) * weight
            if component:
                updated -= (updated @ weights[:component].T) @ weights[:component]
            updated /= torch.linalg.vector_norm(updated)
            limit = torch.abs(torch.abs(updated @ weight) - 1.0)
            weight = updated
        weights[component] = weight
        postfix = {"iterations": max_iter, "limit": f"{float(limit):.2e}"}
        if objective is not None:
            postfix["obj"] = f"{float(objective_value / n_samples):.2f}"
        components.set_postfix(postfix)
    return weights, max_iter


def _source_std(
    values: torch.Tensor,
    *,
    center: torch.Tensor,
    components: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    preprocessing_center: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    source_sum = torch.zeros(components.shape[0], dtype=torch.float64, device=device)
    source_square_sum = torch.zeros_like(source_sum)
    count = 0
    for batch in _preprocessed_batches(
        values,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        preprocessing_center=preprocessing_center,
        norm_eps=norm_eps,
    ):
        sources64 = ((batch - center) @ components.T).to(torch.float64)
        source_sum += sources64.sum(dim=0)
        source_square_sum += sources64.square().sum(dim=0)
        count += int(batch.shape[0])
    variance = source_square_sum / count - (source_sum / count).square()
    return variance.clamp_min(torch.finfo(torch.float64).eps).sqrt().to(dtype)


def _preprocessed_batches(
    values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    preprocessing_center: torch.Tensor | None,
    norm_eps: float,
) -> Iterator[torch.Tensor]:
    for start in range(0, int(values.shape[0]), batch_size):
        batch = values[start : start + batch_size].to(device=device, dtype=dtype)
        if not bool(torch.isfinite(batch).all()):
            raise ValueError("FastICA input contains non-finite values")
        if preprocessing_center is not None:
            batch = batch - preprocessing_center.to(device=device, dtype=dtype)
        if row_normalize:
            norms = torch.linalg.vector_norm(batch, dim=1, keepdim=True)
            batch = batch / norms.clamp_min(norm_eps)
        yield batch


def _whitened_batches(
    values: torch.Tensor,
    *,
    center: object,
    whitening: object,
    device: object,
    dtype: object,
    batch_size: object,
    row_normalize: object,
    preprocessing_center: object,
    norm_eps: object,
) -> Iterator[torch.Tensor]:
    assert isinstance(center, torch.Tensor)
    assert isinstance(whitening, torch.Tensor)
    assert isinstance(device, torch.device)
    assert isinstance(dtype, torch.dtype)
    assert isinstance(batch_size, int)
    assert isinstance(row_normalize, bool)
    assert preprocessing_center is None or isinstance(preprocessing_center, torch.Tensor)
    assert isinstance(norm_eps, float)
    for batch in _preprocessed_batches(
        values,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        preprocessing_center=preprocessing_center,
        norm_eps=norm_eps,
    ):
        yield whitening @ (batch - center).T


def _symmetric_decorrelation(weights: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(weights @ weights.T)
    floor = torch.finfo(weights.dtype).eps
    inverse_root = eigenvectors @ torch.diag(eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
    return cast(torch.Tensor, inverse_root @ weights)


def _nonlinearity(name: str) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if name == "logcosh":

        def logcosh(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            transformed = torch.tanh(values)
            return transformed, (1.0 - transformed.square()).mean(dim=1)

        return logcosh
    if name == "exp":

        def exp(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            exponential = torch.exp(-values.square() / 2.0)
            return values * exponential, ((1.0 - values.square()) * exponential).mean(dim=1)

        return exp

    def cube(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return values.pow(3), (3.0 * values.square()).mean(dim=1)

    return cube


def _contrast(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "logcosh":

        def logcosh(values: torch.Tensor) -> torch.Tensor:
            return (torch.logaddexp(values, -values) - 0.6931471805599453).mean(dim=1)

        return logcosh
    if name == "exp":

        def exp(values: torch.Tensor) -> torch.Tensor:
            return -torch.exp(-values.square() / 2.0).mean(dim=1)

        return exp

    def cube(values: torch.Tensor) -> torch.Tensor:
        return values.pow(4).mean(dim=1) / 4.0

    return cube
