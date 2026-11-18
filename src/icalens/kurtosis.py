"""Excess kurtosis of ICA Lens component scores.

The statistic is the population excess kurtosis ``m4 / m2**2 - 3`` (Gaussian = 0)
of each component's scores along its unit-normalized reading direction. Kurtosis
is invariant to shifting and rescaling, so the value equals the kurtosis of the
ICA source itself and does not depend on centering, whitening, or reading-row
scale. The lens's own input preprocessing is applied before projecting, which is
the identity for ``icalens_preprocessing='none'``.

The same estimator serves both fitting (training) and held-out kurtosis, so the
two are directly comparable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .lens import ICALens

Moments = tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

STATISTIC = (
    "population excess kurtosis m4/m2^2 - 3 (Gaussian = 0) of scores along "
    "unit-normalized reading directions, after the lens input preprocessing"
)


def merge_moments(state: Moments | None, values: torch.Tensor) -> Moments:
    """Merge one ``[samples, columns]`` batch into streaming central moments (order 4)."""
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


def excess_from_moments(state: Moments) -> torch.Tensor:
    """Population excess kurtosis from merged central moments."""
    n, _, m2, _, m4 = state
    return n * m4 / m2.square() - 3.0


def excess_kurtosis(
    activations: Any,
    reading_matrix: Any,
    *,
    preprocessing_center: Any | None = None,
    row_normalize: bool = False,
    norm_eps: float = 1e-12,
    batch_size: int = 8192,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Excess kurtosis of every reading direction over ``[tokens, hidden]`` activations."""
    from ._fastica import _preprocessed_batches

    values = (
        activations
        if isinstance(activations, torch.Tensor)
        else torch.as_tensor(np.asarray(activations))
    )
    values = values.reshape(-1, values.shape[-1])
    target = values.device if device is None else torch.device(device)
    directions = torch.nn.functional.normalize(
        torch.as_tensor(
            np.asarray(reading_matrix)
            if not isinstance(reading_matrix, torch.Tensor)
            else reading_matrix
        ).to(device=target, dtype=torch.float32),
        dim=1,
    )
    pre = (
        None
        if preprocessing_center is None
        else torch.as_tensor(np.asarray(preprocessing_center), dtype=torch.float32)
    )
    state: Moments | None = None
    with torch.inference_mode():
        for batch in _preprocessed_batches(
            values,
            device=target,
            dtype=torch.float32,
            batch_size=batch_size,
            row_normalize=row_normalize,
            preprocessing_center=pre,
            norm_eps=norm_eps,
        ):
            state = merge_moments(state, batch @ directions.T)
    if state is None:
        raise ValueError("no activations were provided")
    return excess_from_moments(state).cpu().numpy()


def lens_excess_kurtosis(
    lens: ICALens,
    activations: Any,
    *,
    layer: int,
    batch_size: int = 8192,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Excess kurtosis of one fitted layer's components, in component order."""
    artifact = lens._get_layer(layer)
    assert artifact.reading_matrix is not None
    return excess_kurtosis(
        activations,
        artifact.reading_matrix,
        preprocessing_center=artifact.preprocessing_center,
        row_normalize=lens.row_normalize,
        norm_eps=lens.norm_eps,
        batch_size=batch_size,
        device=device,
    )
