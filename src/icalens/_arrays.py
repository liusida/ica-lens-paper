"""Array conversion and shape helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def transform_array(
    values: Any,
    *,
    matrix: NDArray[np.float32],
    offset: NDArray[np.float32],
    pre_offset: NDArray[np.float32] | None = None,
    normalize: bool,
    norm_eps: float,
) -> Any:
    """Apply an affine row transform while preserving NumPy or torch type."""
    if _is_torch_tensor(values):
        import torch

        if values.ndim < 2:
            raise ValueError("input must have at least two dimensions")
        if not values.is_floating_point():
            raise TypeError("input must have a floating-point dtype")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("input must contain only finite values")
        work = values
        if pre_offset is not None:
            work = work - torch.as_tensor(pre_offset, dtype=work.dtype, device=work.device)
        if normalize:
            norms = torch.linalg.vector_norm(work, dim=-1, keepdim=True).clamp_min(norm_eps)
            work = work / norms
        torch_offset = torch.as_tensor(offset, dtype=work.dtype, device=work.device)
        torch_matrix = torch.as_tensor(matrix, dtype=work.dtype, device=work.device)
        return (work - torch_offset) @ torch_matrix.transpose(0, 1)

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("input must have at least two dimensions")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("input must have a floating-point dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("input must contain only finite values")
    work = array
    if pre_offset is not None:
        work = work - pre_offset
    if normalize:
        norms = np.linalg.norm(work, axis=-1, keepdims=True)
        work = work / np.maximum(norms, norm_eps)
    return (work - offset) @ matrix.T


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.split(".", maxsplit=1)[0] == "torch"
