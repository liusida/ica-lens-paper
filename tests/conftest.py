from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def mixed_signals() -> np.ndarray:
    rng = np.random.default_rng(7)
    samples = 800
    sources = np.column_stack(
        [
            rng.laplace(size=samples),
            rng.uniform(-2, 2, size=samples),
            rng.standard_t(df=3, size=samples),
        ]
    )
    mixing = np.array([[1.0, 0.4, -0.2], [0.2, 1.1, 0.5], [-0.4, 0.3, 0.9]])
    return np.asarray(sources @ mixing.T, dtype=np.float32)
