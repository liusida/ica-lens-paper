from __future__ import annotations

from ica_lens.nongaussianity import excess_kurtosis
from ica_lens.overlap import absolute_cosine


def test_excess_kurtosis_constant_is_zero() -> None:
    assert excess_kurtosis([1.0, 1.0, 1.0]) == 0.0


def test_absolute_cosine() -> None:
    assert absolute_cosine([1.0, 0.0], [2.0, 0.0]) == 1.0
