from __future__ import annotations

from collections.abc import Iterable


def excess_kurtosis(values: Iterable[float]) -> float:
    xs = [float(value) for value in values]
    if not xs:
        raise ValueError("excess_kurtosis requires at least one value.")
    mean = sum(xs) / len(xs)
    centered = [value - mean for value in xs]
    variance = sum(value**2 for value in centered) / len(centered)
    if variance == 0.0:
        return 0.0
    fourth = sum(value**4 for value in centered) / len(centered)
    return float(fourth / (variance**2) - 3.0)
