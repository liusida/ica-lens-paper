from __future__ import annotations

import math
from collections.abc import Iterable


def absolute_cosine(a: Iterable[float], b: Iterable[float]) -> float:
    va = [float(value) for value in a]
    vb = [float(value) for value in b]
    if len(va) != len(vb):
        raise ValueError("absolute_cosine inputs must have the same length.")
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
    denom = math.sqrt(sum(x * x for x in va)) * math.sqrt(sum(y * y for y in vb))
    if denom == 0.0:
        return 0.0
    return float(abs(dot / denom))
