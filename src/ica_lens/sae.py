from __future__ import annotations


def split_signed_score(score: float) -> tuple[float, float]:
    return max(score, 0.0), max(-score, 0.0)
