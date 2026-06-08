from __future__ import annotations


def erf_bucket(value: float, *, max_window: int = 11) -> str:
    if value >= max_window:
        return f"{max_window}+"
    return str(int(round(value)))
