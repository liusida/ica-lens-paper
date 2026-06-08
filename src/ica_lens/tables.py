from __future__ import annotations


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")
