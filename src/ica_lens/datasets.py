from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasets import load_dataset


def default_text_column() -> str:
    return "text"


def iter_dataset_texts(*, path: str, name: str | None, split: str, text_column: str, streaming: bool) -> Iterable[str]:
    kwargs: dict[str, Any] = {
        "path": path,
        "split": split,
        "streaming": streaming,
    }
    if name:
        kwargs["name"] = name

    dataset = load_dataset(**kwargs)
    for row in dataset:
        text = row.get(text_column) if isinstance(row, dict) else None
        if isinstance(text, str) and text.strip():
            yield text
