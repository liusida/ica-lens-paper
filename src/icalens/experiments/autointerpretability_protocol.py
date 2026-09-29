"""Pure, deterministic autointerpretability protocol helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

FRAGMENT_LENGTH = 64
TOP_RECORDS = 20
RANDOM_RECORDS = 20
EXAMPLES_PER_SPLIT = 5


def normalize_activations(values: np.ndarray) -> np.ndarray:
    """Map nonnegative feature activations to the historical integer 0..10 scale."""
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    scale = float(positive.max(initial=0.0))
    if scale <= 0:
        return np.zeros_like(positive, dtype=np.int64)
    return np.minimum(10, np.floor(10 * positive / scale)).astype(np.int64)


def select_record_indices(
    values: np.ndarray, *, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Return explanation-top, held-out-top, and held-out-random record indices."""
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != FRAGMENT_LENGTH:
        raise ValueError(f"expected [fragments, {FRAGMENT_LENGTH}] activations")
    maxima = array.max(axis=1)
    if len(maxima) < TOP_RECORDS:
        raise ValueError(f"fewer than {TOP_RECORDS} fragments are available")
    top = np.argsort(-maxima, kind="stable")[:TOP_RECORDS].tolist()
    ordering = np.random.default_rng(seed).permutation(len(maxima)).tolist()
    random_records = [index for index in ordering if maxima[index] > 0][:RANDOM_RECORDS]
    if len(random_records) < RANDOM_RECORDS:
        raise ValueError(f"fewer than {RANDOM_RECORDS} fragments activate the feature")
    return (
        top[0::4][:EXAMPLES_PER_SPLIT],
        top[2::4][:EXAMPLES_PER_SPLIT],
        random_records[1::3][:EXAMPLES_PER_SPLIT],
    )


def explanation_messages(
    tokens: Sequence[Sequence[str]], values: np.ndarray
) -> list[dict[str, str]]:
    """Build the version-1 feature-explanation prompt."""
    if len(tokens) != EXAMPLES_PER_SPLIT:
        raise ValueError(f"expected {EXAMPLES_PER_SPLIT} explanation fragments")
    normalized = normalize_activations(values)
    blocks: list[str] = []
    for number, (record_tokens, record_values) in enumerate(
        zip(tokens, normalized, strict=True), 1
    ):
        if len(record_tokens) != FRAGMENT_LENGTH:
            raise ValueError(f"expected {FRAGMENT_LENGTH} tokens per fragment")
        rows = [
            f"{index:02d}\t{json.dumps(token, ensure_ascii=False)}\t{int(value)}"
            for index, (token, value) in enumerate(zip(record_tokens, record_values, strict=True))
        ]
        blocks.append(f"Fragment {number}:\nindex\ttoken\tactivation\n" + "\n".join(rows))
    return [
        {
            "role": "system",
            "content": (
                "You analyze language-model features. Infer one concise pattern that explains "
                "which tokens activate the feature. Do not mention fragment numbers or "
                "speculate about the experiment."
            ),
        },
        {
            "role": "user",
            "content": (
                "0 means inactive and 10 is the strongest positive activation.\n\n"
                "Here are five top-activating examples:\n\n"
                + "\n\n".join(blocks)
                + "\n\nReturn only a concise explanation of the feature's behavior."
            ),
        },
    ]


def simulator_messages(explanation: str, tokens: Sequence[str]) -> list[dict[str, str]]:
    """Build the version-1 indexed simulator prompt."""
    if len(tokens) != FRAGMENT_LENGTH:
        raise ValueError(f"expected {FRAGMENT_LENGTH} tokens")
    indexed = [{"index": index, "token": token} for index, token in enumerate(tokens)]
    return [
        {
            "role": "system",
            "content": (
                "Predict one activation for every token using only the supplied feature "
                "explanation. Return valid JSON with exactly one key, activations. Its value "
                "must be an object with exactly the string keys 0 through 63, each mapped to "
                "one integer activation label from 0 through 10. Do not omit or add keys."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Feature explanation: {explanation}\n\n"
                "Use integer labels from 0 (inactive) to 10 (strongest activation).\n\n"
                "Count against the explicit indices 0 through 63 before answering. "
                "Return compact JSON only.\n\n"
                f"Tokens:\n{json.dumps(indexed, ensure_ascii=False)}"
            ),
        },
    ]


def parse_indexed_activations(raw: str) -> np.ndarray:
    """Strictly parse the simulator's indexed JSON response."""
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("simulator response is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"activations"}:
        raise ValueError("simulator response must contain exactly the 'activations' key")
    activations = payload["activations"]
    expected = [str(index) for index in range(FRAGMENT_LENGTH)]
    if not isinstance(activations, dict) or list(activations) != expected:
        raise ValueError("activations must have exactly the ordered string keys 0 through 63")
    values = list(activations.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("activation labels must be integers")
    if any(value < 0 or value > 10 for value in values):
        raise ValueError("activation labels must be in the range 0 through 10")
    return np.asarray(values, dtype=np.float64)


def pearson_correlation(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return Pearson correlation, or None for constant/invalid vectors."""
    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("actual and predicted values must have the same nonempty shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
