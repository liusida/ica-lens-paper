from __future__ import annotations

from ica_lens.sae import split_signed_score


def test_split_signed_score() -> None:
    assert split_signed_score(1.5) == (1.5, 0.0)
    assert split_signed_score(-2.0) == (0.0, 2.0)
    assert split_signed_score(0.0) == (0.0, -0.0)
