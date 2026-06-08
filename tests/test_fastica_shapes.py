from __future__ import annotations

from ica_lens.fastica import normalize_rows_available


def test_fastica_scaffold_available() -> None:
    assert normalize_rows_available()
