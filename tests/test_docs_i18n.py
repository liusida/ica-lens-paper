from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWED_TRANSLATIONS = (
    "getting-started.md",
    "text-and-chat.md",
    "component-profiles.md",
    "scores-and-energy.md",
    "steering.md",
    "reconstruction.md",
    "fit-and-publish.md",
    "api.md",
)

CODE_PARITY_TRANSLATIONS = tuple(
    filename for filename in REVIEWED_TRANSLATIONS if filename != "api.md"
)


def _heading_levels(text: str) -> list[int]:
    return [len(match.group(1)) for match in re.finditer(r"^(#{1,6})\s", text, re.MULTILINE)]


def _code_blocks(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(r"^```([^\n]*)\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
    ]


def _image_count(text: str) -> int:
    return len(re.findall(r"!\[[^]]*]\([^)]+\)", text))


def test_reviewed_chinese_pages_match_english_structure() -> None:
    for filename in REVIEWED_TRANSLATIONS:
        english = (ROOT / "docs" / filename).read_text()
        chinese = (ROOT / "docs-zh" / filename).read_text()

        assert _heading_levels(chinese) == _heading_levels(english), filename
        assert _image_count(chinese) == _image_count(english), filename

        if filename in CODE_PARITY_TRANSLATIONS:
            assert _code_blocks(chinese) == _code_blocks(english), filename
