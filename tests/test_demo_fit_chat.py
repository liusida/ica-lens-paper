from __future__ import annotations

from typing import Any

import torch

from icalens.cli.fit_chat import ChatDocument, format_conversation, sample_positions


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        return "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        length = min(len(text), int(kwargs["max_length"]))
        return {
            "input_ids": list(range(length)),
            "offset_mapping": [(index, index + 1) for index in range(length)],
        }


MESSAGES = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
]


def test_format_conversation_selects_only_assistant_content() -> None:
    document = format_conversation(
        CharacterTokenizer(), MESSAGES, token_scope="assistant", context_length=100
    )
    assert document is not None
    rendered = "<user>question</user><assistant>answer</assistant>"
    expected = list(range(rendered.index("answer"), rendered.index("answer") + len("answer")))
    assert document.candidate_positions.tolist() == expected


def test_format_conversation_excludes_truncated_assistant_content() -> None:
    assistant_start = len("<user>question</user><assistant>")
    document = format_conversation(
        CharacterTokenizer(),
        MESSAGES,
        token_scope="assistant",
        context_length=assistant_start,
    )
    assert document is None


def test_sample_positions_maps_candidates_back_to_sequence_positions() -> None:
    documents = [
        ChatDocument(torch.arange(10), torch.tensor([2, 5])),
        ChatDocument(torch.arange(10), torch.tensor([1, 7])),
    ]
    selected = sample_positions(documents, token_budget=4, seed=0)
    assert selected[0].tolist() == [2, 5]
    assert selected[1].tolist() == [1, 7]
