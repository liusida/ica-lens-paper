from __future__ import annotations

import torch

from icalens.cli.fit_activations import parse_args as parse_activation_args
from icalens.cli.fit_chat import parse_args as parse_chat_args
from icalens.cli.fit_text import (
    TextDocument,
    load_pile_documents,
    parse_args,
    parse_token_budget,
    resolve_document_framing,
    resolve_text_dataset,
    sample_positions,
)


def test_all_fitting_cli_defaults_use_no_icalens_preprocessing() -> None:
    assert parse_args([]).icalens_preprocessing == "none"
    assert parse_chat_args([]).icalens_preprocessing == "none"
    assert (
        parse_activation_args(["--input", "activations", "--output", "lens"])
        .icalens_preprocessing
        == "none"
    )


def test_parse_all_token_budget() -> None:
    assert parse_token_budget("all") is None
    assert parse_token_budget("1000") == 1000


def test_parse_all_candidate_tokens() -> None:
    args = parse_args(["--candidate-tokens", "all"])
    assert args.candidate_tokens is None


def test_parse_text_source_options() -> None:
    args = parse_args(
        [
            "--model",
            "owner/model",
            "--dataset",
            "owner/dataset",
            "--split",
            "validation",
            "--text-field",
            "body",
            "--context-length",
            "512",
        ]
    )
    assert args.model == "owner/model"
    assert args.dataset == "owner/dataset"
    assert args.split == "validation"
    assert args.text_field == "body"
    assert args.context_length == 512


def test_resolve_local_text_dataset_records_path_and_hash(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"alpha"}\n')

    revision, provenance = resolve_text_dataset(str(dataset), split="train")

    assert revision == provenance["sha256"]
    assert provenance == {
        "path": str(dataset.resolve()),
        "sha256": revision,
        "split": "train",
    }


def test_load_local_jsonl_uses_one_truncated_context_per_row(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"abcdef"}\n{"text":"gh"}\n')

    class Tokenizer:
        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
            max_length: int,
        ) -> dict[str, list[int]]:
            assert add_special_tokens is False
            assert truncation is True
            return {"input_ids": [ord(character) for character in text][:max_length]}

    documents = load_pile_documents(
        Tokenizer(),
        dataset_id=str(dataset),
        split="train",
        text_field="text",
        candidate_token_budget=None,
        context_length=4,
        document_framing={"strategy": "prepend-eos", "token_id": 99},
    )

    assert [document.input_ids.tolist() for document in documents] == [
        [99, ord("a"), ord("b"), ord("c")],
        [99, ord("g"), ord("h")],
    ]
    assert [document.candidate_positions.tolist() for document in documents] == [
        [1, 2, 3],
        [1, 2],
    ]


def test_sample_all_positions_without_random_index() -> None:
    documents = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    selected = sample_positions(documents, token_budget=None, seed=0)
    torch.testing.assert_close(selected[0], torch.tensor([0, 1]))
    torch.testing.assert_close(selected[1], torch.tensor([0, 1, 2]))


def test_sample_positions_excludes_document_prefix() -> None:
    documents = [
        TextDocument(
            input_ids=torch.tensor([99, 1, 2]),
            candidate_positions=torch.tensor([1, 2]),
        ),
        TextDocument(
            input_ids=torch.tensor([99, 3]),
            candidate_positions=torch.tensor([1]),
        ),
    ]
    selected = sample_positions(documents, token_budget=None, seed=0)
    torch.testing.assert_close(selected[0], torch.tensor([1, 2]))
    torch.testing.assert_close(selected[1], torch.tensor([1]))


def test_auto_document_framing_uses_exact_gemma_registry_entry() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "bos_token": "<bos>",
            "bos_token_id": 2,
            "eos_token": "<eos>",
            "eos_token_id": 1,
        },
    )()
    framing = resolve_document_framing(
        tokenizer, "auto", model_id="google/gemma-2-2b"
    )
    assert framing["strategy"] == "prepend-bos"
    assert framing["token_id"] == 2
    assert framing["included_in_fitting_samples"] is False


def test_auto_document_framing_uses_exact_gpt2_registry_entry() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "bos_token": "<|endoftext|>",
            "bos_token_id": 50_256,
            "eos_token": "<|endoftext|>",
            "eos_token_id": 50_256,
        },
    )()
    framing = resolve_document_framing(
        tokenizer, "auto", model_id="openai-community/gpt2"
    )
    assert framing["strategy"] == "prepend-eos"
    assert framing["token_id"] == 50_256
    assert framing["registry_model_entry"] == "openai-community/gpt2"


def test_explicit_document_framing_does_not_require_registry_entry() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {"eos_token": "<eos>", "eos_token_id": 7},
    )()
    framing = resolve_document_framing(
        tokenizer, "prepend-eos", model_id="owner/unknown-model"
    )
    assert framing["strategy"] == "prepend-eos"
    assert framing["policy_source"] == "explicit-cli"
