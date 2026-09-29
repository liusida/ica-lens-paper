"""End-to-end smoke tests for an installed ICA Lens package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from . import ICALens, __version__


@dataclass(frozen=True)
class SmokeCase:
    name: Literal["text", "chat"]
    lens_id: str
    layer: int
    text: str
    response: str | None = None


DEFAULT_TEXT_CASE = SmokeCase(
    name="text",
    lens_id="sida/icalens-gpt2-small-pile10k",
    layer=6,
    text="She deposited the check at the bank.",
)
DEFAULT_CHAT_CASE = SmokeCase(
    name="chat",
    lens_id="sida/icalens-qwen3.5-2b-ultrachat-1m",
    layer=16,
    text="What is the most interesting science? One phrase.",
    response="Quantum entanglement.",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens smoke-test", description=__doc__)
    parser.add_argument(
        "case",
        choices=("text", "chat", "all"),
        nargs="?",
        default="all",
        help="Input path to test (default: all).",
    )
    parser.add_argument("--text-lens", default=DEFAULT_TEXT_CASE.lens_id)
    parser.add_argument("--text-layer", type=int, default=DEFAULT_TEXT_CASE.layer)
    parser.add_argument("--text-input", default=DEFAULT_TEXT_CASE.text)
    parser.add_argument("--chat-lens", default=DEFAULT_CHAT_CASE.lens_id)
    parser.add_argument("--chat-layer", type=int, default=DEFAULT_CHAT_CASE.layer)
    parser.add_argument("--chat-input", default=DEFAULT_CHAT_CASE.text)
    parser.add_argument("--chat-response", default=DEFAULT_CHAT_CASE.response)
    parser.add_argument("--device", default=None, help="Default: CUDA when available, else CPU.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cases = {
        "text": SmokeCase("text", args.text_lens, args.text_layer, args.text_input),
        "chat": SmokeCase(
            "chat",
            args.chat_lens,
            args.chat_layer,
            args.chat_input,
            args.chat_response,
        ),
    }
    selected = (cases["text"], cases["chat"]) if args.case == "all" else (cases[args.case],)

    print(f"ICALens {__version__}")
    for case in selected:
        _run_case(case, device=device, output_dir=args.output_dir)
    print(f"PASS: completed {len(selected)} smoke-test case(s)")


def _run_case(case: SmokeCase, *, device: str, output_dir: Path) -> None:
    print()
    print(f"[{case.name}] Loading lens {case.lens_id}...")
    lens = ICALens.from_pretrained(case.lens_id)
    _require(case.layer in lens.available_layers, f"layer {case.layer} is unavailable")
    if case.name == "text":
        _require(lens.model_type == "base", "text case requires a base-model lens")
        inputs: str | list[dict[str, str]] = case.text
    else:
        _require(lens.model_type == "instruct", "chat case requires an instruct-model lens")
        response = case.response
        if not response:
            raise RuntimeError("smoke test failed: chat response is empty")
        inputs = [
            {"role": "user", "content": case.text},
            {"role": "assistant", "content": response},
        ]

    print(f"[{case.name}] Analyzing with {lens.model_id} on {device}...")
    result = lens.analyze(inputs, layer=case.layer, device=device)
    token_count = len(result.tokens)
    _require(token_count > 0, "analysis returned no tokens")
    _require(result.scores.ndim == 2, "scores must be a two-dimensional tensor")
    _require(result.scores.shape[0] == token_count, "scores are not aligned with tokens")
    _require(result.energy.shape == result.scores.shape, "energy shape differs from scores")
    _require(bool(torch.isfinite(result.activations).all()), "activations contain nonfinite values")
    _require(bool(torch.isfinite(result.scores).all()), "scores contain nonfinite values")
    _require(bool(torch.isfinite(result.energy).all()), "energy contains nonfinite values")
    torch.testing.assert_close(
        result.energy.sum(dim=-1),
        torch.ones(token_count, device=result.energy.device),
        atol=1e-5,
        rtol=1e-5,
    )

    reconstructed = lens.inverse_transform(result.scores, layer=case.layer)
    _require(
        reconstructed.shape == result.activations.shape,
        "reconstructed activation shape differs from captured activations",
    )
    _require(bool(torch.isfinite(reconstructed).all()), "reconstruction contains nonfinite values")

    output = result.to_html(output_dir / f"icalens-smoke-{case.name}.html")
    _require(output.is_file() and output.stat().st_size > 0, "HTML report was not written")

    print(f"[{case.name}] PASS: {token_count} tokens, {result.scores.shape[1]} components")
    print(f"[{case.name}] PASS: energy sums to 1 at every token")
    print(f"[{case.name}] PASS: reconstruction shape {tuple(reconstructed.shape)}")
    print(f"[{case.name}] PASS: HTML report {output}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"smoke test failed: {message}")


if __name__ == "__main__":
    main()
