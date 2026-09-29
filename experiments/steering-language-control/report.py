"""Evaluate saved language-steering runs and generate Markdown and LaTeX tables."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import importlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from icalens import ICALens
from icalens.experiments._sae import SAEFeatureEncoder

ROOT = Path(__file__).parent
MODEL = "gpt-4.1-mini-2025-04-14"
RUBRIC_VERSION = "language-quality-v1"
LANGUAGES = {
    "chinese": "Chinese",
    "french": "French",
    "japanese": "Japanese",
    "spanish": "Spanish",
}
METHODS = {"sae": ("SAE", "feature", "F"), "ica": ("ICA", "component", "C")}
CONVENTION = "all-positions"
DEFAULT_RUN = ROOT / "official"
ARTIFICIAL_INTELLIGENCE_PROMPT = "Artificial intelligence is important because"


class JudgmentSchema(BaseModel):
    detected_language: str
    language_adherence: int = Field(ge=0, le=4)
    quality: int = Field(ge=0, le=4)
    relevance: int = Field(ge=0, le=4)
    degenerate: bool


@dataclass(frozen=True)
class Judgment:
    detected_language: str
    language_adherence: int
    quality: int
    relevance: int
    degenerate: bool

    @property
    def passes(self) -> bool:
        return (
            self.language_adherence >= 3
            and self.quality >= 2
            and self.relevance >= 2
            and not self.degenerate
        )


@dataclass(frozen=True)
class Sample:
    prompt: str
    text: str
    judgment: Judgment


@dataclass(frozen=True)
class Selection:
    identifier: int
    rank: int
    offset: float
    samples: tuple[Sample, ...]

    @property
    def passing_samples(self) -> tuple[Sample, ...]:
        return tuple(sample for sample in self.samples if sample.judgment.passes)

    @property
    def best_sample(self) -> Sample | None:
        if not self.passing_samples:
            return None
        return max(
            self.passing_samples,
            key=lambda sample: (
                sample.judgment.language_adherence,
                sample.judgment.quality,
                sample.judgment.relevance,
            ),
        )


class OpenAIEvaluator:
    def __init__(self, *, client: Any, model: str, cache_path: Path) -> None:
        self.client = client
        self.model = model
        self.cache_path = cache_path
        self.cache = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path.is_file()
            else {}
        )

    async def evaluate(
        self, *, target_language: str, prompt: str, output: str
    ) -> Judgment:
        key = hashlib.sha256(
            json.dumps(
                {
                    "rubric": RUBRIC_VERSION,
                    "model": self.model,
                    "target_language": target_language,
                    "prompt": prompt,
                    "output": output,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if key in self.cache:
            cached = JudgmentSchema.model_validate(self.cache[key])
            return Judgment(**cached.model_dump())
        response = await self.client.responses.parse(
            model=self.model,
            store=False,
            temperature=0,
            max_output_tokens=64,
            instructions=(
                "Evaluate a language-steered continuation, excluding the English prompt. "
                "Use integer scores from 0 (failed) to 4 (excellent). Language adherence "
                "measures use of the requested language; substantial code-mixing scores at "
                "most 2. Quality measures fluency, coherence, grammar, and non-repetition. "
                "Relevance measures whether it sensibly continues the prompt. Set degenerate "
                "true for loops, repeated templates/options, gibberish, or severe malformation."
            ),
            input=(
                f"Target language: {target_language}\n"
                f"Prompt: {prompt}\n"
                f"Continuation:\n{output}"
            ),
            text_format=JudgmentSchema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured evaluation")
        judgment = Judgment(**parsed.model_dump())
        self.cache[key] = asdict(judgment)
        self._save()
        return judgment

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUN,
        help="Run directory containing results/ (default: %(default)s)",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "evaluation-cache" / "openai-language-quality.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "RESULTS.md")
    parser.add_argument(
        "--tables-output",
        type=Path,
        default=None,
        help="LaTeX table directory (default: RUNS/tables).",
    )
    parser.add_argument(
        "--display-prompt",
        choices=("best", "artificial-intelligence"),
        default="best",
        help=(
            "Choose the generation shown for each selected candidate: its best passing "
            "sample, or the shared Artificial intelligence prompt (default: %(default)s)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing the report.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is missing from {args.env_file}")
    try:
        openai = importlib.import_module("openai")
    except ImportError as error:
        raise RuntimeError("Install with `uv sync --extra autointerpretability`") from error
    evaluator = OpenAIEvaluator(
        client=openai.AsyncOpenAI(), model=args.model, cache_path=args.cache
    )
    payloads = load_payloads(args.runs, args.layer)
    selections: dict[tuple[str, str], Selection] = {}
    for method in METHODS:
        for language in LANGUAGES:
            selection, _ = await evaluate_candidates(
                payloads[(method, language)],
                evaluator=evaluator,
                method=method,
                language=language,
            )
            selections[(method, language)] = selection
    cosines = compute_cosines(payloads, selections, layer=args.layer)
    display_prompt = (
        ARTIFICIAL_INTELLIGENCE_PROMPT
        if args.display_prompt == "artificial-intelligence"
        else None
    )
    report = render_report(
        payloads,
        selections,
        cosines,
        layer=args.layer,
        model=args.model,
        display_prompt=display_prompt,
    )
    if args.dry_run:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
        tables_output = args.tables_output or args.runs / "tables"
        tables_output.mkdir(parents=True, exist_ok=True)
        tables = {
            "language-steering-summary.tex": render_latex_summary(selections, cosines),
            "language-steering-generations.tex": render_latex_generations(
                selections, display_prompt=display_prompt
            ),
        }
        for name, contents in tables.items():
            path = tables_output / name
            path.write_text(contents, encoding="utf-8")
            print(f"Wrote {path}")


def load_payloads(runs: Path, layer: int) -> dict[tuple[str, str], dict[str, Any]]:
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    results = runs / "results"
    for method in METHODS:
        for language in LANGUAGES:
            path = results / f"{method}-english-to-{language}-layer{layer}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            configuration = payload.get("configuration", {})
            expected = {
                "layer": layer,
                "method": method,
                "target_language": language,
                "steering_convention": CONVENTION,
            }
            for key, value in expected.items():
                if configuration.get(key) != value:
                    raise ValueError(f"{path}: expected {key}={value!r}")
            payloads[(method, language)] = payload
    return payloads


async def evaluate_candidates(
    payload: dict[str, Any],
    *,
    evaluator: OpenAIEvaluator,
    method: str,
    language: str,
) -> tuple[Selection, list[Selection]]:
    _, identifier_key, _ = METHODS[method]
    candidates = {int(row[identifier_key]): row for row in payload["candidates"]}
    rows_by_candidate: dict[int, list[tuple[str, str]]] = {
        identifier: [] for identifier in candidates
    }
    for generation in payload["generations"]:
        for row in generation["steered"]:
            rows_by_candidate[int(row[identifier_key])].append(
                (str(generation["prompt"]), str(row["text"]))
            )
    evaluated: list[Selection] = []
    for identifier, rows in rows_by_candidate.items():
        candidate = candidates[identifier]
        judgments = await asyncio.gather(
            *(
                evaluator.evaluate(
                    target_language=LANGUAGES[language], prompt=prompt, output=text
                )
                for prompt, text in rows
            )
        )
        evaluated.append(
            Selection(
                identifier=identifier,
                rank=int(candidate["rank"]),
                offset=float(candidate["offset"]),
                samples=tuple(
                    Sample(prompt=prompt, text=text, judgment=judgment)
                    for (prompt, text), judgment in zip(rows, judgments, strict=True)
                ),
            )
        )
    selected = max(
        evaluated,
        key=lambda row: (
            len(row.passing_samples),
            sum(sample.judgment.language_adherence for sample in row.samples),
            sum(sample.judgment.quality for sample in row.samples),
            sum(sample.judgment.relevance for sample in row.samples),
            -row.rank,
        ),
    )
    return selected, evaluated


def render_report(
    payloads: dict[tuple[str, str], dict[str, Any]],
    selections: dict[tuple[str, str], Selection],
    cosines: dict[str, float],
    *,
    layer: int,
    model: str,
    display_prompt: str | None,
) -> str:
    configuration = next(iter(payloads.values()))["configuration"]
    lines = [
        "# English-to-target language steering",
        "",
        "Minimal SAE/ICA language-steering comparison based on",
        "[Causal Language Control in Multilingual Transformers via Sparse Feature Steering]"
        "(https://arxiv.org/abs/2507.13410).",
        "",
        f"- **Model:** `{configuration['model_id']}`",
        f"- **Layer:** {layer}",
        f"- **Calibration pairs:** {configuration['pairs']} per language",
        f"- **Final-token mode:** `{configuration['final_token_mode']}`",
        "- **Steering:** `all-positions`",
        f"- **Evaluator:** `{model}`",
        "",
        "## Selection",
        "",
        "Evaluate each of the three largest activation contrasts on four prompts. Select",
        "the candidate with the most passing outputs.",
        "",
        "| Language | SAE feature | SAE offset | ICA component | ICA offset | Signed cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language, display in LANGUAGES.items():
        sae = selections[("sae", language)]
        ica = selections[("ica", language)]
        lines.append(
            f"| {display} | F{sae.identifier} | {sae.offset:+.4f} | "
            f"C{ica.identifier} | {ica.offset:+.4f} | {cosines[language]:.4f} |"
        )
    lines.extend(["", "## Generations", ""])
    if display_prompt is None:
        lines.extend(
            [
                "For every selected feature and component, show its best passing sample.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "For every selected feature and component, show the generation from the",
                f"shared prompt *{display_prompt}*.",
                "",
            ]
        )
    lines.extend(
        [
            '<table style="width: 100%; table-layout: fixed;">',
            "<colgroup>",
            '<col style="width: 10%;">',
            '<col style="width: 45%;">',
            '<col style="width: 45%;">',
            "</colgroup>",
            "<thead><tr><th>Language</th><th>SAE</th><th>ICA</th></tr></thead>",
            "<tbody>",
        ]
    )
    for language, display in LANGUAGES.items():
        cells = []
        for method, (_, _, prefix) in METHODS.items():
            row = selections[(method, language)]
            sample = (
                row.best_sample
                if display_prompt is None
                else next(
                    (sample for sample in row.samples if sample.prompt == display_prompt),
                    None,
                )
            )
            if sample is None:
                cells.append(
                    "*No generation passed the quality threshold.*"
                    if display_prompt is None
                    else "*The requested prompt is missing from this run.*"
                )
            else:
                prompt = html.escape(sample.prompt)
                text = html.escape(sample.text).replace("\n", "<br>")
                cells.append(
                    f"<strong>{prefix}{row.identifier}</strong><br>"
                    f'<span style="color: #777;">{prompt}</span>{text}'
                )
        lines.append(
            f'<tr><td style="vertical-align: top;">{display}</td>'
            f'<td style="vertical-align: top;">{cells[0]}</td>'
            f'<td style="vertical-align: top;">{cells[1]}</td></tr>'
        )
    lines.extend(["</tbody>", "</table>", ""])
    return "\n".join(lines)


def render_latex_summary(
    selections: dict[tuple[str, str], Selection], cosines: dict[str, float]
) -> str:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"  \toprule",
        r"  Target language & SAE feature & ICA component & Signed cosine \\",
        r"  \midrule",
    ]
    for language, display in LANGUAGES.items():
        sae = selections[("sae", language)]
        ica = selections[("ica", language)]
        lines.append(
            f"  {display} & F{sae.identifier} & C{ica.identifier} & "
            f"{cosines[language]:.3f} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_latex_generations(
    selections: dict[tuple[str, str], Selection], *, display_prompt: str | None
) -> str:
    lines = [
        r"\begin{tabular}{@{}p{0.10\textwidth}p{0.42\textwidth}p{0.42\textwidth}@{}}",
        r"  \toprule",
        r"  Language & SAE & ICA \\",
        r"  \midrule",
    ]
    for language, display in LANGUAGES.items():
        cells: list[str] = []
        cjk_family = {"chinese": "gbsn", "japanese": "min"}.get(language)
        for method, (_, _, prefix) in METHODS.items():
            selection = selections[(method, language)]
            sample = _display_sample(selection, display_prompt)
            if sample is None:
                text = "No qualifying generation."
            else:
                prompt = _latex_escape(sample.prompt)
                continuation = _latex_escape(sample.text)
                text = rf"\textcolor{{gray}}{{{prompt}}}{continuation}"
            cell = rf"\textbf{{{prefix}{selection.identifier}}}\par " + text
            cells.append(
                rf"{{\CJKfamily{{{cjk_family}}} {cell}}}" if cjk_family else cell
            )
        lines.append(f"  {display} & {cells[0]} & {cells[1]} \\\\[3pt]")
    lines.extend([r"  \bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _display_sample(selection: Selection, display_prompt: str | None) -> Sample | None:
    if display_prompt is None:
        return selection.best_sample
    return next(
        (sample for sample in selection.samples if sample.prompt == display_prompt), None
    )


def _latex_escape(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value))
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value).replace(
        "\n", r"\par "
    )


def compute_cosines(
    payloads: dict[tuple[str, str], dict[str, Any]],
    selections: dict[tuple[str, str], Selection],
    *,
    layer: int,
) -> dict[str, float]:
    sae_protocol = payloads[("sae", next(iter(LANGUAGES)))]["protocol"]
    sae_config = dict(sae_protocol["sae"])
    weights = Path(str(sae_config["weights_file"]))
    if not weights.is_file():
        weights = Path(
            hf_hub_download(
                repo_id=str(sae_config["repo_id"]),
                filename=str(sae_config["checkpoint"]),
                revision=str(sae_config["revision"]),
            )
        )
    sae_config["weights_file"] = str(weights)
    lens_id = str(payloads[("ica", next(iter(LANGUAGES)))]["protocol"]["lens"])
    lens = ICALens.from_pretrained(lens_id)
    artifact = lens._get_layer(layer)
    if artifact.writing_matrix is None:
        raise ValueError(f"layer {layer} has no ICA writing matrix")
    encoder = SAEFeatureEncoder(
        {
            "baselines": {"sae": sae_config},
            "hidden_size": artifact.writing_matrix.shape[0],
            "layer": layer,
            "saebench_model_name": lens.model_id,
        },
        device="cpu",
        dtype=torch.float32,
    )
    values = {}
    for language in LANGUAGES:
        sae = selections[("sae", language)]
        ica = selections[("ica", language)]
        sae_vector = sae.offset * encoder.W_dec[sae.identifier].detach().cpu()
        ica_vector = ica.offset * torch.from_numpy(
            artifact.writing_matrix[:, ica.identifier]
        ).float()
        values[language] = float(
            torch.dot(sae_vector, ica_vector)
            / (torch.linalg.vector_norm(sae_vector) * torch.linalg.vector_norm(ica_vector))
        )
    return values


if __name__ == "__main__":
    main()
