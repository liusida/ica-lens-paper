"""Pilot paper-style SAE and ICA language steering on Gemma 2 2B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shlex
import sys
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens._capture import transformer_blocks
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._sae import SAEFeatureEncoder
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.saebench_sparse_probing import (
    _prepare_layer_baselines,
    _resolve_baselines,
)

LENS_ID = "sida/icalens-gemma-2-2b-pile10k"
CORPUS_BASE_URL = "https://www.manythings.org/anki"
TARGET_LANGUAGES = {
    "chinese": {"display": "Chinese", "corpus_code": "cmn"},
    "french": {"display": "French", "corpus_code": "fra"},
    "japanese": {"display": "Japanese", "corpus_code": "jpn"},
    "spanish": {"display": "Spanish", "corpus_code": "spa"},
}
DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (compatible; ICA-Lens-research/0.3; +https://github.com/liusida/ica-lens-paper)"
)
PAPER_PROMPT = "Leaves change color in autumn because"
DEFAULT_PROMPTS = (
    "Artificial intelligence is important because",
    PAPER_PROMPT,
    "A practical way to organize a small kitchen is",
    "The difference between weather and climate is",
)


class _CaptureComplete(Exception):
    """Stop a forward pass after the selected layer has been captured."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens",
        default=LENS_ID,
        help="Local ICA Lens artifact or Hugging Face model repository.",
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layers",
        help=(
            "Layers to search: all, one layer (19), an inclusive range (19-21), "
            "or comma-separated combinations (0,4-6,10). Defaults to all layers."
        ),
    )
    layer_group.add_argument(
        "--layer",
        type=int,
        help="Compatibility alias for selecting one zero-based layer.",
    )
    parser.add_argument(
        "--method",
        choices=("all", "sae", "ica"),
        default="all",
        help="Representation to search and steer; defaults to both SAE and ICA.",
    )
    parser.add_argument(
        "--target-language",
        choices=("all", *TARGET_LANGUAGES),
        default="all",
        help="Target language; defaults to all four registered languages.",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default="layer_20/width_16k/average_l0_71/params.npz",
        help="Gemma Scope checkpoint used for the selected layer.",
    )
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument(
        "--final-token-mode",
        choices=("text", "eos"),
        default="text",
        help="Use the last text token (paper-literal h(x)[-1]) or append EOS explicitly.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument("--offset-multipliers", default="1")
    parser.add_argument(
        "--steering-convention",
        choices=("all-positions",),
        default="all-positions",
        help="Apply the intervention to every position processed during generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        help="Override the target language's ManyThings corpus cache path.",
    )
    parser.add_argument(
        "--calibration-cache-dir",
        type=Path,
        default=Path(__file__).with_name("calibration-cache"),
        help="Directory for reusable component/feature IDs and measured offsets.",
    )
    parser.add_argument(
        "--refresh-calibration",
        action="store_true",
        help="Ignore a matching calibration cache and recompute feature/component offsets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Run directory containing resumable per-condition JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs <= 0 or args.batch_size <= 0 or args.top_features <= 0:
        raise ValueError("--pairs, --batch-size, and --top-features must be positive")
    multipliers = tuple(float(value) for value in args.offset_multipliers.split(","))
    if not multipliers or any(value < 0 for value in multipliers):
        raise ValueError("--offset-multipliers must contain nonnegative numbers")
    lens = ICALens.from_pretrained(args.lens)
    layers = _resolve_layers(args.layers, args.layer, lens.available_layers)
    invalid_layers = sorted(set(layers).difference(lens.available_layers))
    if invalid_layers:
        raise ValueError(
            f"layers {invalid_layers} are unavailable; choose from {lens.available_layers}"
        )
    methods = ("sae", "ica") if args.method == "all" else (args.method,)
    languages = (
        tuple(TARGET_LANGUAGES) if args.target_language == "all" else (args.target_language,)
    )
    if args.corpus_cache is not None and len(languages) != 1:
        raise ValueError("--corpus-cache requires one --target-language")
    if args.sae_checkpoint and ("sae" not in methods or len(layers) != 1):
        raise ValueError(
            "--sae-checkpoint requires a run including SAE and exactly one selected layer"
        )

    experiment_root = Path(__file__).parent
    output = (
        args.output
        or _default_output(
            experiment_root,
            methods,
            languages,
            layers,
            lens.available_layers,
            args.steering_convention,
        )
    ).resolve()
    units = [
        (method, language, layer)
        for layer in layers
        for method in methods
        for language in languages
    ]
    sae_registry = _resolve_baselines(lens.model_id, "sae") if "sae" in methods else {}
    resolved = {
        "format_version": 1,
        "lens": args.lens,
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "layers": list(layers),
        "methods": list(methods),
        "target_languages": list(languages),
        "pairs": args.pairs,
        "final_token_mode": args.final_token_mode,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "top_features": args.top_features,
        "offset_multipliers": list(multipliers),
        "steering_convention": args.steering_convention,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "prompts": list(DEFAULT_PROMPTS),
        "corpora": {
            language: f"{CORPUS_BASE_URL}/{TARGET_LANGUAGES[language]['corpus_code']}-eng.zip"
            for language in languages
        },
        "sae_registry": sae_registry,
        "sae_checkpoint_override": args.sae_checkpoint,
    }
    source = source_provenance()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="searching",
    )
    completed_ids = {
        _unit_id(method, language, layer)
        for method, language, layer in units
        if _condition_checkpoint_valid(
            _condition_output(output, method, language, layer),
            _condition_configuration(args, lens, method, language, layer),
        )
    }
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · language steering search",
        completed=len(completed_ids),
        total=len(units),
        completed_unit_ids=completed_ids,
        unit_label="conditions",
        recent_label="Recent steering output",
        source_dirty=bool(source.get("dirty")),
    )

    with display:
        warn_if_dirty(source)
        log(
            f"Resolved {len(units)} conditions: {len(layers)} layers × "
            f"{len(methods)} methods × {len(languages)} languages."
        )
        if len(completed_ids) == len(units):
            display.phase("Complete")
            run.set_status("complete", complete=True)
            log(f"All language-steering conditions are already complete: {output}")
            return

        display.phase("Loading model", model=lens.model_id)
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            lens.model_id, revision=lens.model_revision, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if device == "cuda":
            model = load_model_to_cuda(
                AutoModelForCausalLM,
                lens.model_id,
                revision=lens.model_revision,
                device=device,
                dtype=torch.bfloat16,
                touch="auto",
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                lens.model_id, revision=lens.model_revision
            ).to(device)
        model.eval()
        hidden_size_value = getattr(model.config, "hidden_size", None)
        if hidden_size_value is None:
            hidden_size_value = model.config.text_config.hidden_size

        pairs_by_language: dict[str, list[tuple[str, str]]] = {}
        try:
            for layer in layers:
                for method in methods:
                    pending_languages = [
                        language
                        for language in languages
                        if _unit_id(method, language, layer) not in completed_ids
                    ]
                    if not pending_languages:
                        continue
                    baseline: dict[str, Any] | None = None
                    sae: SAEFeatureEncoder | None = None
                    if method == "sae":
                        display.phase("Loading SAE", layer=layer)
                        baseline = _prepare_layer_baselines(sae_registry, layer=layer)
                        if args.sae_checkpoint:
                            sae_baseline = baseline["sae"]
                            sae_baseline["checkpoint"] = args.sae_checkpoint
                            sae_baseline["weights_file"] = hf_hub_download(
                                repo_id=str(sae_baseline["repo_id"]),
                                filename=args.sae_checkpoint,
                                revision=str(sae_baseline["revision"]),
                            )
                        snapshot = {
                            "baselines": baseline,
                            "hidden_size": int(hidden_size_value),
                            "layer": layer,
                            "saebench_model_name": lens.model_id,
                        }
                        sae = SAEFeatureEncoder(snapshot, device=device, dtype=torch.float32)

                    for language in pending_languages:
                        condition_args = copy(args)
                        condition_args.method = method
                        condition_args.target_language = language
                        condition_args.layer = layer
                        condition_args.output = _condition_output(output, method, language, layer)
                        display.phase(
                            "Evaluating",
                            layer=layer,
                            method=method.upper(),
                            language=TARGET_LANGUAGES[language]["display"],
                        )
                        if language not in pairs_by_language:
                            pairs_by_language[language] = _load_condition_pairs(
                                condition_args, experiment_root
                            )
                        _run_condition(
                            condition_args,
                            lens=lens,
                            model=model,
                            tokenizer=tokenizer,
                            pairs=pairs_by_language[language],
                            baseline=baseline,
                            sae=sae,
                            multipliers=multipliers,
                        )
                        unit_id = _unit_id(method, language, layer)
                        completed_ids.add(unit_id)
                        display.complete_unit(unit_id, refresh=True)
                    del sae
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        display.phase("Complete")
        run.set_status("complete", complete=True)
        log(f"Language-steering search complete: {output}")


def _run_condition(
    args: argparse.Namespace,
    *,
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    baseline: dict[str, Any] | None,
    sae: SAEFeatureEncoder | None,
    multipliers: tuple[float, ...],
) -> None:
    language = TARGET_LANGUAGES[args.target_language]
    target_display = str(language["display"])
    corpus_code = str(language["corpus_code"])
    corpus_url = f"{CORPUS_BASE_URL}/{corpus_code}-eng.zip"
    output = Path(args.output)
    unit_name = "feature" if args.method == "sae" else "component"
    calibration = calibration_descriptor(
        args=args,
        lens=lens,
        pairs=pairs,
        sae_baseline=baseline["sae"] if baseline is not None else None,
    )
    cache_path = calibration_cache_path(args.calibration_cache_dir, calibration)
    cache_hit = cache_path.is_file() and not args.refresh_calibration
    if cache_hit:
        cached = json.loads(cache_path.read_text())
        if cached.get("configuration") != calibration:
            raise ValueError(f"calibration cache configuration mismatch: {cache_path}")
        candidates = cached["candidates"]
        print(f"Loaded calibration from {cache_path}")
    else:
        english = [pair[0] for pair in pairs]
        target = [pair[1] for pair in pairs]
        print(
            f"Encoding final-token activations for {len(pairs)} English/{target_display} pairs..."
        )
        english_mean = mean_final_codes(
            model,
            tokenizer,
            english,
            method=args.method,
            lens=lens,
            sae=sae,
            layer=args.layer,
            batch_size=args.batch_size,
            append_eos=args.final_token_mode == "eos",
        )
        target_mean = mean_final_codes(
            model,
            tokenizer,
            target,
            method=args.method,
            lens=lens,
            sae=sae,
            layer=args.layer,
            batch_size=args.batch_size,
            append_eos=args.final_token_mode == "eos",
        )
        difference = target_mean - english_mean
        candidate_ids = torch.topk(difference.abs(), k=args.top_features).indices.tolist()
        candidates = [
            {"rank": rank, unit_name: int(feature), "offset": float(difference[feature])}
            for rank, feature in enumerate(candidate_ids)
        ]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            cache_path,
            {"configuration": calibration, "candidates": candidates},
        )
        print(f"Cached calibration at {cache_path}")
    print("Selected candidates:")
    for item in candidates:
        prefix = "F" if args.method == "sae" else "C"
        print(f"  rank {item['rank']}: {prefix}{item[unit_name]} {item['offset']:+.4f}")

    generations: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(DEFAULT_PROMPTS):
        torch.manual_seed(args.seed + prompt_index)
        baseline_text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        row: dict[str, Any] = {"prompt": prompt, "baseline": baseline_text, "steered": []}
        for candidate in candidates:
            for multiplier in multipliers:
                torch.manual_seed(args.seed + prompt_index)
                offset = float(candidate["offset"]) * multiplier
                if args.method == "sae":
                    with additive_sae_steering(
                        model,
                        require_sae(sae),
                        layer=args.layer,
                        feature=int(candidate[unit_name]),
                        offset=offset,
                    ):
                        text = generate(
                            model,
                            tokenizer,
                            prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )
                else:
                    text = lens.generate(
                        prompt,
                        layer=args.layer,
                        steer=(int(candidate[unit_name]), offset),
                        steering_scope=args.steering_convention,
                        max_new_tokens=args.max_new_tokens,
                        model=model,
                        tokenizer=tokenizer,
                        do_sample=True,
                        temperature=args.temperature,
                    )
                row["steered"].append(
                    {
                        "candidate_rank": candidate["rank"],
                        unit_name: candidate[unit_name],
                        "offset_multiplier": multiplier,
                        "offset": offset,
                        "text": text,
                    }
                )
        generations.append(row)

    payload = {
        "experiment": f"{args.method}-language-steering-pilot",
        "configuration": _condition_configuration(
            args, lens, args.method, args.target_language, args.layer
        ),
        "command": f"uv run python {shlex.join(sys.argv)}",
        "protocol": {
            "reference": "https://arxiv.org/abs/2507.13410",
            "method": args.method,
            "selection": "final_token_target_minus_english_mean",
            "source_language": "English",
            "target_language": target_display,
            "parallel_corpus": {
                "source": "Tatoeba via ManyThings",
                "url": corpus_url,
            },
            "final_token_mode": args.final_token_mode,
            "intervention": (
                "x_prime = x + offset * unit_decoder_direction"
                if args.method == "sae"
                else "x_prime = x + score_offset * ICA_writing_direction"
            ),
            "steering_convention": args.steering_convention,
            "model": f"{lens.model_id}@{lens.model_revision}",
            "lens": args.lens,
            "layer": args.layer,
            "pairs": len(pairs),
            "seed": args.seed,
            "top_features": args.top_features,
            "calibration_cache": str(cache_path),
            "calibration_cache_hit": cache_hit,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "offset_multipliers": multipliers,
            **({"sae": baseline["sae"]} if baseline is not None else {}),
        },
        "candidates": candidates,
        "generations": generations,
    }
    atomic_write_json(output, payload)
    print(f"Wrote {output}")
    print_summary(generations, method=args.method)


def _default_output(
    experiment_root: Path,
    methods: tuple[str, ...],
    languages: tuple[str, ...],
    layers: tuple[int, ...],
    available_layers: tuple[int, ...],
    steering_convention: str,
) -> Path:
    method_label = "all-methods" if len(methods) > 1 else methods[0]
    language_label = "all-languages" if len(languages) > 1 else languages[0]
    layer_label = _layer_label(layers, available_layers)
    return (
        experiment_root
        / "runs"
        / (f"{method_label}-{language_label}-{layer_label}-{steering_convention}")
    )


def _layer_label(layers: tuple[int, ...], available_layers: tuple[int, ...]) -> str:
    if layers == available_layers:
        return "all-layers"
    spans: list[str] = []
    start = previous = layers[0]
    for layer in layers[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = layer
    spans.append(str(start) if start == previous else f"{start}-{previous}")
    return "layers" + "_".join(spans)


def _resolve_layers(
    specification: str | None,
    compatibility_layer: int | None,
    available_layers: tuple[int, ...],
) -> tuple[int, ...]:
    if compatibility_layer is not None:
        return (compatibility_layer,)
    if specification is None or specification.strip().lower() == "all":
        return tuple(available_layers)

    selected: set[int] = set()
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid --layers value: {specification!r}")
        if "-" not in part:
            selected.add(int(part))
            continue
        bounds = part.split("-")
        if len(bounds) != 2 or not all(bounds):
            raise ValueError(f"invalid layer range: {part!r}")
        start, end = (int(value) for value in bounds)
        if end < start:
            raise ValueError(f"layer range must be ascending: {part!r}")
        selected.update(range(start, end + 1))
    unavailable = sorted(selected.difference(available_layers))
    if unavailable:
        raise ValueError(f"layers {unavailable} are unavailable; choose from {available_layers}")
    if not selected:
        raise ValueError("--layers must select at least one layer")
    return tuple(layer for layer in available_layers if layer in selected)


def _unit_id(method: str, language: str, layer: int) -> str:
    return f"{method}:{language}:{layer}"


def _condition_output(output: Path, method: str, language: str, layer: int) -> Path:
    return output / "results" / f"{method}-english-to-{language}-layer{layer}.json"


def _condition_configuration(
    args: argparse.Namespace,
    lens: ICALens,
    method: str,
    language: str,
    layer: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": method,
        "target_language": language,
        "layer": layer,
        "lens": args.lens,
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "pairs": args.pairs,
        "seed": args.seed,
        "final_token_mode": args.final_token_mode,
        "top_features": args.top_features,
        "offset_multipliers": args.offset_multipliers,
        "steering_convention": args.steering_convention,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "sae_checkpoint_override": args.sae_checkpoint,
    }


def _condition_checkpoint_valid(path: Path, configuration: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("configuration") == configuration
        and isinstance(payload.get("candidates"), list)
        and isinstance(payload.get("generations"), list)
    )


def _load_condition_pairs(args: argparse.Namespace, experiment_root: Path) -> list[tuple[str, str]]:
    language = TARGET_LANGUAGES[args.target_language]
    corpus_code = str(language["corpus_code"])
    corpus_url = f"{CORPUS_BASE_URL}/{corpus_code}-eng.zip"
    corpus_cache = args.corpus_cache or experiment_root / "data" / f"{corpus_code}-eng.zip"
    return load_parallel_pairs(
        corpus_cache,
        url=corpus_url,
        count=args.pairs,
        seed=args.seed,
    )


def calibration_descriptor(
    *,
    args: argparse.Namespace,
    lens: ICALens,
    pairs: list[tuple[str, str]],
    sae_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    pair_bytes = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()
    artifact = lens._get_layer(args.layer)
    lens_layer_sha256 = None
    if args.method == "ica":
        if artifact.reading_matrix is None or artifact.writing_matrix is None:
            raise ValueError(f"layer {args.layer} has no loaded ICA matrices")
        matrix_hash = hashlib.sha256()
        matrix_hash.update(artifact.reading_matrix.tobytes())
        matrix_hash.update(artifact.writing_matrix.tobytes())
        lens_layer_sha256 = matrix_hash.hexdigest()
    return {
        "format_version": 2,
        "method": args.method,
        "source_language": "english",
        "target_language": args.target_language,
        "lens": args.lens,
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "lens_layer_sha256": lens_layer_sha256,
        "layer": args.layer,
        "pairs": len(pairs),
        "pairs_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "seed": args.seed,
        "final_token_mode": args.final_token_mode,
        "top_features": args.top_features,
        "sae_repo_id": sae_baseline.get("repo_id") if sae_baseline else None,
        "sae_revision": sae_baseline.get("revision") if sae_baseline else None,
        "sae_checkpoint": sae_baseline.get("checkpoint") if sae_baseline else None,
    }


def calibration_cache_path(directory: Path, descriptor: dict[str, Any]) -> Path:
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return directory / (
        f"{descriptor['method']}-english-to-{descriptor['target_language']}"
        f"-layer{descriptor['layer']}-{digest}.json"
    )


def load_parallel_pairs(
    path: Path,
    *,
    url: str,
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        print(f"Downloading {url} to {path}...")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": DOWNLOAD_USER_AGENT, "Accept": "application/zip,*/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        except OSError as error:
            raise RuntimeError(
                f"could not download {url}; download it manually and place it at {path}"
            ) from error
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith(".txt") and not Path(name).name.startswith("_")
        ]
        if not members:
            raise RuntimeError(f"{path} contains no parallel-text file")
        member = max(members, key=lambda name: archive.getinfo(name).file_size)
        text = archive.read(member).decode("utf-8")
    available: list[tuple[str, str]] = []
    for row in csv.reader(io.StringIO(text), delimiter="\t"):
        if len(row) >= 2 and row[0].strip() and row[1].strip():
            available.append((row[0].strip(), row[1].strip()))
    if count > len(available):
        raise ValueError(f"requested {count} pairs but corpus contains {len(available)}")
    return random.Random(seed).sample(available, count)


def mean_final_codes(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    method: str,
    lens: ICALens,
    sae: SAEFeatureEncoder | None,
    layer: int,
    batch_size: int,
    append_eos: bool,
) -> torch.Tensor:
    if method == "sae":
        if sae is None:
            raise ValueError("SAE encoder is required for --method sae")
        width = sae.cfg.d_sae
    else:
        width = lens._get_layer(layer).n_components
    total = torch.zeros(width, dtype=torch.float64)
    seen = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if append_eos:
            eos = str(tokenizer.eos_token)
            framed = [text if text.endswith(eos) else text + eos for text in batch]
        else:
            framed = batch
        encoded = tokenizer(framed, padding=True, truncation=True, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {name: value.to(model_device) for name, value in encoded.items()}
        # Gemma tokenizers left-pad by default, so ``mask.sum() - 1`` points
        # into the padding for shorter rows. Find the last non-padding index
        # explicitly; this also works if a tokenizer is configured to right-pad.
        attention_mask = inputs["attention_mask"]
        final_positions = attention_mask.shape[1] - 1 - attention_mask.flip(dims=(1,)).argmax(dim=1)
        hidden = capture_final_resid_post(
            model, model_inputs=inputs, layer=layer, positions=final_positions
        )
        with torch.inference_mode():
            if method == "sae":
                assert sae is not None
                codes = sae.encode(hidden.float())
            else:
                codes = lens.transform(hidden.float(), layer=layer)
            codes = codes.double().sum(dim=0).cpu()
        total += codes
        seen += len(batch)
        print(f"  {seen}/{len(texts)}", end="\r", flush=True)
    print()
    return (total / seen).float()


def require_sae(sae: SAEFeatureEncoder | None) -> SAEFeatureEncoder:
    if sae is None:
        raise ValueError("SAE encoder is required for --method sae")
    return sae


def capture_final_resid_post(
    model: torch.nn.Module,
    *,
    model_inputs: dict[str, torch.Tensor],
    layer: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    blocks = transformer_blocks(model)
    captured: torch.Tensor | None = None

    def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
        nonlocal captured
        hidden = output[0] if isinstance(output, tuple) else output
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        captured = hidden[rows, positions.to(hidden.device)].detach()
        raise _CaptureComplete

    handle = blocks[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            try:
                model(**model_inputs, use_cache=False)
            except _CaptureComplete:
                pass
    finally:
        handle.remove()
    if captured is None:
        raise RuntimeError("failed to capture final-token residual activations")
    return captured


@contextmanager
def additive_sae_steering(
    model: torch.nn.Module,
    sae: SAEFeatureEncoder,
    *,
    layer: int,
    feature: int,
    offset: float,
) -> Iterator[None]:
    blocks = transformer_blocks(model)

    def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        direction = sae.W_dec[feature].to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden + offset * direction
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {name: value.to(model_device) for name, value in encoded.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        tokens = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    return str(tokenizer.decode(tokens[0, prompt_length:], skip_special_tokens=True))


def print_summary(generations: list[dict[str, Any]], *, method: str) -> None:
    unit_name = "feature" if method == "sae" else "component"
    prefix = "F" if method == "sae" else "C"
    for row in generations:
        print(f"\nPROMPT: {row['prompt']}\nBASELINE: {row['baseline']}")
        for item in row["steered"]:
            print(f"{prefix}{item[unit_name]} ×{item['offset_multiplier']}: {item['text']}")


if __name__ == "__main__":
    main()
