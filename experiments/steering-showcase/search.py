#!/usr/bin/env python3
"""Find ICA components separating English from a target language."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens._capture import transformer_blocks
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORMAT = "icalens.steering_showcase.component_search"
SCHEMA_VERSION = 1
CORPUS_BASE_URL = "https://www.manythings.org/anki"
DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (compatible; ICA-Lens-research/0.3; +https://github.com/liusida/ica-lens-paper)"
)

MODEL_SPECS = {
    "gpt2": {
        "title": "GPT-2 Small",
        "lens": ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k",
        "layer": 6,
    },
    "gemma2": {
        "title": "Gemma 2 2B",
        "lens": ROOT / "local-icalens-models/official/icalens-gemma-2-2b-pile10k",
        "layer": 20,
    },
    "qwen9b": {
        "title": "Qwen 3.5 9B Base",
        "lens": ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k",
        "layer": 20,
    },
}
TARGET_LANGUAGES = {
    "chinese": {"display": "Chinese", "corpus_code": "cmn"},
    "french": {"display": "French", "corpus_code": "fra"},
    "japanese": {"display": "Japanese", "corpus_code": "jpn"},
    "spanish": {"display": "Spanish", "corpus_code": "spa"},
}


class _CaptureComplete(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", *MODEL_SPECS),
        default=["all"],
        help="Models to search; defaults to all three.",
    )
    parser.add_argument(
        "--layers",
        "--layer",
        dest="layer_specs",
        action="append",
        default=[],
        metavar="MODEL=SPEC",
        help=(
            "Override layers with an index, inclusive range, or comma-separated "
            "combination, for example gemma2=18-22 or qwen9b=7,15,23,31."
        ),
    )
    parser.add_argument(
        "--target-language", choices=tuple(TARGET_LANGUAGES), default="chinese"
    )
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-components", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        help="Parallel-corpus ZIP; defaults to the shared language-control cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Resumable run directory; defaults to runs/component-search-LANGUAGE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs <= 0 or args.batch_size <= 0 or args.top_components <= 0:
        raise ValueError("--pairs, --batch-size, and --top-components must be positive")
    labels = list(MODEL_SPECS) if args.models == ["all"] else list(dict.fromkeys(args.models))
    if "all" in labels:
        raise ValueError("--models all cannot be combined with individual model names")
    layers = {label: [int(MODEL_SPECS[label]["layer"])] for label in labels}
    overridden: set[str] = set()
    for value in args.layer_specs:
        try:
            label, raw_layers = value.split("=", 1)
        except ValueError as error:
            raise ValueError(
                "--layers must use MODEL=SPEC, for example gemma2=18-22"
            ) from error
        if label not in labels:
            raise ValueError(f"layer override names unselected model {label!r}")
        selected = parse_layer_spec(raw_layers)
        if label in overridden:
            layers[label] = sorted(set(layers[label]).union(selected))
        else:
            layers[label] = selected
            overridden.add(label)

    language = TARGET_LANGUAGES[args.target_language]
    corpus_code = str(language["corpus_code"])
    corpus_path = (
        args.corpus_cache
        or HERE / "data" / f"{corpus_code}-eng.zip"
    ).expanduser().resolve()
    pairs = load_parallel_pairs(
        corpus_path,
        url=f"{CORPUS_BASE_URL}/{corpus_code}-eng.zip",
        count=args.pairs,
        seed=args.seed,
    )
    pair_bytes = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()

    lenses = {label: ICALens.from_pretrained(MODEL_SPECS[label]["lens"]) for label in labels}
    for label, lens in lenses.items():
        invalid = sorted(set(layers[label]).difference(lens.available_layers))
        if invalid:
            raise ValueError(
                f"{label} layers {invalid} unavailable; choose from {lens.available_layers}"
            )
    resolved = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "models": {
            label: {
                "title": MODEL_SPECS[label]["title"],
                "lens": str(Path(MODEL_SPECS[label]["lens"]).resolve()),
                "lens_manifest_sha256": sha256(
                    Path(MODEL_SPECS[label]["lens"]) / "icalens.json"
                ),
                "model_id": lenses[label].model_id,
                "model_revision": lenses[label].model_revision,
                "layers": layers[label],
            }
            for label in labels
        },
        "source_language": "english",
        "target_language": args.target_language,
        "corpus": str(corpus_path),
        "corpus_sha256": sha256(corpus_path),
        "pairs": args.pairs,
        "pairs_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "pair_sampling": "seeded uniform sample without replacement",
        "seed": args.seed,
        "token_position": "last non-padding text token; EOS is not appended",
        "batch_size": args.batch_size,
        "top_components": args.top_components,
        "ranking": "absolute target-minus-English mean ICA score",
    }
    output = (
        args.output or HERE / "runs" / f"component-search-{args.target_language}"
    ).resolve()
    source = source_provenance()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="searching",
    )
    units = [(label, layer) for label in labels for layer in layers[label]]
    result_paths = {
        (label, layer): output / "results" / f"{label}-layer{layer:02d}.json"
        for label, layer in units
    }
    completed = {
        unit_id(label, layer)
        for label, layer in units
        if result_valid(
            result_paths[(label, layer)], unit_configuration(resolved, label, layer), resolved
        )
    }
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · steering showcase component search",
        completed=len(completed),
        total=len(units),
        completed_unit_ids=completed,
        unit_label="model-layers",
        source_dirty=bool(source.get("dirty")),
    )

    with display:
        warn_if_dirty(source)
        if len(completed) == len(units):
            display.phase("Complete")
            run.set_status("complete", complete=True)
            log(f"All component searches already complete: {output}")
            return
        for label in labels:
            pending_layers = [
                layer for layer in layers[label] if unit_id(label, layer) not in completed
            ]
            if not pending_layers:
                log(f"Reusing all completed {label} layers")
                continue
            lens = lenses[label]
            display.phase("Loading model", model=MODEL_SPECS[label]["title"])
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
            try:
                for layer in pending_layers:
                    display.phase(
                        "Encoding English",
                        model=MODEL_SPECS[label]["title"],
                        layer=layer,
                    )
                    english_mean = mean_final_scores(
                        model,
                        tokenizer,
                        [pair[0] for pair in pairs],
                        lens=lens,
                        layer=layer,
                        batch_size=args.batch_size,
                    )
                    display.phase(
                        f"Encoding {language['display']}",
                        model=MODEL_SPECS[label]["title"],
                        layer=layer,
                    )
                    target_mean = mean_final_scores(
                        model,
                        tokenizer,
                        [pair[1] for pair in pairs],
                        lens=lens,
                        layer=layer,
                        batch_size=args.batch_size,
                    )
                    contrast = target_mean - english_mean
                    count = min(args.top_components, len(contrast))
                    indices = torch.topk(contrast.abs(), k=count).indices.tolist()
                    candidates = [
                        {
                            "rank": rank,
                            "component": int(component),
                            "english_mean": float(english_mean[component]),
                            "target_mean": float(target_mean[component]),
                            "offset": float(contrast[component]),
                        }
                        for rank, component in enumerate(indices)
                    ]
                    payload = {
                        "format": FORMAT,
                        "schema_version": SCHEMA_VERSION,
                        "configuration": unit_configuration(resolved, label, layer),
                        "search": search_configuration(resolved),
                        "candidates": candidates,
                        "english_mean": english_mean.tolist(),
                        "target_mean": target_mean.tolist(),
                        "contrast": contrast.tolist(),
                    }
                    atomic_write_json(result_paths[(label, layer)], payload)
                    log(
                        f"{label} L{layer}: "
                        + ", ".join(
                            f"C{row['component']}={row['offset']:+.4f}"
                            for row in candidates[:3]
                        )
                    )
                    display.complete_unit(unit_id(label, layer), refresh=True)
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        display.phase("Complete")
        run.set_status("complete", complete=True)
        log(f"Component search complete: {output}")


def mean_final_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    lens: ICALens,
    layer: int,
    batch_size: int,
) -> torch.Tensor:
    width = lens._get_layer(layer).n_components
    total = torch.zeros(width, dtype=torch.float64)
    seen = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {name: value.to(device) for name, value in encoded.items()}
        attention_mask = inputs["attention_mask"]
        positions = attention_mask.shape[1] - 1 - attention_mask.flip(dims=(1,)).argmax(dim=1)
        hidden = capture_final_resid_post(
            model, model_inputs=inputs, layer=layer, positions=positions
        )
        with torch.inference_mode():
            scores = lens.transform(hidden.float(), layer=layer).double().sum(dim=0).cpu()
        total += scores
        seen += len(batch)
        print(f"  encoded {seen}/{len(texts)}", end="\r", flush=True)
    print()
    return (total / seen).float()


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


def parse_layer_spec(value: str) -> list[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid descending layer range {part!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected or min(selected) < 0:
        raise ValueError(f"invalid layer specification {value!r}")
    return sorted(selected)


def unit_id(label: str, layer: int) -> str:
    return f"{label}:layer-{layer}"


def unit_configuration(resolved: dict[str, Any], label: str, layer: int) -> dict[str, Any]:
    model = dict(resolved["models"][label])
    model.pop("layers")
    model["layer"] = layer
    return model


def search_configuration(resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        key: resolved[key]
        for key in (
            "source_language",
            "target_language",
            "pairs",
            "pairs_sha256",
            "seed",
            "token_position",
            "ranking",
        )
    }


def load_parallel_pairs(path: Path, *, url: str, count: int, seed: int) -> list[tuple[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        log(f"Downloading {url} to {path}")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": DOWNLOAD_USER_AGENT, "Accept": "application/zip,*/*"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
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
    available = [
        (row[0].strip(), row[1].strip())
        for row in csv.reader(io.StringIO(text), delimiter="\t")
        if len(row) >= 2 and row[0].strip() and row[1].strip()
    ]
    if count > len(available):
        raise ValueError(f"requested {count} pairs but corpus contains {len(available)}")
    return random.Random(seed).sample(available, count)


def result_valid(path: Path, model_config: dict[str, Any], resolved: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_search = search_configuration(resolved)
    return (
        payload.get("format") == FORMAT
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("configuration") == model_config
        and payload.get("search") == expected_search
        and isinstance(payload.get("candidates"), list)
        and isinstance(payload.get("contrast"), list)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
