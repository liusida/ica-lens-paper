#!/usr/bin/env python3
"""Reproduce the three hand-tuned ICA steering notebook demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from icalens import ICALens
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORMAT = "icalens.steering_showcase.generations"
SCHEMA_VERSION = 2

MODEL_SPECS = {
    "gpt2": {
        "title": "GPT-2 Small",
        "lens": ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k",
        "layer": 7,
        "component": 317,
        "prompts": {
            "original": "My favorite thing to do on the weekend is to",
        },
        "initial_steer": (317, -20.0),
        "steer": (317, 0.0),
        "max_new_tokens": 10,
        "demo": "weekend_bar_to_playing_with_kids",
    },
    "gemma2": {
        "title": "Gemma 2 2B",
        "lens": ROOT / "local-icalens-models/official/icalens-gemma-2-2b-pile10k",
        "layer": 20,
        "component": 105,
        "prompts": {
            "english": "Hangzhou is a historic city known for",
            "chinese": "杭州是一座历史悠久的城市，它以",
        },
        "initial_steer": (105, 35.0, 35.0),
        "steer": (105, 0.0, 30.0),
        "max_new_tokens": 24,
        "demo": "english_chinese_bidirectional",
    },
    "qwen9b": {
        "title": "Qwen 3.5 9B Base",
        "lens": ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k",
        "layer": 26,
        "component": 9,
        "prompts": {
            "english": "Deep learning is a machine learning technique that",
            "french": (
                "L’apprentissage profond est une technique d’apprentissage automatique qui"
            ),
        },
        "initial_steer": (9, -40.0, -40.0),
        "steer": (9, -20.0, -20.0),
        "max_new_tokens": 24,
        "demo": "english_french_bidirectional",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", *MODEL_SPECS),
        default=["all"],
        help="Models to run; defaults to all three.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the notebook-specific generation length for every selected model.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print token-aligned component scores before and after each intervention.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Resumable output directory; defaults to a model-selection-specific run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    labels = list(MODEL_SPECS) if args.models == ["all"] else list(dict.fromkeys(args.models))
    if "all" in labels:
        raise ValueError("--models all cannot be combined with individual model names")

    resolved = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "models": {
            label: serializable_spec(MODEL_SPECS[label]) for label in labels
        },
        "max_new_tokens_override": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "document_framing": "model",
        "steering_intervention": (
            "initial_steer on the final prompt position; configured steer on later positions"
        ),
        "ablation": "clamp selected component to zero at all positions",
    }
    selection = "all" if labels == list(MODEL_SPECS) else "-".join(labels)
    output = (
        args.output
        or HERE / "runs" / f"steering-demonstrations-model-framing-{selection}"
    ).expanduser().resolve()
    source = source_provenance()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="generating",
    )
    result_paths = {label: output / "results" / f"{label}.json" for label in labels}
    completed = {
        label
        for label, path in result_paths.items()
        if result_valid(path, resolved["models"][label], resolved)
    }
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · language steering showcase",
        completed=len(completed),
        total=len(labels),
        completed_unit_ids=completed,
        unit_label="models",
        source_dirty=bool(source.get("dirty")),
    )

    with display:
        warn_if_dirty(source)
        if len(completed) == len(labels):
            display.phase("Complete")
            run.set_status("complete", complete=True)
            log(f"All steering demonstrations already complete: {output}")
            return
        for label in labels:
            if label in completed:
                log(f"Reusing completed {label} demonstration: {result_paths[label]}")
                continue
            spec = MODEL_SPECS[label]
            max_new_tokens = args.max_new_tokens or int(spec["max_new_tokens"])
            display.phase(
                "Loading and generating",
                model=spec["title"],
                layer=spec["layer"],
                component=f"C{spec['component']}",
            )
            lens = ICALens.from_pretrained(spec["lens"])
            try:
                generations = run_demo(
                    lens,
                    label=label,
                    spec=spec,
                    max_new_tokens=max_new_tokens,
                    debug=args.debug,
                )
            finally:
                lens.unload_model()

            payload = {
                "format": FORMAT,
                "schema_version": SCHEMA_VERSION,
                "configuration": resolved["models"][label],
                "prompts": spec["prompts"],
                "max_new_tokens": max_new_tokens,
                "generations": generations,
            }
            atomic_write_json(result_paths[label], payload)
            for generation_name, text in generations.items():
                log(f"{label} {generation_name}: {text}")
            display.complete_unit(label, refresh=True)
        display.phase("Complete")
        run.set_status("complete", complete=True)
        log(f"Steering demonstrations complete: {output}")


def serializable_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": spec["title"],
        "lens": str(Path(spec["lens"]).resolve()),
        "layer": spec["layer"],
        "component": spec["component"],
        "demo": spec["demo"],
        "prompts": spec["prompts"],
        "max_new_tokens": spec["max_new_tokens"],
        "initial_steer": list(spec["initial_steer"]),
        "steer": list(spec["steer"]),
        "steering_scope": "current-position",
        "ablation_clamp": [spec["component"], 0.0],
    }


def run_demo(
    lens: ICALens,
    *,
    label: str,
    spec: dict[str, Any],
    max_new_tokens: int,
    debug: bool,
) -> dict[str, str]:
    """Run the same generations as the corresponding hand-tuned notebook."""
    common = {
        "document_framing": "model",
        "max_new_tokens": max_new_tokens,
    }
    if label == "gpt2":
        prompt = spec["prompts"]["original"]
        return {
            "baseline": lens.generate(prompt, **common),
            "steered_to_playing_with_kids": lens.generate(
                prompt,
                layer=int(spec["layer"]),
                initial_steer=spec["initial_steer"],
                steer=spec["steer"],
                steering_scope="current-position",
                debug=debug,
                **common,
            ),
        }

    target_language = "chinese" if label == "gemma2" else "french"
    english_prompt = spec["prompts"]["english"]
    target_prompt = spec["prompts"][target_language]
    return {
        "english_baseline": lens.generate(english_prompt, **common),
        f"{target_language}_baseline": lens.generate(target_prompt, **common),
        f"english_to_{target_language}_steered": lens.generate(
            english_prompt,
            layer=int(spec["layer"]),
            initial_steer=spec["initial_steer"],
            steer=spec["steer"],
            steering_scope="current-position",
            debug=debug,
            **common,
        ),
        f"{target_language}_to_english_ablated": lens.generate(
            target_prompt,
            layer=int(spec["layer"]),
            clamp=(int(spec["component"]), 0.0),
            debug=debug,
            **common,
        ),
    }


def result_valid(path: Path, model_config: dict[str, Any], resolved: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("format") == FORMAT
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("configuration") == model_config
        and payload.get("prompts") == model_config["prompts"]
        and payload.get("max_new_tokens")
        == (resolved["max_new_tokens_override"] or model_config["max_new_tokens"])
        and isinstance(payload.get("generations"), dict)
        and bool(payload["generations"])
    )


if __name__ == "__main__":
    main()
