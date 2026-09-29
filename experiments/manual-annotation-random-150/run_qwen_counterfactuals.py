#!/usr/bin/env python3
"""Measure controlled counterfactual prompts for selected Qwen ICA components."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens.analysis import capture
from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LENS_PATH = ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k"
DEFAULT_TESTS = HERE / "qwen-counterfactual-tests.json"
DEFAULT_OUTPUT = HERE / "results-counterfactual"
FORMAT = "icalens.manual_annotation_counterfactual_results"
SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_test_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.get("format") != "icalens.manual_annotation_counterfactual_tests":
        raise ValueError("unrecognized counterfactual-test format")
    components = spec.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("test specification must contain components")
    seen: set[tuple[int, int]] = set()
    for component in components:
        key = (int(component["layer"]), int(component["component"]))
        if key in seen:
            raise ValueError(f"duplicate component L{key[0]} C{key[1]}")
        seen.add(key)
        if component.get("dominant_sign") not in {"positive", "negative"}:
            raise ValueError(f"invalid dominant sign for L{key[0]} C{key[1]}")
        tests = component.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"L{key[0]} C{key[1]} has no tests")
        for test in tests:
            if test.get("expected") not in {"activate", "not_activate"}:
                raise ValueError(f"invalid expectation for L{key[0]} C{key[1]}")
            prompt, target = str(test["prompt"]), str(test["target"])
            if prompt.count(target) != 1:
                raise ValueError(f"target {target!r} must occur exactly once in prompt {prompt!r}")
    return components


def validate_against_annotations(components: list[dict[str, Any]]) -> None:
    annotations = read_object(HERE / "annotations.json")
    records = annotations.get("qwen3.5-9b")
    if not isinstance(records, list):
        raise ValueError("annotations.json has no qwen3.5-9b annotation list")
    qwen = {(int(record["layer"]), int(record["component"])): record for record in records}
    for component in components:
        key = (int(component["layer"]), int(component["component"]))
        record = qwen.get(key)
        if record is None:
            raise ValueError(f"L{key[0]} C{key[1]} is not in the Qwen random sample")
        if str(record.get("label")) != str(component["label"]):
            raise ValueError(
                f"L{key[0]} C{key[1]} label differs from annotations.json: "
                f"{component['label']!r} != {record.get('label')!r}"
            )


def resolved_configuration(
    lens: ICALens,
    tests_path: Path,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = LENS_PATH / "icalens.json"
    manifest = read_object(manifest_path)
    layers: dict[str, Any] = {}
    for component in components:
        layer = int(component["layer"])
        artifact = LENS_PATH / manifest["layers"][str(layer)]["file"]
        layers[str(layer)] = {
            "artifact": str(artifact.resolve()),
            "artifact_sha256": sha256(artifact),
            "n_components": int(manifest["layers"][str(layer)]["n_components"]),
        }
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "tests": str(tests_path.resolve()),
        "tests_sha256": sha256(tests_path),
        "annotations_sha256": sha256(HERE / "annotations.json"),
        "lens": str(LENS_PATH.resolve()),
        "lens_manifest_sha256": sha256(manifest_path),
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "activation_site": lens.activation_site,
        "document_framing": "use the ICA layer's recorded fitting policy",
        "target_selection": "unique literal character span mapping to exactly one token",
        "rank": "one-based competition rank by descending absolute ICA score",
        "layers": layers,
    }


def unit_identity(resolved: dict[str, Any], component: dict[str, Any]) -> dict[str, Any]:
    layer = int(component["layer"])
    return {
        key: resolved[key]
        for key in (
            "format",
            "schema_version",
            "tests_sha256",
            "annotations_sha256",
            "lens_manifest_sha256",
            "model_id",
            "model_revision",
            "activation_site",
            "document_framing",
            "target_selection",
            "rank",
        )
    } | {
        "layer": layer,
        "component": int(component["component"]),
        "component_spec": component,
        "layer_dependency": resolved["layers"][str(layer)],
    }


def target_token_index(tokenizer: Any, prompt: str, target: str) -> int:
    start = prompt.index(target)
    end = start + len(target)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    exact = [index for index, pair in enumerate(offsets) if pair == (start, end)]
    if len(exact) == 1:
        return exact[0]
    overlapping = [
        (index, pair, tokenizer.convert_ids_to_tokens(encoded["input_ids"][index]))
        for index, pair in enumerate(offsets)
        if pair[0] < end and pair[1] > start
    ]
    if len(overlapping) == 1:
        index, (token_start, token_end), _ = overlapping[0]
        if (
            token_start <= start
            and token_end >= end
            and prompt[token_start:start].isspace()
            and not prompt[end:token_end]
        ):
            return index
    raise ValueError(
        f"target {target!r} in {prompt!r} must map to exactly one token; "
        f"overlapping tokens: {overlapping}"
    )


def competition_rank_by_absolute_score(scores: torch.Tensor, component: int) -> int:
    absolute = scores.abs()
    return int((absolute > absolute[component]).sum().item()) + 1


def measure_component(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    component_spec: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    layer = int(component_spec["layer"])
    component = int(component_spec["component"])
    rows = []
    for index, test in enumerate(component_spec["tests"], start=1):
        prompt, target = str(test["prompt"]), str(test["target"])
        token_index = target_token_index(tokenizer, prompt, target)
        captured = capture(
            lens,
            prompt,
            layer=layer,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        scores = lens.transform(captured.activations.float(), layer=layer)
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores)
        token_scores = scores[token_index].detach().cpu().float()
        rows.append(
            {
                "test_index": index,
                "expected": str(test["expected"]),
                "prompt": prompt,
                "target": target,
                "target_token_index": token_index,
                "target_token": captured.tokens[token_index],
                "target_token_text": captured.token_texts[token_index],
                "score": float(token_scores[component]),
                "absolute_score_rank": competition_rank_by_absolute_score(token_scores, component),
            }
        )
    return {
        "format": FORMAT + ".component",
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "component": component,
        "label": str(component_spec["label"]),
        "dominant_sign": str(component_spec["dominant_sign"]),
        "tests": rows,
    }


def checkpoint_path(output: Path, layer: int, component: int) -> Path:
    return output / f"layer_{layer:02d}-C{component:05d}.json"


def validate_checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_object(path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    tests = value.get("tests")
    if (
        value.get("identity_sha256") != identity_sha256(identity)
        or value.get("format") != FORMAT + ".component"
        or not isinstance(tests, list)
        or len(tests) != len(identity["component_spec"]["tests"])
        or any(
            not isinstance(row.get("score"), (int, float))
            or not isinstance(row.get("absolute_score_rank"), int)
            or row["absolute_score_rank"] < 1
            for row in tests
        )
    ):
        return None
    return value


def write_checkpoint(path: Path, identity: dict[str, Any], result: dict[str, Any]) -> None:
    atomic_write_json(path, result | {"identity_sha256": identity_sha256(identity)})
    if validate_checkpoint(path, identity) is None:
        raise ValueError(f"new checkpoint failed validation: {path}")


def write_summary(output: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        for test in result["tests"]:
            rows.append(
                {
                    "layer": result["layer"],
                    "component": result["component"],
                    "label": result["label"],
                    "dominant_sign": result["dominant_sign"],
                    **test,
                }
            )
    atomic_write_json(
        output / "summary.json",
        {
            "format": FORMAT + ".summary",
            "schema_version": SCHEMA_VERSION,
            "rows": rows,
        },
    )
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")


def load_model(lens: ICALens, device: str) -> torch.nn.Module:
    if not device.startswith("cuda"):
        raise ValueError("this GB10 experiment currently requires a CUDA device")
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        lens.model_id,
        revision=lens.model_revision,
        device=device,
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tests_path = args.tests.expanduser().resolve()
    spec = read_object(tests_path)
    components = validate_test_spec(spec)
    validate_against_annotations(components)
    lens = ICALens.from_pretrained(LENS_PATH)
    if lens.activation_site != "resid_post":
        raise ValueError("the selected Qwen ICA Lens does not use resid_post activations")
    resolved = resolved_configuration(lens, tests_path, components)
    output = args.output.expanduser().resolve()
    source = source_provenance()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source,
        status="measuring",
    )

    cached: dict[tuple[int, int], dict[str, Any]] = {}
    completed: set[tuple[int, int]] = set()
    for component_spec in components:
        layer = int(component_spec["layer"])
        component = int(component_spec["component"])
        identity = unit_identity(resolved, component_spec)
        path = checkpoint_path(output, layer, component)
        value = None if args.force else validate_checkpoint(path, identity)
        if value is not None:
            cached[layer, component] = value
            completed.add((layer, component))

    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · Qwen counterfactual tests",
        completed=len(completed),
        total=len(components),
        completed_unit_ids=completed,
        unit_label="components",
        recent_label="Recent counterfactual-test output",
        detail_filename="counterfactual-test-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        log(
            f"Validated configuration: {len(components)} components, "
            f"{sum(len(item['tests']) for item in components)} prompts."
        )
        pending = [
            item
            for item in components
            if (int(item["layer"]), int(item["component"])) not in completed
        ]
        if pending:
            display.phase("Loading Qwen language model")
            tokenizer = AutoTokenizer.from_pretrained(
                lens.model_id,
                revision=lens.model_revision,
                use_fast=True,
            )
            model = load_model(lens, args.device)
            try:
                for component_spec in pending:
                    layer = int(component_spec["layer"])
                    component = int(component_spec["component"])
                    display.phase(
                        "Measuring controlled prompts",
                        model="qwen3.5-9b",
                        layer=layer,
                        component=f"C{component}",
                    )
                    result = measure_component(
                        lens=lens,
                        model=model,
                        tokenizer=tokenizer,
                        component_spec=component_spec,
                        device=args.device,
                    )
                    identity = unit_identity(resolved, component_spec)
                    path = checkpoint_path(output, layer, component)
                    write_checkpoint(path, identity, result)
                    cached[layer, component] = read_object(path)
                    display.complete_unit((layer, component), refresh=True)
                    log(
                        f"L{layer} C{component}: checkpointed "
                        f"{len(component_spec['tests'])} controlled prompts."
                    )
            finally:
                del model
                torch.cuda.empty_cache()
        else:
            log("All requested components are complete; reusing checkpoints.")

        display.phase("Writing summary")
        ordered = [cached[int(item["layer"]), int(item["component"])] for item in components]
        write_summary(output, ordered)
        run.set_status("complete", complete=True)
        log(f"Counterfactual tests complete: {output}")


if __name__ == "__main__":
    main()
