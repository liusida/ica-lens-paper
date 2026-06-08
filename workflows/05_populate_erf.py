#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm.auto import tqdm


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "ica_lens").is_dir() and (parent / "server").is_dir():
            return parent
    raise RuntimeError("Could not find v6 project root.")


V6_ROOT = _project_root()
V6_SRC = V6_ROOT / "src"
if str(V6_SRC) not in sys.path:
    sys.path.insert(0, str(V6_SRC))
if str(V6_ROOT) not in sys.path:
    sys.path.insert(0, str(V6_ROOT))

from ica_lens.paths import RESULTS_DIR  # noqa: E402
from server.config import load_settings  # noqa: E402
from server.model_runtime import load_model_and_tokenizer  # noqa: E402
from server.probe import _layer_key_to_hidden_index, _load_fastica_artifact, fastica_artifact_path  # noqa: E402
from server.store import connect, list_components, validate_db  # noqa: E402


DEFAULT_MODELS = ("gpt2", "gemma2_2b", "qwen3_5_2b_base")
DEFAULT_DB_PATH = RESULTS_DIR / "reproduced" / "databases" / "ica_probe_reproduced.sqlite"
DEFAULT_ICA_ROOT = RESULTS_DIR / "reproduced" / "ica"
MAX_CONTEXT_TOKENS = 11
TOP_K = 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Populate effective receptive field diagnostics in an explorer DB.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=DEFAULT_MODELS)
    parser.add_argument("--layer", default=None, help="Only measure one layer, for example layer_06.")
    parser.add_argument("--component", type=int, default=None, help="Only measure one component id.")
    parser.add_argument("--limit", type=int, default=None, help="Measure at most this many components per model.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--token-budget", type=int, default=None, help="Use reproduced ICA dirs named <model>_tok<N>.")
    parser.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="Override server runtime device.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    db_path = args.db_path.resolve()
    with connect(db_path) as conn:
        validate_db(conn, None)
        _ensure_erf_table(conn)
        conn.commit()

    report: dict[str, Any] = {"status": "ok", "db_path": str(db_path), "models": {}}
    for model_name in args.models:
        model_settings = settings.models[model_name]
        ica_dir = _resolve_ica_dir(args.ica_root.resolve(), model_name=model_name, token_budget=args.token_budget)
        with connect(db_path) as conn:
            rows = list_components(conn, model_name, layer=args.layer, component=args.component)
            rows = _filter_missing(conn, rows, model_name=model_name, force=bool(args.force))
        if args.limit is not None:
            rows = rows[: max(0, int(args.limit))]
        report["models"][model_name] = {"selected_components": len(rows), "ica_dir": str(ica_dir)}
        print(f"{model_name}: {len(rows)} component(s) selected for ERF")
        if args.dry_run or not rows:
            continue

        runtime_device = str(args.device or settings.device)
        print(f"{model_name}: loading {model_settings.model_id}")
        runtime = load_model_and_tokenizer(model_settings.model_id, device=runtime_device, dtype=model_settings.dtype)
        for row in tqdm(rows, desc=model_name, unit="component", dynamic_ncols=True):
            with connect(db_path) as conn:
                examples = _list_component_examples(conn, model_name, layer=str(row["layer"]), component=int(row["component"]))
            evidence = _evidence_examples(examples)
            result = _measure_component_erf(
                row=row,
                examples=evidence,
                runtime=runtime,
                model_settings=model_settings,
                ica_dir=ica_dir,
                max_context_tokens=int(args.max_context_tokens),
                top_k=int(args.top_k),
                batch_size=int(args.batch_size),
            )
            if result is None:
                continue
            with connect(db_path) as conn:
                _ensure_erf_table(conn)
                _upsert_erf(
                    conn,
                    model_name=model_name,
                    row=row,
                    result=result,
                    max_context_tokens=int(args.max_context_tokens),
                    top_k=int(args.top_k),
                )
                conn.commit()
            if args.verbose:
                print(_format_component_result(model_name, row, result))

    report_path = db_path.with_name(db_path.stem + "_erf_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


def _measure_component_erf(
    *,
    row: dict[str, Any],
    examples: list[dict[str, Any]],
    runtime: tuple[Any, Any],
    model_settings: Any,
    ica_dir: Path,
    max_context_tokens: int,
    top_k: int,
    batch_size: int,
) -> dict[str, Any] | None:
    if not examples:
        return None
    direction = _evidence_direction(examples)
    sample_results = []
    for example in examples:
        sample = _measure_sample_erf(
            row=row,
            example=example,
            runtime=runtime,
            model_settings=model_settings,
            ica_dir=ica_dir,
            direction=direction,
            max_context_tokens=max_context_tokens,
            top_k=top_k,
            batch_size=batch_size,
        )
        if sample is not None:
            sample_results.append(sample)
    if not sample_results:
        return None
    values = [int(sample["erf"]) for sample in sample_results]
    return {
        "direction": direction,
        "sample_count": len(values),
        "min": min(values),
        "mean": mean(values),
        "max": max(values),
        "histogram": Counter(values),
        "samples": sample_results,
    }


def _measure_sample_erf(
    *,
    row: dict[str, Any],
    example: dict[str, Any],
    runtime: tuple[Any, Any],
    model_settings: Any,
    ica_dir: Path,
    direction: str | None,
    max_context_tokens: int,
    top_k: int,
    batch_size: int,
) -> dict[str, Any] | None:
    _, tokenizer = runtime
    text = str(example.get("context_to_target") or example.get("context") or "")
    recovered = _recovered_symbol_token(example)
    if recovered:
        text = text.replace("\ufffd", recovered)
    elif "\ufffd" in text or "\ufffd" in str(example.get("token") or ""):
        return None
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        return None
    target = tokenizer.decode([token_ids[-1]], clean_up_tokenization_spaces=False)
    if recovered:
        target = target.replace("\ufffd", recovered)
    elif "\ufffd" in target:
        return None

    jobs = []
    max_len = min(int(max_context_tokens), len(token_ids))
    for length in range(1, max_len + 1):
        suffix = tokenizer.decode(token_ids[-length:], clean_up_tokenization_spaces=False)
        if recovered:
            suffix = suffix.replace("\ufffd", recovered)
        elif "\ufffd" in suffix:
            continue
        for probe_text in _test_prompts_for_token_text(suffix):
            jobs.append({"length": length, "text": probe_text, "target": target})
    if not jobs:
        return None

    probes = _probe_texts_batched(
        jobs=jobs,
        runtime=runtime,
        layer=str(row["layer"]),
        component=int(row["component"]),
        model_settings=model_settings,
        ica_dir=ica_dir,
        top_k=top_k,
        batch_size=max(1, int(batch_size)),
    )
    probes_by_length: dict[int, list[dict[str, Any]]] = {}
    for probe in probes:
        probes_by_length.setdefault(int(probe["length"]), []).append(probe)

    for length in range(1, max_len + 1):
        matching = [probe for probe in probes_by_length.get(length, []) if _probe_matches_direction(probe, direction)]
        if matching:
            best = max(matching, key=lambda item: abs(float(item["score"])))
            return {
                "target": target,
                "erf": length,
                "passed_erf": length,
                "passed_probe_text": str(best.get("probe_text") or ""),
            }
    return {"target": target, "erf": int(max_context_tokens), "passed_erf": None, "passed_probe_text": None}


_ICA_ARTIFACT_DEVICE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _probe_texts_batched(
    *,
    jobs: list[dict[str, Any]],
    runtime: tuple[Any, Any],
    layer: str,
    component: int,
    model_settings: Any,
    ica_dir: Path,
    top_k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    model, tokenizer = runtime
    device = next(model.parameters()).device
    artifact = _cached_device_artifact(fastica_artifact_path(ica_dir, layer), device=device)
    hidden_index = _layer_key_to_hidden_index(layer, num_transformer_layers=int(model.config.num_hidden_layers))
    out = []
    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        encoded = [_encode_probe_job(tokenizer, job, model_settings=model_settings) for job in batch_jobs]
        max_len = max(len(item["input_ids"]) for item in encoded)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
        for i, item in enumerate(encoded):
            ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
            input_ids[i, : ids.numel()] = ids
            attention_mask[i, : ids.numel()] = 1
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[hidden_index].to(dtype=torch.float32)
            for i, item in enumerate(encoded):
                positions = [pos for pos in item["positions"] if 0 <= pos < len(item["input_ids"])]
                if not positions:
                    positions = list(range(len(item["input_ids"])))
                scores = [_component_score_for_hidden(hidden[i, position], component=component, artifact=artifact, top_k=top_k) for position in positions]
                best = max(scores, key=lambda score: abs(float(score["score"])))
                out.append({**best, "length": int(item["job"]["length"]), "probe_text": str(item["job"]["text"])})
    return out


def _encode_probe_job(tokenizer: Any, job: dict[str, Any], *, model_settings: Any) -> dict[str, Any]:
    text = str(job["text"])
    target = str(job["target"])
    input_ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=model_settings.context_length)["input_ids"]
    return {
        "job": job,
        "input_ids": input_ids,
        "positions": _target_token_positions_from_ids(input_ids, tokenizer(target, add_special_tokens=False)["input_ids"]),
    }


def _target_token_positions_from_ids(probe_ids: list[int], target_ids: list[int]) -> list[int]:
    if not target_ids or len(target_ids) > len(probe_ids):
        return []
    for start in range(len(probe_ids) - len(target_ids), -1, -1):
        if probe_ids[start : start + len(target_ids)] == target_ids:
            return list(range(start, start + len(target_ids)))
    return list(range(max(0, len(probe_ids) - len(target_ids)), len(probe_ids)))


def _cached_device_artifact(path: Path, *, device: torch.device) -> dict[str, Any]:
    key = (str(path.resolve()), str(device))
    cached = _ICA_ARTIFACT_DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    artifact = _load_fastica_artifact(path)
    out = {
        "mean": artifact["mean"].to(device=device, dtype=torch.float32),
        "components": artifact["components"].to(device=device, dtype=torch.float32),
        "norm_eps": float(artifact["norm_eps"]),
    }
    _ICA_ARTIFACT_DEVICE_CACHE[key] = out
    return out


def _component_score_for_hidden(hidden: torch.Tensor, *, component: int, artifact: dict[str, Any], top_k: int) -> dict[str, Any]:
    x = hidden.unsqueeze(0).to(dtype=torch.float32)
    normalized = x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(float(artifact["norm_eps"]))
    scores = ((normalized - artifact["mean"]) @ artifact["components"].T)[0]
    score = float(scores[int(component)].detach().cpu().item())
    k = min(int(top_k), int(scores.numel()))
    top = torch.topk(scores.abs(), k=k).indices
    return {"score": score, "in_top_k": bool((top == int(component)).any().detach().cpu().item())}


def _ensure_erf_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS effective_receptive_fields (
          model_name TEXT NOT NULL,
          layer TEXT NOT NULL,
          component INTEGER NOT NULL,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          direction TEXT,
          sample_count INTEGER NOT NULL,
          min_erf INTEGER NOT NULL,
          mean_erf REAL NOT NULL,
          max_erf INTEGER NOT NULL,
          histogram_json TEXT NOT NULL,
          max_context_tokens INTEGER NOT NULL,
          top_k INTEGER NOT NULL,
          PRIMARY KEY (model_name, layer, component)
        )
        """
    )


def _upsert_erf(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    row: dict[str, Any],
    result: dict[str, Any],
    max_context_tokens: int,
    top_k: int,
) -> None:
    histogram = {str(length): count for length, count in sorted(result["histogram"].items())}
    conn.execute(
        """
        INSERT INTO effective_receptive_fields(
          model_name, layer, component, updated_at, direction, sample_count,
          min_erf, mean_erf, max_erf, histogram_json, max_context_tokens, top_k
        ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_name, layer, component) DO UPDATE SET
          updated_at = excluded.updated_at,
          direction = excluded.direction,
          sample_count = excluded.sample_count,
          min_erf = excluded.min_erf,
          mean_erf = excluded.mean_erf,
          max_erf = excluded.max_erf,
          histogram_json = excluded.histogram_json,
          max_context_tokens = excluded.max_context_tokens,
          top_k = excluded.top_k
        """,
        (
            model_name,
            str(row["layer"]),
            int(row["component"]),
            result["direction"],
            int(result["sample_count"]),
            int(result["min"]),
            float(result["mean"]),
            int(result["max"]),
            json.dumps(histogram, sort_keys=True),
            int(max_context_tokens),
            int(top_k),
        ),
    )


def _list_component_examples(conn: sqlite3.Connection, model_name: str, *, layer: str, component: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT region, rank, token, source_score, context_to_target, context
        FROM examples
        WHERE model_name = ? AND layer = ? AND component = ?
        ORDER BY region, rank
        """,
        (model_name, layer, int(component)),
    ).fetchall()
    return [dict(row) for row in rows]


def _filter_missing(conn: sqlite3.Connection, rows: list[dict[str, Any]], *, model_name: str, force: bool) -> list[dict[str, Any]]:
    if force:
        return rows
    present = {
        (str(row["layer"]), int(row["component"]))
        for row in conn.execute("SELECT layer, component FROM effective_receptive_fields WHERE model_name = ?", (model_name,)).fetchall()
    }
    return [row for row in rows if (str(row["layer"]), int(row["component"])) not in present]


def _evidence_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [example for example in examples if example.get("source_score") is not None]
    if not scored:
        return []
    strongest = max(scored, key=lambda example: abs(float(example["source_score"])))
    strongest_score = float(strongest["source_score"])
    if strongest_score == 0:
        return []
    threshold = abs(strongest_score) / 2
    sign = 1 if strongest_score > 0 else -1
    evidence = []
    seen: set[tuple[str, float]] = set()
    for example in sorted(scored, key=lambda item: abs(float(item["source_score"])), reverse=True):
        score = float(example["source_score"])
        if abs(score) <= threshold or (score > 0) != (sign > 0):
            continue
        key = (str(example.get("token") or "").strip().lower(), round(score, 8))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(example)
    return evidence


def _evidence_direction(examples: list[dict[str, Any]]) -> str | None:
    scored = [example for example in examples if example.get("source_score") is not None]
    if not scored:
        return None
    score = float(max(scored, key=lambda example: abs(float(example["source_score"])))["source_score"])
    if score == 0:
        return None
    return "positive" if score > 0 else "negative"


def _probe_matches_direction(result: dict[str, Any], direction: str | None) -> bool:
    if not result["in_top_k"]:
        return False
    if direction == "positive":
        return float(result["score"]) > 0
    if direction == "negative":
        return float(result["score"]) < 0
    return True


def _test_prompts_for_token_text(text: str) -> list[str]:
    return list(dict.fromkeys([f"test: {text}", f"test-{text}"]))


def _recovered_symbol_token(example: dict[str, Any]) -> str | None:
    if "\ufffd" not in str(example.get("token") or ""):
        return None
    context = str(example.get("context_to_target") or "")
    for char in reversed(context):
        if char == "\ufffd" or char.isspace() or char.isalnum():
            continue
        return char
    return None


def _resolve_ica_dir(root: Path, *, model_name: str, token_budget: int | None) -> Path:
    candidates = []
    if token_budget is not None:
        candidates.append(root / f"{model_name}_tok{token_budget}")
    candidates.append(root / model_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _format_component_result(model_name: str, row: dict[str, Any], result: dict[str, Any]) -> str:
    histogram = ", ".join(f"{length}:{count}" for length, count in sorted(result["histogram"].items()))
    return (
        f"{model_name} {row['layer']} component={row['component']} "
        f"direction={result['direction'] or 'unknown'} samples={result['sample_count']} "
        f"ERF min={result['min']} mean={result['mean']:.2f} max={result['max']} hist=[{histogram}]"
    )


if __name__ == "__main__":
    raise SystemExit(main())
