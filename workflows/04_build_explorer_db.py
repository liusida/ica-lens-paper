#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer


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

from ica_lens.activation_store import iter_layer_shards_with_metadata, layer_shard_records, load_activation_manifest  # noqa: E402
from ica_lens.datasets import iter_dataset_texts  # noqa: E402
from ica_lens.paths import RESULTS_DIR  # noqa: E402
from server.store import init_db  # noqa: E402


DEFAULT_MODELS = ("gpt2", "gemma2_2b", "qwen3_5_2b_base")
DEFAULT_ACTIVATION_ROOT = RESULTS_DIR / "reproduced" / "activations"
DEFAULT_ICA_ROOT = RESULTS_DIR / "reproduced" / "ica"
DEFAULT_DB_PATH = RESULTS_DIR / "reproduced" / "databases" / "ica_probe_reproduced.sqlite"
DEFAULT_EXAMPLES_PER_REGION = 10
DEFAULT_CONTEXT_WINDOW = 24
DEFAULT_SCORE_BATCH_SIZE = 32768
NORM_EPS = 1e-12


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS components (
  model_name TEXT NOT NULL,
  layer TEXT NOT NULL,
  component INTEGER NOT NULL,
  selection_bin TEXT,
  excess_kurtosis REAL,
  n_components INTEGER,
  hidden_size INTEGER,
  PRIMARY KEY (model_name, layer, component)
);

CREATE TABLE IF NOT EXISTS examples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT NOT NULL,
  layer TEXT NOT NULL,
  component INTEGER NOT NULL,
  region TEXT NOT NULL,
  rank INTEGER NOT NULL,
  row_index INTEGER,
  doc_id INTEGER,
  position INTEGER,
  token_id INTEGER,
  token TEXT,
  source_score REAL,
  direction_cosine REAL,
  context_to_target TEXT,
  context TEXT,
  context_score_max_abs REAL,
  UNIQUE (model_name, layer, component, region, rank)
);

CREATE INDEX IF NOT EXISTS idx_examples_model_layer_comp
ON examples(model_name, layer, component);

CREATE INDEX IF NOT EXISTS idx_examples_model_layer_comp_region
ON examples(model_name, layer, component, region);

CREATE TABLE IF NOT EXISTS context_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  example_id INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  token_position INTEGER,
  token_id INTEGER,
  token TEXT,
  source_score REAL,
  direction_cosine REAL,
  is_target INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (example_id) REFERENCES examples(id) ON DELETE CASCADE,
  UNIQUE (example_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_context_tokens_example
ON context_tokens(example_id);

CREATE TABLE IF NOT EXISTS import_meta (
  model_name TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (model_name, key)
);
"""


@dataclass(frozen=True)
class ComponentStats:
    component: int
    excess_kurtosis: float
    n_components: int
    hidden_size: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the base ICA Lens explorer SQLite database from reproduced "
            "activation shards and FastICA artifacts."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--layers", nargs="+", default=None, help="Layer keys to export for every selected model.")
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--examples-per-region", type=int, default=DEFAULT_EXAMPLES_PER_REGION)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--score-batch-size", type=int, default=DEFAULT_SCORE_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-context", action="store_true", help="Do not reload dataset text or populate context/context_tokens.")
    parser.add_argument("--force", action="store_true", help="Replace an existing output DB.")
    args = parser.parse_args(argv)

    started_at = time.time()
    db_path = args.output_db.resolve()
    if db_path.exists() and args.force:
        db_path.unlink()
    elif db_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing DB without --force: {db_path}")

    conn = _connect_output_db(db_path)
    try:
        for model_name in args.models:
            _build_model(
                conn=conn,
                model_name=str(model_name),
                layers=args.layers,
                token_budget=int(args.token_budget),
                activation_root=args.activation_root.resolve(),
                ica_root=args.ica_root.resolve(),
                examples_per_region=int(args.examples_per_region),
                context_window=int(args.context_window),
                score_batch_size=int(args.score_batch_size),
                max_tokens=args.max_tokens,
                device=_resolve_device(str(args.device)),
                include_context=not bool(args.skip_context),
            )
        _analyze(conn)
    finally:
        conn.close()

    report = {
        "status": "ok",
        "analysis": "build_explorer_db",
        "output_db": str(db_path),
        "models": list(args.models),
        "layers": args.layers,
        "token_budget": int(args.token_budget),
        "settings": {
            "activation_root": str(args.activation_root.resolve()),
            "ica_root": str(args.ica_root.resolve()),
            "examples_per_region": int(args.examples_per_region),
            "context_window": int(args.context_window),
            "score_batch_size": int(args.score_batch_size),
            "max_tokens": args.max_tokens,
            "device": str(_resolve_device(str(args.device))),
            "include_context": not bool(args.skip_context),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    report_path = db_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote explorer DB: {db_path}")
    print(f"wrote build report: {report_path}")
    return 0


def _build_model(
    *,
    conn: sqlite3.Connection,
    model_name: str,
    layers: list[str] | None,
    token_budget: int,
    activation_root: Path,
    ica_root: Path,
    examples_per_region: int,
    context_window: int,
    score_batch_size: int,
    max_tokens: int | None,
    device: torch.device,
    include_context: bool,
) -> None:
    manifest_path = activation_root / f"{model_name}_tok{token_budget}" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing activation manifest: {manifest_path}")
    manifest = load_activation_manifest(manifest_path)
    selected_layers = _resolve_layers(layers, manifest)
    tokenizer = AutoTokenizer.from_pretrained(str(manifest["model"]["id"]), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    _write_import_meta(
        conn,
        model_name=model_name,
        metadata={
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "build_source": str(Path(__file__).resolve()),
            "activation_manifest": str(manifest_path.resolve()),
            "token_budget": str(token_budget),
            "layers": json.dumps(selected_layers),
        },
    )
    for layer in selected_layers:
        print(f"{model_name} {layer}: collecting component examples", flush=True)
        artifact_path = _ica_artifact_path(ica_root=ica_root, model_name=model_name, token_budget=token_budget, layer=layer)
        artifact = _load_artifact(artifact_path, device=device)
        scores, cosines, input_ids, doc_ids, positions = _collect_scores(
            capture_dir=manifest_path.parent,
            manifest=manifest,
            layer=layer,
            artifact=artifact,
            device=device,
            max_tokens=max_tokens,
            score_batch_size=score_batch_size,
        )
        components = _build_layer_components(
            model_name=model_name,
            layer=layer,
            manifest=manifest,
            artifact_path=artifact_path,
            scores=scores,
            cosines=cosines,
            input_ids=input_ids,
            doc_ids=doc_ids,
            positions=positions,
            tokenizer=tokenizer,
            examples_per_region=examples_per_region,
            context_window=context_window,
            include_context=include_context,
        )
        if include_context:
            _attach_contexts(
                manifest=manifest,
                tokenizer=tokenizer,
                components=components,
                context_window=context_window,
            )
        _replace_layer_rows(conn, model_name=model_name, layer=layer, components=components)
        conn.commit()
        del scores, cosines, input_ids, doc_ids, positions, artifact
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _build_layer_components(
    *,
    model_name: str,
    layer: str,
    manifest: dict[str, Any],
    artifact_path: Path,
    scores: torch.Tensor,
    cosines: torch.Tensor,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    positions: torch.Tensor,
    tokenizer: Any,
    examples_per_region: int,
    context_window: int,
    include_context: bool,
) -> list[dict[str, Any]]:
    hidden_size = int(manifest["model"]["hidden_size"])
    row_lookup = _row_index_by_doc_position(doc_ids=doc_ids, positions=positions)
    components: list[dict[str, Any]] = []
    for stat in tqdm(_component_stats(scores=scores, hidden_size=hidden_size), desc=f"{model_name} {layer}", unit="component", dynamic_ncols=True):
        component_scores = scores[:, stat.component]
        component_cosines = cosines[:, stat.component]
        components.append(
            {
                "component": stat.component,
                "selection_bin": "full_model",
                "excess_kurtosis": stat.excess_kurtosis,
                "n_components": stat.n_components,
                "hidden_size": stat.hidden_size,
                "artifact_path": str(artifact_path),
                "examples": _examples_for_component(
                    source_scores=component_scores,
                    direction_cosines=component_cosines,
                    input_ids=input_ids,
                    doc_ids=doc_ids,
                    positions=positions,
                    tokenizer=tokenizer,
                    examples_per_region=examples_per_region,
                    context_window=context_window,
                    row_lookup=row_lookup,
                    include_context_scores=include_context,
                ),
            }
        )
    return components


def _collect_scores(
    *,
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: str,
    artifact: dict[str, Any],
    device: torch.device,
    max_tokens: int | None,
    score_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    components = artifact["components"].to(device=device, dtype=torch.float32)
    cosine_components = _normalize_rows(components)
    mean = artifact["mean"].to(device=device, dtype=torch.float32)
    preprocess = str(artifact.get("preprocess") or "with_normalization")

    score_chunks: list[torch.Tensor] = []
    cosine_chunks: list[torch.Tensor] = []
    input_id_chunks: list[torch.Tensor] = []
    doc_id_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    tokens_seen = 0
    total = max_tokens or _layer_token_count(manifest, layer)
    pbar = tqdm(total=total, unit="tok", dynamic_ncols=True, desc=f"{layer} scores")
    try:
        for shard, activations in iter_layer_shards_with_metadata(capture_dir=capture_dir, manifest=manifest, layer=layer):
            remaining = None if max_tokens is None else max_tokens - tokens_seen
            if remaining is not None and remaining <= 0:
                break
            take = int(activations.shape[0]) if remaining is None else min(remaining, int(activations.shape[0]))
            shard_scores: list[torch.Tensor] = []
            shard_cosines: list[torch.Tensor] = []
            for batch in _activation_batches(activations[:take], batch_size=score_batch_size):
                values = batch.to(device=device, dtype=torch.float32)
                normalized = _normalize_rows(values) if preprocess == "with_normalization" else values
                centered = normalized - mean
                shard_scores.append((centered @ components.T).detach().cpu())
                shard_cosines.append((normalized @ cosine_components.T).detach().cpu())
            score_chunks.append(torch.cat(shard_scores, dim=0))
            cosine_chunks.append(torch.cat(shard_cosines, dim=0))
            input_id_chunks.append(torch.load(capture_dir / shard["input_ids"], map_location="cpu")[:take])
            doc_id_chunks.append(torch.load(capture_dir / shard["doc_ids"], map_location="cpu")[:take])
            position_chunks.append(torch.load(capture_dir / shard["positions"], map_location="cpu")[:take])
            tokens_seen += take
            pbar.update(take)
    finally:
        pbar.close()
    return (
        torch.cat(score_chunks, dim=0),
        torch.cat(cosine_chunks, dim=0),
        torch.cat(input_id_chunks, dim=0),
        torch.cat(doc_id_chunks, dim=0),
        torch.cat(position_chunks, dim=0),
    )


def _component_stats(*, scores: torch.Tensor, hidden_size: int) -> list[ComponentStats]:
    x = scores.to(dtype=torch.float64)
    mean = x.mean(dim=0)
    second = x.square().mean(dim=0)
    third = x.pow(3).mean(dim=0)
    fourth = x.pow(4).mean(dim=0)
    variance = (second - mean.square()).clamp_min(1e-30)
    central_fourth = fourth - 4 * mean * third + 6 * mean.square() * second - 3 * mean.pow(4)
    excess = central_fourth / variance.square() - 3.0
    n_components = int(scores.shape[1])
    return [
        ComponentStats(
            component=index,
            excess_kurtosis=float(excess[index].item()),
            n_components=n_components,
            hidden_size=hidden_size,
        )
        for index in range(n_components)
    ]


def _examples_for_component(
    *,
    source_scores: torch.Tensor,
    direction_cosines: torch.Tensor,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    positions: torch.Tensor,
    tokenizer: Any,
    examples_per_region: int,
    context_window: int,
    row_lookup: dict[tuple[int, int], int],
    include_context_scores: bool,
) -> dict[str, list[dict[str, Any]]]:
    regions = _region_indices(values=source_scores, count=examples_per_region)
    return {
        region: [
            _example_row(
                source_scores=source_scores,
                direction_cosines=direction_cosines,
                input_ids=input_ids,
                doc_ids=doc_ids,
                positions=positions,
                index=int(index),
                tokenizer=tokenizer,
                context_window=context_window,
                row_lookup=row_lookup,
                include_context_scores=include_context_scores,
            )
            for index in indices
        ]
        for region, indices in regions.items()
    }


def _region_indices(*, values: torch.Tensor, count: int) -> dict[str, list[int]]:
    n = int(values.numel())
    if n <= 0 or count <= 0:
        return {}
    if n <= count:
        return {"all": torch.arange(n, dtype=torch.long).tolist()}
    abs_values = values.abs()
    top_abs_pool_size = min(n, max(count, 5000))
    top_abs_pool = torch.topk(abs_values, k=top_abs_pool_size, largest=True).indices
    top_abs = [int(index) for index in top_abs_pool[:count].tolist()]
    return {
        "top_abs": top_abs,
        "top_abs_sample_500": _sample_top_abs_indices(top_abs_pool, top_abs, 500, count),
        "top_abs_sample_5000": _sample_top_abs_indices(top_abs_pool, top_abs, 5000, count),
        "opposite_top": _opposite_top_indices(values, top_abs, count),
        "baseline_near_zero": torch.topk(abs_values, k=min(count, n), largest=False).indices.tolist(),
    }


def _example_row(
    *,
    source_scores: torch.Tensor,
    direction_cosines: torch.Tensor,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    positions: torch.Tensor,
    index: int,
    tokenizer: Any,
    context_window: int,
    row_lookup: dict[tuple[int, int], int],
    include_context_scores: bool,
) -> dict[str, Any]:
    token_id = int(input_ids[index])
    row = {
        "row_index": index,
        "source_score": float(source_scores[index]),
        "direction_cosine": float(direction_cosines[index]),
        "token_id": token_id,
        "token": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
        "doc_id": int(doc_ids[index]),
        "position": int(positions[index]),
    }
    if include_context_scores:
        context_scores = _context_scores_by_position(
            source_scores=source_scores,
            direction_cosines=direction_cosines,
            doc_ids=doc_ids,
            positions=positions,
            index=index,
            context_window=context_window,
            row_lookup=row_lookup,
        )
        row["context_scores_by_position"] = context_scores
        row["context_score_max_abs"] = max([abs(float(item["source_score"])) for item in context_scores], default=0.0)
    else:
        row["context_score_max_abs"] = 0.0
    return row


def _context_scores_by_position(
    *,
    source_scores: torch.Tensor,
    direction_cosines: torch.Tensor,
    doc_ids: torch.Tensor,
    positions: torch.Tensor,
    index: int,
    context_window: int,
    row_lookup: dict[tuple[int, int], int],
) -> list[dict[str, Any]]:
    doc_id = int(doc_ids[index])
    target_position = int(positions[index])
    out = []
    for position in range(max(0, target_position - context_window), target_position + context_window + 1):
        candidate = row_lookup.get((doc_id, position))
        if candidate is None:
            continue
        out.append(
            {
                "position": position,
                "source_score": float(source_scores[candidate]),
                "direction_cosine": float(direction_cosines[candidate]),
                "is_target": candidate == index,
            }
        )
    return out


def _attach_contexts(
    *,
    manifest: dict[str, Any],
    tokenizer: Any,
    components: list[dict[str, Any]],
    context_window: int,
) -> None:
    requests: dict[int, set[int]] = {}
    for component in components:
        for examples in component["examples"].values():
            for example in examples:
                requests.setdefault(int(example["doc_id"]), set()).add(int(example["position"]))
    contexts = _load_contexts(manifest=manifest, tokenizer=tokenizer, doc_positions=requests, context_window=context_window)
    for component in components:
        for examples in component["examples"].values():
            for example in examples:
                context = contexts.get((int(example["doc_id"]), int(example["position"])))
                if context is None:
                    continue
                example.update(context)
                _attach_context_token_scores(example)


def _load_contexts(
    *,
    manifest: dict[str, Any],
    tokenizer: Any,
    doc_positions: dict[int, set[int]],
    context_window: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    doc_positions = {doc_id: positions for doc_id, positions in doc_positions.items() if doc_id >= 0}
    if not doc_positions:
        return {}
    max_doc_id = max(doc_positions)
    contexts: dict[tuple[int, int], dict[str, Any]] = {}
    texts = iter_dataset_texts(
        path=str(manifest["dataset"]["path"]),
        name=manifest["dataset"].get("name"),
        split=str(manifest["dataset"]["split"]),
        text_column=str(manifest["dataset"]["text_column"]),
        streaming=bool(manifest["dataset"].get("streaming", False)),
    )
    for doc_id, text in enumerate(tqdm(texts, total=max_doc_id + 1, desc="load contexts", dynamic_ncols=True)):
        if doc_id > max_doc_id:
            break
        positions = doc_positions.get(doc_id)
        if not positions:
            continue
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(manifest["capture"]["context_length"]),
            return_offsets_mapping=True,
        )
        token_ids_tensor = encoded["input_ids"][0]
        offsets = encoded["offset_mapping"][0].tolist()
        for position in positions:
            start = max(0, position - context_window)
            stop = min(int(token_ids_tensor.numel()), position + context_window + 1)
            token_ids = token_ids_tensor[start:stop].tolist()
            token_ids_to_target = token_ids_tensor[start : position + 1].tolist()
            contexts[(doc_id, position)] = {
                "context": tokenizer.decode(token_ids, clean_up_tokenization_spaces=False),
                "context_to_target": tokenizer.decode(token_ids_to_target, clean_up_tokenization_spaces=False),
                "context_token_entries": [
                    {
                        "token_id": int(token_id),
                        "token": _token_text_from_offsets(
                            text=text,
                            offsets=offsets,
                            token_index=start + offset,
                            fallback=tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
                        ),
                        "position": start + offset,
                        "is_target": start + offset == position,
                    }
                    for offset, token_id in enumerate(token_ids)
                ],
            }
    return contexts


def _attach_context_token_scores(example: dict[str, Any]) -> None:
    scores = {int(item["position"]): item for item in example.get("context_scores_by_position", [])}
    token_scores = []
    for token in example.get("context_token_entries", []):
        score = scores.get(int(token["position"]))
        token_scores.append(
            {
                **token,
                "source_score": None if score is None else float(score["source_score"]),
                "direction_cosine": None if score is None else float(score["direction_cosine"]),
            }
        )
    example["context_token_scores"] = token_scores
    example.pop("context_scores_by_position", None)
    example.pop("context_token_entries", None)


def _replace_layer_rows(conn: sqlite3.Connection, *, model_name: str, layer: str, components: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM context_tokens WHERE example_id IN (SELECT id FROM examples WHERE model_name = ? AND layer = ?)", (model_name, layer))
    conn.execute("DELETE FROM examples WHERE model_name = ? AND layer = ?", (model_name, layer))
    conn.execute("DELETE FROM components WHERE model_name = ? AND layer = ?", (model_name, layer))
    for component in components:
        component_id = int(component["component"])
        conn.execute(
            """
            INSERT INTO components(
              model_name, layer, component, selection_bin, excess_kurtosis, n_components, hidden_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                layer,
                component_id,
                str(component.get("selection_bin") or ""),
                float(component.get("excess_kurtosis") or 0.0),
                int(component.get("n_components") or 0),
                int(component.get("hidden_size") or 0),
            ),
        )
        for region, examples in (component.get("examples") or {}).items():
            for rank, example in enumerate(examples, start=1):
                cur = conn.execute(
                    """
                    INSERT INTO examples(
                      model_name, layer, component, region, rank, row_index, doc_id, position,
                      token_id, token, source_score, direction_cosine,
                      context_to_target, context, context_score_max_abs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                    """,
                    (
                        model_name,
                        layer,
                        component_id,
                        str(region),
                        rank,
                        int(example.get("row_index")),
                        int(example.get("doc_id")),
                        int(example.get("position")),
                        int(example.get("token_id")),
                        str(example.get("token") or ""),
                        float(example.get("source_score")),
                        float(example.get("direction_cosine")),
                        str(example.get("context_to_target") or ""),
                        str(example.get("context") or ""),
                        float(example.get("context_score_max_abs") or 0.0),
                    ),
                )
                example_id = int(cur.fetchone()[0])
                for seq, token in enumerate(example.get("context_token_scores") or []):
                    conn.execute(
                        """
                        INSERT INTO context_tokens(
                          example_id, seq, token_position, token_id, token,
                          source_score, direction_cosine, is_target
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            example_id,
                            seq,
                            int(token.get("position")),
                            int(token.get("token_id")),
                            str(token.get("token") or ""),
                            None if token.get("source_score") is None else float(token["source_score"]),
                            None if token.get("direction_cosine") is None else float(token["direction_cosine"]),
                            1 if token.get("is_target") else 0,
                        ),
                    )


def _connect_output_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    conn.executescript(DB_SCHEMA)
    conn.commit()
    return conn


def _write_import_meta(conn: sqlite3.Connection, *, model_name: str, metadata: dict[str, str]) -> None:
    for key, value in metadata.items():
        conn.execute(
            """
            INSERT INTO import_meta(model_name, key, value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(model_name, key) DO UPDATE SET
              value = excluded.value,
              updated_at = excluded.updated_at
            """,
            (model_name, key, value),
        )
    conn.commit()


def _analyze(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ANALYZE")
        conn.commit()
    except sqlite3.DatabaseError:
        pass


def _load_artifact(path: Path, *, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ICA artifact: {path}")
    blob = torch.load(path, map_location="cpu")
    tensors = blob["tensors"]
    mean = tensors["mean"].to(dtype=torch.float32, device=device)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    return {
        "mean": mean,
        "components": tensors["components"].to(dtype=torch.float32, device=device),
        "preprocess": blob.get("metadata", {}).get("preprocess", "with_normalization"),
    }


def _resolve_layers(requested: list[str] | None, manifest: dict[str, Any]) -> list[str]:
    available = list(manifest["capture"]["layers"])
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested layer(s) not present in activation manifest: {', '.join(missing)}")
    return list(requested)


def _ica_artifact_path(*, ica_root: Path, model_name: str, token_budget: int, layer: str) -> Path:
    reproduced = ica_root / f"{model_name}_tok{token_budget}" / f"{layer}_fastica.pt"
    if reproduced.is_file():
        return reproduced
    fetched_style = ica_root / model_name / f"{layer}_fastica.pt"
    if fetched_style.is_file():
        return fetched_style
    return reproduced


def _layer_token_count(manifest: dict[str, Any], layer: str) -> int:
    return sum(int(shard.get("tokens", 0)) for shard in layer_shard_records(manifest, layer))


def _activation_batches(activations: torch.Tensor, *, batch_size: int) -> Iterable[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("--score-batch-size must be positive.")
    for start in range(0, int(activations.shape[0]), batch_size):
        yield activations[start : start + batch_size]


def _row_index_by_doc_position(*, doc_ids: torch.Tensor, positions: torch.Tensor) -> dict[tuple[int, int], int]:
    return {
        (int(doc_id), int(position)): index
        for index, (doc_id, position) in enumerate(zip(doc_ids.tolist(), positions.tolist(), strict=True))
    }


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(NORM_EPS)


def _sample_top_abs_indices(top_abs_pool: torch.Tensor, top_abs_indices: list[int], pool_size: int, count: int) -> list[int]:
    excluded = set(top_abs_indices)
    pool = [int(index) for index in top_abs_pool[:pool_size].tolist() if int(index) not in excluded]
    return _evenly_sample_values(pool, count)


def _opposite_top_indices(values: torch.Tensor, top_abs_indices: list[int], count: int) -> list[int]:
    if not top_abs_indices:
        return []
    dominant_sign = 1 if float(values[top_abs_indices[0]]) >= 0 else -1
    opposite_strength = (-dominant_sign * values).clamp_min(0)
    k = min(max(count + len(top_abs_indices), count), int(values.numel()))
    candidates = torch.topk(opposite_strength, k=k, largest=True).indices.tolist()
    excluded = set(top_abs_indices)
    return [int(index) for index in candidates if int(index) not in excluded and float(opposite_strength[int(index)]) > 0.0][:count]


def _evenly_sample_values(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    positions = torch.linspace(0, len(values) - 1, steps=count).round().to(dtype=torch.long).tolist()
    return [values[int(position)] for position in positions]


def _token_text_from_offsets(*, text: str, offsets: list[list[int]], token_index: int, fallback: str) -> str:
    if token_index < 0 or token_index >= len(offsets):
        return fallback
    start, stop = offsets[token_index]
    if stop <= start:
        return fallback
    return text[start:stop]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return torch.device(value)


if __name__ == "__main__":
    raise SystemExit(main())
