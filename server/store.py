from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
  model_name TEXT NOT NULL,
  layer TEXT NOT NULL,
  component INTEGER NOT NULL,
  label TEXT,
  confidence TEXT,
  positive_label TEXT,
  positive_confidence TEXT,
  negative_label TEXT,
  negative_confidence TEXT,
  positive_interpretation_types_json TEXT,
  negative_interpretation_types_json TEXT,
  interpretation_types_json TEXT,
  summary TEXT,
  notes TEXT,
  include_as_case_study INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (model_name, layer, component)
);

CREATE TABLE IF NOT EXISTS chosen_random_components (
  selection_name TEXT NOT NULL,
  model_name TEXT NOT NULL,
  selection_index INTEGER NOT NULL,
  layer TEXT NOT NULL,
  component INTEGER NOT NULL,
  component_id TEXT,
  source_json TEXT,
  seed INTEGER,
  requested_n INTEGER,
  inventory_size INTEGER,
  selected_size INTEGER,
  fit_converged INTEGER,
  fit_iterations INTEGER,
  fit_final_lim REAL,
  fit_final_lim_p95 REAL,
  fit_seed INTEGER,
  inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (selection_name, model_name, selection_index)
);

CREATE INDEX IF NOT EXISTS idx_chosen_random_components_component
ON chosen_random_components(model_name, layer, component);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _ensure_annotation_columns(conn)
    conn.commit()


def validate_db(conn: sqlite3.Connection, model_name: str | None = None) -> None:
    tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    missing = {"components", "examples", "annotations"} - tables
    if missing:
        raise RuntimeError(f"SQLite DB is missing required table(s): {', '.join(sorted(missing))}")
    if model_name:
        count = conn.execute("SELECT COUNT(*) FROM components WHERE model_name = ?", (model_name,)).fetchone()[0]
        if int(count) == 0:
            raise RuntimeError(f"SQLite DB has no components for model_name={model_name!r}")


def list_models(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT model_name FROM components
        UNION
        SELECT model_name FROM annotations
        ORDER BY model_name
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def list_layers(conn: sqlite3.Connection, model_name: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT layer FROM components WHERE model_name = ?",
        (model_name,),
    ).fetchall()
    return sorted([str(row[0]) for row in rows], key=layer_sort_key)


def list_components(conn: sqlite3.Connection, model_name: str, layer: str | None = None, component: int | None = None, search: str | None = None) -> list[dict[str, Any]]:
    erf_join, erf_select = _erf_join_sql(conn)
    rows = conn.execute(
        f"""
        SELECT c.layer,
               c.component,
               c.excess_kurtosis,
               {erf_select}
               a.positive_label,
               a.positive_confidence,
               a.negative_label,
               a.negative_confidence,
               a.positive_interpretation_types_json,
               a.negative_interpretation_types_json,
               a.summary,
               a.notes,
               a.include_as_case_study
        FROM components c
        LEFT JOIN annotations a
          ON a.model_name = c.model_name
         AND a.layer = c.layer
         AND a.component = c.component
        {erf_join}
        WHERE c.model_name = ?
          AND (? IS NULL OR c.layer = ?)
          AND (? IS NULL OR c.component = ?)
        """,
        (model_name, layer, layer, component, component),
    ).fetchall()
    query = str(search or "").strip().lower()
    out = []
    for row in rows:
        item = {
            "layer": str(row["layer"]),
            "component": int(row["component"]),
            "excess_kurtosis": float(row["excess_kurtosis"]) if row["excess_kurtosis"] is not None else None,
            "effective_context_mean": float(row["effective_context_mean"]) if row["effective_context_mean"] is not None else None,
            **_annotation_row(row),
        }
        text = " ".join(str(item.get(key, "")) for key in ("component", "positive_label", "negative_label", "summary", "notes")).lower()
        if query and query not in text:
            continue
        out.append(item)
    return sorted(out, key=lambda row: (layer_sort_key(row["layer"]), row["component"]))


def list_component_stats(conn: sqlite3.Connection, model_name: str) -> list[dict[str, Any]]:
    erf_join, erf_select = _erf_join_sql(conn)
    rows = conn.execute(
        f"""
        SELECT c.layer,
               c.component,
               c.excess_kurtosis,
               {erf_select}
               a.positive_label,
               a.positive_confidence,
               a.negative_label,
               a.negative_confidence,
               a.positive_interpretation_types_json,
               a.negative_interpretation_types_json,
               a.notes
        FROM components c
        LEFT JOIN annotations a
          ON a.model_name = c.model_name
         AND a.layer = c.layer
         AND a.component = c.component
        {erf_join}
        WHERE c.model_name = ?
        """,
        (model_name,),
    ).fetchall()
    return sorted(
        [
            {
                "layer": str(row["layer"]),
                "effective_context_mean": float(row["effective_context_mean"]) if row["effective_context_mean"] is not None else None,
                **_annotation_row(row),
            }
            for row in rows
        ],
        key=lambda row: (layer_sort_key(row["layer"]), row["component"]),
    )


def list_annotated_components(conn: sqlite3.Connection, model_name: str, layer: str) -> list[dict[str, Any]]:
    erf_join, erf_select = _erf_join_sql(conn)
    rows = conn.execute(
        f"""
        SELECT a.component,
               {erf_select}
               a.positive_label, a.positive_confidence,
               a.negative_label, a.negative_confidence,
               a.positive_interpretation_types_json,
               a.negative_interpretation_types_json,
               a.notes,
               c.excess_kurtosis
        FROM annotations a
        LEFT JOIN components c
          ON c.model_name = a.model_name
         AND c.layer = a.layer
         AND c.component = a.component
        {erf_join}
        WHERE a.model_name = ? AND a.layer = ?
          AND (
            (a.positive_label IS NOT NULL AND TRIM(a.positive_label) != '')
            OR (a.negative_label IS NOT NULL AND TRIM(a.negative_label) != '')
          )
        ORDER BY a.component
        """,
        (model_name, layer),
    ).fetchall()
    out = []
    for row in rows:
        item = _annotation_row(row)
        item["effective_context_mean"] = float(row["effective_context_mean"]) if row["effective_context_mean"] is not None else None
        out.append(item)
    return out



def list_component_metadata(conn: sqlite3.Connection, model_name: str, layer: str, components: list[int]) -> list[dict[str, Any]]:
    if not components:
        return []
    erf_join, erf_select = _erf_join_sql(conn)
    unique_components = sorted({int(component) for component in components})
    placeholders = ",".join("?" for _ in unique_components)
    rows = conn.execute(
        f"""
        SELECT c.component,
               c.excess_kurtosis,
               {erf_select}
               a.positive_label,
               a.positive_confidence,
               a.negative_label,
               a.negative_confidence,
               a.positive_interpretation_types_json,
               a.negative_interpretation_types_json,
               a.summary,
               a.notes,
               a.include_as_case_study
        FROM components c
        LEFT JOIN annotations a
          ON a.model_name = c.model_name
         AND a.layer = c.layer
         AND a.component = c.component
        {erf_join}
        WHERE c.model_name = ? AND c.layer = ? AND c.component IN ({placeholders})
        ORDER BY c.component
        """,
        (model_name, layer, *unique_components),
    ).fetchall()
    out = []
    for row in rows:
        item = _annotation_row(row)
        item["effective_context_mean"] = float(row["effective_context_mean"]) if row["effective_context_mean"] is not None else None
        out.append(item)
    return out


def list_component_neighbors(conn: sqlite3.Connection, model_name: str, layer: str, component: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "component_neighbors"):
        return []
    neighbor_columns = _table_columns(conn, "component_neighbors")
    neighbor_sign_select = "n.neighbor_sign" if "neighbor_sign" in neighbor_columns else "CASE WHEN n.signed_cosine >= 0 THEN 1 ELSE -1 END"
    erf_join, erf_select = _component_neighbor_erf_join_sql(conn)
    rows = conn.execute(
        f"""
        SELECT n.direction,
               n.neighbor_layer,
               n.neighbor_component AS component,
               n.abs_cosine,
               n.signed_cosine,
               {neighbor_sign_select} AS neighbor_sign,
               n.source_n_components,
               n.neighbor_n_components,
               {erf_select}
               a.positive_label,
               a.positive_confidence,
               a.negative_label,
               a.negative_confidence,
               a.positive_interpretation_types_json,
               a.negative_interpretation_types_json,
               a.summary,
               a.notes,
               a.include_as_case_study,
               c.excess_kurtosis
        FROM component_neighbors n
        LEFT JOIN annotations a
          ON a.model_name = n.model_name
         AND a.layer = n.neighbor_layer
         AND a.component = n.neighbor_component
        LEFT JOIN components c
          ON c.model_name = n.model_name
         AND c.layer = n.neighbor_layer
         AND c.component = n.neighbor_component
        {erf_join}
        WHERE n.model_name = ?
          AND n.layer = ?
          AND n.component = ?
        ORDER BY CASE n.direction WHEN 'prev' THEN 0 WHEN 'next' THEN 1 ELSE 2 END
        """,
        (model_name, layer, int(component)),
    ).fetchall()
    out = []
    for row in rows:
        item = _annotation_row(row)
        item.update(
            {
                "direction": str(row["direction"]),
                "neighbor_layer": str(row["neighbor_layer"]),
                "neighbor_component": int(row["component"]),
                "abs_cosine": float(row["abs_cosine"]),
                "signed_cosine": float(row["signed_cosine"]),
                "neighbor_sign": int(row["neighbor_sign"]),
                "source_n_components": int(row["source_n_components"]),
                "neighbor_n_components": int(row["neighbor_n_components"]),
                "effective_context_mean": float(row["effective_context_mean"]) if row["effective_context_mean"] is not None else None,
            }
        )
        out.append(item)
    return out


def list_chosen_random_components(conn: sqlite3.Connection, model_name: str | None = None, selection_name: str | None = None) -> list[dict[str, Any]]:
    if not _table_exists(conn, "chosen_random_components"):
        return []
    erf_join, erf_select = _erf_join_sql(conn)
    rows = conn.execute(
        f"""
        SELECT r.selection_name,
               r.model_name,
               r.selection_index,
               r.layer,
               r.component,
               r.component_id,
               r.source_json,
               r.seed,
               r.requested_n,
               r.inventory_size,
               r.selected_size,
               r.fit_converged,
               r.fit_iterations,
               r.fit_final_lim,
               r.fit_final_lim_p95,
               r.fit_seed,
               c.excess_kurtosis,
               {erf_select}
               a.positive_label,
               a.positive_confidence,
               a.negative_label,
               a.negative_confidence,
               a.positive_interpretation_types_json,
               a.negative_interpretation_types_json,
               a.summary,
               a.notes,
               a.include_as_case_study
        FROM chosen_random_components r
        LEFT JOIN components c
          ON c.model_name = r.model_name
         AND c.layer = r.layer
         AND c.component = r.component
        LEFT JOIN annotations a
          ON a.model_name = r.model_name
         AND a.layer = r.layer
         AND a.component = r.component
        {erf_join}
        WHERE (? IS NULL OR r.model_name = ?)
          AND (? IS NULL OR r.selection_name = ?)
        ORDER BY r.model_name, r.selection_name, r.selection_index
        """,
        (model_name, model_name, selection_name, selection_name),
    ).fetchall()
    out = []
    for row in rows:
        item = _annotation_row(row)
        item.update(
            {
                "selection_name": str(row["selection_name"]),
                "model_name": str(row["model_name"]),
                "selection_index": int(row["selection_index"]),
                "layer": str(row["layer"]),
                "component": int(row["component"]),
                "component_id": str(row["component_id"] or ""),
                "source_json": str(row["source_json"] or ""),
                "seed": int(row["seed"]) if row["seed"] is not None else None,
                "requested_n": int(row["requested_n"]) if row["requested_n"] is not None else None,
                "inventory_size": int(row["inventory_size"]) if row["inventory_size"] is not None else None,
                "selected_size": int(row["selected_size"]) if row["selected_size"] is not None else None,
                "fit_converged": bool(row["fit_converged"]) if row["fit_converged"] is not None else None,
                "fit_iterations": int(row["fit_iterations"]) if row["fit_iterations"] is not None else None,
                "fit_final_lim": float(row["fit_final_lim"]) if row["fit_final_lim"] is not None else None,
                "fit_final_lim_p95": float(row["fit_final_lim_p95"]) if row["fit_final_lim_p95"] is not None else None,
                "fit_seed": int(row["fit_seed"]) if row["fit_seed"] is not None else None,
                "effective_context_mean": float(row["effective_context_mean"]) if row["effective_context_mean"] is not None else None,
            }
        )
        out.append(item)
    return out


def list_component_examples(conn: sqlite3.Connection, model_name: str, layer: str | None = None, component: int | None = None, *, region: str | None = None, limit: int | None = None) -> dict[tuple[str, int], list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT layer,
               component,
               region,
               rank,
               row_index,
               doc_id,
               token_id,
               token,
               source_score,
               direction_cosine,
               position,
               context_to_target,
               context,
               context_score_max_abs
        FROM examples
        WHERE model_name = ?
          AND (? IS NULL OR layer = ?)
          AND (? IS NULL OR component = ?)
          AND (? IS NULL OR region = ?)
          AND (? IS NULL OR rank <= ?)
        ORDER BY layer, component, region, rank
        """,
        (model_name, layer, layer, component, component, region, region, limit, limit),
    ).fetchall()
    examples: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["layer"]), int(row["component"]))
        examples.setdefault(key, []).append(_example_row(row))
    return examples


def list_component_example_details(conn: sqlite3.Connection, model_name: str, *, layer: str, component: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id,
               region,
               rank,
               row_index,
               doc_id,
               token_id,
               token,
               source_score,
               direction_cosine,
               position,
               context_to_target,
               context,
               context_score_max_abs
        FROM examples
        WHERE model_name = ? AND layer = ? AND component = ?
        ORDER BY region, rank
        """,
        (model_name, layer, component),
    ).fetchall()
    examples = []
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        example = _example_row(row)
        example["context_token_scores"] = []
        examples.append(example)
        by_id[int(row["id"])] = example
    for ids in _chunks(list(by_id), 500):
        if not ids:
            continue
        placeholders = ",".join("?" for _ in ids)
        token_rows = conn.execute(
            f"""
            SELECT example_id, seq, token_position, token, source_score, direction_cosine, is_target
            FROM context_tokens
            WHERE example_id IN ({placeholders})
            ORDER BY example_id, seq
            """,
            tuple(ids),
        ).fetchall()
        for token_row in token_rows:
            example = by_id.get(int(token_row["example_id"]))
            if example is None:
                continue
            example["context_token_scores"].append(
                {
                    "position": int(token_row["token_position"]) if token_row["token_position"] is not None else None,
                    "token": str(token_row["token"] or ""),
                    "source_score": float(token_row["source_score"]) if token_row["source_score"] is not None else None,
                    "direction_cosine": float(token_row["direction_cosine"]) if token_row["direction_cosine"] is not None else None,
                    "is_target": bool(token_row["is_target"]),
                }
            )
    return examples


def get_component_row(conn: sqlite3.Connection, model_name: str, layer: str, component: int) -> dict[str, Any] | None:
    rows = list_components(conn, model_name, layer=layer, component=component)
    return rows[0] if rows else None


def get_annotation(conn: sqlite3.Connection, model_name: str, layer: str, component: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT * FROM annotations
        WHERE model_name = ? AND layer = ? AND component = ?
        """,
        (model_name, layer, component),
    ).fetchone()
    return dict(row) if row else None


def update_annotation(conn: sqlite3.Connection, *, model_name: str, layer: str, component: int, positive_label: str, positive_confidence: str, positive_interpretation_types: list[str], negative_label: str, negative_confidence: str, negative_interpretation_types: list[str], summary: str, notes: str, include_as_case_study: bool) -> None:
    conn.execute(
        """
        INSERT INTO annotations(
          model_name, layer, component,
          positive_label, positive_confidence, positive_interpretation_types_json,
          negative_label, negative_confidence, negative_interpretation_types_json,
          summary, notes, include_as_case_study, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(model_name, layer, component) DO UPDATE SET
          positive_label = excluded.positive_label,
          positive_confidence = excluded.positive_confidence,
          positive_interpretation_types_json = excluded.positive_interpretation_types_json,
          negative_label = excluded.negative_label,
          negative_confidence = excluded.negative_confidence,
          negative_interpretation_types_json = excluded.negative_interpretation_types_json,
          summary = excluded.summary,
          notes = excluded.notes,
          include_as_case_study = excluded.include_as_case_study,
          updated_at = datetime('now')
        """,
        (
            model_name,
            layer,
            int(component),
            positive_label,
            _normalize_confidence(positive_confidence),
            json.dumps([str(item) for item in positive_interpretation_types if str(item)]),
            negative_label,
            _normalize_confidence(negative_confidence),
            json.dumps([str(item) for item in negative_interpretation_types if str(item)]),
            summary,
            notes,
            1 if include_as_case_study else 0,
        ),
    )
    conn.commit()


def infer_default_annotation_sign(conn: sqlite3.Connection, model_name: str, layer: str, component: int) -> int:
    row = conn.execute(
        """
        SELECT source_score
        FROM examples
        WHERE model_name = ? AND layer = ? AND component = ?
          AND region IN ('top_abs', 'top_abs_sample_500', 'top_abs_sample_5000')
        ORDER BY ABS(source_score) DESC, rank ASC
        LIMIT 1
        """,
        (model_name, layer, component),
    ).fetchone()
    if row is None or row["source_score"] is None:
        return 1
    return 1 if float(row["source_score"]) >= 0 else -1


def get_examples_by_region(conn: sqlite3.Connection, model_name: str, layer: str, component: int) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    examples = list_component_example_details(conn, model_name, layer=layer, component=component)
    by_region: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_region.setdefault(str(example["region"] or "examples"), []).append(example)
    regions = sorted(by_region, key=_example_band_sort_key)
    return regions, by_region


def pick_default_region(regions: list[str], examples_by_region: dict[str, list[dict[str, Any]]]) -> str:
    for candidate in ("top_abs", "top_abs_sample_500", "top_abs_sample_5000"):
        if candidate in examples_by_region:
            return candidate
    return regions[0] if regions else ""


def search_components(conn: sqlite3.Connection, *, model_name: str | None, query: str, confidence: str, annotation_type: str, include_examples: bool, limit: int) -> list[dict[str, Any]]:
    models = [model_name] if model_name else list_models(conn)
    results = []
    query_l = query.strip().lower()
    for model in models:
        for item in list_components(conn, model):
            labels = [item.get("positive_label", ""), item.get("negative_label", ""), item.get("summary", ""), item.get("notes", "")]
            haystack = " ".join(str(x) for x in labels).lower()
            if query_l and query_l not in haystack:
                continue
            if confidence and confidence not in {item.get("positive_confidence"), item.get("negative_confidence")}:
                continue
            if annotation_type and annotation_type not in set(item.get("positive_types", []) + item.get("negative_types", [])):
                continue
            row = {"model_name": model, **item}
            if include_examples:
                ex = list_component_examples(conn, model, layer=item["layer"], component=item["component"], limit=3).get((item["layer"], item["component"]), [])
                row["examples"] = ex
            results.append(row)
            if len(results) >= limit:
                return results
    return results


def layer_sort_key(layer: str) -> tuple[int, int | str]:
    if layer == "embedding":
        return (0, 0)
    if layer.startswith("layer_"):
        return (1, int(layer.removeprefix("layer_")))
    return (2, layer)


def _erf_join_sql(conn: sqlite3.Connection) -> tuple[str, str]:
    table = _first_existing_table(conn, ["effective_receptive_fields", "effective_context_lengths"])
    if table is None:
        return "", "NULL AS effective_context_mean,"
    columns = _table_columns(conn, table)
    value_column = _first_existing_name(columns, ["mean_erf", "erf_mean", "mean_length", "effective_receptive_field_mean", "effective_context_mean"])
    if value_column is None:
        return "", "NULL AS effective_context_mean,"
    quoted_table = _quote_identifier(table)
    quoted_value = _quote_identifier(value_column)
    return (
        f"""
        LEFT JOIN {quoted_table} erf
          ON erf.model_name = c.model_name
         AND erf.layer = c.layer
         AND erf.component = c.component
        """,
        f"erf.{quoted_value} AS effective_context_mean,",
    )


def _component_neighbor_erf_join_sql(conn: sqlite3.Connection) -> tuple[str, str]:
    table = _first_existing_table(conn, ["effective_receptive_fields", "effective_context_lengths"])
    if table is None:
        return "", "NULL AS effective_context_mean,"
    columns = _table_columns(conn, table)
    value_column = _first_existing_name(columns, ["mean_erf", "erf_mean", "mean_length", "effective_receptive_field_mean", "effective_context_mean"])
    if value_column is None:
        return "", "NULL AS effective_context_mean,"
    quoted_table = _quote_identifier(table)
    quoted_value = _quote_identifier(value_column)
    return (
        f"""
        LEFT JOIN {quoted_table} erf
          ON erf.model_name = n.model_name
         AND erf.layer = n.neighbor_layer
         AND erf.component = n.neighbor_component
        """,
        f"erf.{quoted_value} AS effective_context_mean,",
    )


def _first_existing_table(conn: sqlite3.Connection, names: list[str]) -> str | None:
    for name in names:
        if _table_exists(conn, name):
            return name
    return None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()}


def _first_existing_name(names: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone() is not None


def _ensure_annotation_columns(conn: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(annotations)").fetchall()}
    required = {
        "label": "TEXT",
        "confidence": "TEXT",
        "positive_label": "TEXT",
        "positive_confidence": "TEXT",
        "negative_label": "TEXT",
        "negative_confidence": "TEXT",
        "positive_interpretation_types_json": "TEXT",
        "negative_interpretation_types_json": "TEXT",
        "interpretation_types_json": "TEXT",
        "summary": "TEXT",
        "notes": "TEXT",
        "include_as_case_study": "INTEGER NOT NULL DEFAULT 0",
        "updated_at": "TEXT",
    }
    for column, ddl in required.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE annotations ADD COLUMN {column} {ddl}")


def _annotation_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    keys = row.keys() if hasattr(row, "keys") else row
    def get(key: str, default: Any = None) -> Any:
        return row[key] if key in keys else default
    positive_label = str(get("positive_label") or "")
    negative_label = str(get("negative_label") or "")
    auto_annotated = "auto-annotation" in str(get("notes") or "").lower()
    return {
        "component": int(get("component")),
        "positive_label": positive_label,
        "positive_confidence": _normalize_confidence(get("positive_confidence")),
        "negative_label": negative_label,
        "negative_confidence": _normalize_confidence(get("negative_confidence")),
        "positive_types": _json_list(get("positive_interpretation_types_json")),
        "negative_types": _json_list(get("negative_interpretation_types_json")),
        "summary": str(get("summary") or ""),
        "notes": str(get("notes") or ""),
        "include_as_case_study": bool(get("include_as_case_study") or 0),
        "excess_kurtosis": float(get("excess_kurtosis")) if get("excess_kurtosis") is not None else None,
        "auto_annotated": auto_annotated,
    }


def _example_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "region": str(row["region"] or ""),
        "rank": int(row["rank"]),
        "row_index": int(row["row_index"]) if "row_index" in row.keys() and row["row_index"] is not None else None,
        "doc_id": int(row["doc_id"]) if "doc_id" in row.keys() and row["doc_id"] is not None else None,
        "token_id": int(row["token_id"]) if "token_id" in row.keys() and row["token_id"] is not None else None,
        "token": str(row["token"] or ""),
        "source_score": float(row["source_score"]) if row["source_score"] is not None else None,
        "direction_cosine": float(row["direction_cosine"]) if row["direction_cosine"] is not None else None,
        "position": int(row["position"]) if row["position"] is not None else None,
        "context_to_target": str(row["context_to_target"] or ""),
        "context": str(row["context"] or ""),
        "context_score_max_abs": float(row["context_score_max_abs"]) if row["context_score_max_abs"] is not None else None,
    }


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _normalize_confidence(value: Any) -> str:
    text = str(value or "unclear").strip().lower()
    return text if text in {"high", "medium", "low", "unclear"} else "unclear"


def _chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _example_band_sort_key(region: str) -> tuple[int, str]:
    name = str(region or "").lower().replace("-", "_")
    compact = name.replace("_", "")
    if name == "top_abs" or compact == "topabs":
        return (0, name)
    if "sample" in name:
        return (1, name)
    if "near_zero" in name or "nearzero" in compact:
        return (2, name)
    if "opposite" in name:
        return (3, name)
    return (4, name)
