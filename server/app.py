from __future__ import annotations

import argparse
import gc
import html
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import resolve_db_path, resolve_ica_dir, validate_ica_dir
from .config import Settings, load_settings
from .model_runtime import load_model_and_tokenizer, load_tokenizer
from .probe import _decode_byte_level_token, fastica_artifact_path, interpret_text_probe, list_ica_layer_keys
from .sae_probe import interpret_text_sae_probe, list_sae_layers, load_sae_config
from .store import (
    connect,
    get_annotation,
    get_component_row,
    get_examples_by_region,
    infer_default_annotation_sign,
    init_db,
    list_annotated_components,
    list_component_example_details,
    list_component_examples,
    list_component_metadata,
    list_component_neighbors,
    list_component_stats,
    list_components,
    list_chosen_random_components,
    list_layers,
    list_models,
    pick_default_region,
    search_components,
    update_annotation,
    validate_db,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
V6_ROOT = Path(__file__).resolve().parents[1]
GPT2_LAYER11_PATCH_ROOT = V6_ROOT / "patch_gpt2_layer11"
GPT2_LAYER11_PATCH_DB = GPT2_LAYER11_PATCH_ROOT / "data" / "server" / "db" / "ica_probe_gpt2_layer11_patch.sqlite"
GPT2_LAYER11_PATCH_ICA_ROOT = GPT2_LAYER11_PATCH_ROOT / "data" / "ica"
ANNOTATION_TYPES = ["Form", "Word", "Phrase", "Sentence", "Long-Range Context", "Global", "Position", "Sophisticated"]


class ProbeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    model_name: str = "gpt2"
    layer: str
    top_k: int = Field(5, ge=1, le=32)
    highlights: list[int] = Field(default_factory=list, max_length=256)
    max_length: int | None = Field(None, ge=16, le=8192)
    keep_models: bool = False


class SaeProbeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    model_name: str = "gpt2"
    layer: str
    top_k: int = Field(5, ge=1, le=128)
    max_length: int | None = Field(None, ge=16, le=8192)
    keep_models: bool = False


class AnnotationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    layer: str
    component: int
    positive_label: str = ""
    positive_confidence: str = "unclear"
    positive_interpretation_types: list[str] = Field(default_factory=list)
    negative_label: str = ""
    negative_confidence: str = "unclear"
    negative_interpretation_types: list[str] = Field(default_factory=list)
    summary: str = ""
    notes: str = ""
    include_as_case_study: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    db_path = resolve_db_path(settings)
    ica_dirs = {}
    for model_name, model_settings in settings.models.items():
        try:
            ica_dirs[model_name] = resolve_ica_dir(settings, model_name=model_name, ica_dir=model_settings.ica_dir)
        except FileNotFoundError:
            continue
    if not ica_dirs:
        raise FileNotFoundError("No ICA artifact directories are available for the configured models.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = connect(db_path)
        init_db(conn)
        validate_db(conn, None)
        app.state.conn = conn
        app.state.settings = settings
        app.state.ica_dirs = ica_dirs
        app.state.ica_dir = ica_dirs.get(settings.model_name, next(iter(ica_dirs.values())))
        app.state.use_gpt2_layer11_patch = bool(getattr(settings, "use_gpt2_layer11_patch", False)) or _db_marks_gpt2_layer11_raw_hook(conn)
        app.state.db_lock = threading.RLock()
        app.state.runtime_lock = threading.Lock()
        app.state.runtimes = {}
        app.state.runtime = None
        app.state.runtime_key = None
        app.state.full_context_cache = {}
        try:
            yield
        finally:
            app.state.runtimes.clear()
            app.state.runtime = None
            app.state.runtime_key = None
            conn.close()
            _collect_memory()

    app = FastAPI(title="ICA Lens Explorer", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta")
    def meta(model: str | None = None) -> dict[str, Any]:
        model_name = model or app.state.settings.model_name
        return _model_meta(app, model_name)

    @app.post("/api/probe")
    def probe(body: ProbeRequest) -> dict[str, Any]:
        model_settings = _model_settings(app, body.model_name)
        ica_dir = app.state.ica_dirs.get(model_settings.model_name)
        if ica_dir is None:
            raise HTTPException(status_code=404, detail=f"No ICA artifacts for model {model_settings.model_name!r}")
        artifact = fastica_artifact_path(ica_dir, body.layer)
        if not artifact.is_file():
            raise HTTPException(status_code=404, detail=f"No ICA artifact for {model_settings.model_name!r} layer {body.layer!r}")
        try:
            with app.state.runtime_lock:
                runtimes = app.state.runtimes
                if not body.keep_models:
                    for loaded_name in list(runtimes):
                        if loaded_name != model_settings.model_name:
                            del runtimes[loaded_name]
                    _collect_memory()
                runtime = runtimes.get(model_settings.model_name)
                if runtime is None:
                    runtime = load_model_and_tokenizer(
                        model_settings.model_id,
                        device=app.state.settings.device,
                        dtype=model_settings.dtype,
                    )
                    runtimes[model_settings.model_name] = runtime
                app.state.runtime = runtime
                app.state.runtime_key = model_settings.model_name
                model, tokenizer = runtime
                result = interpret_text_probe(
                    model=model,
                    tokenizer=tokenizer,
                    text=body.text,
                    layer=body.layer,
                    ica_artifact_path=artifact,
                    top_k=body.top_k,
                    highlight_components=body.highlights,
                    max_length=body.max_length or model_settings.context_length,
                    raw_gpt2_block_index=_raw_gpt2_block_index_for_probe(app, model_settings.model_name, body.layer),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        components = sorted({int(pair["component"]) for token in result.get("tokens", []) for pair in token.get("top", [])})
        with app.state.db_lock:
            annotated = list_component_metadata(app.state.conn, model_settings.model_name, body.layer, components)
        return {**result, "annotated_components": annotated}

    @app.get("/api/sae-meta")
    def sae_meta(model: str | None = None) -> dict[str, Any]:
        model_name = model or app.state.settings.model_name
        model_settings = _model_settings(app, model_name)
        try:
            sae_config = load_sae_config(model_settings.model_name)
            layers = list_sae_layers(model_settings.model_name)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "model_name": model_settings.model_name,
            "display_name": model_settings.display_name,
            "model_id": model_settings.model_id,
            "context_length": model_settings.context_length,
            "layers": layers,
            "device": app.state.settings.device,
            "dtype": model_settings.dtype,
            "sae": {
                "repo_id": sae_config.get("repo_id"),
                "width": sae_config.get("width"),
                "top_k": sae_config.get("top_k"),
                "activation": sae_config.get("activation"),
            },
            "model_loaded": model_settings.model_name in getattr(app.state, "runtimes", {}),
        }

    @app.post("/api/sae-probe")
    def sae_probe(body: SaeProbeRequest) -> dict[str, Any]:
        model_settings = _model_settings(app, body.model_name)
        try:
            if body.layer not in list_sae_layers(model_settings.model_name):
                raise ValueError(f"No SAE configured for {model_settings.model_name!r} layer {body.layer!r}")
            with app.state.runtime_lock:
                runtimes = app.state.runtimes
                if not body.keep_models:
                    for loaded_name in list(runtimes):
                        if loaded_name != model_settings.model_name:
                            del runtimes[loaded_name]
                    _collect_memory()
                runtime = runtimes.get(model_settings.model_name)
                if runtime is None:
                    runtime = load_model_and_tokenizer(
                        model_settings.model_id,
                        device=app.state.settings.device,
                        dtype=model_settings.dtype,
                    )
                    runtimes[model_settings.model_name] = runtime
                app.state.runtime = runtime
                app.state.runtime_key = model_settings.model_name
                model, tokenizer = runtime
                return interpret_text_sae_probe(
                    model=model,
                    tokenizer=tokenizer,
                    text=body.text,
                    model_name=model_settings.model_name,
                    layer=body.layer,
                    top_k=body.top_k,
                    max_length=body.max_length or model_settings.context_length,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/component-stats")
    def component_stats(model: str | None = None) -> dict[str, Any]:
        model_name = model or app.state.settings.model_name
        with app.state.db_lock:
            rows = list_component_stats(app.state.conn, model_name)
        layers: list[dict[str, Any]] = []
        for row in rows:
            if not layers or layers[-1]["layer"] != row["layer"]:
                layers.append({"layer": row["layer"], "components": []})
            layers[-1]["components"].append({key: value for key, value in row.items() if key != "layer"})
        return {"model_name": model_name, "layers": layers}

    @app.get("/api/component-examples")
    def component_examples(layer: str, component: int, model: str | None = None) -> dict[str, Any]:
        model_name = model or app.state.settings.model_name
        with app.state.db_lock:
            rows = list_components(app.state.conn, model_name, layer=layer, component=component)
            if not rows:
                raise HTTPException(status_code=404, detail="Unknown layer/component")
            examples = list_component_example_details(app.state.conn, model_name, layer=layer, component=component)
        tokenizer = None
        model_settings = app.state.settings.models.get(model_name)
        if model_settings is not None:
            tokenizer = load_tokenizer(model_settings.model_id)
        for example in examples:
            example["token"] = _visible_token_text(example.get("token"), token_id=example.get("token_id"), tokenizer=tokenizer)
        bands: dict[str, list[dict[str, Any]]] = {}
        for example in examples:
            bands.setdefault(str(example["region"] or "examples"), []).append(example)
        return {
            "model_name": model_name,
            "layer": layer,
            "component": component,
            "excess_kurtosis": rows[0]["excess_kurtosis"],
            "bands": [{"region": region, "examples": items} for region, items in sorted(bands.items(), key=lambda item: _example_band_sort_key(item[0]))],
        }

    @app.get("/api/component-token-stats")
    def component_token_stats(layer: str, component: int, model: str | None = None) -> dict[str, Any]:
        model_name = model or app.state.settings.model_name
        with app.state.db_lock:
            rows = list_components(app.state.conn, model_name, layer=layer, component=component)
            if not rows:
                raise HTTPException(status_code=404, detail="Unknown layer/component")
            examples = list_component_examples(app.state.conn, model_name, layer=layer, component=component).get((layer, component), [])
        tokenizer = None
        model_settings = app.state.settings.models.get(model_name)
        if model_settings is not None:
            tokenizer = load_tokenizer(model_settings.model_id)
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        total_count = 0
        for example in examples:
            if example["region"] not in {"top_abs", "top_abs_sample_500"}:
                continue
            token = _visible_token_text(example["token"], token_id=example.get("token_id"), tokenizer=tokenizer)
            if not token:
                continue
            total_count += 1
            first_seen.setdefault(token, total_count)
            counts[token] = counts.get(token, 0) + 1
        ordered_tokens = sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
        return {"model_name": model_name, "layer": layer, "component": component, "total_count": total_count, "tokens": [{"token": token, "count": count} for token, count in ordered_tokens]}

    @app.get("/api/component-neighbors")
    def component_neighbors(layer: str, component: int, model: str | None = None) -> dict[str, Any]:
        model_name = model or app.state.settings.model_name
        with app.state.db_lock:
            rows = list_components(app.state.conn, model_name, layer=layer, component=component)
            if not rows:
                raise HTTPException(status_code=404, detail="Unknown layer/component")
            neighbors = list_component_neighbors(app.state.conn, model_name, layer=layer, component=component)
        return {"model_name": model_name, "layer": layer, "component": component, "neighbors": neighbors}

    @app.get("/api/models")
    def api_models() -> dict[str, Any]:
        models = []
        with app.state.db_lock:
            model_names = list_models(app.state.conn)
        for model_name in model_names:
            model_settings = app.state.settings.models.get(model_name)
            ica_dir = app.state.ica_dirs.get(model_name)
            models.append(
                {
                    "model_name": model_name,
                    "display_name": model_settings.display_name if model_settings else model_name,
                    "model_id": model_settings.model_id if model_settings else "",
                    "context_length": model_settings.context_length if model_settings else None,
                    "has_examples": True,
                    "probe_supported": model_settings is not None and ica_dir is not None,
                    "ica_layers": list_ica_layer_keys(ica_dir) if ica_dir else [],
                }
            )
        return {"models": models}

    @app.get("/api/layers")
    def api_layers(model: str) -> dict[str, Any]:
        with app.state.db_lock:
            layers = list_layers(app.state.conn, model)
        return {"layers": layers}

    @app.get("/api/components")
    def api_components(model: str, layer: str | None = None, search: str | None = None) -> dict[str, Any]:
        with app.state.db_lock:
            components = list_components(app.state.conn, model_name=model, layer=layer, search=search)
        return {"components": components}

    @app.get("/api/search/components")
    def api_search_components(model: str | None = None, q: str | None = None, confidence: str | None = None, type: str | None = None, include_examples: bool = False, limit: int = 200) -> dict[str, Any]:
        with app.state.db_lock:
            results = search_components(app.state.conn, model_name=model, query=q or "", confidence=confidence or "", annotation_type=type or "", include_examples=include_examples, limit=limit)
        return {"results": results}

    @app.get("/api/random-components")
    def api_random_components(model: str | None = None, selection: str | None = None) -> dict[str, Any]:
        with app.state.db_lock:
            rows = list_chosen_random_components(app.state.conn, model_name=model, selection_name=selection)
        runs_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            model_name = str(row["model_name"])
            selection_name = str(row["selection_name"])
            key = (model_name, selection_name)
            run = runs_by_key.setdefault(
                key,
                {
                    "model": model_name,
                    "selection_name": selection_name,
                    "source_json": row.get("source_json"),
                    "settings": {
                        "n": row.get("requested_n"),
                        "seed": row.get("seed"),
                    },
                    "inventory_size": row.get("inventory_size"),
                    "selected_size": row.get("selected_size"),
                    "selected_components": [],
                },
            )
            component = int(row["component"])
            layer = str(row["layer"])
            default_sign = infer_default_annotation_sign(app.state.conn, model_name, layer, component)
            run["selected_components"].append(
                {
                    **row,
                    "component_index": component,
                    "top_abs_sign": default_sign,
                    "annotation": _annotation_response(row),
                    "annotate_url": f"/annotate?model={model_name}&layer={layer}&component={component}",
                }
            )
        return {"runs": list(runs_by_key.values())}

    @app.get("/api/annotations/component")
    def api_get_annotation(model: str, layer: str, component: int) -> dict[str, Any]:
        with app.state.db_lock:
            row = get_annotation(app.state.conn, model, layer, component)
            if not row:
                sign = infer_default_annotation_sign(app.state.conn, model, layer, component)
                return _blank_annotation(model, layer, component, default_sign=sign)
        return _annotation_response(row)

    @app.post("/api/annotations/component")
    def api_post_annotation(body: AnnotationUpdate) -> dict[str, str]:
        with app.state.db_lock:
            if get_component_row(app.state.conn, body.model_name, body.layer, body.component) is None:
                raise HTTPException(status_code=404, detail="Unknown model/layer/component")
            update_annotation(
                app.state.conn,
                model_name=body.model_name,
                layer=body.layer,
                component=body.component,
                positive_label=body.positive_label,
                positive_confidence=body.positive_confidence,
                positive_interpretation_types=body.positive_interpretation_types,
                negative_label=body.negative_label,
                negative_confidence=body.negative_confidence,
                negative_interpretation_types=body.negative_interpretation_types,
                summary=body.summary,
                notes=body.notes,
                include_as_case_study=body.include_as_case_study,
            )
        return {"status": "ok"}

    @app.get("/api/examples/component")
    def api_examples_component(model: str, layer: str, component: int) -> dict[str, Any]:
        with app.state.db_lock:
            if get_component_row(app.state.conn, model, layer, component) is None:
                raise HTTPException(status_code=404, detail="Unknown model/layer/component")
            regions, examples_by_region = get_examples_by_region(app.state.conn, model, layer, component)
        tokenizer = None
        model_settings = app.state.settings.models.get(model)
        if model_settings is not None:
            tokenizer = load_tokenizer(model_settings.model_id)
        for examples in examples_by_region.values():
            for example in examples:
                example["token"] = _visible_token_text(example.get("token"), token_id=example.get("token_id"), tokenizer=tokenizer)
        return {"model_name": model, "layer": layer, "component": component, "regions": regions, "default_region": pick_default_region(regions, examples_by_region), "examples_by_region": examples_by_region}

    @app.get("/api/annotation-types")
    def api_annotation_types() -> dict[str, Any]:
        return {"types": ANNOTATION_TYPES}

    @app.get("/context")
    def full_context_page(model: str, doc_id: int, position: int | None = None, target: str | None = None) -> HTMLResponse:
        model_settings = _model_settings(app, model)
        text = _load_dataset_doc_text(app, model_settings, doc_id)
        target_span = _target_token_char_span(model_settings, text, position)
        if target_span is None:
            target_span = _fallback_target_char_span(text, target)
        return HTMLResponse(
            _render_full_context_page(
                model_name=model_settings.model_name,
                doc_id=doc_id,
                position=position,
                text=text,
                target_span=target_span,
            )
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/stats")
    def stats() -> FileResponse:
        return FileResponse(STATIC_DIR / "stats.html")

    @app.get("/component")
    def component_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "component.html")

    @app.get("/annotate")
    def annotate_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "annotate.html")

    @app.get("/random-components")
    def random_components_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "random_components.html")

    @app.get("/sae-explorer")
    def sae_explorer_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "sae_explorer.html")

    return app


def _model_settings(app: FastAPI, model_name: str):
    model_settings = app.state.settings.models.get(model_name)
    if model_settings is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name!r} is not configured for text probing.")
    return model_settings


def _raw_gpt2_block_index_for_probe(app: FastAPI, model_name: str, layer: str) -> int | None:
    if not getattr(app.state, "use_gpt2_layer11_patch", False):
        return None
    if model_name == "gpt2" and layer == "layer_11":
        return 11
    return None


def _db_marks_gpt2_layer11_raw_hook(conn) -> bool:
    try:
        row = conn.execute(
            """
            SELECT value
            FROM import_meta
            WHERE model_name = ?
              AND key = ?
            """,
            ("gpt2", "gpt2_layer11_probe_site"),
        ).fetchone()
    except Exception:
        return False
    if row is None:
        return False
    value = str(row[0]).strip().lower()
    return value in {"raw_block_11_resid_post", "patched_raw_block_11_resid_post"}


def _settings_with_gpt2_layer11_patch(settings: Settings) -> Settings:
    gpt2_settings = settings.models.get("gpt2")
    if gpt2_settings is None:
        raise RuntimeError("GPT-2 settings are required for --use-gpt2-layer11-patch.")
    if not GPT2_LAYER11_PATCH_DB.is_file():
        raise FileNotFoundError(f"Patch SQLite database does not exist: {GPT2_LAYER11_PATCH_DB}")
    if not (GPT2_LAYER11_PATCH_ICA_ROOT / "gpt2" / "layer_11_fastica.pt").is_file():
        raise FileNotFoundError(f"Patch ICA artifact does not exist: {GPT2_LAYER11_PATCH_ICA_ROOT / 'gpt2' / 'layer_11_fastica.pt'}")
    models = dict(settings.models)
    models["gpt2"] = replace(gpt2_settings, ica_dir=GPT2_LAYER11_PATCH_ICA_ROOT / "gpt2")
    patched = replace(
        settings,
        db_path=GPT2_LAYER11_PATCH_DB,
        ica_root=GPT2_LAYER11_PATCH_ICA_ROOT,
        ica_dir=GPT2_LAYER11_PATCH_ICA_ROOT / "gpt2",
        model_name="gpt2",
        download_missing=False,
        models=models,
        use_gpt2_layer11_patch=True,
    )
    return patched


def _model_meta(app: FastAPI, model_name: str) -> dict[str, Any]:
    model_settings = _model_settings(app, model_name)
    ica_dir = app.state.ica_dirs.get(model_settings.model_name)
    layers = list_ica_layer_keys(ica_dir) if ica_dir else []
    with app.state.db_lock:
        db_layers = set(list_layers(app.state.conn, model_settings.model_name))
    return {
        "model_name": model_settings.model_name,
        "display_name": model_settings.display_name,
        "model_id": model_settings.model_id,
        "context_length": model_settings.context_length,
        "layers": [layer for layer in layers if layer in db_layers],
        "device": app.state.settings.device,
        "dtype": model_settings.dtype,
        "model_loaded": model_settings.model_name in getattr(app.state, "runtimes", {}),
    }


def _blank_annotation(model: str, layer: str, component: int, *, default_sign: int) -> dict[str, Any]:
    positive_label = "" if default_sign >= 0 else "?"
    negative_label = "?" if default_sign >= 0 else ""
    return {
        "model_name": model,
        "layer": layer,
        "component": component,
        "positive_label": positive_label,
        "positive_confidence": "unclear",
        "positive_interpretation_types": [],
        "negative_label": negative_label,
        "negative_confidence": "unclear",
        "negative_interpretation_types": [],
        "summary": "",
        "notes": "",
        "include_as_case_study": False,
        "updated_at": None,
    }


def _annotation_response(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": row["model_name"],
        "layer": row["layer"],
        "component": int(row["component"]),
        "positive_label": row.get("positive_label") or "",
        "positive_confidence": row.get("positive_confidence") or "unclear",
        "positive_interpretation_types": _json_list(row.get("positive_interpretation_types_json")),
        "negative_label": row.get("negative_label") or "",
        "negative_confidence": row.get("negative_confidence") or "unclear",
        "negative_interpretation_types": _json_list(row.get("negative_interpretation_types_json")),
        "summary": row.get("summary") or "",
        "notes": row.get("notes") or "",
        "include_as_case_study": bool(row.get("include_as_case_study")),
        "updated_at": row.get("updated_at"),
    }


def _json_list(value: Any) -> list[str]:
    import json
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _load_dataset_doc_text(app: FastAPI, model_settings: Any, doc_id: int) -> str:
    if doc_id < 0:
        raise HTTPException(status_code=400, detail="doc_id must be non-negative")
    key = (model_settings.model_name, doc_id)
    cached = app.state.full_context_cache.get(key)
    if cached is not None:
        return cached
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="datasets is required to open full document context") from exc
    dataset_kwargs: dict[str, Any] = {
        "path": model_settings.dataset_path,
        "split": model_settings.dataset_split,
        "streaming": model_settings.dataset_streaming,
    }
    if model_settings.dataset_name:
        dataset_kwargs["name"] = model_settings.dataset_name
    try:
        dataset = load_dataset(**dataset_kwargs)
        if model_settings.dataset_streaming:
            for idx, row in enumerate(dataset):
                if idx == doc_id:
                    text = str(row[model_settings.dataset_text_column])
                    app.state.full_context_cache[key] = text
                    return text
                if idx > doc_id:
                    break
        else:
            row = dataset[doc_id]
            text = str(row[model_settings.dataset_text_column])
            app.state.full_context_cache[key] = text
            return text
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in {model_settings.dataset_path}/{model_settings.dataset_split}") from exc
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Dataset row does not contain text column {model_settings.dataset_text_column!r}") from exc
    raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in {model_settings.dataset_path}/{model_settings.dataset_split}")


def _target_token_char_span(model_settings: Any, text: str, position: int | None) -> tuple[int, int] | None:
    if position is None or position < 0:
        return None
    tokenizer = load_tokenizer(model_settings.model_id)
    try:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=model_settings.context_length,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    offsets = encoded.get("offset_mapping")
    if offsets is None or position >= len(offsets):
        return None
    start, stop = offsets[position]
    start = int(start)
    stop = int(stop)
    if stop <= start:
        return None
    return start, stop


def _fallback_target_char_span(text: str, target: str | None) -> tuple[int, int] | None:
    if not target:
        return None
    candidates = [target.replace("\ufffd", "")]
    stripped = candidates[0].strip()
    if stripped and stripped != candidates[0]:
        candidates.append(stripped)
    for candidate in candidates:
        if not candidate:
            continue
        start = text.find(candidate)
        if start >= 0:
            return start, start + len(candidate)
    return None


def _highlighted_text_html(text: str, target_span: tuple[int, int] | None) -> str:
    if target_span is None:
        return html.escape(text)
    start, stop = target_span
    start = max(0, min(start, len(text)))
    stop = max(start, min(stop, len(text)))
    return f'{html.escape(text[:start])}<mark id="target-token">{html.escape(text[start:stop])}</mark>{html.escape(text[stop:])}'


def _render_full_context_page(*, model_name: str, doc_id: int, position: int | None, text: str, target_span: tuple[int, int] | None) -> str:
    pos = "" if position is None else f" - pos={html.escape(str(position))}"
    title = f"{model_name} - doc={doc_id}{pos}"
    content = _highlighted_text_html(text, target_span)
    target_status = "" if target_span is not None else '<span class="target-status">target token not located</span>'
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin:0; background:#f8fafc; color:#0f172a; font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ position:sticky; top:0; background:#fff; border-bottom:1px solid #cbd5e1; padding:14px 18px; color:#475569; font-size:13px; box-shadow:0 1px 2px rgb(15 23 42 / .08); }}
    pre {{ max-width:980px; margin:24px auto; padding:0 20px 48px; white-space:pre-wrap; overflow-wrap:anywhere; font:inherit; }}
    mark {{ background:#bfdbfe; color:#172554; border-radius:4px; padding:1px 3px; box-shadow:0 0 0 1px #93c5fd inset; }}
    .target-status {{ margin-left:10px; color:#b45309; }}
  </style>
</head>
<body>
  <header>{html.escape(title)}{target_status}</header>
  <pre>{content}</pre>
  <script>const target=document.getElementById("target-token"); if(target) target.scrollIntoView({{block:"center"}});</script>
</body>
</html>'''


def _collect_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def _visible_token_text(token: Any, *, token_id: Any = None, tokenizer: Any = None) -> str:
    raw = str(token or "")
    text = raw.strip()
    if (not text or "\ufffd" in text) and tokenizer is not None and token_id is not None:
        try:
            token_id_int = int(token_id)
            decoded = str(tokenizer.decode([token_id_int], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        except Exception:
            decoded = ""
        if decoded and "\ufffd" not in decoded:
            raw = decoded
            text = raw.strip()
        else:
            try:
                raw_token = str(tokenizer.convert_ids_to_tokens(token_id_int))
            except Exception:
                raw_token = ""
            byte_decoded = _decode_byte_level_token(raw_token) if raw_token else None
            if byte_decoded:
                raw = byte_decoded
                text = raw.strip()
            elif raw_token and "\ufffd" not in raw_token:
                raw = raw_token
                text = raw.strip()
    if text:
        return text
    if raw == " ":
        return "[space]"
    if raw == "\n":
        return "[newline]"
    if raw == "\t":
        return "[tab]"
    return raw.replace("\ufffd", "[invalid]")


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description="Run the ICA Lens explorer/annotator server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8764)
    parser.add_argument(
        "--use-gpt2-layer11-patch",
        action="store_true",
        help="Use the isolated corrected GPT-2 layer-11 patch DB/artifacts and raw block hook for live layer-11 probes.",
    )
    args = parser.parse_args()
    settings = load_settings()
    if args.use_gpt2_layer11_patch:
        settings = _settings_with_gpt2_layer11_patch(settings)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
