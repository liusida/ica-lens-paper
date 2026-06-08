#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "ica_lens").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Could not find v6 project root.")


V6_ROOT = _project_root()
V6_SRC = V6_ROOT / "src"
if str(V6_SRC) not in sys.path:
    sys.path.insert(0, str(V6_SRC))

from ica_lens.config import load_toml  # noqa: E402
from ica_lens.paths import CONFIGS_DIR, RESULTS_DIR  # noqa: E402
from ica_lens.saebench_adapters import _decoder_weight, _load_sae_weights, _resolve_sae_weights_path, canonical_ica_preprocess  # noqa: E402


DEFAULT_MODELS = ("gpt2", "gemma2_2b", "qwen3_5_2b_base")
DEFAULT_ICA_ROOT = RESULTS_DIR / "reproduced" / "ica"
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "reproduced" / "ica_sae_overlap"
NORM_EPS = 1e-12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ICA component directions to nearest public SAE decoder directions.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=DEFAULT_MODELS)
    parser.add_argument("--layers", nargs="*", default=None, help="Layer names like layer_06, or integer ids.")
    parser.add_argument("--token-budget", type=int, default=None, help="Use reproduced ICA artifacts named <model>_tok<N>.")
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sae-chunk-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-unconverged", action="store_true")
    parser.add_argument("--accept-p95-threshold", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    started_at = time.time()
    results = {}
    for model_name in args.models:
        results[model_name] = _process_model(model_name=str(model_name), args=args)

    manifest_path = args.output_root.resolve() / "manifest.json"
    _write_json(
        manifest_path,
        {
            "analysis": "ica_sae_direction_overlap",
            "models": list(args.models),
            "settings": {
                "ica_root": str(args.ica_root.resolve()),
                "token_budget": args.token_budget,
                "output_root": str(args.output_root.resolve()),
                "layers": args.layers,
                "device": str(_resolve_device(str(args.device))),
                "include_unconverged": bool(args.include_unconverged),
                "accept_p95_threshold": args.accept_p95_threshold,
            },
            "results": results,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
    )
    print(f"wrote {manifest_path}")
    return 0


def _process_model(*, model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = CONFIGS_DIR / "comparisons" / f"{model_name}.toml"
    cfg = load_toml(cfg_path)
    overlap_cfg = dict(cfg.get("direction_overlap", {}))
    top_k = int(args.top_k or overlap_cfg.get("top_k", 1))
    sae_chunk_size = int(args.sae_chunk_size or overlap_cfg.get("sae_chunk_size", 4096))
    accept_p95_threshold = float(
        args.accept_p95_threshold if args.accept_p95_threshold is not None else overlap_cfg.get("accept_p95_threshold", 1e-4)
    )
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    ica_dir = _resolve_ica_dir(args.ica_root.resolve(), model_name=model_name, token_budget=args.token_budget)
    if not ica_dir.is_dir():
        raise FileNotFoundError(f"Missing ICA directory: {ica_dir}")
    layers = _selected_layers(args.layers, ica_dir)
    device = _resolve_device(str(args.device))

    output_dir = args.output_root.resolve() / model_name
    metrics_path = output_dir / f"{model_name}.json"
    summary_table_path = output_dir / f"{model_name}_summary.csv"
    component_table_path = output_dir / f"{model_name}_components.csv"
    if metrics_path.exists() and not args.force:
        print(f"reusing {metrics_path}")
        return {"files": {"metrics": str(metrics_path), "summary": str(summary_table_path), "components": str(component_table_path)}}

    results: dict[str, Any] = {
        "analysis": "ica_sae_direction_overlap",
        "model_name": model_name,
        "config": str(cfg_path.resolve()),
        "ica_dir": str(ica_dir),
        "settings": {
            "nearest_metric": "maximum_absolute_cosine",
            "top_k": top_k,
            "sae_chunk_size": sae_chunk_size,
            "device": str(device),
            "include_unconverged": bool(args.include_unconverged),
            "accept_p95_threshold": accept_p95_threshold,
        },
        "sae": dict(cfg["sae"]),
        "layers": {},
    }
    layer_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    all_nearest_abs: list[torch.Tensor] = []

    for layer in layers:
        metadata_path = ica_dir / f"{layer}_fastica.json"
        artifact_path = ica_dir / f"{layer}_fastica.pt"
        metadata = _load_json(metadata_path) if metadata_path.exists() else {}
        acceptance = _ica_fit_acceptance(metadata, p95_threshold=accept_p95_threshold)
        if not args.include_unconverged and not bool(acceptance["accepted"]):
            print(f"skipping {model_name} {layer}: ICA fit not accepted ({acceptance['reason']})")
            continue

        layer_started = time.time()
        component_dirs = _load_ica_component_directions(artifact_path=artifact_path, device=device)
        hidden_size = int(component_dirs.shape[1])
        sae_decoder = _load_sae_decoder(sae_config=dict(cfg["sae"]), layer=_layer_index(layer), hidden_size=hidden_size)
        matches = _nearest_sae_features(
            component_dirs=component_dirs,
            decoder=sae_decoder,
            top_k=top_k,
            chunk_size=sae_chunk_size,
            device=device,
            desc=f"{model_name} {layer} ICA->SAE",
        )
        nearest_abs = matches["abs_cosine"][:, 0].detach().cpu()
        all_nearest_abs.append(nearest_abs)
        components = _component_results(matches, top_k=top_k)
        component_rows.extend(_component_csv_rows(model_name, layer, components))
        layer_result = {
            "elapsed_seconds": round(time.time() - layer_started, 3),
            "artifact": str(artifact_path),
            "converged": bool(metadata.get("converged", False)),
            "accepted": bool(acceptance["accepted"]),
            "acceptance_reason": str(acceptance["reason"]),
            "final_lim_p95": acceptance["final_p95"],
            "n_components": int(component_dirs.shape[0]),
            "hidden_size": hidden_size,
            "sae_layer": _layer_index(layer),
            "sae_width": int(sae_decoder.shape[0]),
            "nearest_abs_cosine_summary": _tensor_summary(nearest_abs),
            "components": components,
        }
        results["layers"][layer] = layer_result
        layer_rows.append(_layer_csv_row(model_name, layer, layer_result))
        del component_dirs, sae_decoder, matches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_nearest_abs:
        results["nearest_abs_cosine_summary"] = _tensor_summary(torch.cat(all_nearest_abs))
    _write_json(metrics_path, results)
    _write_csv(summary_table_path, layer_rows)
    _write_csv(component_table_path, component_rows)
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_table_path}")
    print(f"wrote {component_table_path}")
    return {"files": {"metrics": str(metrics_path), "summary": str(summary_table_path), "components": str(component_table_path)}}


def _resolve_ica_dir(root: Path, *, model_name: str, token_budget: int | None) -> Path:
    candidates = []
    if token_budget is not None:
        candidates.append(root / f"{model_name}_tok{token_budget}")
    candidates.append(root / model_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _selected_layers(values: list[str] | None, ica_dir: Path) -> list[str]:
    if values:
        return [_layer_name(value) for value in values]
    return sorted((path.name.removesuffix("_fastica.pt") for path in ica_dir.glob("layer_*_fastica.pt")), key=_layer_index)


def _load_ica_component_directions(*, artifact_path: Path, device: torch.device) -> torch.Tensor:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing ICA artifact: {artifact_path}")
    blob = torch.load(artifact_path, map_location="cpu")
    metadata = blob.get("metadata", {})
    preprocess = canonical_ica_preprocess(metadata.get("preprocess"))
    if preprocess != "with_normalization":
        raise ValueError(f"Expected with_normalization ICA artifact, got {preprocess}: {artifact_path}")
    components = blob["tensors"]["components"].to(dtype=torch.float32)
    components = _normalize_rows(components)
    return components.to(device=device).contiguous()


def _load_sae_decoder(*, sae_config: dict[str, Any], layer: int, hidden_size: int) -> torch.Tensor:
    weights_path = _resolve_sae_weights_path(sae_config=sae_config, layer=layer)
    weights = _load_sae_weights(weights_path, sae_config=sae_config)
    decoder = _decoder_weight(weights, hidden_size=hidden_size, preferred_key=sae_config.get("decoder_key"))
    return decoder.to(dtype=torch.float32).contiguous()


def _nearest_sae_features(
    *,
    component_dirs: torch.Tensor,
    decoder: torch.Tensor,
    top_k: int,
    chunk_size: int,
    device: torch.device,
    desc: str,
) -> dict[str, torch.Tensor]:
    n_components = int(component_dirs.shape[0])
    width = int(decoder.shape[0])
    best_abs = torch.empty((n_components, 0), device=device, dtype=torch.float32)
    best_signed = torch.empty((n_components, 0), device=device, dtype=torch.float32)
    best_feature = torch.empty((n_components, 0), device=device, dtype=torch.long)
    component_dirs = component_dirs.to(device=device, dtype=torch.float32)
    for start in tqdm(range(0, width, chunk_size), unit="chunk", dynamic_ncols=True, desc=desc):
        end = min(start + chunk_size, width)
        decoder_chunk = _normalize_rows(decoder[start:end].to(device=device, dtype=torch.float32, non_blocking=True))
        cosine = component_dirs @ decoder_chunk.T
        local_k = min(top_k, int(cosine.shape[1]))
        local_abs, local_pos = torch.topk(cosine.abs(), k=local_k, dim=1)
        local_signed = cosine.gather(1, local_pos)
        local_feature = local_pos + start
        candidate_abs = torch.cat([best_abs, local_abs], dim=1)
        candidate_signed = torch.cat([best_signed, local_signed], dim=1)
        candidate_feature = torch.cat([best_feature, local_feature], dim=1)
        keep_abs, keep_pos = torch.topk(candidate_abs, k=min(top_k, int(candidate_abs.shape[1])), dim=1)
        best_abs = keep_abs
        best_signed = candidate_signed.gather(1, keep_pos)
        best_feature = candidate_feature.gather(1, keep_pos)
    return {"feature": best_feature.cpu(), "cosine": best_signed.cpu(), "abs_cosine": best_abs.cpu()}


def _component_results(matches: dict[str, torch.Tensor], *, top_k: int) -> list[dict[str, Any]]:
    feature_ids = matches["feature"]
    signed = matches["cosine"]
    absolute = matches["abs_cosine"]
    out = []
    for component in range(int(feature_ids.shape[0])):
        nearest = [
            {
                "rank": rank + 1,
                "sae_feature": int(feature_ids[component, rank]),
                "cosine": float(signed[component, rank]),
                "abs_cosine": float(absolute[component, rank]),
            }
            for rank in range(top_k)
        ]
        item: dict[str, Any] = {
            "component": component,
            "nearest_sae_feature": nearest[0]["sae_feature"],
            "cosine": nearest[0]["cosine"],
            "abs_cosine": nearest[0]["abs_cosine"],
        }
        if top_k > 1:
            item["nearest_sae_features"] = nearest
        out.append(item)
    return out


def _ica_fit_acceptance(metadata: dict[str, Any], *, p95_threshold: float) -> dict[str, object]:
    final_p95 = _final_lim_p95(metadata)
    if bool(metadata.get("converged", False)):
        return {"accepted": True, "reason": "strict_converged", "final_p95": final_p95}
    if p95_threshold >= 0 and final_p95 is not None and final_p95 <= p95_threshold:
        return {"accepted": True, "reason": f"final_p95<={p95_threshold:g}", "final_p95": final_p95}
    if not metadata:
        return {"accepted": True, "reason": "metadata_missing", "final_p95": final_p95}
    return {"accepted": False, "reason": "strict_unconverged_and_p95_not_accepted", "final_p95": final_p95}


def _final_lim_p95(metadata: dict[str, Any]) -> float | None:
    history = metadata.get("lim_stats_history")
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict) and last.get("p95") is not None:
            return float(last["p95"])
    return None


def _tensor_summary(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().cpu().to(dtype=torch.float64)
    q = torch.quantile(x, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], dtype=torch.float64))
    return {
        "min": float(x.min().item()),
        "p01": float(q[0].item()),
        "p05": float(q[1].item()),
        "median": float(q[2].item()),
        "p95": float(q[3].item()),
        "p99": float(q[4].item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def _layer_csv_row(model_name: str, layer: str, result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model_name": model_name,
        "layer": layer,
        "n_components": result["n_components"],
        "hidden_size": result["hidden_size"],
        "sae_width": result["sae_width"],
        "converged": result["converged"],
        "accepted": result["accepted"],
        "acceptance_reason": result["acceptance_reason"],
        "final_lim_p95": result["final_lim_p95"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    for stat, value in result["nearest_abs_cosine_summary"].items():
        row[f"nearest_abs_cosine_{stat}"] = value
    return row


def _component_csv_rows(model_name: str, layer: str, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_name,
            "layer": layer,
            "component": int(item["component"]),
            "nearest_sae_feature": int(item["nearest_sae_feature"]),
            "cosine": float(item["cosine"]),
            "abs_cosine": float(item["abs_cosine"]),
        }
        for item in components
    ]


def _layer_name(value: str) -> str:
    return value if value.startswith("layer_") else f"layer_{int(value):02d}"


def _layer_index(layer: str) -> int:
    return int(layer.removeprefix("layer_"))


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(NORM_EPS)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return torch.device(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
