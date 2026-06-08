#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ica_lens.config import load_toml, write_json
from ica_lens.paths import CONFIGS_DIR, RESULTS_DIR, V6_ROOT


DEFAULT_CONFIG = CONFIGS_DIR / "demo" / "mini_3k.toml"
DEFAULT_MODELS = ["gpt2", "gemma2_2b", "qwen3_5_2b_base"]
DEMO_DEFAULTS = {
    "gpt2": {"layer": "layer_06", "n_components": 128},
    "gemma2_2b": {"layer": "layer_12", "n_components": 128},
    "qwen3_5_2b_base": {"layer": "layer_12", "n_components": 128},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the miniature ICA Lens reproduction path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--model", default=None, help="Run one model. Prefer --models for multi-model runs.")
    parser.add_argument("--models", nargs="+", default=None, choices=DEFAULT_MODELS)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--n-components", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-documents", type=int, default=256)
    parser.add_argument("--skip-saebench", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    started_at = time.time()
    demo_cfg = dict(load_toml(args.config)["demo"])
    token_budget = int(args.token_budget or demo_cfg.get("token_budget", 3000))
    models = _selected_models(args, demo_cfg)
    if len(models) > 1 and args.layer is not None:
        raise ValueError("--layer can only be used with a single selected model.")
    if len(models) > 1 and args.n_components is not None:
        raise ValueError("--n-components can only be used with a single selected model.")

    reports = [
        _run_model_demo(
            model=model,
            layer=str(args.layer or DEMO_DEFAULTS[model]["layer"]),
            token_budget=token_budget,
            n_components=str(args.n_components or DEMO_DEFAULTS[model]["n_components"]),
            device=str(args.device),
            max_documents=int(args.max_documents),
            skip_saebench=bool(args.skip_saebench),
            force=bool(args.force),
        )
        for model in models
    ]

    report = {
        "status": "ok",
        "models": reports,
        "token_budget": token_budget,
        "skipped_saebench": bool(args.skip_saebench),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    output = RESULTS_DIR / "demo" / "demo_report.json"
    write_json(output, report)
    print(f"wrote demo report: {output}")
    return 0


def _selected_models(args: argparse.Namespace, demo_cfg: dict[str, object]) -> list[str]:
    if args.models is not None:
        return list(args.models)
    if args.model is not None:
        if args.model not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported demo model {args.model!r}; expected one of {', '.join(DEFAULT_MODELS)}.")
        return [str(args.model)]
    config_model = str(demo_cfg.get("model", ""))
    if config_model and config_model != "all":
        if config_model not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported demo config model {config_model!r}.")
        return [config_model]
    return list(DEFAULT_MODELS)


def _run_model_demo(
    *,
    model: str,
    layer: str,
    token_budget: int,
    n_components: str,
    device: str,
    max_documents: int,
    skip_saebench: bool,
    force: bool,
) -> dict[str, object]:
    started_at = time.time()

    activation_root = RESULTS_DIR / "demo" / "activations"
    ica_root = RESULTS_DIR / "demo" / "ica"
    activation_config = CONFIGS_DIR / "activations" / f"{model}.toml"
    fit_config = CONFIGS_DIR / "fit_ica" / f"{model}.toml"
    activation_manifest = activation_root / f"{model}_tok{token_budget}" / "manifest.json"
    ica_manifest = ica_root / f"{model}_tok{token_budget}" / "manifest.json"
    sparse_probe_output_root = RESULTS_DIR / "demo" / "saebench_sparse_probe"
    sparse_probe_root = sparse_probe_output_root / f"{model}_tok{token_budget}"
    tpp_root = RESULTS_DIR / "demo" / "saebench_tpp" / f"{model}_tok{token_budget}"
    sparse_probe_summary = sparse_probe_root / "summary.json"
    tpp_summary = tpp_root / "summary.json"
    comparison_cfg = load_toml(CONFIGS_DIR / "comparisons" / f"{model}.toml")
    tpp_cfg = dict(comparison_cfg["saebench_tpp"])

    capture_cmd = [
        sys.executable,
        str(V6_ROOT / "workflows" / "01_capture_activations.py"),
        "--config",
        str(activation_config),
        "--output-root",
        str(activation_root),
        "--token-budget",
        str(token_budget),
        "--shard-token-budget",
        str(token_budget),
        "--max-documents",
        str(max_documents),
        "--device",
        device,
        "--layers",
        layer,
    ]
    fit_cmd = [
        sys.executable,
        str(V6_ROOT / "workflows" / "02_fit_ica.py"),
        "--config",
        str(fit_config),
        "--activation-root",
        str(activation_root),
        "--token-budget",
        str(token_budget),
        "--output-root",
        str(ica_root),
        "--layers",
        layer,
        "--n-components",
        n_components,
        "--device",
        device,
        "--no-progress",
    ]
    if force:
        capture_cmd.append("--force")
        fit_cmd.append("--force")

    _run(capture_cmd)
    _run(fit_cmd)
    if not skip_saebench:
        sparse_probe_cmd = [
            sys.executable,
            str(V6_ROOT / "workflows" / "08_run_saebench_sparse_probe.py"),
            "--model",
            model,
            "--layer",
            layer,
            "--token-budget",
            str(token_budget),
            "--ica-root",
            str(ica_root),
            "--output-root",
            str(sparse_probe_output_root),
            "--saebench-model-name",
            str(tpp_cfg["saebench_model_name"]),
            "--hook-name-template",
            str(tpp_cfg["hook_name_template"]),
            "--llm-dtype",
            str(tpp_cfg.get("llm_dtype", "float32")),
            "--device",
            device,
            "--no-save-activations",
        ]
        tpp_cmd = [
            sys.executable,
            str(V6_ROOT / "workflows" / "07_run_saebench_tpp.py"),
            "--model",
            model,
            "--layer",
            layer,
            "--token-budget",
            str(token_budget),
            "--ica-root",
            str(ica_root),
            "--output-root",
            str(tpp_root),
            "--saebench-model-name",
            str(tpp_cfg["saebench_model_name"]),
            "--hook-name-template",
            str(tpp_cfg["hook_name_template"]),
            "--llm-dtype",
            str(tpp_cfg.get("llm_dtype", "float32")),
            "--device",
            device,
            "--no-save-activations",
        ]
        if force:
            sparse_probe_cmd.append("--force-rerun")
            tpp_cmd.append("--force-rerun")
        _run(tpp_cmd)
        _run(sparse_probe_cmd)

    return {
        "status": "ok",
        "model": model,
        "layer": layer,
        "token_budget": token_budget,
        "n_components": int(n_components),
        "activation_manifest": str(activation_manifest),
        "ica_manifest": str(ica_manifest),
        "sparse_probe_summary": None if skip_saebench else str(sparse_probe_summary),
        "tpp_summary": None if skip_saebench else str(tpp_summary),
        "skipped_saebench": bool(skip_saebench),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=V6_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
