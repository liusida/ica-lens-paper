#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "ica_lens").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Could not find v6 project root containing src/ica_lens and configs.")


V6_ROOT = _project_root()
V6_SRC = V6_ROOT / "src"
SAEBENCH_ROOT = V6_ROOT / "vendor" / "SAEBench"
SAEBENCH_PYTHON = SAEBENCH_ROOT / ".venv" / "bin" / "python"
QWEN35_SAEBENCH_ROOT = V6_ROOT / "vendor" / "SAEBench-qwen35"
QWEN35_SAEBENCH_PYTHON = QWEN35_SAEBENCH_ROOT / ".venv" / "bin" / "python"
RESULTS_DIR = V6_ROOT / "results"
DEFAULT_ACTIVATION_ROOT = RESULTS_DIR / "demo" / "activations"
DEFAULT_ICA_ROOT = RESULTS_DIR / "demo" / "ica"
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "demo" / "saebench_sparse_probe"
DEFAULT_ARTIFACTS_ROOT = RESULTS_DIR / "demo" / "saebench_artifacts"
ALL_METHODS = ("pca", "ica", "sae_baseline", "itda", "matryoshka_128", "matryoshka_512")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SAEBench sparse probing for ICA Lens demo artifacts and baselines.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--saebench-model-name", default=None)
    parser.add_argument("--layer", default="layer_06")
    parser.add_argument("--token-budget", type=int, default=3000)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--saebench-root", type=Path, default=None)
    parser.add_argument("--saebench-python", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["all"], choices=[*ALL_METHODS, "all"])
    parser.add_argument("--hook-name-template", default=None)
    parser.add_argument("--dataset", default="LabHC/bias_in_bios_class_set1")
    parser.add_argument("--probe-train-size", type=int, default=64)
    parser.add_argument("--probe-test-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 5])
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--sae-batch-size", type=int, default=32)
    parser.add_argument("--llm-dtype", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--itda-k", type=int, default=8)
    parser.add_argument("--itda-tau", type=float, default=4e-4)
    parser.add_argument("--itda-max-atoms", type=int, default=128)
    parser.add_argument("--itda-batch-size", type=int, default=128)
    parser.add_argument("--itda-max-train-tokens", type=int, default=3000)
    parser.add_argument("--itda-encode-chunk-size", type=int, default=256)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-save-activations", action="store_true")
    parser.add_argument("--train-full-feature-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-reexec", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    args.methods = _selected_methods(args.methods, model=str(args.model), layer=str(args.layer))
    args.saebench_root, args.saebench_python = _resolve_saebench_env(args)
    _reexec_under_saebench_python(args)
    _prepare_imports(args.saebench_root)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    from ica_lens.config import load_toml
    from ica_lens.paths import CONFIGS_DIR
    from ica_lens.sparse_probe_baselines import (
        RunTarget,
        collect_run_rows,
        prepare_artifacts,
        prepare_dry_run,
        run_sparse_probe_worker,
    )

    started_at = time.time()
    target_methods = [RunTarget(model=str(args.model), layer=str(args.layer), method=method) for method in args.methods]
    output_root = args.output_root.resolve()
    activation_root = args.activation_root.resolve()
    ica_root = args.ica_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    run_name = f"demo_sparse_probe_{args.model}_{args.layer}_tok{args.token_budget}"

    if args.dry_run:
        plan = prepare_dry_run(
            output_root=output_root,
            activation_root=activation_root,
            targets=target_methods,
            itda_k=int(args.itda_k),
            itda_tau=float(args.itda_tau),
            itda_max_atoms=int(args.itda_max_atoms),
        )
        print(json.dumps({"targets": [target.__dict__ for target in target_methods], "plan": plan}, indent=2))
        return 0

    prepare_artifacts(
        output_root=output_root,
        activation_root=activation_root,
        ica_root=ica_root,
        token_budget=int(args.token_budget),
        targets=target_methods,
        itda_k=int(args.itda_k),
        itda_tau=float(args.itda_tau),
        itda_max_atoms=int(args.itda_max_atoms),
        itda_batch_size=int(args.itda_batch_size),
        itda_max_train_tokens=int(args.itda_max_train_tokens),
        device=str(args.device),
        force=bool(args.force_rerun),
    )

    comparison_cfg = load_toml(CONFIGS_DIR / "comparisons" / f"{args.model}.toml")
    tpp_cfg = dict(comparison_cfg["saebench_tpp"])
    saebench_model_name = str(args.saebench_model_name or tpp_cfg["saebench_model_name"])
    llm_dtype = str(args.llm_dtype or tpp_cfg.get("llm_dtype", "float32"))

    rows = []
    for target in target_methods:
        row = run_sparse_probe_worker(
            target=target,
            output_root=output_root,
            activation_root=activation_root,
            ica_root=ica_root,
            token_budget=int(args.token_budget),
            saebench_artifacts_path=artifacts_root,
            datasets=[str(args.dataset)],
            k_values=[int(value) for value in args.k_values],
            probe_train_size=int(args.probe_train_size),
            probe_test_size=int(args.probe_test_size),
            context_length=int(args.context_length),
            llm_batch_size=int(args.llm_batch_size),
            sae_batch_size=int(args.sae_batch_size),
            llm_dtype=llm_dtype,
            saebench_model_name=saebench_model_name,
            hook_name_template=str(args.hook_name_template or tpp_cfg["hook_name_template"]),
            run_name=run_name,
            itda_k=int(args.itda_k),
            itda_tau=float(args.itda_tau),
            itda_max_atoms=int(args.itda_max_atoms),
            itda_encode_chunk_size=int(args.itda_encode_chunk_size),
            force_rerun=bool(args.force_rerun),
            save_activations=not bool(args.no_save_activations),
            train_full_feature_probe=bool(args.train_full_feature_probe),
        )
        rows.append(row)

    all_rows = collect_run_rows(output_root=output_root, run_name=run_name)
    report = {
        "status": "ok",
        "analysis": "saebench_sparse_probe_demo_comparison",
        "model": args.model,
        "layer": args.layer,
        "token_budget": int(args.token_budget),
        "methods": args.methods,
        "settings": {
            "saebench_model_name": saebench_model_name,
            "dataset": str(args.dataset),
            "probe_train_set_size": int(args.probe_train_size),
            "probe_test_set_size": int(args.probe_test_size),
            "context_length": int(args.context_length),
            "k_values": [int(value) for value in args.k_values],
            "llm_batch_size": int(args.llm_batch_size),
            "sae_batch_size": int(args.sae_batch_size),
            "llm_dtype": llm_dtype,
            "device": str(args.device),
            "itda_k": int(args.itda_k),
            "itda_max_atoms": int(args.itda_max_atoms),
        },
        "rows": rows,
        "all_rows": all_rows,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    output_path = output_root / f"{args.model}_tok{args.token_budget}" / "summary.json"
    _write_json(output_path, report)
    print(f"wrote sparse probe summary: {output_path}")
    return 0


def _selected_methods(values: list[str], *, model: str, layer: str) -> list[str]:
    methods = list(ALL_METHODS) if "all" in values else list(dict.fromkeys(values))
    if (model, layer) != ("gemma2_2b", "layer_12"):
        methods = [method for method in methods if not method.startswith("matryoshka_")]
    return methods


def _resolve_saebench_env(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.saebench_root is not None or args.saebench_python is not None:
        root = args.saebench_root or SAEBENCH_ROOT
        python = args.saebench_python or root / ".venv" / "bin" / "python"
        return root.resolve(), _absolute_path(python)
    if args.model == "qwen3_5_2b_base":
        return QWEN35_SAEBENCH_ROOT.resolve(), _absolute_path(QWEN35_SAEBENCH_PYTHON)
    return SAEBENCH_ROOT.resolve(), _absolute_path(SAEBENCH_PYTHON)


def _absolute_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _reexec_under_saebench_python(args: argparse.Namespace) -> None:
    if args.no_reexec:
        return
    target = args.saebench_python
    target_venv = target.parent.parent
    if Path(sys.prefix).resolve() == target_venv.resolve():
        return
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing SAEBench interpreter: {target}. "
            "Run `bash scripts/setup_saebench_envs.sh` from v6 first."
        )
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(target_venv)
    env["PATH"] = f"{target.parent}{os.pathsep}{env.get('PATH', '')}"
    env.pop("UV_RUN_RECURSION_DEPTH", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    os.execve(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:], "--no-reexec"], env)


def _prepare_imports(saebench_root: Path) -> None:
    if not saebench_root.is_dir() or not (saebench_root / "sae_bench").is_dir():
        raise RuntimeError(
            f"Missing vendored SAEBench source at {saebench_root}. "
            "Run `git submodule update --init --recursive` or restore vendor/SAEBench."
        )
    if str(V6_SRC) not in sys.path:
        sys.path.insert(0, str(V6_SRC))
    if str(saebench_root) not in sys.path:
        sys.path.insert(0, str(saebench_root))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
