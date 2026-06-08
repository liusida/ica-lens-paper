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
DEFAULT_ICA_ROOT = RESULTS_DIR / "demo" / "ica"
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "demo" / "saebench_tpp"
DEFAULT_ARTIFACTS_ROOT = RESULTS_DIR / "demo" / "saebench_artifacts"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small SAEBench TPP evaluation on a demo ICA artifact.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--saebench-model-name", default="gpt2-small")
    parser.add_argument("--layer", default="layer_06")
    parser.add_argument("--token-budget", type=int, default=3000)
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--saebench-root", type=Path, default=None)
    parser.add_argument("--saebench-python", type=Path, default=None)
    parser.add_argument("--hook-name-template", default="blocks.{layer}.hook_resid_post")
    parser.add_argument("--dataset", default="LabHC/bias_in_bios_class_set1")
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=3)
    parser.add_argument("--n-values", nargs="*", type=int, default=[1, 2, 5])
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--sae-batch-size", type=int, default=32)
    parser.add_argument("--llm-dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-save-activations", action="store_true")
    parser.add_argument("--no-reexec", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    args.saebench_root, args.saebench_python = _resolve_saebench_env(args)
    _reexec_under_saebench_python(args)
    _prepare_imports(args.saebench_root)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    import torch
    from ica_lens.saebench_adapters import load_fastica_sae
    from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
    from sae_bench.evals.scr_and_tpp.main import run_eval
    from sae_bench.sae_bench_utils.general_utils import setup_environment, str_to_dtype

    started_at = time.time()
    device = setup_environment() if args.device == "auto" else args.device
    layer_index = _layer_index(args.layer)
    artifact_prefix = args.ica_root.resolve() / f"{args.model}_tok{args.token_budget}" / f"{args.layer}_fastica"
    if not artifact_prefix.with_suffix(".pt").is_file():
        raise FileNotFoundError(f"Missing ICA artifact: {artifact_prefix.with_suffix('.pt')}")

    cfg = ScrAndTppEvalConfig(model_name=args.saebench_model_name, perform_scr=False)
    cfg.dataset_names = [args.dataset]
    cfg.train_set_size = int(args.train_size)
    cfg.test_set_size = int(args.test_size)
    cfg.context_length = int(args.context_length)
    cfg.probe_epochs = int(args.probe_epochs)
    cfg.n_values = [int(value) for value in args.n_values]
    cfg.llm_batch_size = int(args.llm_batch_size)
    cfg.sae_batch_size = int(args.sae_batch_size)
    cfg.llm_dtype = str(args.llm_dtype)
    cfg.random_seed = 42

    sae = load_fastica_sae(
        artifact_prefix=artifact_prefix,
        model_name=args.saebench_model_name,
        hook_layer=layer_index,
        hook_name=str(args.hook_name_template).format(layer=layer_index),
        device=device,
        dtype=str_to_dtype(str(args.llm_dtype)),
        sign_mode="two_sign",
        expected_preprocess="with_normalization",
    )
    sae_name = f"demo_ica_{args.model}_{args.layer}_tok{args.token_budget}"
    result = _lookup_result(
        run_eval(
            cfg,
            selected_saes=[(sae_name, sae)],
            device=device,
            output_path=str(args.output_root.resolve()),
            force_rerun=bool(args.force_rerun),
            clean_up_activations=False,
            save_activations=not bool(args.no_save_activations),
            artifacts_path=str(args.artifacts_root.resolve()),
        ),
        args.output_root.resolve(),
        sae_name,
    )
    report = {
        "status": "ok",
        "analysis": "saebench_tpp_mini",
        "model": args.model,
        "layer": args.layer,
        "token_budget": args.token_budget,
        "ica_artifact": str(artifact_prefix.with_suffix(".pt")),
        "settings": {
            "dataset_names": cfg.dataset_names,
            "train_set_size": cfg.train_set_size,
            "test_set_size": cfg.test_set_size,
            "context_length": cfg.context_length,
            "probe_epochs": cfg.probe_epochs,
            "n_values": cfg.n_values,
            "llm_batch_size": cfg.llm_batch_size,
            "sae_batch_size": cfg.sae_batch_size,
            "llm_dtype": cfg.llm_dtype,
            "device": str(device),
        },
        "metrics": result.get("eval_result_metrics", {}),
        "details": result.get("eval_result_details", []),
        "sae_cfg_dict": result.get("sae_cfg_dict", {}),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    output_path = args.output_root.resolve() / "summary.json"
    _write_json(output_path, report)
    print(f"wrote TPP summary: {output_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


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


def _lookup_result(results: dict[str, Any], output_root: Path, sae_name: str) -> dict[str, Any]:
    for key in (sae_name, f"{sae_name}_custom_sae"):
        if key in results:
            return results[key]
    for key in (sae_name, f"{sae_name}_custom_sae"):
        path = output_root / "tpp" / f"{key}_eval_results.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    if len(results) == 1:
        return next(iter(results.values()))
    raise KeyError(f"Could not find SAEBench result for {sae_name!r}; got {sorted(results)}")


def _layer_index(layer: str) -> int:
    return int(layer.removeprefix("layer_"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
