#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ica_lens.config import write_json
from ica_lens.paths import RESULTS_DIR, V6_ROOT


DEMO_MODELS = ["gpt2", "gemma2_2b", "qwen3_5_2b_base"]
DEMO_LAYERS = {
    "gpt2": "layer_06",
    "gemma2_2b": "layer_12",
    "qwen3_5_2b_base": "layer_12",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ICA Lens reproduction workflows.")
    parser.add_argument("--mode", choices=["demo", "paper"], default="demo")
    parser.add_argument("--clean", action="store_true", help="Delete generated results and fetched artifacts before running.")
    parser.add_argument("--skip-fetch", action="store_true", help="Do not fetch released Hugging Face artifacts.")
    parser.add_argument("--skip-saebench", action="store_true", help="Skip SAEBench TPP and sparse-probe demo steps.")
    parser.add_argument("--token-budget", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-documents", type=int, default=256)
    parser.add_argument("--examples-per-region", type=int, default=2)
    parser.add_argument("--erf-limit", type=int, default=1, help="ERF components per model in demo mode; use -1 for all.")
    parser.add_argument("--random-directions", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="Force recomputation of workflow outputs.")
    args = parser.parse_args(argv)

    if args.mode != "demo":
        raise SystemExit("paper mode is not implemented yet; use --mode demo for the standalone smoke reproduction.")

    started_at = time.time()
    report: dict[str, Any] = {
        "mode": args.mode,
        "status": "running",
        "settings": {
            "clean": bool(args.clean),
            "skip_fetch": bool(args.skip_fetch),
            "skip_saebench": bool(args.skip_saebench),
            "token_budget": int(args.token_budget),
            "device": str(args.device),
            "max_documents": int(args.max_documents),
            "examples_per_region": int(args.examples_per_region),
            "erf_limit": int(args.erf_limit),
            "random_directions": int(args.random_directions),
            "force": bool(args.force),
        },
        "steps": [],
    }
    output = RESULTS_DIR / "demo" / "reproduce_all_report.json"

    try:
        if args.clean:
            _clean_generated()
            report["steps"].append({"name": "clean_generated", "status": "ok"})

        if not args.skip_fetch:
            _run_step(report, "fetch_artifacts", [sys.executable, "scripts/fetch_artifacts.py", "--models", "--databases"])
            _run_step(report, "verify_artifacts", [sys.executable, "scripts/verify_artifacts.py"])

        run_demo_cmd = [
            sys.executable,
            "scripts/run_demo.py",
            "--models",
            *DEMO_MODELS,
            "--token-budget",
            str(args.token_budget),
            "--device",
            str(args.device),
            "--max-documents",
            str(args.max_documents),
        ]
        if args.force or args.clean:
            run_demo_cmd.append("--force")
        if args.skip_saebench:
            run_demo_cmd.append("--skip-saebench")
        _run_step(report, "run_demo", run_demo_cmd)

        db_path = RESULTS_DIR / "demo" / "databases" / "ica_probe_demo.sqlite"
        _run_step(
            report,
            "build_explorer_db",
            [
                sys.executable,
                "workflows/04_build_explorer_db.py",
                "--models",
                *DEMO_MODELS,
                "--token-budget",
                str(args.token_budget),
                "--activation-root",
                "results/demo/activations",
                "--ica-root",
                "results/demo/ica",
                "--output-db",
                str(db_path),
                "--examples-per-region",
                str(args.examples_per_region),
                "--force",
            ],
        )

        _run_step(
            report,
            "compute_nongaussianity",
            [
                sys.executable,
                "workflows/03_compute_nongaussianity.py",
                "--models",
                *DEMO_MODELS,
                "--token-budget",
                str(args.token_budget),
                "--activation-root",
                "results/demo/activations",
                "--ica-root",
                "results/demo/ica",
                "--output-root",
                "results/demo/nongaussianity",
                "--families",
                "ica",
                "random",
                "--random-directions",
                str(args.random_directions),
                "--force",
            ],
        )

        erf_cmd = [
            sys.executable,
            "workflows/05_populate_erf.py",
            "--models",
            *DEMO_MODELS,
            "--db-path",
            str(db_path),
            "--ica-root",
            "results/demo/ica",
            "--token-budget",
            str(args.token_budget),
            "--device",
            str(args.device),
            "--force",
        ]
        if int(args.erf_limit) >= 0:
            erf_cmd.extend(["--limit", str(args.erf_limit)])
        _run_step(report, "populate_erf", erf_cmd)

        _run_step(
            report,
            "compare_ica_sae_overlap",
            [
                sys.executable,
                "workflows/06_compare_ica_sae_overlap.py",
                "--models",
                *DEMO_MODELS,
                "--token-budget",
                str(args.token_budget),
                "--ica-root",
                "results/demo/ica",
                "--output-root",
                "results/demo/ica_sae_overlap",
                "--force",
            ],
        )

        report["status"] = "ok"
        return_code = 0
    except subprocess.CalledProcessError as exc:
        report["status"] = "failed"
        report["failed_step_returncode"] = int(exc.returncode)
        return_code = int(exc.returncode) or 1
    finally:
        report["elapsed_seconds"] = round(time.time() - started_at, 3)
        write_json(output, report)
        print(f"Wrote {output}")

    return return_code


def _run_step(report: dict[str, Any], name: str, cmd: list[str]) -> None:
    started_at = time.time()
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=V6_ROOT, check=True)
    report["steps"].append({"name": name, "status": "ok", "elapsed_seconds": round(time.time() - started_at, 3), "cmd": cmd})


def _clean_generated() -> None:
    for path in [
        V6_ROOT / "artifacts" / "fetched",
        RESULTS_DIR / "demo",
        RESULTS_DIR / "reproduced",
        RESULTS_DIR / "verification",
    ]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()
    for path in [
        V6_ROOT / "artifacts" / "fetched" / "models",
        V6_ROOT / "artifacts" / "fetched" / "databases",
    ]:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()


if __name__ == "__main__":
    raise SystemExit(main())
