#!/usr/bin/env python3
"""Run repository-level integrity checks against accepted project artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from integrity import (
    capture_replay,
    downstream_replays,
    fitting_replay,
    profile_occurrence_replay,
    profile_readout_replay,
    profiling_replay,
    reference_preflight,
    sae_adapter_replay,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Canary:
    lens: Path
    activations: Path
    r_lens: Path
    layer: int


ACTIVATION_ROOT = Path.home() / "Expansion/research/ICA-data/icalens-activations"
CANARIES = {
    "gpt2": Canary(
        lens=ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k",
        activations=ACTIVATION_ROOT / "gpt2-pile10k-1m",
        r_lens=ROOT / "local-r-lens-models/official/gpt2-small/lens.pt",
        layer=6,
    ),
    "gemma": Canary(
        lens=ROOT / "local-icalens-models/official/icalens-gemma-2-2b-pile10k",
        activations=ACTIVATION_ROOT / "gemma-2-2b-pile10k-1m",
        r_lens=ROOT / "local-r-lens-models/official/gemma-2-2b/lens.pt",
        layer=13,
    ),
    "qwen": Canary(
        lens=ROOT / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k",
        activations=ACTIVATION_ROOT / "qwen3.5-9b-base-pile10k-1m",
        r_lens=ROOT / "local-r-lens-models/official/qwen3.5-9b-base/lens.pt",
        layer=16,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*CANARIES, "all"), default="gpt2")
    parser.add_argument(
        "--check",
        choices=("reference-preflight", "capture", "fit", "profile", "sae", "downstream", "all"),
        default="all",
    )
    parser.add_argument("--reference-lens", type=Path)
    parser.add_argument("--reference-activations", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--verification-rows", type=int, default=32)
    parser.add_argument("--fit-verification-rows", type=int, default=1024)
    parser.add_argument("--verification-components", type=int, default=64)
    parser.add_argument("--verification-seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output-root", type=Path, default=ROOT / "integrity/runs")
    parser.add_argument("--verbose", action="store_true", help="Print the complete JSON report.")
    return parser.parse_args()


def git_output(repository: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repository)


def repository_identity(repository: Path) -> dict[str, Any]:
    commit = git_output(repository, "rev-parse", "HEAD").decode().strip()
    changed = git_output(repository, "diff", "--name-only", "HEAD", "-z").split(b"\0")
    untracked = git_output(repository, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    )
    paths = sorted({value for value in (*changed, *untracked) if value})
    digest = hashlib.sha256()
    digest.update(commit.encode())
    digest.update(b"\0")
    changed_paths: list[str] = []
    for encoded_path in paths:
        path_text = encoded_path.decode(errors="surrogateescape")
        path = repository / path_text
        changed_paths.append(path_text)
        digest.update(encoded_path)
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"<missing>")
    fingerprint = digest.hexdigest()
    return {
        "commit": commit,
        "dirty": bool(paths),
        "changed_paths": sorted(changed_paths),
        "fingerprint": fingerprint,
    }


def source_identity() -> dict[str, Any]:
    repositories = {"project": repository_identity(ROOT)}
    paper = ROOT / "paper/overleaf-v2.1"
    if (paper / ".git").exists():
        repositories["paper"] = repository_identity(paper)
    digest = hashlib.sha256()
    for name, identity in sorted(repositories.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(identity["fingerprint"]).encode())
        digest.update(b"\0")
    return {
        "dirty": any(bool(item["dirty"]) for item in repositories.values()),
        "fingerprint": digest.hexdigest(),
        "repositories": repositories,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def execute_check(
    check_id: str,
    operation: Callable[[], dict[str, Any]],
    *,
    log_path: Path,
    verbose: bool,
) -> dict[str, Any]:
    """Run one check, preserving detailed chatter without flooding the terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    print(
        f"START {check_id} ({started_at.isoformat(timespec='seconds')})",
        flush=True,
    )
    started = perf_counter()
    try:
        if verbose:
            result = operation()
        else:
            with log_path.open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    result = operation()
            result.setdefault("artifacts", {})["runtime_log"] = str(log_path)
    except Exception as error:
        if verbose:
            traceback.print_exc()
        else:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
        result = {
            "check": check_id,
            "status": "fail",
            "scope": {},
            "error": f"{type(error).__name__}: {error}",
            "failed_checks": ["exception"],
            "artifacts": ({"runtime_log": str(log_path)} if not verbose else {}),
        }
    finished_at = datetime.now().astimezone()
    result["timing"] = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round(perf_counter() - started, 3),
    }
    print_check_result(result)
    return result


def print_check_result(result: dict[str, Any]) -> None:
    label = "PASS" if result["status"] == "pass" else "FAIL"
    scope = result.get("scope", {})
    ids = scope.get("data_ids", []) or [
        *scope.get("input_data_ids", []),
        *scope.get("code_ids", []),
        *scope.get("output_data_ids", []),
    ]
    duration = float(result.get("timing", {}).get("duration_seconds", 0.0))
    print(f"{label} {result['check']} [{','.join(ids)}] ({duration:.1f}s)", flush=True)
    for failed in result.get("failed_checks", []):
        print(f"  failed: {failed}", flush=True)


def selected_canaries(args: argparse.Namespace) -> list[tuple[str, Canary]]:
    if args.model == "all":
        if args.reference_lens is not None or args.reference_activations is not None:
            raise ValueError("reference path overrides require selecting one --model")
        if args.layer is not None:
            raise ValueError("--layer requires selecting one --model")
        return list(CANARIES.items())
    default = CANARIES[args.model]
    return [
        (
            args.model,
            Canary(
                lens=(args.reference_lens or default.lens).expanduser().resolve(),
                activations=(args.reference_activations or default.activations)
                .expanduser()
                .resolve(),
                r_lens=default.r_lens.expanduser().resolve(),
                layer=default.layer if args.layer is None else args.layer,
            ),
        )
    ]


def run_canary(
    *,
    canary_id: str,
    canary: Canary,
    args: argparse.Namespace,
    run_root: Path,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    lens_root = canary.lens.expanduser().resolve()
    activation_root = canary.activations.expanduser().resolve()
    preflight_id = f"reference-preflight-{canary_id}-layer{canary.layer}"
    preflight = execute_check(
        preflight_id,
        lambda: reference_preflight.run(
            lens_root=lens_root,
            activation_root=activation_root,
            r_lens_path=canary.r_lens,
            layer=canary.layer,
            canary_id=canary_id,
        ),
        log_path=run_root / preflight_id / "runtime.log",
        verbose=args.verbose,
    )
    reports.append(preflight)
    if preflight["status"] != "pass":
        return reports

    if args.check in {"capture", "all"}:
        check_id = f"C10-capture-{canary_id}-layer{canary.layer}"
        check_root = run_root / check_id
        reports.append(
            execute_check(
                check_id,
                lambda: capture_replay.run(
                    lens_root=lens_root,
                    activation_root=activation_root,
                    layer=canary.layer,
                    verification_rows=args.verification_rows,
                    verification_seed=args.verification_seed,
                    output=check_root,
                    canary_id=canary_id,
                ),
                log_path=check_root / "runtime.log",
                verbose=args.verbose,
            )
        )
    if args.check in {"fit", "all"}:
        check_id = f"C11-fit-{canary_id}-layer{canary.layer}"
        check_root = run_root / check_id
        reports.append(
            execute_check(
                check_id,
                lambda: fitting_replay.run(
                    lens_root=lens_root,
                    activation_root=activation_root,
                    layer=canary.layer,
                    verification_rows=args.fit_verification_rows,
                    verification_components=args.verification_components,
                    verification_seed=args.verification_seed,
                    rtol=args.rtol,
                    atol=args.atol,
                    output=check_root,
                    canary_id=canary_id,
                ),
                log_path=check_root / "runtime.log",
                verbose=args.verbose,
            )
        )
    if args.check in {"profile", "all"}:
        check_id = f"C12-profile-statistics-{canary_id}-layer{canary.layer}"
        check_root = run_root / check_id
        reports.append(
            execute_check(
                check_id,
                lambda: profiling_replay.run(
                    lens_root=lens_root,
                    activation_root=activation_root,
                    layer=canary.layer,
                    verification_components=args.verification_components,
                    verification_seed=args.verification_seed,
                    rtol=args.rtol,
                    atol=args.atol,
                    output=check_root,
                    canary_id=canary_id,
                ),
                log_path=check_root / "runtime.log",
                verbose=args.verbose,
            )
        )
        check_id = f"C12-profile-readouts-{canary_id}-layer{canary.layer}"
        check_root = run_root / check_id
        reports.append(
            execute_check(
                check_id,
                lambda: profile_readout_replay.run(
                    lens_root=lens_root,
                    r_lens_path=canary.r_lens,
                    layer=canary.layer,
                    verification_components=args.verification_components,
                    verification_seed=args.verification_seed,
                    rtol=args.rtol,
                    atol=args.atol,
                    output=check_root,
                    canary_id=canary_id,
                ),
                log_path=check_root / "runtime.log",
                verbose=args.verbose,
            )
        )
        check_id = f"C12-profile-occurrences-{canary_id}-layer{canary.layer}"
        check_root = run_root / check_id
        reports.append(
            execute_check(
                check_id,
                lambda: profile_occurrence_replay.run(
                    lens_root=lens_root,
                    activation_root=activation_root,
                    layer=canary.layer,
                    verification_components=args.verification_components,
                    verification_seed=args.verification_seed,
                    rtol=args.rtol,
                    atol=args.atol,
                    output=check_root,
                    canary_id=canary_id,
                ),
                log_path=check_root / "runtime.log",
                verbose=args.verbose,
            )
        )
    if args.check in {"sae", "all"}:
        check_id = f"C13-sae-adapter-{canary_id}-layer{canary.layer}"
        check_root = run_root / check_id
        reports.append(
            execute_check(
                check_id,
                lambda: sae_adapter_replay.run(
                    lens_root=lens_root,
                    layer=canary.layer,
                    rtol=args.rtol,
                    atol=args.atol,
                    output=check_root,
                    canary_id=canary_id,
                ),
                log_path=check_root / "runtime.log",
                verbose=args.verbose,
            )
        )
    return reports


def main() -> None:
    args = parse_args()
    source = source_identity()
    run_root = args.output_root.expanduser().resolve() / source["fingerprint"][:16]
    reports: list[dict[str, Any]] = []
    if args.check != "downstream":
        for canary_id, canary in selected_canaries(args):
            reports.extend(
                run_canary(
                    canary_id=canary_id,
                    canary=canary,
                    args=args,
                    run_root=run_root,
                )
            )
    blockers_path = None
    if args.check in {"downstream", "all"}:
        for operation in downstream_replays.CHECKS:
            check_id = operation.__name__.replace("_", "-", 1)
            check_root = run_root / check_id
            reports.append(
                execute_check(
                    check_id,
                    lambda operation=operation, check_root=check_root: operation(check_root),
                    log_path=check_root / "runtime.log",
                    verbose=args.verbose,
                )
            )
        blockers_path = Path("/tmp") / f"icalens-integrity-blockers-{source['fingerprint'][:16]}.md"
        blockers_path.write_text(
            "# Skipped integrity relations\n\n"
            "- **C14 reconstruction capture:** skipped because the retained reconstruction "
            "cache records sampled token positions and activations, but the independent "
            "recapture path for its six heterogeneous dataset loaders has not yet been "
            "implemented. D14 identity and its downstream C21 aggregation are checked.\n"
            "- **C30 steering/language-control:** skipped because reproducing accepted "
            "generation and language judgments requires stochastic generation plus an "
            "external OpenAI evaluator. The accepted cache is not an independent numerical "
            "oracle for those model calls.\n",
            encoding="utf-8",
        )
    status = "pass" if reports and all(item["status"] == "pass" for item in reports) else "fail"
    report = {
        "format": "icalens.repository_integrity",
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "reports": reports,
    }
    report_path = run_root / f"report-{args.model}-{args.check}.json"
    atomic_write_json(report_path, report)
    if args.verbose:
        print(json.dumps(report, indent=2))
    passed = sum(item["status"] == "pass" for item in reports)
    total_duration = sum(
        float(item.get("timing", {}).get("duration_seconds", 0.0)) for item in reports
    )
    print(f"Overall: {status.upper()} ({passed}/{len(reports)}, {total_duration:.1f}s)")
    print(f"Completed: {report['created_at']}")
    print(f"Report: {report_path}")
    if blockers_path is not None:
        print(f"Skipped-check notes: {blockers_path}")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
