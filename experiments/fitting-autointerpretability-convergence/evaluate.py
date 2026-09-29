#!/usr/bin/env python3
"""Run the unchanged autointerpretability scorer over trajectory checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prepare import DEFAULT_OUTPUT, EVALUATED_CHECKPOINTS, LAYERS
from trajectory import parse_layers

from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parent
DEFAULT_EVALUATIONS = ROOT / "runs/evaluated"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_TINKER_EXPLAINER = "thinkingmachines/Inkling"
DEFAULT_TINKER_SIMULATOR = "Qwen/Qwen3.8-27B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--provider", choices=("tinker", "openai"), default="tinker")
    parser.add_argument("--model")
    parser.add_argument("--explainer-model")
    parser.add_argument("--simulator-model")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument(
        "--n-components",
        type=int,
        help="Evaluate this many prefix-stable cohort positions (default: all prepared).",
    )
    parser.add_argument(
        "--max-concurrent-checkpoints",
        type=int,
        default=11,
        help="Checkpoint evaluations to run concurrently within one component (default: 11).",
    )
    parser.add_argument(
        "--max-concurrent-simulations",
        type=int,
        default=10,
        help="Concurrent simulator requests within each checkpoint (default: 10).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    if args.provider == "tinker" and args.model:
        raise ValueError("Tinker uses --explainer-model and --simulator-model, not --model")
    if args.provider == "openai" and (args.explainer_model or args.simulator_model):
        raise ValueError("OpenAI uses --model, not Tinker model-role options")
    if args.max_concurrent_checkpoints < 1:
        raise ValueError("--max-concurrent-checkpoints must be positive")
    if args.max_concurrent_simulations < 1:
        raise ValueError("--max-concurrent-simulations must be positive")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model = args.model or DEFAULT_OPENAI_MODEL if args.provider == "openai" else None
    explainer = (
        args.explainer_model or DEFAULT_TINKER_EXPLAINER
        if args.provider == "tinker"
        else None
    )
    simulator = (
        args.simulator_model or DEFAULT_TINKER_SIMULATOR
        if args.provider == "tinker"
        else None
    )
    preparation_protocols = {}
    prepared_counts: set[int] = set()
    for iteration in EVALUATED_CHECKPOINTS:
        path = source / f"iter-{iteration:03d}" / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"prepared checkpoint is missing: {path}")
        prepared = json.loads(path.read_text())["resolved"]
        preparation_protocols[str(iteration)] = {
            key: prepared.get(key)
            for key in ("selection", "split_sizes", "seed_rule", "iteration")
        }
        for layer in layers:
            selection = json.loads(
                (source / f"iter-{iteration:03d}" / f"layer_{layer:02d}"
                 / "ica/selection.json").read_text()
            )
            prepared_counts.add(len(selection["accepted"]))
    if len(prepared_counts) != 1:
        raise ValueError("prepared checkpoints do not have one consistent cohort size")
    prepared_count = prepared_counts.pop()
    target_count = args.n_components if args.n_components is not None else prepared_count
    if not 0 < target_count <= prepared_count:
        raise ValueError(
            f"--n-components must be between 1 and prepared capacity {prepared_count}"
        )
    resolved = {
        "format": "icalens.fastica_autointerpretability_evaluation",
        "format_version": 1,
        "input": str(source),
        "checkpoints": list(EVALUATED_CHECKPOINTS),
        "layers": list(layers),
        "preparation_protocols": preparation_protocols,
        "provider": args.provider,
        "model": model,
        "explainer_model": explainer,
        "simulator_model": simulator,
        "sampling_seed": 0,
        "methods": ["ica"],
        "cohort_order": "prefix-stable persistent-row permutation",
    }
    if args.dry_run:
        for iteration in EVALUATED_CHECKPOINTS:
            subprocess.run(
                _command(
                    args, source, output, iteration, 0, model, explainer, simulator
                ),
                check=True,
            )
        return
    provenance = source_provenance()
    run = ResumableRun.open(
        output=output,
        filename="launcher-run.json",
        resolved=resolved,
        source=provenance,
        status="evaluating",
    )
    completed = {
        position
        for position in range(target_count)
        if all(
            _position_complete(
                output / f"iter-{iteration:03d}",
                source / f"iter-{iteration:03d}",
                position,
                args.provider,
                model,
                explainer,
                simulator,
                layers,
            )
            for iteration in EVALUATED_CHECKPOINTS
        )
    }
    try:
        with ExperimentDisplay(
            output=output / "logs",
            title=f"ICA Lens · autointerpretability convergence · {args.provider}",
            completed=len(completed),
            total=target_count,
            completed_unit_ids=completed,
            source_dirty=provenance.get("dirty"),
            unit_label="matched cohort positions",
            detail_filename="evaluation-detail.log",
        ) as display:
            warn_if_dirty(provenance)
            for position in range(target_count):
                if position in completed:
                    log(f"Reused matched cohort position {position + 1}/{target_count}.")
                    continue
                log(f"Starting matched cohort position {position + 1}/{target_count}.")
                pending = []
                for iteration in EVALUATED_CHECKPOINTS:
                    preparation = source / f"iter-{iteration:03d}"
                    destination = output / f"iter-{iteration:03d}"
                    if _position_complete(
                        destination, preparation, position, args.provider,
                        model, explainer, simulator,
                        layers,
                    ):
                        log(
                            f"Reused cohort position {position + 1}/{target_count} at "
                            f"iteration {iteration}."
                        )
                        continue
                    pending.append(iteration)

                display.phase(
                    "Evaluating checkpoints concurrently",
                    component=f"{position + 1}/{target_count}",
                    checkpoints=len(pending),
                    provider=args.provider,
                )
                with ThreadPoolExecutor(
                    max_workers=min(args.max_concurrent_checkpoints, len(pending) or 1)
                ) as executor:
                    futures = {
                        executor.submit(
                            _run_checkpoint,
                            _command(
                                args, source, output, iteration, position,
                                model, explainer, simulator,
                            ),
                            output / "logs" / (
                                f"component-{position + 1:02d}-iter-{iteration:03d}.log"
                            ),
                        ): iteration
                        for iteration in pending
                    }
                    for future in as_completed(futures):
                        iteration = futures[future]
                        child_log = output / "logs" / (
                            f"component-{position + 1:02d}-iter-{iteration:03d}.log"
                        )
                        try:
                            future.result()
                        except subprocess.CalledProcessError as error:
                            raise RuntimeError(
                                f"checkpoint iteration {iteration} failed; see {child_log}"
                            ) from error
                        preparation = source / f"iter-{iteration:03d}"
                        destination = output / f"iter-{iteration:03d}"
                        if not _position_complete(
                            destination, preparation, position, args.provider,
                            model, explainer, simulator,
                            layers,
                        ):
                            raise ValueError(
                                f"child did not complete cohort position {position} "
                                f"at iteration {iteration}"
                            )
                        log(
                            f"Completed cohort position {position + 1}/{target_count} at "
                            f"iteration {iteration}."
                        )
                display.complete_unit(position, refresh=True)
                log(
                    f"Completed matched cohort position {position + 1}/{target_count} "
                    "across checkpoints."
                )
            run.set_status("complete", complete=True)
            log(
                f"Complete: {target_count} matched cohort positions across all "
                "fitting checkpoints."
            )
    except BaseException:
        run.set_status("interrupted")
        raise


def _command(args, source, output, iteration, position, model, explainer, simulator):
    command = [
        "icalens", "experiment", "autointerpretability", "evaluate",
        "--input", str(source / f"iter-{iteration:03d}"),
        "--output", str(output / f"iter-{iteration:03d}"),
        "--methods", "ica", "--provider", args.provider,
        "--feature-position", str(position),
        "--sampling-seed", "0",
        "--max-concurrent", str(args.max_concurrent_simulations),
        "--env-file", str(args.env_file),
    ]
    if model:
        command.extend(("--model", model))
    if explainer:
        command.extend(("--explainer-model", explainer))
    if simulator:
        command.extend(("--simulator-model", simulator))
    if args.dry_run:
        command.append("--dry-run")
    return command


def _run_checkpoint(command: list[str], log_path: Path) -> None:
    """Run one child evaluator without letting its live display corrupt the parent."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _position_complete(
    directory: Path,
    preparation: Path,
    position: int,
    provider: str,
    model: str | None,
    explainer: str | None,
    simulator: str | None,
    layers: tuple[int, ...],
) -> bool:
    try:
        for layer in layers:
            selection = json.loads(
                (preparation / f"layer_{layer:02d}/ica/selection.json").read_text()
            )
            feature = int(selection["accepted"][position]["feature"])
            result = json.loads(
                (directory / f"layer_{layer:02d}/ica/results/feature_{feature}.json").read_text()
            )
            if (
                result.get("status") not in {"complete", "disabled"}
                or result.get("provider") != provider
                or (provider == "tinker" and result.get("sampling_seed") != 0)
            ):
                return False
            if model is not None and (
                result.get("explainer_model") != model
                or result.get("simulator_model") != model
            ):
                return False
            if explainer is not None and result.get("explainer_model") != explainer:
                return False
            if simulator is not None and result.get("simulator_model") != simulator:
                return False
        return True
    except (OSError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


if __name__ == "__main__":
    main()
