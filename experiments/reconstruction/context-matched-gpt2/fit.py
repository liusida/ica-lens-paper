#!/usr/bin/env python3
"""Fit a context-matched GPT-2 ICA lens for the OAI SAE comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens._activation_dataset import (
    ActivationDataset,
    ActivationDatasetWriter,
    check_disk_space,
    sample_metadata,
)
from icalens._capture import capture_resid_post
from icalens.cli.fit_text import (
    TextDocument,
    load_pile_documents,
    peak_rss_gib,
    resolve_document_framing,
    sample_positions,
    set_cuda_memory_limit,
)
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"
OFFICIAL_CONTEXT_LENGTH = 1024
CONTEXT_LENGTH = 64
CANDIDATE_TOKENS = 5_465_620
FITTING_TOKENS = 1_000_000
FRAMING_TOKEN_ID = 50_256
LAYERS = tuple(range(12))
HIDDEN_SIZE = 768

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_RUN_OUTPUT = HERE / "run"
DEFAULT_LENS_OUTPUT = (
    ROOT / "local-icalens-models" / "experimental" / "icalens-gpt2-small-pile10k-context-length-64"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, default=DEFAULT_RUN_OUTPUT)
    parser.add_argument("--lens-output", type=Path, default=DEFAULT_LENS_OUTPUT)
    parser.add_argument("--capture-layers-at-once", type=int, default=len(LAYERS))
    parser.add_argument("--fit-batch-size", type=int, default=32768)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--objective-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-vram-gb", type=float)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate pinned Hub objects and tokenizer framing without fitting.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_arguments(args)
    api = HfApi()
    _validate_pins(api)
    first_row = next(
        iter(
            load_dataset(
                DATASET_ID,
                split=DATASET_SPLIT,
                revision=DATASET_REVISION,
                streaming=True,
            )
        )
    )
    _validate_dataset_row(first_row)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.eos_token_id != FRAMING_TOKEN_ID:
        raise ValueError(
            f"pinned GPT-2 EOS token changed: {tokenizer.eos_token_id} != {FRAMING_TOKEN_ID}"
        )
    framing = resolve_document_framing(tokenizer, "auto", model_id=MODEL_ID)
    if framing["strategy"] != "prepend-eos" or framing["token_id"] != FRAMING_TOKEN_ID:
        raise ValueError(f"unexpected GPT-2 document-framing policy: {framing}")
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if int(config.n_layer) != len(LAYERS) or int(config.n_embd) != HIDDEN_SIZE:
        raise ValueError(
            f"pinned GPT-2 shape changed: layers={config.n_layer}, hidden={config.n_embd}"
        )
    if args.validate_only:
        print(
            "Validated context-matched fit inputs: "
            f"{MODEL_ID}@{MODEL_REVISION}, {DATASET_ID}@{DATASET_REVISION}, "
            f"candidate text field={TEXT_FIELD!r}, fit context={CONTEXT_LENGTH}, "
            f"framing token={FRAMING_TOKEN_ID}."
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("the context-matched fit requires CUDA")
    if args.max_vram_gb is not None:
        set_cuda_memory_limit(args.max_vram_gb)

    run_output = args.run_output.expanduser().resolve()
    lens_output = args.lens_output.expanduser().resolve()
    activation_output = run_output / "activations"
    tokens_path = run_output / "prepared-inputs.safetensors"
    _validate_existing_lens_output(lens_output, activation_output)
    source = source_provenance()
    warn_if_dirty(source)
    resolved = _resolved(
        args,
        run_output=run_output,
        lens_output=lens_output,
        document_framing=framing,
    )
    run = ResumableRun.open(
        output=run_output,
        resolved=resolved,
        source=source,
        status="running",
    )

    completed = _completed_units(tokens_path, activation_output, lens_output)
    total = 1 + 2 * len(LAYERS)
    with (
        _mark_failed(run),
        ExperimentDisplay(
            output=run_output,
            title="Context-matched GPT-2 ICA fit",
            completed=len(completed),
            total=total,
            source_dirty=cast(bool, source.get("dirty", False)),
            command=sys.argv,
            unit_label="checkpoints",
            completed_unit_ids=completed,
        ) as display,
    ):
        if "tokens" not in completed:
            display.phase("Preparing controlled 64-token inputs", dataset=DATASET_ID)
            prepared = _prepare_inputs(tokenizer, document_framing=framing, seed=args.seed)
            _save_safetensors_atomic(tokens_path, prepared)
            display.complete_unit("tokens", refresh=True)
        else:
            prepared = _load_prepared_inputs(tokens_path)

        writer = _activation_writer(
            activation_output,
            prepared=prepared,
            args=args,
            document_framing=framing,
        )
        missing = writer.missing_layers
        if missing:
            required, available = check_disk_space(
                activation_output, required_bytes=writer.required_bytes
            )
            print(
                f"Activation cache: {required / 1024**3:.1f} GiB recommended; "
                f"{available / 1024**3:.1f} GiB available."
            )
            display.phase("Loading pinned GPT-2", revision=MODEL_REVISION[:12])
            model = _load_model()
            group_size = min(args.capture_layers_at_once, len(missing))
            for start in range(0, len(missing), group_size):
                group = missing[start : start + group_size]
                display.phase(
                    "Capturing resid_post activations",
                    layers=",".join(map(str, group)),
                )
                _capture_group(
                    model,
                    prepared=prepared,
                    layers=group,
                    writer=writer,
                )
                for layer in group:
                    display.complete_unit(f"capture:{layer}", refresh=True)
            del model
            torch.cuda.empty_cache()
        writer.finish()

        dataset = ActivationDataset(activation_output)
        lens = _load_or_create_lens(lens_output, expected_provenance=dataset.provenance)
        for layer in LAYERS:
            if layer in lens.available_layers:
                continue
            display.phase("Fitting FastICA", layer=layer, max_iter=args.max_iter)
            lens.fit(
                dataset.layer(layer),
                layer=layer,
                n_components=HIDDEN_SIZE,
                algorithm="parallel",
                fun="logcosh",
                max_iter=args.max_iter,
                random_state=args.seed,
                progress=True,
                device="cuda",
                batch_size=args.fit_batch_size,
                objective_every=args.objective_every,
                provenance=dataset.provenance,
            )
            lens.save(lens_output)
            display.complete_unit(f"fit:{layer}", refresh=True)
            print(f"Checkpointed fitted layer {layer} to {lens_output}")

        run.update(
            available_layers=list(lens.available_layers),
            lens_output=str(lens_output),
            peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            peak_rss_gib=peak_rss_gib(),
        )
        run.set_status("complete", complete=True)
        display.phase("Complete", lens=lens_output)


@contextmanager
def _mark_failed(run: ResumableRun) -> Iterator[None]:
    try:
        yield
    except BaseException:
        run.set_status("failed")
        raise


def _validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "capture_layers_at_once",
        "fit_batch_size",
        "max_iter",
        "objective_every",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _validate_pins(api: HfApi) -> None:
    model_sha = api.model_info(MODEL_ID, revision=MODEL_REVISION).sha
    dataset_sha = api.dataset_info(DATASET_ID, revision=DATASET_REVISION).sha
    if model_sha != MODEL_REVISION:
        raise RuntimeError(f"model revision resolved to {model_sha}, expected {MODEL_REVISION}")
    if dataset_sha != DATASET_REVISION:
        raise RuntimeError(
            f"dataset revision resolved to {dataset_sha}, expected {DATASET_REVISION}"
        )


def _validate_dataset_row(row: Any) -> None:
    text = row.get(TEXT_FIELD) if isinstance(row, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"dataset rows must contain nonempty {TEXT_FIELD!r} text")


def _resolved(
    args: argparse.Namespace,
    *,
    run_output: Path,
    lens_output: Path,
    document_framing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "gpt2_pile10k_context_length_control_v2",
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {
            "repo_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "field": TEXT_FIELD,
        },
        "selection": {
            "context_length": CONTEXT_LENGTH,
            "official_candidate_context_length": OFFICIAL_CONTEXT_LENGTH,
            "candidate_tokens": CANDIDATE_TOKENS,
            "fitting_tokens": FITTING_TOKENS,
            "sampling_seed": args.seed,
            "candidate_policy": "official_1024_context_pile10k_token_stream",
            "selected_token_policy": "same_flat_candidate_indices_as_official_seeded_sampler",
            "reframing": "prepend_eos_then_up_to_63_candidate_tokens",
            "document_framing": document_framing,
            "sample_metadata": {
                "identity_fields": ["document_index", "position", "token_id"],
                "context_field": "context_position",
                "position_convention": "official_input_position_including_framing_offset",
            },
        },
        "fit": {
            "layers": list(LAYERS),
            "activation_site": "resid_post",
            "preprocessing": "none",
            "components": HIDDEN_SIZE,
            "algorithm": "parallel",
            "fun": "logcosh",
            "max_iter": args.max_iter,
            "objective_every": args.objective_every,
            "batch_size": args.fit_batch_size,
            "seed": args.seed,
        },
        "capture": {
            "strategy": "official_one_context_at_a_time",
            "layers_at_once": args.capture_layers_at_once,
            "dtype": "bfloat16",
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("datasets", "transformers", "torch", "gb10-load-llm")
        },
        "run_output": str(run_output),
        "lens_output": str(lens_output),
    }


def _prepare_inputs(
    tokenizer: Any, *, document_framing: dict[str, Any], seed: int
) -> dict[str, torch.Tensor]:
    documents = load_pile_documents(
        tokenizer,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        split=DATASET_SPLIT,
        text_field=TEXT_FIELD,
        candidate_token_budget=CANDIDATE_TOKENS,
        context_length=OFFICIAL_CONTEXT_LENGTH,
        document_framing=document_framing,
    )
    selected = sample_positions(documents, token_budget=FITTING_TOKENS, seed=seed)
    official_samples = sample_metadata(documents, selected)
    prepared = _reframe_selected_documents(documents, selected)
    for name, expected in official_samples.items():
        if not torch.equal(prepared[name], expected):
            raise RuntimeError(
                f"context-64 reframing changed official sample metadata field {name!r}"
            )
    return prepared


def _reframe_selected_documents(
    documents: list[TextDocument], selected: dict[int, torch.Tensor]
) -> dict[str, torch.Tensor]:
    input_rows: list[torch.Tensor] = []
    attention_rows: list[torch.Tensor] = []
    selection_rows: list[torch.Tensor] = []
    token_rows: list[torch.Tensor] = []
    document_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    context_position_rows: list[torch.Tensor] = []
    content_length = CONTEXT_LENGTH - 1
    progress = tqdm(total=FITTING_TOKENS, desc="Reframe selected tokens", unit="tok")
    try:
        for document_index, document in enumerate(documents):
            positions = selected.get(document_index)
            if positions is None:
                continue
            content = document.input_ids[1:]
            selected_content = torch.zeros(content.shape[0], dtype=torch.bool)
            selected_content[positions - 1] = True
            for start in range(0, int(content.shape[0]), content_length):
                stop = min(start + content_length, int(content.shape[0]))
                chunk_selection = selected_content[start:stop]
                if not bool(chunk_selection.any()):
                    continue
                chunk = content[start:stop]
                valid = int(chunk.shape[0]) + 1
                input_row = torch.full((CONTEXT_LENGTH,), FRAMING_TOKEN_ID, dtype=torch.long)
                input_row[1:valid] = chunk
                attention_row = torch.zeros(CONTEXT_LENGTH, dtype=torch.long)
                attention_row[:valid] = 1
                selection_row = torch.zeros(CONTEXT_LENGTH, dtype=torch.bool)
                selection_row[1:valid] = chunk_selection
                chosen = torch.nonzero(chunk_selection, as_tuple=False).flatten()
                input_rows.append(input_row)
                attention_rows.append(attention_row)
                selection_rows.append(selection_row)
                token_rows.append(chunk[chosen])
                document_rows.append(
                    torch.full((chosen.numel(),), document_index, dtype=torch.long)
                )
                # Preserve the official cache's position convention: positions
                # include the framing token at position 0.
                position_rows.append(chosen + start + 1)
                context_position_rows.append(chosen + 1)
                progress.update(int(chosen.numel()))
    finally:
        progress.close()
    result = {
        "input_ids": torch.stack(input_rows),
        "attention_mask": torch.stack(attention_rows),
        "selection_mask": torch.stack(selection_rows),
        "token_id": torch.cat(token_rows),
        "document_index": torch.cat(document_rows),
        "position": torch.cat(position_rows),
        "context_position": torch.cat(context_position_rows),
    }
    if int(result["selection_mask"].sum()) != FITTING_TOKENS:
        raise RuntimeError("reframing did not preserve the exact fitting-token count")
    return result


def _load_prepared_inputs(path: Path) -> dict[str, torch.Tensor]:
    values = load_file(path)
    required = {
        "input_ids",
        "attention_mask",
        "selection_mask",
        "token_id",
        "document_index",
        "position",
        "context_position",
    }
    if set(values) != required:
        raise ValueError(f"invalid prepared input keys at {path}: {sorted(values)}")
    shape = values["input_ids"].shape
    if len(shape) != 2 or shape[1] != CONTEXT_LENGTH:
        raise ValueError(f"invalid prepared input shape at {path}: {tuple(shape)}")
    if values["attention_mask"].shape != shape or values["selection_mask"].shape != shape:
        raise ValueError(f"prepared masks do not match input shape at {path}")
    if int(values["selection_mask"].sum()) != FITTING_TOKENS:
        raise ValueError(f"prepared selection does not contain {FITTING_TOKENS} tokens")
    for name in ("token_id", "document_index", "position", "context_position"):
        if values[name].shape != (FITTING_TOKENS,):
            raise ValueError(f"invalid prepared {name} shape at {path}: {values[name].shape}")
    context_positions = values["context_position"]
    if int(context_positions.min()) < 1 or int(context_positions.max()) >= CONTEXT_LENGTH:
        raise ValueError(f"prepared context positions are outside 1..{CONTEXT_LENGTH - 1}")
    return values


def _activation_writer(
    path: Path,
    *,
    prepared: dict[str, torch.Tensor],
    args: argparse.Namespace,
    document_framing: dict[str, Any],
) -> ActivationDatasetWriter:
    samples = {
        "document_index": prepared["document_index"],
        "position": prepared["position"],
        "context_position": prepared["context_position"],
        "token_id": prepared["token_id"],
    }
    return ActivationDatasetWriter(
        path,
        model={"repo_id": MODEL_ID, "revision": MODEL_REVISION, "type": "base"},
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        layers=LAYERS,
        sample_count=FITTING_TOKENS,
        hidden_size=HIDDEN_SIZE,
        dtype=torch.bfloat16,
        provenance={
            "dataset": {
                "repo_id": DATASET_ID,
                "revision": DATASET_REVISION,
                "split": DATASET_SPLIT,
            },
            "text_field": TEXT_FIELD,
            "candidate_tokens": CANDIDATE_TOKENS,
            "context_length": CONTEXT_LENGTH,
            "candidate_context_length": OFFICIAL_CONTEXT_LENGTH,
            "document_framing": document_framing,
            "fitting_tokens": FITTING_TOKENS,
            "sampling_seed": args.seed,
            "token_scope": "all",
        },
        samples=samples,
    )


def _load_model() -> torch.nn.Module:
    model = cast(
        torch.nn.Module,
        load_model_to_cuda(
            AutoModelForCausalLM,
            MODEL_ID,
            device="cuda",
            dtype=torch.bfloat16,
            touch="auto",
            low_cpu_mem_usage=True,
            revision=MODEL_REVISION,
        ),
    )
    model.eval()
    return model


def _capture_group(
    model: torch.nn.Module,
    *,
    prepared: dict[str, torch.Tensor],
    layers: tuple[int, ...],
    writer: ActivationDatasetWriter,
) -> None:
    with writer.group(layers) as sink:
        progress = tqdm(total=FITTING_TOKENS, desc="Capture activations", unit="tok")
        try:
            sequence_count = int(prepared["input_ids"].shape[0])
            for sequence_index in range(sequence_count):
                valid = int(prepared["attention_mask"][sequence_index].sum())
                input_ids = prepared["input_ids"][sequence_index, :valid].unsqueeze(0).to("cuda")
                positions = torch.nonzero(
                    prepared["selection_mask"][sequence_index, :valid], as_tuple=False
                ).flatten().to("cuda")
                captured = capture_resid_post(
                    model,
                    model_inputs={"input_ids": input_ids},
                    layers=layers,
                    positions=positions,
                )
                sink.append(captured)
                progress.update(int(positions.numel()))
        finally:
            progress.close()


def _load_or_create_lens(
    path: Path, *, expected_provenance: dict[str, Any] | None = None
) -> ICALens:
    manifest = path / "icalens.json"
    if manifest.is_file():
        lens = ICALens.from_pretrained(path)
        expected = (MODEL_ID, MODEL_REVISION, "resid_post", "none")
        actual = (
            lens.model_id,
            lens.model_revision,
            lens.activation_site,
            lens.icalens_preprocessing,
        )
        if actual != expected:
            raise ValueError(f"incompatible existing lens at {path}: {actual} != {expected}")
        for layer in lens.available_layers:
            artifact = lens._get_layer(layer)
            if expected_provenance is not None:
                actual_provenance = artifact.fitting.get("provenance")
                if actual_provenance != expected_provenance:
                    raise ValueError(
                        f"existing lens layer {layer} was fitted from a different "
                        f"activation dataset: {path}"
                    )
        return lens
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"lens output is nonempty but has no icalens.json: {path}")
    return ICALens(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_type="base",
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        icalens_preprocessing="none",
    )


def _validate_existing_lens_output(lens_output: Path, activation_output: Path) -> None:
    if not (lens_output / "icalens.json").is_file():
        return
    activation_manifest = activation_output / "activations.json"
    if not activation_manifest.is_file():
        raise ValueError(
            f"existing lens output cannot be matched to a missing activation cache: "
            f"{lens_output}"
        )
    import json

    payload = json.loads(activation_manifest.read_text())
    if payload.get("status") != "complete":
        raise ValueError(
            f"existing lens output cannot be reused while the activation cache is incomplete: "
            f"{lens_output}"
        )
    dataset = ActivationDataset(activation_output)
    _load_or_create_lens(lens_output, expected_provenance=dataset.provenance)


def _completed_units(tokens_path: Path, activation_output: Path, lens_output: Path) -> set[str]:
    completed: set[str] = set()
    if tokens_path.is_file():
        _load_prepared_inputs(tokens_path)
        completed.add("tokens")
    manifest = activation_output / "activations.json"
    if manifest.is_file():
        import json

        payload = json.loads(manifest.read_text())
        for layer, entry in payload.get("layers", {}).items():
            if entry.get("status") == "complete":
                completed.add(f"capture:{int(layer)}")
    if (lens_output / "icalens.json").is_file():
        lens = _load_or_create_lens(lens_output)
        completed.update(f"fit:{layer}" for layer in lens.available_layers)
    return completed


def _save_safetensors_atomic(path: Path, values: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({name: value.contiguous() for name, value in values.items()}, temporary)
    load_file(temporary)
    temporary.replace(path)


if __name__ == "__main__":
    main()
