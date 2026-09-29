"""Profile fitted ICA components from a dataset or captured activations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from icalens import ICALens
from icalens._activation_dataset import ActivationDataset
from icalens.cli.fit_chat import load_chat_documents
from icalens.cli.fit_text import load_pile_documents

from ._status import log


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens profile", description=__doc__)
    parser.add_argument(
        "operation",
        nargs="?",
        choices=(
            "add-r-lens",
            "refresh-statistics",
            "refresh-examples",
            "refresh-examples-rank",
        ),
        help=(
            "Optional profile operation. Use 'add-r-lens' to enrich existing "
            "profiles, or 'refresh-statistics' to recompute moments and tail "
            "selection from captured activations, or 'refresh-examples' to gather "
            "top-score examples on the already selected tail, or "
            "'refresh-examples-rank' to add ranks to already stored examples."
        ),
    )
    parser.add_argument("--lens", required=True, help="Local lens directory or Hub repository.")
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int, help="One fitted layer (compatibility form).")
    layer_group.add_argument("--layers", help="Comma-separated fitted layers, or 'all'.")
    parser.add_argument(
        "--dataset",
        help="Hugging Face dataset repository (not used by 'add-r-lens').",
    )
    parser.add_argument(
        "--activations",
        type=Path,
        help="Previously captured activation dataset; mutually exclusive with --dataset.",
    )
    parser.add_argument("--activation-batch-size", type=int, default=8192)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--input-type", choices=("auto", "text", "chat"), default="auto")
    parser.add_argument(
        "--token-scope", choices=("assistant", "user", "content", "all"), default="all"
    )
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--top-k-examples", type=int, default=20)
    parser.add_argument("--logit-lens-top-k", type=int, default=20)
    parser.add_argument("--logit-lens-batch-size", type=int, default=64)
    parser.add_argument(
        "--r-lens",
        type=Path,
        help="Compatible local R-lens artifact to add R-lens vocabulary readouts.",
    )
    parser.add_argument("--r-lens-top-k", type=int, default=20)
    parser.add_argument("--r-lens-batch-size", type=int, default=8)
    parser.add_argument(
        "--allow-base-model-transfer",
        action="store_true",
        help=(
            "Explicitly allow a dimension-compatible base-model R-lens to enrich "
            "an instruct-model ICA Lens and record the transfer in provenance."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deliberately recompute requested profiles after compatibility validation.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination artifact directory. By default, a local --lens directory "
            "is updated in place; Hub sources require this option."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = _resolve_output(args.lens, args.output)
    lens = ICALens.from_pretrained(args.lens)
    layers = _parse_layers(args.layer, args.layers, lens.available_layers)
    log(f"Requested layers: {','.join(map(str, layers))}")
    if args.operation == "add-r-lens":
        if args.r_lens is None:
            raise ValueError("'add-r-lens' requires --r-lens")
        for layer in layers:
            log(f"Profiling layer {layer} (R-lens enrichment)...")
            profile = lens.add_r_lens_profile(
                layer=layer,
                r_lens=args.r_lens,
                top_k=args.r_lens_top_k,
                batch_size=args.r_lens_batch_size,
                device=args.device,
                progress=not args.no_progress,
                allow_base_model_transfer=args.allow_base_model_transfer,
            )
            saved_to = lens.checkpoint_component_profile(output, layer=layer)
            log(f"Added R-lens readouts to layer {layer}: {len(profile['components'])} components.")
            log(f"Checkpointed profiled lens to {saved_to}")
        return
    if args.operation == "refresh-statistics":
        if args.activations is None or args.dataset is not None:
            raise ValueError("'refresh-statistics' requires --activations and no --dataset")
        _refresh_cached_statistics(args, lens, layers, output)
        return
    if args.operation == "refresh-examples":
        if args.activations is None or args.dataset is not None:
            raise ValueError("'refresh-examples' requires --activations and no --dataset")
        _refresh_cached_examples(args, lens, layers, output)
        return
    if args.operation == "refresh-examples-rank":
        if args.activations is None or args.dataset is not None:
            raise ValueError("'refresh-examples-rank' requires --activations and no --dataset")
        _refresh_cached_example_ranks(args, lens, layers, output)
        return
    if (args.dataset is None) == (args.activations is None):
        raise ValueError("full profiling requires exactly one of --dataset or --activations")
    if args.activations is not None:
        _profile_cached_activations(args, lens, layers, output)
        return
    input_type = (
        ("chat" if lens.model_type == "instruct" else "text")
        if args.input_type == "auto"
        else args.input_type
    )
    revision = HfApi().dataset_info(args.dataset).sha
    if revision is None:
        raise RuntimeError("could not resolve the dataset revision")
    for layer in layers:
        log(f"Profiling layer {layer}...")
        dataset = load_dataset(args.dataset, split=args.split, revision=revision, streaming=True)
        inputs = _dataset_inputs(
            dataset,
            input_type=input_type,
            text_field=args.text_field,
            messages_field=args.messages_field,
        )
        profile = lens.profile_components(
            inputs,
            layer=layer,
            token_scope=args.token_scope,
            max_tokens=args.max_tokens,
            top_k_examples=args.top_k_examples,
            logit_lens_top_k=args.logit_lens_top_k,
            logit_lens_batch_size=args.logit_lens_batch_size,
            r_lens=args.r_lens,
            r_lens_top_k=args.r_lens_top_k,
            r_lens_batch_size=args.r_lens_batch_size,
            allow_base_model_transfer=args.allow_base_model_transfer,
            context_length=args.context_length,
            provenance={
                "dataset": {
                    "repo_id": args.dataset,
                    "revision": str(revision),
                    "split": args.split,
                },
                "input_type": input_type,
                "token_scope": args.token_scope,
                "text_field": args.text_field if input_type == "text" else None,
                "messages_field": args.messages_field if input_type == "chat" else None,
            },
            device=args.device,
            progress=not args.no_progress,
        )
        if (output / "icalens.json").is_file():
            saved_to = lens.checkpoint_component_profile(output, layer=layer)
        else:
            saved_to = lens.save(output)
        log(
            f"Profiled layer {layer}: {len(profile['components'])} components from "
            f"{profile['n_tokens']} tokens."
        )
        log(f"Checkpointed profiled lens to {saved_to}")


def _profile_cached_activations(
    args: argparse.Namespace, lens: ICALens, layers: tuple[int, ...], output: Path
) -> None:
    dataset = ActivationDataset(args.activations)
    _validate_cached_dataset(dataset, lens, layers)
    count = min(args.max_tokens, dataset.sample_count)
    provenance = dataset.provenance
    provenance["profile_sampling"] = {
        "policy": "uniform_without_replacement",
        "seed": args.sample_seed,
        "selected_tokens": count,
        "population_tokens": dataset.sample_count,
    }
    pending = _pending_profile_layers(
        args,
        lens,
        layers,
        output=output,
        provenance=provenance,
        n_tokens=count,
    )
    if not pending:
        log("All requested component profiles are already complete and compatible.")
        return
    generator = torch.Generator(device="cpu").manual_seed(args.sample_seed)
    rows = torch.randperm(dataset.sample_count, generator=generator)[:count].sort().values
    records = _recover_cached_records(dataset, rows, lens)
    for layer in pending:
        log(f"Profiling layer {layer} from cached activations...")
        selected = dataset.layer(layer).index_select(0, rows)
        profile = lens.profile_components_from_activations(
            selected,
            records,
            layer=layer,
            batch_size=args.activation_batch_size,
            top_k_examples=args.top_k_examples,
            logit_lens_top_k=args.logit_lens_top_k,
            logit_lens_batch_size=args.logit_lens_batch_size,
            r_lens=args.r_lens,
            r_lens_top_k=args.r_lens_top_k,
            r_lens_batch_size=args.r_lens_batch_size,
            allow_base_model_transfer=args.allow_base_model_transfer,
            provenance=provenance,
            device=args.device,
            progress=not args.no_progress,
        )
        saved_to = (
            lens.checkpoint_component_profile(output, layer=layer)
            if (output / "icalens.json").is_file()
            else lens.save(output)
        )
        log(
            f"Profiled layer {layer} from cached activations: "
            f"{len(profile['components'])} components from {profile['n_tokens']} tokens."
        )
        log(f"Checkpointed profiled lens to {saved_to}")


def _refresh_cached_statistics(
    args: argparse.Namespace, lens: ICALens, layers: tuple[int, ...], output: Path
) -> None:
    if not (output / "icalens.json").is_file():
        raise ValueError("'refresh-statistics' requires an existing local lens artifact")
    dataset = ActivationDataset(args.activations)
    _validate_cached_dataset(dataset, lens, layers)
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    count = min(args.max_tokens, dataset.sample_count)
    rows = None
    sampling_policy = "all" if count == dataset.sample_count else "uniform_without_replacement"
    if count != dataset.sample_count:
        generator = torch.Generator(device="cpu").manual_seed(args.sample_seed)
        rows = torch.randperm(dataset.sample_count, generator=generator)[:count].sort().values
    provenance = dataset.provenance
    provenance["statistics_sampling"] = {
        "policy": sampling_policy,
        "seed": args.sample_seed if rows is not None else None,
        "selected_tokens": count,
        "population_tokens": dataset.sample_count,
    }
    pending, completed = _pending_statistics_layers(
        args,
        lens,
        layers,
        provenance=provenance,
    )
    completed_text = ",".join(map(str, completed)) if completed else "none"
    log(f"Durable completed layers: {completed_text}")
    if not pending:
        log("All requested profile statistics are already complete and compatible.")
        return
    for layer in pending:
        log(f"Refreshing score statistics for layer {layer} from cached activations...")
        profile = lens.refresh_profile_statistics_from_activations(
            dataset.layer(layer),
            layer=layer,
            rows=rows,
            batch_size=args.activation_batch_size,
            provenance=provenance,
            device=args.device,
            progress=not args.no_progress,
        )
        saved_to = lens.checkpoint_component_profile(output, layer=layer)
        log(
            f"Refreshed layer {layer}: {len(profile['components'])} components "
            f"from {count} activation rows."
        )
        log(f"Checkpointed profiled lens to {saved_to}")


def _refresh_cached_examples(
    args: argparse.Namespace, lens: ICALens, layers: tuple[int, ...], output: Path
) -> None:
    if not (output / "icalens.json").is_file():
        raise ValueError("'refresh-examples' requires an existing local lens artifact")
    dataset = ActivationDataset(args.activations)
    _validate_cached_dataset(dataset, lens, layers)
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    count = min(args.max_tokens, dataset.sample_count)
    generator = torch.Generator(device="cpu").manual_seed(args.sample_seed)
    rows = torch.randperm(dataset.sample_count, generator=generator)[:count].sort().values
    provenance = dataset.provenance
    provenance["profile_sampling"] = {
        "policy": "uniform_without_replacement",
        "seed": args.sample_seed,
        "selected_tokens": count,
        "population_tokens": dataset.sample_count,
    }
    pending, completed = _pending_example_layers(args, lens, layers, provenance=provenance)
    log("Durable completed layers: " + (",".join(map(str, completed)) if completed else "none"))
    if not pending:
        log("All requested profile examples are already complete and compatible.")
        return
    records = _recover_cached_records(dataset, rows, lens)
    for layer in pending:
        log(f"Refreshing selected-tail examples for layer {layer}...")
        profile = lens.refresh_profile_examples_from_activations(
            dataset.layer(layer),
            records,
            layer=layer,
            rows=rows,
            batch_size=args.activation_batch_size,
            top_k_examples=args.top_k_examples,
            provenance=provenance,
            device=args.device,
            progress=not args.no_progress,
        )
        saved_to = lens.checkpoint_component_profile(output, layer=layer)
        log(
            f"Refreshed layer {layer}: selected-tail examples for "
            f"{len(profile['components'])} components from {count} activation rows."
        )
        log(f"Checkpointed profiled lens to {saved_to}")


def _refresh_cached_example_ranks(
    args: argparse.Namespace, lens: ICALens, layers: tuple[int, ...], output: Path
) -> None:
    if not (output / "icalens.json").is_file():
        raise ValueError("'refresh-examples-rank' requires an existing local lens artifact")
    dataset = ActivationDataset(args.activations)
    _validate_cached_dataset(dataset, lens, layers)
    profiles: dict[int, dict[str, Any]] = {}
    required_positions: set[tuple[int, int]] = set()
    for layer in layers:
        artifact = lens._layers[layer]
        if artifact.profile_file is None:
            raise ValueError(f"layer {layer} has no component profile")
        profile = lens._get_profile(artifact)
        expected_cache = dataset.provenance["activation_dataset"]["manifest_sha256"]
        actual_cache = (
            profile.get("example_provenance", {})
            .get("activation_dataset", {})
            .get("manifest_sha256")
        )
        if actual_cache != expected_cache:
            raise ValueError(
                f"layer {layer} examples came from activation cache {actual_cache!r}, "
                f"not the requested cache {expected_cache!r}"
            )
        profiles[layer] = profile
        for component in profile.get("components", []):
            for sign in ("positive", "negative"):
                for occurrence in (
                    component.get("examples", {}).get(sign, {}).get("occurrences", [])
                ):
                    required_positions.add(
                        (int(occurrence["source_index"]), int(occurrence["position"]))
                    )
    metadata = dataset.samples()
    row_by_position: dict[tuple[int, int], tuple[int, int]] = {}
    for row, (source, position, token_id) in enumerate(
        zip(
            metadata["document_index"].tolist(),
            metadata["position"].tolist(),
            metadata["token_id"].tolist(),
            strict=True,
        )
    ):
        key = (int(source), int(position))
        if key in required_positions:
            if key in row_by_position:
                raise ValueError(f"activation cache contains duplicate token position {key}")
            row_by_position[key] = (row, int(token_id))
    for layer in layers:
        profile = profiles[layer]
        occurrence_rows: list[tuple[int, str, int, int]] = []
        missing: list[str] = []
        for component_index, component in enumerate(profile.get("components", [])):
            for sign in ("positive", "negative"):
                occurrences = component.get("examples", {}).get(sign, {}).get("occurrences", [])
                for occurrence_index, occurrence in enumerate(occurrences):
                    key = (int(occurrence["source_index"]), int(occurrence["position"]))
                    match = row_by_position.get(key)
                    if match is None:
                        missing.append(
                            f"C{component_index} {sign} occurrence {occurrence_index + 1}"
                        )
                    else:
                        row, token_id = match
                        if int(occurrence["token_id"]) != token_id:
                            raise ValueError(
                                f"layer {layer} C{component_index} occurrence token does not "
                                "match the activation cache"
                            )
                        occurrence_rows.append((component_index, sign, occurrence_index, row))
        if missing:
            raise ValueError(
                f"layer {layer} has {len(missing)} occurrences absent from the activation cache; "
                f"first missing: {missing[0]}"
            )
        already_ranked = profile.get("selection", {}).get(
            "example_absolute_score_rank"
        ) == "competition_rank_by_absolute_score" and all(
            "absolute_score_rank"
            in profile["components"][component]["examples"][sign]["occurrences"][occurrence]
            for component, sign, occurrence, _row in occurrence_rows
        )
        if already_ranked and not args.force:
            log(f"Layer {layer} example ranks are already complete.")
            continue
        log(f"Adding absolute-score ranks to {len(occurrence_rows)} examples in layer {layer}...")
        lens.refresh_profile_example_ranks_from_activations(
            dataset.layer(layer),
            occurrence_rows,
            layer=layer,
            batch_size=args.activation_batch_size,
            device=args.device,
            progress=not args.no_progress,
        )
        saved_to = lens.checkpoint_component_profile(output, layer=layer)
        log(f"Checkpointed ranked examples for layer {layer} to {saved_to}")


def _pending_example_layers(
    args: argparse.Namespace,
    lens: ICALens,
    layers: tuple[int, ...],
    *,
    provenance: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pending: list[int] = []
    completed: list[int] = []
    incompatible: list[str] = []
    for layer in layers:
        artifact = lens._layers[layer]
        if artifact.profile_file is None:
            raise ValueError(
                f"layer {layer} has no component profile; compute statistics/full profile first"
            )
        profile = lens._get_profile(artifact)
        selection = profile.get("selection", {})
        components = profile.get("components", [])
        if not all(
            component.get("tail_direction") in ("positive", "negative") for component in components
        ):
            raise ValueError(
                f"layer {layer} has no computed tail directions; run refresh-statistics first"
            )
        has_new_examples = (
            selection.get("example_selection") == "top_absolute_score_on_selected_tail"
            and selection.get("top_k_examples_on_selected_tail") == args.top_k_examples
            and selection.get("example_absolute_score_rank") == "competition_rank_by_absolute_score"
            and "example_provenance" in profile
        )
        if args.force or not has_new_examples:
            pending.append(layer)
        elif profile.get("example_provenance") == provenance:
            completed.append(layer)
        else:
            incompatible.append(f"layer {layer}: example provenance differs")
    if incompatible:
        raise ValueError(
            "existing refreshed examples are incompatible with this request ("
            + "; ".join(incompatible)
            + "); use --force to deliberately replace them"
        )
    return tuple(pending), tuple(completed)


def _pending_statistics_layers(
    args: argparse.Namespace,
    lens: ICALens,
    layers: tuple[int, ...],
    *,
    provenance: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate refreshed statistics and split requested layers into pending/completed."""
    pending: list[int] = []
    completed: list[int] = []
    incompatible: list[str] = []
    for layer in layers:
        artifact = lens._layers[layer]
        if artifact.profile_file is None:
            raise ValueError(
                f"layer {layer} has no component profile to refresh; run full profiling first"
            )
        profile = lens._get_profile(artifact)
        selection = profile.get("selection", {})
        already_refreshed = (
            selection.get("score_statistics") == "population_mean_variance_skewness_excess_kurtosis"
            and selection.get("sign_selection") == "population_skewness"
        )
        if args.force:
            pending.append(layer)
        elif not already_refreshed:
            pending.append(layer)
        elif profile.get("score_statistics_provenance") == provenance:
            completed.append(layer)
        else:
            incompatible.append(f"layer {layer}: statistics provenance differs")
    if incompatible:
        raise ValueError(
            "existing refreshed profile statistics are incompatible with this request ("
            + "; ".join(incompatible)
            + "); use --force to deliberately replace them"
        )
    return tuple(pending), tuple(completed)


def _pending_profile_layers(
    args: argparse.Namespace,
    lens: ICALens,
    layers: tuple[int, ...],
    *,
    output: Path,
    provenance: dict[str, Any],
    n_tokens: int,
) -> tuple[int, ...]:
    """Return layers needing work, rejecting stale profiles before corpus replay."""
    if not (output / "icalens.json").is_file():
        return layers
    expected_selection = {
        "top_k_examples_on_selected_tail": args.top_k_examples,
        "example_selection": "top_absolute_score_on_selected_tail",
        "example_absolute_score_rank": "competition_rank_by_absolute_score",
        "logit_lens_top_k": args.logit_lens_top_k,
        "logit_lens_batch_size": args.logit_lens_batch_size,
        "score_statistics": "population_mean_variance_skewness_excess_kurtosis",
        "sign_selection": "population_skewness",
    }
    expected_r_lens_sha = _sha256(args.r_lens) if args.r_lens is not None else None
    pending: list[int] = []
    incompatible: list[str] = []
    for layer in layers:
        # Inspect profile metadata without loading the much larger fitted tensor.
        artifact = lens._layers[layer]
        if artifact.profile_file is None or args.force:
            pending.append(layer)
            continue
        profile = lens._get_profile(artifact)
        differences: list[str] = []
        selection = profile.get("selection", {})
        if any(selection.get(key) != value for key, value in expected_selection.items()):
            differences.append("selection")
        if profile.get("provenance") != provenance:
            differences.append("provenance")
        if profile.get("n_tokens") != n_tokens:
            differences.append("n_tokens")
        if expected_r_lens_sha is not None:
            stored_r_lens = profile.get("r_lens_provenance") or {}
            if stored_r_lens.get("sha256") != expected_r_lens_sha:
                differences.append("r_lens_artifact")
            unavailable_for_layer = (
                selection.get("r_lens_top_k") is None
                and selection.get("r_lens_batch_size") is None
                and stored_r_lens.get("sha256") == expected_r_lens_sha
            )
            if not unavailable_for_layer:
                if selection.get("r_lens_top_k") != args.r_lens_top_k:
                    differences.append("r_lens_top_k")
                if selection.get("r_lens_batch_size") != args.r_lens_batch_size:
                    differences.append("r_lens_batch_size")
        if differences:
            incompatible.append(f"layer {layer}: {', '.join(differences)}")
        else:
            log(f"Reusing compatible completed profile for layer {layer}.")
    if incompatible:
        details = "; ".join(incompatible)
        raise ValueError(
            "existing component profiles are incompatible with this request "
            f"({details}); use --force to deliberately replace the requested profiles"
        )
    return tuple(pending)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_cached_dataset(
    dataset: ActivationDataset, lens: ICALens, layers: tuple[int, ...]
) -> None:
    model = dataset.model
    problems = []
    if model.get("repo_id") != lens.model_id:
        problems.append(f"model {model.get('repo_id')!r} != {lens.model_id!r}")
    if lens.model_revision and model.get("revision") != lens.model_revision:
        problems.append(f"revision {model.get('revision')!r} != {lens.model_revision!r}")
    if dataset.manifest["activation_site"] != lens.activation_site:
        problems.append("activation site differs")
    if dataset.manifest["layer_indexing"] != lens.layer_indexing:
        problems.append("layer indexing differs")
    if lens._hidden_size is not None and dataset.hidden_size != lens._hidden_size:
        problems.append("hidden size differs")
    missing = sorted(set(layers) - set(dataset.available_layers))
    if missing:
        problems.append(f"missing activation layers {missing}")
    if problems:
        raise ValueError("incompatible activation dataset: " + "; ".join(problems))


def _recover_cached_records(
    dataset: ActivationDataset, rows: torch.Tensor, lens: ICALens
) -> list[dict[str, Any]]:
    """Replay pinned tokenization and validate every selected activation row."""
    provenance = dataset.manifest["provenance"]
    recorded_runtime = provenance.get("tokenization_runtime")
    if isinstance(recorded_runtime, dict):
        current_runtime = {
            package: importlib.metadata.version(package)
            for package in ("datasets", "tokenizers", "transformers")
        }
        if recorded_runtime != current_runtime:
            raise ValueError(
                "activation records require the tokenization runtime used during capture: "
                f"recorded={recorded_runtime}, current={current_runtime}"
            )
    source = provenance["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(
        lens.model_id, revision=lens.model_revision, trust_remote_code=True
    )
    dataset_id, dataset_revision = _replay_dataset_source(source)
    candidate_tokens = int(provenance["candidate_tokens"])
    context_length = int(provenance["context_length"])
    documents: Any
    if lens.model_type == "instruct":
        documents = load_chat_documents(
            tokenizer,
            dataset_id=dataset_id,
            dataset_revision=str(dataset_revision),
            split=str(source["split"]),
            messages_field=str(provenance.get("messages_field", "messages")),
            token_scope=str(provenance.get("token_scope", "all")),
            candidate_token_budget=candidate_tokens,
            context_length=context_length,
        )
    else:
        documents = load_pile_documents(
            tokenizer,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=str(source["split"]),
            text_field=str(provenance.get("text_field", "text")),
            candidate_token_budget=candidate_tokens,
            context_length=context_length,
            document_framing=provenance.get("document_framing"),
        )
    metadata = dataset.samples()
    records = []
    for activation_row in rows.tolist():
        source_index = int(metadata["document_index"][activation_row])
        position = int(metadata["position"][activation_row])
        token_id = int(metadata["token_id"][activation_row])
        if source_index >= len(documents):
            raise ValueError(f"sample references unavailable document {source_index}")
        input_ids = documents[source_index].input_ids
        if position >= len(input_ids) or int(input_ids[position]) != token_id:
            raise ValueError(
                "captured token metadata does not match replayed dataset/tokenizer at "
                f"activation row {activation_row}; refusing to write a mixed profile"
            )
        start, end = max(0, position - 4), min(len(input_ids), position + 5)
        token = tokenizer.convert_ids_to_tokens(token_id)
        context_ids = input_ids[start:end].tolist()
        target_offset = position - start
        context = tokenizer.decode(context_ids, clean_up_tokenization_spaces=False)
        context_target_start = len(
            tokenizer.decode(context_ids[:target_offset], clean_up_tokenization_spaces=False)
        )
        context_target_end = len(
            tokenizer.decode(context_ids[: target_offset + 1], clean_up_tokenization_spaces=False)
        )
        records.append(
            {
                "token": str(token),
                "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                "token_id": token_id,
                "position": position,
                "context": context,
                "context_target_start": context_target_start,
                "context_target_end": context_target_end,
                "source_index": source_index,
            }
        )
    log(
        f"Validated {len(records)} sampled activation rows against the pinned "
        "dataset and tokenizer."
    )
    return records


def _replay_dataset_source(source: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve and validate Hub or local dataset provenance for token replay."""
    if "repo_id" in source:
        return str(source["repo_id"]), str(source["revision"])
    if "path" not in source or "sha256" not in source:
        raise ValueError("activation dataset has incomplete dataset provenance")
    path = Path(str(source["path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"captured local dataset is unavailable: {path}")
    expected = str(source["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            "captured local dataset checksum differs from activation provenance: "
            f"expected {expected}, got {actual}"
        )
    return str(path), None


def _resolve_output(lens_source: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    local_source = Path(lens_source).expanduser()
    if local_source.is_dir():
        return local_source.resolve()
    raise ValueError(
        "--output is required when --lens is a Hugging Face repository; "
        "local lens directories are profiled in place by default"
    )


def _dataset_inputs(
    dataset: Iterable[dict[str, Any]],
    *,
    input_type: str,
    text_field: str,
    messages_field: str,
) -> Iterable[str | list[dict[str, str]]]:
    field = text_field if input_type == "text" else messages_field
    for row in dataset:
        if field not in row:
            raise KeyError(f"dataset row does not contain field {field!r}")
        value = row[field]
        if input_type == "text":
            if isinstance(value, str) and value:
                yield value
        elif isinstance(value, list) and value:
            yield value


def _parse_layers(
    layer: int | None, layers: str | None, available: tuple[int, ...]
) -> tuple[int, ...]:
    requested: tuple[int, ...]
    if layer is not None:
        requested = (layer,)
    elif layers == "all":
        requested = available
    else:
        assert layers is not None
        try:
            requested = tuple(int(value.strip()) for value in layers.split(","))
        except ValueError as error:
            raise ValueError("--layers must be comma-separated integers or 'all'") from error
    if not requested:
        raise ValueError("no layers were requested")
    unavailable = sorted(set(requested) - set(available))
    if unavailable:
        raise ValueError(f"unavailable layers: {unavailable}; available layers: {list(available)}")
    return requested
