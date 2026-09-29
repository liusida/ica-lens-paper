"""Official resumable SAE suffix ERF using the shared ICA sweep engine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from icalens import __version__
from icalens._activation_dataset import ActivationDataset
from icalens.cli._status import log
from icalens.experiments import erf_gradient, erf_suffix_sweep
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._sae import SAEFeatureEncoder
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.erf_gradient import _stable_seed
from icalens.experiments.saebench_sparse_probing import _prepare_layer_baselines, _resolve_baselines

CHUNK = 32768


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class SAEReadout:
    """Adapt only score computation/provenance, without copying the ERF algorithm."""

    def __init__(self, cache: ActivationDataset, encoder: SAEFeatureEncoder, layer: int) -> None:
        self.model_id = cache.model["repo_id"]
        self.model_revision = cache.model["revision"]
        self.encoder = encoder
        self.layer = layer
        self.metadata = {"layers": {str(layer): {"fitting": {"provenance": cache.provenance}}}}

    def transform(self, hidden: torch.Tensor, *, layer: int) -> torch.Tensor:
        if layer != self.layer:
            raise ValueError("incorrect SAE layer")
        return self.encoder.encode(hidden)


def profile(
    cache: ActivationDataset,
    encoder: SAEFeatureEncoder,
    ids: list[int],
    count: int,
    display: ExperimentDisplay,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep top positive occurrences for a deterministic candidate feature set."""
    values = torch.zeros((len(ids), count), device="cuda")
    indices = torch.full((len(ids), count), -1, dtype=torch.long, device="cuda")
    hidden = cache.layer(layer)
    ids_tensor = torch.tensor(ids, device="cuda")
    for start in range(0, cache.sample_count, CHUNK):
        stop = min(start + CHUNK, cache.sample_count)
        display.phase("Profiling SAE candidates", layer=layer, dataset="Pile-10k")
        for batch in range(start, stop, 512):
            batch_end = min(batch + 512, stop)
            codes = encoder.encode(hidden[batch:batch_end].to("cuda", dtype=torch.float32))
            selected = codes[:, ids_tensor].T
            if not torch.isfinite(selected).all() or (selected < 0).any():
                raise ValueError("SAE scores must be finite and nonnegative")
            new_values, local = selected.topk(min(count, batch_end - batch), dim=1)
            candidates = torch.cat((values, new_values), dim=1)
            candidate_indices = torch.cat((indices, local + batch), dim=1)
            values, order = candidates.topk(count, dim=1)
            indices = candidate_indices.gather(1, order)
        if stop == cache.sample_count or stop % (CHUNK * 8) == 0:
            log(f"Layer {layer}: profiled {stop}/{cache.sample_count} activations")
    return values.cpu(), indices.cpu()


def prepare_feature(
    feature: int,
    values: torch.Tensor,
    indices: torch.Tensor,
    cache: ActivationDataset,
    encoder: SAEFeatureEncoder,
    tokenizer: Any,
    layer: int,
) -> dict[str, Any]:
    samples = cache.samples()
    hidden = cache.layer(layer)
    occurrences: list[dict[str, Any]] = []
    for score, index in zip(values.tolist(), indices.tolist(), strict=True):
        if score <= 0 or index < 0:
            continue
        codes = encoder.encode(hidden[index : index + 1].to("cuda", dtype=torch.float32))[0]
        rank = int((codes > codes[feature]).sum()) + 1
        token = int(samples["token_id"][index])
        position = int(samples["position"][index])
        if not 1 <= position < 1024:
            raise ValueError(
                "expected content positions after the framing token within context 1024"
            )
        occurrences.append(
            {
                "source_index": int(samples["document_index"][index]),
                "position": position,
                "token_id": token,
                "text": tokenizer.decode([token]),
                "context": "",
                "score": float(codes[feature]),
                "energy": float(codes[feature].square() / codes.square().sum()),
                "absolute_score_rank": rank,
                "activation_row": index,
            }
        )
    return {"component": feature, "tail_direction": "positive", "occurrences": occurrences}


def validate_result(
    value: dict[str, Any],
    expected_count: int,
    thresholds: Sequence[int] = (1, 3, 5, 10, 15),
) -> None:
    if value.get("n_occurrences") != expected_count:
        raise ValueError("incomplete feature checkpoint")
    if expected_count == 0:
        if value.get("status") != "no_positive_examples":
            raise ValueError("missing inactive-feature status")
        return
    if len(value.get("occurrences", [])) != expected_count:
        raise ValueError("missing occurrence results")
    if value.get("rank_thresholds") != list(thresholds):
        raise ValueError("incorrect result thresholds")
    for threshold in thresholds:
        estimates = []
        recovered = 0
        for occurrence in value["occurrences"]:
            recovery = occurrence["thresholds"][str(threshold)]
            erf = recovery["erf_estimate"]
            if not math.isfinite(erf) or not 1 <= erf <= occurrence["content_tokens"]:
                raise ValueError("invalid ERF estimate")
            estimates.append(erf)
            recovered += recovery["recovered"]
        summary = value["threshold_results"][str(threshold)]
        if not np.isclose(np.mean(estimates), summary["suffix_erf_mean"]):
            raise ValueError("invalid component aggregation")
        if recovered != summary["n_recovered"]:
            raise ValueError("invalid recovery count")


FORMAT = "icalens.erf_suffix_sweep.sae"
SCHEMA_VERSION = 2


def _prepared_path(output: Path, layer: int) -> Path:
    return output / "prepared" / f"layer_{layer:02d}.json"


def _result_path(output: Path, layer: int) -> Path:
    return output / "results" / f"layer_{layer:02d}.json"


def _parse_layers(value: str, available: Sequence[int]) -> list[int]:
    if value == "all":
        return list(available)
    layers = sorted({int(item) for item in value.split(",") if item.strip()})
    if not layers or set(layers).difference(available):
        raise ValueError(f"invalid layers {layers}; available: {list(available)}")
    return layers


def _identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_layer(path: Path, identity: dict[str, Any], kind: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"invalid {kind} artifact: {path}")
    if value.get("identity_sha256") != _identity_sha256(identity) or value.get("kind") != kind:
        raise ValueError(f"incompatible {kind} artifact: {path}")
    return value


def _validate_result_bundle(bundle: dict[str, Any], identity: dict[str, Any], layer: int) -> None:
    results = bundle.get("results")
    features = bundle.get("features")
    if (
        not isinstance(results, dict)
        or not isinstance(features, list)
        or set(results) != set(features)
    ):
        raise ValueError(f"incomplete result bundle for layer {layer}")
    if len(features) != identity["features_per_layer"]:
        raise ValueError(f"incorrect feature count for layer {layer}")
    for feature_text, result in results.items():
        count = result.get("n_occurrences")
        if not isinstance(count, int) or not 1 <= count <= identity["occurrences_per_feature"]:
            raise ValueError(f"invalid occurrence count for layer {layer}, F{feature_text}")
        validate_result(result, count, tuple(identity["rank_thresholds"]))
        if result["component"] != int(feature_text) or result["layer"] != layer:
            raise ValueError(f"incorrect result identity for layer {layer}, F{feature_text}")


def _write_summaries(output: Path, layers: Sequence[int], identity: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        bundle = _load_layer(_result_path(output, layer), identity, "result")
        if bundle is None:
            raise ValueError(f"missing result bundle for layer {layer}")
        _validate_result_bundle(bundle, identity, layer)
        for feature_text, result in bundle["results"].items():
            for threshold_text, summary in result["threshold_results"].items():
                rows.append(
                    {
                        "model": identity["label"],
                        "layer": layer,
                        "component": int(feature_text),
                        "top_k": int(threshold_text),
                        **summary,
                    }
                )
    atomic_write_json(output / "summary.json", {"format": FORMAT, "rows": rows})
    path = output / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--occurrences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rank-thresholds", type=erf_suffix_sweep._parse_rank_thresholds, default=(1, 3, 5, 10, 15)
    )
    parser.add_argument("--exact-suffix-length", type=int, default=10)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--batch-token-budget", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.features,
            args.occurrences,
            args.exact_suffix_length,
            args.max_batch_size,
            args.batch_token_budget,
        )
        < 1
    ):
        parser.error("feature, occurrence, suffix, and batch values must be positive")
    cache = ActivationDataset(args.cache)
    layers = _parse_layers(args.layers, cache.available_layers)
    provenance = cache.provenance
    if (
        cache.manifest["activation_site"] != "resid_post"
        or cache.sample_count != 1_000_000
        or provenance["context_length"] != 1024
        or provenance["dataset"]["repo_id"] != "NeelNanda/pile-10k"
        or provenance["document_framing"]["strategy"] not in {"prepend-bos", "prepend-eos"}
    ):
        raise ValueError("requires the official 1M-token, 1024-context Pile-10k resid_post cache")
    registry = _resolve_baselines(cache.model["repo_id"], "sae")
    baselines = {layer: _prepare_layer_baselines(registry, layer=layer) for layer in layers}
    identity = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "icalens_version": __version__,
        "method": erf_suffix_sweep.METHOD,
        "label": args.label,
        "model": cache.model,
        "activation_site": "resid_post",
        "layers": layers,
        "activation_manifest_sha256": digest(cache.path / "activations.json"),
        "provenance": provenance,
        "sae": {str(k): v["sae"] for k, v in baselines.items()},
        "features_per_layer": args.features,
        "occurrences_per_feature": args.occurrences,
        "rank_thresholds": list(args.rank_thresholds),
        "seed": args.seed,
        "selection": "first active features in a model/layer-seeded random permutation",
        "tail_direction": "positive",
        "score": "encoder activation times decoder-row norm",
        "recovery": "positive score and full-dictionary competition rank <= threshold",
        "exact_suffix_length": args.exact_suffix_length,
        "max_batch_size": args.max_batch_size,
        "batch_token_budget": args.batch_token_budget,
    }
    if args.dry_run:
        print(json.dumps(identity, indent=2))
        return
    output = args.output.expanduser().resolve()
    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=identity, source=source, status="measuring")
    completed = set()
    for layer in layers:
        bundle = _load_layer(_result_path(output, layer), identity, "result")
        if bundle is not None:
            _validate_result_bundle(bundle, identity, layer)
            completed.add(layer)
    try:
        with ExperimentDisplay(
            output=output / "logs",
            title=f"SAE suffix ERF · {args.label}",
            total=len(layers),
            completed=len(completed),
            completed_unit_ids=completed,
            source_dirty=source.get("dirty"),
            unit_label="layers",
            recent_label="Recent ERF output",
        ) as display:
            warn_if_dirty(source)
            if len(completed) == len(layers):
                log(f"{args.label}: all layers complete; skipping model load")
            else:
                model_identity = type(
                    "ModelIdentity",
                    (),
                    {"model_id": cache.model["repo_id"], "model_revision": cache.model["revision"]},
                )()
                display.phase("Loading model", model=args.label)
                model = erf_gradient._load_model(model_identity, device="cuda")
                tokenizer = AutoTokenizer.from_pretrained(
                    cache.model["repo_id"],
                    revision=cache.model["revision"],
                    use_fast=True,
                    trust_remote_code=True,
                )
                datasets: dict[str, Any] = {}
                token_cache: dict[tuple[str, int, int], list[int]] = {}
                try:
                    for layer in layers:
                        if layer in completed:
                            continue
                        display.phase("Loading SAE", model=args.label, layer=layer)
                        encoder = SAEFeatureEncoder(
                            {
                                "hidden_size": cache.hidden_size,
                                "layer": layer,
                                "saebench_model_name": cache.model["repo_id"],
                                "baselines": baselines[layer],
                            },
                            device="cuda",
                            dtype=torch.float32,
                        ).eval()
                        try:
                            prepared_bundle = _load_layer(
                                _prepared_path(output, layer), identity, "prepared"
                            )
                            if prepared_bundle is None:
                                width = int(baselines[layer]["sae"]["width"])
                                permutation = np.random.default_rng(
                                    _stable_seed(args.seed, args.label, layer)
                                ).permutation(width)
                                candidate_count = min(width, max(args.features * 2, args.features))
                                while True:
                                    candidates = permutation[:candidate_count].tolist()
                                    values, indices = profile(
                                        cache, encoder, candidates, args.occurrences, display, layer
                                    )
                                    active_rows = [
                                        i for i in range(candidate_count) if float(values[i, 0]) > 0
                                    ]
                                    if len(active_rows) >= args.features:
                                        chosen_rows = active_rows[: args.features]
                                        break
                                    if candidate_count == width:
                                        raise ValueError(
                                            f"layer {layer} has fewer than "
                                            f"{args.features} active SAE features"
                                        )
                                    candidate_count = min(width, candidate_count * 2)
                                chosen = sorted((candidates[row], row) for row in chosen_rows)
                                prepared = {
                                    str(feature): prepare_feature(
                                        feature,
                                        values[row],
                                        indices[row],
                                        cache,
                                        encoder,
                                        tokenizer,
                                        layer,
                                    )
                                    for feature, row in chosen
                                }
                                prepared_bundle = {
                                    "kind": "prepared",
                                    "identity_sha256": _identity_sha256(identity),
                                    "layer": layer,
                                    "candidate_features_profiled": candidate_count,
                                    "features": prepared,
                                }
                                atomic_write_json(_prepared_path(output, layer), prepared_bundle)
                            prepared = prepared_bundle["features"]
                            adapter = SAEReadout(cache, encoder, layer)
                            results: dict[str, dict[str, Any]] = {}

                            def checkpoint(
                                feature: int,
                                result: dict[str, Any],
                                *,
                                _results: dict[str, dict[str, Any]] = results,
                            ) -> None:
                                _results[str(feature)] = result

                            display.phase("Sweeping suffixes", model=args.label, layer=layer)
                            erf_suffix_sweep._measure_layer(
                                lens=adapter,
                                model=model,
                                tokenizer=tokenizer,
                                datasets=datasets,
                                token_cache=token_cache,
                                layer=layer,
                                prepared_components={int(k): v for k, v in prepared.items()},
                                exact_suffix_length=args.exact_suffix_length,
                                rank_thresholds=args.rank_thresholds,
                                max_batch_size=args.max_batch_size,
                                batch_token_budget=args.batch_token_budget,
                                device="cuda",
                                checkpoint=checkpoint,
                            )
                            if set(results) != set(prepared):
                                raise ValueError(f"layer {layer} produced incomplete results")
                            bundle = {
                                "kind": "result",
                                "identity_sha256": _identity_sha256(identity),
                                "layer": layer,
                                "features": list(prepared),
                                "results": results,
                            }
                            _validate_result_bundle(bundle, identity, layer)
                            atomic_write_json(_result_path(output, layer), bundle)
                            display.complete_unit(layer, refresh=True)
                            print(
                                f"ERF_LAYER_COMPLETE {args.label} {layer} ({len(results)} features)"
                            )
                        finally:
                            del encoder
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                finally:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            display.phase("Summarizing", model=args.label)
            _write_summaries(output, layers, identity)
            run.set_status("complete", complete=True)
            log(f"Complete: {args.label}, {len(layers)}/{len(layers)} layers")
    except BaseException:
        run.set_status("interrupted")
        raise


if __name__ == "__main__":
    main()
