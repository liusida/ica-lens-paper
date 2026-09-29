"""Numerically replay a deterministic slice of C10 activation capture."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens._activation_dataset import ActivationDataset, sample_metadata
from icalens._capture import capture_resid_post
from icalens.cli.fit_text import load_pile_documents, sample_positions


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sampled_rows(size: int, count: int, seed: int) -> torch.Tensor:
    if count < 1 or count > size:
        raise ValueError(f"verification rows must be in 1..{size}, got {count}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[:count].sort().values


def run(
    *,
    lens_root: Path,
    activation_root: Path,
    layer: int,
    verification_rows: int,
    verification_seed: int,
    output: Path,
    canary_id: str = "gpt2",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("C10 capture replay requires CUDA")
    lens = load_json(lens_root / "icalens.json")
    dataset = ActivationDataset(activation_root)
    provenance = dataset.manifest["provenance"]
    source = provenance["dataset"]
    model = lens["model"]
    rows = sampled_rows(dataset.sample_count, verification_rows, verification_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model["repo_id"], revision=model["revision"], use_fast=True
    )
    documents = load_pile_documents(
        tokenizer,
        dataset_id=source["repo_id"],
        dataset_revision=source["revision"],
        split=source["split"],
        text_field=provenance["text_field"],
        candidate_token_budget=int(provenance["candidate_tokens"]),
        context_length=int(provenance["context_length"]),
        document_framing=provenance["document_framing"],
    )
    selected = sample_positions(
        documents,
        token_budget=int(provenance["fitting_tokens"]),
        seed=int(provenance["sampling_seed"]),
    )
    regenerated_metadata = sample_metadata(documents, selected)
    reference_metadata = dataset.samples()
    metadata_fields = sorted(reference_metadata)
    metadata_equal_by_field = {
        field: torch.equal(
            reference_metadata[field].index_select(0, rows),
            regenerated_metadata[field].index_select(0, rows),
        )
        for field in metadata_fields
    }

    by_document: dict[int, list[tuple[int, int]]] = defaultdict(list)
    sampled_documents = reference_metadata["document_index"].index_select(0, rows)
    sampled_positions = reference_metadata["position"].index_select(0, rows)
    for output_index, (document, position) in enumerate(
        zip(sampled_documents.tolist(), sampled_positions.tolist(), strict=True)
    ):
        by_document[int(document)].append((output_index, int(position)))

    loaded_model = load_model_to_cuda(
        AutoModelForCausalLM,
        model["repo_id"],
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        revision=model["revision"],
    )
    loaded_model.eval()
    actual = torch.empty((len(rows), dataset.hidden_size), dtype=torch.bfloat16)
    try:
        for document_index, targets in sorted(by_document.items()):
            positions = torch.tensor([position for _, position in targets], device="cuda")
            captured = capture_resid_post(
                loaded_model,
                model_inputs={"input_ids": documents[document_index].input_ids[None].to("cuda")},
                layers=(layer,),
                positions=positions,
            )[layer].cpu()
            for local_index, (output_index, _) in enumerate(targets):
                actual[output_index] = captured[local_index]
    finally:
        del loaded_model
        torch.cuda.empty_cache()

    expected = dataset.layer(layer).index_select(0, rows)
    exact = torch.equal(expected, actual)
    difference = (expected.float() - actual.float()).abs()
    metadata_exact = all(metadata_equal_by_field.values())
    status = "pass" if metadata_exact and exact else "fail"

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "fragment.npz",
        row_indices=rows.numpy(),
        document_indices=sampled_documents.numpy(),
        positions=sampled_positions.numpy(),
        expected=expected.float().numpy(),
        actual=actual.float().numpy(),
    )
    return {
        "check": f"C10-capture-{canary_id}-layer{layer}",
        "status": status,
        "scope": {
            "code_ids": ["C10"],
            "input_data_ids": ["D01", "D02", "D03"],
            "output_data_ids": ["D10"],
            "canary": canary_id,
            "layer": layer,
            "verification_rows": len(rows),
            "verification_seed": verification_seed,
            "sampled_documents": len(by_document),
        },
        "comparison": {
            "metadata_exact": metadata_exact,
            "metadata_exact_by_field": metadata_equal_by_field,
            "activation_values_exact": exact,
            "maximum_absolute_error": float(difference.max()),
            "differing_activation_values": int(torch.count_nonzero(difference)),
            "compared_activation_values": int(expected.numel()),
        },
        "artifacts": {"comparison_fragment": str(output / "fragment.npz")},
        "failed_checks": [
            name
            for name, passed in (
                ("sample_metadata", metadata_exact),
                ("activation_values", exact),
            )
            if not passed
        ],
    }
