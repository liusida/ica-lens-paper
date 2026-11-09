#!/usr/bin/env python3
"""Capture a document-disjoint additional 1M-token assessment set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from icalens._activation_dataset import ActivationDataset
from icalens.cli.capture import main as capture_main

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUTPUT = WORK / "activations/gpt2-pile10k-extra-1m"


def main():
    training = ActivationDataset(WORK / "activations/gpt2-pile10k-layer11-1m")
    provenance = training.provenance
    source = provenance["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(training.model["repo_id"], revision=training.model["revision"])
    content_limit = provenance["context_length"] - 1
    assert provenance["document_framing"]["strategy"] == "prepend-eos"
    records = load_dataset(source["repo_id"], revision=source["revision"], split=source["split"], streaming=True)
    train_count, eval_count = 0, 0
    train_hashes, eval_hashes = set(), set()
    train_refs, eval_refs, train_ids = [], [], []
    raw_path = WORK / "extra-evaluation-documents.jsonl"
    with raw_path.open("w") as handle:
        for row_index, row in enumerate(records):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            fitting = train_count < training.sample_count
            if not fitting and (digest in train_hashes or digest in eval_hashes):
                continue
            remaining = training.sample_count - train_count if fitting else 1_000_000 - eval_count
            ids = tokenizer(text, add_special_tokens=False, truncation=True,
                            max_length=min(content_limit, remaining))["input_ids"]
            if not ids:
                continue
            reference = {"source_row": row_index, "text_sha256": digest, "tokens": len(ids)}
            if fitting:
                train_hashes.add(digest)
                train_refs.append(reference)
                train_ids.extend(ids)
                train_count += len(ids)
            else:
                eval_hashes.add(digest)
                eval_refs.append(reference)
                handle.write(json.dumps({"text": text, "source_row": row_index}, ensure_ascii=False) + "\n")
                eval_count += len(ids)
                if eval_count == 1_000_000:
                    break
    if eval_count != 1_000_000:
        raise RuntimeError(f"only {eval_count} additional tokens available")
    assert torch.equal(torch.tensor(train_ids), training.samples()["token_id"]), "failed to replay original fitting-token identities"
    assert not train_hashes.intersection(eval_hashes)
    assert not {r["source_row"] for r in train_refs}.intersection(r["source_row"] for r in eval_refs)
    audit = {"assessment_type": "held-out documents from the same Pile distribution; not cross-domain OOD",
             "dataset": source, "model": training.model,
             "training_manifest_sha256": provenance["activation_dataset"]["manifest_sha256"],
             "replayed_training_token_ids_match": True,
             "training_tokens": train_count, "assessment_tokens": eval_count,
             "training_documents": train_refs, "assessment_documents": eval_refs,
             "overlapping_source_documents": 0, "overlapping_exact_document_hashes": 0,
             "selection": "continue after original training documents, excluding exact-text duplicates",
             "raw_documents_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    audit_path = WORK / "extra-evaluation-selection.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Replayed {train_count:,} original tokens from {len(train_refs)} documents; selected {eval_count:,} new tokens from {len(eval_refs)} disjoint documents", flush=True)
    capture_main("text", ["--model", training.model["repo_id"], "--model-revision", training.model["revision"],
        "--dataset", str(raw_path), "--layers", "11", "--candidate-tokens", "1000000",
        "--token-budget", "1000000", "--context-length", str(provenance["context_length"]),
        "--capture-layers-at-once", "1", "--seed", "0", "--output", str(OUTPUT)])
    manifest_path = OUTPUT / "activations.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["evaluation_selection"] = {
        "audit_path": str(audit_path), "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "training_manifest_sha256": audit["training_manifest_sha256"], "source_dataset": source,
        "assessment_type": audit["assessment_type"], "overlapping_source_documents": 0,
        "overlapping_exact_document_hashes": 0}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
