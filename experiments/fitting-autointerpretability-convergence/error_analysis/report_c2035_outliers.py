#!/usr/bin/env python3
"""Recover source contexts for sampled L7/C2035 scores above a z threshold."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from icalens._activation_dataset import ActivationDataset

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ACTIVATIONS = Path(
    "/media/liusida/Expansion/research/ICA-data/icalens-activations/"
    "qwen3.5-9b-base-pile10k-1m"
)
DATA = RESULTS / "c2035-distribution-data.npz"
MODEL = "Qwen/Qwen3.5-9B-Base"
MODEL_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
DATASET = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
CONTEXT_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--context-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = np.load(DATA)
    sampled_rows = archive["sample_indices"]
    z = archive["standardized_scores"]
    selected = np.flatnonzero(z > args.threshold)

    capture = ActivationDataset(ACTIVATIONS)
    metadata = capture.samples()
    document_indices = metadata["document_index"][sampled_rows[selected]].tolist()
    needed = set(map(int, document_indices))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    source = load_dataset(
        DATASET, split="train", revision=DATASET_REVISION, streaming=True
    )
    documents: dict[int, list[int]] = {}
    valid_index = 0
    for row in source:
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=CONTEXT_LENGTH - 1,
        )["input_ids"]
        if not ids:
            continue
        if valid_index in needed:
            documents[valid_index] = list(map(int, ids))
            if len(documents) == len(needed):
                break
        valid_index += 1

    records = []
    for sample_position in selected:
        activation_row = int(sampled_rows[sample_position])
        doc = int(metadata["document_index"][activation_row])
        captured_position = int(metadata["position"][activation_row])
        expected_token_id = int(metadata["token_id"][activation_row])
        content_position = captured_position - 1  # Capture prepends Qwen's EOS separator.
        ids = documents[doc]
        if ids[content_position] != expected_token_id:
            raise RuntimeError(
                f"token mismatch at document {doc}, position {captured_position}: "
                f"capture={expected_token_id}, replay={ids[content_position]}"
            )
        start = max(0, content_position - args.context_tokens)
        stop = min(len(ids), content_position + args.context_tokens + 1)
        before = tokenizer.decode(ids[start:content_position])
        target = tokenizer.decode([ids[content_position]])
        after = tokenizer.decode(ids[content_position + 1 : stop])
        records.append({
            "standardized_signed_score": float(z[sample_position]),
            "sample_position": int(sample_position),
            "activation_row": activation_row,
            "document_index": doc,
            "captured_position": captured_position,
            "token_id": expected_token_id,
            "target": target,
            "context_before": before,
            "context_after": after,
            "context": before + target + after,
        })
    records.sort(key=lambda item: item["standardized_signed_score"], reverse=True)

    payload = {
        "model": MODEL,
        "layer": 7,
        "component": 2035,
        "iteration": 50,
        "threshold": args.threshold,
        "comparison": "standardized_signed_score > threshold",
        "count": len(records),
        "records": records,
    }
    json_path = RESULTS / "c2035-outliers-z-gt-20.json"
    html_path = RESULTS / "c2035-outliers-z-gt-20.html"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    rows = []
    for index, record in enumerate(records, 1):
        context = (
            html.escape(record["context_before"])
            + "<mark>"
            + html.escape(record["target"])
            + "</mark>"
            + html.escape(record["context_after"])
        )
        rows.append(
            f"<tr><td>{index}</td><td>{record['standardized_signed_score']:.2f}</td>"
            f"<td>{record['document_index']}</td><td>{record['captured_position']}</td>"
            f"<td><code>{html.escape(record['target'])}</code></td>"
            f"<td class='context'>{context}</td></tr>"
        )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>C2035 outliers</title>"
        "<style>body{font:15px system-ui;margin:32px;max-width:1200px}"
        "table{border-collapse:collapse}th,td{padding:8px 10px;border-bottom:1px solid #ddd;"
        "vertical-align:top}th{text-align:left}.context{line-height:1.5}"
        "mark{background:#efc7bc;padding:1px 2px}</style>"
        f"<h1>L7/C2035: standardized signed score &gt; {args.threshold:g}</h1>"
        f"<p>{len(records)} occurrences among the 100,000 plotted samples.</p>"
        "<table><thead><tr><th>#</th><th>z</th><th>Document</th><th>Position</th>"
        "<th>Target</th><th>Context</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>",
        encoding="utf-8",
    )

    for index, record in enumerate(records, 1):
        context = (
            record["context_before"]
            + "⟦"
            + record["target"]
            + "⟧"
            + record["context_after"]
        ).replace("\n", "↵")
        print(f"{index}. z={record['standardized_signed_score']:.2f}  {context}")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
