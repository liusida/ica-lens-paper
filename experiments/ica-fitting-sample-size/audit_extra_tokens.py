#!/usr/bin/env python3
"""Audit all evaluation states and independently recompute final metrics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from safetensors.torch import load_file

from icalens._activation_dataset import ActivationDataset
from icalens._fastica import GAUSSIAN_OBJECTIVES
from run import BUDGETS

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"


def main():
    dataset = ActivationDataset(WORK / "activations/gpt2-pile10k-extra-1m")
    values = dataset.layer(11)
    audit_path = WORK / "extra-evaluation-selection.json"
    audit = json.loads(audit_path.read_text())
    training_rows = {v["source_row"] for v in audit["training_documents"]}
    evaluation_rows = {v["source_row"] for v in audit["assessment_documents"]}
    training_hashes = {v["text_sha256"] for v in audit["training_documents"]}
    evaluation_hashes = {v["text_sha256"] for v in audit["assessment_documents"]}
    assert not training_rows.intersection(evaluation_rows)
    assert not training_hashes.intersection(evaluation_hashes)
    assert audit["replayed_training_token_ids_match"]
    assert sum(v["tokens"] for v in audit["assessment_documents"]) == dataset.sample_count == 1_000_000
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == dataset.provenance["evaluation_selection"]["audit_sha256"]
    report = {"document_overlap": 0, "exact_text_overlap": 0,
              "training_documents": len(training_rows), "evaluation_documents": len(evaluation_rows),
              "evaluation_tokens": dataset.sample_count, "settings": {}}
    selected = np.array([0, 100, 200, 300, 400, 500, 600, 767])
    for label, count in BUDGETS.items():
        source = WORK / "runs" / label
        output = WORK / "extra-evaluation" / label
        configuration = json.loads((output / "configuration.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        assert complete["evaluation_tokens"] == 1_000_000 and complete["states"] == 51
        assert configuration["evaluation_manifest_sha256"] == dataset.provenance["activation_dataset"]["manifest_sha256"]
        for name, expected in configuration["checkpoint_sha256"].items():
            assert hashlib.sha256((source / "checkpoints" / name).read_bytes()).hexdigest() == expected
        with np.load(output / "trajectory.npz") as archive:
            record = {key: archive[key].copy() for key in archive.files}
        assert np.array_equal(record["step"], np.arange(51))
        for key, value in record.items():
            assert np.isfinite(value).all(), (label, key)
            if key.startswith("component_"):
                assert value.shape == (51, 768)
        shared = load_file(source / "preprocessing.safetensors", device="cuda")
        w = load_file(source / "checkpoints/step-050.safetensors", device="cuda")["unmixing"]
        reading = w[selected] @ shared["whitening"]
        projected = []
        for start in range(0, len(values), 16_384):
            x = values[start:start+16_384].cuda().float()
            projected.append(((x - shared["center"]) @ reading.T).cpu().numpy())
        z = np.concatenate(projected).astype(np.float64)
        independent_kurtosis = kurtosis(z, axis=0, fisher=True, bias=True)
        independent_skewness = skew(z, axis=0, bias=True)
        standardized = (z - z.mean(0)) / z.std(0)
        contrast = np.mean(np.logaddexp(standardized, -standardized) - math.log(2), axis=0)
        deviation = np.abs(contrast - GAUSSIAN_OBJECTIVES["logcosh"])
        assert np.allclose(independent_kurtosis, record["component_excess_kurtosis"][-1, selected], rtol=1e-4, atol=1e-3)
        assert np.allclose(independent_skewness, record["component_skewness"][-1, selected], rtol=1e-4, atol=1e-4)
        assert np.allclose(deviation, record["component_logcosh_deviation"][-1, selected], rtol=1e-4, atol=1e-5)
        report["settings"][label] = {
            "fitting_tokens": count, "evaluation_tokens": len(values), "states": 51,
            "median_final_excess_kurtosis": float(np.median(record["component_excess_kurtosis"][-1])),
            "median_final_absolute_skewness": float(np.median(np.abs(record["component_skewness"][-1]))),
            "median_final_logcosh_deviation": float(np.median(record["component_logcosh_deviation"][-1])),
            "median_final_raw_score_variance": float(np.median(record["component_variance"][-1])),
            "independent_scipy_max_abs_kurtosis_error": float(np.max(np.abs(independent_kurtosis-record["component_excess_kurtosis"][-1, selected]))),
            "independent_component_indices": selected.tolist()}
        print(label, report["settings"][label], flush=True)
    report["all_checks_passed"] = True
    (WORK / "extra-evaluation/audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
