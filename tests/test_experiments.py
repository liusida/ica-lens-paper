from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from icalens.experiments._sae import SAEFeatureEncoder, _load_sae_tensors
from icalens.experiments._saebench_environment import resolve_backend
from icalens.experiments._saebench_multilayer_worker import (
    _capture_layers,
    _hook_aliases,
)
from icalens.experiments._saebench_worker import (
    ICAFeatureEncoder,
    RandomFeatureEncoder,
    _merge_dataset_results,
    _prepare_model_inputs,
    _remove_dataset_artifacts,
)
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.figure import parse_args as parse_figure_args
from icalens.experiments.figure import (
    render_sparse_probing_figure,
    render_sparse_probing_panels,
)
from icalens.experiments.reconstruction import (
    CAPTURE_SUITE_MANIFEST,
    _activation_path,
    _capture_suite_identity,
    _completed_measurement_units,
    _configuration_differences,
    _dataset_cache_dir,
    _dataset_value_to_text,
    _linear_dictionary_metrics,
    _load_preset,
    _load_run,
    _merge_method_results,
    _method_result_path,
    _parse_capture_layers_at_once,
    _remove_activation_cache,
    _save_activation_cache,
    _validate_or_create_capture_suite,
    parse_capture_args,
    parse_measure_args,
)
from icalens.experiments.reconstruction import (
    parse_args as parse_reconstruction_args,
)
from icalens.experiments.reconstruction_figure import (
    _dataset_title,
    _legend_order,
    _linear_endpoint_segment,
    _mean_curve,
    _same_plot_point,
)
from icalens.experiments.reconstruction_figure import (
    render as render_reconstruction,
)
from icalens.experiments.saebench_sparse_probing import (
    _completed_evaluations_at_start,
    _completed_methods,
    _estimate_activation_cache_bytes,
    _load_or_initialize_run,
    _parse_k_values,
    _parse_layers,
    _resolve_baselines,
    collect_result_rows,
)


def test_k_values_override_parser() -> None:
    assert _parse_k_values("500,200,500") == [200, 500]
    with pytest.raises(ValueError, match="positive integers"):
        _parse_k_values("0,200")


def test_ica_saebench_encoder_compensates_unit_decoder_norms(tmp_path: Path) -> None:
    center = torch.tensor([0.25, -0.5])
    reading = torch.tensor([[2.0, 0.0], [0.0, 0.25]])
    writing = torch.linalg.pinv(reading)
    layer_file = tmp_path / "layer.safetensors"
    save_file(
        {
            "center": center,
            "reading_matrix": reading,
            "writing_matrix": writing,
            "tail_signs": torch.tensor([-1.0, 1.0]),
        },
        layer_file,
    )
    snapshot = {
        "layer_file": str(layer_file),
        "saebench_model_name": "test-model",
        "layer": 0,
        "row_normalize": False,
        "norm_eps": 1e-12,
    }
    encoder = ICAFeatureEncoder(snapshot, device="cpu", dtype=torch.float32)
    activations = torch.tensor([[[1.25, -2.5], [-0.75, 3.5]]])
    features = encoder.encode(activations)
    reconstructed = features @ encoder.W_dec + center

    assert encoder.cfg.d_sae == 2
    assert encoder.cfg.activation_fn_str == "identity"
    assert torch.allclose(encoder.W_dec.norm(dim=-1), torch.ones(2))
    assert torch.allclose(reconstructed, activations, atol=1e-6)


def test_source_provenance_records_checkout_revision() -> None:
    provenance = source_provenance()
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert provenance["git_available"] is True
    assert provenance["commit"] == expected
    assert isinstance(provenance["dirty"], bool)


def test_dirty_source_warning_is_non_blocking(capsys: pytest.CaptureFixture[str]) -> None:
    warn_if_dirty({"git_available": True, "commit": "abc", "dirty": True})
    assert "WARNING" in capsys.readouterr().out
    warn_if_dirty({"git_available": True, "commit": "abc", "dirty": False})
    assert capsys.readouterr().out == ""


def test_resume_eta_initial_count_includes_dataset_checkpoints(tmp_path: Path) -> None:
    partial = tmp_path / "layers" / "layer_10" / "saebench-datasets" / "00_owner__dataset-a"
    partial.mkdir(parents=True)
    (partial / "ica_custom_sae_eval_results.json").write_text(
        json.dumps({"eval_result_metrics": {}}), encoding="utf-8"
    )

    completed = _completed_evaluations_at_start(
        tmp_path,
        layers=[6, 10],
        methods=["ica", "sae"],
        datasets=["owner/dataset-a", "owner/dataset-b"],
        completed_by_layer={6: {"ica", "sae"}, 10: set()},
    )

    assert completed == 5


def test_backend_registry_resolves_exact_gpt2_revision() -> None:
    backend = resolve_backend("openai-community/gpt2")
    assert backend.name == "paper-standard"
    assert backend.commit == "2a7480bf67f9f82f79e742be55fc1f5ec404bfa9"
    assert backend.saebench_model_name == "gpt2-small"


def test_figure_output_defaults_to_experiment_directory() -> None:
    args = parse_figure_args(["results/gpt2-paper"])
    assert args.output is None


def test_reconstruction_paper_uses_script_free_code_parquet() -> None:
    preset = _load_preset("paper")
    assert preset["context_length"] == 1024
    code = preset["datasets"][2]
    assert code["repo_id"] == "codeparrot/github-code"
    assert code["loader"] == "parquet"
    assert code["data_files"].endswith(".parquet")
    assert code["text_column"] == "content"


def test_reconstruction_paper_spans_six_domains() -> None:
    datasets = _load_preset("paper")["datasets"]
    assert [item["repo_id"] for item in datasets] == [
        "fancyzhx/ag_news",
        "Salesforce/wikitext",
        "codeparrot/github-code",
        "wikimedia/wikipedia",
        "wikimedia/wikipedia",
        "HuggingFaceH4/ultrachat_200k",
    ]
    assert datasets[3]["config"] == "20231101.es"
    assert datasets[4]["config"] == "20231101.zh"
    assert datasets[5]["format"] == "conversation"


def test_reconstruction_figure_distinguishes_wikipedia_languages() -> None:
    datasets = _load_preset("paper")["datasets"]
    assert _dataset_title(datasets[3]) == "Spanish Wikipedia"
    assert _dataset_title(datasets[4]) == "Chinese Wikipedia"
    assert _dataset_title(datasets[5]) == "HuggingFaceH4/ultrachat_200k"


def test_reconstruction_conversation_serialization_is_model_independent() -> None:
    rendered = _dataset_value_to_text(
        [
            {"role": "user", "content": " Hello "},
            {"role": "assistant", "content": "Hi!"},
        ],
        {"format": "conversation"},
    )
    assert rendered == "User: Hello\nAssistant: Hi!"


def test_reconstruction_context_length_override_parser() -> None:
    args = parse_reconstruction_args(
        [
            "--lens",
            "owner/lens",
            "--layers",
            "6",
            "--output",
            "results/run",
            "--context-length",
            "512",
        ]
    )
    assert args.context_length == 512


def test_reconstruction_endpoint_marker_overlap_rule() -> None:
    assert _same_plot_point((768.0, 0.999), (768.0, 1.0))
    assert not _same_plot_point((300.0, 0.98), (768.0, 1.0))
    assert not _same_plot_point((768.0, 0.98), (768.0, 1.0))


def test_reconstruction_connects_only_linear_full_endpoint() -> None:
    curve = [(100.0, 0.9), (300.0, 0.98)]
    full = (768.0, 1.0)
    assert _linear_endpoint_segment(curve, full, method="ica") == [curve[-1], full]
    assert _linear_endpoint_segment(curve, full, method="sae") is None


def test_reconstruction_legend_starts_with_ica_without_changing_draw_order() -> None:
    assert _legend_order(False) == ("ica", "sae", "pca", "random")
    assert _legend_order(True) == ("ica", "sae", "sae_context_64", "pca", "random")


def test_reconstruction_pile10k_preset_is_labeled_in_distribution() -> None:
    preset = _load_preset("pile10k")
    assert preset["datasets"] == [
        {"repo_id": "NeelNanda/pile-10k", "split": "train", "text_column": "text"}
    ]
    assert preset["evaluation_scope"] == "in_distribution_fitting_corpus_diagnostic"


def test_reconstruction_capture_layers_accepts_all() -> None:
    assert _parse_capture_layers_at_once("all", selected_layer_count=12) == 12
    assert _parse_capture_layers_at_once("2", selected_layer_count=12) == 2
    with pytest.raises(ValueError, match="positive integer or 'all'"):
        _parse_capture_layers_at_once("everything", selected_layer_count=12)


def test_reconstruction_capture_defaults_to_all() -> None:
    args = parse_reconstruction_args(
        ["--lens", "owner/lens", "--layers", "all", "--output", "results/run"]
    )
    assert args.capture_layers_at_once == "all"


def test_reconstruction_split_cli_paths_and_k_values() -> None:
    capture = parse_capture_args(
        [
            "--lens",
            "owner/lens",
            "--layers",
            "all",
            "--preset",
            "paper",
            "--output",
            "~/Expansion/reconstruction/gpt2",
        ]
    )
    assert capture.output == Path("~/Expansion/reconstruction/gpt2")
    measure = parse_measure_args(
        [
            "--lens",
            "owner/lens",
            "--activations",
            "~/Expansion/reconstruction/gpt2",
            "--k-values",
            "300,30,1,30",
            "--evaluation-context-length",
            "64",
            "--output",
            "results/gpt2",
        ]
    )
    assert measure.k_values == [1, 30, 300]
    assert measure.evaluation_context_length == 64


def test_reconstruction_capture_identity_ignores_only_runtime_state() -> None:
    value = {
        "format": "icalens.reconstruction-activations",
        "status": "complete",
        "icalens_source": {"commit": "old"},
        "model": {"repo_id": "model", "revision": "revision"},
    }
    assert _capture_suite_identity(value) == {
        "format": "icalens.reconstruction-activations",
        "model": {"repo_id": "model", "revision": "revision"},
    }
    assert CAPTURE_SUITE_MANIFEST == "reconstruction-activations.json"


def test_reconstruction_completed_capture_manifest_is_stable_on_resume(tmp_path: Path) -> None:
    path = tmp_path / CAPTURE_SUITE_MANIFEST
    complete = {
        "format": "icalens.reconstruction-activations",
        "format_version": 1,
        "status": "complete",
        "icalens_source": {"commit": "original"},
        "model": {"repo_id": "model", "revision": "revision"},
    }
    path.write_text(json.dumps(complete), encoding="utf-8")
    requested = {
        **complete,
        "status": "capturing",
        "icalens_source": {"commit": "new-checkout"},
    }
    stored = _validate_or_create_capture_suite(path, requested)
    assert stored == complete
    assert json.loads(path.read_text(encoding="utf-8")) == complete


def test_reconstruction_activation_checkpoint_round_trip_and_cleanup(tmp_path: Path) -> None:
    cache = _dataset_cache_dir(tmp_path, 2)
    values = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    positions = torch.tensor([0, 1, 2])

    _save_activation_cache(cache, {6: values}, positions)

    from safetensors.torch import load_file

    stored = load_file(_activation_path(cache, 6))
    assert torch.equal(stored["activations"], values)
    assert torch.equal(stored["positions"], positions)
    _remove_activation_cache(cache)
    assert not cache.exists()


def test_reconstruction_measurement_checkpoints_each_method(tmp_path: Path) -> None:
    for method, payload_methods in (
        ("ica", {"ica": {"curve": {}}}),
        ("sae", {"sae": {"curve": {}}, "sae_context_64": {"curve": {}}}),
    ):
        path = _method_result_path(tmp_path, 6, 2, method)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"n_tokens": 50, "methods": payload_methods}), encoding="utf-8")
    assert (
        _completed_measurement_units(tmp_path, layers=[6], dataset_count=3, methods=["ica", "sae"])
        == 2
    )
    merged = _merge_method_results(
        tmp_path,
        layer=6,
        dataset_index=2,
        methods=["ica", "sae"],
        dataset={"repo_id": "owner/data"},
    )
    assert merged["n_tokens"] == 50
    assert set(merged["methods"]) == {"ica", "sae", "sae_context_64"}


def test_failed_empty_reconstruction_run_rejects_corrected_preset(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "resolved": {"preset": "old"},
                "layer_runs": {"0": {"status": "failed"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preset: 'old' != 'corrected'"):
        _load_run(path, {"preset": "corrected"})


def test_reconstruction_configuration_difference_names_nested_fields() -> None:
    assert _configuration_differences(
        {"preset": {"context_length": 1024}, "layers": [0, 1]},
        {"preset": {"context_length": 512}, "layers": [1]},
    ) == [
        "layers: [0, 1] != [1]",
        "preset.context_length: 1024 != 512",
    ]


def test_reconstruction_rejects_orphan_checkpoints_without_run_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoint.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty but has no run.json"):
        _load_run(tmp_path / "run.json", {"preset": "paper"})


def test_qwen9b_backend_registry_resolves_standard_backend() -> None:
    backend = resolve_backend("Qwen/Qwen3.5-9B-Base")
    assert backend.name == "paper-standard"
    assert backend.saebench_model_name == "Qwen/Qwen3.5-9B-Base"


def test_layer_parser_validates_artifact_layers() -> None:
    assert _parse_layers("all", (0, 6, 11)) == [0, 6, 11]
    assert _parse_layers("6,11,6", (0, 6, 11)) == [6, 11]
    with pytest.raises(ValueError, match="does not contain"):
        _parse_layers("7", (0, 6, 11))


def test_paper_activation_cache_estimate_matches_saebench_sampling() -> None:
    settings = {
        "datasets": [
            "LabHC/bias_in_bios_class_set1",
            "LabHC/bias_in_bios_class_set2",
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "canrager/amazon_reviews_mcauley_1and5_sentiment",
            "codeparrot/github-code",
            "fancyzhx/ag_news",
            "Helsinki-NLP/europarl",
        ],
        "probe_train_size": 4000,
        "probe_test_size": 1000,
        "context_length": 128,
        "llm_dtype": "float32",
    }
    assert _estimate_activation_cache_bytes(settings, hidden_size=768) == (5 * 2500 * 128 * 768 * 4)


def test_dataset_result_merger_uses_saebench_unweighted_average() -> None:
    def payload(dataset: str, value: float) -> dict[str, object]:
        return {
            "eval_config": {"dataset_names": [dataset]},
            "eval_result_metrics": {
                "llm": {"llm_test_accuracy": value + 0.1},
                "sae": {"sae_top_1_test_accuracy": value},
            },
            "eval_result_details": [
                {"dataset_name": f"{dataset}_results", "sae_top_1_test_accuracy": value}
            ],
            "eval_result_unstructured": {f"{dataset}_results": {"class": value}},
        }

    merged = _merge_dataset_results(
        [payload("dataset-a", 0.6), payload("dataset-b", 0.8)],
        ["dataset-a", "dataset-b"],
    )
    assert merged["eval_config"]["dataset_names"] == ["dataset-a", "dataset-b"]
    assert merged["eval_result_metrics"]["sae"]["sae_top_1_test_accuracy"] == pytest.approx(0.7)
    assert len(merged["eval_result_details"]) == 2


def test_remove_dataset_artifacts_removes_complete_cache_tree(tmp_path: Path) -> None:
    dataset_cache = tmp_path / "dataset_00"
    nested = dataset_cache / "sparse_probing" / "model" / "hook"
    nested.mkdir(parents=True)
    (nested / "activations.pt").write_bytes(b"temporary activations")

    _remove_dataset_artifacts(dataset_cache)
    assert not dataset_cache.exists()
    # Repeated cleanup is safe, including after an interrupted partial cleanup.
    _remove_dataset_artifacts(dataset_cache)


def test_shared_capture_runs_model_once_per_batch_and_masks_special_tokens() -> None:
    class Block(torch.nn.Module):
        def __init__(self, amount: float) -> None:
            super().__init__()
            self.amount = amount

        def forward(self, values: torch.Tensor) -> tuple[torch.Tensor]:
            return (values + self.amount,)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block(1), Block(2), Block(3)])
            self.calls = 0

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool
        ) -> torch.Tensor:
            del attention_mask, use_cache
            self.calls += 1
            values = input_ids[..., None].to(torch.float32).expand(-1, -1, 2)
            for block in self.model.layers:
                values = block(values)[0]
            return values

    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 9, "bos_token_id": 8})()
    model = TinyModel()
    tokenized = {"class": {"input_ids": torch.tensor([[1, 9], [2, 3], [8, 4]], dtype=torch.long)}}

    captured = _capture_layers(
        tokenized,
        model,
        tokenizer,
        [0, 2],
        batch_size=2,
        input_protocol={"boundary_source": "tokenizer"},
    )

    assert model.calls == 2
    assert captured[0]["class"].shape == (3, 2, 2)
    assert torch.equal(captured[0]["class"][0, 1], torch.zeros(2))
    assert torch.equal(captured[2]["class"][2, 0], torch.zeros(2))
    assert captured[0]["class"][1, 0, 0].item() == 3.0
    assert captured[2]["class"][1, 0, 0].item() == 8.0


def test_saebench_model_inputs_prepend_visible_boundary_and_hide_padding() -> None:
    tokenizer = type("Tokenizer", (), {"pad_token_id": 9})()
    tokens = torch.tensor([[4, 5, 9], [6, 9, 9]])
    model_tokens, attention_mask, strip_prefix = _prepare_model_inputs(
        tokens,
        tokenizer,
        {"boundary_source": "adapter", "token_id": 9},
    )

    assert torch.equal(model_tokens, torch.tensor([[9, 4, 5, 9], [9, 6, 9, 9]]))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
    assert strip_prefix == 1


def test_saebench_model_inputs_do_not_duplicate_tokenizer_boundary() -> None:
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    tokens = torch.tensor([[0, 2, 4, 5]])
    model_tokens, attention_mask, strip_prefix = _prepare_model_inputs(
        tokens,
        tokenizer,
        {"boundary_source": "tokenizer", "token_id": 2},
    )

    assert torch.equal(model_tokens, tokens)
    assert torch.equal(attention_mask, torch.tensor([[0, 1, 1, 1]]))
    assert strip_prefix == 0


def test_shared_capture_uses_saebench_and_sae_hook_aliases() -> None:
    snapshot = {
        "layer": 12,
        "baselines": {"sae": {"hook_name_template": "model.layers.{layer}"}},
    }
    assert _hook_aliases(snapshot) == {
        "blocks.12.hook_resid_post",
        "model.layers.12",
    }


def test_gpt2_baseline_registry_resolves_sae_and_pca() -> None:
    baselines = _resolve_baselines("openai-community/gpt2", "sae,pca")
    assert baselines["sae"]["repo_id"] == "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs"
    assert baselines["sae"]["feature_source"] == "encoder"
    assert baselines["sae"]["activation"] == "topk"
    assert baselines["sae"]["top_k"] == 32
    assert baselines["sae"]["normalize_activations"] == "layer_norm"
    assert baselines["pca"]["feature_sides"] == "one_signed_coordinate"


def test_random_baseline_is_model_independent_and_seeded() -> None:
    baseline = _resolve_baselines("owner/unregistered-model", "random")
    assert baseline["random"]["components"] == "model_hidden_size"
    assert baseline["random"]["seed"] == 0


def test_unregistered_model_rejects_baselines() -> None:
    with pytest.raises(ValueError, match="no sparse-probing baselines"):
        _resolve_baselines("owner/unknown-model", "pca")


def test_gemma_and_qwen_baseline_formats() -> None:
    gemma = _resolve_baselines("google/gemma-2-2b", "sae,pca")
    assert gemma["sae"]["checkpoint_format"] == "npz"
    assert gemma["sae"]["activation"] == "jumprelu"
    # Gemma Scope's official loader feeds raw residual activations directly to
    # W_enc. Unlike several SAE Lens checkpoints, it does not subtract b_dec.
    assert gemma["sae"]["apply_b_dec_to_input"] is False
    assert gemma["sae"]["layer_checkpoints"]["12"].endswith("params.npz")

    qwen = _resolve_baselines("Qwen/Qwen3.5-9B-Base", "sae,pca")
    assert qwen["sae"]["checkpoint_format"] == "torch"
    assert qwen["sae"]["activation"] == "topk"
    assert qwen["sae"]["top_k"] == 50
    assert qwen["sae"]["width"] == 65536


def test_sae_tensor_loader_supports_npz_and_safe_torch(tmp_path: Path) -> None:
    npz_path = tmp_path / "params.npz"
    np.savez(npz_path, W_enc=np.ones((2, 3), dtype=np.float32))
    assert _load_sae_tensors(npz_path, checkpoint_format="npz")["W_enc"].shape == (2, 3)

    torch_path = tmp_path / "layer.sae.pt"
    torch.save({"W_dec": torch.ones(3, 2)}, torch_path)
    assert _load_sae_tensors(torch_path, checkpoint_format="torch")["W_dec"].shape == (3, 2)


def test_sae_encoder_can_skip_decoder_bias_input_subtraction(tmp_path: Path) -> None:
    weights = tmp_path / "params.npz"
    np.savez(
        weights,
        W_enc=np.eye(2, dtype=np.float32),
        W_dec=np.eye(2, dtype=np.float32),
        b_enc=np.zeros(2, dtype=np.float32),
        b_dec=np.array([10.0, 0.0], dtype=np.float32),
    )
    baseline = {
        "weights_file": str(weights),
        "checkpoint_format": "npz",
        "width": 2,
        "activation": "relu",
        "apply_b_dec_to_input": False,
        "hook_name_template": "model.layers.{layer}",
    }
    encoder = SAEFeatureEncoder(
        {
            "baselines": {"sae": baseline},
            "hidden_size": 2,
            "model_id": "owner/model",
            "saebench_model_name": "owner/model",
            "layer": 0,
        },
        device="cpu",
        dtype=torch.float32,
    )

    # Gemma Scope applies W_enc to x directly; b_dec is only added on decode.
    assert torch.equal(encoder.encode(torch.tensor([[1.0, 2.0]])), torch.tensor([[1.0, 2.0]]))


def test_resume_ignores_backend_cache_state(tmp_path: Path) -> None:
    resolved = {
        "experiment": "saebench-sparse-probing",
        "lens": "owner/lens",
        "model_id": "openai-community/gpt2",
        "model_revision": "revision",
        "activation_site": "resid_post",
        "layers": [6],
        "preset": {"name": "smoke"},
        "backend": {
            "name": "paper-standard",
            "repository": "https://example.test/SAEBench.git",
            "commit": "abc123",
            "saebench_model_name": "gpt2-small",
            "cached": False,
            "cache_path": "/old/cache",
        },
    }
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"resolved": resolved, "status": "failed", "layer_runs": {}}),
        encoding="utf-8",
    )
    current = json.loads(json.dumps(resolved))
    current["backend"]["cached"] = True
    current["backend"]["cache_path"] = "/new/cache"

    run = _load_or_initialize_run(path, current)

    assert run["status"] == "running"


def test_resume_allows_adding_a_baseline(tmp_path: Path) -> None:
    resolved = {
        "experiment": "saebench-sparse-probing",
        "lens": "owner/lens",
        "model_id": "openai-community/gpt2",
        "model_revision": "revision",
        "activation_site": "resid_post",
        "layers": [6],
        "preset": {"name": "smoke"},
        "baselines": {"pca": {"name": "PCA"}},
        "backend": {
            "name": "paper-standard",
            "repository": "https://example.test/SAEBench.git",
            "commit": "abc123",
            "saebench_model_name": "gpt2-small",
        },
    }
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"resolved": resolved, "status": "complete", "layer_runs": {}}),
        encoding="utf-8",
    )
    current = json.loads(json.dumps(resolved))
    current["baselines"]["random"] = {"name": "Random"}

    run = _load_or_initialize_run(path, current)

    assert run["status"] == "running"
    assert "random" in run["resolved"]["baselines"]


def test_completed_methods_combines_raw_and_durable_method_files(tmp_path: Path) -> None:
    layer_dir = tmp_path / "layers" / "layer_06"
    saebench_dir = layer_dir / "saebench"
    saebench_dir.mkdir(parents=True)
    (layer_dir / "raw-result.json").write_text(
        json.dumps({"methods": {"ica": {"eval_result_metrics": {}}}}), encoding="utf-8"
    )
    (saebench_dir / "pca_custom_sae_eval_results.json").write_text(
        json.dumps({"eval_result_metrics": {}}), encoding="utf-8"
    )
    assert _completed_methods(tmp_path, 6) == {"ica", "pca"}


def test_completed_methods_ignores_corrupt_and_incomplete_results(tmp_path: Path) -> None:
    layer_dir = tmp_path / "layers" / "layer_06"
    saebench_dir = layer_dir / "saebench"
    saebench_dir.mkdir(parents=True)
    (layer_dir / "raw-result.json").write_text("{not-json", encoding="utf-8")
    (saebench_dir / "ica_custom_sae_eval_results.json").write_text(
        json.dumps({"eval_result_unstructured": {}}), encoding="utf-8"
    )

    assert _completed_methods(tmp_path, 6) == set()


def test_random_encoder_is_orthogonal_and_reproducible(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    layer_file = tmp_path / "layer.safetensors"
    save_file({"center": torch.zeros(4)}, layer_file)
    snapshot = {
        "layer_file": str(layer_file),
        "baselines": {"random": {"seed": 7}},
        "layer": 2,
        "saebench_model_name": "tiny",
        "row_normalize": False,
        "norm_eps": 1e-12,
    }
    first = RandomFeatureEncoder(snapshot, device="cpu", dtype=torch.float32)
    second = RandomFeatureEncoder(snapshot, device="cpu", dtype=torch.float32)
    identity = first.components @ first.components.T
    assert torch.allclose(identity, torch.eye(4), atol=1e-5)
    assert torch.equal(first.components, second.components)
    assert first.cfg.random_seed == 9


def test_resume_reports_result_defining_mismatch(tmp_path: Path) -> None:
    previous = {
        "experiment": "saebench-sparse-probing",
        "lens": "owner/lens",
        "model_id": "openai-community/gpt2",
        "model_revision": "revision",
        "activation_site": "resid_post",
        "layers": [6],
        "preset": {"name": "smoke"},
        "backend": {
            "name": "paper-standard",
            "repository": "https://example.test/SAEBench.git",
            "commit": "abc123",
            "saebench_model_name": "gpt2-small",
        },
    }
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"resolved": previous, "status": "failed", "layer_runs": {}}),
        encoding="utf-8",
    )
    current = json.loads(json.dumps(previous))
    current["layers"] = [11]

    with pytest.raises(ValueError, match=r"resolved.layers: \[6\] -> \[11\]"):
        _load_or_initialize_run(path, current)


def test_collect_rows_and_plot_saved_results(tmp_path: Path) -> None:
    layer_dir = tmp_path / "layers" / "layer_06"
    layer_dir.mkdir(parents=True)
    raw = {
        "eval_config": {"k_values": [1, 5, 10]},
        "eval_result_metrics": {
            "sae": {
                "sae_top_1_test_accuracy": 0.71,
                "sae_top_5_test_accuracy": 0.82,
                "sae_top_10_test_accuracy": 0.86,
            }
        },
    }
    (layer_dir / "raw-result.json").write_text(json.dumps(raw), encoding="utf-8")
    rows = collect_result_rows(tmp_path, [6])
    assert rows[-1] == {
        "method": "ica",
        "layer": 6,
        "k": 10,
        "mean_probe_accuracy": 0.86,
    }

    pytest.importorskip("matplotlib")
    payload = {
        "experiment": {"experiment": "saebench-sparse-probing"},
        "rows": rows,
    }
    outputs = render_sparse_probing_figure(
        payload,
        output=tmp_path / "figures",
        stem="gpt2-smoke",
        formats=["png"],
        force=False,
    )
    assert outputs == [
        tmp_path / "figures" / "gpt2-smoke.png",
        tmp_path / "figures" / "gpt2-smoke.txt",
    ]
    assert outputs[0].stat().st_size > 0
    assert "SAEBench sparse-probing performance" in outputs[1].read_text(encoding="utf-8")


def test_collect_comparison_methods(tmp_path: Path) -> None:
    layer_dir = tmp_path / "layers" / "layer_06"
    layer_dir.mkdir(parents=True)
    method_result = {
        "eval_config": {"k_values": [1]},
        "eval_result_metrics": {"sae": {"sae_top_1_test_accuracy": 0.75}},
    }
    (layer_dir / "raw-result.json").write_text(
        json.dumps({"methods": {"ica": method_result, "pca": method_result}}),
        encoding="utf-8",
    )

    rows = collect_result_rows(tmp_path, [6])

    assert [row["method"] for row in rows] == ["ica", "pca"]


def test_collect_rows_recovers_custom_k_from_unstructured_metrics(tmp_path: Path) -> None:
    layer_dir = tmp_path / "layers" / "layer_06"
    layer_dir.mkdir(parents=True)
    raw = {
        "eval_config": {"k_values": [100, 200, 500]},
        "eval_result_metrics": {
            "sae": {"sae_top_100_test_accuracy": 0.8},
        },
        "eval_result_unstructured": {
            "dataset-a_results": {
                "sae_top_200_test_accuracy": {"0": 0.7, "1": 0.9},
                "sae_top_500_test_accuracy": {"0": 0.8, "1": 1.0},
            },
            "dataset-b_results": {
                "sae_top_200_test_accuracy": {"0": 0.6, "1": 0.8},
                "sae_top_500_test_accuracy": {"0": 0.7, "1": 0.9},
            },
        },
    }
    (layer_dir / "raw-result.json").write_text(json.dumps(raw), encoding="utf-8")

    rows = collect_result_rows(tmp_path, [6])

    assert [(row["k"], row["mean_probe_accuracy"]) for row in rows] == [
        (100, 0.8),
        (200, pytest.approx(0.75)),
        (500, pytest.approx(0.85)),
    ]


def test_render_paper_comparison_with_pending_panel(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    payload = {
        "experiment": {
            "experiment": "saebench-sparse-probing",
            "layers": [6, 10],
        },
        "rows": [
            {"method": method, "layer": 6, "k": k, "mean_probe_accuracy": value}
            for method, value in (("ica", 0.8), ("sae", 0.78), ("pca", 0.7))
            for k in (1, 10)
        ],
    }
    outputs = render_sparse_probing_panels(
        [payload, None, payload],
        titles=["GPT-2 Small", "Gemma 2 2B", "Qwen 3.5 9B Base"],
        output=tmp_path / "figures",
        stem="comparison",
        formats=["png"],
        force=False,
    )
    assert outputs == [
        tmp_path / "figures" / "comparison.png",
        tmp_path / "figures" / "comparison.txt",
    ]
    assert outputs[0].stat().st_size > 0
    assert "Gemma 2 2B" in outputs[1].read_text(encoding="utf-8")


def test_full_linear_dictionary_reconstruction_is_exact() -> None:
    target = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    identity = torch.eye(2)
    result = _linear_dictionary_metrics(
        target,
        target,
        target.norm(dim=-1, keepdim=True),
        torch.zeros(2),
        identity,
        identity,
        [1],
        restore_norm=False,
    )
    assert result["curve"]["2"]["nmse"]["mean"] == pytest.approx(0.0)
    assert result["curve"]["2"]["cosine"]["mean"] == pytest.approx(1.0)
    assert result["curve"]["1"]["nmse"]["mean"] > 0


def test_render_reconstruction_writes_both_metrics(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    payload = {
        "experiment": {"model_id": "owner/model", "layers": [0]},
        "rows": [
            {
                "layer": 0,
                "method": method,
                "k": str(k),
                "effective_k": k,
                "nmse_mean": value,
                "cosine_mean": 1.0 - value,
            }
            for method in ("ica", "sae")
            for k, value in ((1, 0.5), (10, 0.1))
        ],
    }
    outputs = render_reconstruction(
        [payload],
        titles=["Model"],
        output=tmp_path,
        formats=["png"],
        force=False,
    )
    assert {path.name for path in outputs} == {
        "reconstruction-nmse.png",
        "reconstruction-nmse.txt",
        "reconstruction-cosine.png",
        "reconstruction-cosine.txt",
    }
    caption = (tmp_path / "reconstruction-cosine.txt").read_text(encoding="utf-8")
    assert "top-k budgets 1, 10" in caption
    assert "top-k budgets 1, 3, 10, 32" not in caption


def test_reconstruction_curve_uses_effective_k_and_collapses_saturation() -> None:
    rows = [
        {
            "method": "sae",
            "k": str(k),
            "effective_k": effective,
            "cosine_mean": value,
        }
        for k, effective, value in (
            (1, 1.0, 0.6),
            (1, 1.0, 0.8),
            (32, 31.99999, 0.9),
            (32, 32.0, 0.94),
            (100, 31.99999, 0.9),
            (100, 32.0, 0.94),
        )
    ]
    curve = _mean_curve(rows, "sae", "cosine")
    assert len(curve) == 2
    assert curve[0] == pytest.approx((1.0, 0.7))
    assert curve[1] == pytest.approx((31.999995, 0.92))
