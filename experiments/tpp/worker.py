"""Isolated SAEBench TPP worker for the five-representation pilot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from gb10_load_llm import load_model_to_cuda
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# The worker runs inside SAEBench's isolated environment rather than the
# project's uv environment. Make the checked-out ICA Lens source explicit,
# matching the direct-source fallback used by the core SAEBench worker.
PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from icalens.experiments._sae import SAEFeatureEncoder  # noqa: E402
from icalens.experiments._saebench_worker import (  # noqa: E402
    HFHookedModel,
    _unwrap_runtime_types,
)

METHODS = ("ica", "unfitted_ica", "sae", "untrained_sae_matched_l0", "pca")
METHOD_DEFINITION_VERSION = 4


@dataclass
class Config:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    architecture: str
    activation_fn_str: str
    dtype: str
    device: str
    random_seed: int | None = None
    top_k: int | None = None


class SignedLinearDictionary(torch.nn.Module):
    """A full-rank signed linear dictionary with an exact encode/decode pair."""

    def __init__(
        self,
        *,
        center: torch.Tensor,
        reading: torch.Tensor,
        writing: torch.Tensor,
        snapshot: dict[str, Any],
        architecture: str,
        device: str,
        dtype: torch.dtype,
        seed: int | None = None,
        orientation: torch.Tensor | None = None,
        activation: str = "identity",
    ) -> None:
        super().__init__()
        self.register_buffer("center", center.to(torch.float32))
        self.register_buffer("reading", reading.to(torch.float32))
        signs = (
            torch.ones(reading.shape[0], dtype=torch.float32)
            if orientation is None
            else orientation.to(torch.float32)
        )
        if signs.shape != (reading.shape[0],) or not bool(torch.all((signs == 1) | (signs == -1))):
            raise ValueError("orientation must contain one +1 or -1 per linear component")
        self.register_buffer("orientation", signs)
        decoder = writing.T.to(torch.float32) * signs[:, None]
        decoder_norms = torch.linalg.vector_norm(decoder, dim=-1).clamp_min(1e-12)
        self.register_buffer("decoder_norms", decoder_norms)
        self.W_dec = torch.nn.Parameter(decoder / decoder_norms[:, None], requires_grad=False)
        if activation not in {"identity", "relu"}:
            raise ValueError(f"unsupported linear dictionary activation: {activation!r}")
        self.activation = activation
        self.row_normalize = bool(snapshot["row_normalize"])
        self.norm_eps = float(snapshot["norm_eps"])
        self.device = torch.device(device)
        self.dtype = dtype
        self.cfg = Config(
            model_name=str(snapshot["saebench_model_name"]),
            d_in=int(reading.shape[1]),
            d_sae=int(reading.shape[0]),
            hook_layer=int(snapshot["layer"]),
            hook_name=f"blocks.{int(snapshot['layer'])}.hook_resid_post",
            architecture=architecture,
            activation_fn_str=activation,
            dtype=str(dtype).removeprefix("torch."),
            device=device,
            random_seed=seed,
        )
        self.to(device=self.device, dtype=dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        work = x.to(device=self.device, dtype=self.dtype)
        if self.row_normalize:
            work = work / torch.linalg.vector_norm(work, dim=-1, keepdim=True).clamp_min(
                self.norm_eps
            )
        score = (work - self.center) @ self.reading.T
        code = score * self.orientation
        if self.activation == "relu":
            code = torch.relu(code)
        return code * self.decoder_norms

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return code.to(device=self.device, dtype=self.dtype) @ self.W_dec + self.center


class TrainedSAE(torch.nn.Module):
    """Give the shared SAE encoder SAEBench's stateful decode signature."""

    def __init__(self, snapshot: dict[str, Any], *, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.encoder = SAEFeatureEncoder(snapshot, device=device, dtype=dtype)
        self.W_dec = self.encoder.W_dec
        self.cfg = self.encoder.cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.reference: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.reference = x
        return self.encoder.encode(x)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if self.reference is None:
            raise RuntimeError("decode must follow encode for the same batch")
        return self.encoder.decode(code, reference=self.reference)


class UntrainedMatchedL0SAE(torch.nn.Module):
    """Tied random SAE with the published checkpoint's sparsity budget."""

    def __init__(self, snapshot: dict[str, Any], *, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        baseline = snapshot["baselines"]["sae"]
        hidden = int(snapshot["hidden_size"])
        width = int(baseline["width"])
        seed = 10_000 + int(snapshot["layer"])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        decoder = torch.randn((width, hidden), generator=generator, dtype=torch.float32)
        decoder /= torch.linalg.vector_norm(decoder, dim=1, keepdim=True).clamp_min(1e-12)
        self.W_dec = torch.nn.Parameter(decoder, requires_grad=False)
        self.W_enc = torch.nn.Parameter(decoder.T.contiguous(), requires_grad=False)
        self.device = torch.device(device)
        self.dtype = dtype
        self.normalize = str(baseline.get("normalize_activations", "none"))
        configured_k = baseline.get("top_k")
        self.top_k = int(configured_k) if configured_k is not None else _average_l0(baseline)
        self.cfg = Config(
            model_name=str(snapshot["saebench_model_name"]),
            d_in=hidden,
            d_sae=width,
            hook_layer=int(snapshot["layer"]),
            hook_name=str(baseline["hook_name_template"]).format(layer=snapshot["layer"]),
            architecture="untrained_tied_random_matched_l0",
            activation_fn_str="topk",
            dtype=str(dtype).removeprefix("torch."),
            device=device,
            random_seed=seed,
            top_k=self.top_k,
        )
        self.to(device=self.device, dtype=dtype)

    def _preprocess(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        work = x.to(device=self.device, dtype=self.dtype)
        if self.normalize == "none":
            return work, None, None
        mean = work.mean(dim=-1, keepdim=True)
        std = (work - mean).std(dim=-1, keepdim=True)
        return (work - mean) / (std + 1e-5), mean, std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        work, mean, std = self._preprocess(x)
        self.reference_stats = (mean, std)
        values, indices = torch.topk(torch.relu(work @ self.W_enc), k=self.top_k, dim=-1)
        code = torch.zeros(
            (*work.shape[:-1], self.W_dec.shape[0]), device=work.device, dtype=work.dtype
        )
        code.scatter_(-1, indices, values)
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        reconstruction = code @ self.W_dec
        mean, std = self.reference_stats
        if self.normalize != "none":
            assert mean is not None and std is not None
            reconstruction = reconstruction * std + mean
        return reconstruction


def _average_l0(baseline: dict[str, Any]) -> int:
    checkpoint = str(baseline.get("weights_file", baseline.get("checkpoint", "")))
    match = re.search(r"average_l0_(\d+)", checkpoint)
    if not match:
        raise ValueError(
            "JumpReLU untrained control requires an average_l0_N checkpoint path; "
            "refusing to reuse trained thresholds"
        )
    return int(match.group(1))


def _fastica_initial_unmixing(shape: torch.Size, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(shape, generator=generator, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(initial @ initial.T)
    inverse_sqrt = (
        eigenvectors * eigenvalues.clamp_min(torch.finfo(torch.float64).eps).rsqrt()
    ) @ eigenvectors.T
    return inverse_sqrt @ initial


def build_methods(
    snapshot: dict[str, Any], *, methods: tuple[str, ...], device: str, dtype: torch.dtype
) -> dict[str, torch.nn.Module]:
    tensors = load_file(snapshot["layer_file"], device="cpu")
    center = tensors["center"].to(torch.float32)
    fitted_reading = tensors["reading_matrix"].to(torch.float64)
    fitted_writing = tensors["writing_matrix"].to(torch.float64)
    tail_signs = tensors["tail_signs"].to(torch.float32)
    result: dict[str, torch.nn.Module] = {}
    if "ica" in methods:
        result["ica"] = SignedLinearDictionary(
            center=center,
            reading=fitted_reading,
            writing=fitted_writing,
            snapshot=snapshot,
            architecture="fitted_ica_profile_oriented_relu",
            device=device,
            dtype=dtype,
            orientation=tail_signs,
            activation="relu",
        )
    if "unfitted_ica" in methods:
        covariance_whitener_gram = fitted_reading.T @ fitted_reading
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_whitener_gram)
        symmetric_whitener = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
        initial_unmixing = _fastica_initial_unmixing(
            fitted_reading.shape, seed=int(snapshot["fitting_seed"])
        )
        initial_reading = initial_unmixing @ symmetric_whitener
        initial_writing = torch.linalg.pinv(initial_reading)
        result["unfitted_ica"] = SignedLinearDictionary(
            center=center,
            reading=initial_reading,
            writing=initial_writing,
            snapshot=snapshot,
            architecture="fastica_initialization",
            device=device,
            dtype=dtype,
            seed=int(snapshot["fitting_seed"]),
        )
    if "sae" in methods:
        result["sae"] = TrainedSAE(snapshot, device=device, dtype=dtype)
    if "untrained_sae_matched_l0" in methods:
        result["untrained_sae_matched_l0"] = UntrainedMatchedL0SAE(
            snapshot, device=device, dtype=dtype
        )
    if "pca" in methods:
        covariance = fitted_writing @ fitted_writing.T
        _, pca_vectors = torch.linalg.eigh(covariance)
        pca_reading = pca_vectors.flip(1).T.contiguous()
        pca_writing = pca_reading.T
        result["pca"] = SignedLinearDictionary(
            center=center,
            reading=pca_reading,
            writing=pca_writing,
            snapshot=snapshot,
            architecture="pca",
            device=device,
            dtype=dtype,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saebench-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-cache", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.saebench_root))
    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE
    import sae_bench.evals.scr_and_tpp.main as tpp_main
    from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
    from sae_bench.sae_bench_utils import activation_collection

    for name in ("get_all_llm_activations", "get_llm_activations", "get_sae_meaned_activations"):
        if hasattr(activation_collection, name):
            setattr(
                activation_collection,
                name,
                _unwrap_runtime_types(getattr(activation_collection, name)),
            )

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    experiment_config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = experiment_config["settings"]
    methods = tuple(experiment_config["methods"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot["model_id"], revision=snapshot["model_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, settings["llm_dtype"])
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        snapshot["model_id"],
        device="cuda",
        dtype=dtype,
        touch="auto",
        low_cpu_mem_usage=True,
        revision=snapshot["model_revision"],
    )
    model.eval()

    class Factory:
        @staticmethod
        def from_pretrained_no_processing(_: str, device: str, dtype: torch.dtype) -> Any:
            return HFHookedModel(model, tokenizer, snapshot["evaluation_input_protocol"])

    tpp_main.HookedTransformer = Factory
    encoders = build_methods(snapshot, methods=methods, device="cuda", dtype=dtype)
    config = ScrAndTppEvalConfig(model_name=snapshot["saebench_model_name"], perform_scr=False)
    config.dataset_names = list(settings["datasets"])
    config.n_values = list(settings["n_values"])
    config.train_set_size = int(settings["train_size"])
    config.test_set_size = int(settings["test_size"])
    config.context_length = int(settings["context_length"])
    config.probe_epochs = int(settings["probe_epochs"])
    config.early_stopping_patience = config.probe_epochs
    config.llm_batch_size = int(settings["llm_batch_size"])
    config.sae_batch_size = int(settings["sae_batch_size"])
    config.llm_dtype = str(settings["llm_dtype"])
    config.random_seed = int(settings["random_seed"])
    config.lower_vram_usage = False
    args.output.mkdir(parents=True, exist_ok=True)
    args.activation_cache.mkdir(parents=True, exist_ok=True)
    tpp_main.run_eval(
        config,
        selected_saes=[(name, encoders[name]) for name in methods],
        device="cuda",
        output_path=str(args.output / "saebench"),
        force_rerun=False,
        clean_up_activations=False,
        save_activations=True,
        artifacts_path=str(args.activation_cache),
    )
    method_results = {}
    for name in methods:
        result_path = args.output / "saebench" / "tpp" / f"{name}_custom_sae_eval_results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"SAEBench did not produce the expected result: {result_path}")
        method_results[f"{name}_custom_sae"] = json.loads(result_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "method_definition_version": METHOD_DEFINITION_VERSION,
        "methods": method_results,
        "feature_configs": {name: asdict(encoders[name].cfg) for name in methods},
    }
    temporary = args.output / "result.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(args.output / "result.json")


if __name__ == "__main__":
    main()
