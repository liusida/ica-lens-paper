"""Minimal SAEBench RAVEL trial: ICA Lens vs Gemma Scope SAE on Gemma 2 2B.

Run with the pinned SAEBench interpreter (paper-standard backend), e.g.

    SAEBENCH=~/.cache/icalens/experiments/saebench/paper-standard/2a7480bf67f9f82f79e742be55fc1f5ec404bfa9
    $SAEBENCH/.venv/bin/python experiments/ravel/trial/run_trial.py --method ica

This is an exploratory trial, not the official protocol. It deliberately uses a
tiny RAVEL slice (one entity class, two attributes, few pairs, one epoch) so the
whole pipeline can be inspected end to end.

Compatibility patches applied to the pinned SAEBench RAVEL code:

1. ``transformers.TRANSFORMERS_CACHE`` shim (same as the sparse-probing worker).
2. transformers 5 decoder layers return a bare tensor, while SAEBench RAVEL
   hooks assume a tuple. Without a patch, ``get_layer_activations`` silently
   takes ``outputs[0]`` = the first batch row, and the MDBM hook raises. Both are
   replaced with tensor-aware versions that keep the original logic.
3. The MDBM hook's ``add_error`` flag is never set by SAEBench (always False).
   ``--add-error`` exposes it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub.constants import HF_HUB_CACHE

if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE  # type: ignore[attr-defined]

from safetensors.torch import load_file  # noqa: E402

import sae_bench.evals.ravel.main as ravel_main  # noqa: E402
import sae_bench.evals.ravel.mdbm as mdbm_module  # noqa: E402
import sae_bench.sae_bench_utils.activation_collection as activation_collection  # noqa: E402
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LENS_REPO = "sida/icalens-gemma-2-2b-pile10k"
LOCAL_LENS_DIR = Path(
    "/home/liusida/research/ICA-paper/icalens_pypi/local-icalens-models/official/"
    "icalens-gemma-2-2b-pile10k"
)
LENS_DIR = LOCAL_LENS_DIR
GEMMA_SCOPE_REPO = "google/gemma-scope-2b-pt-res"
GEMMA_SCOPE_REVISION = "fd571b47c1c64851e9b1989792367b9babb4af63"


def lens_file(relative: str) -> Path:
    """Local lens file if present, else download it from the Hub (pods)."""
    local = LENS_DIR / relative
    if local.is_file():
        return local
    return Path(hf_hub_download(LENS_REPO, relative))
# Same checkpoints as src/icalens/experiments/baseline_registry.json.
GEMMA_SCOPE_L0 = {12: "average_l0_41"}

ADD_ERROR = False
MASK_LOG: list[dict] = []

_original_train_mdbm = mdbm_module.train_mdbm


FULL_SWAP = False


def train_mdbm_logged(*args, **kwargs):
    """Record how many features each trained mask selects (or skip training for full swap)."""
    if FULL_SWAP:
        model, tokenizer, config, sae = args[:4]
        trained = mdbm_module.MDBM(model, tokenizer, config, sae).to(model.device)
        with torch.no_grad():
            trained.binary_mask.fill_(1.0)
    else:
        trained = _original_train_mdbm(*args, **kwargs)
    mask = mask_values(trained.binary_mask.detach(), 1.0, training_mode=False)
    MASK_LOG.append({"selected": int(mask.sum()), "total": int(mask.numel())})
    print(f"[trial] mask selects {MASK_LOG[-1]['selected']} / {MASK_LOG[-1]['total']} features")
    return trained


mdbm_module.train_mdbm = train_mdbm_logged


# --------------------------------------------------------------------------- #
# Patches for transformers 5 (decoder layers return a tensor, not a tuple).   #
# --------------------------------------------------------------------------- #
def _hidden(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


@torch.no_grad()
def get_layer_activations(model, target_layer, inputs, source_pos_B):
    acts_BLD = None

    def hook(module, inp, outputs):
        nonlocal acts_BLD
        acts_BLD = _hidden(outputs)
        return outputs

    handle = activation_collection.get_module(model, target_layer).register_forward_hook(hook)
    model(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs.get("attention_mask", None),
    )
    handle.remove()
    assert acts_BLD is not None
    return acts_BLD[list(range(acts_BLD.shape[0])), source_pos_B, :]


BUDGET: int | None = None  # max number of features the mask may select


def topk_indicator(logits_F, k):
    ind = torch.zeros_like(logits_F)
    ind[torch.topk(logits_F, min(k, logits_F.numel())).indices] = 1.0
    return ind


def mask_values(logits_F, temperature, training_mode):
    """SAEBench MDBM mask, optionally restricted to the top-k logits.

    Training: forward uses sigmoid(logit/T) on the top-k features only; a
    straight-through term passes gradients to every feature so others can enter.
    Eval: hard mask = top-k features with positive logits (at most k).
    """
    if not training_mode:
        hard = (logits_F > 0).to(logits_F.dtype)
        return hard if BUDGET is None else hard * topk_indicator(logits_F.detach(), BUDGET)
    soft = torch.sigmoid(logits_F / temperature)
    if BUDGET is None:
        return soft
    ind = topk_indicator(logits_F.detach(), BUDGET)
    return soft * ind + (soft - soft.detach()) * (1 - ind)


def create_intervention_hook(self, source_rep_BD, base_pos_B, training_mode=False, add_error=False):
    add_error = add_error or ADD_ERROR

    def intervention_hook(module, inputs, outputs):
        resid_BLD = _hidden(outputs)
        if resid_BLD.shape[1] == 1:  # KV-cached generation step
            return outputs
        rows = list(range(resid_BLD.shape[0]))
        with torch.no_grad():
            source_act_BF = self.sae.encode(source_rep_BD)
            resid_BD = resid_BLD[rows, base_pos_B, :]
            base_act_BF = self.sae.encode(resid_BD)
        mask_F = mask_values(self.binary_mask, self.temperature, training_mode)
        modified_BF = (1 - mask_F) * base_act_BF + mask_F * source_act_BF
        modified_BD = self.sae.decode(modified_BF)
        if add_error:
            modified_BD = modified_BD + (resid_BD - self.sae.decode(base_act_BF))
        resid_BLD = resid_BLD.clone()
        resid_BLD[rows, base_pos_B, :] = modified_BD.to(resid_BLD.dtype)
        if isinstance(outputs, tuple):
            return (resid_BLD, *outputs[1:])
        return resid_BLD

    return intervention_hook


activation_collection.get_layer_activations = get_layer_activations
mdbm_module.MDBM.create_intervention_hook = create_intervention_hook


# --------------------------------------------------------------------------- #
# Featurizers exposing the minimal SAE interface SAEBench RAVEL uses.          #
# --------------------------------------------------------------------------- #
@dataclass
class FeaturizerCfg:
    hook_layer: int
    hook_name: str
    d_sae: int
    d_in: int
    architecture: str
    model_name: str = "google/gemma-2-2b"
    notes: dict = field(default_factory=dict)


class Featurizer(torch.nn.Module):
    """float32 internally; ignores dtype casts so ICA stays exact."""

    def to(self, *args, **kwargs):  # type: ignore[override]
        kwargs.pop("dtype", None)
        args = tuple(a for a in args if not isinstance(a, torch.dtype))
        return super().to(*args, **kwargs)

    @property
    def device(self):
        return self.W_dec.device

    @property
    def dtype(self):
        return torch.float32


class ICASplitFeaturizer(Featurizer):
    """Split-sign ICA: features [relu(z)·n, relu(-z)·n], decoder ±w/n. Exact inverse."""

    def __init__(self, layer: int):
        super().__init__()
        meta = json.loads(lens_file("icalens.json").read_text())
        assert meta["input_preprocessing"]["row_normalization"] == "none"
        t = load_file(str(lens_file(meta["layers"][str(layer)]["file"])))
        reading, writing, center = t["reading_matrix"], t["writing_matrix"], t["center"]
        norms = writing.norm(dim=0).clamp_min(1e-12)  # (C,)
        unit = (writing / norms).T  # (C, D)
        self.register_buffer("reading", reading.float())
        self.register_buffer("center", center.float())
        self.register_buffer("norms", norms.float())
        self.W_dec = torch.nn.Parameter(torch.cat([unit, -unit]).float(), requires_grad=False)
        c, d = reading.shape
        self.cfg = FeaturizerCfg(layer, f"model.layers.{layer}", 2 * c, d, "ica_split_sign",
                                 notes={"lens": LENS_DIR.name, "components": c})

    def encode(self, x):
        z = (x.float() - self.center) @ self.reading.T * self.norms
        return torch.cat([z.clamp_min(0), (-z).clamp_min(0)], dim=-1)

    def decode(self, f):
        return f.float() @ self.W_dec + self.center


class PCASplitFeaturizer(ICASplitFeaturizer):
    """Split-sign PCA from the lens's own whitening.

    reading = R·Wh with R orthogonal, so readingᵀreading = Σ⁻¹. Its eigenvectors
    are the principal directions; PCA coordinates are whitened (unit variance),
    matching ICA's scale. Writing matrix = reading⁻¹.
    """

    def __init__(self, layer: int):
        super().__init__(layer)
        reading = self.reading.double()
        precision = reading.T @ reading  # Σ⁻¹
        evals, evecs = torch.linalg.eigh(precision)  # ascending precision = descending variance
        pca_reading = evals.clamp_min(1e-30).sqrt()[:, None] * evecs.T  # (C, D)
        pca_writing = torch.linalg.inv(pca_reading)  # (D, C)
        norms = pca_writing.norm(dim=0).clamp_min(1e-12)
        unit = (pca_writing / norms).T
        self.reading = pca_reading.float()
        self.norms = norms.float()
        self.W_dec = torch.nn.Parameter(torch.cat([unit, -unit]).float(), requires_grad=False)
        self.cfg.architecture = "pca_split_sign"
        self.cfg.notes = {**self.cfg.notes, "derived_from": "ica_lens_whitening"}


class RandomSplitFeaturizer(ICASplitFeaturizer):
    """Split-sign features in a seeded random orthonormal basis (Haar), centered at the lens center."""

    def __init__(self, layer: int, seed: int = 0):
        super().__init__(layer)
        d = self.reading.shape[1]
        g = torch.Generator().manual_seed(seed)
        q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        q = q * torch.sign(torch.diagonal(r))  # Haar-uniform
        self.reading = q.T.float()  # rows are directions
        self.norms = torch.ones(d)
        self.W_dec = torch.nn.Parameter(torch.cat([q.T, -q.T]).float(), requires_grad=False)
        self.cfg.architecture = "random_orthogonal_split_sign"
        self.cfg.notes = {**self.cfg.notes, "seed": seed}


class NeuronSplitFeaturizer(ICASplitFeaturizer):
    """Split-sign raw residual dimensions (standard basis), centered at the lens center."""

    def __init__(self, layer: int):
        super().__init__(layer)
        d = self.reading.shape[1]
        eye = torch.eye(d)
        self.reading = eye
        self.norms = torch.ones(d)
        self.W_dec = torch.nn.Parameter(torch.cat([eye, -eye]), requires_grad=False)
        self.cfg.architecture = "neuron_split_sign"


class GemmaScopeFeaturizer(Featurizer):
    """Gemma Scope JumpReLU SAE with unit-norm decoder rows (norms folded into features)."""

    def __init__(self, layer: int):
        super().__init__()
        p = np.load(hf_hub_download(
            GEMMA_SCOPE_REPO, f"layer_{layer}/width_16k/{GEMMA_SCOPE_L0[layer]}/params.npz",
            revision=GEMMA_SCOPE_REVISION))
        W_dec = torch.from_numpy(p["W_dec"]).float()  # (F, D)
        norms = W_dec.norm(dim=1).clamp_min(1e-12)
        self.register_buffer("W_enc", torch.from_numpy(p["W_enc"]).float())
        self.register_buffer("b_enc", torch.from_numpy(p["b_enc"]).float())
        self.register_buffer("b_dec", torch.from_numpy(p["b_dec"]).float())
        self.register_buffer("threshold", torch.from_numpy(p["threshold"]).float())
        self.register_buffer("norms", norms)
        self.W_dec = torch.nn.Parameter(W_dec / norms[:, None], requires_grad=False)
        f, d = W_dec.shape
        self.cfg = FeaturizerCfg(layer, f"model.layers.{layer}", f, d, "jumprelu",
                                 notes={"checkpoint": f"layer_{layer}/width_16k/{GEMMA_SCOPE_L0[layer]}"})

    def encode(self, x):
        pre = x.float() @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold) * self.norms

    def decode(self, f):
        return f.float() @ self.W_dec + self.b_dec


def build_datasets(config: RAVELEvalConfig, dest: Path, workdir: Path) -> None:
    """Build SAEBench's filtered RAVEL datasets exactly as run_eval_single_dataset does.

    Raw downloads (which include a hidden .cache folder) stay in ``workdir``; only the
    small filtered dataset JSONs are copied to ``dest`` so every method pod can reuse them.
    """
    import shutil

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_bench.evals.ravel.instance import RAVELInstance, get_instance_name
    from sae_bench.sae_bench_utils import general_utils

    import random as _random

    _random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    config.model_name = ravel_main.LLM_NAME_MAP[config.model_name]
    config.artifact_dir = str(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="cuda", dtype=general_utils.str_to_dtype(config.llm_dtype),
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    for entity_class, attributes in config.entity_attribute_selection.items():
        start = time.time()
        orig = config.llm_batch_size
        config.llm_batch_size = orig * 8  # as in run_eval_single_dataset
        full = RAVELInstance.create_from_files(
            config=config, entity_type=entity_class, tokenizer=tokenizer,
            data_dir=config.artifact_dir, model=model, model_name=config.model_name,
            attribute_types=attributes, downsample=config.full_dataset_downsample)
        config.llm_batch_size = orig
        full.create_and_save_filtered_dataset(artifact_dir=config.artifact_dir,
                                              top_n_entities=config.top_n_entities)
        name = get_instance_name(entity_class, config.model_name, config.full_dataset_downsample,
                                 config.top_n_entities)
        shutil.copy(workdir / name, dest / name)
        print(f"[trial] built {name} in {time.time() - start:.0f}s")


def main() -> None:
    global ADD_ERROR, FULL_SWAP, BUDGET
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", choices=["ica", "pca", "random", "sae", "neurons"], default="ica")
    ap.add_argument("--default-config", action="store_true",
                    help="use SAEBench's default RAVEL config (ignores the trial size knobs)")
    ap.add_argument("--dataset-only", action="store_true",
                    help="only build the filtered RAVEL dataset(s) into <output>/dataset, then exit")
    ap.add_argument("--dataset-workdir", default="/tmp/ravel_dataset_work",
                    help="scratch dir for raw RAVEL downloads in --dataset-only mode")
    ap.add_argument("--tag", default="", help="suffix for repeat runs")
    ap.add_argument("--seed", type=int, default=None,
                    help="RAVEL eval seed (pairs, split, mask training); default 42 = SAEBench")
    ap.add_argument("--budget", type=int, default=None, help="max features the mask may select (top-k)")
    ap.add_argument("--full-swap", action="store_true", help="skip mask training; swap every feature")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--attributes", default="Country,Continent")
    ap.add_argument("--downsample", type=int, default=6000, help="prompts kept before the knowledge filter")
    ap.add_argument("--top-n-entities", type=int, default=100)
    ap.add_argument("--pairs", type=int, default=400, help="intervention pairs per attribute")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--add-error", action="store_true")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/ravel/pilot-runs/trial")
    ap.add_argument("--dataset-dir", type=Path, default=None,
                    help="pre-built filtered RAVEL dataset dir to copy into <output>/dataset")
    args = ap.parse_args()
    ADD_ERROR = args.add_error
    FULL_SWAP = args.full_swap
    BUDGET = args.budget

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset_dir is not None and not (out / "dataset").exists():
        import shutil
        shutil.copytree(args.dataset_dir, out / "dataset")
    featurizer = {"ica": ICASplitFeaturizer, "pca": PCASplitFeaturizer, "random": RandomSplitFeaturizer,"sae": GemmaScopeFeaturizer,
                  "neurons": NeuronSplitFeaturizer}[args.method](args.layer)

    # Reconstruction sanity check on random-ish input (exactness of the featurizer).
    with torch.no_grad():
        x = torch.randn(4, featurizer.cfg.d_in) * 50
        rel = ((featurizer.decode(featurizer.encode(x)) - x).norm() / x.norm()).item()
    print(f"[trial] {args.method} layer {args.layer}: d_sae={featurizer.cfg.d_sae}, "
          f"random-input relative reconstruction error={rel:.2e}")

    if args.default_config:
        # Exactly what `python -m sae_bench.evals.ravel.main --model_name gemma-2-2b` uses:
        # RAVELEvalConfig defaults, batch = LLM_NAME_TO_BATCH_SIZE // 4, dtype from table.
        config = RAVELEvalConfig(model_name="gemma-2-2b", artifact_dir=str(out / "dataset"))
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE["gemma-2-2b"] // 4
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE["gemma-2-2b"]
    else:
        config = RAVELEvalConfig(
            entity_attribute_selection={"city": args.attributes.split(",")},
            top_n_entities=args.top_n_entities,
            full_dataset_downsample=args.downsample,
            num_pairs_per_attribute=args.pairs,
            num_epochs=args.epochs,
            llm_batch_size=args.batch_size,
            llm_dtype="bfloat16",
            model_name="gemma-2-2b",
            artifact_dir=str(out / "dataset"),
        )
    if args.dataset_only:
        build_datasets(config, out / "dataset", Path(args.dataset_workdir))
        return
    if args.seed is not None:
        # Seeds pair sampling, the train/test split and mask training. The shared
        # filtered dataset (built once with seed 42) and the random basis (seed 0)
        # stay fixed.
        config.random_seed = args.seed
    name = (f"{args.method}_layer{args.layer}" + ("_adderror" if args.add_error else "")
            + ("_fullswap" if args.full_swap else "") + (f"_k{args.budget}" if args.budget else "")
            + (f"_{args.tag}" if args.tag else ""))
    start = time.time()
    results = ravel_main.run_eval(
        config,
        [(name, featurizer)],
        device="cuda",
        output_path=str(out / "results"),
        force_rerun=True,
        artifacts_path=str(out / "artifacts"),
    )
    elapsed = time.time() - start
    summary = next(iter(results.values()))
    compact = {
        "method": args.method,
        "layer": args.layer,
        "add_error": args.add_error,
        "full_swap": args.full_swap,
        "budget": args.budget,
        "default_config": args.default_config,
        "seed": config.random_seed,
        "featurizer_cfg": {k: str(v) for k, v in vars(featurizer.cfg).items()},
        "ravel_config": {k: str(v) for k, v in vars(config).items()},
        "d_sae": featurizer.cfg.d_sae,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": summary["eval_result_metrics"]["ravel"],
        "per_attribute": summary["eval_result_unstructured"],
        "mask_selection_in_attribute_order": MASK_LOG,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (out / f"{name}.json").write_text(json.dumps(compact, indent=2))
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
