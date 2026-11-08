#!/usr/bin/env python3
"""Perturbation size of TPP ablations, recomputed from cached activations.

For one official TPP run (model, layer, seed), this script
1. loads the cached SAEBench activations and trained probes (no model forward);
2. rebuilds the ICA and SAE wrappers exactly as `experiments/tpp/worker.py` does;
3. reselects the top-k features per class with SAEBench's own functions;
4. reproduces the TPP metrics (a sanity check against the stored results); and
5. measures the size of each ablation, Δx = decode(f) − decode(f with top-k zeroed),
   on the held-out test tokens.

Reported per (method, dataset, class, k):
- token_rel: mean over valid tokens of ‖Δx‖ / ‖x‖;
- token_abs: mean over valid tokens of ‖Δx‖;
- token_nonzero: fraction of valid tokens where Δx ≠ 0;
- pooled_rel: mean over examples of ‖mean_t Δx‖ / ‖mean_t x‖ (what the probe sees).

Run with the pinned SAEBench environment, from the repository root:
  $SAEBENCH/.venv/bin/python experiments/tpp/perturbation-norm/compute.py \
      --run <official results dir for one model> --layer 12
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub.constants import HF_HUB_CACHE
from tqdm.auto import tqdm

if not hasattr(transformers, "TRANSFORMERS_CACHE"):  # same shim as worker.py
    transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # experiments/tpp (worker.py)

import worker  # noqa: E402  (also puts the project src on sys.path)
from sae_bench.evals.scr_and_tpp import main as tpp  # noqa: E402
from sae_bench.sae_bench_utils import dataset_info  # noqa: E402

# SAEBench caches activations under the wrapper's hook name, so each method has its own
# cache folder (and its own trained probes).
METHOD_HOOK = {"ica": "blocks.{layer}.hook_resid_post", "sae": "model.layers.{layer}"}
RESULT_FILE = {"ica": "ica_custom_sae_eval_results.json", "sae": "sae_custom_sae_eval_results.json"}


@torch.no_grad()
def perturbation(sae, test_acts: dict[str, torch.Tensor], mask: torch.Tensor, batch: int,
                 device: str) -> dict[str, float]:
    sums = {"token_rel": 0.0, "token_abs": 0.0, "token_nonzero": 0.0, "pooled_rel": 0.0}
    n_tok = n_ex = 0
    for acts in test_acts.values():
        for i in range(0, acts.shape[0], batch):
            x = acts[i:i + batch].to(device)
            valid = x.abs().sum(-1) != 0  # SAEBench zeroes masked and boundary tokens
            f = sae.encode(x)
            full = sae.decode(f)
            f[..., mask] = 0.0
            dx = (full - sae.decode(f)) * valid[..., None]
            dn = dx.norm(dim=-1)
            xn = x.norm(dim=-1).clamp_min(1e-12)
            sums["token_rel"] += float((dn / xn)[valid].sum())
            sums["token_abs"] += float(dn[valid].sum())
            sums["token_nonzero"] += float((dn[valid] > 0).sum())
            n_tok += int(valid.sum())
            count = valid.sum(-1).clamp_min(1)[:, None]
            pooled_dx = dx.sum(1) / count
            pooled_x = (x * valid[..., None]).sum(1) / count
            sums["pooled_rel"] += float((pooled_dx.norm(dim=-1)
                                         / pooled_x.norm(dim=-1).clamp_min(1e-12)).sum())
            n_ex += x.shape[0]
    return {k: v / (n_ex if k == "pooled_rel" else n_tok) for k, v in sums.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="official results dir, e.g. .../seed-0/gemma2")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=125)
    p.add_argument("--max-per-class", type=int, default=200,
                   help="examples kept per class and split (light mode; 0 = all)")
    args = p.parse_args()

    run = args.run.resolve()
    layer_dir = run / "layers" / f"layer_{args.layer:02d}"
    snapshot = json.loads((run / "checkpoints" / f"layer_{args.layer:02d}" / "snapshot.json").read_text())
    cache = Path(json.loads((run / "storage.json").read_text())["layer_cache_dirs"][str(args.layer)])
    model_dir = next((cache / "tpp").iterdir())
    methods = worker.build_methods(snapshot, methods=("ica", "sae"), device=args.device,
                                   dtype=torch.float32)

    out: dict = {"run": str(run), "layer": args.layer, "methods": {}}
    for method, sae in methods.items():
        stored = json.loads((layer_dir / "saebench" / "tpp" / RESULT_FILE[method]).read_text())
        cfg = stored["eval_config"]
        n_values = cfg["n_values"]
        rows, reproduced = [], {}
        for dataset in cfg["dataset_names"]:
            stem = dataset.replace("/", "_")
            folder = model_dir / METHOD_HOOK[method].format(layer=args.layer)
            print(f"[{method}] loading {folder.name}/{stem} activations ...", flush=True)
            # Memory-mapped, then a small per-class subset copied out (keeps RAM low).
            full_acts = torch.load(folder / f"{stem}_activations.pt", mmap=True)
            m = args.max_per_class or None
            acts = {split: {c: t[:m].clone().float() for c, t in d.items()}
                    for split, d in full_acts.items()}
            del full_acts
            with open(folder / f"{stem}_probes.pkl", "rb") as fh:
                probes = pickle.load(fh)
            classes = dataset_info.chosen_classes_per_dataset[dataset]
            effects = tpp.get_all_node_effects_for_one_sae(
                sae, probes["llm_probes"], classes, False, acts["train"], args.batch)
            # Sanity check: reproduce SAEBench's TPP accuracies with the same selection.
            accs = tpp.perform_feature_ablations(
                probes["llm_probes"], sae, args.batch, acts["test"], effects, n_values,
                classes, cfg["probe_test_batch_size"], False)
            metrics, _ = tpp.create_tpp_plotting_dict(accs, probes["llm_test_accuracies"])
            reproduced[dataset] = metrics
            for cls in tqdm(classes, desc=f"{method} {stem} norms"):
                for k in n_values:
                    mask = tpp.select_top_n_features(effects[cls], k, cls)
                    row = perturbation(sae, acts["test"], mask, args.batch, args.device)
                    rows.append({"dataset": dataset, "class": cls, "k": k,
                                 "n_selected": int(mask.sum()), **row})
            del acts
            gc.collect()
        # Average reproduced TPP over datasets, the way SAEBench reports it.
        repro = {key: sum(m[key] for m in reproduced.values()) / len(reproduced)
                 for key in next(iter(reproduced.values()))}
        stored_tpp = stored["eval_result_metrics"]["tpp_metrics"]
        check = {k: {"stored": stored_tpp.get(f"tpp_threshold_{k}_total_metric"),
                     "reproduced": repro.get(f"tpp_threshold_{k}_total_metric")} for k in n_values}
        print(f"[{method}] TPP check:", {k: (round(v['stored'] or 0, 4), round(v['reproduced'] or 0, 4))
                                         for k, v in check.items()})
        out["methods"][method] = {"d_features": int(sae.W_dec.shape[0]), "tpp_check": check,
                                  "tpp_reproduced": repro, "rows": rows}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(args.output)


if __name__ == "__main__":
    main()
