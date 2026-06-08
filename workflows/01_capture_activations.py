#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from ica_lens.activation_capture import capture_random_token_hidden_state_shards
from ica_lens.config import load_toml
from ica_lens.datasets import iter_dataset_texts
from ica_lens.models import load_model_and_tokenizer, torch_dtype
from ica_lens.paths import CONFIGS_DIR, RESULTS_DIR


DEFAULT_CONFIG = CONFIGS_DIR / "activations" / "gpt2.toml"
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "reproduced" / "activations"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture model activations for ICA Lens reproduction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--shard-token-budget", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dtype", default="float32")
    parser.add_argument("--layers", nargs="+", default=None)
    parser.add_argument(
        "--include-embedding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store model input embeddings as an embedding layer. Default: true when all layers are captured.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_toml(args.config)
    activation_cfg = dict(cfg["activation"])
    model_cfg = _load_model_config(args.config, activation_cfg)
    model = dict(model_cfg["model"])

    token_budget = int(args.token_budget or activation_cfg["token_budget"])
    run_name = str(model["short_name"])
    capture_dir = args.output_root.resolve() / f"{run_name}_tok{token_budget}"
    manifest_path = capture_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"using existing activation manifest: {manifest_path}")
        return 0
    if args.force and capture_dir.exists():
        import shutil

        shutil.rmtree(capture_dir)

    texts_iter = iter_dataset_texts(
        path=str(activation_cfg["dataset_path"]),
        name=activation_cfg.get("dataset_name"),
        split=str(activation_cfg["dataset_split"]),
        text_column=str(activation_cfg["text_column"]),
        streaming=bool(activation_cfg.get("streaming", False)),
    )
    texts = list(islice(texts_iter, args.max_documents)) if args.max_documents is not None else list(texts_iter)
    if not texts:
        raise RuntimeError("No dataset texts were loaded.")

    loaded_model, tokenizer = load_model_and_tokenizer(str(model["id"]), device=args.device, dtype=args.model_dtype)
    manifest = capture_random_token_hidden_state_shards(
        texts=texts,
        model=loaded_model,
        tokenizer=tokenizer,
        output_dir=args.output_root.resolve(),
        run_name=run_name,
        model_id=str(model["id"]),
        model_short_name=run_name,
        dataset_manifest={
            "path": str(activation_cfg["dataset_path"]),
            "name": activation_cfg.get("dataset_name"),
            "split": str(activation_cfg["dataset_split"]),
            "text_column": str(activation_cfg["text_column"]),
            "loaded_documents": len(texts),
            "streaming": bool(activation_cfg.get("streaming", False)),
        },
        context_length=int(model["context_length"]),
        token_budget=token_budget,
        activation_dtype=str(activation_cfg.get("activation_dtype", "bfloat16")),
        shard_token_budget=int(args.shard_token_budget or activation_cfg.get("shard_token_budget", token_budget)),
        seed=int(activation_cfg.get("seed", 0)),
        layers=list(args.layers) if args.layers is not None else None,
    )
    include_embedding = bool(args.include_embedding) if args.include_embedding is not None else args.layers is None
    if include_embedding:
        _store_input_embedding_layer(
            manifest_path=manifest,
            model=loaded_model,
            storage_dtype=str(activation_cfg.get("activation_dtype", "bfloat16")),
            shard_token_budget=int(args.shard_token_budget or activation_cfg.get("shard_token_budget", token_budget)),
        )
    print(f"wrote activation manifest: {manifest}")
    return 0


def _load_model_config(config_path: Path, activation_cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(activation_cfg["model_config"]))
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return load_toml(path)


def _store_input_embedding_layer(
    *,
    manifest_path: Path,
    model: torch.nn.Module,
    storage_dtype: str,
    shard_token_budget: int,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture_dir = manifest_path.parent
    dtype = torch_dtype(storage_dtype)
    embedding = model.get_input_embeddings().weight.detach().to(dtype=dtype, device="cpu")
    if embedding.dim() != 2:
        raise RuntimeError(f"Expected a 2D embedding matrix, got shape {tuple(embedding.shape)}.")
    hidden_size = int(manifest["model"]["hidden_size"])
    if int(embedding.shape[1]) != hidden_size:
        raise RuntimeError(f"Embedding width {int(embedding.shape[1])} does not match hidden_size {hidden_size}.")

    vocab_size = int(embedding.shape[0])
    layer_shards = []
    for shard_index, start in enumerate(range(0, vocab_size, shard_token_budget)):
        stop = min(vocab_size, start + shard_token_budget)
        shard_name = f"shard_{shard_index:05d}.pt"
        token_ids = torch.arange(start, stop, dtype=torch.long)
        layer_shards.append(
            {
                "index": shard_index,
                "tokens": int(stop - start),
                "input_ids": _save_tensor(capture_dir, "embedding_input_ids", shard_name, token_ids),
                "doc_ids": _save_tensor(capture_dir, "embedding_doc_ids", shard_name, torch.full_like(token_ids, -1)),
                "positions": _save_tensor(capture_dir, "embedding_positions", shard_name, token_ids),
                "layers": {
                    "embedding": _save_tensor(capture_dir, "embedding", shard_name, embedding[start:stop].contiguous()),
                },
                "hidden_size": hidden_size,
                "row_source": "input_embedding_matrix",
            }
        )

    layers = list(manifest["capture"].get("layers") or [])
    manifest["capture"]["layers"] = ["embedding", *[layer for layer in layers if layer != "embedding"]]
    manifest["capture"]["embedding_layer_source"] = "input_embedding_matrix"
    manifest["capture"]["embedding_rows"] = vocab_size
    manifest.setdefault("layer_shards", {})["embedding"] = layer_shards
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _save_tensor(capture_dir: Path, subdir: str, shard_name: str, tensor: torch.Tensor) -> str:
    output_dir = capture_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_dir / shard_name)
    return f"{subdir}/{shard_name}"


if __name__ == "__main__":
    raise SystemExit(main())
