"""Standalone SAEBench sparse-probing worker executed in its pinned environment."""

from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from huggingface_hub.constants import HF_HUB_CACHE
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from icalens.experiments._display import ExperimentDisplay
    from icalens.experiments._sae import SAEFeatureEncoder
except ModuleNotFoundError:  # Direct execution inside the isolated environment.
    from _display import ExperimentDisplay  # type: ignore[import-not-found,no-redef]
    from _sae import SAEFeatureEncoder  # type: ignore[import-not-found,no-redef]


class _StopForward(Exception):
    pass


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class _CapturedBenchmarkOutput(io.TextIOBase):
    def __init__(self, display: Any) -> None:
        self.display = display

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self.display.capture(value)
        return len(value)

    def flush(self) -> None:
        self.display.flush()

    def isatty(self) -> bool:
        return False


class _LegacyBenchmarkDisplay:
    """Keep SAEBench chatter in a small live panel and a complete detail log."""

    def __init__(
        self,
        *,
        output: Path,
        completed: int,
        total: int,
        run_initial: int,
        run_started_at: float,
        title: str = "ICA Lens · SAEBench sparse probing",
        item_label: str = "Method",
        items_label: str = "Methods",
        recent_label: str = "Recent SAEBench output",
        detail_filename: str = "saebench-detail.log",
        source_dirty: bool = False,
    ) -> None:
        self.total = total
        self.completed = completed
        self.run_initial = run_initial
        self.run_started_at = run_started_at
        self.title = title
        self.item_label = item_label
        self.items_label = items_label
        self.recent_label = recent_label
        self.source_dirty = source_dirty
        self.dataset = "preparing"
        self.dataset_index = 0
        self.dataset_total = 0
        self.method = "—"
        self.method_order: list[str] = []
        self.dataset_completed_methods: set[str] = set()
        self.recent: deque[str] = deque(maxlen=7)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.terminal_out = sys.stdout
        self.terminal_err = sys.stderr
        self.interactive = bool(self.terminal_err.isatty())
        self.console = Console(file=self.terminal_err, force_terminal=self.interactive)
        output.mkdir(parents=True, exist_ok=True)
        self.detail_path = output / detail_filename
        self.detail = self.detail_path.open("a", encoding="utf-8")
        self.stream = _CapturedBenchmarkOutput(self)
        self.live: Live | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> _LegacyBenchmarkDisplay:
        self.detail.write(
            "# ICA Lens experiment run\n"
            f"started_at: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"working_directory: {Path.cwd()}\n"
            f"command: {shlex.join(sys.argv)}\n\n"
        )
        self.detail.flush()
        if self.interactive:
            self.live = Live(
                self.render(),
                console=self.console,
                refresh_per_second=4,
                transient=False,
            )
            self.live.__enter__()
            self.thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self.thread.start()
        sys.stdout = self.stream
        sys.stderr = self.stream
        return self

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        sys.stdout = self.terminal_out
        sys.stderr = self.terminal_err
        if self.interactive:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join()
            if self.live is not None:
                self.live.update(self.render(), refresh=True)
                self.live.__exit__(error_type, error, traceback)
        self.detail.write(
            "\n# ICA Lens experiment run ended\n"
            f"ended_at: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"status: {'failed' if error is not None else 'complete'}\n"
        )
        self.detail.flush()
        self.detail.close()
        print(f"Full output: {self.detail_path}")

    def set_dataset(
        self,
        dataset: str,
        *,
        index: int,
        total: int,
        methods: list[str],
        completed_methods: set[str],
    ) -> None:
        with self.lock:
            self.dataset = dataset
            self.dataset_index = index
            self.dataset_total = total
            self.method = "—"
            self.method_order = list(methods)
            self.dataset_completed_methods = set(completed_methods)
        self.refresh()

    def track_methods(self, iterable: Any, **_: Any) -> Any:
        for item in iterable:
            with self.lock:
                self.method = str(item[0])
            self.refresh()
            completed = False
            try:
                yield item
                completed = True
            finally:
                if completed:
                    with self.lock:
                        self.completed += 1
                        self.dataset_completed_methods.add(str(item[0]))
                    self.refresh()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.method = phase
        self.refresh()

    def capture(self, value: str) -> None:
        with self.lock:
            self.detail.write(value)
            self.detail.flush()
            clean = _ANSI_ESCAPE.sub("", value)
            for line in re.split(r"[\r\n]+", clean):
                line = line.strip()
                if line:
                    self.recent.append(line[-180:])
        if not self.interactive:
            self.terminal_out.write(value)
            self.terminal_out.flush()

    def flush(self) -> None:
        with self.lock:
            if not self.detail.closed:
                self.detail.flush()

    def refresh(self) -> None:
        if self.live is not None:
            self.live.update(self.render(), refresh=True)

    def _refresh_loop(self) -> None:
        while not self.stop_event.wait(1.0):
            self.refresh()

    def render(self) -> Any:
        with self.lock:
            completed = self.completed
            total = self.total
            dataset = self.dataset
            dataset_index = self.dataset_index
            dataset_total = self.dataset_total
            method = self.method
            method_order = list(self.method_order)
            completed_methods = set(self.dataset_completed_methods)
            elapsed = max(0.0, time.time() - self.run_started_at)
            recent = list(self.recent)
        percentage = 100.0 * completed / max(total, 1)
        completed_this_run = max(0, completed - self.run_initial)
        remaining = max(0, total - completed)
        if completed_this_run:
            eta = elapsed * remaining / completed_this_run
            timing = f"elapsed {_format_duration(elapsed)} · ETA ~{_format_duration(eta)}"
        else:
            timing = f"elapsed {_format_duration(elapsed)} · ETA estimating…"
        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        header.add_row(
            Text(f"Overall {completed}/{total} ({percentage:.1f}%)", style="bold"),
            Text(timing, style="cyan"),
        )
        header.add_row(ProgressBar(total=total, completed=completed, width=48), Text(""))
        if method in method_order:
            method_position = method_order.index(method) + 1
            method_label = f"{method.upper()} ({method_position}/{len(method_order)})"
        else:
            method_label = method
        task = Text.assemble(
            ("Dataset: ", "bold"),
            f"{dataset} ({dataset_index}/{dataset_total})" if dataset_total else dataset,
            (f"    {self.item_label}: ", "bold"),
            method_label,
        )
        method_status = Text(f"{self.items_label}: ", style="bold")
        if len(method_order) > 12:
            method_status.append(
                f" ✓ {len(completed_methods)}/{len(method_order)} complete ",
                style="bold green",
            )
            if method in method_order:
                method_status.append(f" ▶ {method.upper()} ", style="bold yellow")
            method_status.append(
                f" · {max(0, len(method_order) - len(completed_methods))} remaining",
                style="dim",
            )
        else:
            for name in method_order:
                if name in completed_methods:
                    method_status.append(f" ✓ {name.upper()} ", style="bold green")
                elif name == method:
                    method_status.append(f" ▶ {name.upper()} ", style="bold yellow")
                else:
                    method_status.append(f" · {name.upper()} ", style="dim")
        tail = Text("\n".join(recent) if recent else "Waiting for SAEBench output…", style="dim")
        main = Panel(
            Group(header, task, method_status, Text(self.recent_label, style="bold"), tail),
            title=self.title,
            border_style="blue",
        )
        if not self.source_dirty:
            return main
        warning = Panel(
            Text(
                "WARNING: uncommitted source; the recorded commit alone cannot reproduce this run.",
                style="bold bright_yellow",
            ),
            title="Source warning",
            border_style="bright_yellow",
        )
        return Group(main, warning)


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class _PolicyBenchmarkDisplay(ExperimentDisplay):
    """Expose the legacy worker callbacks through the shared policy display."""

    def __init__(
        self,
        *,
        output: Path,
        completed: int,
        total: int,
        run_initial: int,
        run_started_at: float,
        title: str = "ICA Lens · SAEBench sparse probing",
        item_label: str = "Method",
        items_label: str = "Methods",
        recent_label: str = "Recent SAEBench output",
        detail_filename: str = "saebench-detail.log",
        source_dirty: bool = False,
    ) -> None:
        del run_initial, item_label, items_label
        self.dataset = "preparing"
        self.dataset_index = 0
        self.dataset_total = 0
        super().__init__(
            output=output,
            title=title,
            completed=completed,
            total=total,
            source_dirty=source_dirty,
            detail_filename=detail_filename,
            recent_label=recent_label,
            unit_label="dataset-layer-method evaluations",
            started_at=run_started_at,
        )

    def set_dataset(
        self,
        dataset: str,
        *,
        index: int,
        total: int,
        methods: list[str],
        completed_methods: set[str],
    ) -> None:
        del methods, completed_methods
        self.dataset = dataset
        self.dataset_index = index
        self.dataset_total = total
        self.phase("Evaluating", dataset=f"{dataset} ({index}/{total})", method="—")

    def track_methods(self, iterable: Any, **_: Any) -> Any:
        for item in iterable:
            method = str(item[0])
            self.phase(
                "Evaluating",
                dataset=f"{self.dataset} ({self.dataset_index}/{self.dataset_total})",
                method=method,
            )
            completed = False
            try:
                yield item
                completed = True
            finally:
                if completed:
                    self.advance(refresh=True)

    def set_phase(self, phase: str) -> None:
        self.phase(
            phase,
            dataset=f"{self.dataset} ({self.dataset_index}/{self.dataset_total})",
        )


# Keep the worker API stable while using the shared long-run implementation.
_BenchmarkDisplay = _PolicyBenchmarkDisplay


def _blocks(model: torch.nn.Module) -> Any:
    candidates = (
        ("transformer", "h"),
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        blocks: Any = model
        for name in path:
            blocks = getattr(blocks, name, None)
            if blocks is None:
                break
        if isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)):
            return blocks
    raise ValueError(f"transformer blocks were not found on {type(model).__name__}")


class HFHookedModel:
    """The small TransformerLens surface used by SAEBench sparse probing."""

    def __init__(
        self, model: torch.nn.Module, tokenizer: Any, input_protocol: dict[str, Any] | None = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.input_protocol = input_protocol or {"boundary_source": "tokenizer"}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: str) -> HFHookedModel:
        self.model.to(device)
        return self

    def run_with_hooks(
        self,
        tokens: torch.Tensor,
        *,
        stop_at_layer: int,
        fwd_hooks: list[tuple[str, Any]],
    ) -> None:
        if len(fwd_hooks) != 1:
            raise ValueError("the sparse-probing adapter expects exactly one forward hook")
        hook_name, callback = fwd_hooks[0]
        try:
            layer = next(int(part) for part in reversed(hook_name.split(".")) if part.isdigit())
        except StopIteration as error:
            raise ValueError(f"unsupported hook name: {hook_name!r}") from error
        if stop_at_layer != layer + 1:
            raise ValueError("stop_at_layer must immediately follow the captured block")

        model_tokens, attention_mask, strip_prefix = _prepare_model_inputs(
            tokens.to(self.device), self.tokenizer, self.input_protocol
        )

        def capture(_: Any, __: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            callback(hidden[:, strip_prefix:], None)
            raise _StopForward

        handle = _blocks(self.model)[layer].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                try:
                    self.model(
                        input_ids=model_tokens,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                except _StopForward:
                    pass
        finally:
            handle.remove()


def _prepare_model_inputs(
    tokens: torch.Tensor, tokenizer: Any, input_protocol: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Apply artifact-matched document framing and construct a real attention mask."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    attention_mask = torch.ones_like(tokens, dtype=torch.long)
    if pad_token_id is not None:
        attention_mask = (tokens != int(pad_token_id)).to(dtype=torch.long)
    if input_protocol.get("boundary_source") != "adapter":
        return tokens, attention_mask, 0
    token_id = int(input_protocol["token_id"])
    boundary = torch.full((tokens.shape[0], 1), token_id, dtype=tokens.dtype, device=tokens.device)
    boundary_mask = torch.ones(
        (tokens.shape[0], 1), dtype=attention_mask.dtype, device=tokens.device
    )
    return (
        torch.cat((boundary, tokens), dim=1),
        torch.cat((boundary_mask, attention_mask), dim=1),
        1,
    )


@dataclass
class ICAFeatureConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    architecture: str = "icalens_profile_oriented_signed"
    activation_fn_str: str = "identity"
    dtype: str = "float32"
    device: str = "cuda"
    random_seed: int | None = None


class ICAFeatureEncoder(torch.nn.Module):
    """Expose one signed, profile-oriented feature per ICA component."""

    center: torch.Tensor
    reading: torch.Tensor
    tail_signs: torch.Tensor
    decoder_norms: torch.Tensor

    def __init__(self, snapshot: dict[str, Any], *, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        tensors = load_file(snapshot["layer_file"], device="cpu")
        center = tensors["center"].to(torch.float32)
        reading = tensors["reading_matrix"].to(torch.float32)
        writing = tensors["writing_matrix"].to(torch.float32)
        tail_signs = tensors["tail_signs"].to(torch.float32)
        if tail_signs.shape != (reading.shape[0],) or not bool(
            torch.all((tail_signs == 1) | (tail_signs == -1))
        ):
            raise ValueError("tail_signs must contain one +1 or -1 per ICA component")
        self.register_buffer("center", center)
        self.register_buffer("reading", reading)
        self.register_buffer("tail_signs", tail_signs)
        decoder = writing.T * tail_signs[:, None]
        decoder_norms = torch.linalg.vector_norm(decoder, dim=-1).clamp_min(
            float(snapshot["norm_eps"])
        )
        decoder = decoder / decoder_norms[:, None]
        self.register_buffer("decoder_norms", decoder_norms)
        self.W_dec = torch.nn.Parameter(decoder, requires_grad=False)
        self.dtype = dtype
        self.device = torch.device(device)
        self.row_normalize = bool(snapshot["row_normalize"])
        self.norm_eps = float(snapshot["norm_eps"])
        components = int(reading.shape[0])
        self.cfg = ICAFeatureConfig(
            model_name=str(snapshot["saebench_model_name"]),
            d_in=int(reading.shape[1]),
            d_sae=components,
            hook_layer=int(snapshot["layer"]),
            hook_name=f"blocks.{int(snapshot['layer'])}.hook_resid_post",
            dtype=str(dtype).removeprefix("torch."),
            device=device,
        )
        self.to(device=self.device, dtype=dtype)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        work = activations.to(device=self.device, dtype=self.dtype)
        if self.row_normalize:
            work = work / torch.linalg.vector_norm(work, dim=-1, keepdim=True).clamp_min(
                self.norm_eps
            )
        scores = (work - self.center) @ self.reading.T
        signed_scores = scores * self.tail_signs
        # W_dec is unit-normalized for SAEBench. Move the removed writing-vector
        # norms into the corresponding feature activations so the represented
        # residual contribution remains exactly unchanged.
        return signed_scores * self.decoder_norms


class PCAFeatureEncoder(torch.nn.Module):
    """Recover the fitted PCA basis from a full-rank ICA whitening transform."""

    center: torch.Tensor
    components: torch.Tensor

    def __init__(self, snapshot: dict[str, Any], *, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        tensors = load_file(snapshot["layer_file"], device="cpu")
        center = tensors["center"].to(torch.float32)
        writing = tensors["writing_matrix"].to(torch.float64)
        if writing.shape != (center.numel(), center.numel()):
            raise ValueError(
                "PCA recovery requires a full-rank ICA Lens layer; "
                f"got writing matrix {tuple(writing.shape)} for hidden size {center.numel()}"
            )
        covariance = writing @ writing.T
        _, eigenvectors = torch.linalg.eigh(covariance)
        components = eigenvectors.flip(1).T.contiguous().to(torch.float32)
        self.register_buffer("center", center)
        self.register_buffer("components", components)
        decoder = components
        self.W_dec = torch.nn.Parameter(decoder, requires_grad=False)
        self.dtype = dtype
        self.device = torch.device(device)
        self.row_normalize = bool(snapshot["row_normalize"])
        self.norm_eps = float(snapshot["norm_eps"])
        hidden_size = int(components.shape[1])
        self.cfg = ICAFeatureConfig(
            model_name=str(snapshot["saebench_model_name"]),
            d_in=hidden_size,
            d_sae=hidden_size,
            hook_layer=int(snapshot["layer"]),
            hook_name=f"blocks.{int(snapshot['layer'])}.hook_resid_post",
            architecture="pca_signed",
            activation_fn_str="identity",
            dtype=str(dtype).removeprefix("torch."),
            device=device,
        )
        self.to(device=self.device, dtype=dtype)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        work = activations.to(device=self.device, dtype=self.dtype)
        if self.row_normalize:
            work = work / torch.linalg.vector_norm(work, dim=-1, keepdim=True).clamp_min(
                self.norm_eps
            )
        scores = (work - self.center) @ self.components.T
        return scores


class RandomFeatureEncoder(torch.nn.Module):
    """A seeded full-rank random orthogonal basis with ICA-matched preprocessing."""

    center: torch.Tensor
    components: torch.Tensor

    def __init__(self, snapshot: dict[str, Any], *, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        tensors = load_file(snapshot["layer_file"], device="cpu")
        center = tensors["center"].to(torch.float32)
        hidden_size = int(center.numel())
        baseline = snapshot["baselines"]["random"]
        seed = int(baseline.get("seed", 0)) + int(snapshot["layer"])
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        matrix = torch.randn(
            (hidden_size, hidden_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        basis, triangular = torch.linalg.qr(matrix)
        diagonal = torch.diagonal(triangular)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        components = (basis * signs).T.contiguous()
        del matrix, basis, triangular
        self.register_buffer("center", center)
        self.register_buffer("components", components)
        decoder = components
        self.W_dec = torch.nn.Parameter(decoder, requires_grad=False)
        self.dtype = dtype
        self.device = torch.device(device)
        self.row_normalize = bool(snapshot["row_normalize"])
        self.norm_eps = float(snapshot["norm_eps"])
        self.cfg = ICAFeatureConfig(
            model_name=str(snapshot["saebench_model_name"]),
            d_in=hidden_size,
            d_sae=hidden_size,
            hook_layer=int(snapshot["layer"]),
            hook_name=f"blocks.{int(snapshot['layer'])}.hook_resid_post",
            architecture="random_orthogonal_signed",
            activation_fn_str="identity",
            dtype=str(dtype).removeprefix("torch."),
            device=device,
            random_seed=seed,
        )
        self.to(device=self.device, dtype=dtype)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        work = activations.to(device=self.device, dtype=self.dtype)
        if self.row_normalize:
            work = work / torch.linalg.vector_norm(work, dim=-1, keepdim=True).clamp_min(
                self.norm_eps
            )
        scores = (work - self.center) @ self.components.T
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saebench-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--progress-initial", type=int, default=0)
    parser.add_argument("--progress-total", type=int, default=1)
    parser.add_argument("--progress-run-initial", type=int, default=0)
    parser.add_argument("--progress-started-at", type=float, default=None)
    parser.add_argument("--methods", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.saebench_root))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    selected_methods = [name for name in args.methods.split(",") if name]

    # TransformerLens 2.11 reads this removed module attribute while SAEBench is
    # imported. Its model loader is replaced below, but legacy imports still need
    # a valid cache path under Transformers 5.
    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE  # type: ignore[attr-defined]

    import sae_bench.evals.sparse_probing.main as sparse_main  # type: ignore[import-not-found]
    from sae_bench.evals.sparse_probing.eval_config import (  # type: ignore[import-not-found]
        SparseProbingEvalConfig,
    )
    from sae_bench.sae_bench_utils import activation_collection  # type: ignore[import-not-found]

    # This pinned commit annotates its duck-typed model and SAE inputs as concrete
    # TransformerLens/SAELens classes.  Our adapters intentionally implement only
    # the operations sparse probing uses, so remove the legacy runtime wrappers at
    # this integration boundary without changing the benchmark implementation.
    for name in (
        "get_all_llm_activations",
        "get_llm_activations",
        "get_sae_meaned_activations",
    ):
        function = _unwrap_runtime_types(getattr(activation_collection, name))
        setattr(activation_collection, name, function)

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
            del device, dtype
            return HFHookedModel(model, tokenizer, snapshot["evaluation_input_protocol"])

    sparse_main.HookedTransformer = Factory
    encoders: dict[str, torch.nn.Module] = {}
    feature_configs: dict[str, dict[str, Any]] = {}
    if "ica" in selected_methods:
        encoder = ICAFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        encoders["ica"] = encoder
        feature_configs["ica"] = asdict(encoder.cfg)
    if "sae" in selected_methods and "sae" in snapshot.get("baselines", {}):
        sae_encoder = SAEFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        encoders["sae"] = sae_encoder
        feature_configs["sae"] = asdict(sae_encoder.cfg)
    if "pca" in selected_methods and "pca" in snapshot.get("baselines", {}):
        pca_encoder = PCAFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        encoders["pca"] = pca_encoder
        feature_configs["pca"] = asdict(pca_encoder.cfg)
    if "random" in selected_methods and "random" in snapshot.get("baselines", {}):
        random_encoder = RandomFeatureEncoder(snapshot, device="cuda", dtype=dtype)
        encoders["random"] = random_encoder
        feature_configs["random"] = asdict(random_encoder.cfg)
    config = SparseProbingEvalConfig(model_name=snapshot["saebench_model_name"])
    config.dataset_names = list(settings["datasets"])
    config.k_values = list(settings["k_values"])
    config.probe_train_set_size = int(settings["probe_train_size"])
    config.probe_test_set_size = int(settings["probe_test_size"])
    config.context_length = int(settings["context_length"])
    config.random_seed = int(settings["random_seed"])
    config.llm_batch_size = int(settings["llm_batch_size"])
    config.sae_batch_size = int(settings["sae_batch_size"])
    config.llm_dtype = str(settings["llm_dtype"])
    config.lower_vram_usage = False

    args.output.mkdir(parents=True, exist_ok=True)
    unknown_methods = sorted(set(selected_methods).difference(encoders))
    if unknown_methods:
        raise ValueError(f"requested methods are unavailable: {unknown_methods}")

    datasets = list(settings["datasets"])
    dataset_root = args.output / "saebench-datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    existing_tasks = sum(
        _dataset_result_path(dataset_root, index, dataset, method).is_file()
        for index, dataset in enumerate(datasets)
        for method in selected_methods
    )
    original_tqdm = sparse_main.tqdm
    display = _BenchmarkDisplay(
        output=args.output,
        completed=args.progress_initial + existing_tasks,
        total=args.progress_total,
        run_initial=args.progress_run_initial,
        run_started_at=args.progress_started_at or time.time(),
    )
    sparse_main.tqdm = display.track_methods
    try:
        with display:
            for index, dataset in enumerate(datasets):
                dataset_output = _dataset_output(dataset_root, index, dataset)
                missing = [
                    method
                    for method in selected_methods
                    if not _dataset_result_path(dataset_root, index, dataset, method).is_file()
                ]
                display.set_dataset(
                    dataset,
                    index=index + 1,
                    total=len(datasets),
                    methods=selected_methods,
                    completed_methods=set(selected_methods).difference(missing),
                )
                dataset_artifacts = args.artifacts / f"dataset_{index:02d}"
                if not missing:
                    # A previous run may have committed every method result but
                    # stopped before removing its temporary activation cache.
                    _remove_dataset_artifacts(dataset_artifacts)
                    continue
                config.dataset_names = [dataset]
                # Dataset result JSON files are the resume checkpoints. Raw
                # activations remain available if evaluation fails, allowing a
                # resumed run to reuse them. Once every missing method returns
                # successfully, the result files are durable and the complete
                # per-dataset cache can be removed.
                result = sparse_main.run_eval(
                    config,
                    selected_saes=[(name, encoders[name]) for name in missing],
                    device="cuda",
                    output_path=str(dataset_output),
                    force_rerun=False,
                    # SAEBench only removes the final hook directory when
                    # methods use different hook names. ICALens therefore
                    # owns cleanup of the complete per-dataset directory.
                    clean_up_activations=False,
                    save_activations=len(missing) > 1,
                    artifacts_path=str(dataset_artifacts),
                )
                del result
                _remove_dataset_artifacts(dataset_artifacts)
    finally:
        sparse_main.tqdm = original_tqdm
    config.dataset_names = datasets
    final_output = args.output / "saebench"
    final_output.mkdir(parents=True, exist_ok=True)
    for method in selected_methods:
        payloads = [
            json.loads(
                _dataset_result_path(dataset_root, index, dataset, method).read_text(
                    encoding="utf-8"
                )
            )
            for index, dataset in enumerate(datasets)
        ]
        merged = _merge_dataset_results(payloads, datasets)
        (final_output / f"{method}_custom_sae_eval_results.json").write_text(
            json.dumps(merged, indent=2, sort_keys=False, default=str) + "\n",
            encoding="utf-8",
        )
    methods: dict[str, Any] = {}
    available_methods = ["ica", *snapshot.get("baselines", {})]
    for name in available_methods:
        result_path = args.output / "saebench" / f"{name}_custom_sae_eval_results.json"
        if name in selected_methods and not result_path.is_file():
            raise RuntimeError(f"SAEBench produced no {name} result at {result_path}")
        if result_path.is_file():
            methods[name] = json.loads(result_path.read_text(encoding="utf-8"))
    (args.output / "raw-result.json").write_text(
        json.dumps({"methods": methods}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output / "worker.json").write_text(
        json.dumps(
            {
                "feature_configs": feature_configs,
                "preset": settings,
                "baselines": snapshot.get("baselines", {}),
                "execution_order": "dataset_first",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _dataset_output(root: Path, index: int, dataset: str) -> Path:
    safe_name = dataset.replace("/", "__")
    return root / f"{index:02d}_{safe_name}"


def _remove_dataset_artifacts(path: Path) -> None:
    """Remove one dataset's temporary shared activation cache."""
    if path.exists():
        shutil.rmtree(path)


def _dataset_result_path(root: Path, index: int, dataset: str, method: str) -> Path:
    return _dataset_output(root, index, dataset) / f"{method}_custom_sae_eval_results.json"


def _merge_dataset_results(payloads: list[dict[str, Any]], datasets: list[str]) -> dict[str, Any]:
    """Recreate SAEBench's unweighted cross-dataset sparse-probing result."""
    if len(payloads) != len(datasets) or not payloads:
        raise ValueError("one SAEBench payload is required for every dataset")
    merged = json.loads(json.dumps(payloads[0]))
    merged["eval_config"]["dataset_names"] = datasets
    merged["eval_result_details"] = []
    merged["eval_result_unstructured"] = {}
    for payload in payloads:
        details = payload.get("eval_result_details", [])
        if len(details) != 1:
            raise ValueError("dataset-first SAEBench output must contain one result detail")
        merged["eval_result_details"].append(details[0])
        merged["eval_result_unstructured"].update(payload.get("eval_result_unstructured", {}))
    categories = set.intersection(*(set(payload["eval_result_metrics"]) for payload in payloads))
    merged_metrics: dict[str, dict[str, float | None]] = {}
    for category in categories:
        metric_names = set.intersection(
            *(set(payload["eval_result_metrics"][category]) for payload in payloads)
        )
        merged_metrics[category] = {
            metric: _average_optional_metric(
                [payload["eval_result_metrics"][category][metric] for payload in payloads]
            )
            for metric in metric_names
        }
    merged["eval_result_metrics"] = merged_metrics
    return cast(dict[str, Any], merged)


def _average_optional_metric(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    if len(numeric) != len(values):
        raise ValueError("SAEBench metric is missing from only some datasets")
    return sum(numeric) / len(numeric)


def _unwrap_runtime_types(function: Any) -> Any:
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


if __name__ == "__main__":
    main()
