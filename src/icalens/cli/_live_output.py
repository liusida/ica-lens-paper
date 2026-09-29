"""Compact live terminal output for long-running ICA Lens CLI operations."""

from __future__ import annotations

import importlib.metadata
import io
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_FIT = re.compile(r"Fitting layer (\d+)")
_FIT_DONE = re.compile(r"Checkpointed layer (\d+)")
_PROFILE = re.compile(r"Profiling layer (\d+)")
_PROFILE_DONE = re.compile(r"Profiled layer (\d+)")
_PROFILE_REFRESH = re.compile(r"Refreshing score statistics for layer (\d+)")
_PROFILE_REFRESH_DONE = re.compile(r"Refreshed layer (\d+)")
_R_LENS_DONE = re.compile(r"Added R-lens readouts to layer (\d+)")
_CAPTURE = re.compile(r"Capturing layers ([0-9,]+)")
_CAPTURE_DONE = re.compile(r"Checkpointed activation layer (\d+)")
_REQUESTED = re.compile(r"Requested layers: ([0-9,]+)")
_DURABLE_COMPLETED = re.compile(r"Durable completed layers: ([0-9,]+|none)")


class _CapturedOutput(io.TextIOBase):
    def __init__(self, display: _OperationDisplay) -> None:
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


class _OperationDisplay:
    def __init__(
        self,
        *,
        title: str,
        detail_path: Path,
        command: str,
        source_dirty: bool | None,
        requested_layers: tuple[int, ...] | None,
        completed_layers: set[int],
    ) -> None:
        self.title = title
        self.detail_path = detail_path
        self.command = command
        self.source_dirty = source_dirty
        self.requested_layers = requested_layers
        self.layer_total = len(requested_layers) if requested_layers is not None else None
        self.current_layer: int | None = None
        self.current_layers: tuple[int, ...] = ()
        self.completed_layers = set(completed_layers)
        self.initial_completed = len(completed_layers)
        self.phase = "Preparing"
        self.recent: deque[str] = deque(maxlen=7)
        self.started_at = time.time()
        self.first_completion_at: float | None = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.original_out = sys.stdout
        self.original_err = sys.stderr
        self.console = Console(file=self.original_err, force_terminal=True)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        self.detail = detail_path.open("a", encoding="utf-8")
        self.stream = _CapturedOutput(self)
        self.live: Live | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> _OperationDisplay:
        self.detail.write(_run_header(self.command, source_dirty=self.source_dirty))
        self.detail.flush()
        self.live = Live(self.render(), console=self.console, refresh_per_second=4, transient=False)
        self.live.__enter__()
        self.thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.thread.start()
        sys.stdout = self.stream
        sys.stderr = self.stream
        warning = _dirty_warning(self.source_dirty)
        if warning is not None:
            from ._status import log

            log(warning)
        return self

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        sys.stdout = self.original_out
        sys.stderr = self.original_err
        if error is not None:
            with self.lock:
                self.phase = "Failed"
                self.recent.append(f"{type(error).__name__}: {error}"[-180:])
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.live is not None:
            self.live.update(self.render(), refresh=True)
            self.live.__exit__(error_type, error, traceback)
        self.detail.write(_run_footer(error))
        self.detail.flush()
        self.detail.close()
        print(f"Full output: {self.detail_path}")

    def capture(self, value: str) -> None:
        with self.lock:
            if self.detail.closed:
                return
            self.detail.write(value)
            self.detail.flush()
            clean = _ANSI.sub("", value)
            for raw in re.split(r"[\r\n]+", clean):
                line = raw.strip()
                if not line:
                    continue
                self._observe(line)
                self.recent.append(line[-180:])

    def _observe(self, line: str) -> None:
        requested = _REQUESTED.search(line)
        durable = _DURABLE_COMPLETED.search(line)
        fit = _FIT.search(line)
        profile = _PROFILE.search(line)
        refresh = _PROFILE_REFRESH.search(line)
        capture = _CAPTURE.search(line)
        completed = (
            _FIT_DONE.search(line)
            or _PROFILE_DONE.search(line)
            or _PROFILE_REFRESH_DONE.search(line)
            or _R_LENS_DONE.search(line)
            or _CAPTURE_DONE.search(line)
        )
        if requested:
            self.requested_layers = tuple(int(value) for value in requested.group(1).split(","))
            self.layer_total = len(self.requested_layers)
        if durable:
            values = durable.group(1)
            discovered = set() if values == "none" else {int(value) for value in values.split(",")}
            if self.requested_layers is not None:
                discovered &= set(self.requested_layers)
            self.completed_layers = discovered
            self.initial_completed = len(discovered)
        if fit:
            self.current_layer = int(fit.group(1))
            self.current_layers = (self.current_layer,)
            self.phase = "Fitting"
        elif profile:
            self.current_layer = int(profile.group(1))
            self.current_layers = (self.current_layer,)
            self.phase = "Profiling"
        elif refresh:
            self.current_layer = int(refresh.group(1))
            self.current_layers = (self.current_layer,)
            self.phase = "Refreshing statistics"
        elif capture:
            self.current_layers = tuple(int(value) for value in capture.group(1).split(","))
            self.current_layer = self.current_layers[0] if len(self.current_layers) == 1 else None
            self.phase = "Capturing activations"
        elif completed:
            layer = int(completed.group(1))
            if layer not in self.completed_layers and self.first_completion_at is None:
                self.first_completion_at = time.time()
            self.completed_layers.add(layer)
            self.current_layers = tuple(value for value in self.current_layers if value != layer)
            self.current_layer = self.current_layers[0] if len(self.current_layers) == 1 else None
            self.phase = "Checkpointing"
        elif "Captur" in line and "activation" in line:
            self.phase = "Capturing activations"
        elif "Tokenize" in line or "candidate token" in line:
            self.phase = "Tokenizing dataset"
        elif "Loading weights" in line or "Loading " in line and "CUDA" in line:
            self.phase = "Loading model"
        elif "Profiling logit-lens" in line:
            self.phase = "Logit Lens"
        elif "Profile cached activations" in line or "Profile components" in line:
            self.phase = "Profiling components"
        elif "Saved " in line or "Experiment complete" in line:
            self.phase = "Complete"

    def flush(self) -> None:
        with self.lock:
            if not self.detail.closed:
                self.detail.flush()

    def _refresh_loop(self) -> None:
        while not self.stop.wait(1):
            if self.live is not None:
                self.live.update(self.render(), refresh=True)

    def render(self) -> Panel:
        with self.lock:
            phase = self.phase
            current = self.current_layer
            current_layers = self.current_layers
            completed = len(self.completed_layers)
            newly_completed = max(0, completed - self.initial_completed)
            recent = list(self.recent)
        elapsed_seconds = time.time() - self.started_at
        elapsed = _duration(elapsed_seconds)
        eta = _eta(
            elapsed_seconds=elapsed_seconds,
            newly_completed=newly_completed,
            remaining=max(0, (self.layer_total or completed) - completed),
        )
        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        if len(current_layers) > 1:
            layer_text = " · layers " + ",".join(map(str, current_layers))
        else:
            layer_text = f" · layer {current}" if current is not None else ""
        timing = f"elapsed {elapsed}" + (f" · ETA ~{eta}" if eta is not None else "")
        if self.layer_total is not None:
            percent = 100 * completed / self.layer_total if self.layer_total else 100
            header.add_row(
                Text(
                    f"Overall {completed}/{self.layer_total} layers ({percent:.1f}%)",
                    style="bold",
                ),
                Text(timing, style="cyan"),
            )
            header.add_row(
                ProgressBar(total=self.layer_total, completed=completed, width=48),
                Text(""),
            )
            header.add_row(Text(f"{phase}{layer_text}", style="bold"), Text(""))
        elif completed:
            header.add_row(
                Text(f"Completed layers: {completed}", style="green"), Text(timing, style="cyan")
            )
            header.add_row(Text(f"{phase}{layer_text}", style="bold"), Text(""))
        else:
            header.add_row(Text(f"{phase}{layer_text}", style="bold"), Text(timing, style="cyan"))
        tail = Text("\n".join(recent) if recent else "Waiting for output…", style="dim")
        command = Panel(
            Text(self.command, style="cyan"),
            title="Command",
            title_align="left",
            border_style="dim",
            padding=(0, 1),
        )
        sections: list[Any] = [header, command]
        warning = _dirty_warning(self.source_dirty)
        if warning is not None:
            warning_text = Text(warning)
            warning_text.stylize("bold bright_yellow", 0, len("WARNING"))
            sections.append(
                Panel(
                    warning_text,
                    title="Source warning",
                    title_align="left",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        sections.extend((Text("Recent output", style="bold"), tail))
        return Panel(
            Group(*sections),
            title=self.title,
            border_style="blue",
        )


@contextmanager
def compact_output(title: str, args: list[str]) -> Iterator[None]:
    """Use a compact live panel on terminals and ordinary output elsewhere."""
    if any(value in {"-h", "--help"} for value in args):
        yield
        return
    source_dirty = _source_worktree_dirty()
    if not sys.stderr.isatty():
        warning = _dirty_warning(source_dirty)
        if warning is not None:
            from ._status import log

            log(warning)
        yield
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    detail = Path.cwd() / ".icalens-runs" / f"{timestamp}-{slug}.log"
    requested, completed = _layer_progress(args, title=title)
    with _OperationDisplay(
        title=title,
        detail_path=detail,
        command=_portable_command(title, args),
        source_dirty=source_dirty,
        requested_layers=requested,
        completed_layers=completed,
    ):
        yield


def _portable_command(title: str, args: list[str]) -> str:
    prefixes = {
        "ICA Lens · component profiling": ("icalens", "profile"),
        "ICA Lens · text fitting": ("icalens", "fit", "text"),
        "ICA Lens · chat fitting": ("icalens", "fit", "chat"),
        "ICA Lens · activation fitting": ("icalens", "fit", "activations"),
        "ICA Lens · text activation capture": ("icalens", "capture", "text"),
        "ICA Lens · chat activation capture": ("icalens", "capture", "chat"),
    }
    prefix = prefixes.get(title, ("icalens",))
    return shlex.join([*prefix, *args])


def _run_header(command: str, *, source_dirty: bool | None = None) -> str:
    try:
        package_version = importlib.metadata.version("icalens")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    return (
        "# ICA Lens run\n"
        f"started_at: {started}\n"
        f"working_directory: {Path.cwd()}\n"
        f"icalens_version: {package_version}\n"
        f"source_worktree_dirty: {_format_optional_bool(source_dirty)}\n"
        f"command: {command}\n\n"
    )


def _source_worktree_dirty() -> bool | None:
    """Return whether the checkout containing this package has local changes."""
    package_dir = Path(__file__).resolve().parents[1]
    try:
        root = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if root.returncode != 0:
            return None
        status = subprocess.run(
            [
                "git",
                "-C",
                root.stdout.strip(),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if status.returncode != 0:
        return None
    return bool(status.stdout.strip())


def _dirty_warning(source_dirty: bool | None) -> str | None:
    if source_dirty:
        return (
            "WARNING: The ICALens Git worktree has uncommitted changes. "
            "The exact source for this run cannot be reproduced from a commit alone."
        )
    return None


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if value else "false"


def _run_footer(error: Any) -> str:
    ended = datetime.now().astimezone().isoformat(timespec="seconds")
    status = "failed" if error is not None else "complete"
    return f"\n# ICA Lens run ended\nended_at: {ended}\nstatus: {status}\n"


def _layer_progress(args: list[str], *, title: str) -> tuple[tuple[int, ...] | None, set[int]]:
    requested = _requested_layers(args)
    if requested is None:
        return None, set()
    # A forced run deliberately replaces the requested durable units, so its
    # progress begins at zero even if prior artifacts are present.
    profile_suboperation = any(
        operation in args for operation in ("add-r-lens", "refresh-statistics", "refresh-examples")
    )
    completed = (
        set() if "--force" in args or profile_suboperation else _completed_layers(args, title=title)
    )
    return requested, set(requested) & completed


def _requested_layers(args: list[str]) -> tuple[int, ...] | None:
    for index, value in enumerate(args):
        if value == "--layer" and index + 1 < len(args):
            return (int(args[index + 1]),)
        if value == "--layers" and index + 1 < len(args):
            requested = args[index + 1].strip()
            if requested.lower() == "all":
                return _layers_from_artifact(args)
            else:
                return tuple(
                    sorted({int(part.strip()) for part in requested.split(",") if part.strip()})
                )
        if value.startswith("--layers="):
            requested = value.partition("=")[2].strip()
            if requested.lower() == "all":
                return _layers_from_artifact(args)
            else:
                return tuple(
                    sorted({int(part.strip()) for part in requested.split(",") if part.strip()})
                )
    return None


def _layers_from_artifact(args: list[str]) -> tuple[int, ...] | None:
    for option, manifest_name in (
        ("--input", "activations.json"),
        ("--activations", "activations.json"),
        ("--lens", "icalens.json"),
        ("--output", "activations.json"),
    ):
        value = _option_value(args, option)
        if value is None:
            continue
        manifest = Path(value).expanduser() / manifest_name
        if not manifest.is_file():
            continue
        try:
            layers = json.loads(manifest.read_text())["layers"]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(layers, dict):
            return tuple(sorted(int(layer) for layer in layers))
    return None


def _completed_layers(args: list[str], *, title: str) -> set[int]:
    """Read durable completed units without importing the command implementation."""
    if "profiling" in title:
        source = _option_value(args, "--output") or _option_value(args, "--lens")
        manifest_name = "icalens.json"
        completion_field = "component_profile"
    elif "capture" in title:
        source = _option_value(args, "--output")
        manifest_name = "activations.json"
        completion_field = "status"
    else:
        source = _option_value(args, "--output")
        manifest_name = "icalens.json"
        completion_field = "file"
    if source is None:
        return set()
    manifest = Path(source).expanduser() / manifest_name
    if not manifest.is_file():
        return set()
    try:
        layers = json.loads(manifest.read_text())["layers"]
        return {
            int(layer)
            for layer, entry in layers.items()
            if isinstance(entry, dict)
            and (
                entry.get(completion_field) == "complete"
                if completion_field == "status"
                else bool(entry.get(completion_field))
            )
        }
    except (KeyError, TypeError, ValueError, OSError):
        return set()


def _option_value(args: list[str], option: str) -> str | None:
    for index, value in enumerate(args):
        if value == option and index + 1 < len(args):
            return args[index + 1]
        prefix = option + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _eta(*, elapsed_seconds: float, newly_completed: int, remaining: int) -> str | None:
    """Estimate only from work completed in this invocation, never reused work."""
    if newly_completed <= 0 or remaining <= 0:
        return None
    return _duration(elapsed_seconds / newly_completed * remaining)
