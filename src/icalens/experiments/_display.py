"""Generic compact display and complete logging for long experiment commands."""

from __future__ import annotations

import io
import re
import shlex
import sys
import threading
import time
from collections import deque
from collections.abc import Hashable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class _CapturedOutput(io.TextIOBase):
    def __init__(self, display: ExperimentDisplay) -> None:
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


class ExperimentDisplay:
    """Render policy-compliant progress while retaining complete command output."""

    def __init__(
        self,
        *,
        output: Path,
        title: str,
        completed: int,
        total: int,
        source_dirty: bool | None = False,
        detail_filename: str = "experiment-detail.log",
        recent_label: str = "Recent output",
        command: str | Sequence[str] | None = None,
        unit_label: str = "units",
        started_at: float | None = None,
        completed_unit_ids: Iterable[Hashable] | None = None,
    ) -> None:
        if total < 0 or completed < 0 or completed > total:
            raise ValueError("progress must satisfy 0 <= completed <= total")
        self.total = total
        self.completed = completed
        self.initial_completed = completed
        self.started_at = time.time() if started_at is None else started_at
        self.title = title
        self.source_dirty = source_dirty
        self.recent_label = recent_label
        self.phase_name = "Preparing"
        self.context: dict[str, str] = {}
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
        self.stream = _CapturedOutput(self)
        self.live: Live | None = None
        self.thread: threading.Thread | None = None
        if command is None:
            self.command = shlex.join(sys.argv)
        elif isinstance(command, str):
            self.command = command
        else:
            self.command = shlex.join(command)
        self.unit_label = unit_label
        self.completed_unit_ids = set(completed_unit_ids or ())
        if self.completed_unit_ids and len(self.completed_unit_ids) != completed:
            raise ValueError("completed_unit_ids must contain exactly completed unique IDs")
        self.outcome = "complete"

    def __enter__(self) -> ExperimentDisplay:
        self.detail.write(
            "# ICA Lens experiment run\n"
            f"started_at: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"working_directory: {Path.cwd()}\n"
            f"command: {self.command}\n"
            f"initial_completed: {self.initial_completed}\n"
            f"total: {self.total}\n\n"
        )
        self.detail.flush()
        if self.interactive:
            self.live = Live(self.render(), console=self.console, refresh_per_second=4)
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
            f"status: {'failed' if error is not None else self.outcome}\n"
            f"completed: {self.completed}/{self.total}\n"
        )
        self.detail.flush()
        self.detail.close()
        print(f"Full output: {self.detail_path}")

    def phase(self, name: str, **context: Any) -> None:
        """Set the current phase and its optional user-facing context labels."""
        with self.lock:
            self.phase_name = str(name)
            self.context = {str(key): str(value) for key, value in context.items()}
        self.refresh()

    def set_outcome(self, outcome: str) -> None:
        """Set the final non-exception outcome written to the detail log."""
        self.outcome = outcome

    def advance(self, count: int = 1, *, refresh: bool = False) -> None:
        """Record newly durable work units."""
        if count < 0:
            raise ValueError("progress count must be nonnegative")
        with self.lock:
            if self.completed + count > self.total:
                raise ValueError("progress would exceed the configured total")
            self.completed += count
        if refresh:
            self.refresh()

    def complete_unit(self, unit_id: Hashable, *, refresh: bool = False) -> bool:
        """Record one named durable unit once; return whether it was newly completed."""
        with self.lock:
            if unit_id in self.completed_unit_ids:
                return False
            if self.completed + 1 > self.total:
                raise ValueError("progress would exceed the configured total")
            self.completed_unit_ids.add(unit_id)
            self.completed += 1
        if refresh:
            self.refresh()
        return True

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

    def render(self) -> Panel | Group:
        with self.lock:
            completed = self.completed
            total = self.total
            phase = self.phase_name
            context = dict(self.context)
            recent = list(self.recent)
            elapsed = max(0.0, time.time() - self.started_at)
        completed_now = completed - self.initial_completed
        remaining = total - completed
        if completed_now > 0:
            eta = elapsed * remaining / completed_now
            timing = f"elapsed {_duration(elapsed)} · ETA ~{_duration(eta)}"
        else:
            timing = f"elapsed {_duration(elapsed)} · ETA estimating…"
        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        percentage = 100.0 * completed / max(total, 1)
        header.add_row(
            Text(
                f"Overall {completed}/{total} {self.unit_label} ({percentage:.1f}%)",
                style="bold",
            ),
            Text(timing, style="cyan"),
        )
        header.add_row(ProgressBar(total=max(total, 1), completed=completed, width=48), Text(""))
        current = Text.assemble(("Phase: ", "bold"), phase)
        for key, value in context.items():
            current.append(f"    {key.replace('_', ' ').title()}: ", style="bold")
            current.append(value)
        tail = Text("\n".join(recent) if recent else "Waiting for output…", style="dim")
        command = Panel(
            Text(self.command, style="cyan"),
            title="Command",
            title_align="left",
            border_style="dim",
            padding=(0, 1),
        )
        panel = Panel(
            Group(header, current, command, Text(self.recent_label, style="bold"), tail),
            title=self.title,
            border_style="blue",
        )
        if not self.source_dirty:
            return panel
        warning = Panel(
            Text(
                "WARNING: uncommitted source; the recorded commit alone cannot reproduce this run.",
                style="bold bright_yellow",
            ),
            title="Source warning",
            border_style="bright_yellow",
        )
        return Group(panel, warning)


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
