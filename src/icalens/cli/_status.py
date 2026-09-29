"""Consistent lifecycle messages for long-running CLI operations."""

from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    """Print one timestamped, immediately flushed lifecycle message."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)
