"""Shared resumable-run metadata and atomic checkpoint helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

Normalizer = Callable[[dict[str, Any], dict[str, Any]], None]


class ResumableRun:
    """Validate one output-bound run and update its metadata atomically."""

    def __init__(self, path: Path, state: dict[str, Any]) -> None:
        self.path = path
        self.state = state

    @classmethod
    def open(
        cls,
        *,
        output: Path,
        resolved: dict[str, Any],
        source: dict[str, Any],
        status: str,
        filename: str = "run.json",
        normalize_previous: Normalizer | None = None,
    ) -> ResumableRun:
        output.mkdir(parents=True, exist_ok=True)
        path = output / filename
        if path.is_file():
            state = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
            previous = state.get("resolved")
            if not isinstance(previous, dict):
                raise ValueError(f"{path} has no valid resolved configuration")
            if normalize_previous is not None:
                normalize_previous(previous, resolved)
            differences = configuration_differences(previous, resolved)
            if differences:
                raise ValueError(
                    f"{path} belongs to a different configuration "
                    f"({'; '.join(differences)}); choose another output"
                )
        else:
            state = {
                "started_at": _timestamp(),
                "completed_at": None,
                "resolved": resolved,
            }
        state["resolved"] = resolved
        state["icalens_source"] = source
        state["status"] = status
        if status != "complete":
            state["completed_at"] = None
        run = cls(path, state)
        run.save()
        return run

    def validate_section(self, name: str, resolved: dict[str, Any]) -> None:
        """Create or validate an independently resumable configuration section."""
        previous = self.state.get(name)
        if previous is not None:
            if not isinstance(previous, dict):
                raise ValueError(f"{self.path} field {name!r} is not an object")
            differences = configuration_differences(previous, resolved, prefix=name)
            if differences:
                raise ValueError(
                    f"{self.path} belongs to a different {name} configuration "
                    f"({'; '.join(differences)}); choose another output"
                )
        self.state[name] = resolved
        self.save()

    def set_status(self, status: str, *, complete: bool = False) -> None:
        self.state["status"] = status
        self.state["completed_at"] = _timestamp() if complete else None
        self.save()

    def update(self, **fields: Any) -> None:
        """Atomically update lifecycle metadata without changing run identity."""
        self.state.update(fields)
        self.save()

    def save(self) -> None:
        atomic_write_json(self.path, self.state)


def configuration_differences(
    previous: Any, current: Any, *, prefix: str = "resolved"
) -> list[str]:
    """Return readable recursive differences between two JSON-like values."""
    if isinstance(previous, dict) and isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}"
            if key not in previous:
                differences.append(f"{path} was added")
            elif key not in current:
                differences.append(f"{path} was removed")
            else:
                differences.extend(
                    configuration_differences(previous[key], current[key], prefix=path)
                )
        return differences
    if previous != current:
        return [f"{prefix}: {previous!r} -> {current!r}"]
    return []


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON by validated same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Parse before replacement so a programming or serialization error cannot
    # replace the last good metadata file.
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
