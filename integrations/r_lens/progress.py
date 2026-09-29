"""Progress display for the upstream Jacobian-lens fitter."""

from __future__ import annotations

import logging
import re

from tqdm.auto import tqdm


class _JacobianProgressHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._bar: tqdm | None = None
        self._completed = 0

    def emit(self, record: logging.LogRecord) -> None:
        match = re.search(r"pass (\d+)/(\d+)", record.getMessage())
        if match is None:
            return
        completed, total = (int(value) for value in match.groups())
        if self._bar is None or self._bar.total != total or completed < self._completed:
            self._close_bar()
            self._bar = tqdm(total=total, desc="R-lens backward", unit="pass")
            self._completed = 0
        self._bar.update(max(0, completed - self._completed))
        self._completed = completed
        if completed >= total:
            self._close_bar()

    def close(self) -> None:
        self._close_bar()
        super().close()

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
        self._bar = None
        self._completed = 0


def configure_jlens_progress() -> None:
    """Show upstream debug pass checkpoints as a tqdm progress bar."""
    logger = logging.getLogger("jlens.fitting")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    info = logging.StreamHandler()
    info.setLevel(logging.INFO)
    info.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(info)
    logger.addHandler(_JacobianProgressHandler())
