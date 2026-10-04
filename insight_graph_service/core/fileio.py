"""Atomic file replacement that survives Windows sharing violations (docs/06_graph_and_storage.md §6.3).

On Windows a file cannot be renamed over while another handle has it open (Python opens files
without delete sharing), and a file cannot be opened while it is being renamed over: both fail
with ``PermissionError`` (WinError 5 / 32). The console polls the batch records while the worker
rewrites them, so both sides retry briefly. Elsewhere a ``PermissionError`` is real and raised
at once. Standard library only.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

SHARING_RETRY_S = 2.0  # readers hold a record for microseconds; this only bounds a real permission error


def retry_sharing[T](action: Callable[[], T], timeout: float = SHARING_RETRY_S) -> T:
    """Run ``action()``; on Windows retry a ``PermissionError`` with backoff until ``timeout`` seconds."""
    if os.name != "nt":
        return action()
    delay, deadline = 0.002, time.monotonic() + timeout
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 0.05)


def replace_file(src: str | Path, dst: str | Path) -> None:
    """``os.replace`` that waits out readers of ``dst``; the temporary ``src`` is removed if it fails."""
    try:
        retry_sharing(lambda: os.replace(src, dst))
    except BaseException:
        Path(src).unlink(missing_ok=True)
        raise
