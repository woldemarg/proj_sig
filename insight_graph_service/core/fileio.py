"""Atomic file replacement (docs/06_graph_and_storage.md §6.3). Standard library only."""

from __future__ import annotations

import os
from pathlib import Path


def replace_file(src: str | Path, dst: str | Path) -> None:
    """``os.replace``; the temporary ``src`` is removed if it fails."""
    try:
        os.replace(src, dst)
    except BaseException:
        Path(src).unlink(missing_ok=True)
        raise
