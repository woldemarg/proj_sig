"""Shared helpers for the measurement scripts: throwaway workspaces under ``sig/.scratch/``.

A script imports it first: it puts the repository on ``sys.path`` (scripts run from ``scripts/``, the packages are not
installed). The configured settings (``.env``, as the service reads them) are loaded when a helper builds an engine
(``settings``), never at import, so the tests that import the scripts never load the repository's ``.env``.
Never touches ``workspace/`` (the user's knowledge base). Neo4j publishing is always off here.
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# scripts run from scripts/; the package lives one level up (not pip-installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insight_graph_service.core.demo import generate_retail_dataset  # noqa: E402
from insight_graph_service.core.engine import Engine  # noqa: E402
from insight_graph_service.core.settings import PROJECT_ROOT, Settings, load_env_file, load_settings  # noqa: E402

SCRATCH = PROJECT_ROOT / ".scratch"


def settings(**overrides: Any) -> Settings:
    """The configured settings (``.env`` loaded first, as the service does) with Neo4j off and ``overrides`` applied."""
    load_env_file()
    return load_settings(**{"neo4j_enabled": False, **overrides})


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """``--override NAME=VALUE`` arguments as settings fields: lower-case names, numbers as int or float."""
    overrides = {}
    for item in items:
        name, value = item.split("=", 1)
        overrides[name.lower()] = float(value) if "." in value else int(value)
    return overrides


def scratch_dir(name: str, *, fresh: bool = False) -> Path:
    """``.scratch/<name>/``, emptied first when ``fresh``."""
    path = SCRATCH / name
    if fresh:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_demo_csv(folder: Path) -> Path:
    """The synthetic retail dataset (``insight_graph_service.core.demo``) as a CSV inside ``folder``."""
    path = folder / "retail_synthetic_seed7.csv"
    if not path.exists():
        generate_retail_dataset(5000, 7).to_csv(path, index=False)
    return path


def weekly_retail(folder: Path, weeks: int = 6, spiked: tuple[str, ...] = ("discount",), rows: int = 5000, channel: str | None = None) -> Path:
    """``rows`` of the demo's orders over ``weeks`` ISO weeks from a Monday (``when``; its first ten rows without a
    readable time), the ``spiked`` metrics of phones in the US — those sold through ``channel`` only, when named — raised
    by 3 sd in the fourth week: a PEAKED_AT there (docs/02_discovery.md §2.10)."""
    frame = generate_retail_dataset(rows, seed=7)
    rng = np.random.default_rng(1)
    week = rng.integers(0, weeks, len(frame))
    when = pd.Timestamp("2031-03-03") + pd.to_timedelta(week * 7 * 86_400 + rng.uniform(0, 7 * 86_400, len(frame)), unit="s")
    frame.insert(0, "when", when.strftime("%Y-%m-%d %H:%M:%S"))
    spike = (frame["category"] == "phones") & (frame["region"] == "US") & (week == 3) & (frame["channel"] == channel if channel else True)
    for metric in spiked:
        frame.loc[spike, metric] += 3 * frame[metric].std()
    frame.loc[:9, "when"] = "unknown"
    path = folder / "weekly.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def build_engine(workspace: Path, *, embedder: Any = None, **overrides: Any) -> Engine:
    """Engine on a scratch workspace; ``embedder`` reuses a loaded model; ``overrides`` are settings fields."""
    return Engine(settings(workspace_dir=workspace, **overrides), embedder=embedder)


def ingest_ready(engine: Engine, path: Path) -> dict[str, Any]:
    """Run one batch; exits with the batch status and error unless it ends READY."""
    record = engine.ingest_file(path)
    if record["status"] != "READY":
        raise SystemExit(f"{path.name}: {record['status']} {record.get('error')}")
    return record


def demo_engine(name: str) -> Engine:
    """Fresh scratch workspace with the demo ingested."""
    root = scratch_dir(name, fresh=True)
    engine = build_engine(root / "ws")
    ingest_ready(engine, write_demo_csv(root))
    return engine


def save_result(name: str, label: str, payload: Any) -> Path:
    """Persist a measurement next to the scratch workspaces (``.scratch/results/<name>_<label>.json``)."""
    path = scratch_dir("results") / f"{name}_{label}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve(app, port: int | None = None) -> Iterator[str]:
    """Run an ASGI app on a local port (a free one by default) in a background thread; yields its base URL."""
    import uvicorn

    port = port or free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
