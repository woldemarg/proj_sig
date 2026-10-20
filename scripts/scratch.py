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

# scripts run from scripts/; the package lives one level up (not pip-installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insight_graph_service.core.demo import generate_retail_dataset  # noqa: E402
from insight_graph_service.core.engine import Engine  # noqa: E402
from insight_graph_service.core.settings import PROJECT_ROOT, Settings, load_env_file, load_settings  # noqa: E402

SCRATCH = PROJECT_ROOT / ".scratch"


def settings(**overrides: Any) -> Settings:
    """The configured settings (``.env`` loaded first, as the service does) with Neo4j off and ``overrides`` applied."""
    load_env_file()
    return load_settings(neo4j_enabled=False, **overrides)


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
