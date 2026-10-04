"""Shared helpers for the measurement scripts: throwaway workspaces under ``sig/.scratch/``.

Never touches ``workspace/`` (the user's knowledge base). Neo4j publishing is always off here.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

# scripts run from scripts/; the package lives one level up (not pip-installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insight_graph_service.core.demo import generate_retail_dataset  # noqa: E402
from insight_graph_service.core.engine import Engine  # noqa: E402
from insight_graph_service.core.settings import PROJECT_ROOT, load_env_file, load_settings  # noqa: E402

load_env_file()  # the measurements use the configured settings (.env), like the service
SCRATCH = PROJECT_ROOT / ".scratch"


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


def build_engine(workspace: Path, *, embedder: Any = None) -> Engine:
    """Engine on a scratch workspace; ``embedder`` reuses a loaded model."""
    return Engine(load_settings(workspace_dir=workspace, neo4j_enabled=False), embedder=embedder)


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
