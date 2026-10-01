"""Shared helpers for the measurement scripts: throwaway workspaces under ``sig/.scratch/``.

Never touches ``workspace/`` (the user's knowledge base). Neo4j publishing and the
sphere export are always off here.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

# scripts run from scripts/; the package lives one level up (not pip-installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ltir.config import PROJECT_ROOT, load_config  # noqa: E402
from ltir.pipeline import Engine  # noqa: E402
from ltir.synth import generate_retail_dataset  # noqa: E402

SCRATCH = PROJECT_ROOT / ".scratch"
DEMO_QUESTION = "Why is margin lower for phones in the US?"


def scratch_dir(name: str, *, fresh: bool = False) -> Path:
    """``.scratch/<name>/``, emptied first when ``fresh``."""
    path = SCRATCH / name
    if fresh:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_demo_csv(folder: Path, seed: int = 7) -> Path:
    """The synthetic retail dataset (``ltir.synth``) as a CSV inside ``folder``."""
    path = folder / f"retail_synthetic_seed{seed}.csv"
    if not path.exists():
        generate_retail_dataset(5000, seed).to_csv(path, index=False)
    return path


def build_engine(workspace: Path, *, llm: Any = None, embedder: Any = None, **overrides: Any) -> Engine:
    """Engine on a scratch workspace; ``overrides`` are ``Config`` fields, ``embedder`` reuses a loaded model."""
    cfg = load_config(workspace_dir=workspace, neo4j_enabled=False, sphere_export=False, **overrides)
    return Engine(cfg, llm=llm, embedder=embedder)


def ingest_ready(engine: Engine, path: Path, **options: Any) -> dict[str, Any]:
    """Run one batch; exits with the batch status and error unless it ends READY."""
    record = engine.ingest_file(path, **options)
    if record["status"] != "READY":
        raise SystemExit(f"{path.name}: {record['status']} {record.get('error')}")
    return record


def demo_engine(name: str, *, llm: Any = None, **overrides: Any) -> Engine:
    """Fresh scratch workspace with the demo ingested."""
    root = scratch_dir(name, fresh=True)
    engine = build_engine(root / "ws", llm=llm, **overrides)
    ingest_ready(engine, write_demo_csv(root))
    return engine


def save_result(name: str, label: str, payload: Any) -> Path:
    """Persist a measurement next to the scratch workspaces (``.scratch/results/<name>_<label>.json``)."""
    path = scratch_dir("results") / f"{name}_{label}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path
