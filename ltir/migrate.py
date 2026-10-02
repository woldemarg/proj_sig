"""Rebuild a workspace with the current code by re-ingesting its stored sources (docs/06_graph_and_storage.md §6.5).

A representation or canonical version bump makes an existing workspace refuse new work
(``representation_mismatch``). Every READY batch keeps its uploaded file
(``datasets/<id>/source.<ext>``) and its options (``bins``, ``categories``), so the
knowledge base can be reproduced: the batches are re-ingested in their original order into
``<workspace>.migrating``; on success the old workspace is kept as
``<workspace>.bak-<timestamp>`` (never deleted) and the new one takes its place.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ltir.config import Config
from ltir.pipeline import TERMINAL, Engine, PipelineError
from ltir.store import Workspace, acquire_writer_lock


def _source_of(ws: Workspace, batch: dict[str, Any]) -> Path:
    """The stored upload of a READY batch."""
    found = sorted((ws.datasets_dir / batch["dataset_id"]).glob("source.*"))
    if not found:
        raise PipelineError("migration_failed", f"batch {batch['batch_id']}: no stored source for dataset {batch['dataset_id']}")
    return found[0]


def migrate_workspace(config: Config) -> dict[str, Any]:
    """Re-ingest every READY batch of ``config.workspace_dir`` into a fresh workspace and swap it in.

    Takes the workspace's writer lock (``WorkspaceBusy`` while the web app or another writer
    runs) and refuses while a batch is unfinished. A failed re-ingest leaves the old workspace
    untouched and the partial ``.migrating`` folder for inspection. Neo4j publishing is off
    during the re-ingest; with ``NEO4J_ENABLED`` the mirror is synced to the new workspace
    once it is in place.
    """
    root = Path(config.workspace_dir)
    acquire_writer_lock(root)
    old = Workspace(config)
    batches = old.list_batches()
    busy = [b["batch_id"] for b in batches if b["status"] not in TERMINAL]
    if busy:
        raise PipelineError("busy", f"batches in progress: {busy}; stop the web app and retry")
    if old.pending_path.exists():
        raise PipelineError("busy", "an interrupted write is pending; start the web app once (it rolls it back), then retry")
    ready = sorted((b for b in batches if b["status"] == "READY"), key=lambda b: b.get("batch_seq", 0))
    if not ready:
        raise PipelineError("nothing_to_migrate", f"no READY batch in {root}")

    staging = root.with_name(f"{root.name}.migrating")
    shutil.rmtree(staging, ignore_errors=True)
    engine = Engine(replace(config, workspace_dir=staging, neo4j_enabled=False, sphere_export=False), recover=False)  # no thread in the folder
    report = []
    for batch in ready:
        record = engine.ingest_file(_source_of(old, batch), filename=batch["filename"], bins=batch.get("bins"), categories=batch.get("categories"))
        report.append(
            {
                "old_batch": batch["batch_id"],
                "new_batch": record["batch_id"],
                "filename": batch["filename"],
                "dataset_id": record.get("dataset_id"),
                "status": record["status"],
                "insights": (record.get("metrics") or {}).get("validated_insights"),
            }
        )
        if record["status"] != "READY":
            raise PipelineError("migration_failed", f"{batch['filename']}: {record['status']} {record.get('error')}; {root} is unchanged")

    backup = root.with_name(f"{root.name}.bak-{datetime.now(UTC):%Y%m%dT%H%M%S}")
    root.rename(backup)
    try:
        staging.rename(root)
    except OSError:  # put the old workspace back; the staging folder stays for inspection
        backup.rename(root)
        raise
    neo4j = Engine(config, recover=False).sync_neo4j()
    return {"workspace": str(root), "backup": str(backup), "batches": report, "neo4j": neo4j["status"]}
