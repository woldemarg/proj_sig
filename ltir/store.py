"""Local persistence: journals, ontology state, dataset artifacts, batch registry (docs/06_graph_and_storage.md §6.3–6.5).

Layout under ``WORKSPACE_DIR``::

    registry/batches/<batch_id>.json      lifecycle status + metrics (UI polls this)
    datasets/<dataset_id>/source.<ext>     uploaded file (provenance)
    datasets/<dataset_id>/profile.json     schema / profile summary
    datasets/<dataset_id>/covers.npz       pattern_id -> covered row positions
    datasets/<dataset_id>/rejections.json  pruned candidates with reason codes
    journal/patterns.jsonl                 append-only Pattern records (lac ChunkJournal)
    journal/activations.jsonl              append-only ACTIVATES records
    journal/embeddings.mmap (+ _meta.json) float32 unit insight vectors, row = row_id
    journal/blocks/<batch_id>.npz          scope / target / phenomenon blocks (tripartite parts)
    state/                                 lac ConceptStore.save() + representation.json + sig_state.json
    graph/snapshot.json                    materialised dual-layer graph (derived, rebuildable)
    logs/queries.jsonl                     query observability

Recovery: ``checkpoint()`` marks the journal extent and copies state before a batch;
``rollback()`` restores both, so a failed batch leaves no partial knowledge.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ltir.config import Config
from ltir.engines.lac.chunk_journal import ChunkJournal
from ltir.models import CANONICAL_VERSION, REPRESENTATION_VERSION, EmbeddingSpec


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class RepresentationMismatch(RuntimeError):
    code = "representation_mismatch"


class Workspace:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = Path(config.workspace_dir)
        self.registry_dir = self.root / "registry" / "batches"
        self.datasets_dir = self.root / "datasets"
        self.journal_dir = self.root / "journal"
        self.state_dir = self.root / "state"
        self.graph_path = self.root / "graph" / "snapshot.json"
        self.query_log = self.root / "logs" / "queries.jsonl"
        for d in (self.registry_dir, self.datasets_dir, self.journal_dir / "blocks", self.state_dir, self.query_log.parent):
            d.mkdir(parents=True, exist_ok=True)
        self.journal = ChunkJournal(self.journal_dir, records_name="patterns.jsonl")

    def batch_path(self, batch_id: str) -> Path:
        return self.registry_dir / f"{batch_id}.json"

    def save_batch(self, record: dict[str, Any]) -> None:
        record["updated_at"] = utc_now()
        atomic_write_json(self.batch_path(record["batch_id"]), record)

    def load_batch(self, batch_id: str) -> dict[str, Any] | None:
        return read_json(self.batch_path(batch_id))

    def list_batches(self) -> list[dict[str, Any]]:
        batches = [read_json(p) for p in self.registry_dir.glob("*.json")]
        return sorted((b for b in batches if b), key=lambda b: b.get("created_at", ""))

    def ready_batch_for(self, dataset_id: str) -> dict[str, Any] | None:
        return next((b for b in self.list_batches() if b["dataset_id"] == dataset_id and b["status"] == "READY"), None)

    def next_batch_seq(self) -> int:
        state = read_json(self.state_dir / "sig_state.json", {"next_batch_seq": 0})
        return int(state["next_batch_seq"])

    def commit_batch_seq(self, seq: int) -> None:
        atomic_write_json(self.state_dir / "sig_state.json", {"next_batch_seq": seq + 1})

    def dataset_dir(self, dataset_id: str) -> Path:
        path = self.datasets_dir / dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_covers(self, dataset_id: str, covers: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.dataset_dir(dataset_id) / "covers.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in covers.items()})

    def check_representation(self, spec: EmbeddingSpec) -> None:
        """Refuse to mix vectors from a different model/canonicalisation/composition.

        Validation only: the contract is written by ``record_representation`` inside the
        checkpointed commit, so a failed first batch leaves no stale contract behind.
        """
        stored = self.representation()
        if stored is not None and stored.get("fingerprint") != spec.fingerprint:
            raise RepresentationMismatch(
                f"workspace vectors use representation {stored.get('fingerprint')} "
                f"({stored.get('model_id')}), current is {spec.fingerprint} ({spec.model_id}); "
                "rebuild it with the current code: python -m ltir migrate --yes (the old workspace is kept as a backup)"
            )

    def check_versions(self) -> None:
        """Refuse journals written by another canonical/representation version (no model needed)."""
        stored = self.representation()
        if stored is None:
            return
        found = (stored.get("canonical_version"), stored.get("representation_version"))
        if found != (CANONICAL_VERSION, REPRESENTATION_VERSION):
            raise RepresentationMismatch(
                f"workspace was built with {found[0]} / {found[1]}, this code writes {CANONICAL_VERSION} / "
                f"{REPRESENTATION_VERSION}; rebuild it with the current code: python -m ltir migrate --yes (the old workspace is kept as a backup)"
            )

    def record_representation(self, spec: EmbeddingSpec) -> None:
        """Persist the vector contract on the first committed batch."""
        if self.representation() is None:
            atomic_write_json(self.state_dir / "representation.json", spec.to_dict())

    def representation(self) -> dict[str, Any] | None:
        return read_json(self.state_dir / "representation.json")

    def append(
        self, patterns: list[dict[str, Any]], vectors: np.ndarray, activations: list[dict[str, Any]], blocks: dict[str, np.ndarray], batch_id: str
    ) -> None:
        self.journal.append_batch(patterns, vectors, activations)
        np.savez_compressed(self.journal_dir / "blocks" / f"{batch_id}.npz", **blocks)

    def patterns(self) -> list[dict[str, Any]]:
        return self.journal.load_chunks()

    def activations(self) -> list[dict[str, Any]]:
        return self.journal.load_activations()

    def vectors(self) -> np.ndarray:
        """Unit insight vectors (the one frame shared by the ontology and retrieval), row = ``row_id``."""
        return self.journal.load_embeddings()

    def checkpoint(self) -> dict[str, Any]:
        cp_dir = self.state_dir.parent / "checkpoint"
        if cp_dir.exists():
            shutil.rmtree(cp_dir)
        shutil.copytree(self.state_dir, cp_dir)
        return {**self.journal.mark(), "dir": str(cp_dir)}

    def rollback(self, cp: dict[str, Any]) -> None:
        self.journal.truncate(cp)
        shutil.rmtree(self.state_dir)
        shutil.copytree(cp["dir"], self.state_dir)

    def discard_checkpoint(self, cp: dict[str, Any]) -> None:
        shutil.rmtree(cp["dir"], ignore_errors=True)

    def save_graph(self, snapshot: dict[str, Any]) -> None:
        atomic_write_json(self.graph_path, snapshot)

    def load_graph(self) -> dict[str, Any] | None:
        return read_json(self.graph_path)

    def log_query(self, entry: dict[str, Any]) -> None:
        with self.query_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def reset(self) -> None:
        """Delete all SIG workspace data (datasets, journals, state, graph, logs)."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.__init__(self.config)
