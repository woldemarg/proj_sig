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
    journal/blocks/<batch_id>.npz          scope / target / phenomenon blocks, document vectors, pattern ids
    state/                                 lac ConceptStore.save() + representation.json + sig_state.json
    graph/snapshot.json                    materialised dual-layer graph (derived, rebuildable)
    logs/queries.jsonl                     query observability

Recovery: ``checkpoint()`` marks the journal extent and copies state before a batch;
``rollback()`` restores both, so a failed batch leaves no partial knowledge.

Writers: one process at a time holds ``<workspace>.writer.lock`` (beside the folder, so a
migration can rename the folder while it is held); see ``acquire_writer_lock``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ltir.config import Config
from ltir.engines.lac.chunk_journal import ChunkJournal
from ltir.fileio import replace_file, retry_sharing
from ltir.models import CANONICAL_VERSION, REPRESENTATION_VERSION, EmbeddingSpec


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    replace_file(tmp, path)  # waits out a reader of the old file (Windows)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(retry_sharing(lambda: path.read_text(encoding="utf-8")))  # waits out a replace (Windows)


class WorkspaceBusy(RuntimeError):
    """Another live process holds the workspace's writer lock."""


_LOCK_OFFSET = 1 << 20  # the locked byte, far past the pid at offset 0, which stays readable for the error message
_HELD_LOCKS: dict[str, BinaryIO] = {}


def writer_lock_path(workspace_dir: Path | str) -> Path:
    """``<workspace>.writer.lock`` beside the workspace folder."""
    root = Path(workspace_dir).resolve()
    return root.with_name(f"{root.name}.writer.lock")


def _lock_byte(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def acquire_writer_lock(workspace_dir: Path | str) -> None:
    """Hold the workspace's exclusive writer lock until this process exits (re-entrant).

    One writer process per workspace — the web app or one CLI writer. A second process gets
    ``WorkspaceBusy`` instead of interleaving batches with the first or failing its queued
    uploads during recovery. The operating system releases the lock when the holder exits,
    also after a crash.
    """
    path = writer_lock_path(workspace_dir)
    if str(path) in _HELD_LOCKS:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        _lock_byte(handle)
    except OSError:
        os.lseek(handle.fileno(), 0, os.SEEK_SET)  # unbuffered: a buffered read would reach the locked byte
        owner = os.read(handle.fileno(), 32).decode("ascii", "ignore").strip() or "unknown"
        handle.close()
        raise WorkspaceBusy(
            f"workspace {workspace_dir} is in use by another writer process (pid {owner}); stop it first (while the web app runs, upload through it)"
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()).encode("ascii"))
    handle.flush()
    _HELD_LOCKS[str(path)] = handle


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

    def document_vectors(self, batch_ids: set[str]) -> dict[str, np.ndarray]:
        """Canonical-document embeddings stored at ingest (the naive text baseline), pattern id -> vector."""
        out: dict[str, np.ndarray] = {}
        for batch_id in batch_ids:
            path = self.journal_dir / "blocks" / f"{batch_id}.npz"
            if not path.is_file():
                continue
            with retry_sharing(lambda path=path: np.load(path)) as data:
                if "document" in data.files and "pattern_ids" in data.files:
                    out.update(zip(data["pattern_ids"].tolist(), data["document"]))
        return out

    def add_document_vectors(self, batch_id: str, pattern_ids: list[str], vectors: np.ndarray) -> None:
        """Back-fill a batch's document vectors (``pattern_ids`` in the batch's journal order, so they
        align with its blocks); the other arrays are kept and the file is replaced atomically."""
        path = self.journal_dir / "blocks" / f"{batch_id}.npz"
        arrays: dict[str, np.ndarray] = {}
        if path.is_file():
            with retry_sharing(lambda: np.load(path)) as data:
                arrays = {k: data[k] for k in data.files}
        arrays["document"] = np.asarray(vectors, dtype=np.float32)
        arrays["pattern_ids"] = np.array(pattern_ids)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        replace_file(tmp, path)

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
