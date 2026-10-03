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
    graph/literals.npz                     literal-catalog vectors by text (a cache, docs/07_question_answering.md §7.1.1)
    graph/sphere.html                      the standalone 3D sphere, written on request (python -m ltir sphere)
    logs/queries.jsonl                     query observability
    uploads/<id>_<name>                    files uploaded through the web app
    experiments/<time>.json                hypothesis experiment results (python -m ltir experiment)
    pending.json                           an unfinished transaction: its intent and checkpoint

Every path below the workspace root is named here and only here: callers use these methods.

Recovery: a batch commit and a dataset deletion each run in ``transaction()``, which
checkpoints the journal extent, state and snapshot and writes ``pending.json``; an error rolls
back at once, a crash leaves the marker and ``recover()`` rolls back at the next writer start.

Writers: one process at a time holds ``<workspace>.writer.lock`` (beside the folder, so a
migration can rename the folder while it is held); see ``acquire_writer_lock``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ltir.config import Config
from ltir.engines.lac.chunk_journal import ChunkJournal
from ltir.fileio import replace_file, retry_sharing
from ltir.models import CANONICAL_VERSION, REPRESENTATION_VERSION, EmbeddingSpec, utc_now

log = logging.getLogger(__name__)


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


TERMINAL = frozenset({"READY", "FAILED", "SKIPPED"})  # final batch states


class RepresentationMismatch(RuntimeError):
    code = "representation_mismatch"


class RollbackPending(RuntimeError):
    code = "rollback_pending"


class Workspace:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = Path(config.workspace_dir)
        self.registry_dir = self.root / "registry" / "batches"
        self.datasets_dir = self.root / "datasets"
        self.journal_dir = self.root / "journal"
        self.state_dir = self.root / "state"
        self.graph_path = self.root / "graph" / "snapshot.json"
        self.sphere_path = self.root / "graph" / "sphere.html"
        self.literals_path = self.root / "graph" / "literals.npz"
        self.query_log = self.root / "logs" / "queries.jsonl"
        self.uploads_dir = self.root / "uploads"
        self.experiments_dir = self.root / "experiments"
        self.pending_path = self.root / "pending.json"  # an unfinished transaction (``transaction``)
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

    def unfinished_batches(self) -> list[dict[str, Any]]:
        """Batches not yet READY, FAILED or SKIPPED (queued or running)."""
        return [b for b in self.list_batches() if b["status"] not in TERMINAL]

    def next_batch_seq(self) -> int:
        state = read_json(self.state_dir / "sig_state.json", {"next_batch_seq": 0})
        return int(state["next_batch_seq"])

    def commit_batch_seq(self, seq: int) -> None:
        atomic_write_json(self.state_dir / "sig_state.json", {"next_batch_seq": seq + 1})

    def dataset_dir(self, dataset_id: str) -> Path:
        path = self.datasets_dir / dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def keep_source(self, dataset_id: str, source: Path, filename: str) -> None:
        """Copy an uploaded file next to the dataset's artefacts (``datasets/<id>/source.<ext>``, provenance)."""
        copy = self.dataset_dir(dataset_id) / f"source{Path(filename).suffix.lower() or '.csv'}"
        if not copy.exists():
            shutil.copyfile(source, copy)

    def source_of(self, dataset_id: str) -> Path | None:
        """The stored upload of a dataset, or ``None``."""
        found = sorted((self.datasets_dir / dataset_id).glob("source.*"))
        return found[0] if found else None

    def save_upload(self, name: str, stream: BinaryIO) -> Path:
        """Store a file received by the web app under ``uploads/``; returns its path."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = self.uploads_dir / f"{uuid.uuid4().hex[:8]}_{Path(name).name}"
        with dest.open("wb") as f:
            shutil.copyfileobj(stream, f)
        return dest

    def save_profile(self, dataset_id: str, profile: dict[str, Any]) -> None:
        atomic_write_json(self.dataset_dir(dataset_id) / "profile.json", profile)

    def save_rejections(self, dataset_id: str, rejections: list[dict[str, Any]]) -> None:
        atomic_write_json(self.dataset_dir(dataset_id) / "rejections.json", rejections)

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

    def save_literals(self, fingerprint: str, texts: list[str], vectors: np.ndarray) -> None:
        """Literal-catalog vectors by text (docs/07_question_answering.md §7.1.1) in ``graph/literals.npz``. A cache checked on load (the
        representation fingerprint), so it lives outside ``state/`` and no rollback has to restore it."""
        tmp = self.literals_path.with_name(self.literals_path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, texts=np.array(texts), vectors=np.asarray(vectors, dtype=np.float32), fingerprint=np.array(fingerprint))
        replace_file(tmp, self.literals_path)

    def load_literals(self, fingerprint: str) -> dict[str, np.ndarray]:
        """Cached literal vectors (text -> unit vector) written for this representation; ``{}`` otherwise."""
        if not self.literals_path.is_file():
            return {}
        with retry_sharing(lambda: np.load(self.literals_path)) as data:
            if str(data["fingerprint"]) != fingerprint:
                return {}
            return dict(zip(map(str, data["texts"]), data["vectors"]))

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

    def save_document_vectors(self, documents: dict[str, np.ndarray], changed: set[str]) -> None:
        """Back-fill document vectors: every batch holding a ``changed`` pattern gets all its patterns' vectors from
        ``documents``, in the batch's journal order (the row order of its blocks file)."""
        by_batch: dict[str, list[str]] = {}
        for record in self.patterns():
            if record["id"] in documents:
                by_batch.setdefault(record["batch_id"], []).append(record["id"])
        for batch_id, ids in by_batch.items():
            if changed.intersection(ids):
                self._write_document_vectors(batch_id, ids, np.stack([documents[i] for i in ids]))

    def _write_document_vectors(self, batch_id: str, pattern_ids: list[str], vectors: np.ndarray) -> None:
        """One batch's document vectors into its blocks file; the other arrays are kept, the file replaced atomically."""
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

    @contextmanager
    def transaction(self, intent: dict[str, Any], *, journal: bool = False) -> Iterator[dict[str, Any]]:
        """One all-or-nothing write (a batch commit, a dataset deletion): checkpoint, then ``pending.json`` holding
        ``intent`` and the checkpoint. An error in the body rolls back; a crash leaves the marker for ``recover``.
        A rollback that fails itself leaves the marker too, and no transaction starts again until ``recover`` ran."""
        if self.pending_path.exists():
            raise RollbackPending("an earlier write was not rolled back; restart the writer (web app or CLI) to recover")
        cp = self.checkpoint(journal=journal)
        atomic_write_json(self.pending_path, {**intent, "checkpoint": cp})
        try:
            yield cp
        except Exception:  # (a BaseException is a dying process: the marker stays for recover)
            try:
                self.rollback(cp)
            except Exception:
                log.exception("rollback of %s failed; the next writer start retries it", intent)
            else:
                self._close(cp)
            raise
        self._close(cp)

    def recover(self) -> dict[str, Any] | None:
        """Finish an interrupted transaction (writer start: no other writer alive). A batch whose READY record was
        written had committed and keeps its writes; anything else is rolled back. Returns the rolled-back intent."""
        pending = read_json(self.pending_path)
        if pending is None:
            return None
        cp = pending.pop("checkpoint")
        batch = self.load_batch(pending["batch_id"]) if "batch_id" in pending else None
        committed = batch is not None and batch["status"] == "READY"
        if not committed:
            self.rollback(cp)
        self._close(cp)
        return None if committed else pending

    def _close(self, cp: dict[str, Any]) -> None:
        self.pending_path.unlink()
        shutil.rmtree(cp["dir"], ignore_errors=True)

    def checkpoint(self, *, journal: bool = False) -> dict[str, Any]:
        """Copy ``state/`` and the snapshot aside (and, for a rewrite that may shrink the journal, ``journal/``);
        a batch only appends, so for it the journal extent (``mark``) is enough. ``rollback`` restores all of it."""
        cp_dir = self.state_dir.parent / "checkpoint"
        if cp_dir.exists():
            shutil.rmtree(cp_dir)
        shutil.copytree(self.state_dir, cp_dir)
        if journal:
            shutil.copytree(self.journal_dir, cp_dir / "journal")
        had_snapshot = self.graph_path.is_file()
        if had_snapshot:  # readers trust the snapshot: a rolled-back batch must not leave its graph behind
            (cp_dir / "graph").mkdir()
            shutil.copyfile(self.graph_path, cp_dir / "graph" / "snapshot.json")
        return {**self.journal.mark(), "dir": str(cp_dir), "snapshot": had_snapshot}

    def rollback(self, cp: dict[str, Any]) -> None:
        """Restore a checkpoint; repeatable, so ``recover`` can finish a rollback that failed halfway."""
        cp_dir = Path(cp["dir"])
        if (cp_dir / "journal").exists():
            shutil.rmtree(self.journal_dir, ignore_errors=True)
            shutil.copytree(cp_dir / "journal", self.journal_dir)
        else:
            self.journal.truncate(cp)
        shutil.rmtree(self.state_dir, ignore_errors=True)
        shutil.copytree(cp_dir, self.state_dir, ignore=shutil.ignore_patterns("journal", "graph", "removed"))
        if cp["snapshot"]:
            tmp = self.graph_path.with_name("snapshot.json.restore")
            shutil.copyfile(cp_dir / "graph" / "snapshot.json", tmp)
            replace_file(tmp, self.graph_path)
        else:
            self.graph_path.unlink(missing_ok=True)
        removed = cp_dir / "removed"  # files a dataset deletion moved aside go back where they were
        for f in sorted(p for p in removed.rglob("*") if p.is_file()) if removed.exists() else []:
            dest = self.root / f.relative_to(removed)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))

    def rewrite_journal(self, patterns: list[dict[str, Any]], vectors: np.ndarray, activations: list[dict[str, Any]]) -> None:
        self.journal.rewrite(patterns, vectors, activations)

    def remove_dataset(self, dataset_id: str, batches: list[dict[str, Any]], trash: Path) -> None:
        """Move a dataset's artefacts — its folder, its batches' blocks, records and web uploads, and the standalone
        sphere page, which may show it — into ``trash`` (inside the checkpoint), so a rollback can put them back and
        discarding the checkpoint deletes them."""
        root = self.root.resolve()
        paths = [self.datasets_dir / dataset_id, self.sphere_path]
        for b in batches:
            source = Path(b.get("source_path") or "")
            paths += [self.journal_dir / "blocks" / f"{b['batch_id']}.npz", self.batch_path(b["batch_id"])]
            if source.is_file() and source.resolve().is_relative_to(self.uploads_dir.resolve()):
                paths.append(source)
        for path in paths:
            if path.exists():
                dest = trash / path.resolve().relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))

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
