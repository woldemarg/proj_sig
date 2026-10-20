"""Local persistence: journals, ontology state, dataset artifacts, batch registry (docs/06_graph_and_storage.md §6.3–6.5).

Layout under ``WORKSPACE_DIR``::

    registry/batches/<batch_id>.json      lifecycle status + metrics (UI polls this)
    datasets/<dataset_id>/profile.json     schema / profile summary
    datasets/<dataset_id>/covers.npz       pattern_id -> covered row positions
    datasets/<dataset_id>/rejections.json  pruned candidates with reason codes
    datasets/<dataset_id>/geo.npz          a geo dataset: H3 cells, point -> cell, points, neighbour graph
    datasets/<dataset_id>/cells.geojson    a geo dataset: the map's GeoJSON (cells, patterns' extents), written with the batch
    journal/patterns.jsonl                 append-only Pattern records (lac ChunkJournal)
    journal/activations.jsonl              append-only ACTIVATES records
    journal/embeddings.mmap (+ _meta.json) float32 unit insight vectors, row = row_id
    journal/blocks/<batch_id>.npz          scope / target / phenomenon blocks, pattern ids
    state/                                 lac ConceptStore.save() + representation.json + sig_state.json
    graph/snapshot.json                    materialised dual-layer graph (derived, rebuildable)
    graph/literals.npz                     literal-catalog vectors by text (a cache, docs/07_question_answering.md §7.1.1)
    uploads/<id>_<name>                    uploaded files (the batch record points at its upload: provenance)
    pending.json                           an unfinished transaction: its intent and checkpoint

Every path below the workspace root is named here and only here: callers use these methods.

Recovery: a batch commit and a dataset deletion each run in ``transaction()``, which
checkpoints the journal extent, state and snapshot and writes ``pending.json``; an error rolls
back at once, a crash leaves the marker and ``recover()`` rolls back at the next writer start.

Writers: one process at a time holds ``<workspace>.writer.lock`` (beside the folder, so a
reset can delete the folder while it is held); see ``acquire_writer_lock``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np

from attractor_topology.models import CANONICAL_VERSION, REPRESENTATION_VERSION, EmbeddingSpec
from insight_graph_service.core.chunk_journal import ChunkJournal
from insight_graph_service.core.fileio import replace_file
from insight_graph_service.core.settings import Settings

if TYPE_CHECKING:
    from subgroup_miner.spatial import SpatialContext

log = logging.getLogger(__name__)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    replace_file(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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

    One writer process per workspace — the service. A second process gets
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
            f"workspace {workspace_dir} is in use by another writer process (pid {owner}); stop it first (while the service runs, upload through its API)"
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()).encode("ascii"))
    handle.flush()
    _HELD_LOCKS[str(path)] = handle


def utc_now() -> str:
    """Timestamps of records and snapshots: ISO 8601, UTC, whole seconds."""
    return datetime.now(UTC).isoformat(timespec="seconds")


TERMINAL = frozenset({"READY", "FAILED", "SKIPPED"})  # final batch states


class RepresentationMismatch(RuntimeError):
    code = "representation_mismatch"


class RollbackPending(RuntimeError):
    code = "rollback_pending"


class UploadTooLarge(RuntimeError):
    code = "file_too_large"


class Workspace:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.workspace_dir)
        self.registry_dir = self.root / "registry" / "batches"
        self.datasets_dir = self.root / "datasets"
        self.journal_dir = self.root / "journal"
        self.state_dir = self.root / "state"
        self.graph_path = self.root / "graph" / "snapshot.json"
        self.literals_path = self.root / "graph" / "literals.npz"
        self.uploads_dir = self.root / "uploads"
        self.pending_path = self.root / "pending.json"  # an unfinished transaction (``transaction``)
        for d in (self.registry_dir, self.datasets_dir, self.journal_dir / "blocks", self.state_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.journal = ChunkJournal(self.journal_dir, records_name="patterns.jsonl")
        self._batches: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """The batch registry and the vector contract, read once and written through: the console polls them, and a
        reader never opens a file the writer is replacing. Merged, not replaced: a record saved meanwhile stays."""
        self._batches.update({b["batch_id"]: b for b in map(read_json, self.registry_dir.glob("*.json")) if b})
        self._representation = read_json(self.state_dir / "representation.json")

    def batch_path(self, batch_id: str) -> Path:
        return self.registry_dir / f"{batch_id}.json"

    def save_batch(self, record: dict[str, Any]) -> None:
        record["updated_at"] = utc_now()
        atomic_write_json(self.batch_path(record["batch_id"]), record)
        self._batches[record["batch_id"]] = json.loads(json.dumps(record, default=str))  # what the file holds

    def load_batch(self, batch_id: str) -> dict[str, Any] | None:
        """A copy: a caller edits it and saves it, or drops it."""
        return copy.deepcopy(self._batches.get(batch_id))

    def list_batches(self) -> list[dict[str, Any]]:
        """Every batch record, oldest first: the cached records themselves, to read (``load_batch`` gives a copy to edit). A
        record is replaced whole when it is saved, never edited in place, so a reader holds a consistent one."""
        return sorted(self._batches.values(), key=lambda b: b.get("created_at", ""))

    def ready_batch_for(self, dataset_id: str) -> dict[str, Any] | None:
        """A copy of the dataset's READY record (a caller may edit it and save it)."""
        found = next((b for b in self.list_batches() if b["dataset_id"] == dataset_id and b["status"] == "READY"), None)
        return copy.deepcopy(found)

    def unfinished_batches(self) -> list[dict[str, Any]]:
        """Copies of the batches not yet READY, FAILED or SKIPPED (queued or running): recovery edits and saves them."""
        return [copy.deepcopy(b) for b in self.list_batches() if b["status"] not in TERMINAL]

    def next_batch_seq(self) -> int:
        state = read_json(self.state_dir / "sig_state.json", {"next_batch_seq": 0})
        return int(state["next_batch_seq"])

    def commit_batch_seq(self, seq: int) -> None:
        atomic_write_json(self.state_dir / "sig_state.json", {"next_batch_seq": seq + 1})

    def dataset_dir(self, dataset_id: str) -> Path:
        path = self.datasets_dir / dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload(self, name: str, stream: BinaryIO, max_bytes: int) -> Path:
        """Store an uploaded file under ``uploads/``; ``UploadTooLarge`` once the stream passes ``max_bytes`` (the copy
        stops one byte past the cap and the partial file is removed). This bounds the copy under ``uploads/``; the
        request body itself is bounded by the HTTP layer, which refuses a ``Content-Length`` above the cap before
        reading it (a body sent without one is spooled by the framework first)."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = self.uploads_dir / f"{uuid.uuid4().hex[:8]}_{Path(name).name}"
        copied = 0
        with dest.open("wb") as f:
            while chunk := stream.read(min(1 << 20, max_bytes + 1 - copied)):
                copied += len(chunk)
                f.write(chunk)
        if copied > max_bytes:
            dest.unlink(missing_ok=True)
            raise UploadTooLarge(f"{name} exceeds {max_bytes >> 20} MB (MAX_UPLOAD_MB)")
        return dest

    def save_profile(self, dataset_id: str, profile: dict[str, Any]) -> None:
        atomic_write_json(self.dataset_dir(dataset_id) / "profile.json", profile)

    def save_rejections(self, dataset_id: str, rejections: list[dict[str, Any]]) -> None:
        atomic_write_json(self.dataset_dir(dataset_id) / "rejections.json", rejections)

    def covers_ref(self, dataset_id: str, pattern_id: str) -> str:
        """Where a pattern's covered row positions are stored (its ``provenance.rows_ref``)."""
        return f"datasets/{dataset_id}/covers.npz#{pattern_id}"

    def save_covers(self, dataset_id: str, covers: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.dataset_dir(dataset_id) / "covers.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in covers.items()})

    def load_covers(self, dataset_id: str) -> dict[str, np.ndarray]:
        path = self.datasets_dir / dataset_id / "covers.npz"
        if not path.is_file():
            return {}
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    def save_geo(self, dataset_id: str, spatial: SpatialContext) -> None:
        """A geo dataset's spatial context: its cells, each point's cell, the points and the neighbour graph."""
        np.savez_compressed(
            self.dataset_dir(dataset_id) / "geo.npz",
            cells=spatial.cells.astype(str),
            point_cell=spatial.point_cell,
            lonlat=spatial.lonlat,
            nbr_ptr=spatial.nbr_ptr,
            nbr_idx=spatial.nbr_idx,
        )

    def save_geo_cells(self, dataset_id: str, payload: dict[str, Any]) -> None:
        """The map's GeoJSON of a geo dataset (``geo_views.geo_cells``), written with its batch."""
        atomic_write_json(self.geo_cells_path(dataset_id), payload)

    def geo_cells_path(self, dataset_id: str) -> Path:
        """The map's GeoJSON of a geo dataset (``save_geo_cells``); absent for any other dataset."""
        return self.datasets_dir / dataset_id / "cells.geojson"

    def load_geo(self, dataset_id: str, keys: tuple[str, ...] | None = None) -> dict[str, Any] | None:
        """The spatial context ``save_geo`` stored for a geo dataset (only ``keys``, if given); None for any other dataset."""
        path = self.datasets_dir / dataset_id / "geo.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in keys or data.files}

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
                "reset the workspace (POST /api/reset) and ingest the data again"
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
                f"{REPRESENTATION_VERSION}; reset the workspace (POST /api/reset) and ingest the data again"
            )

    def record_representation(self, spec: EmbeddingSpec) -> None:
        """Persist the vector contract on the first committed batch."""
        if self._representation is None:
            atomic_write_json(self.state_dir / "representation.json", spec.to_dict())
            self._representation = spec.to_dict()

    def representation(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._representation)

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
        with np.load(self.literals_path) as data:
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

    def activations(self) -> list[dict[str, Any]]:
        return self.journal.load_activations()

    def vectors(self) -> np.ndarray:
        """Unit insight vectors (the one frame shared by the ontology and retrieval), row = ``row_id``."""
        return self.journal.load_embeddings()

    @contextmanager
    def transaction(self, intent: dict[str, Any], *, journal: bool = False) -> Iterator[dict[str, Any]]:
        """One all-or-nothing write (a batch commit, a dataset deletion, a metadata change): checkpoint, then ``pending.json`` holding
        ``intent`` and the checkpoint. An error in the body rolls back; a crash leaves the marker for ``recover``.
        A rollback that fails itself leaves the marker too, and no transaction starts again until ``recover`` ran."""
        if self.pending_path.exists():
            raise RollbackPending(
                "an earlier write was not finished; restart the service to finish it (POST /api/reset deletes the workspace instead)"
            )
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
                self._finish(cp, intent)  # the body's error is the one to raise; recover rolls back again
            raise
        if "batch_id" in intent:  # a batch committed at its READY record, the body's last write: nothing after it may undo it
            self._finish(cp, intent)
        else:  # a deletion or a metadata change commits when its marker goes: a marker that stays fails the write
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

    def _finish(self, cp: dict[str, Any], intent: dict[str, Any]) -> None:
        """Close the marker of a batch that committed, or of a body that was rolled back. A marker that cannot be removed (a file
        another process holds open) is logged and left: ``recover`` at the next writer start closes it after a READY batch and
        rolls back anything else again; until then writes are refused (``rollback_pending``)."""
        try:
            self._close(cp)
        except OSError:
            log.exception("marker of %s not removed; the next writer start closes it", intent)

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
        self._load_cache()  # the restored state and batch records

    def rewrite_journal(self, patterns: list[dict[str, Any]], vectors: np.ndarray, activations: list[dict[str, Any]]) -> None:
        self.journal.rewrite(patterns, vectors, activations)

    def remove_datasets(self, dataset_ids: set[str], batches: list[dict[str, Any]], trash: Path) -> None:
        """Move the datasets' artefacts — their folders, their batches' blocks, records and uploads — into ``trash`` (inside
        the checkpoint), so a rollback can put them back and discarding the checkpoint deletes them. An upload another
        batch still points at (another level of the same upload) stays."""
        root = self.root.resolve()
        for b in batches:
            self._batches.pop(b["batch_id"], None)
        shared = {b.get("source_path") for b in self._batches.values()}
        paths = [self.datasets_dir / ds for ds in sorted(dataset_ids)]
        for b in batches:
            source = Path(b.get("source_path") or "")
            paths += [self.journal_dir / "blocks" / f"{b['batch_id']}.npz", self.batch_path(b["batch_id"])]
            if source.is_file() and source.resolve().is_relative_to(self.uploads_dir.resolve()) and str(source) not in shared:
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

    def reset(self) -> None:
        """Delete all SIG workspace data (datasets, journals, state, graph, uploads). The folder is renamed aside first —
        one step, so a failure leaves it whole — then deleted; a ``<name>.deleting-*`` folder left behind is safe to remove."""
        if self.root.exists():
            trash = self.root.with_name(f"{self.root.name}.deleting-{uuid.uuid4().hex[:8]}")
            os.replace(self.root, trash)
            shutil.rmtree(trash, ignore_errors=True)
        self.__init__(self.settings)
