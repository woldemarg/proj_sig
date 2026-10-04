"""The application service (docs/09_operations.md, docs/12_architecture.md).

``Engine`` orchestrates the analytical core over one workspace for the service:
the batch lifecycle

UPLOADED -> VALIDATING -> PROFILING -> DISCOVERING -> VALIDATING_INSIGHTS -> EMBEDDING
-> UPDATING_ONTOLOGY -> BUILDING_GRAPH -> PERSISTING -> READY   (| FAILED | SKIPPED)

(strictly one batch at a time: the ontology is stateful; each commits completely or is rolled
back), dataset deletion, the committed read state, retrieval (``search``) and the evidence the narrator
verbalises (``evidence``).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from attractor_topology.canonical import canonicalize
from attractor_topology.encoder import InsightEncoder, SentenceTransformerEmbedder, TextEmbedder
from attractor_topology.models import CanonicalInsight, EmbeddingSpec
from attractor_topology.ontology import LatentOntology
from graph_query_engine.graph import DualGraph, LatentFrame
from graph_query_engine.question import build_catalog
from graph_query_engine.search import CommittedState, SearchResult, empty_payload, search
from insight_contracts import SNAPSHOT_VERSION, EvidencePayload, Insight
from insight_graph_service.core.batch import admit_insights, batch_metrics, pattern_records
from insight_graph_service.core.settings import Settings
from insight_graph_service.core.snapshot import build_snapshot
from insight_graph_service.core.workspace import TERMINAL, RepresentationMismatch, Workspace, acquire_writer_lock, utc_now
from subgroup_miner.discovery import build_insights, covers_of, run_discovery
from subgroup_miner.ingestion import load_dataset
from subgroup_miner.selection import select_insights

log = logging.getLogger(__name__)

BLOCKS = ("scope", "target", "phenomenon")  # tripartite parts persisted beside the composite vector


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _healthy(method):
    """Refuse the call (``workspace_degraded``) while the engine serves a workspace it cannot use (``Engine.problem``)."""

    @functools.wraps(method)
    def guarded(self: Engine, *args: Any, **kwargs: Any) -> Any:
        if self.problem:
            raise PipelineError("workspace_degraded", f"{self.problem}; reset the workspace (POST /api/reset)")
        return method(self, *args, **kwargs)

    return guarded


class Engine:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: TextEmbedder | None = None,
        recover: bool = True,
    ) -> None:
        """A writer (``recover=True``: the service) takes the workspace's writer lock — ``WorkspaceBusy`` if another
        process holds it — and then recovers unfinished batches. ``recover=False`` reads a workspace another process writes.
        A workspace it cannot open starts degraded (``_open``)."""
        self.settings = settings
        if recover:
            acquire_writer_lock(settings.workspace_dir)
        self.ws = Workspace(settings)
        self.encoder = InsightEncoder(embedder or SentenceTransformerEmbedder(settings.topology), settings.topology)
        self._lock = threading.RLock()  # batches, deletion, reset and rebuild: one write at a time
        self._writer = recover  # holds the writer lock: may save derived files (literal cache, document back-fill)
        self._prepare_lock = threading.Lock()  # catalog build and document back-fill: never the batch lock
        self._open()

    def committed(self) -> CommittedState:
        """The committed read state. One object per commit, so a reader never mixes two commits; readers never
        touch the journal or a lock, so they neither race nor wait for a batch."""
        return self._state

    def prepared(self) -> CommittedState:
        """The committed state made ready for questions, once per commit (``_prepare``)."""
        state = self._state
        self._prepare(state)
        return state

    def graph(self) -> DualGraph:
        return self.committed().graph

    def frame(self) -> LatentFrame:
        """Committed pattern vectors and attractor centroids (retrieval and the sphere)."""
        return self.committed().frame

    def _open(self) -> None:
        """Recover the workspace (a writer) and load its committed state. A workspace this code cannot serve — a
        recovery that failed, journals of another canonical / representation version — leaves the empty state and
        ``problem`` (why); every guarded call is refused until ``reset()``. So is a snapshot or state file that cannot
        be read: the service starts, and its reset stays reachable."""
        problem: str | None = None
        try:
            if self._writer:
                self._recover()
            self.ws.check_versions()
            state = self._load_state()
        except RepresentationMismatch as exc:
            problem, state = str(exc), self._empty_state()
        except Exception as exc:  # a rollback that cannot finish (its marker keeps blocking writes), an unreadable file
            problem, state = f"cannot open the workspace: {type(exc).__name__}: {exc}", self._empty_state()
        if problem:
            log.error("workspace degraded: %s", problem)
        self.problem, self._state = problem, state  # together, after the work: readers never see a half-open engine

    def _load_state(self) -> CommittedState:
        """The read state of the saved snapshot; a snapshot of another version is rebuilt (derived data: the journals
        stay the source of truth)."""
        snapshot = self.ws.load_graph()
        if snapshot and snapshot.get("version") != SNAPSHOT_VERSION:
            return self._derive(self.ontology())[1]
        graph = DualGraph(snapshot) if snapshot else DualGraph.empty()
        return CommittedState(graph, self._load_frame(self.ws.patterns(), self.ws.vectors(), self.ontology()))

    @staticmethod
    def _empty_state() -> CommittedState:
        return CommittedState(DualGraph.empty(), LatentFrame({}, {}, {}))

    def _prepare(self, state: CommittedState) -> None:
        """Fill what a question needs and the commit did not compute: the canonical-document vector of every pattern
        (stored at ingest; one missing, from a batch written before documents were embedded, is embedded here) and the
        literal catalog (docs/07_question_answering.md §7.1.1; vectors of known literals from the workspace's literal
        cache). Both stay on ``state``; a writer saves them, so no later question or restart pays for them again."""
        graph, documents = state.graph, state.frame.documents
        with self._prepare_lock:
            missing = [pid for pid in graph.insights if pid not in documents]
            if missing:
                filled = dict(zip(missing, self.encoder.embedder.embed([graph.canonical_document(i) for i in missing])))
                documents.update(filled)
                self._persist(state, lambda: self.ws.save_document_vectors(documents, set(filled)))
            if state.catalog is None and graph.insights:
                fingerprint = self.encoder.spec.fingerprint
                known = self.ws.load_literals(fingerprint)
                state.catalog = build_catalog(graph, self.encoder.embedder, known)
                if known.keys() != set(state.catalog.texts):  # new literals, or a deleted dataset's gone
                    catalog = state.catalog
                    self._persist(state, lambda: self.ws.save_literals(fingerprint, catalog.texts, catalog.vectors))

    def _persist(self, state: CommittedState, save: Callable[[], None]) -> None:
        """Save a cache derived on a question (a writer only), unless a write runs — a reset or a deletion may be
        removing its files — or a newer commit replaced ``state``; a later question saves it instead."""
        if self._writer and self._lock.acquire(blocking=False):
            try:
                if state is self._state:
                    save()
            finally:
                self._lock.release()

    def ontology(self) -> LatentOntology:
        return LatentOntology(self.settings.topology, self.ws.state_dir)

    @_healthy
    def search(self, question: str) -> SearchResult:
        """Retrieval only: the evidence for ``question`` over the committed graph (no LLM; docs/07_question_answering.md §7.1–7.4)."""
        return self._search(question, self.committed())

    @_healthy
    def evidence(self, question: str) -> EvidencePayload:
        """The evidence for ``question`` as the narrator receives it (``POST /api/search``); an empty graph is an
        ordinary answer, not an error."""
        state = self.committed()
        return self._search(question, state).payload() if state.graph.insights else empty_payload(question)

    def _search(self, question: str, state: CommittedState) -> SearchResult:
        start = time.perf_counter()
        self.ws.check_representation(self.encoder.spec)  # query and stored vectors must share one frame
        self._prepare(state)  # the literal catalog; every document vector for the naive baseline
        found = search(question, state, self.encoder, self.settings.query)
        found.seconds = time.perf_counter() - start  # with the preparation of the first question after a commit
        return found

    def _load_frame(self, records: list[dict[str, Any]], vectors: np.ndarray, ontology: LatentOntology) -> LatentFrame:
        return LatentFrame(
            patterns={r["id"]: vectors[r["row_id"]] for r in records} if len(vectors) else {},
            attractors={a.id: a.centroid for a in ontology.attractors()},
            documents=self.ws.document_vectors({r["batch_id"] for r in records}),
        )

    def _derive(self, ontology: LatentOntology, pending: dict[str, Any] | None = None) -> tuple[dict[str, Any], CommittedState]:
        """The snapshot of the workspace's state (saved) and the read state of the same journal read; ``pending`` (the
        batch being committed) counts as READY. Runs inside a write, before its commit point."""
        batches = {b["batch_id"]: b for b in self.ws.list_batches()}
        if pending is not None:
            batches[pending["batch_id"]] = {**pending, "status": "READY"}
        records, vectors = self.ws.patterns(), self.ws.vectors()
        snapshot = build_snapshot(
            records=records,
            activations=self.ws.activations(),
            vectors=vectors,
            batches=batches,
            representation=self.ws.representation() or {},
            ontology=ontology,
            settings=self.settings,
        )
        if self._writer:  # a read-only engine rebuilds an outdated snapshot in memory only
            self.ws.save_graph(snapshot)
        return snapshot, CommittedState(DualGraph(snapshot), self._load_frame(records, vectors, ontology))

    @_healthy
    def upload(self, name: str, stream: BinaryIO, *, bins: str | None = None, categories: str | None = None) -> dict[str, Any]:
        """Store an uploaded file (at most ``MAX_UPLOAD_MB``) and register it (UPLOADED); processing is a separate call."""
        path = self.ws.save_upload(name, stream)
        if not path.exists():  # the workspace was reset while the file streamed in
            raise PipelineError("busy", "the workspace was reset during the upload; upload again")
        if path.stat().st_size > self.settings.max_upload_mb * 1024 * 1024:
            path.unlink()
            raise PipelineError("file_too_large", f"{name} exceeds {self.settings.max_upload_mb} MB (MAX_UPLOAD_MB)")
        return self.submit(path, filename=name, bins=bins, categories=categories)

    @_healthy
    def submit(
        self,
        path: Path,
        *,
        filename: str | None = None,
        bins: str | None = None,
        categories: str | None = None,
    ) -> dict[str, Any]:
        """Register an upload (status UPLOADED). Processing is a separate call.

        ``bins`` / ``categories`` = None means "use BIN_COLUMNS / CATEGORICAL_COLUMNS"
        (applied leniently); an explicit string (even "") overrides them strictly.
        """
        path = Path(path)
        created = utc_now()
        record = {
            "batch_id": f"B{created[:19].replace('-', '').replace(':', '')}-{uuid.uuid4().hex[:6]}",
            "dataset_id": None,
            "filename": filename or path.name,
            "source_path": str(path),
            "bins": bins,
            "categories": categories,
            "status": "UPLOADED",
            "stage_times": {"UPLOADED": created},
            "created_at": created,
            "metrics": {},
            "profile": {},
            "warnings": [],
            "error": None,
        }
        self.ws.save_batch(record)
        return record

    def ingest_file(self, path: Path) -> dict[str, Any]:
        """Submit and process one file in the calling thread (tests and scripts)."""
        return self.process(self.submit(path)["batch_id"])

    def process(self, batch_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.ws.load_batch(batch_id)
            if record is None:
                raise PipelineError("unknown_batch", batch_id)
            if record["status"] in TERMINAL:
                return record
            t0 = time.perf_counter()
            timings: dict[str, float] = {}
            try:
                with self._stage(record, "VALIDATING", timings, "validate_s"):
                    loaded = load_dataset(
                        Path(record["source_path"]),
                        self.settings.miner,
                        filename=record["filename"],
                        bins=record.get("bins"),
                        categories=record.get("categories"),
                    )
                    ds = record["dataset_id"] = loaded.dataset_id
                    record["warnings"] += loaded.warnings
                    duplicate = self.ws.ready_batch_for(ds)
                    if duplicate:
                        return self._skip(record, duplicate)

                with self._stage(record, "PROFILING", timings, "discover_s"):
                    result = run_discovery(loaded.frame, self.settings.miner, on_stage=lambda s: self._enter(record, s))
                    record["profile"] = {
                        **result.profile.to_dict(),
                        "derived_columns": loaded.derived_columns,
                        "bins": loaded.bins,
                        "categorical_overrides": loaded.categories,
                    }
                    self.ws.save_profile(ds, record["profile"])

                with self._stage(record, "VALIDATING_INSIGHTS", timings, "select_s"):
                    covers = covers_of(result)
                    insights = build_insights(result, self.settings.miner, dataset_id=ds, batch_id=batch_id, filename=record["filename"])
                    selection = select_insights(insights, self.settings.miner)
                    kept, refused = admit_insights(selection.kept, self.settings.min_insight_weight, self.settings.max_insights_per_batch)
                    rejections = [*result.rejections, *selection.rejections, *refused]
                    self.ws.save_rejections(ds, [asdict(r) for r in rejections])
                    if not kept:
                        raise PipelineError("no_viable_insights", f"No insight passed selection ({dict(Counter(r.reason for r in rejections))})")

                with self._stage(record, "EMBEDDING", timings, "embed_s"):
                    canon = [canonicalize(i, self.settings.topology, result.profile.rows) for i in kept]
                    enc, spec = self._encode(canon)
                    self.ws.check_representation(spec)  # validate only; recorded at commit

                with self.ws.transaction({"batch_id": batch_id}):
                    with self._stage(record, "UPDATING_ONTOLOGY", timings, "ontology_s"):
                        seq = record["batch_seq"] = self.ws.next_batch_seq()
                        ontology = self.ontology()
                        update = ontology.ingest(
                            enc["vector"], np.array([i.weight for i in kept]), [i.id for i in kept], batch_seq=seq, batch_id=batch_id
                        )

                    with self._stage(record, "BUILDING_GRAPH", timings, "graph_s"):
                        if update.row_ids[0] != len(self.ws.vectors()):  # the ontology numbers rows after the journal
                            raise RuntimeError(f"ontology row {update.row_ids[0]} does not continue the journal ({len(self.ws.vectors())} rows)")
                        patterns = pattern_records(kept, canon, update.row_ids, spec)
                        for rec in patterns:
                            rec["provenance"]["rows_ref"] = self.ws.covers_ref(ds, rec["id"])
                        self.ws.save_covers(ds, {i.id: covers[i.expression] for i in kept})
                        self._enter(record, "PERSISTING")
                        self._refuse_journaled(kept)
                        blocks = {**{k: enc[k] for k in BLOCKS}, "document": enc["document"], "pattern_ids": np.array([i.id for i in kept])}
                        self.ws.append(patterns, enc["vector"], update.activations, blocks, batch_id)
                        ontology.save()
                        self.ws.record_representation(spec)
                        self.ws.commit_batch_seq(seq)
                        snapshot, state = self._derive(ontology, pending=record)  # loaded before the commit point

                    record["metrics"] = batch_metrics(result, kept, rejections, update, spec, snapshot, timings, t0)
                    record["status"] = "READY"
                    record["stage_times"]["READY"] = utc_now()
                    self.ws.save_batch(record)  # the commit point: Workspace.recover keeps the writes of a READY batch
            except Exception as exc:  # failures before READY were rolled back by the transaction
                return self._fail(record, exc, t0)
            record["neo4j"] = self._publish(state, snapshot)  # committed: nothing below may turn READY into FAILED
            if record["neo4j"]["status"] == "failed":
                record["warnings"].append(f"graph persistence (Neo4j) failed: {record['neo4j']['error']}")
            self.ws.save_batch(record)
            return record

    @contextmanager
    def _stage(self, record: dict[str, Any], stage: str, timings: dict[str, float], key: str) -> Iterator[None]:
        """Enter ``stage`` and record the block's wall time as ``timings[key]``."""
        self._enter(record, stage)
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[key] = time.perf_counter() - start

    def _enter(self, record: dict[str, Any], stage: str) -> None:
        """Move the batch to ``stage`` and persist the record (the UI polls it)."""
        record["status"] = stage
        record["stage_times"][stage] = utc_now()
        self.ws.save_batch(record)
        log.info("[%s] %s", record["batch_id"], stage)

    def _skip(self, record: dict[str, Any], duplicate: dict[str, Any]) -> dict[str, Any]:
        """Same content + options already READY: idempotent skip."""
        record["status"] = "SKIPPED"
        record["duplicate_of"] = duplicate["batch_id"]
        record["warnings"].append(f"dataset already ingested in {duplicate['batch_id']} (idempotent skip)")
        self.ws.save_batch(record)
        return record

    def _encode(self, canon: list[CanonicalInsight]) -> tuple[dict[str, np.ndarray], EmbeddingSpec]:
        """Tripartite vectors + their spec; any encoder error becomes ``embedding_failure``."""
        try:
            enc = self.encoder.encode(canon)
            # embedded once here, so the per-question naive text baseline is a dot product, not a corpus pass
            enc["document"] = self.encoder.embedder.embed([c.document() for c in canon])
            return enc, self.encoder.spec
        except Exception as exc:
            raise PipelineError("embedding_failure", f"Embedding failed: {exc}") from exc

    def _refuse_journaled(self, kept: list[Insight]) -> None:
        """Pattern ids are deterministic: a second copy would double-count the ontology."""
        journaled = {r["id"] for r in self.ws.patterns()}
        dupes = [i.id for i in kept if i.id in journaled]
        if dupes:
            raise PipelineError("duplicate_patterns", f"{len(dupes)} patterns already journaled (e.g. {dupes[:3]})")

    def _fail(self, record: dict[str, Any], exc: Exception, t0: float) -> dict[str, Any]:
        """Record the failure of a batch (called inside ``except``; ``Workspace.transaction`` has rolled its writes back)."""
        if self.ws.pending_path.exists():  # the rollback failed too (logged): its marker blocks every write until a restart
            record["warnings"].append("rollback failed; restart the service to retry it, no write runs until then")
        code = getattr(exc, "code", "internal_error")
        record["status"] = "FAILED"
        record["failed_stage"] = list(record["stage_times"])[-1]
        record["error"] = {"code": code, "message": str(exc)}
        if code == "internal_error":
            record["error"]["trace"] = traceback.format_exc(limit=6)
        record["metrics"]["processing_duration_s"] = time.perf_counter() - t0
        self.ws.save_batch(record)
        log.warning("[%s] FAILED %s: %s", record["batch_id"], code, exc)
        return record

    def _publish(self, state: CommittedState, snapshot: dict[str, Any]) -> dict[str, Any]:
        """After a commit (batch or deletion): readers switch to the new state in one assignment (the literal catalog
        follows on the next question) and the optional Neo4j mirror is synced; returns the mirror status."""
        self._state = state
        return self.sync_neo4j(snapshot)

    @_healthy
    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Remove a dataset: its batches, patterns, vectors, memberships and artefacts (docs/06_graph_and_storage.md §6.8).

        Anchors that keep a member, or that are still RELATED_TO another anchor, survive (their
        centroids are not un-averaged); the others go. Journal rows are renumbered, the snapshot is
        rebuilt and the Neo4j mirror synced. All or nothing (``Workspace.transaction``): the checkpoint holds the
        state, the journal, the snapshot and the moved-aside files; an error rolls back at once, and a crash
        leaves the ``pending.json`` marker, so the next writer start rolls back (``Workspace.recover``).
        ``dataset_id`` may also be the batch id of a batch that failed before its dataset id was known.
        """
        with self._lock:
            batches = [b for b in self.ws.list_batches() if dataset_id in (b.get("dataset_id"), b["batch_id"])]
            if not batches:
                raise PipelineError("unknown_dataset", dataset_id)
            if any(b["status"] not in TERMINAL for b in batches):
                raise PipelineError("busy", f"dataset {dataset_id} has a batch in progress")
            ontology = self.ontology()
            records, vectors = self.ws.patterns(), self.ws.vectors()
            kept = [r for r in records if r["dataset_id"] != dataset_id]
            row_of = {r["row_id"]: i for i, r in enumerate(kept)}  # journal order, renumbered
            acts = [{**a, "row_id": row_of[a["row_id"]]} for a in self.ws.activations() if a["row_id"] in row_of]
            with self.ws.transaction({"dataset_id": dataset_id}, journal=True) as cp:  # commit point: the marker goes
                self.ws.remove_dataset(dataset_id, batches, Path(cp["dir"]) / "removed")  # first: the snapshot must not see them
                dropped = ontology.forget(acts, len(kept))
                self.ws.rewrite_journal([{**r, "row_id": row_of[r["row_id"]]} for r in kept], vectors[[r["row_id"] for r in kept]], acts)
                ontology.save()
                snapshot, state = self._derive(ontology)
            return {
                "dataset_id": dataset_id,
                "batches": [b["batch_id"] for b in batches],
                "patterns_removed": len(records) - len(kept),
                "anchors_removed": dropped,
                "neo4j": self._publish(state, snapshot),
            }

    def reset(self) -> dict[str, Any]:
        """Delete the workspace; with ``NEO4J_ENABLED`` the mirror is cleared too (failure = warning). Refused with
        ``busy``, never queued, while a batch or a deletion runs or an upload waits. Also the way out of a degraded start."""
        if not self._lock.acquire(blocking=False):
            raise PipelineError("busy", "a batch or a dataset deletion is running")
        try:
            busy = [b["batch_id"] for b in self.ws.unfinished_batches()]
            if busy:
                raise PipelineError("busy", f"batches in progress: {busy}")
            self.ws.reset()
            self._open()  # the empty workspace
            return {"neo4j": self.sync_neo4j({"nodes": [], "edges": []})}
        finally:
            self._lock.release()

    def startup_sync(self) -> dict[str, Any]:
        """Bring the Neo4j mirror up to the committed snapshot after a start (a commit made while it was down).
        Never from a degraded or empty workspace: that would wipe a mirror for no operator action."""
        with self._lock:  # never interleaved with a deletion's or a reset's own sync
            if self.problem or not self.committed().graph.insights:
                log.warning("skipping Neo4j sync: workspace degraded or empty")
                return {"status": "skipped"}
            return self.sync_neo4j(self.committed().graph.snapshot)

    def sync_neo4j(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Make the Neo4j mirror equal to ``snapshot`` when ``NEO4J_ENABLED``."""
        if not self.settings.neo4j_enabled:
            return {"status": "disabled"}
        from insight_graph_service.core.neo4j_mirror import publish_snapshot

        try:
            return {"status": "ok", **publish_snapshot(snapshot, self.settings)}
        except Exception as exc:  # the mirror is optional: report, never fail the caller
            log.warning("Neo4j sync failed: %s", exc)
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def _recover(self) -> None:
        """Under the writer lock no other writer is alive: an unfinished transaction is rolled back (unless its
        batch had committed) and every unfinished batch fails — also when the rollback cannot finish, so a reset is
        never refused for a batch that no process runs."""
        try:
            intent = self.ws.recover()
            if intent is not None:
                log.warning("an interrupted write was rolled back: %s", intent)
        finally:
            for record in self.ws.unfinished_batches():
                record["status"] = "FAILED"
                record["error"] = {"code": "interrupted", "message": "processing was interrupted; state rolled back"}
                self.ws.save_batch(record)
