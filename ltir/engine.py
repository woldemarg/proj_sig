"""The application service (docs/09_operations.md, docs/12_architecture.md).

``Engine`` orchestrates the analytical core over one workspace for the CLI and the web app:
the batch lifecycle

UPLOADED -> VALIDATING -> PROFILING -> DISCOVERING -> VALIDATING_INSIGHTS -> EMBEDDING
-> UPDATING_ONTOLOGY -> BUILDING_GRAPH -> PERSISTING -> READY   (| FAILED | SKIPPED)

(strictly one batch at a time: the ontology is stateful; each commits completely or is rolled
back), dataset deletion, the committed read state, retrieval (``search``) and chat answers
(``ask``: retrieval, then the LLM).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ltir.analysis.canonical import canonicalize
from ltir.analysis.discovery import DiscoveryResult, build_insights, covers_of, run_discovery
from ltir.analysis.encoder import InsightEncoder, TextEmbedder, make_text_embedder
from ltir.analysis.graph import SNAPSHOT_VERSION, DualGraph, build_snapshot
from ltir.analysis.ingestion import load_dataset
from ltir.analysis.ontology import LatentOntology, OntologyUpdate
from ltir.analysis.quality import SelectionResult, select_insights
from ltir.answering import QAResult, answer, empty_answer
from ltir.config import Config
from ltir.llm_client import ChatModel, OpenAICompatibleLLM
from ltir.models import CanonicalInsight, EmbeddingSpec, Insight, LatentFrame, utc_now
from ltir.retrieval.question import build_catalog
from ltir.retrieval.search import CommittedState, SearchResult, search
from ltir.storage.workspace import TERMINAL, Workspace, acquire_writer_lock

log = logging.getLogger(__name__)

BLOCKS = ("scope", "target", "phenomenon")  # tripartite parts persisted beside the composite vector


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@contextmanager
def _clock(timings: dict[str, float], key: str) -> Iterator[None]:
    """Record the wall time of the enclosed block as ``timings[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - start


def _pattern_records(kept: list[Insight], canon: list[CanonicalInsight], row_ids: list[int], spec: EmbeddingSpec) -> list[dict[str, Any]]:
    """Journal records: the Insight plus its journal row, canonical form and vector contract."""
    embedding = {"fingerprint": spec.fingerprint, "model_id": spec.model_id, "dim": spec.dim, "representation_version": spec.representation_version}
    records = []
    for ins, can, row_id in zip(kept, canon, row_ids):
        rec = ins.to_record()
        rec.update(
            row_id=row_id,
            canonical={**asdict(can), "document": can.document(), "components": [list(c) for c in can.components]},
            embedding=embedding,
        )
        records.append(rec)
    return records


def _batch_metrics(
    result: DiscoveryResult,
    selection: SelectionResult,
    update: OntologyUpdate,
    spec: EmbeddingSpec,
    snapshot: dict[str, Any],
    timings: dict[str, float],
    t0: float,
) -> dict[str, Any]:
    """Batch-record metrics (docs/09_operations.md §9.5)."""
    kept, om, graph = selection.kept, update.metrics, snapshot["stats"]
    pruned = Counter(r.reason for r in [*result.rejections, *selection.rejections])
    return {
        "input_rows": result.profile.rows,
        "columns": result.profile.columns,
        "numeric_targets": len(result.profile.numerics),
        "dimensions": len(result.profile.selected_dimensions),
        "search_space": result.profile.search_space_size,
        "pass1_subgroups": result.pass1_subgroups,
        "candidate_patterns": len(result.candidates),
        "validated_candidates": len(result.validated),
        "validated_insights": len(kept),
        "pruned": dict(pruned),
        "pruned_total": sum(pruned.values()),
        "avg_insight_support": float(np.mean([i.support for i in kept])),
        "avg_insight_weight": float(np.mean([i.weight for i in kept])),
        "embedding_count": len(kept),
        "embedding_dim": spec.dim,
        "attractors_total": om["total_concepts"],
        "attractors_new": len(update.new_attractors),
        "orphan_rate": om["orphan_rate"],
        "activation_count": om["activation_count"],
        "soft_merged": om["soft_merged"],
        "centroid_drift": om["centroid_drift"],
        "adaptive_thresh": om["adaptive_thresh"],
        "max_concept_density_pct": om["max_concept_density_pct"],
        "density_threshold": om["density_threshold"],
        "damped_attractors": om["damped_attractors"],
        "max_centroid_step": om["max_centroid_step"],
        "clamped_attractors": om["clamped_attractors"],
        "avg_attractor_degree": graph["avg_attractor_degree"],
        "graph_edges": graph["edges"],
        "edge_counts": graph["edge_counts"],
        "graph_patterns": graph["patterns"],
        "ontology_warnings": om["warnings"],
        "timings": timings,
        "processing_duration_s": time.perf_counter() - t0,
    }


class Engine:
    def __init__(
        self,
        config: Config,
        *,
        embedder: TextEmbedder | None = None,
        llm: ChatModel | None = None,
        recover: bool = True,
    ) -> None:
        """A writer (``recover=True``: the web app, CLI demo/ingest/reset/rebuild-graph) takes the
        workspace's writer lock — ``WorkspaceBusy`` if another process holds it — and then
        recovers unfinished batches. ``recover=False`` for read-only callers (CLI status/query)."""
        self.config = config
        if recover:
            acquire_writer_lock(config.workspace_dir)
        self.ws = Workspace(config)
        self.encoder = InsightEncoder(embedder or make_text_embedder(config), config)
        self.llm: ChatModel = llm or OpenAICompatibleLLM(config)  # building the client does no I/O
        self._lock = threading.RLock()  # batches, deletion, reset and rebuild: one write at a time
        self._writer = recover  # holds the writer lock: may save derived files (literal cache, document back-fill)
        self._prepare_lock = threading.Lock()  # catalog build and document back-fill: never the batch lock
        if recover:
            self._recover()
        self._state = self._load_state()  # the committed graph, vectors and literal catalog, replaced whole per commit

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

    def _load_state(self) -> CommittedState:
        """The read state of the saved snapshot; a snapshot of another version is rebuilt (derived data: the journals
        stay the source of truth; ``RepresentationMismatch`` names the migration when they are outdated too)."""
        snapshot = self.ws.load_graph()
        if snapshot and snapshot.get("version") != SNAPSHOT_VERSION:
            self.ws.check_versions()
            return self._derive(self.ontology())[1]
        graph = DualGraph(snapshot) if snapshot else DualGraph.empty()
        return CommittedState(graph, self._load_frame(self.ws.patterns(), self.ws.vectors(), self.ontology()))

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
                if self._writer:
                    self.ws.save_document_vectors(documents, set(filled))
            if state.catalog is None and graph.insights:
                fingerprint = self.encoder.spec.fingerprint
                known = self.ws.load_literals(fingerprint)
                state.catalog = build_catalog(graph, self.encoder.embedder, known)
                if self._writer and known.keys() != set(state.catalog.texts):  # new literals, or a deleted dataset's gone
                    self.ws.save_literals(fingerprint, state.catalog.texts, state.catalog.vectors)

    def ontology(self) -> LatentOntology:
        return LatentOntology(self.config, self.ws.state_dir)

    def search(self, question: str) -> SearchResult:
        """Retrieval only: the evidence for ``question`` over the committed graph (no LLM; docs/07_question_answering.md §7.1–7.4)."""
        return self._search(question, self.committed())

    def ask(self, question: str, *, use_llm: bool = True) -> QAResult:
        """Chat answer: ``search``, then the LLM (or the evidence-only summary); logged to ``logs/queries.jsonl``."""
        state = self.committed()
        if not state.graph.insights:
            return empty_answer(question)
        qa = answer(self._search(question, state), self.llm if use_llm else None)
        self.ws.log_query(
            {
                "at": utc_now(),
                "question": question,
                "mode": qa.answer_mode,
                "metrics": qa.metrics,
                "seeds": qa.highlight["seeds"],
                "evidence": qa.highlight["evidence"],
                "citations": qa.citations,
            }
        )
        return qa

    def _search(self, question: str, state: CommittedState) -> SearchResult:
        start = time.perf_counter()
        self.ws.check_representation(self.encoder.spec)  # query and stored vectors must share one frame
        self._prepare(state)  # the literal catalog; every document vector for the naive baseline
        found = search(question, state, self.encoder, self.config)
        found.seconds = time.perf_counter() - start  # with the preparation of the first question after a commit
        return found

    def _load_frame(self, records: list[dict[str, Any]], vectors: np.ndarray, ontology: LatentOntology) -> LatentFrame:
        st = ontology.store
        return LatentFrame(
            patterns={r["id"]: vectors[r["row_id"]] for r in records} if len(vectors) else {},
            attractors={int(c): st.embeddings[i].copy() for i, c in enumerate(st.concept_ids)},
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
            config=self.config,
        )
        self.ws.save_graph(snapshot)
        return snapshot, CommittedState(DualGraph(snapshot), self._load_frame(records, vectors, ontology))

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

    def ingest_file(
        self,
        path: Path,
        *,
        filename: str | None = None,
        bins: str | None = None,
        categories: str | None = None,
    ) -> dict[str, Any]:
        return self.process(self.submit(path, filename=filename, bins=bins, categories=categories)["batch_id"])

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
                        self.config,
                        filename=record["filename"],
                        bins=record.get("bins"),
                        categories=record.get("categories"),
                    )
                    ds = record["dataset_id"] = loaded.dataset_id
                    record["warnings"] += loaded.warnings
                    duplicate = self.ws.ready_batch_for(ds)
                    if duplicate:
                        return self._skip(record, duplicate)
                    self.ws.keep_source(ds, Path(record["source_path"]), record["filename"])

                with self._stage(record, "PROFILING", timings, "discover_s"):
                    result = run_discovery(loaded.frame, self.config, on_stage=lambda s: self._enter(record, s))
                    record["profile"] = {
                        **result.profile.to_dict(),
                        "derived_columns": loaded.derived_columns,
                        "bins": loaded.bins,
                        "categorical_overrides": loaded.categories,
                    }
                    self.ws.save_profile(ds, record["profile"])

                with self._stage(record, "VALIDATING_INSIGHTS", timings, "select_s"):
                    covers = covers_of(result)
                    insights = build_insights(result, self.config, dataset_id=ds, batch_id=batch_id, filename=record["filename"])
                    selection = select_insights(insights, self.config)
                    self.ws.save_rejections(ds, [asdict(r) for r in [*result.rejections, *selection.rejections]])
                    kept = selection.kept
                    if not kept:
                        raise PipelineError("no_viable_insights", f"No insight passed selection ({selection.stats})")

                with self._stage(record, "EMBEDDING", timings, "embed_s"):
                    canon = [canonicalize(i, self.config, result.profile.rows) for i in kept]
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
                        patterns = _pattern_records(kept, canon, update.row_ids, spec)
                        self.ws.save_covers(ds, {i.id: covers[i.expression] for i in kept})
                        self._enter(record, "PERSISTING")
                        self._refuse_journaled(kept)
                        blocks = {**{k: enc[k] for k in BLOCKS}, "document": enc["document"], "pattern_ids": np.array([i.id for i in kept])}
                        self.ws.append(patterns, enc["vector"], update.activations, blocks, batch_id)
                        ontology.save()
                        self.ws.record_representation(spec)
                        self.ws.commit_batch_seq(seq)
                        snapshot, state = self._derive(ontology, pending=record)  # loaded before the commit point

                    record["metrics"] = _batch_metrics(result, selection, update, spec, snapshot, timings, t0)
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
        with _clock(timings, key):
            yield

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
            record["warnings"].append("rollback failed; restart the writer (web app or CLI) to retry it, no write runs until then")
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

    def rebuild_graph(self) -> dict[str, Any]:
        with self._lock:
            self.ws.check_versions()
            snapshot, self._state = self._derive(self.ontology())
            return snapshot

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
        ``busy``, never queued, while a batch or a deletion runs or an upload waits."""
        if not self._lock.acquire(blocking=False):
            raise PipelineError("busy", "a batch or a dataset deletion is running")
        try:
            busy = [b["batch_id"] for b in self.ws.unfinished_batches()]
            if busy:
                raise PipelineError("busy", f"batches in progress: {busy}")
            self.ws.reset()
            self._state = self._load_state()  # the empty workspace
            return {"neo4j": self.sync_neo4j({"nodes": [], "edges": []})}
        finally:
            self._lock.release()

    def sync_neo4j(self, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make the Neo4j mirror equal to ``snapshot`` (default: the committed one) when ``NEO4J_ENABLED``."""
        if not self.config.neo4j_enabled:
            return {"status": "disabled"}
        from ltir.storage.neo4j_mirror import publish_snapshot

        snapshot = snapshot if snapshot is not None else self.ws.load_graph() or {"nodes": [], "edges": []}
        try:
            return {"status": "ok", **publish_snapshot(snapshot, self.config)}
        except Exception as exc:  # the mirror is optional: report, never fail the caller
            log.warning("Neo4j sync failed: %s", exc)
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def _recover(self) -> None:
        """Under the writer lock no other writer is alive: an unfinished transaction is rolled back (unless its
        batch had committed) and every unfinished batch fails."""
        intent = self.ws.recover()
        if intent is not None:
            log.warning("an interrupted write was rolled back: %s", intent)
        for record in self.ws.unfinished_batches():
            record["status"] = "FAILED"
            record["error"] = {"code": "interrupted", "message": "processing was interrupted; state rolled back"}
            self.ws.save_batch(record)
