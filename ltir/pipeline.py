"""End-to-end dataset lifecycle (docs/09_operations.md).

UPLOADED -> VALIDATING -> PROFILING -> DISCOVERING -> VALIDATING_INSIGHTS -> EMBEDDING
-> UPDATING_ONTOLOGY -> BUILDING_GRAPH -> PERSISTING -> READY   (| FAILED | SKIPPED)

``Engine`` is the single stateful service used by the CLI and the web app. Batches
run strictly one at a time (the ontology is stateful); every batch either commits
completely or is rolled back to the pre-batch checkpoint.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ltir.canonical import canonicalize
from ltir.config import Config
from ltir.discovery import DiscoveryResult, build_insights, covers_of, run_discovery
from ltir.encoder import InsightEncoder, TextEmbedder, make_text_embedder
from ltir.graph import SNAPSHOT_VERSION, DualGraph, build_snapshot
from ltir.ingestion import load_dataset
from ltir.models import CanonicalInsight, EmbeddingSpec, Insight
from ltir.ontology import LatentOntology, OntologyUpdate
from ltir.quality import SelectionResult, select_insights
from ltir.store import Workspace, acquire_writer_lock, atomic_write_json, utc_now

if TYPE_CHECKING:
    from ltir.llm import OpenAICompatibleLLM
    from ltir.query import LiteralCatalog

log = logging.getLogger("ltir.pipeline")

TERMINAL = {"READY", "FAILED", "SKIPPED"}
BLOCKS = ("scope", "target", "phenomenon")  # tripartite parts persisted beside the composite vector


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LatentFrame:
    """Committed vectors of the single insight frame: pattern id -> unit vector, attractor id -> centroid,
    plus the canonical-document embeddings the naive text baseline compares questions with."""

    patterns: dict[str, np.ndarray]
    attractors: dict[int, np.ndarray]
    documents: dict[str, np.ndarray]

    @classmethod
    def load(cls, ws: Workspace, ontology: LatentOntology) -> LatentFrame:
        vectors = ws.vectors()
        st = ontology.store
        records = ws.patterns()
        return cls(
            patterns={r["id"]: vectors[r["row_id"]] for r in records} if len(vectors) else {},
            attractors={int(c): st.embeddings[i].copy() for i, c in enumerate(st.concept_ids)},
            documents=ws.document_vectors({r["batch_id"] for r in records}),
        )


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
        llm: OpenAICompatibleLLM | None = None,
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
        self._llm = llm
        self._lock = threading.RLock()
        self._writer = recover  # holds the writer lock: may write derived files (document back-fill)
        self._documents_lock = threading.Lock()  # the document back-fill, never the batch lock
        self._sphere_pool: ThreadPoolExecutor | None = None  # standalone sphere export, off the batch thread
        self._sphere_generation = 0  # the latest requested export; older queued ones are skipped
        self._graph: DualGraph | None = None
        self._frame: LatentFrame | None = None
        self._catalog: tuple[DualGraph, LiteralCatalog | None] | None = None  # (graph, its literal catalog)
        self._catalog_lock = threading.Lock()  # never the batch lock: a question does not wait for a batch
        if recover:
            self._recover()

    @property
    def llm(self) -> OpenAICompatibleLLM:
        """The configured client, created on first use. Tests inject any object with
        ``model``, ``generate(system, user) -> LLMResponse`` and ``health(fresh=...)``."""
        if self._llm is None:
            from ltir.llm import OpenAICompatibleLLM

            self._llm = OpenAICompatibleLLM(self.config)
        return self._llm

    def graph(self) -> DualGraph:
        """Committed graph. Readers never touch the journal, so they cannot race a batch."""
        if self._graph is None:
            with self._lock:
                if self._graph is None:
                    snap = self.ws.load_graph()
                    if snap and snap.get("version") != SNAPSHOT_VERSION:
                        self.rebuild_graph()  # derived data; journals stay the source of truth
                    else:
                        self._graph = DualGraph(snap) if snap else DualGraph.empty()
        return self._graph

    def catalog(self) -> LiteralCatalog | None:
        """The literal catalog of the committed graph (docs/07 §7.1.1), built on the first question after a commit.
        Vectors of known literals come from ``graph/literals.npz`` (same fingerprint); a writer saves new ones there."""
        graph = self.graph()
        with self._catalog_lock:
            if self._catalog is None or self._catalog[0] is not graph:
                from ltir.query import build_catalog

                fingerprint = self.encoder.spec.fingerprint
                stored = self.ws.load_literals()
                known = dict(zip(map(str, stored["texts"]), stored["vectors"])) if stored and str(stored["fingerprint"]) == fingerprint else {}
                catalog = build_catalog(graph, self.encoder.embedder, known)
                if catalog is not None and self._writer and not known.keys() >= set(catalog.texts):
                    self.ws.save_literals({"texts": np.array(catalog.texts), "vectors": catalog.vectors, "fingerprint": np.array(fingerprint)})
                self._catalog = (graph, catalog)
            return self._catalog[1]

    def document_vectors(self) -> dict[str, np.ndarray]:
        """Canonical-document embedding of every committed pattern (stored at ingest).

        A pattern without a stored vector (a batch written before documents were embedded) is
        embedded once: the vector stays on the committed frame and, from a writer engine, is saved
        into the batch's blocks file, so no later question pays for it again.
        """
        frame, graph = self.frame(), self.graph()
        if all(n["id"] in frame.documents for n in graph.of_kind("Pattern")):
            return frame.documents
        with self._documents_lock:
            missing = [n["id"] for n in graph.of_kind("Pattern") if n["id"] not in frame.documents]
            if missing:
                filled = dict(zip(missing, self.encoder.embedder.embed([graph.canonical_document(i) for i in missing])))
                frame.documents.update(filled)
                if self._writer:
                    by_batch: dict[str, list[str]] = defaultdict(list)
                    for record in self.ws.patterns():  # journal order = the row order of the batch's blocks
                        if record["id"] in frame.documents:
                            by_batch[record["batch_id"]].append(record["id"])
                    for batch_id, ids in by_batch.items():
                        if any(i in filled for i in ids):
                            self.ws.add_document_vectors(batch_id, ids, np.stack([frame.documents[i] for i in ids]))
        return frame.documents

    def frame(self) -> LatentFrame:
        """Committed pattern vectors and attractor centroids (retrieval and the sphere)."""
        if self._frame is None:
            with self._lock:
                if self._frame is None:
                    self._frame = LatentFrame.load(self.ws, self.ontology())
        return self._frame

    def ontology(self) -> LatentOntology:
        return LatentOntology(self.config, self.ws.state_dir)

    def ask(self, question: str, **kwargs: Any):
        from ltir.qa import answer_question

        return answer_question(self, question, **kwargs)

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
                self._enter(record, "VALIDATING")
                with _clock(timings, "validate_s"):
                    loaded = load_dataset(
                        Path(record["source_path"]),
                        self.config,
                        filename=record["filename"],
                        bins=record.get("bins"),
                        categories=record.get("categories"),
                    )
                    record["dataset_id"] = loaded.dataset_id
                    record["warnings"] += loaded.warnings
                    duplicate = self.ws.ready_batch_for(loaded.dataset_id)
                    if duplicate:
                        return self._skip(record, duplicate)
                    ds_dir = self._keep_source(record, loaded.dataset_id)

                self._enter(record, "PROFILING")
                with _clock(timings, "discover_s"):
                    result = run_discovery(loaded.frame, self.config, on_stage=lambda s: self._enter(record, s))
                    record["profile"] = {
                        **result.profile.to_dict(),
                        "derived_columns": loaded.derived_columns,
                        "bins": loaded.bins,
                        "categorical_overrides": loaded.categories,
                    }
                    atomic_write_json(ds_dir / "profile.json", record["profile"])

                self._enter(record, "VALIDATING_INSIGHTS")
                with _clock(timings, "select_s"):
                    covers = covers_of(result)
                    insights = build_insights(result, self.config, dataset_id=loaded.dataset_id, batch_id=batch_id, filename=record["filename"])
                    selection = select_insights(insights, self.config)
                    atomic_write_json(ds_dir / "rejections.json", [asdict(r) for r in [*result.rejections, *selection.rejections]])
                    kept = selection.kept
                    if not kept:
                        raise PipelineError("no_viable_insights", f"No insight passed selection ({selection.stats})")

                self._enter(record, "EMBEDDING")
                with _clock(timings, "embed_s"):
                    canon = [canonicalize(i, self.config, result.profile.rows) for i in kept]
                    enc, spec = self._encode(canon)
                    self.ws.check_representation(spec)  # validate only; recorded at commit

                self._enter(record, "UPDATING_ONTOLOGY")
                with self.ws.transaction({"batch_id": batch_id}):
                    with _clock(timings, "ontology_s"):
                        seq = record["batch_seq"] = self.ws.next_batch_seq()
                        ontology = self.ontology()
                        update = ontology.ingest(
                            enc["vector"], np.array([i.weight for i in kept]), [i.id for i in kept], batch_seq=seq, batch_id=batch_id
                        )

                    self._enter(record, "BUILDING_GRAPH")
                    with _clock(timings, "graph_s"):
                        patterns = _pattern_records(kept, canon, update.row_ids, spec)
                        self.ws.save_covers(loaded.dataset_id, {i.id: covers[i.expression] for i in kept})
                        self._enter(record, "PERSISTING")
                        self._refuse_journaled(kept)
                        blocks = {**{k: enc[k] for k in BLOCKS}, "document": enc["document"], "pattern_ids": np.array([i.id for i in kept])}
                        self.ws.append(patterns, enc["vector"], update.activations, blocks, batch_id)
                        ontology.save()
                        self.ws.record_representation(spec)
                        self.ws.commit_batch_seq(seq)
                        snapshot = build_snapshot(self.ws, ontology, self.config, pending=record)
                        self.ws.save_graph(snapshot)
                        graph, frame = DualGraph(snapshot), LatentFrame.load(self.ws, ontology)  # loaded before the commit point

                    record["metrics"] = _batch_metrics(result, selection, update, spec, snapshot, timings, t0)
                    record["status"] = "READY"
                    record["stage_times"]["READY"] = utc_now()
                    self.ws.save_batch(record)  # the commit point: Workspace.recover keeps the writes of a READY batch
            except Exception as exc:  # failures before READY were rolled back by the transaction
                return self._fail(record, exc, t0)
            record["neo4j"] = self._publish(graph, frame, snapshot)  # committed: nothing below may turn READY into FAILED
            if record["neo4j"]["status"] == "failed":
                record["warnings"].append(f"graph persistence (Neo4j) failed: {record['neo4j']['error']}")
            self.ws.save_batch(record)
            return record

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

    def _keep_source(self, record: dict[str, Any], dataset_id: str) -> Path:
        """Copy the upload next to the dataset artifacts (provenance); returns the dataset dir."""
        ds_dir = self.ws.dataset_dir(dataset_id)
        copy = ds_dir / f"source{Path(record['filename']).suffix.lower() or '.csv'}"
        if not copy.exists():
            shutil.copyfile(record["source_path"], copy)
        return ds_dir

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

    def _publish(self, graph: DualGraph, frame: LatentFrame, snapshot: dict[str, Any]) -> dict[str, Any]:
        """After a commit (batch or deletion): readers switch to the new state (the literal catalog follows on the
        next question), the sphere export is queued and the optional Neo4j mirror synced; returns the mirror status."""
        self._graph, self._frame = graph, frame
        self._export_sphere()
        return self.sync_neo4j(snapshot)

    def _export_sphere(self) -> None:
        """Queue the standalone 3D sphere (KernelPCA + Plotly) on its own thread; callers hold ``_lock``.

        The writer returns at once, so the next upload never waits on a plot; when several commits
        come quickly only the latest export runs. ``/api/sphere`` builds the page on demand.
        """
        if not self.config.sphere_export:
            return
        self._sphere_generation += 1
        if self._sphere_pool is None:
            self._sphere_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltir-sphere")
        self._sphere_pool.submit(self._write_sphere, self._sphere_generation)

    def _write_sphere(self, generation: int) -> None:
        """Export unless a newer export or a reset superseded it; a failure is logged, never raised."""
        if generation != self._sphere_generation:
            return
        try:
            from ltir.sphere import export_sphere

            page = export_sphere(self)
        except Exception as exc:
            log.warning("sphere export failed: %s", exc)
            return
        with self._lock:  # reset holds it
            if generation != self._sphere_generation:  # superseded while it ran (a reset must leave nothing behind)
                page.unlink(missing_ok=True)

    def rebuild_graph(self) -> dict[str, Any]:
        with self._lock:
            self.ws.check_versions()
            ontology = self.ontology()
            snapshot = build_snapshot(self.ws, ontology, self.config)
            self.ws.save_graph(snapshot)
            self._graph, self._frame = DualGraph(snapshot), LatentFrame.load(self.ws, ontology)
            return snapshot

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Remove a dataset: its batches, patterns, vectors, memberships and artefacts (docs/06 §6.8).

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
                snapshot = build_snapshot(self.ws, ontology, self.config)
                self.ws.save_graph(snapshot)
                graph, frame = DualGraph(snapshot), LatentFrame.load(self.ws, ontology)
            return {
                "dataset_id": dataset_id,
                "batches": [b["batch_id"] for b in batches],
                "patterns_removed": len(records) - len(kept),
                "anchors_removed": dropped,
                "neo4j": self._publish(graph, frame, snapshot),
            }

    def reset(self) -> dict[str, Any]:
        """Delete the workspace; with ``NEO4J_ENABLED`` the mirror is cleared too (failure = warning)."""
        with self._lock:
            self._sphere_generation += 1  # drop a queued export of the old workspace
            self.ws.reset()
            self._graph = self._frame = self._catalog = None
            return {"neo4j": self.sync_neo4j({"nodes": [], "edges": []})}

    def sync_neo4j(self, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make the Neo4j mirror equal to ``snapshot`` (default: the committed one) when ``NEO4J_ENABLED``."""
        if not self.config.neo4j_enabled:
            return {"status": "disabled"}
        from ltir.neo4j_sink import publish_snapshot

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
        for record in self.ws.list_batches():
            if record["status"] not in TERMINAL:
                record["status"] = "FAILED"
                record["error"] = {"code": "interrupted", "message": "processing was interrupted; state rolled back"}
                self.ws.save_batch(record)
