"""End-to-end dataset lifecycle (docs/09_operations.md).

UPLOADED -> VALIDATING -> PROFILING -> DISCOVERING -> VALIDATING_INSIGHTS -> EMBEDDING
-> UPDATING_ONTOLOGY -> BUILDING_GRAPH -> PERSISTING -> READY   (| FAILED | SKIPPED)

``Engine`` is the single stateful service used by the CLI and the web app. Batches
run strictly one at a time (the ontology is stateful); every batch either commits
completely or is rolled back to the pre-batch checkpoint.
"""

from __future__ import annotations

import logging
import os
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

log = logging.getLogger("ltir.pipeline")

STAGES = [
    "UPLOADED",
    "VALIDATING",
    "PROFILING",
    "DISCOVERING",
    "VALIDATING_INSIGHTS",
    "EMBEDDING",
    "UPDATING_ONTOLOGY",
    "BUILDING_GRAPH",
    "PERSISTING",
    "READY",
]
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
        self._sphere_lock = threading.Lock()
        self._graph: DualGraph | None = None
        self._frame: LatentFrame | None = None
        if recover:
            self._recover_interrupted()

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

    def _refresh_caches(self, snapshot: dict[str, Any], ontology: LatentOntology) -> None:
        """Swap in the committed state for readers."""
        self._graph = DualGraph(snapshot)
        self._frame = LatentFrame.load(self.ws, ontology)

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
            record["owner_pid"] = os.getpid()
            t0 = time.perf_counter()
            timings: dict[str, float] = {}
            checkpoint = None
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
                with _clock(timings, "ontology_s"):
                    checkpoint = record["checkpoint"] = self.ws.checkpoint()
                    seq = record["batch_seq"] = self.ws.next_batch_seq()
                    self.ws.save_batch(record)
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

                record["metrics"] = _batch_metrics(result, selection, update, spec, snapshot, timings, t0)
                record["status"] = "READY"
                record["stage_times"]["READY"] = utc_now()
                record.pop("checkpoint", None)
                self.ws.save_batch(record)  # READY is durable before the checkpoint goes
                self.ws.discard_checkpoint(checkpoint)
                checkpoint = None
                self._refresh_caches(snapshot, ontology)
            except Exception as exc:  # failures before READY roll back and become FAILED
                return self._fail(record, exc, checkpoint, t0)
            # Journals, snapshot and READY are committed. Publish and sphere export
            # must not be able to rewrite that status.
            self._publish_neo4j(record, snapshot)
            self._export_sphere(record)
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

    def _fail(self, record: dict[str, Any], exc: Exception, checkpoint: dict[str, Any] | None, t0: float) -> dict[str, Any]:
        """Roll back to the pre-batch checkpoint and record the failure (called inside ``except``)."""
        if checkpoint is not None:
            try:
                self.ws.rollback(checkpoint)
                self.ws.discard_checkpoint(checkpoint)
            except Exception:  # pragma: no cover - surfaced in the record
                record["warnings"].append("rollback failed: " + traceback.format_exc(limit=2))
        record.pop("checkpoint", None)
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

    def _publish_neo4j(self, record: dict[str, Any], snapshot: dict[str, Any]) -> None:
        """Optional mirror; failure never invalidates the committed batch."""
        if not self.config.neo4j_enabled:
            record["neo4j"] = {"status": "disabled"}
        else:
            try:
                from ltir.neo4j_sink import publish_snapshot

                record["neo4j"] = {"status": "ok", **publish_snapshot(snapshot, self.config)}
            except Exception as exc:
                record["neo4j"] = {"status": "failed", "error": str(exc)}
                record["warnings"].append(f"graph persistence (Neo4j) failed: {exc}")
        self.ws.save_batch(record)

    def _export_sphere(self, record: dict[str, Any]) -> None:
        """Queue the standalone 3D sphere (KernelPCA + Plotly) on its own thread.

        The batch thread returns at once, so the next upload never waits on a plot; when several
        batches finish quickly only the latest export runs. ``/api/sphere`` builds the page on demand.
        """
        if not self.config.sphere_export:
            return
        with self._sphere_lock:
            self._sphere_generation += 1
            generation = self._sphere_generation
            if self._sphere_pool is None:
                self._sphere_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltir-sphere")
        self._sphere_pool.submit(self._write_sphere, record, generation)

    def _write_sphere(self, record: dict[str, Any], generation: int) -> None:
        """Export for a READY batch unless a newer export (or a reset) superseded it; failure is a warning."""
        if generation != self._sphere_generation:
            return
        try:
            from ltir.sphere import export_sphere

            record["sphere"] = str(export_sphere(self))
        except Exception as exc:
            record["warnings"].append(f"sphere export failed: {exc}")
        with self._lock:  # reset holds it: the record is saved before a reset or not at all
            if self.ws.load_batch(record["batch_id"]) is not None:
                self.ws.save_batch(record)
            elif "sphere" in record:  # the workspace was reset during the export: drop the stale page
                Path(record["sphere"]).unlink(missing_ok=True)

    def rebuild_graph(self) -> dict[str, Any]:
        with self._lock:
            self.ws.check_versions()
            ontology = self.ontology()
            snapshot = build_snapshot(self.ws, ontology, self.config)
            self.ws.save_graph(snapshot)
            self._refresh_caches(snapshot, ontology)
            return snapshot

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Remove a dataset: its batches, patterns, vectors, memberships and artefacts (docs/06 §6.8).

        Anchors that keep a member, or that are still RELATED_TO another anchor, survive (their
        centroids are not un-averaged); the others go. Journal rows are renumbered, the snapshot is
        rebuilt and the Neo4j mirror synced. Exception-safe through a state + journal checkpoint.
        ``dataset_id`` may also be the batch id of a batch that failed before its dataset id was known.
        """
        with self._lock:
            batches = [b for b in self.ws.list_batches() if dataset_id in (b.get("dataset_id"), b["batch_id"])]
            if not batches:
                raise PipelineError("unknown_dataset", dataset_id)
            if any(b["status"] not in TERMINAL for b in batches):
                raise PipelineError("busy", f"dataset {dataset_id} has a batch in progress")
            self._sphere_generation += 1  # a queued export would draw the old workspace
            ontology = self.ontology()
            records, vectors = self.ws.patterns(), self.ws.vectors()
            kept = [r for r in records if r["dataset_id"] != dataset_id]
            row_of = {r["row_id"]: i for i, r in enumerate(kept)}  # journal order, renumbered
            acts = [{**a, "row_id": row_of[a["row_id"]]} for a in self.ws.activations() if a["row_id"] in row_of]
            cp = self.ws.checkpoint(journal=True)
            try:
                dropped = ontology.forget(acts, len(kept))
                self.ws.rewrite_journal([{**r, "row_id": row_of[r["row_id"]]} for r in kept], vectors[[r["row_id"] for r in kept]], acts)
                ontology.save()
                self.ws.remove_dataset(dataset_id, batches)
                snapshot = build_snapshot(self.ws, ontology, self.config)
                self.ws.save_graph(snapshot)
            except Exception:
                self.ws.rollback(cp)
                self.ws.discard_checkpoint(cp)
                raise
            self.ws.discard_checkpoint(cp)
            self._refresh_caches(snapshot, ontology)
            return {
                "dataset_id": dataset_id,
                "batches": [b["batch_id"] for b in batches],
                "patterns_removed": len(records) - len(kept),
                "anchors_removed": dropped,
                "neo4j": self.sync_neo4j(snapshot),
            }

    def reset(self) -> dict[str, Any]:
        """Delete the workspace; with ``NEO4J_ENABLED`` the mirror is cleared too (failure = warning)."""
        with self._lock:
            self._sphere_generation += 1  # drop a queued export of the old workspace
            self.ws.reset()
            self._graph = None
            self._frame = None
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

    def _recover_interrupted(self) -> None:
        """Roll back and fail every unfinished batch: under the writer lock no other writer is alive."""
        for record in self.ws.list_batches():
            if record["status"] in TERMINAL:
                continue
            cp = record.pop("checkpoint", None)
            if cp and Path(cp["dir"]).exists():
                self.ws.rollback(cp)
                self.ws.discard_checkpoint(cp)
            record["status"] = "FAILED"
            record["error"] = {"code": "interrupted", "message": "processing was interrupted; state rolled back"}
            self.ws.save_batch(record)
