"""End-to-end dataset lifecycle (SDD 14).

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
from collections import Counter
from collections.abc import Iterator
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
from ltir.store import Workspace, atomic_write_json, utc_now

if TYPE_CHECKING:
    from ltir.llm import LLMClient

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
    """Committed vectors of the single insight frame: pattern id -> unit vector, attractor id -> centroid."""

    patterns: dict[str, np.ndarray]
    attractors: dict[int, np.ndarray]

    @classmethod
    def load(cls, ws: Workspace, ontology: LatentOntology) -> LatentFrame:
        vectors = ws.vectors()
        st = ontology.store
        return cls(
            patterns={r["id"]: vectors[r["row_id"]] for r in ws.patterns()} if len(vectors) else {},
            attractors={int(c): st.embeddings[i].copy() for i, c in enumerate(st.concept_ids)},
        )


def _pid_alive(pid: int | None) -> bool:
    """True when another live process owns a batch (portable; never signals the process)."""
    if not pid or pid <= 0 or pid == os.getpid():
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
    """Batch-record metrics (SDD 14 §Metrics)."""
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
        llm: LLMClient | None = None,
        recover: bool = True,
    ) -> None:
        """``recover=False`` for read-only callers (CLI status/query): they must never roll
        back a batch another process is committing."""
        self.config = config
        self.ws = Workspace(config)
        self.encoder = InsightEncoder(embedder or make_text_embedder(config), config)
        self._llm = llm
        self._lock = threading.RLock()
        self._graph: DualGraph | None = None
        self._frame: LatentFrame | None = None
        if recover:
            self._recover_interrupted()

    @property
    def llm(self) -> LLMClient:
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
                    self.ws.append(patterns, enc["vector"], update.activations, {k: enc[k] for k in BLOCKS}, batch_id)
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
            return self.encoder.encode(canon), self.encoder.spec
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
        """Standalone 3D sphere (lac prosphera projector); failure never invalidates the batch."""
        if not self.config.sphere_export:
            return
        try:
            from ltir.sphere import export_sphere

            record["sphere"] = str(export_sphere(self))
        except Exception as exc:
            record["warnings"].append(f"sphere export failed: {exc}")
        self.ws.save_batch(record)

    def rebuild_graph(self) -> dict[str, Any]:
        with self._lock:
            self.ws.check_versions()
            ontology = self.ontology()
            snapshot = build_snapshot(self.ws, ontology, self.config)
            self.ws.save_graph(snapshot)
            self._refresh_caches(snapshot, ontology)
            return snapshot

    def reset(self) -> None:
        with self._lock:
            self.ws.reset()
            self._graph = None
            self._frame = None

    def _recover_interrupted(self) -> None:
        """A batch left mid-flight by a *dead* process is rolled back and marked FAILED.

        Batches owned by a live process (e.g. the web server while the CLI starts) are
        left alone.
        """
        for record in self.ws.list_batches():
            if record["status"] in TERMINAL or _pid_alive(record.get("owner_pid")):
                continue
            cp = record.pop("checkpoint", None)
            if cp and Path(cp["dir"]).exists():
                self.ws.rollback(cp)
                self.ws.discard_checkpoint(cp)
            record["status"] = "FAILED"
            record["error"] = {"code": "interrupted", "message": "processing was interrupted; state rolled back"}
            self.ws.save_batch(record)
