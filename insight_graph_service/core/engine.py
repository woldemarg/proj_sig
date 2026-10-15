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
import hashlib
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
from insight_contracts.text import RELIABILITY
from insight_graph_service.core.batch import admit_insights, batch_metrics, pattern_records
from insight_graph_service.core.settings import Settings
from insight_graph_service.core.snapshot import build_snapshot
from insight_graph_service.core.workspace import TERMINAL, RepresentationMismatch, Workspace, acquire_writer_lock, atomic_write_json, utc_now
from subgroup_miner.discovery import build_insights, covers_of, run_discovery
from subgroup_miner.ingestion import IngestionError, auto_levels, load_dataset, parse_indicators
from subgroup_miner.selection import select_insights
from subgroup_miner.spatial import GeoDataset, SpatialError, cell_boundary, compactness, derive_spatial, parse_geo_levels, sac_summary

log = logging.getLogger(__name__)

BLOCKS = ("scope", "target", "phenomenon")  # tripartite parts persisted beside the composite vector


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


SOURCE_MAX_CHARS = 120


def resolve_dataset_metadata(source: str | None, reliability: str | None, indicators: str | None = None) -> dict[str, Any]:
    """The given dataset metadata, normalised: ``source`` on one line (whitespace collapsed), ``reliability`` an
    upper-case Admiralty code A-F, ``indicators`` the parsed registry (``parse_indicators``); an empty value is None
    (undeclared). A value not given (None) is left out."""
    options: dict[str, Any] = {}
    if source is not None:
        options["source"] = " ".join(source.split()) or None
        if len(options["source"] or "") > SOURCE_MAX_CHARS:
            raise PipelineError("invalid_options", f"source longer than {SOURCE_MAX_CHARS} characters")
    if reliability is not None:
        options["reliability"] = reliability.strip().upper() or None
        if options["reliability"] not in (None, *RELIABILITY):
            raise PipelineError("invalid_options", f"reliability '{reliability}' is not one of {', '.join(RELIABILITY)}")
    if indicators is not None:
        try:
            options["indicators"] = parse_indicators(indicators) or None
        except IngestionError as exc:
            raise PipelineError(exc.code, str(exc)) from exc
    return options


def resolve_levels(geo: str | None, table: bool, region_levels: tuple[int, ...], default_resolution: int) -> list[tuple[str | None, str]]:
    """The levels one upload makes, as (``geo`` option of the level's dataset, level): each listed resolution, finest
    first, then ``table`` (no geo) when asked. A coarser resolution must be one of ``GEO_REGION_LEVELS``."""
    if not geo:
        if table:
            raise PipelineError("invalid_options", "table=true needs the geo option: a plain upload is the table level already")
        return [(None, "table")]
    try:
        levels = parse_geo_levels(geo, default_resolution)
    except SpatialError as exc:
        raise PipelineError("invalid_options", str(exc)) from exc
    if stray := [r for _, r in levels[1:] if r not in region_levels]:
        raise PipelineError("invalid_options", f"coarser geo levels {stray} are not GEO_REGION_LEVELS {list(region_levels)}")
    return [(f"{loc}@{r}", str(r)) for loc, r in levels] + ([(None, "table")] if table else [])


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
        self._open()

    def committed(self) -> CommittedState:
        """The committed read state. One object per commit, so a reader never mixes two commits; readers never
        touch the journal or a lock, so they neither race nor wait for a batch."""
        return self._state

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
        return self._committed_state(graph, self._load_frame(self.ws.patterns(), self.ws.vectors(), self.ontology()))

    @staticmethod
    def _empty_state() -> CommittedState:
        return CommittedState(DualGraph.empty(), LatentFrame({}, {}, {}))

    def _committed_state(self, graph: DualGraph, frame: LatentFrame) -> CommittedState:
        """The read state of a commit, complete before readers see it: every pattern's canonical-document vector (stored
        at ingest; one missing, from a batch written before documents were embedded, is embedded here) and the literal
        catalog (docs/07_question_answering.md §7.1.1; known literals from the workspace's literal cache). A writer
        saves both, so no later commit or restart pays for them again. Neither is built for a workspace of another
        representation (its questions are refused) nor when the model fails: then ``search`` builds the catalog on the
        first question, and a pattern without a document vector stays out of the naive baseline."""
        if not graph.insights:
            return CommittedState(graph, frame)
        try:
            self.ws.check_representation(self.encoder.spec)  # never embed or save with a model the workspace does not use
            missing = [pid for pid in graph.insights if pid not in frame.documents]
            if missing:
                filled = dict(zip(missing, self.encoder.embedder.embed([graph.canonical_document(i) for i in missing])))
                frame.documents.update(filled)
                if self._writer:
                    self.ws.save_document_vectors(frame.documents, set(filled))
            fingerprint = self.encoder.spec.fingerprint
            known = self.ws.load_literals(fingerprint)
            catalog = build_catalog(graph, self.encoder.embedder, known)
            if self._writer and known.keys() != set(catalog.texts):  # new literals, or a deleted dataset's gone
                self.ws.save_literals(fingerprint, catalog.texts, catalog.vectors)
        except Exception as exc:  # the graph is committed; a missing cache costs the first question, never the workspace
            log.warning("committed state without its literal catalog: %s: %s", type(exc).__name__, exc)
            return CommittedState(graph, frame)
        return CommittedState(graph, frame, catalog)

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
        found = search(question, state, self.encoder, self.settings.query)
        found.seconds = time.perf_counter() - start
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
        geo = []  # every READY geo dataset's stored cells and extents: the spatial plane is derived, never journaled
        for b in batches.values():
            stored = self.ws.load_geo(b["dataset_id"]) if b.get("status") == "READY" and (b.get("profile") or {}).get("geo") else None
            if stored is not None:
                geo.append(GeoDataset(b["dataset_id"], stored["cells"], stored["nbr_ptr"], stored["nbr_idx"], self.ws.load_covers(b["dataset_id"])))
        snapshot = build_snapshot(
            records=records,
            activations=self.ws.activations(),
            vectors=vectors,
            batches=batches,
            representation=self.ws.representation() or {},
            ontology=ontology,
            settings=self.settings,
            spatial=derive_spatial(geo, self.settings.region_levels),
        )
        if self._writer:  # a read-only engine rebuilds an outdated snapshot in memory only
            self.ws.save_graph(snapshot)
        return snapshot, self._committed_state(DualGraph(snapshot), self._load_frame(records, vectors, ontology))

    @_healthy
    def upload(
        self,
        name: str,
        stream: BinaryIO,
        *,
        bins: str | None = None,
        categories: str | None = None,
        geo: str | None = None,
        source: str | None = None,
        reliability: str | None = None,
        indicators: str | None = None,
        table: bool = False,
    ) -> list[dict[str, Any]]:
        """Store an uploaded file (at most ``MAX_UPLOAD_MB``) and register one batch per level (UPLOADED), each an ordinary
        dataset with its own id, all with the file's ``upload_id`` (``resolve_levels``); ``<locator>@auto`` reads the
        levels from the points (``select_levels``). Processing is a separate call."""
        resolve_dataset_metadata(source, reliability, indicators)  # a bad value is refused before the file is stored
        locator, _, res = (geo or "").partition("@")
        auto = res.strip().lower() == "auto"
        levels = [] if auto else resolve_levels(geo, table, self.settings.region_levels, self.settings.miner.geo_resolution)
        path = self.ws.save_upload(name, stream, self.settings.max_upload_mb * 1024 * 1024)  # UploadTooLarge past the cap
        selection, refused = None, None
        if auto:
            try:
                selection = auto_levels(path, locator.strip(), self.settings.miner, self.settings.region_levels)
            except IngestionError as exc:
                refused = PipelineError(exc.code, str(exc))
        if not path.exists():  # the workspace was reset while the file streamed in or was read
            raise PipelineError("busy", "the workspace was reset during the upload; upload again")
        if refused:
            path.unlink(missing_ok=True)  # refused: the file is not kept
            raise refused
        if selection:
            chosen = ",".join(map(str, selection["chosen"]))
            levels = resolve_levels(f"{locator}@{chosen}", table, self.settings.region_levels, self.settings.miner.geo_resolution)
        with path.open("rb") as stored:
            upload_id = "up-" + hashlib.file_digest(stored, "sha256").hexdigest()[:12]
        options = {"filename": name, "bins": bins, "categories": categories, "source": source, "reliability": reliability, "indicators": indicators}
        if table:  # the geo and the table levels share bins and categories: each applies those it has
            options["strict"] = False
        located = {"locator": locator.strip()} if geo else {}  # the table level leaves the point columns out
        return [
            self.submit(path, geo=g, upload=(upload_id, level), **({"geo_levels": selection} if g else located), **options) for g, level in levels
        ]

    @_healthy
    def submit(
        self,
        path: Path,
        *,
        filename: str | None = None,
        bins: str | None = None,
        categories: str | None = None,
        geo: str | None = None,
        source: str | None = None,
        reliability: str | None = None,
        indicators: str | None = None,
        upload: tuple[str, str] | None = None,
        strict: bool = True,
        geo_levels: dict[str, Any] | None = None,
        locator: str | None = None,
    ) -> dict[str, Any]:
        """Register an upload (status UPLOADED). Processing is a separate call.

        ``bins`` / ``categories`` = None means "use BIN_COLUMNS / CATEGORICAL_COLUMNS"
        (applied leniently); an explicit string (even "") overrides them strictly.
        ``geo`` (``auto``, a point column or ``lat,lon``, optionally ``@<res>``) analyses H3 cells (docs/02_discovery.md §2.9).
        ``source`` / ``reliability`` (A-F) describe where the data comes from, ``indicators`` what its metrics measure
        (``metric:indicator[:unit],...``); none enters the dataset id. ``upload`` is (``upload_id``, level) for a batch
        of ``upload``; ``strict`` = False skips the ``bins`` / ``categories`` columns the dataset lacks (with a warning);
        ``geo_levels`` is the ``@auto`` selection, moved into ``profile.geo.levels`` by ``process``; ``locator`` the point
        columns a table level leaves out.
        """
        trust = {k: v for k, v in resolve_dataset_metadata(source, reliability, indicators).items() if v}
        path = Path(path)
        created = utc_now()
        record = {
            "batch_id": f"B{created[:19].replace('-', '').replace(':', '')}-{uuid.uuid4().hex[:6]}",
            "dataset_id": None,
            "filename": filename or path.name,
            "source_path": str(path),
            "bins": bins,
            "categories": categories,
            **({"geo": geo} if geo else {}),
            **({"upload_id": upload[0], "level": upload[1]} if upload else {}),
            **({} if strict else {"strict": False}),
            **({"geo_levels": geo_levels} if geo_levels else {}),
            **({"locator": locator} if locator else {}),
            **trust,
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
                        geo=record.get("geo"),
                        strict=record.get("strict", True),
                        locator=record.get("locator"),
                    )
                    ds = record["dataset_id"] = loaded.dataset_id
                    record["warnings"] += loaded.warnings
                    duplicate = self.ws.ready_batch_for(ds)
                    if duplicate:
                        return self._skip(record, duplicate)

                with self._stage(record, "PROFILING", timings, "discover_s"):
                    result = run_discovery(loaded.frame, self.settings.miner, on_stage=lambda s: self._enter(record, s), spatial=loaded.spatial)
                    record["profile"] = {
                        **result.profile.to_dict(),
                        "derived_columns": loaded.derived_columns,
                        "bins": loaded.bins,
                        "categorical_overrides": loaded.categories,
                    }
                    if loaded.spatial:
                        record["profile"]["geo"] = self._geo_profile(loaded) | (
                            {"levels": record.pop("geo_levels")} if "geo_levels" in record else {}
                        )
                    if unknown := sorted(set(record.get("indicators", {})) - set(result.profile.numerics)):
                        record["warnings"].append(
                            f"indicators: no metric named {unknown} in the dataset (its metrics: {result.profile.numerics}); dropped"
                        )
                        kept = {k: v for k, v in record.pop("indicators").items() if k not in unknown}
                        record.update({"indicators": kept} if kept else {})
                    self.ws.save_profile(ds, record["profile"])

                with self._stage(record, "VALIDATING_INSIGHTS", timings, "select_s"):
                    covers = covers_of(result)
                    insights = build_insights(result, self.settings.miner, dataset_id=ds, batch_id=batch_id, filename=record["filename"])
                    selection = select_insights(insights, self.settings.miner)
                    kept, refused = admit_insights(selection.kept, self.settings.min_insight_weight, self.settings.max_insights_per_batch)
                    if loaded.spatial:
                        record["profile"]["geo"]["sac"] = {**sac_summary(insights, self.settings.miner.max_p_adjusted), "admitted": len(kept)}
                        self.ws.save_profile(ds, record["profile"])
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
                        ontology.recount_mass(self.ws.activations())  # the store's inertia is the journal's summed member weights
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
                        if loaded.spatial:  # the map's file, from the journaled patterns
                            ctx = loaded.spatial
                            self.ws.save_geo(
                                ds,
                                {
                                    "cells": ctx.cells.astype(str),
                                    "point_cell": ctx.point_cell,
                                    "lonlat": ctx.lonlat,
                                    "nbr_ptr": ctx.nbr_ptr,
                                    "nbr_idx": ctx.nbr_idx,
                                },
                            )
                            atomic_write_json(self.ws.geo_cells_path(ds), self.geo_cells(ds, loaded))
                        ontology.save()
                        self.ws.record_representation(spec)
                        self.ws.commit_batch_seq(seq)
                        snapshot, state = self._derive(ontology, pending=record)  # loaded before the commit point
                        if state.catalog is None:
                            record["warnings"].append("literal catalog not built at the commit (see the service log); the first question builds it")

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

    @staticmethod
    def _geo_profile(loaded: Any) -> dict[str, Any]:
        ctx = loaded.spatial
        return {
            "option": loaded.geo,
            "resolution": ctx.resolution,
            "location": ctx.location,
            "cells": len(ctx.cells),
            "points": int((ctx.point_cell >= 0).sum()),
            "sparse_cells": ctx.sparse_cells,
            "min_cell_points": ctx.min_cell_points,
            "base": "H3 cells holding at least one point",
            "moran_i": {k: round(v, 4) for k, v in ctx.moran.items()},
            "glossary": ctx.glossary,
        }

    def geo_datasets(self) -> list[dict[str, Any]]:
        """READY datasets analysed as H3 cells, with what the map needs to offer them."""
        out = []
        for b in self.ws.list_batches():
            geo = (b.get("profile") or {}).get("geo")
            if b.get("status") == "READY" and geo:
                out.append({"dataset_id": b["dataset_id"], "filename": b["filename"], **geo, "metrics": b["profile"].get("numerics", [])})
        return out

    def _geo_ready(self, dataset_id: str) -> bool:
        """Only a dataset with a READY geo batch is served: its files survive a failed batch, its listing does not."""
        return any(d["dataset_id"] == dataset_id for d in self.geo_datasets())

    def geo_cells_file(self, dataset_id: str) -> Path:
        """``geo_cells`` as written with the dataset's batch (the cells, their rings and the patterns' extents never
        change afterwards): the map reads the file on every open."""
        if not (self._geo_ready(dataset_id) and self.ws.geo_cells_path(dataset_id).is_file()):
            raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
        return self.ws.geo_cells_path(dataset_id)

    def geo_cells(self, dataset_id: str, loaded: Any) -> dict[str, Any]:
        """The dataset's cells as a GeoJSON FeatureCollection with the journaled patterns' extents; properties are the
        columns of the analysed cell table (``loaded.frame``), read once, when the batch writes the file."""
        ctx, frame = loaded.spatial, loaded.frame
        cols = {c: frame[c].to_numpy(dtype=float if frame[c].dtype.kind in "fiu" else str) for c in frame.columns}
        features = []
        for i, cell in enumerate(ctx.cells):
            props: dict[str, Any] = {"h3": str(cell), "row": i}
            for name, values in cols.items():
                v = values[i]
                props[name] = (None if not np.isfinite(v) else round(float(v), 4)) if values.dtype.kind == "f" else str(v)
            features.append(
                {"type": "Feature", "id": i, "geometry": {"type": "Polygon", "coordinates": [cell_boundary(str(cell))]}, "properties": props}
            )
        lonlat = ctx.lonlat[ctx.point_cell >= 0]
        bbox = [float(lonlat[:, 0].min()), float(lonlat[:, 1].min()), float(lonlat[:, 0].max()), float(lonlat[:, 1].max())]
        covers = self.ws.load_covers(dataset_id)
        patterns = [
            {
                "id": r["id"],
                "rows": covers[r["id"]].tolist(),
                "scope": [f"{c['attribute']} = {c['value']}" for c in r["conditions"]],
                "target": r["target"],
                "effect_size": r["effect_size"],
                "phenomenon_type": r["phenomenon_type"],
                "support": r["support"],
                "weight": r["weight"],
                "spatial": {
                    **r["provenance"].get("spatial", {}),
                    "compactness": round(compactness(covers[r["id"]], ctx.nbr_ptr, ctx.nbr_idx), 4),
                },
            }
            for r in self.ws.patterns()
            if r["dataset_id"] == dataset_id and r["id"] in covers
        ]
        return {
            "type": "FeatureCollection",
            "features": features,
            "bbox": bbox,
            "meta": {"resolution": ctx.resolution, "location": ctx.location, "columns": list(frame.columns)},
            "patterns": patterns,
        }

    def geo_pattern(self, dataset_id: str, pattern_id: str, max_points: int = 5000) -> dict[str, Any]:
        """A pattern's cells (rows of the cell table) and a deterministic sample of the points inside them."""
        geo = self._geo(dataset_id)
        rows = self.ws.load_covers(dataset_id).get(pattern_id)
        if rows is None:
            raise PipelineError("unknown_dataset", f"no pattern {pattern_id} in {dataset_id}")
        inside = np.flatnonzero(np.isin(geo["point_cell"], rows))
        step = max(1, int(np.ceil(len(inside) / max_points)))
        points = geo["lonlat"][inside[::step]]
        return {
            "pattern_id": pattern_id,
            "cells": [str(c) for c in geo["cells"][rows]],
            "rows": rows.tolist(),
            "n_points": int(len(inside)),
            "points": np.round(points.astype(float), 5).tolist(),
        }

    def _geo(self, dataset_id: str) -> dict[str, Any]:
        geo = self.ws.load_geo(dataset_id) if self._geo_ready(dataset_id) else None
        if geo is None:
            raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
        return geo

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
        if record.get("source") or record.get("reliability") or record.get("indicators"):
            record["warnings"].append(
                f"a re-upload does not change the source, reliability or indicators: PATCH /api/datasets/{record['dataset_id']}"
            )
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
        """After a commit (batch, deletion or metadata change): readers switch to the new state in one assignment and
        the optional Neo4j mirror is synced; returns the mirror status."""
        self._state = state
        return self.sync_neo4j(snapshot)

    @_healthy
    def update_dataset_metadata(
        self, dataset_id: str, *, source: str | None = None, reliability: str | None = None, indicators: str | None = None
    ) -> dict[str, Any]:
        """Change a READY dataset's ``source`` / ``reliability`` / ``indicators`` (None: unchanged, "": undeclared) on its
        batch record and recompile the snapshot: no new batch, no mining, the same pattern ids
        (docs/06_graph_and_storage.md §6.2). An indicator for a metric the dataset does not have is refused."""
        changes = resolve_dataset_metadata(source, reliability, indicators)
        with self._lock:
            record = self.ws.ready_batch_for(dataset_id)
            if record is None:
                raise PipelineError("unknown_dataset", dataset_id)
            unknown = sorted(set(changes.get("indicators") or {}) - set(record["profile"].get("numerics", [])))
            if unknown:
                raise PipelineError("invalid_options", f"indicators: no metric named {unknown} in {dataset_id}")
            record.update(changes)
            for cleared in [k for k, v in changes.items() if v is None]:
                del record[cleared]
            # ponytail: a crash after save_batch and before the marker goes rolls the snapshot back under the new record until the next commit
            with self.ws.transaction({"dataset_id": dataset_id}):
                snapshot, state = self._derive(self.ontology(), pending=record)
                self.ws.save_batch(record)
            return {
                "dataset_id": dataset_id,
                "source": record.get("source"),
                "reliability": record.get("reliability", "F"),
                "indicators": record.get("indicators", {}),
                "neo4j": self._publish(state, snapshot),
            }

    @_healthy
    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Remove a dataset: its batches, patterns, vectors, memberships and artefacts (docs/06_graph_and_storage.md §6.8).

        Anchors that keep a member, or that are still RELATED_TO another anchor, survive (their mass recounted;
        those that lost a member moved to their surviving members' direction); the others go. Journal rows are renumbered, the snapshot is
        rebuilt and the Neo4j mirror synced. All or nothing (``Workspace.transaction``): the checkpoint holds the
        state, the journal, the snapshot and the moved-aside files; an error rolls back at once, and a crash
        leaves the ``pending.json`` marker, so the next writer start rolls back (``Workspace.recover``).
        ``dataset_id`` may also be the batch id of a batch that failed before its dataset id was known.
        """
        return {
            "dataset_id": dataset_id,
            **self._delete(lambda b: dataset_id in (b.get("dataset_id"), b["batch_id"]), {"dataset_id": dataset_id}, "unknown_dataset"),
        }

    @_healthy
    def delete_upload(self, upload_id: str) -> dict[str, Any]:
        """Remove every level of one upload (the datasets its batches made) in one transaction, as ``delete_dataset``."""
        return {"upload_id": upload_id, **self._delete(lambda b: b.get("upload_id") == upload_id, {"upload_id": upload_id}, "unknown_upload")}

    def _delete(self, chosen: Callable[[dict[str, Any]], bool], intent: dict[str, str], unknown: str) -> dict[str, Any]:
        """Delete the datasets of the ``chosen`` batches with all their batches (``delete_dataset``); ``intent`` names
        what is deleted (``dataset_id`` or ``upload_id``) for the transaction marker, ``unknown`` the refusal's code."""
        key = next(iter(intent.values()))
        with self._lock:
            every = self.ws.list_batches()
            picked = [b for b in every if chosen(b)]
            if not picked:
                raise PipelineError(unknown, key)
            ids = {b["dataset_id"] for b in picked if b.get("dataset_id")}
            batches = [b for b in every if chosen(b) or b.get("dataset_id") in ids]
            if any(b["status"] not in TERMINAL for b in batches):
                raise PipelineError("busy", f"{key} has a batch in progress")
            ontology = self.ontology()
            records, vectors = self.ws.patterns(), self.ws.vectors()
            kept = [r for r in records if r["dataset_id"] not in ids]
            row_of = {r["row_id"]: i for i, r in enumerate(kept)}  # journal order, renumbered
            all_acts = self.ws.activations()
            acts = [{**a, "row_id": row_of[a["row_id"]]} for a in all_acts if a["row_id"] in row_of]
            lost = {int(a["attractor_id"]) for a in all_acts if a["row_id"] not in row_of}  # the anchors the deletion touches
            with self.ws.transaction(intent, journal=True) as cp:  # commit point: the marker goes
                self.ws.remove_datasets(ids, batches, Path(cp["dir"]) / "removed")  # first: the snapshot must not see them
                kept_vectors = vectors[[r["row_id"] for r in kept]]
                dropped = ontology.forget(acts, kept_vectors, lost)
                self.ws.rewrite_journal([{**r, "row_id": row_of[r["row_id"]]} for r in kept], kept_vectors, acts)
                ontology.save()
                snapshot, state = self._derive(ontology)
            return {
                "datasets": sorted(ids),
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
