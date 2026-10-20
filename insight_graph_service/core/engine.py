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
import math
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
from insight_contracts import Condition, EvidencePayload, Insight, pattern_id
from insight_contracts.text import RELIABILITY
from insight_graph_service.core import geo_views
from insight_graph_service.core.batch import admit_insights, batch_metrics, pattern_records
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.settings import Settings
from insight_graph_service.core.snapshot import build_snapshot, compute_metadata_integrity, metadata_digest
from insight_graph_service.core.workspace import TERMINAL, RepresentationMismatch, Workspace, acquire_writer_lock, atomic_write_json, utc_now
from subgroup_miner.discovery import NOT_RECOMPUTED, build_insights, covers_of, recompute_statistics, run_discovery
from subgroup_miner.ingestion import IngestionError, LoadedDataset, auto_levels, load_dataset, parse_indicators
from subgroup_miner.selection import select_insights
from subgroup_miner.spatial import (
    GeoDataset,
    SpatialError,
    derive_spatial,
    parse_geo_levels,
    sac_summary,
)

log = logging.getLogger(__name__)

BLOCKS = ("scope", "target", "phenomenon")  # tripartite parts persisted beside the composite vector


SOURCE_MAX_CHARS = 120
EMPTY_LEVEL = frozenset({"invalid_schema", "no_numeric_targets", "no_candidates", "no_viable_insights"})  # a level with nothing to mine


def verify_diff(record: dict[str, Any], fresh: dict[str, Any], covers: np.ndarray | None, family: int) -> list[dict[str, Any]]:
    """What of the journal record the recomputation (``recompute_statistics``) does not reproduce, each as ``{field,
    journal, recomputed}``: its id and selector against its conditions, the rows against ``covers.npz`` and the record's
    row hash, its target as its first shift (on a covariance insight, the stronger of its pair), a missing p only on a
    covariance insight, its test family against its dataset's, then every number; floats agree within a relative 1e-9 (the same code over the same bytes)."""
    rows = fresh["rows"]
    out = []

    def differs(field: str, journal: Any, recomputed: Any) -> None:
        out.append({"field": field, "journal": journal, "recomputed": recomputed})

    if record["id"] != pattern_id(record["dataset_id"], tuple(Condition(c["attribute"], c["value"]) for c in record["conditions"])):
        differs("id", record["id"], "another id for these conditions")
    if fresh["expression_rows"] is None or not np.array_equal(fresh["expression_rows"], rows):
        differs("expression", record["expression"], "selects other rows than the conditions")
    if covers is not None and not np.array_equal(np.sort(covers), rows):
        differs("rows", f"{len(covers)} rows in covers.npz", f"{len(rows)} rows by the conditions")
    if record.get("row_hash") != hashlib.sha1(rows.astype(np.int64).tobytes()).hexdigest()[:16]:
        differs("row_hash", record.get("row_hash"), hashlib.sha1(rows.astype(np.int64).tobytes()).hexdigest()[:16])
    if record["provenance"].get("multiple_testing_family") != family:
        differs("multiple_testing_family", record["provenance"].get("multiple_testing_family"), f"{family} for its dataset")
    covariance = record.get("phenomenon_type") == "covariance" and bool((record.get("covariance") or {}).get("pair"))
    if record["p_value"] is None and not covariance:
        differs("p_value", None, fresh.get("p_value"))
    if "shifts" not in fresh:
        return out + [{"field": "support", "journal": record["support"], "recomputed": 0}]
    primary = next((s for s in fresh["shifts"] if s["metric"] == record["target"]), None)
    if primary is None:
        differs("target", record["target"], "not among its shifts")
        primary = fresh["shifts"][0]
    elif not covariance and record["shifts"][0]["metric"] != record["target"]:  # its p and spatial test are of its first shift
        differs("target", record["target"], f"not its first shift ({record['shifts'][0]['metric']}, whose test its p is)")
    pair = [s for s in fresh["shifts"] if s["metric"] in (record.get("covariance") or {}).get("pair", [])]
    if covariance and pair and record["target"] != (chosen := min(pair, key=lambda s: (-abs(s["robust_z"]), s["metric"]))["metric"]):
        differs("target", record["target"], f"{chosen} (the stronger metric of its pair)")
    pairs = [
        ("support", record["support"], fresh["support"]),
        ("support_fraction", record["support_fraction"], fresh["support_fraction"]),
        ("baseline", record["baseline"], primary["global_median"]),
        ("local", record["local"], primary["local_median"]),
        ("effect_size", record["effect_size"], primary["robust_z"]),
    ]
    if record["p_value"] is not None:
        pairs += [("p_value", record["p_value"], fresh["p_value"]), ("p_adjusted", record["p_adjusted"], fresh["p_adjusted"])]
    for stored, again in zip(record["shifts"], fresh["shifts"], strict=True):
        pairs += [(f"shifts.{stored['metric']}.{k}", stored[k], again[k]) for k in ("robust_z", "local_median", "global_median", "global_mad")]
    spatial = record["provenance"].get("spatial") or {}
    pairs += [(f"spatial.{k}", spatial[k], v) for k, v in fresh.get("spatial", {}).items() if k in spatial]
    for name, a, b in pairs:
        if not (a == b or (isinstance(a, int | float) and isinstance(b, int | float) and math.isclose(a, b, rel_tol=1e-9))):
            differs(name, a, b)
    return out


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
        self._writer = recover  # holds the writer lock: may save derived files (the literal cache)
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
        ``problem`` (why); every guarded call is refused until ``reset()``. So is a state file that cannot be read (the
        snapshot is rebuilt, whatever its file holds): the service starts, and its reset stays reachable."""
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
        """The read state, its snapshot rebuilt from the journals (derived data: the journals stay the source of truth).
        Integrity and credibility are computed at compile, never read back from the file, so a snapshot edited or left
        stale since the last commit is never served."""
        if not self.ws.patterns():  # nothing journaled, nothing to compile: no snapshot file is written
            return self._committed_state(DualGraph.empty(), self._load_frame([], self.ws.vectors(), self.ontology()))
        return self._derive(self.ontology())[1]

    @staticmethod
    def _empty_state() -> CommittedState:
        return CommittedState(DualGraph.empty(), LatentFrame({}, {}))

    def _committed_state(self, graph: DualGraph, frame: LatentFrame) -> CommittedState:
        """The read state of a commit, complete before readers see it: the literal catalog
        (docs/07_question_answering.md §7.1.1; known literals from the workspace's literal cache, which a writer saves, so
        no later commit or restart pays for them again). It is not built for a workspace of another representation (its
        questions are refused) nor when the model fails: then ``search`` builds it on the first question."""
        if not graph.insights:
            return CommittedState(graph, frame)
        try:
            self.ws.check_representation(self.encoder.spec)  # never embed or save with a model the workspace does not use
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
    def search(self, question: str, **zone: Any) -> SearchResult:
        """Retrieval only: the evidence for ``question`` over the committed graph (no LLM; docs/07_question_answering.md
        §7.1–7.4); ``zone``: the ``cells``, ``geometry`` and ``resolution`` of ``place_scores``."""
        return self._search(question, self.committed(), **zone)

    @_healthy
    def evidence(self, question: str, **zone: Any) -> EvidencePayload:
        """The evidence for ``question`` (and a map zone, as ``search``) as the narrator receives it (``POST /api/search``);
        an empty graph is an ordinary answer, not an error."""
        state = self.committed()
        return self._search(question, state, **zone).payload() if state.graph.insights else empty_payload(question)

    def _search(self, question: str, state: CommittedState, **zone: Any) -> SearchResult:
        start = time.perf_counter()
        self.ws.check_representation(self.encoder.spec)  # query and stored vectors must share one frame
        found = search(question, state, self.encoder, self.settings.query, self.place_scores(state, **zone))
        found.seconds = time.perf_counter() - start
        return found

    def place_scores(
        self, state: CommittedState, cells: list[str] | None = None, geometry: dict[str, Any] | None = None, resolution: int | None = None
    ) -> dict[str, float] | None:
        return geo_views.place_scores(self.ws, state.graph, cells, geometry, resolution)

    def _load_frame(self, records: list[dict[str, Any]], vectors: np.ndarray, ontology: LatentOntology) -> LatentFrame:
        return LatentFrame(
            patterns={r["id"]: vectors[r["row_id"]] for r in records} if len(vectors) else {},
            attractors={a.id: a.centroid for a in ontology.attractors()},
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
            stored = (
                self.ws.load_geo(b["dataset_id"], ("cells", "nbr_ptr", "nbr_idx"))
                if b.get("status") == "READY" and (b.get("profile") or {}).get("geo")
                else None
            )
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
        supersedes: str | None = None,
    ) -> list[dict[str, Any]]:
        """Store an uploaded file (at most ``MAX_UPLOAD_MB``) and register one batch per level (UPLOADED), each an ordinary
        dataset with its own id, all with the file's ``upload_id`` (``resolve_levels``); ``<locator>@auto`` reads the
        levels from the points (``select_levels``); ``supersedes`` names the dataset or upload this one replaces once it is
        READY (``_superseded``). Processing is a separate call."""
        resolve_dataset_metadata(source, reliability, indicators)  # a bad value is refused before the file is stored
        supersedes = self._check_supersedes(supersedes)
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
        if supersedes and supersedes in {upload_id, *(b.get("dataset_id") for b in self.ws.list_batches() if b.get("upload_id") == upload_id)}:
            path.unlink(missing_ok=True)  # its upload, or a level of it: the same bytes
            raise PipelineError("invalid_options", f"supersedes {supersedes}: the file is the version it replaces")
        options = {
            "filename": name,
            "bins": bins,
            "categories": categories,
            "source": source,
            "reliability": reliability,
            "indicators": indicators,
            "supersedes": supersedes,
        }
        if table:  # the geo and the table levels share bins and categories: each applies those it has
            options["strict"] = False
        return [  # the table level leaves the point columns out
            self.submit(
                path,
                geo=g,
                upload={"upload_id": upload_id, "level": level},
                geo_levels=selection if g else None,
                locator=None if g else locator.strip() or None,
                **options,
            )
            for g, level in levels
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
        upload: dict[str, Any] | None = None,
        strict: bool = True,
        geo_levels: dict[str, Any] | None = None,
        locator: str | None = None,
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        """Register an upload (status UPLOADED). Processing is a separate call.

        ``bins`` / ``categories`` = None means "use BIN_COLUMNS / CATEGORICAL_COLUMNS"
        (applied leniently); an explicit string (even "") overrides them strictly.
        ``geo`` (``auto``, a point column or ``lat,lon``, optionally ``@<res>``) analyses H3 cells (docs/02_discovery.md §2.9).
        ``source`` / ``reliability`` (A-F) describe where the data comes from, ``indicators`` what its metrics measure
        (``metric:indicator[:unit],...``); none enters the dataset id. ``upload`` is ``{upload_id, level}`` for
        a batch of ``upload``; ``strict`` = False skips the ``bins`` / ``categories`` columns the dataset lacks (with a warning);
        ``geo_levels`` is the ``@auto`` selection, kept as ``profile.geo.levels`` whatever the batch's end; ``locator`` the point
        columns a table level leaves out; ``supersedes`` a dataset or upload id this one replaces (outside the dataset id too).
        """
        trust = {k: v for k, v in resolve_dataset_metadata(source, reliability, indicators).items() if v}
        supersedes = self._check_supersedes(supersedes)
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
            **(upload or {}),
            **({} if strict else {"strict": False}),
            **({"locator": locator} if locator else {}),
            **({"supersedes": supersedes} if supersedes else {}),
            **trust,
            "status": "UPLOADED",
            "stage_times": {"UPLOADED": created},
            "created_at": created,
            "metrics": {},
            "profile": {"geo": {"levels": geo_levels}} if geo_levels else {},
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
                    loaded = self._load(record)
                    ds = record["dataset_id"] = loaded.dataset_id
                    record["warnings"] += loaded.warnings
                    duplicate = self.ws.ready_batch_for(ds)
                    if duplicate:
                        return self._skip(record, duplicate)

                with self._stage(record, "PROFILING", timings, "discover_s"):
                    result = run_discovery(loaded.frame, self.settings.miner, on_stage=lambda s: self._enter(record, s), spatial=loaded.spatial)
                    selection = record["profile"].get("geo", {}).get("levels")  # @auto's, from the upload
                    record["profile"] = {
                        **result.profile.to_dict(),
                        "derived_columns": loaded.derived_columns,
                        "bins": loaded.bins,
                        "categorical_overrides": loaded.categories,
                    }
                    if loaded.spatial:
                        record["profile"]["geo"] = geo_views.geo_profile(loaded) | ({"levels": selection} if selection else {})
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

                replaced = self._superseded(record, ("READY", None))
                with self.ws.transaction({"batch_id": batch_id}, journal=bool(replaced)) as cp:
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
                        blocks = {**{k: enc[k] for k in BLOCKS}, "pattern_ids": np.array([i.id for i in kept])}
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
                            atomic_write_json(self.ws.geo_cells_path(ds), geo_views.geo_cells(self.ws, ds, loaded))
                        if replaced:  # after the new version joined its anchors: those it shares keep their ids
                            record["superseded"] = self._remove(*replaced, ontology, Path(cp["dir"]) / "removed")
                        ontology.save()
                        self.ws.record_representation(spec)
                        self.ws.commit_batch_seq(seq)
                        record["metadata_digest"] = metadata_digest(record)  # what the graph reads of the dataset, signed
                        snapshot, state = self._derive(ontology, pending=record)  # loaded before the commit point
                        if state.catalog is None:
                            record["warnings"].append("literal catalog not built at the commit (see the service log); the first question builds it")

                    record["metrics"] = batch_metrics(result, kept, rejections, update, spec, snapshot, timings, t0)
                    if replaced:  # the anchors left after the old version went
                        record["metrics"]["attractors_total"] = len(ontology.attractor_ids)
                    record["status"] = "READY"
                    record["stage_times"]["READY"] = utc_now()
                    self.ws.save_batch(record)  # the commit point: Workspace.recover keeps the writes of a READY batch
            except Exception as exc:  # failures before READY were rolled back by the transaction
                return self._fail(record, exc, t0)
            start = time.perf_counter()
            record["neo4j"] = self._publish(state, snapshot)  # committed: nothing below may turn READY into FAILED
            record["metrics"]["timings"]["neo4j_s"] = time.perf_counter() - start
            if record["neo4j"]["status"] == "failed":
                record["warnings"].append(f"graph persistence (Neo4j) failed: {record['neo4j']['error']}")
            self.ws.save_batch(record)
            return record

    # the map's reads (core/geo_views.py): refused on a degraded workspace, each over the graph of one commit

    @_healthy
    def geo_datasets(self) -> list[dict[str, Any]]:
        return geo_views.geo_datasets(self.ws)

    @_healthy
    def geo_cells_file(self, dataset_id: str) -> Path:
        return geo_views.geo_cells_file(self.ws, dataset_id)

    def _load(self, record: dict[str, Any]) -> LoadedDataset:
        """The batch's stored file read with its own options, as its processing read it."""
        return load_dataset(
            Path(record["source_path"]),
            self.settings.miner,
            filename=record["filename"],
            bins=record.get("bins"),
            categories=record.get("categories"),
            geo=record.get("geo"),
            strict=record.get("strict", True),
            locator=record.get("locator"),
        )

    @_healthy
    def verify_pattern(self, pattern_id: str) -> dict[str, Any]:
        """``pattern_id`` recomputed from its stored upload (docs/06_graph_and_storage.md §6.7): ``reproduced`` when its rows
        and every recomputed number match the journal, ``differs`` with what does not, ``source_changed`` when the file
        read with its options (under the current miner settings) no longer gives the dataset id, ``source_missing`` when
        it is gone. Reads only, without the writer lock: a file removed meanwhile is ``source_missing``."""
        records = self.ws.patterns()
        record = next((r for r in records if r["id"] == pattern_id), None)
        if record is None:
            raise PipelineError("unknown_pattern", pattern_id)
        ds = record["dataset_id"]
        batch = self.ws.ready_batch_for(ds)
        out = {"pattern_id": pattern_id, "dataset_id": ds, "diff": [], "not_checked": list(NOT_RECOMPUTED)}
        try:
            loaded = self._load(batch) if batch else None
        except IngestionError as exc:  # unreadable now, or refused by the current settings
            return out | {"status": "source_missing" if exc.code == "unsupported_file" else "source_changed", "detail": str(exc)}
        if loaded is None:
            return out | {"status": "source_missing"}
        if loaded.dataset_id != ds:
            return out | {"status": "source_changed", "detail": f"the file read with its options now gives {loaded.dataset_id}"}
        family = Counter(r["provenance"].get("multiple_testing_family") for r in records if r["dataset_id"] == ds).most_common(1)[0][0]
        fresh = recompute_statistics(
            loaded.frame,
            [(c["attribute"], c["value"]) for c in record["conditions"]],
            record["expression"],
            [s["metric"] for s in record["shifts"]],
            loaded.spatial,
            family,
        )
        try:
            covers = self.ws.load_covers(ds).get(pattern_id)
        except (OSError, ValueError):  # rewritten meanwhile: the conditions decide the rows
            covers = None
        if covers is None:
            out["not_checked"].append("its rows in covers.npz (none stored)")
        diff = verify_diff(record, fresh, covers, family)
        return out | {"status": "differs" if diff else "reproduced", "diff": diff}

    @_healthy
    def geo_pattern(self, dataset_id: str, pattern_id: str) -> dict[str, Any]:
        return geo_views.geo_pattern(self.ws, self.graph(), dataset_id, pattern_id, self.settings.geo_max_points)

    @_healthy
    def geo_overlay(self, dataset_id: str) -> dict[str, Any]:
        return geo_views.geo_overlay(self.ws, self.graph(), dataset_id)

    @_healthy
    def geo_projected(self, dataset_id: str) -> list[dict[str, Any]]:
        return geo_views.geo_projected(self.ws, self.graph(), dataset_id)

    @_healthy
    def geo_window(self, dataset_id: str, bbox: str) -> dict[str, Any]:
        return geo_views.geo_window(self.ws, self.graph(), dataset_id, bbox)

    @_healthy
    def export_geosparql(self) -> str:
        return geo_views.export_geosparql(self.ws, self.graph())

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
        self._supersede_alone(record, ("SKIPPED", None))  # the last level of the new version may be one already ingested
        self.ws.save_batch(record)
        return record

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

    def _fail(self, record: dict[str, Any], exc: Exception, t0: float) -> dict[str, Any]:
        """Record the failure of a batch (called inside ``except``; ``Workspace.transaction`` has rolled its writes back)."""
        if self.ws.pending_path.exists():  # the rollback failed too (logged): its marker blocks every write until a restart
            record["warnings"].append("rollback failed; restart the service to retry it, no write runs until then")
        code = getattr(exc, "code", "internal_error")
        record.pop("superseded", None)  # the transaction rolled the removal back with the batch
        record["status"] = "FAILED"
        record["failed_stage"] = list(record["stage_times"])[-1]
        record["error"] = {"code": code, "message": str(exc)}
        levels = record["profile"].get("geo", {}).get("levels") or {}
        if code in EMPTY_LEVEL and levels.get("capped") and str(levels["chosen"][0]) == str(record.get("level")):  # @auto's finest
            finer = f"{record['geo'].partition('@')[0]}@{min(levels['capped'])}"  # the nearest level the cap left out
            record["error"]["message"] += f"; @auto left {finer} out for GEO_AUTO_MAX_CELLS alone, try it"
        if code == "internal_error":
            record["error"]["trace"] = traceback.format_exc(limit=6)
        record["metrics"]["processing_duration_s"] = time.perf_counter() - t0
        if not self.ws.pending_path.exists():  # a coarser level with nothing to mine still completes a new version
            self._supersede_alone(record, ("FAILED", code))
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
        self,
        dataset_id: str,
        *,
        source: str | None = None,
        reliability: str | None = None,
        indicators: str | None = None,
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        """Change a READY dataset's ``source`` / ``reliability`` / ``indicators`` / ``supersedes`` (None: unchanged, "":
        undeclared) on its batch record and recompile the snapshot: no new batch, no mining, the same pattern ids
        (docs/06_graph_and_storage.md §6.2). An indicator for a metric the dataset does not have is refused, and so is a
        record that fails its metadata digest (``metadata_mismatch``); a dataset or upload it now supersedes is deleted in
        the same transaction (docs/06_graph_and_storage.md §6.8)."""
        changes = resolve_dataset_metadata(source, reliability, indicators)
        if supersedes is not None:
            changes["supersedes"] = self._check_supersedes(supersedes) or None
        with self._lock:
            record = self.ws.ready_batch_for(dataset_id)
            if record is None:
                raise PipelineError("unknown_dataset", dataset_id)
            if compute_metadata_integrity(record) == "mismatch":  # a PATCH would sign the forged fields as genuine
                raise PipelineError(
                    "metadata_mismatch", f"{dataset_id}: its batch record no longer matches its digest; delete it and upload it again"
                )
            own = {b.get("dataset_id") for b in self.ws.list_batches() if record.get("upload_id") and b.get("upload_id") == record["upload_id"]}
            if changes.get("supersedes") and changes["supersedes"] in {dataset_id, record.get("upload_id"), *own}:
                raise PipelineError("invalid_options", f"supersedes {changes['supersedes']}: {dataset_id} cannot replace itself or its own upload")
            unknown = sorted(set(changes.get("indicators") or {}) - set(record["profile"].get("numerics", [])))
            if unknown:
                raise PipelineError("invalid_options", f"indicators: no metric named {unknown} in {dataset_id}")
            record.update(changes)
            for cleared in [k for k, v in changes.items() if v is None]:
                del record[cleared]
            replaced = self._targets(record, own) if changes.get("supersedes") else None
            ontology = self.ontology()
            # ponytail: a crash after save_batch and before the marker goes rolls the snapshot back under the new record until the next commit
            with self.ws.transaction({"dataset_id": dataset_id}, journal=bool(replaced)) as cp:
                superseded = self._remove(*replaced, ontology, Path(cp["dir"]) / "removed") if replaced else None
                if replaced:
                    ontology.save()
                record["metadata_digest"] = metadata_digest(record)  # a PATCH signs the record again
                snapshot, state = self._derive(ontology, pending=record)
                self.ws.save_batch(record)
            return {
                "dataset_id": dataset_id,
                "source": record.get("source"),
                "reliability": record.get("reliability", "F"),
                "indicators": record.get("indicators", {}),
                "supersedes": record.get("supersedes"),
                **({"superseded": superseded} if superseded else {}),
                "neo4j": self._publish(state, snapshot),
            }

    def _check_supersedes(self, target: str | None) -> str | None:
        """``target`` stripped (None when empty), refused unless it names a READY dataset or an upload with batches."""
        target = (target or "").strip() or None
        if target and not any(
            target in (b.get("dataset_id"), b.get("upload_id")) and b["status"] in ("READY", "SKIPPED") for b in self.ws.list_batches()
        ):
            raise PipelineError("invalid_options", f"supersedes {target}: no READY dataset or upload has this id")
        return target

    def _superseded(self, record: dict[str, Any], outcome: tuple[str, str | None]) -> tuple[set[str], list[dict[str, Any]]] | None:
        """What ``record``'s batch replaces now (``_targets``), its own ``outcome`` (status, error code) given: its
        ``supersedes`` once every level of the same upload call (one stored file) is done, one holds insights and none
        failed but for having nothing to mine (``EMPTY_LEVEL``), so the last level applies it; otherwise the old version
        stays, with a warning."""
        if not record.get("supersedes"):
            return None
        levels = [b for b in self.ws.list_batches() if b["batch_id"] != record["batch_id"] and b.get("source_path") == record["source_path"]]
        if any(b["status"] not in TERMINAL for b in levels):
            return None
        done = [outcome, *((b["status"], (b.get("error") or {}).get("code")) for b in levels)]
        if any(status == "FAILED" and code not in EMPTY_LEVEL for status, code in done) or all(status == "FAILED" for status, _ in done):
            record["warnings"].append(
                f"supersedes {record['supersedes']} not applied: a level of this upload failed or none holds insights; the old version stays"
            )
            return None
        try:
            return self._targets(record, {b.get("dataset_id") for b in levels})
        except PipelineError as exc:  # busy: the batch commits, the old version stays
            record["warnings"].append(f"supersedes {record['supersedes']} not applied: {exc}")
            return None

    def _supersede_alone(self, record: dict[str, Any], outcome: tuple[str, str | None]) -> None:
        """``supersedes`` applied by a last level that commits nothing (SKIPPED, or no insight): a deletion of its own."""
        try:
            if replaced := self._superseded(record, outcome):
                chosen = {b["batch_id"] for b in replaced[1]}
                # not a batch intent: recover rolls an unfinished removal back whatever the record's status
                removed = self._delete(lambda b: b["batch_id"] in chosen, {"superseded_by": record["batch_id"]}, "unknown_dataset")
                record["neo4j"], record["superseded"] = removed.pop("neo4j"), removed
        except Exception as exc:  # the batch's own outcome stands; the old version stays
            record["warnings"].append(f"supersedes {record['supersedes']} not applied: {type(exc).__name__}: {exc}")

    def _targets(self, record: dict[str, Any], own: set[str | None]) -> tuple[set[str], list[dict[str, Any]]] | None:
        """The datasets named by ``record['supersedes']`` (a dataset id, or every dataset of an upload id) and all their
        batches, never ``record``'s own dataset nor those in ``own``; None, with a warning, when none is left; ``busy`` while
        one of those batches is still to run."""
        target = record["supersedes"]
        every = [b for b in self.ws.list_batches() if b["batch_id"] != record["batch_id"]]
        ids = {b["dataset_id"] for b in every if target in (b.get("dataset_id"), b.get("upload_id")) and b.get("dataset_id")}
        ids -= own | {record.get("dataset_id")}
        if not ids:
            record["warnings"].append(f"supersedes {target}: nothing left to replace")
            return None
        batches = [b for b in every if b.get("dataset_id") in ids or b.get("upload_id") == target]
        if any(b["status"] not in TERMINAL for b in batches):
            raise PipelineError("busy", f"supersedes {target}: a batch of it is still to run")
        return ids, batches

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
            with self.ws.transaction(intent, journal=True) as cp:  # commit point: the marker goes
                removed = self._remove(ids, batches, ontology, Path(cp["dir"]) / "removed")
                ontology.save()
                snapshot, state = self._derive(ontology)
            return {**removed, "neo4j": self._publish(state, snapshot)}

    def _remove(self, ids: set[str], batches: list[dict[str, Any]], ontology: LatentOntology, trash: Path) -> dict[str, Any]:
        """Inside a journal transaction: move the datasets' artefacts to ``trash``, drop their journal rows (renumbered) and
        let ``ontology`` forget their memberships (orphan anchors go); the caller saves the ontology and derives."""
        records, vectors = self.ws.patterns(), self.ws.vectors()
        kept = [r for r in records if r["dataset_id"] not in ids]
        row_of = {r["row_id"]: i for i, r in enumerate(kept)}  # journal order, renumbered
        all_acts = self.ws.activations()
        acts = [{**a, "row_id": row_of[a["row_id"]]} for a in all_acts if a["row_id"] in row_of]
        lost = {int(a["attractor_id"]) for a in all_acts if a["row_id"] not in row_of}  # the anchors the deletion touches
        self.ws.remove_datasets(ids, batches, trash)  # first: the snapshot must not see them
        kept_vectors = vectors[[r["row_id"] for r in kept]]
        dropped = ontology.forget(acts, kept_vectors, lost)
        self.ws.rewrite_journal([{**r, "row_id": row_of[r["row_id"]]} for r in kept], kept_vectors, acts)
        return {
            "datasets": sorted(ids),
            "batches": [b["batch_id"] for b in batches],
            "patterns_removed": len(records) - len(kept),
            "anchors_removed": dropped,
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
