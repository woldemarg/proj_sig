"""The graph service's HTTP API (docs/08_interface.md): uploads and batches, the graph and its nodes, the sphere,
dataset deletion and reset, and ``POST /api/search`` — the evidence for one question, which the narrator verbalises.

Run: ``python -m insight_graph_service.server`` (http://127.0.0.1:8765). Batches are processed by a single background
worker, so the ontology is updated strictly sequentially.
"""

from __future__ import annotations

import io
import logging
import re
import time
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from attractor_topology.encoder import model_folder
from insight_graph_service.core.demo import generate_retail_dataset
from insight_graph_service.core.engine import Engine, PipelineError
from insight_graph_service.core.model_store import problems
from insight_graph_service.core.settings import PROJECT_ROOT, Settings
from insight_graph_service.core.workspace import RepresentationMismatch, RollbackPending, UploadTooLarge, WorkspaceBusy
from insight_graph_service.server.views import cytoscape_elements, node_details, sphere_points

log = logging.getLogger(__name__)
STATUS = {
    "file_too_large": 413,
    "unknown_dataset": 404,
    "unknown_upload": 404,
    "unknown_pattern": 404,
    "invalid_options": 400,
}  # every other refusal is 409: busy, degraded, mismatch, rollback


SLOW_REQUEST_S = 2.0  # a request slower than this is logged as a warning (docs/12_architecture.md §12.7)


class RequestTimer:
    """ASGI middleware: every request's method, path, status and wall time in the service log (a streamed file until its
    last byte); a slow one as a warning."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start, status = time.perf_counter(), [None]

        async def sent(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status[0] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, sent)
        finally:
            seconds = time.perf_counter() - start
            (log.warning if seconds > SLOW_REQUEST_S else log.info)("%s %s %s %.3f s", scope["method"], scope["path"], status[0], seconds)


class SearchIn(BaseModel):
    question: str = ""  # may be empty when a zone is given
    cells: list[str] | None = None  # a map zone as H3 cells
    geometry: dict[str, Any] | None = None  # or as GeoJSON, {center, radius_m} or {bbox}
    resolution: int | None = Field(None, ge=0, le=15)  # the zone weighs the geo datasets of at most this resolution


class DatasetMetadataIn(BaseModel):
    source: str | None = None  # None: unchanged; "": undeclared
    reliability: str | None = None
    indicators: str | None = None
    supersedes: str | None = None  # a dataset or upload id this dataset replaces: deleted now


def _log_failure(future: Future) -> None:
    if future.exception() is not None:  # process() records its own failures; this is what escaped it
        log.error("background job failed", exc_info=future.exception())


def create_app(engine: Engine) -> FastAPI:
    settings = engine.settings
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sig-batch")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        worker.shutdown(wait=False, cancel_futures=True)  # a queued batch stays UPLOADED and fails at the next start

    app = FastAPI(title="SIG insight graph service", lifespan=lifespan)
    app.add_middleware(RequestTimer)

    async def refused(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=STATUS.get(exc.code, 409))

    async def failed(_: Request, exc: Exception) -> JSONResponse:
        log.exception("request failed", exc_info=exc)
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}", "code": "internal_error"}, status_code=500)

    for error in (PipelineError, RepresentationMismatch, RollbackPending, UploadTooLarge):
        app.add_exception_handler(error, refused)
    app.add_exception_handler(Exception, failed)

    def run(job, *args: Any) -> None:
        worker.submit(job, *args).add_done_callback(_log_failure)

    def accept(name: str, stream: BinaryIO, bins: str | None, categories: str | None, geo: str | None = None, **options: Any) -> JSONResponse:
        records = engine.upload(name, stream, bins=bins, categories=categories, geo=geo, **options)
        for record in records:  # one worker thread: the levels run in turn
            run(engine.process, record["batch_id"])
        levels = {record["level"]: record["batch_id"] for record in records}
        return JSONResponse(records[0] | {"levels": levels}, status_code=202)  # the finest level's record, every level's batch

    run(engine.startup_sync)  # on the batch thread: Neo4j writes stay sequential

    @app.get("/api/health")
    async def health() -> dict[str, Any]:  # on the event loop: answers while the batch and search threads are busy
        return {
            "neo4j": {
                "enabled": settings.neo4j_enabled,
                "uri": settings.neo4j_uri if settings.neo4j_enabled else None,
                "database": settings.neo4j_database,
            },
            "embedding": engine.ws.representation(),
            "embedding_device": getattr(engine.encoder.embedder, "device", "n/a"),
            "graph": engine.graph().snapshot.get("stats", {}),
            "workspace": str(settings.workspace_dir),
            "problem": engine.problem,
        }

    @app.get("/api/batches")
    def batches(upload_id: str | None = None) -> list[dict[str, Any]]:
        return [engine.polled(b) for b in reversed(engine.ws.list_batches()) if upload_id in (None, b.get("upload_id"))]

    @app.get("/api/batches/{batch_id}")
    def batch(batch_id: str) -> dict[str, Any]:
        record = engine.ws.load_batch(batch_id) if re.fullmatch(r"[\w-]+", batch_id) else None  # an id, never a path
        if record is None:
            raise HTTPException(404, "unknown batch")
        return engine.polled(record)

    @app.post("/api/upload")
    def upload(
        request: Request,
        file: UploadFile = File(...),
        bins: str = Form(""),
        categories: str = Form(""),
        geo: str = Form(""),
        source: str = Form(""),
        reliability: str = Form(""),
        indicators: str = Form(""),
        table: bool = Form(False),
        supersedes: str = Form(""),
    ) -> JSONResponse:
        if (length := request.headers.get("content-length")) and int(length) > engine.settings.max_upload_bytes:  # refused before the body is read
            raise UploadTooLarge(f"{file.filename} exceeds {engine.settings.max_upload_mb} MB (MAX_UPLOAD_MB)")
        # empty form fields mean "use the workspace defaults" (BIN_COLUMNS / CATEGORICAL_COLUMNS); no geo = a plain table
        name = Path(file.filename or "upload.csv").name
        return accept(
            name,
            file.file,
            bins.strip() or None,
            categories.strip() or None,
            geo.strip() or None,
            source=source,
            reliability=reliability,
            indicators=indicators,
            table=table,
            supersedes=supersedes,
        )

    @app.post("/api/demo")
    def demo() -> JSONResponse:
        table = io.BytesIO(generate_retail_dataset().to_csv(index=False).encode("utf-8"))
        return accept("retail_synthetic.csv", table, "", "")  # the demo is self-describing: explicitly no bands / overrides

    @app.post("/api/demo/geo")
    def demo_geo() -> JSONResponse:
        # the shipped California housing table (located block groups) as H3 cells at resolution 5, with the cells' typical
        # income as a band dimension (docs/10_verification.md §10.2); the Dockerfile copies this one file of data/
        with (PROJECT_ROOT / "data" / "demo" / "california_housing.csv").open("rb") as table:
            return accept("california_housing.csv", table, "median_income_median:4", "", "auto@5")

    @app.get("/api/graph")
    def graph(dataset: str | None = None, regions: bool = False) -> dict[str, Any]:
        return cytoscape_elements(engine.graph(), dataset, regions)

    @app.get("/api/nodes/{node_id:path}")  # column ids hold raw column names, '/' included
    def node(node_id: str) -> dict[str, Any]:
        return node_details(engine.graph(), node_id)

    @app.post("/api/search")
    def search(body: SearchIn) -> dict[str, Any]:
        if body.cells is not None and body.geometry is not None:
            raise HTTPException(400, "a zone is cells or geometry, not both")
        if not body.question.strip() and body.cells is None and body.geometry is None:
            raise HTTPException(400, "empty question")
        return engine.evidence(body.question, cells=body.cells, geometry=body.geometry, resolution=body.resolution).to_dict()

    @app.post("/api/patterns/{pattern_id}/verify")
    def verify_pattern(pattern_id: str) -> dict[str, Any]:
        return engine.verify_pattern(pattern_id)

    @app.get("/api/sphere")
    def sphere(dataset: str | None = None) -> dict[str, Any]:
        return sphere_points(engine.committed(), settings.topology.random_seed, dataset or None)

    @app.get("/api/geo")
    def geo_datasets() -> list[dict[str, Any]]:
        return engine.geo_datasets()

    @app.get("/api/geo/{dataset_id}/cells")
    def geo_cells(dataset_id: str) -> Response:
        return Response(engine.geo_cells_pack(dataset_id), media_type="application/json")

    @app.get("/api/export/geosparql.ttl")
    def export_geosparql() -> Response:
        return Response(engine.export_geosparql(), media_type="text/turtle")

    @app.get("/api/geo/{dataset_id}/overlay")
    def geo_overlay(dataset_id: str) -> dict[str, Any]:
        return engine.geo_overlay(dataset_id)

    @app.get("/api/geo/{dataset_id}/projected")
    def geo_projected(dataset_id: str) -> list[dict[str, Any]]:
        return engine.geo_projected(dataset_id)

    @app.get("/api/geo/{dataset_id}/window")
    def geo_window(dataset_id: str, bbox: str) -> dict[str, Any]:
        return engine.geo_window(dataset_id, bbox)

    @app.get("/api/geo/{dataset_id}/patterns/{pattern_id}")
    def geo_pattern(dataset_id: str, pattern_id: str) -> dict[str, Any]:
        return engine.geo_pattern(dataset_id, pattern_id)

    @app.patch("/api/datasets/{dataset_id}")
    def update_dataset(dataset_id: str, body: DatasetMetadataIn) -> dict[str, Any]:
        return engine.update_dataset_metadata(
            dataset_id, source=body.source, reliability=body.reliability, indicators=body.indicators, supersedes=body.supersedes
        )

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str) -> dict[str, Any]:
        return engine.delete_dataset(dataset_id)

    @app.delete("/api/uploads/{upload_id}")
    def delete_upload(upload_id: str) -> dict[str, Any]:
        return engine.delete_upload(upload_id)

    @app.post("/api/reset")
    def reset() -> dict[str, Any]:
        return {"status": "reset", **engine.reset()}

    return app


def log_gpu_headroom() -> None:
    """Report free GPU memory after the embedder is loaded: what a local LLM server on the same GPU has left."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            log.info("GPU memory after warm-up: %.1f GB free of %.1f GB", free / 2**30, total / 2**30)
    except Exception as exc:  # observability only; never blocks startup
        log.info("GPU memory unavailable: %s", exc)


def ready_app(settings: Settings) -> FastAPI:
    """The app as a service start needs it (also used by scripts/dev.py): the model files checked before anything
    touches the workspace, then the engine opened and the model warmed up."""
    folder = model_folder(settings.topology.model_dir)
    missing = problems(folder, hashes=False)
    if missing:  # before the engine opens: a clear message instead of a stack trace
        log.error("embedding model in %s is incomplete (%s): run python -m insight_graph_service.core.model_store", folder, "; ".join(missing))
        raise SystemExit(3)
    try:
        engine = Engine(settings)
    except WorkspaceBusy as exc:  # another writer (a second server, a script) holds the workspace
        log.error("%s", exc)
        raise SystemExit(2) from None
    log.info("warming up the embedding model")
    engine.encoder.embedder.embed(["warmup"])
    log.info("embedding device: %s", getattr(engine.encoder.embedder, "device", "n/a"))
    log_gpu_headroom()
    return create_app(engine)
