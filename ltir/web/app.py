"""Local web app: upload, processing status, graph, node inspection, chat (docs/08_interface.md).

Run: ``python -m ltir.web`` (http://127.0.0.1:8765). Batches are processed by a
single background worker so the ontology is updated strictly sequentially.
"""

from __future__ import annotations

import html
import importlib
import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ltir.canonical import humanize
from ltir.config import Config, load_config
from ltir.pipeline import TERMINAL, Engine, PipelineError
from ltir.store import RepresentationMismatch, WorkspaceBusy
from ltir.synth import DEMO_PATH, write_demo

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("ltir.web")


class QueryIn(BaseModel):
    question: str
    use_llm: bool = True


class SphereIn(BaseModel):
    dataset: str | None = None
    highlight: dict[str, list[str]] | None = None
    palette: dict[str, str] | None = None  # the UI's theme colours, so both views share one visual language
    layers: dict[str, bool] | None = None  # the legend toggles' state


def _sphere_page(
    engine: Engine, dataset: str | None, highlight: dict | None, palette: dict | None = None, layers: dict | None = None
) -> HTMLResponse:
    from ltir.sphere import SphereError, sphere_html

    try:
        return HTMLResponse(sphere_html(engine, dataset=dataset or None, highlight=highlight, palette=palette, layers=layers))
    except SphereError as exc:
        return HTMLResponse(f"<body style='background:#0f172a;color:#cbd5e1;font:14px system-ui;padding:24px'>{html.escape(str(exc))}</body>")


def _short_label(graph, node: dict[str, Any]) -> str:
    if node["kind"] != "Pattern":
        return node["label"]
    ins = graph.insight(node["id"])
    scope = " · ".join(c.value for c in ins.conditions)
    if ins.phenomenon_type == "covariance" and ins.covariance:
        a, b = sorted(ins.covariance["pair"])
        return f"{scope}\ncorr({humanize(a)}, {humanize(b)})"
    return f"{scope}\n{humanize(ins.target)} {ins.effect_size:+.2f} sd"


def cytoscape_elements(engine: Engine, dataset: str | None = None) -> dict[str, Any]:
    graph = engine.graph()
    keep: set[str] = set()
    nodes = []
    for n in graph.nodes.values():
        p = n["props"]
        if dataset and n["kind"] in {"Pattern", "Dimension", "Metric", "Dataset"} and p.get("dataset_id") != dataset:
            continue
        if dataset and n["kind"] == "Attractor" and dataset not in p.get("datasets", []):
            continue
        keep.add(n["id"])
        data = {"id": n["id"], "kind": n["kind"], "label": _short_label(graph, n), "full_label": n["label"]}
        if n["kind"] == "Pattern":
            ins = graph.insight(n["id"])
            anchor = max((e for e, _ in graph.incident(n["id"], ["ACTIVATES"])), key=lambda e: e["weight"], default=None)
            data.update(
                weight=round(ins.weight, 3),
                direction=1 if ins.effect_size > 0 else -1,
                ptype=ins.phenomenon_type,
                dataset=ins.dataset_id,
                support=ins.support,
                target=ins.target,
                effect=round(ins.effect_size, 3),
                scope=[c.expr for c in ins.conditions],
                anchor=anchor["target"] if anchor else None,
                anchor_label=graph.nodes[anchor["target"]]["label"] if anchor else None,
            )
        elif n["kind"] == "Attractor":
            data.update(mass=p["mass"], n_patterns=p["n_patterns"])
        nodes.append({"data": data})
    edges = [
        {
            "data": {
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "type": e["type"],
                "plane": e["plane"],
                "weight": round(float(e["weight"]), 3),
                "weak": bool(e["props"].get("weak", False)),  # a coverage-only membership: drawn, not walked
            }
        }
        for e in graph.edges
        if e["source"] in keep and e["target"] in keep
    ]
    return {"nodes": nodes, "edges": edges, "stats": graph.snapshot.get("stats", {})}


def node_details(engine: Engine, node_id: str) -> dict[str, Any]:
    graph = engine.graph()
    node = graph.nodes.get(node_id)
    if node is None:
        raise HTTPException(404, f"unknown node {node_id}")
    neighbors: dict[str, list[dict[str, Any]]] = {}
    for e in graph.out.get(node_id, []) + graph.inc.get(node_id, []):
        other = e["target"] if e["source"] == node_id else e["source"]
        key = e["type"] + ("" if e["source"] == node_id else " (in)")
        neighbors.setdefault(key, []).append(
            {"id": other, "label": graph.nodes[other]["label"], "kind": graph.nodes[other]["kind"], "weight": e["weight"], "props": e["props"]}
        )
    for items in neighbors.values():
        items.sort(key=lambda x: -float(x["weight"]))
    return {"node": node, "neighbors": neighbors}


def create_app(config: Config | None = None, engine: Engine | None = None) -> FastAPI:
    config = config or load_config()
    engine = engine or Engine(config)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltir-batch")
    uploads = Path(config.workspace_dir) / "uploads"
    app = FastAPI(title="SIG — Latent Transversal Insight Representation")
    app.state.engine = engine

    def schedule(record: dict[str, Any]) -> dict[str, Any]:
        worker.submit(engine.process, record["batch_id"])
        return record

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "llm": engine.llm.health(),
            "neo4j": {"enabled": config.neo4j_enabled, "uri": config.neo4j_uri if config.neo4j_enabled else None, "database": config.neo4j_database},
            "embedding": engine.ws.representation(),
            "embedding_device": getattr(engine.encoder.embedder, "device", "n/a"),
            "graph": engine.graph().snapshot.get("stats", {}),
            "workspace": str(config.workspace_dir),
        }

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return config.public_dict()

    @app.get("/api/batches")
    def batches() -> list[dict[str, Any]]:
        return [{k: v for k, v in b.items() if k not in {"checkpoint"}} for b in reversed(engine.ws.list_batches())]

    @app.get("/api/batches/{batch_id}")
    def batch(batch_id: str) -> dict[str, Any]:
        record = engine.ws.load_batch(batch_id)
        if record is None:
            raise HTTPException(404, "unknown batch")
        return record

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), bins: str = Form(""), categories: str = Form("")) -> JSONResponse:
        name = Path(file.filename or "upload.csv").name
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / f"{uuid.uuid4().hex[:8]}_{name}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        # empty form fields mean "use the workspace defaults" (BIN_COLUMNS / CATEGORICAL_COLUMNS)
        record = engine.submit(dest, filename=name, bins=bins.strip() or None, categories=categories.strip() or None)
        return JSONResponse(schedule(record), status_code=202)

    @app.post("/api/demo")
    def demo() -> JSONResponse:
        path = DEMO_PATH if DEMO_PATH.exists() else write_demo()
        # the demo is self-describing: explicitly no bands / overrides
        return JSONResponse(schedule(engine.submit(path, filename=path.name, bins="", categories="")), status_code=202)

    @app.get("/api/graph")
    def graph(dataset: str | None = None) -> dict[str, Any]:
        return cytoscape_elements(engine, dataset)

    @app.get("/api/nodes/{node_id}")
    def node(node_id: str) -> dict[str, Any]:
        return node_details(engine, node_id)

    @app.post("/api/query")
    def query(body: QueryIn) -> dict[str, Any]:
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        try:
            return engine.ask(body.question, use_llm=body.use_llm).to_dict()
        except RepresentationMismatch as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/sphere", response_class=HTMLResponse)
    def sphere(dataset: str | None = None) -> HTMLResponse:
        return _sphere_page(engine, dataset, None)

    @app.post("/api/sphere", response_class=HTMLResponse)
    def sphere_highlight(body: SphereIn) -> HTMLResponse:
        return _sphere_page(engine, body.dataset, body.highlight, body.palette, body.layers)

    @app.get("/vendor/plotly.min.js")
    def plotly_js() -> FileResponse:
        from ltir.sphere import plotly_js_path

        return FileResponse(plotly_js_path(), media_type="application/javascript")

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            return engine.delete_dataset(dataset_id)
        except PipelineError as exc:
            raise HTTPException(404 if exc.code == "unknown_dataset" else 409, str(exc)) from exc

    @app.post("/api/reset")
    def reset() -> dict[str, Any]:
        busy = [b["batch_id"] for b in engine.ws.list_batches() if b["status"] not in TERMINAL]
        if busy:
            raise HTTPException(409, f"batches in progress: {busy}")
        return {"status": "reset", **engine.reset()}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def log_gpu_headroom() -> None:
    """Report free GPU memory after the embedder is loaded. SIG never loads the LLM in-process
    (it is only reached through LLM_BASE_URL), so a local LLM server must fit in what is left."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            log.info("GPU memory after warm-up: %.1f GB free of %.1f GB (LLM is external: LLM_BASE_URL)", free / 2**30, total / 2**30)
    except Exception as exc:  # observability only; never blocks startup
        log.info("GPU memory unavailable: %s", exc)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    try:
        app = create_app(config)
    except WorkspaceBusy as exc:  # another writer (a CLI ingest, a second server) holds the workspace
        log.error("%s", exc)
        raise SystemExit(2) from None
    engine: Engine = app.state.engine
    log.info("warming up embedding model %s", config.embedding_model)
    engine.encoder.embedder.embed(["warmup"])
    log.info("embedding device: %s", getattr(engine.encoder.embedder, "device", "n/a"))
    log_gpu_headroom()

    def _warm_viz() -> None:
        # Deferred on purpose: plotly/prosphera take a few seconds and must not block startup.
        importlib.import_module("ltir.engines.lac.projector")
        log.info("3D sphere projector ready")

    threading.Thread(target=_warm_viz, daemon=True).start()
    uvicorn.run(app, host=config.web_host, port=config.web_port, log_level="info")
