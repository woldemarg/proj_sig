"""The bridge's HTTP host: ``GET /api/ads/health``, ``POST /api/ads/preview`` (the task's text before the ask),
``POST /api/ads/jobs`` (a zone's question run through the graph, ADS and the final report) and
``GET /api/ads/jobs/{id}`` (its stage and results)."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from jinja2 import Template
from pydantic import BaseModel, Field

from ads_bridge_service.settings import AdsSettings
from ads_bridge_service.task import build_aoi_md, build_domain_md, build_query_parts, finding_files, name_tables, select_findings

log = logging.getLogger(__name__)

JOINT_TEMPLATE = Path(__file__).resolve().parent / "prompts" / "joint_response.jinja"
JOINT_MAX_TOKENS = 16384  # the prompt keeps every row of the report's tables
JOBS_KEPT = 20
STATUS_RETRIES = 5  # failed reads of an ADS job's status in a row before it is given up
_WORKSPACE_REFS = re.compile(r"<iframe[^>]*>.*?</iframe>|!?\[([^\]]*)\]\(/workspace/[^)]*\)", re.S)  # files of ADS's own workspace


class TableIn(BaseModel):
    dataset_id: str
    filename: str


class AskIn(BaseModel):
    question: str = ""
    tables: list[TableIn] = Field(min_length=1)  # the first is the base table, the rest its context layers
    polygon: dict[str, Any] | None = None  # the drawn zone (GeoJSON Polygon)
    cells: list[str] = []  # its cells on the open map: the zone the graph is asked about


class StageError(RuntimeError):
    """A job's failure with its code (docs/09_operations.md §9.4)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _checked(resp: httpx.Response, code: str, what: str) -> httpx.Response:
    if resp.status_code >= 300:
        raise StageError(code, f"{what}: HTTP {resp.status_code} {resp.text[:300]}")
    return resp


def ads_report(status: dict[str, Any]) -> str:
    """The report of a finished ADS job, the links into ADS's own workspace taken out."""
    last = (status.get("result") or {}).get("last_execute_response") or {}
    text = last.get("report") or ""
    if not text and last.get("answer") is not None:
        text = json.dumps(last["answer"], ensure_ascii=False, indent=2)
    return _WORKSPACE_REFS.sub(lambda m: m.group(1) or "", text).strip()


def create_app(settings: AdsSettings | None = None) -> FastAPI:
    settings = settings or AdsSettings.from_env()
    http = httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0))
    jobs: dict[str, dict[str, Any]] = {}  # ponytail: in memory, the last JOBS_KEPT ended ones; a store if jobs must survive a restart
    jobs_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        http.close()

    app = FastAPI(title="SIG ADS bridge", lifespan=lifespan)

    def ask_graph(question: str, cells: list[str]) -> list[dict[str, Any]]:
        body = {"question": question, "cells": cells, "use_llm": False}
        return select_findings(_checked(http.post(f"{settings.ads_narrator_url}/api/chat/query", json=body), "sig_failed", "SIG").json())

    def read_tables(body: AskIn) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
        """Each picked table's rows against the zone, the layers' cells at the base table's resolution: one cell column to join on."""
        tables, files, resolution = [], {}, None
        for t in name_tables([t.model_dump() for t in body.tables]):
            url = f"{settings.ads_graph_url}/api/geo/{quote(t['dataset_id'], safe='')}/aoi"  # one path segment, whatever the id holds
            resp = _checked(http.post(url, json={"polygon": body.polygon, "resolution": resolution}), "sig_failed", f"SIG rows of {t['filename']}")
            tables.append(t | json.loads(resp.headers["x-aoi"]))
            files[t["file"]] = resp.content
            resolution = tables[0]["resolution"]
        return tables, files

    def run_ads(query: str, files: dict[str, bytes]) -> str:
        """The task run in a session of its own, closed whatever happens; a status read that fails is read again, a
        job is given up after STATUS_RETRIES failures in a row or ADS_JOB_TIMEOUT_S."""
        ads = settings.ads_url
        sid = _checked(http.post(f"{ads}/sessions"), "ads_failed", "ADS session").json()["session_id"]
        try:
            for name, data in files.items():
                _checked(http.post(f"{ads}/sessions/{sid}/datasets", files={"file": (name, data)}), "ads_failed", f"ADS upload {name}")
            _checked(http.post(f"{ads}/sessions/{sid}/jobs", json={"user_query": query}), "ads_failed", "ADS job")
            deadline, failures = time.monotonic() + settings.ads_job_timeout_s, 0
            while True:
                try:
                    status = http.get(f"{ads}/sessions/{sid}/status")
                    if status.status_code == 200:
                        break
                    failures = 0 if status.status_code == 202 else failures + 1
                    if failures > STATUS_RETRIES:
                        _checked(status, "ads_failed", "ADS status")
                except httpx.HTTPError:
                    failures += 1
                    if failures > STATUS_RETRIES:
                        raise
                if time.monotonic() > deadline:
                    raise StageError("ads_timeout", f"the ADS job still ran after {settings.ads_job_timeout_s:g} s")
                time.sleep(settings.ads_poll_s)
        finally:
            try:
                http.post(f"{ads}/sessions/{sid}/close")
            except httpx.HTTPError as exc:  # the report, if any, stands
                log.warning("ADS session %s not closed: %s", sid, exc)
        report = ads_report(status.json())
        if not report:
            raise StageError("ads_failed", f"ADS returned no report (outcome {status.json().get('outcome')})")
        return report

    def write_report(question: str, report: str, domain: str) -> str:
        """The final report from the ADS report with the zone's methodology and the glossary; "" without its required sections."""
        prompt = Template(JOINT_TEMPLATE.read_text(encoding="utf-8")).render(
            user_query=question, report=report, domain_knowledge=domain, query_language="uk"
        )
        body = {"model": "synthesis", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": JOINT_MAX_TOKENS}
        resp = _checked(http.post(f"{settings.ads_llm_url}/v1/chat/completions", json=body), "ads_failed", "ADS LLM gateway")
        out = resp.json()["choices"][0]["message"]["content"]
        return out if "## Результати аналізу" in out and "## Методологія" in out else ""

    def run(job: dict[str, Any], body: AskIn) -> None:
        started = time.monotonic()

        def enter(stage: str) -> None:
            nonlocal started
            job["timings"][job["stage"]] = round(time.monotonic() - started, 1)
            started, job["stage"] = time.monotonic(), stage

        try:
            findings = ask_graph(body.question, body.cells)
            if not any(f["inside"] for f in findings):  # the text pulled the seeds out of the zone: the zone alone, if it holds any
                alone = ask_graph("", body.cells)
                findings = alone if any(f["inside"] for f in alone) else findings
            tables, files = read_tables(body)
            sources = finding_files(findings)
            job["insights"] = findings
            job["ads_query"] = "".join(p["text"] for p in build_query_parts(body.question, tables, findings))
            files["aoi.md"] = build_aoi_md(body.polygon, tables, sources).encode("utf-8")
            enter("ads")
            job["ads_report"] = run_ads(job["ads_query"], files)
            enter("joint")
            try:
                job["joint_report"] = write_report(body.question, job["ads_report"], build_domain_md(body.polygon, tables, len(body.cells), sources))
            except Exception as exc:  # ADS's report stands without the final one
                log.exception("ADS job %s: no final report", job["job_id"])
                job.update(joint_report="", joint_error=f"{type(exc).__name__}: {exc}")
            enter("done")
        except Exception as exc:  # recorded on the job: the console shows the stage and the reason
            log.exception("ADS job %s failed", job["job_id"])
            unreachable = isinstance(exc, httpx.HTTPError)
            code = (
                exc.code
                if isinstance(exc, StageError)
                else ("ads_unavailable" if job["stage"] != "search" else "sig_failed")
                if unreachable
                else "internal_error"
            )
            job.update(failed_stage=job["stage"], stage="error", error={"code": code, "detail": f"{type(exc).__name__}: {exc}"})

    @app.get("/api/ads/health")
    def health() -> dict[str, Any]:
        def reachable(url: str) -> bool:
            try:
                return http.get(f"{url}/health", timeout=2).status_code == 200
            except httpx.HTTPError:
                return False

        ads, llm = reachable(settings.ads_url), reachable(settings.ads_llm_url)
        return {"ok": ads and llm, "ads": {"reachable": ads, "url": settings.ads_url}, "llm": {"reachable": llm, "url": settings.ads_llm_url}}

    @app.post("/api/ads/preview")
    def preview(body: AskIn) -> dict[str, Any]:
        return {"parts": build_query_parts(body.question, name_tables([t.model_dump() for t in body.tables]), None)}

    @app.post("/api/ads/jobs")
    def start(body: AskIn) -> JSONResponse:
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        if not body.polygon or not body.cells:
            return JSONResponse({"detail": "a question to ADS needs a zone drawn on the map", "code": "empty_zone"}, status_code=400)
        job = {"job_id": uuid.uuid4().hex[:12], "stage": "search", "question": body.question, "timings": {}}
        with jobs_lock:  # a running job is never dropped
            ended = [k for k, j in jobs.items() if j["stage"] in ("done", "error")]
            for old in ended[: max(0, len(ended) - JOBS_KEPT + 1)]:
                del jobs[old]
            jobs[job["job_id"]] = job
        threading.Thread(target=run, args=(job, body), name=f"ads-job-{job['job_id']}", daemon=True).start()
        return JSONResponse({"job_id": job["job_id"]}, status_code=202)

    @app.get("/api/ads/jobs/{job_id}")
    def job_state(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise HTTPException(404, f"no ADS job {job_id}")
        return jobs[job_id]

    return app
