"""The bridge's HTTP host: ``GET /api/ads/health``, ``POST /api/ads/preview`` (the task's text before the ask),
``POST /api/ads/jobs`` (a question about zones run through the graph, ADS and the final report, in the conversation's
ADS session), ``GET /api/ads/jobs/{id}`` (its stage and results) and ``POST /api/ads/sessions/{id}/close`` (the
conversation's end: its tables deleted from ADS, its session closed)."""

from __future__ import annotations

import contextlib
import copy
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
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse
from jinja2 import Template
from pydantic import BaseModel, Field, model_validator

from ads_bridge_service.settings import AdsSettings
from ads_bridge_service.task import (
    ZONES_FILE,
    ZONES_MD,
    build_domain_md,
    build_query_parts,
    build_zones_md,
    finding_files,
    merge_findings,
    name_tables,
    select_findings,
)

log = logging.getLogger(__name__)

JOINT_TEMPLATE = Path(__file__).resolve().parent / "prompts" / "joint_response.jinja"
JOINT_MAX_TOKENS = 16384  # the prompt keeps every row of the report's tables
JOBS_KEPT = 20
STATUS_RETRIES = 5  # failed reads of an ADS job's status in a row before it is given up
MAX_FOCUS = 5  # the zones one question may compare
SESSION_ID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"  # ADS's: a uuid4, a path segment as it is
ZONE_ID = r"^[A-Za-z0-9_-]{1,16}$"  # the graph service's zone ids
# the calendar periods the graph reads in a question's text (insight_contracts.wording.PERIOD)
_PERIODS = re.compile(r"(?<![\w-])(?:\d{4}-W\d{2}|\d{4}-Q[1-4]|Q[1-4]\s+\d{4}|\d{4}-\d{2}(?:-\d{2})?)(?![\d-])", re.I)
_HEADER = re.compile(r"^Період:.*$", re.M)  # the console's period line: ADS's, no words of the question
_WORKSPACE_REFS = re.compile(r"<iframe[^>]*>.*?</iframe>|!?\[([^\]]*)\]\(/workspace/[^)]*\)", re.S)  # files of ADS's own workspace


class TableIn(BaseModel):
    dataset_id: str
    filename: str
    resolution: int = Field(ge=0, le=15)  # the dataset's own; the base table's is every table's cell columns'


class ZoneIn(BaseModel):
    id: str = Field(pattern=ZONE_ID)
    polygon: dict[str, Any]  # a drawn zone (GeoJSON Polygon): the graph's evidence and the tables' rows both read it


class AskIn(BaseModel):
    question: str = ""
    tables: list[TableIn] = Field(min_length=1)  # the first is the base table, the rest its context layers
    zones: list[ZoneIn] = Field(min_length=1)  # every zone of the conversation: a later question may name an earlier one
    focus: list[str] = Field(min_length=1, max_length=MAX_FOCUS)  # the zones this question is about
    session_id: str | None = Field(None, pattern=SESSION_ID)  # the conversation's ADS session; None: its first question
    period: list[str] | None = Field(None, min_length=2, max_length=2)  # the console's period bar, [start, end): the graph's only period

    @model_validator(mode="after")
    def _focus_in_zones(self) -> AskIn:
        ids = [z.id for z in self.zones]
        if len(set(ids)) != len(ids) or not set(self.focus) <= set(ids):
            raise ValueError("zone ids repeat, or the question names a zone the conversation lacks")
        return self


class StageError(RuntimeError):
    """A job's failure with its code (docs/09_operations.md §9.4)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _checked(resp: httpx.Response, code: str, what: str) -> httpx.Response:
    if resp.status_code >= 300:
        raise StageError(code, f"{what}: HTTP {resp.status_code} {resp.text[:300]}")
    return resp


def error_code(exc: Exception, stage: str) -> str:
    """A failed job's code (docs/09_operations.md §9.4): its stage's own, else a service that did not answer, else a fault here."""
    if isinstance(exc, StageError):
        return exc.code
    if isinstance(exc, httpx.HTTPError):
        return "sig_failed" if stage == "search" else "ads_unavailable"
    return "internal_error"


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
    # each job's state as last published; ponytail: in memory, the last JOBS_KEPT ended ones; a store if jobs must survive a restart
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    def publish(job: dict[str, Any]) -> None:
        """The job's state for the console: a copy, so a read never meets the worker's own dict half-changed."""
        with jobs_lock:
            jobs[job["job_id"]] = copy.deepcopy(job)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        http.close()

    app = FastAPI(title="SIG ADS bridge", lifespan=lifespan)

    def ask_graph(question: str, polygon: dict[str, Any], period: list[str] | None) -> list[dict[str, Any]]:
        """A zone's evidence: the graph reads the polygon as the tables' rows are read against it (``geometry_to_cells``),
        one place, and the period bar's days, never a period the question's text names (that one is ADS's)."""
        ask = {"question": _PERIODS.sub(" ", _HEADER.sub(" ", question)), "geometry": polygon, "use_llm": False}
        if period:
            ask["period"] = period
        return select_findings(_checked(http.post(f"{settings.ads_narrator_url}/api/chat/query", json=ask), "sig_failed", "SIG").json())

    def zone_findings(body: AskIn, zones: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Each asked zone's findings, merged; a zone whose answer to the text holds none inside it is asked alone."""
        by_zone = {}
        for z in body.focus:
            found = ask_graph(body.question, zones[z], body.period)
            if not any(f["inside"] for f in found):  # the text pulled the seeds out of the zone: the zone alone, if it holds any
                alone = ask_graph("", zones[z], body.period)
                found = alone if any(f["inside"] for f in alone) else found
            by_zone[z] = found
        return merge_findings(by_zone)

    def place_tables(body: AskIn) -> tuple[list[dict[str, Any]], bytes]:
        """The picked tables, each with what the graph service tells of it and its rows per zone, and the zones file: which
        rows each zone of the conversation holds, the layers' cells at the base table's resolution, one column to join on."""
        zones = [{"id": z.id, "polygon": z.polygon} for z in body.zones]
        tables, lines = [], ["zone_id,table,row_id,aoi_zone"]
        for t in name_tables([t.model_dump() for t in body.tables]):
            url = f"{settings.ads_graph_url}/api/geo/{quote(t['dataset_id'], safe='')}/zone_rows"  # one path segment, whatever the id holds
            resp = _checked(http.post(url, json={"zones": zones, "resolution": t["cells_at"]}), "sig_failed", f"SIG zones of {t['filename']}")
            tables.append(t | json.loads(resp.headers["x-zones"]))
            for line in resp.text.splitlines()[1:]:  # zone_id,row_id,aoi_zone: an id, a number and a word, no quoting
                zone, _, rest = line.partition(",")
                lines.append(f"{zone},{t['file']},{rest}")
        return tables, ("\n".join(lines) + "\n").encode("utf-8")

    def session_files(sid: str) -> set[str]:
        """The files the session holds already: a table sent on an earlier question is not sent again."""
        files = _checked(http.get(f"{settings.ads_url}/sessions/{sid}/sources"), "ads_failed", "ADS session files").json()["files"]
        return {f["name"] for f in files if f.get("source_kind") == "uploaded_input"}

    def upload(sid: str, tables: list[dict[str, Any]], files: dict[str, bytes]) -> None:
        """The tables the session lacks, read from the graph service, and ``files`` (the zones, replacing their last copy)."""
        held, sent = session_files(sid), {}
        for t in tables:
            if t["file"] not in held:
                url = f"{settings.ads_graph_url}/api/geo/{quote(t['dataset_id'], safe='')}/rows"
                rows = _checked(http.get(url, params={"resolution": t["cells_at"]}), "sig_failed", f"SIG rows of {t['filename']}")
                sent[t["file"]] = rows.content
        for name, data in (sent | files).items():
            _checked(http.post(f"{settings.ads_url}/sessions/{sid}/datasets", files={"file": (name, data)}), "ads_failed", f"ADS upload {name}")

    def run_ads(sid: str, query: str) -> str:
        """The task run as the session's next job; the session stays open for the conversation's next question. A status
        read that fails is read again, a job is given up after STATUS_RETRIES failures in a row or ADS_JOB_TIMEOUT_S."""
        ads = settings.ads_url
        started = http.post(f"{ads}/sessions/{sid}/jobs", json={"user_query": query})
        if started.status_code == 409:
            raise StageError("ads_busy", "the conversation's previous question still runs in ADS")
        _checked(started, "ads_failed", "ADS job")
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
                with contextlib.suppress(httpx.HTTPError):  # the session takes the next question
                    http.post(f"{ads}/sessions/{sid}/jobs/cancel")
                raise StageError("ads_timeout", f"the ADS job still ran after {settings.ads_job_timeout_s:g} s")
            time.sleep(settings.ads_poll_s)
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
        """The job's stages; ``job`` is this worker's alone, the console reads what each stage's end publishes."""
        started = time.monotonic()

        def enter(stage: str) -> None:
            nonlocal started
            job["timings"][job["stage"]] = round(time.monotonic() - started, 1)
            started, job["stage"] = time.monotonic(), stage
            publish(job)

        try:
            zones = {z.id: z.polygon for z in body.zones}
            findings = zone_findings(body, zones)
            tables, placed = place_tables(body)
            sources = finding_files(findings)
            job["insights"] = findings
            job["ads_query"] = "".join(p["text"] for p in build_query_parts(body.question, tables, body.focus, list(zones), findings))
            enter("ads")
            if not job["session_id"]:
                job["session_id"] = _checked(http.post(f"{settings.ads_url}/sessions"), "ads_failed", "ADS session").json()["session_id"]
                publish(job)  # the console keeps the session for the conversation's next question
            upload(job["session_id"], tables, {ZONES_FILE: placed, ZONES_MD: build_zones_md(zones, tables, sources).encode("utf-8")})
            job["ads_report"] = run_ads(job["session_id"], job["ads_query"])
            enter("joint")
            try:
                job["joint_report"] = write_report(body.question, job["ads_report"], build_domain_md(zones, body.focus, tables, sources))
            except Exception as exc:  # ADS's report stands without the final one
                log.exception("ADS job %s: no final report", job["job_id"])
                job.update(joint_report="", joint_error=f"{type(exc).__name__}: {exc}")
            enter("done")
        except Exception as exc:  # recorded on the job: the console shows the stage and the reason
            log.exception("ADS job %s failed", job["job_id"])
            error = {"code": error_code(exc, job["stage"]), "detail": f"{type(exc).__name__}: {exc}"}
            job.update(failed_stage=job["stage"], stage="error", error=error)
            publish(job)

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
        tables = name_tables([t.model_dump() for t in body.tables])
        return {"parts": build_query_parts(body.question, tables, body.focus, [z.id for z in body.zones], None)}

    @app.post("/api/ads/jobs")
    def start(body: AskIn) -> JSONResponse:
        if not _HEADER.sub("", body.question).strip():
            raise HTTPException(400, "empty question")
        job = {"job_id": uuid.uuid4().hex[:12], "stage": "search", "question": body.question, "session_id": body.session_id, "timings": {}}
        with jobs_lock:  # a running job is never dropped
            ended = [k for k, j in jobs.items() if j["stage"] in ("done", "error")]
            for old in ended[: max(0, len(ended) - JOBS_KEPT + 1)]:
                del jobs[old]
        publish(job)
        threading.Thread(target=run, args=(job, body), name=f"ads-job-{job['job_id']}", daemon=True).start()
        return JSONResponse({"job_id": job["job_id"]}, status_code=202)

    @app.get("/api/ads/jobs/{job_id}")
    def job_state(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise HTTPException(404, f"no ADS job {job_id}")
        return jobs[job_id]

    @app.post("/api/ads/sessions/{session_id}/close")
    def close(session_id: str = PathParam(pattern=SESSION_ID)) -> dict[str, Any]:
        """The conversation's end: the tables it sent deleted from ADS (its .md files and reports ADS keeps: its API deletes
        no .md), its session closed. Best effort: what ADS refuses is logged, the console has moved on."""
        ads, deleted = settings.ads_url, 0
        try:
            for f in _checked(http.get(f"{ads}/sessions/{session_id}/sources"), "ads_failed", "ADS session files").json()["files"]:
                if f.get("source_kind") == "uploaded_input" and f["name"].endswith(".csv"):
                    deleted += http.post(f"{ads}/sessions/{session_id}/sources/delete", json={"path": f["path"]}).status_code == 200
            http.post(f"{ads}/sessions/{session_id}/close")
        except (httpx.HTTPError, StageError) as exc:
            log.warning("ADS session %s not closed: %s", session_id, exc)
            return {"session_id": session_id, "deleted": deleted, "closed": False}
        return {"session_id": session_id, "deleted": deleted, "closed": True}

    return app
