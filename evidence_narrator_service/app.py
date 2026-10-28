"""The narrator's HTTP host: ``POST /api/chat/query`` (the evidence from the graph service, verbalised and
citation-checked) and ``GET /api/chat/health`` (always 200; reports the LLM and the graph service)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from evidence_narrator_service.llm_client import ChatModel, OpenAICompatibleLLM
from evidence_narrator_service.narration import answer
from evidence_narrator_service.settings import NarratorSettings
from insight_contracts import EvidencePayload

log = logging.getLogger(__name__)
queries = logging.getLogger("evidence_narrator_service.queries")  # one JSON line per answered question


class QueryIn(BaseModel):
    question: str = ""
    use_llm: bool = True
    cells: list[str] | None = None  # a map zone, passed to POST /api/search as given
    geometry: dict[str, Any] | None = None
    resolution: int | None = None
    as_of: str | None = None  # the reference date, passed to POST /api/search as given


def create_app(settings: NarratorSettings | None = None, llm: ChatModel | None = None) -> FastAPI:
    settings = settings or NarratorSettings.from_env()
    built = llm is None  # the app closes only the client it built
    llm = llm or OpenAICompatibleLLM(settings)  # building the client does no I/O
    graph = httpx.Client(base_url=settings.insight_graph_url, timeout=httpx.Timeout(settings.insight_graph_timeout_s, connect=5.0))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        graph.close()
        if built:
            llm.close()

    app = FastAPI(title="SIG evidence narrator", lifespan=lifespan)

    @app.post("/api/chat/query", response_model=None)
    async def query(body: QueryIn, request: Request) -> dict[str, Any] | JSONResponse | Response:
        if not body.question.strip() and body.cells is None and body.geometry is None:
            raise HTTPException(400, "empty question")
        try:
            resp = await run_in_threadpool(graph.post, "/api/search", json=body.model_dump(exclude={"use_llm"}, exclude_none=True))
        except httpx.ReadTimeout as exc:  # the console reads a 504 as no answer in time
            raise HTTPException(504, f"graph service did not answer within {settings.insight_graph_timeout_s:g} s") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"graph service unreachable: {type(exc).__name__}: {exc}") from exc
        if resp.status_code in (400, 409, 422):  # a zone it refused, a workspace that cannot answer: the reason, as sent
            return JSONResponse(resp.json(), status_code=resp.status_code)
        if resp.status_code != 200:
            raise HTTPException(502, f"graph service HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            payload = EvidencePayload.from_dict(resp.json())
        except (KeyError, ValueError) as exc:  # another version of the graph service
            raise HTTPException(502, f"graph service broke the evidence contract: {type(exc).__name__}: {exc}") from exc
        if body.use_llm and await request.is_disconnected():  # the console cancelled or left: no paid call, no log line
            return Response(status_code=499)
        qa = await run_in_threadpool(answer, payload, llm if body.use_llm else None)
        if not payload.graph_empty:
            queries.info(
                json.dumps(
                    {
                        "at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "question": qa.question,
                        **body.model_dump(include={"cells", "geometry", "resolution", "as_of"}, exclude_none=True),  # a zone and a date, to replay it
                        "mode": qa.answer_mode,
                        "metrics": qa.metrics,
                        "evidence": [c["pattern_id"] for c in payload.citations.values()],
                        "citations": qa.citations,
                    },
                    ensure_ascii=False,
                )
            )
        return asdict(qa)

    # ponytail: sync probes share the request thread pool with /query; make them async if many users ask at once
    @app.get("/api/chat/health")
    def health() -> dict[str, Any]:
        try:
            reachable = graph.get("/api/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            reachable = False
        return {"llm": llm.health(), "insight_graph": {"reachable": reachable, "url": settings.insight_graph_url}}

    return app
