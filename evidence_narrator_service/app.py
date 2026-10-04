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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from evidence_narrator_service.llm_client import ChatModel, OpenAICompatibleLLM
from evidence_narrator_service.narration import answer
from evidence_narrator_service.settings import NarratorSettings
from insight_contracts import EvidencePayload

log = logging.getLogger(__name__)
queries = logging.getLogger("evidence_narrator_service.queries")  # one JSON line per answered question


class QueryIn(BaseModel):
    question: str
    use_llm: bool = True


def create_app(settings: NarratorSettings | None = None, llm: ChatModel | None = None) -> FastAPI:
    settings = settings or NarratorSettings.from_env()
    llm = llm or OpenAICompatibleLLM(settings)  # building the client does no I/O
    graph = httpx.Client(base_url=settings.insight_graph_url, timeout=httpx.Timeout(settings.insight_graph_timeout_s, connect=5.0))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        graph.close()

    app = FastAPI(title="SIG evidence narrator", lifespan=lifespan)

    @app.post("/api/chat/query", response_model=None)
    def query(body: QueryIn) -> dict[str, Any] | JSONResponse:
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        try:
            resp = graph.post("/api/search", json={"question": body.question})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"graph service unreachable: {type(exc).__name__}: {exc}") from exc
        if resp.status_code == 409:  # a workspace that cannot answer (mismatch, degraded): its reason, as sent
            return JSONResponse(resp.json(), status_code=409)
        if resp.status_code != 200:
            raise HTTPException(502, f"graph service HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            payload = EvidencePayload.from_dict(resp.json())
        except (KeyError, ValueError) as exc:  # another version of the graph service
            raise HTTPException(502, f"graph service broke the evidence contract: {type(exc).__name__}: {exc}") from exc
        qa = answer(payload, llm if body.use_llm else None)
        if not payload.graph_empty:
            queries.info(
                json.dumps(
                    {
                        "at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "question": qa.question,
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
