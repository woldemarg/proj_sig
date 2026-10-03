"""The gateway service: ``GET /health``, ``GET /v1/models``, ``POST /v1/chat/completions`` (OpenAI-compatible).

It serves exactly one model: a request's ``model`` field is accepted and replaced by the configured
upstream model; every other field of the request is forwarded as sent (``stream: true`` is refused —
answers are not streamed). Settings (environment, or ``.env.gemma`` in the repository root when run
on the host with ``python -m llm_gateway``):

    GEMMA_BASE_URL               upstream base URL, e.g. https://openrouter.ai/api/v1 (required)
    GEMMA_MODEL_NAME             upstream model id, e.g. google/gemma-4-26b-a4b-it (required)
    GEMMA_CHAT_ENDPOINT          path appended to the base URL (/chat/completions)
    GEMMA_API_KEY                bearer token for the upstream ("" for a local server)
    GEMMA_PROVIDER               OpenRouter provider slugs tried in order, comma-separated; no fallbacks ("" = OpenRouter's routing)
    GEMMA_CONNECT_TIMEOUT_SEC    5
    GEMMA_READ_TIMEOUT_SEC       120
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

log = logging.getLogger("llm_gateway")


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    chat_endpoint: str = "/chat/completions"
    api_key: str = ""
    provider_order: tuple[str, ...] = ()
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 120.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        return cls(
            base_url=env.get("GEMMA_BASE_URL", "").strip().rstrip("/"),
            model=env.get("GEMMA_MODEL_NAME", "").strip(),
            chat_endpoint=env.get("GEMMA_CHAT_ENDPOINT", "").strip() or "/chat/completions",
            api_key=env.get("GEMMA_API_KEY", "").strip(),
            provider_order=tuple(s.strip() for s in env.get("GEMMA_PROVIDER", "").split(",") if s.strip()),
            connect_timeout_s=float(env.get("GEMMA_CONNECT_TIMEOUT_SEC") or 5),
            read_timeout_s=float(env.get("GEMMA_READ_TIMEOUT_SEC") or 120),
        )


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # temperature, max_tokens, top_p, stop, response_format, tools, … pass through

    messages: list[dict[str, Any]]
    stream: bool = False


def _error(status: int, message: str, kind: str, headers: dict[str, str] | None = None) -> JSONResponse:
    """An OpenAI-shaped error body."""
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status, headers=headers)


def create_app(settings: Settings) -> FastAPI:
    configured = bool(settings.base_url and settings.model)
    upstream_url = settings.base_url + settings.chat_endpoint

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with httpx.Client(timeout=httpx.Timeout(settings.read_timeout_s, connect=settings.connect_timeout_s)) as client:
            app.state.client = client  # one connection pool for the app's lifetime
            yield

    app = FastAPI(title="SIG LLM gateway", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        """A malformed body is a 400 in the OpenAI error shape (clients read ``error.message``), not FastAPI's 422."""
        where = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return _error(400, f"invalid request: {where}", "invalid_request_error")

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness; ``status`` says whether an upstream is configured."""
        return {"status": "ok" if configured else "unconfigured", "model": settings.model, "upstream": urlsplit(settings.base_url).hostname}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": settings.model, "object": "model", "owned_by": "sig-llm-gateway"}] if configured else []}

    @app.post("/v1/chat/completions", response_model=None)
    def chat(req: ChatRequest) -> dict[str, Any] | JSONResponse:
        if not configured:
            return _error(503, "gateway is not configured: set GEMMA_BASE_URL and GEMMA_MODEL_NAME", "gateway_unconfigured")
        if not req.messages:
            return _error(400, "messages must not be empty", "invalid_request_error")
        if req.stream:
            return _error(400, "streaming is not supported: send stream=false", "invalid_request_error")
        payload = {**req.model_dump(exclude={"stream"}), "model": settings.model, "stream": False}
        if settings.provider_order:  # OpenRouter: only the pinned providers, in this order
            payload["provider"] = {"order": list(settings.provider_order), "allow_fallbacks": False}
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        start = time.perf_counter()
        try:
            resp = app.state.client.post(upstream_url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            log.warning("upstream timeout after %.1f s", time.perf_counter() - start)
            return _error(504, f"upstream timeout: {exc}", "upstream_timeout")
        except httpx.HTTPError as exc:
            return _error(502, f"upstream unreachable: {type(exc).__name__}: {exc}", "upstream_error")
        if resp.status_code >= 400:  # a client error (bad request, key, credit, rate limit) keeps its status
            log.warning("upstream HTTP %s", resp.status_code)
            status = resp.status_code if resp.status_code < 500 else 502
            retry = {"Retry-After": resp.headers["retry-after"]} if "retry-after" in resp.headers else None
            return _error(status, f"upstream HTTP {resp.status_code}: {resp.text[:300]}", "upstream_error", retry)
        try:
            data = resp.json()
        except ValueError:
            return _error(502, f"upstream sent no JSON: {resp.text[:200]}", "upstream_error")
        usage = data.get("usage") or {}
        log.info(
            "chat model=%s messages=%d prompt_tokens=%s completion_tokens=%s latency_ms=%d",
            data.get("model", settings.model),
            len(req.messages),
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            round((time.perf_counter() - start) * 1000),
        )
        return data

    return app


ENV_FILE = Path(__file__).resolve().parents[1] / ".env.gemma"


def load_env_file(path: Path = ENV_FILE) -> None:
    """``KEY=value`` lines into the environment (existing variables win); a missing file is no error."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
