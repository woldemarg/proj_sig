"""OpenAI-compatible chat client (docs/07_question_answering.md §7.5).

It talks to one endpoint, ``LLM_BASE_URL``: the LLM gateway (``llm_gateway/``, which owns the
provider, the model and its key) or any other OpenAI-compatible server (OpenRouter, Ollama,
LM Studio). It needs only ``POST /chat/completions``; ``GET /models`` serves the status display
and, when ``LLM_MODEL`` is empty, names the one model the endpoint serves. It knows nothing of
providers or of the answer rules (``ltir.answering``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ltir.config import Config

HEALTH_TTL_S = 15.0  # a refused connection is remembered this long (a refused localhost connect costs ~4 s on Windows)


@dataclass
class LLMResponse:
    text: str
    model: str
    ok: bool
    latency_s: float
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class ChatModel(Protocol):
    """What the answer and the status display need from an LLM (tests use a fake)."""

    def generate(self, system: str, user: str) -> LLMResponse: ...

    def health(self, *, fresh: bool = False) -> dict[str, Any]: ...


class OpenAICompatibleLLM:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.base_url = config.llm_base_url.rstrip("/")
        self.model = config.llm_model  # "" = the one model the endpoint lists (the gateway serves exactly one)
        self._health: tuple[float, dict[str, Any]] | None = None
        self._down_until = 0.0  # after a refused connection, answers fall back at once until then

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.llm_api_key}"} if self.config.llm_api_key else {}

    def _listed(self) -> list[str]:
        resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=10.0)
        resp.raise_for_status()
        return [str(m.get("id")) for m in resp.json().get("data", [])]

    def _resolve(self, listed: list[str]) -> str:
        if self.model:
            return self.model
        if len(listed) != 1:
            raise ValueError(f"LLM_MODEL is empty and the endpoint lists {len(listed)} models: set LLM_MODEL")
        return listed[0]

    def health(self, *, fresh: bool = False) -> dict[str, Any]:
        """Status for the UI and ``llm-check``: reachable, the model in use, whether the endpoint lists it; cached for ``HEALTH_TTL_S``."""
        if self._health and not fresh and time.monotonic() - self._health[0] < HEALTH_TTL_S:
            return self._health[1]
        result: dict[str, Any] = {"reachable": False, "model": self.model, "model_available": False, "base_url": self.base_url}
        try:
            listed = self._listed()
        except httpx.HTTPStatusError as exc:  # it answers, but lists no models (a gateway with only /chat/completions)
            result.update(reachable=True, model_available=bool(self.model), error=f"GET /models: HTTP {exc.response.status_code}")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        else:
            result.update(reachable=True, models=listed[:20])
            try:
                model = self._resolve(listed)
                # an Ollama tag ("gemma4:latest") answers to its name ("gemma4")
                result.update(model=model, model_available=any(i == model or i.startswith(model + ":") for i in listed))
            except ValueError as exc:
                result["error"] = str(exc)
        self._health = (time.monotonic(), result)
        return result

    def generate(self, system: str, user: str) -> LLMResponse:
        start = time.perf_counter()
        if time.monotonic() < self._down_until:  # fail fast; the evidence-only fallback answers instead
            return LLMResponse("", self.model, False, 0.0, error="endpoint unreachable (refused moments ago)")
        model = self.model
        try:
            model = self._resolve([] if self.model else self._listed())
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens,
                "stream": False,
            }
            timeout = httpx.Timeout(self.config.llm_timeout_s, connect=5.0)
            resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers(), timeout=timeout)
            if resp.status_code >= 400:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty completion")
            return LLMResponse(text, data.get("model", model), True, time.perf_counter() - start, usage=data.get("usage", {}))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            self._down_until = time.monotonic() + HEALTH_TTL_S
            return LLMResponse("", model, False, time.perf_counter() - start, error=f"endpoint unreachable: {exc}")
        except Exception as exc:
            return LLMResponse("", model, False, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
