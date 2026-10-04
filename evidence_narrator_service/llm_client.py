"""The narrator's chat client (docs/07_question_answering.md §7.5).

It talks to one endpoint, ``LLM_BASE_URL``: the LLM model broker (``llm_model_broker/``), which owns the upstream
provider, the model and its key. It sends no model name (the broker sets its own) and needs only
``POST /chat/completions``; ``GET /models`` serves the status display. It knows nothing of providers or of the answer
rules (``evidence_narrator_service.narration``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from evidence_narrator_service.settings import NarratorSettings

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

    def health(self) -> dict[str, Any]: ...


class OpenAICompatibleLLM:
    def __init__(self, settings: NarratorSettings) -> None:
        self.settings = settings
        self.base_url = settings.llm_base_url.rstrip("/")
        self._health: tuple[float, dict[str, Any]] | None = None
        self._down_until = 0.0  # after a refused connection, answers fall back at once until then

    def health(self) -> dict[str, Any]:
        """Status for the UI: reachable and the model the broker serves; cached for ``HEALTH_TTL_S``."""
        if self._health and time.monotonic() - self._health[0] < HEALTH_TTL_S:
            return self._health[1]
        result: dict[str, Any] = {"reachable": False, "model": None, "model_available": False, "base_url": self.base_url}
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=2.0)
            resp.raise_for_status()
            listed = [str(m.get("id")) for m in resp.json().get("data", [])]
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        else:
            result.update(reachable=True, model=listed[0] if listed else None, model_available=bool(listed))
        self._health = (time.monotonic(), result)
        return result

    def generate(self, system: str, user: str) -> LLMResponse:
        start = time.perf_counter()
        if time.monotonic() < self._down_until:  # fail fast; the evidence-only fallback answers instead
            return LLMResponse("", "", False, 0.0, error="endpoint unreachable (refused moments ago)")
        payload = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": False,
        }
        try:
            timeout = httpx.Timeout(self.settings.llm_timeout_s, connect=5.0)
            resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty completion")
            return LLMResponse(text, data.get("model", ""), True, time.perf_counter() - start, usage=data.get("usage", {}))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            self._down_until = time.monotonic() + HEALTH_TTL_S
            return LLMResponse("", "", False, time.perf_counter() - start, error=f"endpoint unreachable: {exc}")
        except Exception as exc:
            return LLMResponse("", "", False, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
