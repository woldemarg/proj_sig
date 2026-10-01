"""LLM interface (SDD 12): local Gemma 4 over an OpenAI-compatible endpoint.

Works with any OpenAI-compatible endpoint: OpenRouter (``https://openrouter.ai/api/v1``,
``google/gemma-4-26b-a4b-it`` with ``LLM_PROVIDER_ORDER`` pinning, as in
spectr/agentic-data-science), Ollama (``http://localhost:11434/v1``, ``gemma4``) or
LM Studio (``http://localhost:1234/v1``, ``google/gemma-4-26b-a4b``).
The LLM only *verbalises* evidence; it never retrieves or computes statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ltir.config import Config

HEALTH_TTL_S = 15.0  # a refused localhost connect costs ~4 s on Windows

SYSTEM_PROMPT = """You are a careful data analyst answering questions about a tabular dataset.
You receive EVIDENCE: statistically validated subgroup findings retrieved from a knowledge graph.
Shifts are robust standard deviations ("sd": the median difference scaled by the MAD).
Rules:
1. Use ONLY the evidence. Do not invent numbers, subgroups, metrics or datasets.
2. Cite every factual statement with its key, e.g. [P1] or [P2][P4].
3. Structure the answer in two labelled parts:
   "Observations:" - what the verified statistics show (medians, shifts in sd, support).
   "Interpretation (hypotheses):" - possible explanations, explicitly marked as hypotheses.
4. These are observational subgroup statistics. Do not claim causation; say "is associated with".
5. If items were reached through a latent anchor and are scope-disjoint from the seeds (no shared
   condition), point out that the same phenomenon recurs in a different part of the data.
6. If the evidence does not answer the question, say so plainly.
Be concise (at most ~250 words)."""


@dataclass
class LLMResponse:
    text: str
    model: str
    ok: bool
    latency_s: float
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, user: str) -> LLMResponse: ...

    def health(self, *, fresh: bool = False) -> dict[str, Any]: ...


class OpenAICompatibleLLM:
    def __init__(self, config: Config) -> None:
        self.base_url = config.llm_base_url.rstrip("/")
        self.model = config.llm_model
        self.api_key = config.llm_api_key
        self.timeout = config.llm_timeout_s
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.reasoning_effort = config.llm_reasoning_effort
        self.provider_order = [p.strip() for p in config.llm_provider_order.split(",") if p.strip()]
        self.app_title = config.llm_app_title
        self._health: tuple[float, dict[str, Any]] | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"X-Title": self.app_title} if self.app_title else {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self, *, fresh: bool = False) -> dict[str, Any]:
        """GET /models, cached for ``HEALTH_TTL_S``."""
        if self._health and not fresh and time.monotonic() - self._health[0] < HEALTH_TTL_S:
            return self._health[1]
        try:
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            ids = [m.get("id") for m in resp.json().get("data", [])]
            available = any(self.model == i or str(i).startswith(self.model) for i in ids)
            result = {"reachable": True, "model": self.model, "model_available": available, "models": ids[:20], "base_url": self.base_url}
        except Exception as exc:
            result = {"reachable": False, "model": self.model, "model_available": False, "base_url": self.base_url, "error": str(exc)}
        self._health = (time.monotonic(), result)
        return result

    def generate(self, system: str, user: str) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.provider_order:  # OpenRouter routing, same contract as spectr's GemmaChatModel
            payload["provider"] = {"order": self.provider_order, "allow_fallbacks": False}
        start = time.perf_counter()
        status = self.health()
        if not status["reachable"]:  # fail fast; the evidence-only fallback answers instead
            return LLMResponse("", self.model, False, time.perf_counter() - start, error=f"endpoint unreachable: {status.get('error')}")
        try:
            timeout = httpx.Timeout(self.timeout, connect=5.0)
            resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty completion")
            return LLMResponse(text, data.get("model", self.model), True, time.perf_counter() - start, usage=data.get("usage", {}))
        except Exception as exc:
            return LLMResponse("", self.model, False, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
