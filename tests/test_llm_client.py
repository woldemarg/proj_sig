"""The OpenAI-compatible LLM client and the citation check (docs/07_question_answering.md §7.5).

The shared stub endpoint (``conftest.llm_stub``) stands in for the LLM so the real HTTP client code path is exercised.
"""

from __future__ import annotations

import time

from conftest import free_port

from evidence_narrator_service.llm_client import OpenAICompatibleLLM
from evidence_narrator_service.narration import SYSTEM_PROMPT, check_citations
from evidence_narrator_service.settings import NarratorSettings


def test_openai_compatible_roundtrip(llm_stub):
    """A plain OpenAI request without a model name: the broker sets the model, the provider and the key."""
    base, seen = llm_stub
    llm = OpenAICompatibleLLM(NarratorSettings(llm_base_url=base))
    health = llm.health()
    assert health["reachable"] and health["model_available"] and health["model"] == "gemma4:latest"
    resp = llm.generate(SYSTEM_PROMPT, "EVIDENCE ... [P1] ...")
    assert resp.ok and resp.text.endswith("[P1].") and resp.model == "gemma4:latest" and resp.usage["completion_tokens"] == 5
    body = seen[-1]["body"]
    assert set(body) == {"messages", "temperature", "max_tokens", "stream"} and body["stream"] is False and seen[-1]["auth"] is None
    assert [m["role"] for m in body["messages"]] == ["system", "user"] and "ONLY the evidence" in body["messages"][0]["content"]
    assert check_citations(resp.text, {"P1": "P-abc"})["grounded"]
    resp = llm.generate("s", "fail")
    assert not resp.ok and "HTTP 500" in resp.error and "boom" in resp.error  # the upstream's reason, not only a status


def test_unreachable_endpoint_fails_fast():
    llm = OpenAICompatibleLLM(NarratorSettings(llm_base_url=f"http://127.0.0.1:{free_port()}/v1"))
    resp = llm.generate("s", "u")
    assert not resp.ok and "unreachable" in resp.error
    t = time.perf_counter()
    llm.generate("s", "u")  # the refused connection is remembered -> no second attempt
    assert time.perf_counter() - t < 0.5


def test_citation_check_flags_unknown_keys():
    res = check_citations("see [P1] and [P9]", {"P1": "P-a", "P2": "P-b"})
    assert res["unknown"] == ["P9"] and res["uncited"] == ["P2"] and not res["grounded"]


def test_grouped_citations_are_parsed():
    res = check_citations("margins fall [P1, P3] and recur elsewhere [P2; P4]", {f"P{i}": f"P-{i}" for i in range(1, 5)})
    assert [c["key"] for c in res["cited"]] == ["P1", "P2", "P3", "P4"] and res["grounded"]
