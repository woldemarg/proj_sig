"""The OpenAI-compatible LLM client and the citation check (docs/07_question_answering.md §7.5).

The shared stub endpoint (``conftest.llm_stub``) stands in for the LLM so the real HTTP client code path is exercised.
"""

from __future__ import annotations

import logging
import time

from conftest import free_port, serve
from fastapi import FastAPI, Request

from ltir.answering import SYSTEM_PROMPT, check_citations
from ltir.config import load_config
from ltir.llm_client import OpenAICompatibleLLM


def test_openai_compatible_roundtrip(llm_stub):
    """A plain OpenAI request: provider routing is the gateway's business, not the client's."""
    base, seen = llm_stub
    llm = OpenAICompatibleLLM(load_config(llm_base_url=base, llm_model="gemma4"))
    health = llm.health(fresh=True)
    assert health["reachable"] and health["model_available"]  # an Ollama tag answers to its name
    resp = llm.generate(SYSTEM_PROMPT, "EVIDENCE ... [P1] ...")
    assert resp.ok and resp.text.endswith("[P1].") and resp.usage["completion_tokens"] == 5
    body = seen[-1]["body"]
    assert set(body) == {"model", "messages", "temperature", "max_tokens", "stream"} and body["model"] == "gemma4" and body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"] and "ONLY the evidence" in body["messages"][0]["content"]
    assert check_citations(resp.text, {"P1": "P-abc"})["grounded"]
    assert not OpenAICompatibleLLM(load_config(llm_base_url=base, llm_model="gemma")).health(fresh=True)["model_available"]


def test_unset_model_is_the_one_the_endpoint_serves(llm_stub):
    base, seen = llm_stub
    llm = OpenAICompatibleLLM(load_config(llm_base_url=base, llm_model=""))
    assert llm.health(fresh=True)["model"] == "gemma4:latest" and llm.generate("s", "u").ok
    assert seen[-1]["body"]["model"] == "gemma4:latest"
    resp = llm.generate("s", "fail")
    assert not resp.ok and "HTTP 500" in resp.error and "boom" in resp.error  # the upstream's reason, not only a status


def test_an_endpoint_without_a_model_list():
    """Only POST /chat/completions is needed when LLM_MODEL is set; with it empty, one listed model is required."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        return {"model": body["model"], "choices": [{"message": {"content": "ok [P1]"}}]}

    with serve(app) as url:
        named = OpenAICompatibleLLM(load_config(llm_base_url=f"{url}/v1", llm_model="default"))
        assert named.generate("s", "u").ok and named.health(fresh=True)["reachable"]
        unnamed = OpenAICompatibleLLM(load_config(llm_base_url=f"{url}/v1", llm_model=""))
        assert not unnamed.generate("s", "u").ok


def test_an_empty_model_against_a_multi_model_endpoint_is_refused():
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": "a"}, {"id": "b"}]}

    with serve(app) as url:
        llm = OpenAICompatibleLLM(load_config(llm_base_url=f"{url}/v1", llm_model=""))
        resp = llm.generate("s", "u")
        health = llm.health(fresh=True)
    assert not resp.ok and "set LLM_MODEL" in resp.error and health["reachable"] and not health["model_available"]


def test_unreachable_endpoint_fails_fast():
    llm = OpenAICompatibleLLM(load_config(llm_base_url=f"http://127.0.0.1:{free_port()}/v1", llm_model="m"))
    resp = llm.generate("s", "u")
    assert not resp.ok and "unreachable" in resp.error
    t = time.perf_counter()
    llm.generate("s", "u")  # the refused connection is remembered -> no second attempt
    assert time.perf_counter() - t < 0.5


def test_retired_settings_are_named(monkeypatch, caplog):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "dekallm/bf16")
    with caplog.at_level(logging.WARNING, logger="ltir.config"):
        load_config()
    assert "LLM_PROVIDER_ORDER is set but not read" in caplog.text and "GEMMA_PROVIDER" in caplog.text


def test_citation_check_flags_unknown_keys():
    res = check_citations("see [P1] and [P9]", {"P1": "P-a", "P2": "P-b"})
    assert res["unknown"] == ["P9"] and res["uncited"] == ["P2"] and not res["grounded"]


def test_grouped_citations_are_parsed():
    res = check_citations("margins fall [P1, P3] and recur elsewhere [P2; P4]", {f"P{i}": f"P-{i}" for i in range(1, 5)})
    assert [c["key"] for c in res["cited"]] == ["P1", "P2", "P3", "P4"] and res["grounded"]
