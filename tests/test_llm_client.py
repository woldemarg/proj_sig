"""LLM interface over the OpenAI-compatible protocol used by Ollama / LM Studio (docs/07_question_answering.md §7.5).

A stub server stands in for Gemma 4 so the real HTTP client code path is exercised.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request

from ltir.config import load_config
from ltir.llm import SYSTEM_PROMPT, OpenAICompatibleLLM
from ltir.qa import check_citations


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def stub():
    app = FastAPI()
    seen: list[dict] = []

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "gemma4:latest"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        seen.append(body)
        return {
            "model": body["model"],
            "choices": [{"message": {"role": "assistant", "content": "Observations: margin is lower [P1]."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1", seen
    server.should_exit = True
    thread.join(timeout=5)


def test_openai_compatible_roundtrip(stub):
    base, seen = stub
    llm = OpenAICompatibleLLM(
        load_config(llm_base_url=base, llm_model="gemma4", llm_reasoning_effort="none", llm_provider_order="dekallm/bf16, parasail/bf16")
    )
    health = llm.health(fresh=True)
    assert health["reachable"] and health["model_available"]
    resp = llm.generate(SYSTEM_PROMPT, "EVIDENCE ... [P1] ...")
    assert resp.ok and resp.text.endswith("[P1].") and resp.usage["completion_tokens"] == 5
    body = seen[-1]
    assert body["model"] == "gemma4" and body["stream"] is False and body["reasoning_effort"] == "none"
    assert body["provider"] == {"order": ["dekallm/bf16", "parasail/bf16"], "allow_fallbacks": False}
    assert [m["role"] for m in body["messages"]] == ["system", "user"] and "ONLY the evidence" in body["messages"][0]["content"]
    assert check_citations(resp.text, {"P1": "P-abc"})["grounded"]


def test_unreachable_endpoint_fails_fast():
    llm = OpenAICompatibleLLM(load_config(llm_base_url=f"http://127.0.0.1:{_free_port()}/v1"))
    resp = llm.generate("s", "u")
    assert not resp.ok and "unreachable" in resp.error
    t = time.perf_counter()
    llm.generate("s", "u")  # cached health -> no second connection attempt
    assert time.perf_counter() - t < 0.5


def test_citation_check_flags_unknown_keys():
    res = check_citations("see [P1] and [P9]", {"P1": "P-a", "P2": "P-b"})
    assert res["unknown"] == ["P9"] and res["uncited"] == ["P2"] and not res["grounded"]


def test_grouped_citations_are_parsed():
    res = check_citations("margins fall [P1, P3] and recur elsewhere [P2; P4]", {f"P{i}": f"P-{i}" for i in range(1, 5)})
    assert [c["key"] for c in res["cited"]] == ["P1", "P2", "P3", "P4"] and res["grounded"]
