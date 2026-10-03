"""The LLM gateway (docs/12_architecture.md §12.4): one OpenAI-compatible endpoint in front of the upstream model.

The shared stub endpoint (``conftest.llm_stub``) is the upstream and records what the gateway forwards; the last
test runs the whole chain backend client -> gateway -> upstream over real local HTTP.
"""

from __future__ import annotations

from conftest import serve
from fastapi.testclient import TestClient

from llm_gateway.server import Settings, create_app
from ltir.config import load_config
from ltir.llm_client import OpenAICompatibleLLM

MESSAGES = [{"role": "user", "content": "hi"}]


def settings(base_url: str) -> Settings:
    return Settings(base_url=base_url, model="google/gemma-4-26b-a4b-it", api_key="sk-test", provider_order=("dekallm/bf16", "parasail/bf16"))


def test_forwards_to_the_configured_model_with_pinned_providers(llm_stub):
    base, seen = llm_stub
    with TestClient(create_app(settings(base))) as client:  # runs the lifespan, which owns the upstream client
        assert client.get("/health").json() == {"status": "ok", "model": "google/gemma-4-26b-a4b-it", "upstream": "127.0.0.1"}
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["google/gemma-4-26b-a4b-it"]
        resp = client.post("/v1/chat/completions", json={"model": "anything", "messages": MESSAGES, "max_tokens": 50, "top_p": 0.9})
    assert resp.status_code == 200 and resp.json()["choices"][0]["message"]["content"].endswith("[P1].")
    sent = seen[-1]
    assert sent["auth"] == "Bearer sk-test" and sent["body"]["model"] == "google/gemma-4-26b-a4b-it"
    assert sent["body"]["max_tokens"] == 50 and sent["body"]["top_p"] == 0.9 and "temperature" not in sent["body"]  # sent as given
    assert sent["body"]["provider"] == {"order": ["dekallm/bf16", "parasail/bf16"], "allow_fallbacks": False}


def test_upstream_failures_and_bad_requests_are_http_errors(llm_stub):
    base, _ = llm_stub
    with TestClient(create_app(settings(base))) as client:
        busy = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "busy"}]})
        assert busy.status_code == 429 and busy.headers["retry-after"] == "7" and "rate limited" in busy.json()["error"]["message"]
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "fail"}]}).status_code == 502
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
        assert client.post("/v1/chat/completions", json={"messages": MESSAGES, "stream": True}).status_code == 400
        malformed = client.post("/v1/chat/completions", json={"model": "x"})  # no messages: OpenAI-shaped 400, not a 422
        assert malformed.status_code == 400 and malformed.json()["error"]["type"] == "invalid_request_error"
        assert "messages" in malformed.json()["error"]["message"]
    with TestClient(create_app(Settings(base_url="", model=""))) as bare:
        assert bare.get("/health").json()["status"] == "unconfigured" and bare.get("/v1/models").json()["data"] == []
        assert bare.post("/v1/chat/completions", json={"messages": MESSAGES}).status_code == 503
    assert Settings.from_env({"GEMMA_PROVIDER": " a/bf16, b ", "GEMMA_BASE_URL": "https://x/v1/"}).provider_order == ("a/bf16", "b")


def test_backend_client_reaches_the_model_through_the_gateway(llm_stub):
    base, seen = llm_stub
    with serve(create_app(settings(base))) as gateway:
        llm = OpenAICompatibleLLM(load_config(llm_base_url=f"{gateway}/v1", llm_model="", llm_api_key=""))
        resp = llm.generate("system", "user")
    assert resp.ok and resp.text.endswith("[P1].") and resp.model == "google/gemma-4-26b-a4b-it"
    assert seen[-1]["auth"] == "Bearer sk-test"  # the provider key lives in the gateway, never in the backend
