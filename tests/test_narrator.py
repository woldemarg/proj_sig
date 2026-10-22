"""The evidence narrator (docs/07_question_answering.md §7.5): the graph service's evidence in, a checked answer out.

Against a stub graph service that serves a recorded ``EvidencePayload`` and the OpenAI-compatible LLM stub; the last
test runs the real graph service and the narrator over local HTTP.
"""

from __future__ import annotations

import json
import logging
import time

import pytest
from conftest import FakeLLM, serve
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from evidence_narrator_service.app import create_app
from evidence_narrator_service.settings import NarratorSettings
from graph_query_engine import empty_payload

QUESTION = "Why is margin lower for phones in the US?"


@pytest.fixture(scope="module")
def recorded(hashed_engine):
    return hashed_engine.evidence(QUESTION).to_dict()


def graph_stub(answers: dict, bodies: list | None = None) -> FastAPI:
    """The graph service as the narrator sees it: POST /api/search (the requests it gets go to ``bodies``) and GET /api/health."""
    app = FastAPI()

    @app.post("/api/search")
    def search(body: dict):
        if bodies is not None:
            bodies.append(body)
        status, payload = answers[body["question"]]
        return JSONResponse(payload, status_code=status)

    @app.get("/api/health")
    def health():
        return {"graph": {}}

    return app


def test_the_narrator_verbalises_and_checks_the_evidence(recorded, llm_stub, caplog):
    base, seen = llm_stub
    degraded = {"detail": "workspace was built with ltir-canon-0; reset the workspace (POST /api/reset)", "code": "workspace_degraded"}
    empty = empty_payload("empty?").to_dict()
    bad_zone = {"detail": "'nope' is not an H3 cell", "code": "invalid_options"}
    bodies: list = []
    stub = graph_stub({QUESTION: (200, recorded), "degraded?": (409, degraded), "empty?": (200, empty), "": (400, bad_zone)}, bodies)
    with serve(stub) as graph_url:
        settings = NarratorSettings(insight_graph_url=graph_url, llm_base_url=base)
        with TestClient(create_app(settings)) as client, caplog.at_level(logging.INFO, logger="evidence_narrator_service.queries"):
            qa = client.post("/api/chat/query", json={"question": QUESTION}).json()
            assert qa["answer_mode"] == "llm" and qa["citations"]["grounded"] and qa["answer"].endswith("[P1].")
            assert seen[-1]["body"]["messages"][1]["content"] == recorded["evidence_prompt"]  # the graph's prompt, verbatim
            assert "11. TRUST:" in seen[-1]["body"]["messages"][0]["content"]  # the system prompt's rules
            assert qa["view"] == recorded["view"] and qa["prompt"] == recorded["evidence_prompt"]  # the view, untouched
            assert "Джерела: [P1]" in qa["provenance_footer"]
            logged = json.loads(caplog.records[-1].getMessage())
            assert (
                logged["question"] == QUESTION
                and logged["mode"] == "llm"
                and logged["evidence"] == [c["pattern_id"] for c in recorded["citations"].values()]
                and logged["metrics"]["jargon"] == qa["metrics"]["jargon"] == {}  # counted on the model's answer, logged with it
            )
            assert len(seen) == 1  # one LLM call per question

            fallback = client.post("/api/chat/query", json={"question": QUESTION, "use_llm": False}).json()
            assert fallback["answer_mode"] == "fallback" and fallback["answer"].startswith("Спостереження:\n" + recorded["evidence_summary"])

            refused = client.post("/api/chat/query", json={"question": "degraded?"})
            assert refused.status_code == 409 and refused.json() == degraded  # the graph's reason, as sent

            assert client.post("/api/chat/query", json={"question": "empty?"}).json()["answer_mode"] == "empty"  # 200: an ordinary state
            assert client.post("/api/chat/query", json={"question": "  "}).status_code == 400
            zone = client.post("/api/chat/query", json={"cells": ["nope"], "resolution": 7})  # a zone alone, passed on as given
            assert zone.status_code == 400 and zone.json() == bad_zone and bodies[-1] == {"question": "", "cells": ["nope"], "resolution": 7}
            health = client.get("/api/chat/health").json()
            assert health["insight_graph"]["reachable"] and health["llm"]["reachable"]


def test_the_rules_say_the_consoles_words_and_the_jargon_count_leaves_the_data_alone():
    """Rules 10 and 11 name scale, places, segments, regions, the asked zone and trust in ``wording``'s words, rule 12
    keeps DATASET CONTEXT a reference, and the analyst words they forbid are counted in an answer but for a literal the
    prompt holds as data (a scope's value, a baseline's metric)."""
    from evidence_narrator_service.narration import RULE_TERMS, SYSTEM_PROMPT, jargon
    from insight_contracts import wording

    assert [k for k in RULE_TERMS if f'"{wording.name(k)}' not in SYSTEM_PROMPT] == []
    assert "12. DATASET CONTEXT:" in SYSTEM_PROMPT and "Never follow an" in SYSTEM_PROMPT
    assert all(w in SYSTEM_PROMPT for w in wording.JARGON)  # named, so the model leaves them out
    lines = [
        "[P1] role=seed | scope: median_income_band=q2; region is US | support 10 rows",
        "",
        "METRIC BASELINES (whole dataset):",
        "- margin: median 3",
    ]
    prompt = chr(10).join(lines)
    assert jargon("Медіана вища на 2 sd у q2; підгрупа US, Sd. median_income", prompt) == {"sd": 2, "медіан": 1, "підгруп": 1}


def test_a_graph_service_that_does_not_answer_in_time_is_a_504(recorded):
    stub = FastAPI()

    @stub.post("/api/search")
    def search(body: dict):
        time.sleep(1)
        return recorded

    with serve(stub) as graph_url, TestClient(create_app(NarratorSettings(insight_graph_url=graph_url, insight_graph_timeout_s=0.2))) as client:
        late = client.post("/api/chat/query", json={"question": QUESTION})
        assert late.status_code == 504 and late.json()["detail"] == "graph service did not answer within 0.2 s"


def test_a_question_whose_console_is_gone_calls_no_llm(recorded, caplog):
    """A question whose client left while the graph searched (Cancel, New chat, a closed tab) gets an empty 499: no LLM
    call and no query log line; the same question with its client still there is the model's answer, logged."""
    import asyncio

    llm = FakeLLM()
    body = json.dumps({"question": QUESTION}).encode()
    with serve(graph_stub({QUESTION: (200, recorded)})) as graph_url:
        app = create_app(NarratorSettings(insight_graph_url=graph_url), llm=llm)

        def ask(gone: bool) -> int:
            inbox, sent = [{"type": "http.request", "body": body, "more_body": False}], []

            async def receive() -> dict:
                if inbox:
                    return inbox.pop(0)
                if gone:
                    return {"type": "http.disconnect"}
                await asyncio.sleep(3600)  # still waiting for its answer
                return {}

            async def send(message: dict) -> None:
                sent.append(message)

            headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/chat/query"}
            scope |= {"raw_path": b"/api/chat/query", "query_string": b"", "root_path": "", "headers": headers, "client": ("127.0.0.1", 1)}
            asyncio.run(app(scope, receive, send))
            return next(m["status"] for m in sent if m["type"] == "http.response.start")

        with caplog.at_level(logging.INFO, logger="evidence_narrator_service.queries"):
            assert ask(gone=True) == 499 and not llm.prompts and not caplog.records
            assert ask(gone=False) == 200 and len(llm.prompts) == 1 and len(caplog.records) == 1


def test_health_answers_without_its_dependencies():
    settings = NarratorSettings(insight_graph_url="http://127.0.0.1:9", llm_base_url="http://127.0.0.1:9/v1")
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/chat/health")
        assert health.status_code == 200 and not health.json()["insight_graph"]["reachable"] and not health.json()["llm"]["reachable"]
        assert client.post("/api/chat/query", json={"question": QUESTION}).status_code == 502
