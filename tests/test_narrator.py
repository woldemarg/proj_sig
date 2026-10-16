"""The evidence narrator (docs/07_question_answering.md §7.5): the graph service's evidence in, a checked answer out.

Against a stub graph service that serves a recorded ``EvidencePayload`` and the OpenAI-compatible LLM stub; the last
test runs the real graph service and the narrator over local HTTP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from conftest import serve
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from evidence_narrator_service.app import create_app
from evidence_narrator_service.settings import NarratorSettings
from graph_query_engine import empty_payload

QUESTION = "Why is margin lower for phones in the US?"
DOCS = Path(__file__).resolve().parents[1] / "docs" / "07_question_answering.md"


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
            system = seen[-1]["body"]["messages"][0]["content"]
            assert "11. TRUST:" in system and system in DOCS.read_text(encoding="utf-8")  # docs/07 §7.5 quotes the rules verbatim
            assert qa["view"] == recorded["view"] and qa["prompt"] == recorded["evidence_prompt"]  # the view, untouched
            assert "Sources: [P1]" in qa["provenance_footer"]
            logged = json.loads(caplog.records[-1].getMessage())
            assert (
                logged["question"] == QUESTION
                and logged["mode"] == "llm"
                and logged["evidence"] == [c["pattern_id"] for c in recorded["citations"].values()]
            )

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


def test_health_answers_without_its_dependencies():
    settings = NarratorSettings(insight_graph_url="http://127.0.0.1:9", llm_base_url="http://127.0.0.1:9/v1")
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/chat/health")
        assert health.status_code == 200 and not health.json()["insight_graph"]["reachable"] and not health.json()["llm"]["reachable"]
        assert client.post("/api/chat/query", json={"question": QUESTION}).status_code == 502
