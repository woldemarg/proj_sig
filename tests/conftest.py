"""Shared fixtures (docs/10_verification.md §10.1).

* ``hashing`` backend: fast, offline, deterministic — used by most contract tests.
* ``model`` marker: tests that need the real local sentence-transformers model;
  skipped automatically when the model cache is missing.
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

os.environ["LTIR_NO_DOTENV"] = "1"  # never read sig/.env (API keys, NEO4J_ENABLED) in tests

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ltir.config import load_config
from ltir.evaluation.synthetic import generate_retail_dataset
from ltir.llm_client import LLMResponse
from ltir.models import Condition, Insight


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve(app) -> Iterator[str]:
    """Run an ASGI app on a free local port in a background thread; yields its base URL."""
    import uvicorn

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def llm_stub():
    """An OpenAI-compatible endpoint on a local port: ``GET /v1/models`` lists one model; ``POST /v1/chat/completions``
    records ``{"body", "auth"}`` and answers — a 500 for the message "fail", a 429 with Retry-After for "busy".
    Yields (base URL ending in /v1, the recorded requests)."""
    app = FastAPI()
    seen: list[dict] = []

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "gemma4:latest"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        seen.append({"body": body, "auth": req.headers.get("authorization")})
        last = body["messages"][-1]["content"]
        if last == "fail":
            return JSONResponse({"error": {"message": "boom"}}, status_code=500)
        if last == "busy":
            return JSONResponse({"error": {"message": "rate limited"}}, status_code=429, headers={"Retry-After": "7"})
        reply = {"role": "assistant", "content": "Observations: margin is lower [P1]."}
        return {"model": body["model"], "choices": [{"message": reply}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    with serve(app) as url:
        yield f"{url}/v1", seen


def _model_available(cfg) -> bool:
    from ltir.analysis.encoder import model_folder

    return (model_folder(cfg) / "modules.json").is_file()


def pytest_collection_modifyitems(config, items):
    cfg = load_config()
    if _model_available(cfg):
        return
    skip = pytest.mark.skip(reason=f"embedding model {cfg.embedding_model} not found in {cfg.model_dir}")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)


class FakeLLM:
    """Deterministic stand-in for Gemma 4: cites the first evidence keys it sees."""

    model = "fake-gemma"

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.prompts: list[str] = []

    def health(self, **_):
        return {"reachable": self.ok, "model": self.model, "model_available": self.ok}

    def generate(self, system: str, user: str) -> LLMResponse:
        self.prompts.append(user)
        if not self.ok:
            return LLMResponse("", self.model, False, 0.01, error="ConnectError: offline")
        keys = re.findall(r"^\[(P\d+)\]", user, flags=re.M)[:3]
        cites = "".join(f"[{k}]" for k in keys)
        return LLMResponse(
            f"Observations: the retrieved subgroups show the shift {cites}.\nInterpretation (hypotheses): associated, not causal.",
            self.model,
            True,
            0.01,
        )


@pytest.fixture(scope="session")
def demo_df():
    return generate_retail_dataset(5000, seed=7)


@pytest.fixture(scope="session")
def demo_csv(tmp_path_factory, demo_df):
    path = tmp_path_factory.mktemp("data") / "retail_synthetic.csv"
    demo_df.to_csv(path, index=False)
    return path


def toy_insight(scope, shifts, **fields) -> Insight:
    """A test Insight: ``scope`` [(column, value)] in order, ``shifts`` Shift records. Target, baseline, local and
    effect follow the first shift; the other statistics are neutral, and any field can be overridden."""
    first = shifts[0]
    values = {
        "id": "P-x",
        "dataset_id": "d",
        "batch_id": "b",
        "conditions": tuple(Condition(k, v) for k, v in scope),
        "expression": "e",
        "target": first.metric,
        "shifts": tuple(shifts),
        "support": 100,
        "support_fraction": 0.1,
        "baseline": first.global_median,
        "local": first.local_median,
        "effect_size": first.robust_z,
        "sd_score": 1.0,
        "sd_raw_score": 1.0,
        "emm_score": 0.0,
        "volume_utility": 0.2,
        "stability": 0.9,
        "p_value": 1e-9,
        "p_adjusted": 1e-6,
        "drivers": (),
        "row_hash": fields.get("id", "P-x"),
    }
    return Insight(**{**values, **fields})


def make_config(workspace: Path, **overrides):
    overrides.setdefault("neo4j_enabled", False)
    return load_config(workspace_dir=workspace, embedding_backend=overrides.pop("embedding_backend", "hashing"), **overrides)


@pytest.fixture(scope="session")
def hashed_engine(tmp_path_factory, demo_csv):
    """Engine with the synthetic demo ingested (hashing embedder) — shared, read-only use."""
    from ltir.engine import Engine

    engine = Engine(make_config(tmp_path_factory.mktemp("ws_hash")), llm=FakeLLM())
    record = engine.ingest_file(demo_csv)
    assert record["status"] == "READY", record.get("error")
    return engine


@pytest.fixture(scope="session")
def model_engine(tmp_path_factory, demo_csv):
    """Engine with the synthetic demo ingested using the real local embedding model."""
    from ltir.engine import Engine

    engine = Engine(make_config(tmp_path_factory.mktemp("ws_model"), embedding_backend="sentence-transformers"), llm=FakeLLM())
    record = engine.ingest_file(demo_csv)
    assert record["status"] == "READY", record.get("error")
    return engine
