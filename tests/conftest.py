"""Shared fixtures (docs/10_verification.md §10.1).

* ``hashing`` backend: fast, offline, deterministic — used by most contract tests.
* ``model`` marker: tests that need the real local sentence-transformers model;
  skipped automatically when the model cache is missing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ["LTIR_NO_DOTENV"] = "1"  # never read sig/.env (API keys, NEO4J_ENABLED) in tests

import pytest

from ltir.config import load_config
from ltir.llm import LLMResponse
from ltir.models import Condition, Insight
from ltir.synth import generate_retail_dataset


def _model_available(cfg) -> bool:
    from ltir.encoder import model_folder

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
    overrides.setdefault("sphere_export", False)
    return load_config(workspace_dir=workspace, embedding_backend=overrides.pop("embedding_backend", "hashing"), **overrides)


@pytest.fixture(scope="session")
def hashed_engine(tmp_path_factory, demo_csv):
    """Engine with the synthetic demo ingested (hashing embedder) — shared, read-only use."""
    from ltir.pipeline import Engine

    engine = Engine(make_config(tmp_path_factory.mktemp("ws_hash")), llm=FakeLLM())
    record = engine.ingest_file(demo_csv)
    assert record["status"] == "READY", record.get("error")
    return engine


@pytest.fixture(scope="session")
def model_engine(tmp_path_factory, demo_csv):
    """Engine with the synthetic demo ingested using the real local embedding model."""
    from ltir.pipeline import Engine

    engine = Engine(make_config(tmp_path_factory.mktemp("ws_model"), embedding_backend="sentence-transformers"), llm=FakeLLM())
    record = engine.ingest_file(demo_csv)
    assert record["status"] == "READY", record.get("error")
    return engine
