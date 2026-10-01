"""End-to-end: upload -> discover -> validate -> embed -> ontology -> graph -> query -> evidence -> answer (SDD 14)."""

from __future__ import annotations

import pytest
from conftest import FakeLLM

QUESTION = "Why is margin lower for phones in the US?"
EROSION_ANALOGUES = {
    "category=tablets AND region=APAC",
    "category=tablets AND channel=retail AND region=EU",
    "category=tablets AND channel=retail AND region=APAC",
    "category=tablets AND channel=retail",
}


def _check_pipeline_record(engine):
    rec = next(b for b in engine.ws.list_batches() if b["status"] == "READY")
    m = rec["metrics"]
    assert m["input_rows"] == 5000 and 0 < m["candidate_patterns"] <= m["search_space"]
    assert m["validated_insights"] >= 15 and m["attractors_total"] >= 3
    assert set(rec["stage_times"]) >= {
        "UPLOADED",
        "VALIDATING",
        "PROFILING",
        "DISCOVERING",
        "VALIDATING_INSIGHTS",
        "EMBEDDING",
        "UPDATING_ONTOLOGY",
        "BUILDING_GRAPH",
        "PERSISTING",
        "READY",
    }
    for key in ("orphan_rate", "activation_count", "avg_attractor_degree", "graph_edges", "processing_duration_s", "avg_insight_support"):
        assert key in m


def _check_grounded_answer(engine):
    qa = engine.ask(QUESTION)
    assert qa.answer_mode == "llm" and qa.citations["grounded"]
    assert qa.citations["cited"] and not qa.citations["unknown"]
    items = qa.evidence["items"]
    assert items[0]["role"] == "seed" and items[0]["scope"] == ["category=phones", "region=US"]
    for it in items:  # provenance down to dataset, batch, pattern, statistics and graph path
        assert it["provenance"]["dataset_id"] and it["provenance"]["batch_id"] and it["provenance"]["expression"]
        assert "support" in it["statistics"] and "shifts" in it["statistics"]
    cross = [it for it in items if it["transversal_only"]]
    assert cross, "expected scope-disjoint analogues reached through latent anchors"
    for it in cross:
        assert [s["edge_type"] for s in it["path"]][:1] == ["ACTIVATES"]
        assert any(s["target"].startswith("A-") for s in it["path"])
    assert {" AND ".join(sorted(it["scope"])) for it in cross} & EROSION_ANALOGUES
    h = qa.highlight
    assert h["seeds"] and h["anchors"] and h["edges"] and set(h["evidence"]) <= set(h["traversed"])
    assert "Sources:" in qa.provenance_footer and items[0]["pattern_id"] in qa.provenance_footer
    prompt = engine.llm.prompts[-1]
    assert "EVIDENCE — verified statistical observations" in prompt and "[P1]" in prompt
    return qa


def test_e2e_hashing_embedder(hashed_engine):
    _check_pipeline_record(hashed_engine)
    _check_grounded_answer(hashed_engine)


@pytest.mark.model
def test_e2e_local_embedding_model(model_engine):
    _check_pipeline_record(model_engine)
    qa = _check_grounded_answer(model_engine)
    assert qa.evidence["attractors"][0]["label"].startswith("discount ↑")


@pytest.mark.model
def test_hypothesis_apparatus(model_engine):
    from ltir.experiment import run_experiment

    res = run_experiment(model_engine, k=3)
    s = res["summary"]
    assert res["cases"] >= 5
    assert s["transversal"]["mrr"] > s["text_nn"]["mrr"] and s["transversal"]["mrr"] > s["structural"]["mrr"]
    assert s["transversal"]["recall"] > s["text_nn"]["recall"]


def test_llm_failure_keeps_knowledge(hashed_engine):
    llm = hashed_engine._llm
    hashed_engine._llm = FakeLLM(ok=False)
    try:
        qa = hashed_engine.ask(QUESTION)
    finally:
        hashed_engine._llm = llm
    assert qa.answer_mode == "fallback" and "Observations:" in qa.answer and "[P1]" in qa.answer
    assert qa.llm["error"] and qa.citations["grounded"]
    assert hashed_engine.graph().of_kind("Pattern")  # graph untouched


def test_empty_graph_answer(tmp_path):
    from conftest import make_config

    from ltir.pipeline import Engine

    qa = Engine(make_config(tmp_path / "empty"), llm=FakeLLM()).ask("anything?")
    assert qa.answer_mode == "empty" and qa.metrics["error"] == "empty_graph"
