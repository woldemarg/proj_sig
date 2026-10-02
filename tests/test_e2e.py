"""End-to-end: upload -> discover -> validate -> embed -> ontology -> graph -> query -> evidence -> answer (docs/09_operations.md)."""

from __future__ import annotations

import pytest
from conftest import FakeLLM, make_config

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
    assert "EVIDENCE (verified statistical observations" in prompt and "[P1]" in prompt
    assert prompt.isascii()  # no byte-fallback symbols reach the LLM (docs/07_question_answering.md §7.4)
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
    assert qa.answer_mode == "fallback" and qa.answer.startswith("Спостереження:") and "[P1]" in qa.answer
    assert "category=phones" in qa.answer and "margin" in qa.answer  # literals as stored, Ukrainian around them
    assert qa.llm["error"] and qa.citations["grounded"]
    assert hashed_engine.graph().of_kind("Pattern")  # graph untouched


def test_answering_embeds_the_question_not_the_corpus(hashed_engine, monkeypatch):
    """Document vectors are stored at ingest: an answer (and its naive text baseline) embeds no document."""
    graph = hashed_engine.graph()
    documents = {graph.canonical_document(n["id"]) for n in graph.of_kind("Pattern")}
    embedder = hashed_engine.encoder.embedder
    seen: list[str] = []
    for name in ("embed", "embed_queries"):
        original = getattr(embedder, name)
        monkeypatch.setattr(embedder, name, lambda texts, original=original: seen.extend(texts) or original(texts))
    qa = hashed_engine.ask("Why is margin lower for phones in the US?", use_llm=False)
    assert qa.traversal["baselines"]["naive_nearest"]
    assert seen and not documents & set(seen)


def test_missing_document_vectors_are_filled_once(tmp_path, demo_csv, monkeypatch):
    """A batch without stored document vectors is embedded on the first question only, and a writer saves them."""
    import numpy as np

    from ltir.pipeline import Engine

    cfg = make_config(tmp_path / "ws")
    Engine(cfg, llm=FakeLLM()).ingest_file(demo_csv)
    for path in (cfg.workspace_dir / "journal" / "blocks").glob("*.npz"):  # as written before documents were stored
        with np.load(path) as data:
            kept = {k: data[k] for k in data.files if k not in ("document", "pattern_ids")}
        np.savez_compressed(path, **kept)
    engine = Engine(cfg, llm=FakeLLM())
    graph = engine.graph()
    documents = {graph.canonical_document(n["id"]) for n in graph.of_kind("Pattern")}
    seen: list[str] = []
    original = engine.encoder.embedder.embed
    monkeypatch.setattr(engine.encoder.embedder, "embed", lambda texts: seen.extend(texts) or original(texts))
    engine.ask("Why is margin lower for phones in the US?", use_llm=False)
    assert documents <= set(seen)  # filled on the first question
    seen.clear()
    engine.ask("Why is margin lower for phones in the US?", use_llm=False)
    assert not documents & set(seen)  # kept on the frame
    assert set(Engine(cfg, llm=FakeLLM(), recover=False).frame().documents) == {n["id"] for n in graph.of_kind("Pattern")}  # saved


def test_empty_graph_answer(tmp_path):

    from ltir.pipeline import Engine

    qa = Engine(make_config(tmp_path / "empty"), llm=FakeLLM()).ask("anything?")
    assert qa.answer_mode == "empty" and qa.metrics["error"] == "empty_graph"
