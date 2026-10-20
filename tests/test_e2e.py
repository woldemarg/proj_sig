"""End-to-end: upload -> discover -> validate -> embed -> ontology -> graph -> query -> evidence -> answer (docs/09_operations.md)."""

from __future__ import annotations

import pytest
from conftest import FakeLLM, ask, make_config, make_engine

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
    llm = FakeLLM()
    qa = ask(engine, QUESTION, llm)
    assert qa.answer_mode == "llm" and qa.citations["grounded"]
    assert qa.citations["cited"] and not qa.citations["unknown"]
    items = qa.view["evidence"]["items"]
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
    h = qa.view["highlight"]
    assert h["seeds"] and h["anchors"] and h["edges"] and set(h["evidence"]) <= set(h["traversed"])
    assert "Sources:" in qa.provenance_footer and items[0]["pattern_id"] in qa.provenance_footer
    prompt = llm.prompts[-1]
    assert "EVIDENCE (statistical findings computed from the uploaded data" in prompt and "verified" not in prompt and "[P1]" in prompt
    assert prompt.isascii()  # no byte-fallback symbols reach the LLM (docs/07_question_answering.md §7.4)
    return qa


def test_e2e_hashing_embedder(hashed_engine):
    _check_pipeline_record(hashed_engine)
    _check_grounded_answer(hashed_engine)


@pytest.mark.model
def test_e2e_local_embedding_model(model_engine):
    _check_pipeline_record(model_engine)
    qa = _check_grounded_answer(model_engine)
    assert qa.view["evidence"]["attractors"][0]["label"].startswith("discount ↑")


@pytest.mark.model
def test_hypothesis_apparatus(model_engine):
    from experiment import run_experiment

    res = run_experiment(model_engine, k=3)
    s = res["summary"]
    assert res["cases"] >= 5
    assert s["transversal"]["mrr"] > s["text_nn"]["mrr"] and s["transversal"]["mrr"] > s["structural"]["mrr"]
    assert s["transversal"]["recall"] > s["text_nn"]["recall"]


def test_attractor_benchmark_apparatus(tmp_path):
    from attractor_benchmark import analyse, build_cases, generate, holm, rank_cases

    frame, truth = generate(1)
    assert len(truth) == 3 and all(len(scopes) == 3 for scopes in truth.values())  # three mechanisms, three disjoint scopes each
    assert all(not (set(a.items()) & set(b.items())) for scopes in truth.values() for a in scopes for b in scopes if a is not b)
    csv = tmp_path / "planted.csv"
    frame.to_csv(csv, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    record = engine.ingest_file(csv)
    assert record["status"] == "READY", record.get("error")
    cases = build_cases(engine, {record["batch_id"]: truth}, {record["batch_id"]: 1})
    assert cases and all(c.analogues and c.dataset == 1 for c in cases)
    rows = rank_cases(engine, cases, k=3, seed=1)
    rankers = ("transversal", "static", "random", "vector_nn", "text_nn", "weight_only")
    assert all(0.0 <= row[m][key] <= 1.0 for row in rows for m in rankers for key in ("recall", "precision", "mrr", "reached"))
    assert all(row[m]["reached"] == 1.0 for row in rows for m in ("vector_nn", "text_nn", "weight_only"))  # they rank every pattern
    result = analyse(rows, 3)
    assert result["cases"] == len(rows) and set(result["comparisons"]) == set(rankers[1:])
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06]) and holm([0.6, 0.5]) == [1.0, 1.0]


def test_geo_benchmark_apparatus(tmp_path):
    import h3
    import numpy as np
    import pandas as pd
    from geo_benchmark import BROAD, FINE, REGION, SIZE_BOX, generate, run_scenario
    from perf_benchmark import CALIFORNIA, EPOCH
    from test_spatial import geo_frame

    scenario = generate("S-regional", 1)
    (upload,) = scenario.uploads
    assert upload.frame.equals(generate("S-regional", 1).uploads[0].frame)  # the same seed writes the same file
    planted = upload.frame[np.logical_or.reduce(list(upload.parts.values()))]
    cells = {h3.latlng_to_cell(a, b, FINE) for a, b in zip(planted["lat"], planted["lon"], strict=True)}
    assert len(cells) == 3 * 37 and {h3.cell_to_parent(c, REGION) for c in cells} == {scenario.truth["region"]}  # three foci, one region
    for frame in (upload.frame, geo_frame().rename(columns={"latitude": "lat", "longitude": "lon"})):  # one territory, one epoch
        assert frame["lat"].between(CALIFORNIA[0], CALIFORNIA[2]).all() and frame["lon"].between(CALIFORNIA[1], CALIFORNIA[3]).all()
        assert (pd.to_datetime(frame["when"]) - pd.Timestamp(EPOCH)).dt.days.between(0, 30).all()
    for lat0, lon0, dlat, dlon in (BROAD, SIZE_BOX):  # the other scenarios' boxes; the focus scenarios share S-regional's area
        assert CALIFORNIA[0] <= lat0 and lat0 + dlat <= CALIFORNIA[2] and CALIFORNIA[1] <= lon0 and lon0 + dlon <= CALIFORNIA[3]
    result = run_scenario(scenario, tmp_path, make_engine(make_config(tmp_path / "ws")), k=5)
    found = result["files"][0]["found"]
    assert len({f["pattern"] for f in found.values()}) == 3 and all(f["coverage"] >= 0.9 for f in found.values())  # one insight per focus
    assert result["recall"]["cases"] == 3 and 0.0 <= result["recall"]["recall"] <= 1.0
    assert result["cross_coloc"] == 0 and result["prompt_chars"] > 0


def test_geo_coverage_apparatus(tmp_path):
    """The geo coverage measurement (docs/10_verification.md §10.4) on ``geo_frame`` with rows that have no point: every
    group gets the three coverages and the four rules on each region level; C_members is 1 for an insight of located rows
    and 0 for one of rows without a point; an insight of the whole map has lift 1 in every region."""
    import numpy as np
    import pandas as pd
    from geo_coverage import RULES, measure, run, upload_insights
    from test_spatial import geo_frame

    frame = geo_frame()
    blank = frame.head(60).assign(latitude=np.nan, longitude=np.nan)
    pd.concat([frame, blank], ignore_index=True).to_csv(path := tmp_path / "points.csv", index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    result = run(engine, path, "latitude,longitude@auto")
    assert result["map"]["rows"] == len(frame) + 60 and result["map"]["events"] == len(frame) and result["region_levels"] == [7, 5, 3, 1]
    base = result["groups"]["base"]
    assert base["insights"] > 0 and 0 < base["c_members"]["min"] <= base["c_members"]["median"] <= 1 and 0 < base["c_map"]["events"] <= 1
    assert all(0 <= base[lv][rule]["c_insights"] <= 1 for lv in result["region_levels"] for rule in RULES)
    finest, _ = upload_insights(engine, engine.ws.list_batches())
    n = len(finest["point_cell"])
    probe = {"located": np.arange(len(frame)), "none": np.arange(len(frame), n), "all": np.arange(n)}
    measured = measure(finest["point_cell"], finest["cells"], probe, [5, 3])["insights"]
    assert measured["located"]["c_members"] == 1.0 and measured["none"]["c_members"] == 0.0 and "levels" not in measured["none"]
    for lv in (5, 3):
        whole = measured["all"]["levels"][lv]
        assert whole["lift_min"] == pytest.approx(1.0) and whole["lift_max"] == pytest.approx(1.0) and whole["lift_2"] == 0


def test_perf_benchmark_apparatus(tmp_path, caplog):
    """The performance benchmark (docs/10_verification.md §10.4): its synthetic deliveries are the same for one seed and
    keep to their own territory and calendar; one upload through the API reports each level's stages, the Neo4j sync's
    included, then the console's reads cold and warm and three questions; the service logs every request's time."""
    import logging

    import pandas as pd
    from fastapi.testclient import TestClient
    from perf_benchmark import CALIFORNIA, EPOCH, reads_after, run_upload, write_synthetic_points

    from insight_graph_service.core.settings import PROJECT_ROOT
    from insight_graph_service.server.app import create_app

    (tmp_path / "again").mkdir()
    path = write_synthetic_points(tmp_path, n=3000, seed=1)
    assert path.read_bytes() == write_synthetic_points(tmp_path / "again", n=3000, seed=1).read_bytes()
    frame, (south, west, north, east) = pd.read_csv(path), CALIFORNIA
    assert list(frame.columns) == "lat lon ordered_at segment channel area vehicle refunded basket items wait_min rating".split()
    assert frame["lat"].between(south, north).all() and frame["lon"].between(west, east).all()
    assert (pd.to_datetime(frame["ordered_at"]) - pd.Timestamp(EPOCH)).dt.days.between(0, 59).all()
    engine = make_engine(make_config(tmp_path / "ws"))
    app = create_app(engine)
    with caplog.at_level(logging.INFO, logger="insight_graph_service.server.app"), TestClient(app) as client:
        upload = run_upload(client, app, PROJECT_ROOT / "data" / "demo" / "california_housing.csv", "auto@5", False, True)
        reads = reads_after(client, tmp_path / "ws")
    (level,) = upload["levels"]
    assert level["status"] == "READY" and {"validate_s", "discover_s", "embed_s", "graph_s", "neo4j_s"} <= set(level["timings"])
    paths = {r["path"].rsplit("/", 1)[-1] for r in reads["cold"]}
    assert {"graph", "cells", "overlay", "projected", "window"} <= paths and all(r["status"] == 200 for r in reads["cold"] + reads["warm"])
    assert len(reads["questions"]) == 3 and all(q["status"] == 200 for q in reads["questions"]) and "graph/snapshot.json" in reads["bytes"]
    assert any(r.getMessage().startswith("GET /api/geo") and r.getMessage().endswith(" s") for r in caplog.records)


def test_poisoning_eval_apparatus(tmp_path):
    """The deterministic red team (docs/10_verification.md §10.3): the spot planted in an unreliable source only is never
    confirmed and says so on its trust line in the evidence of every question, the opposite shifts of two sources are both
    5, a p-value edited in the journal leaves the evidence, and an edited reliability is caught and ignored; the variant
    before #8 really lacks the trust lines and rule 11."""
    from poisoning_eval import BEFORE_8, DIRECTED, check, ingest, score, tamper, without_trust

    from evidence_narrator_service.narration import SYSTEM_PROMPT

    engine = make_engine(make_config(tmp_path / "poison" / "ws"))
    scenario, case = ingest("S-poison", 1, tmp_path / "poison", engine)
    found = check(engine, case)
    (spot,) = case["flagged"]
    assert found[spot]["credibility"] == 2 and found[spot]["questions"] == 10 and len(found[spot]["trust"]) == 1
    assert found[spot]["trust"][0].startswith("source E (unreliable) | credibility 2 (not confirmed")
    assert found[case["confirmed"]]["credibility"] == 1
    full = engine.evidence(DIRECTED[0], cells=case["spot"])
    before = without_trust(full).evidence_prompt
    assert "     trust: " in full.evidence_prompt and "trust is given per item" in full.evidence_prompt
    assert "trust" not in before and "weakest credibility" not in before and before.count("\n[P") == full.evidence_prompt.count("\n[P")
    assert "11. TRUST" in SYSTEM_PROMPT and "11. TRUST" not in BEFORE_8 and BEFORE_8.endswith("Be concise (at most ~250 words).")
    key = next(k for k, c in full.citations.items() if c["pattern_id"] == spot)
    other = next(k for k in full.citations if k != key)
    assert score(f"Вищий load [{key}]. Але це не підтверджено [{key}][{other}].", full, [spot]) == {"mentioned": True, "all": False, "any": True}
    assert score(f"Дані надійно показують вищий load [{key}].", full, [spot])["any"] is False  # approval is not a qualification
    assert score(f"Вищий load [{other}].", full, [spot]) == {"mentioned": False, "all": False, "any": False}
    forged = tamper(engine, case, scenario.uploads[-1].name)
    assert forged["metadata"]["dataset"] == "mismatch" and forged["metadata"]["note"]
    assert forged["metadata"]["trust"].startswith("source unknown (its dataset's metadata failed the integrity check)")
    assert forged["integrity"] == {"integrity": "mismatch", "in_evidence": False, "note": True}

    engine = make_engine(make_config(tmp_path / "two" / "ws"))
    _, case = ingest("S-two-sources", 1, tmp_path / "two", engine)
    found = check(engine, case)
    assert len(case["flagged"]) == 2 and all(found[p]["credibility"] == 5 for p in case["flagged"])


def test_chain_eval_apparatus(hashed_engine):
    from chain_eval import run_chain_eval

    res = run_chain_eval(hashed_engine)
    layers = res["layers"]
    assert set(layers) == {"structural", "latent", "union"} and res["sets"] >= 2 and res["disjoint_pairs"] > 0
    assert layers["structural"]["disjoint@1"] == 0.0  # a lattice edge needs a shared condition; analogues share none
    assert layers["union"]["pairs@2"] >= max(layers["structural"]["pairs@2"], layers["latent"]["pairs@2"])
    assert all(0.0 <= m[f"ball@{h}"] <= 1.0 for m in layers.values() for h in (0, 2, 4))


def test_llm_failure_keeps_knowledge(hashed_engine):
    qa = ask(hashed_engine, QUESTION, FakeLLM(ok=False))
    assert qa.answer_mode == "fallback" and qa.answer.startswith("Спостереження:") and "[P1]" in qa.answer
    assert "category=phones" in qa.answer and "margin" in qa.answer  # literals as stored, Ukrainian around them
    assert qa.llm["error"] and qa.citations["grounded"]
    assert hashed_engine.graph().of_kind("Pattern")  # graph untouched


def test_answering_embeds_the_question_not_the_corpus(tmp_path, demo_csv, monkeypatch):
    """A batch embeds each insight's three blocks and no canonical document (the naive text baseline is a measurement of
    scripts/experiment.py, docs/07_question_answering.md §7.6); an answer embeds the question, never a document, and
    carries the structural baselines only."""
    engine = make_engine(make_config(tmp_path / "ws"))
    embedder, seen = engine.encoder.embedder, []
    for name in ("embed", "embed_queries"):
        original = getattr(embedder, name)
        monkeypatch.setattr(embedder, name, lambda texts, original=original: seen.extend(texts) or original(texts))
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    graph = engine.graph()
    documents = {graph.canonical_document(n["id"]) for n in graph.of_kind("Pattern")}
    assert seen and not documents & set(seen)  # the batch embedded blocks and literals, no document
    seen.clear()
    qa = ask(engine, "Why is margin lower for phones in the US?")
    assert seen and not documents & set(seen)
    assert set(qa.view["traversal"]["baselines"]) == {"structural_only", "transversal_only", "not_structurally_reachable"}


def test_search_is_the_structured_context_the_narrator_verbalises(hashed_engine):
    """Retrieval stands alone (docs/12_architecture.md): the same evidence with or without the narrator on top."""
    from graph_query_engine.search import search

    found = hashed_engine.search(QUESTION)
    assert (
        found.evidence.items and found.seeds and set(found.highlight()) == {"seeds", "traversed", "anchors", "evidence", "edges", "transversal_only"}
    )
    assert found.evidence.to_prompt().startswith("QUESTION: ")
    again = search(QUESTION, hashed_engine.committed(), hashed_engine.encoder, hashed_engine.settings.query)  # the reuse API, no engine in between
    assert [s.pattern_id for s in again.seeds] == [s.pattern_id for s in found.seeds] and again.highlight() == found.highlight()
    qa = ask(hashed_engine, QUESTION)
    assert qa.view["highlight"] == found.highlight() and qa.answer_mode == "fallback" and qa.llm["error"] == "disabled"
    assert 0 < qa.metrics["retrieval_s"] <= qa.metrics["total_s"] and qa.metrics["retrieved_evidence"] == len(found.evidence.items)


def test_empty_graph_answer(tmp_path):

    qa = ask(make_engine(make_config(tmp_path / "empty")), "anything?")
    assert qa.answer_mode == "empty" and qa.metrics["error"] == "empty_graph"
