"""Literal grounding and multilingual retrieval (docs/07_question_answering.md §7.1.1, §7.7).

The gate logic runs on a dictionary embedder with hand-made unit vectors, so it is deterministic and
needs no model; the four buckets of the document's benchmark run on the hashing engine (B1, B2) and,
under the ``model`` marker, on the real embedder (B3, B4).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeLLM, make_config
from test_traversal import pattern

from ltir import query as q
from ltir.graph import DualGraph
from ltir.pipeline import Engine
from ltir.query import LiteralCatalog, build_catalog, parse_query, resolve_seeds, score_pattern
from ltir.synth import generate_retail_dataset


class DictEmbedder:
    """Unit vectors by word: translations share a vector, a hub literal sits close to everything."""

    model_id, dim, query_instruction, revision, compute_dtype, truncate_dim = "dict", 12, "", "", "float64", 0
    BASE = {
        "margin": 0,
        "маржа": 0,
        "phones": 1,
        "телефонів": 1,
        "телефони": 1,
        "US": 2,
        "США": 2,
        "us": 2,
        "EU": 3,
        "Харків": 4,
        "Харкові": 4,
        "delivery days": 5,
        "дні доставки": 5,
        "delivery_days": 5,
        "return rate": 6,
        "return_rate": 6,
        "higher increase growth up": 8,
        "lower decrease drop down reduction": 9,
        "retail": 7,
    }

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        if text in self.BASE:
            v[self.BASE[text]] = 1.0
        elif text == "hub":  # the hub literal: 0.7 cosine to every other literal direction
            v[:7] = 1.0
        elif text == "обидва":  # equidistant to two literals: ambiguous
            v[0] = v[1] = 1.0
        else:
            v[10] = 1.0  # everything unknown points the same, far-away way (no literal lives there)
        return v / np.linalg.norm(v)

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)

    embed_queries = embed


def toy_graph(extra: list | None = None) -> DualGraph:
    nodes = [
        pattern("P1", ["region=US", "category=phones"]),
        pattern("P2", ["region=EU", "category=phones"], z=1.0),
        pattern("P3", ["city=Харків", "category=phones"], target="return_rate"),
        pattern("P4", ["region=EU", "channel=retail"], target="delivery_days"),
        {"id": "M:ds1:margin", "kind": "Metric", "label": "margin", "props": {"name": "margin"}},
        {"id": "M:ds1:return_rate", "kind": "Metric", "label": "return rate", "props": {"name": "return_rate"}},
        {"id": "M:ds1:delivery_days", "kind": "Metric", "label": "delivery days", "props": {"name": "delivery_days"}},
        {"id": "D:ds1:region", "kind": "Dimension", "label": "region", "props": {"name": "region"}},
        {"id": "D:ds1:city", "kind": "Dimension", "label": "city", "props": {"name": "city"}},
        *(extra or []),
    ]
    return DualGraph({"nodes": nodes, "edges": []})


def grounded(graph: DualGraph, embedder=None) -> DualGraph:
    graph._catalog = build_catalog(graph, embedder or DictEmbedder(), "fp")
    return graph


def test_dense_gate_accepts_translations_and_rejects_case():
    g = grounded(toy_graph())
    p = parse_query("Чому маржа нижча для телефонів у США?", g)
    assert p.targets == ["margin"] and set(p.conditions) == {("category", "phones"), ("region", "US")} and p.direction == -1
    assert {x["span"]: x["literal"] for x in p.grounding} == {"маржа": "margin", "телефонів": "phones", "США": "US"}
    assert all(x["layer"] == "dense" and x["score"] > 0 for x in p.grounding)
    assert ("region", "US") not in parse_query("tell us about margins", g).conditions  # the acronym literal needs an upper-case span


def test_hub_and_ambiguity_are_rejected():
    g = grounded(toy_graph([pattern("P5", ["region=hub"])]))
    p = parse_query("Що обидва?", g)  # "обидва" is equidistant to margin and phones: Lowe's ratio rejects it
    assert p.grounding == [] and p.targets == [] and p.conditions == []
    p = parse_query("маржа", g)  # the hub literal is close to everything: the margin gate keeps margin on top
    assert [x["literal"] for x in p.grounding] == ["margin"]


def test_chars_match_same_script_inflections_only():
    g = grounded(toy_graph())
    p = parse_query("Чому маржа нижча у Харкові?", g)
    assert ("city", "Харків") in p.conditions and [x["layer"] for x in p.grounding if x["literal"] == "Харків"] == ["chars"]
    assert ("category", "phones") in parse_query("Why is margin lower for fones?", g).conditions  # a typo, same script
    assert "margin" not in [x["literal"] for x in parse_query("Did sales change marginally?", g).grounding]  # 4 letters longer: no


def test_longest_span_claims_its_tokens_and_the_rest_is_not_embedded():
    emb = DictEmbedder()
    g = grounded(toy_graph(), emb)
    emb.calls.clear()
    p = parse_query("Які сегменти мають довші дні доставки?", g)
    assert p.targets == ["delivery_days"] and len(p.grounding) == 1 and p.grounding[0]["span"] == "дні доставки"
    assert len(emb.calls) == 1 and len(emb.calls[0]) <= 8  # one forward pass over the unresolved spans only
    emb.calls.clear()
    assert parse_query("Why is margin lower for phones in the US?", g).grounding == [] and emb.calls == []  # fully lexical: no embedding


def test_shared_value_binds_the_named_column_or_a_wildcard(tmp_path):
    g = grounded(toy_graph([pattern("P6", ["origin=US", "category=phones"])]))
    p = parse_query("Чому маржа нижча для США?", g)
    assert ("*", "US") in p.conditions  # US lives under region and origin: any column
    ins = g.insight("P1")
    s = score_pattern(ins, p, np.zeros(1), None, make_config(tmp_path))
    assert s.matched["scope"] == 1.0 and s.matched["scope_conflicts"] == 0
    p = parse_query("Чому маржа нижча для США за region?", grounded(toy_graph([pattern("P6", ["origin=US", "category=phones"])])))
    assert ("region", "US") in p.conditions and ("*", "US") not in p.conditions


def test_ukrainian_comparatives_and_directed_drivers():
    g = toy_graph()  # no catalog: lexicon and rules only
    assert parse_query("Де margin більш низький?", g).direction == -1
    assert parse_query("Де margin менш низький?", g).direction == 1
    assert parse_query("Де margin менш високий?", g).direction == -1
    assert parse_query("Де margin більший?", g).direction == 1
    p = parse_query("Що пов'язано з вищим return_rate?", g)
    assert not p.covariance and p.direction == 1 and p.targets == ["return_rate"]
    p = parse_query("What is associated with higher return_rate?", g)
    assert not p.covariance and p.direction == 1
    assert parse_query("Де руйнується зв'язок між discount і margin?", g).covariance
    assert parse_query("Де вища знижка?", g).direction == 1  # знижка is a discount, not "зниження"


def seeds(engine, question):
    graph = engine.graph()
    parsed = parse_query(question, graph, engine.config)
    return sorted(s.pattern_id for s in resolve_seeds(parsed, graph, engine.encoder, engine.frame().patterns, engine.config))


def test_b1_b2_code_switched_questions_seed_identically(hashed_engine):
    for en, uk in (
        ("Why is margin lower for phones in the US?", "Чому margin нижчий для phones у US?"),
        ("Where does the correlation between discount and margin break down?", "Де руйнується кореляція між discount і margin?"),
        ("What drives higher return_rate?", "Що спричиняє вищий return_rate?"),
    ):
        assert seeds(hashed_engine, en) == seeds(hashed_engine, uk), uk


def test_catalog_is_persisted_and_reloaded(tmp_path, demo_csv):
    cfg = make_config(tmp_path / "ws")
    writer = Engine(cfg, llm=FakeLLM())
    assert writer.ingest_file(demo_csv)["status"] == "READY"
    stored = writer.ws.load_literals()
    assert stored is not None and str(stored["fingerprint"]) == writer.encoder.spec.fingerprint
    reader = Engine(cfg, llm=FakeLLM(), recover=False)
    calls = []
    original = reader.encoder.embedder.embed
    reader.encoder.embedder.embed = lambda texts: calls.append(list(texts)) or original(texts)
    catalog = reader.graph().catalog
    assert isinstance(catalog, LiteralCatalog) and catalog.texts == writer.graph().catalog.texts and calls == []  # loaded, not re-embedded
    assert "margin" in catalog.texts and "phones" in catalog.texts and not (cfg.workspace_dir / "state" / "literals.npz.tmp").exists()


@pytest.mark.model
def test_b3_translated_question_parses_like_english(model_engine):
    en = parse_query("Why is margin lower for phones in the US?", model_engine.graph(), model_engine.config)
    uk = parse_query("Чому маржа нижча для телефонів у США?", model_engine.graph(), model_engine.config)
    assert set(uk.conditions) == set(en.conditions) == {("category", "phones"), ("region", "US")} and uk.direction == en.direction == -1
    assert {x["literal"] for x in uk.grounding} >= {"phones", "US"}  # маржа itself is the one literal this model confuses (§7.1.1)
    assert set(seeds(model_engine, "Why is margin lower for phones in the US?")) <= set(seeds(model_engine, "Чому маржа нижча для телефонів у США?"))
    for distractor in ("tell us about margins", "Did sales change marginally?", "Де низька ціна?"):
        p = parse_query(distractor, model_engine.graph(), model_engine.config)
        assert ("region", "US") not in p.conditions and "margin" not in [x["literal"] for x in p.grounding], distractor


@pytest.mark.model
def test_b4_inflected_ukrainian_value(tmp_path):
    df = generate_retail_dataset(5000, 7)
    df["region"] = df["region"].map({"EU": "Львів", "US": "Харків", "APAC": "Київ", "LATAM": "Одеса"})
    df["category"] = df["category"].map({"laptops": "ноутбуки", "phones": "телефони", "tablets": "планшети", "accessories": "аксесуари"})
    path = tmp_path / "uk.csv"
    df.rename(columns={"region": "city"}).to_csv(path, index=False)
    engine = Engine(make_config(tmp_path / "ws", embedding_backend="sentence-transformers"), llm=FakeLLM())
    assert engine.ingest_file(path)["status"] == "READY"
    p = parse_query("Чому margin нижчий для телефонів у Харкові?", engine.graph(), engine.config)
    assert ("city", "Харків") in p.conditions and ("category", "телефони") in p.conditions
    assert seeds(engine, "Чому margin нижчий для телефонів у Харкові?") == seeds(engine, "Why is margin lower for телефони in Харків?")


def test_neo4j_mirror_does_not_change_retrieval(tmp_path, demo_csv, monkeypatch):
    """The document's falsification step 6: identical seed scores with the mirror on (fake driver) and off."""
    from dataclasses import replace

    from test_persistence import _Driver

    from ltir import neo4j_sink

    cfg = make_config(tmp_path / "ws")
    engine = Engine(cfg, llm=FakeLLM())
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    question = "Чому margin нижчий для phones у US?"
    off = engine.ask(question, use_llm=False).traversal["seeds"]
    real = neo4j_sink.publish_snapshot
    monkeypatch.setattr(neo4j_sink, "publish_snapshot", lambda snapshot, config, driver=None: real(snapshot, config, driver=_Driver()))
    engine.config = replace(cfg, neo4j_enabled=True)
    assert engine.sync_neo4j()["status"] == "ok"
    assert engine.ask(question, use_llm=False).traversal["seeds"] == off


def test_grounding_switches_are_production_defaults():
    assert q.GROUNDING == {"layers": "ABC", "gates": True, "rules": True}
