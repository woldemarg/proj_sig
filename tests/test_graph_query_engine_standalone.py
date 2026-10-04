"""graph_query_engine on its own: a snapshot and its vectors in, evidence and an LLM-ready payload out
(graph_query_engine/README.md). No engine, no workspace, no attractor_topology: a hand-written snapshot and an
encoder that only has the ``QueryEncoder`` shape."""

from __future__ import annotations

import json

import numpy as np
from doubles import HashingEmbedder
from test_traversal import edge, pattern

QUESTION = "Why is margin lower for phones in the US?"


class ShapeOnlyEncoder:
    """Anything with ``embedder`` and ``encode_query`` will do; here every question points at the margin patterns."""

    def __init__(self) -> None:
        self.embedder = HashingEmbedder()

    def encode_query(self, scope_text, target_text, components, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def snapshot() -> dict:
    from insight_contracts import SNAPSHOT_VERSION

    signature = [{"component": "margin", "value": -1.0}]
    return {
        "version": SNAPSHOT_VERSION,
        "nodes": [
            pattern("P1", ["region=US", "category=phones"]),
            pattern("P2", ["region=EU", "category=tablets"]),
            {"id": "A-1", "kind": "Attractor", "label": "margin ↓", "props": {"n_patterns": 2, "distinct_scopes": 2, "signature": signature}},
            {"id": "M:ds1:margin", "kind": "Metric", "label": "margin", "props": {"name": "margin", "global_median": 12.0, "global_mad": 2.0}},
            {"id": "D:ds1:region", "kind": "Dimension", "label": "region", "props": {"name": "region"}},
        ],
        "edges": [edge("P1", "A-1", "ACTIVATES", 0.9), edge("P2", "A-1", "ACTIVATES", 0.8)],
    }


def test_a_question_becomes_an_evidence_payload():
    from graph_query_engine import CommittedState, DualGraph, LatentFrame, QueryConfig, empty_payload, search
    from insight_contracts import EvidencePayload

    graph, encoder = DualGraph(snapshot()), ShapeOnlyEncoder()
    vectors = {"P1": np.array([1.0, 0.0, 0.0]), "P2": np.array([0.8, 0.6, 0.0])}
    documents = {pid: encoder.embedder.embed([graph.canonical_document(pid)])[0] for pid in graph.insights}
    state = CommittedState(graph, LatentFrame(vectors, {}, documents))
    found = search(QUESTION, state, encoder, QueryConfig())
    payload = found.payload()

    assert state.catalog is not None  # built on first use: the graph's literals ground "phones" and "US"
    assert found.seeds and found.seeds[0].pattern_id == "P1"
    assert {c["pattern_id"] for c in payload.citations.values()} >= {"P1"} and payload.evidence_prompt and payload.evidence_summary
    assert {k: v["pattern_id"] for k, v in payload.citations.items()} == {it.key: it.pattern_id for it in found.evidence.items}  # the manifest
    assert payload.view["evidence"]["parsed"]["targets"] and "prompt" not in payload.view["evidence"]  # each fact travels once
    wire = json.loads(json.dumps(payload.to_dict()))  # the service boundary
    assert EvidencePayload.from_dict(wire) == payload
    empty = empty_payload("anything?")
    assert empty.graph_empty and set(empty.view["highlight"]) == set(payload.view["highlight"])


def test_the_citations_come_from_the_insight_not_its_free_form_provenance():
    """Pattern records whose provenance is empty still answer with a full citation manifest (no file name)."""
    from graph_query_engine import CommittedState, DualGraph, LatentFrame, QueryConfig, search

    snap = snapshot()
    snap["nodes"] = [{**n, "props": {**n["props"], "provenance": {}}} if n["kind"] == "Pattern" else n for n in snap["nodes"]]
    graph, encoder = DualGraph(snap), ShapeOnlyEncoder()
    documents = {pid: encoder.embedder.embed([graph.canonical_document(pid)])[0] for pid in graph.insights}
    state = CommittedState(graph, LatentFrame({"P1": np.array([1.0, 0.0, 0.0]), "P2": np.array([0.8, 0.6, 0.0])}, {}, documents))
    payload = search(QUESTION, state, encoder, QueryConfig()).payload()
    cited = {c["pattern_id"]: c for c in payload.citations.values()}
    assert cited["P1"] == {"pattern_id": "P1", "expression": "region=US AND category=phones", "dataset_id": "ds1", "filename": "", "batch_id": "B1"}
    assert payload.view["evidence"]["datasets"][0]["batch_id"] == "B1"


def test_the_payload_contract_is_strict():
    import pytest

    from insight_contracts import EvidencePayload

    payload = EvidencePayload("q?", evidence_prompt="QUESTION: q?", evidence_summary="- a=1 | m +1.00 sd | n=10 [P1]", metrics={"retrieval_s": 0.1})
    wire = payload.to_dict()
    assert EvidencePayload.from_dict(wire) == payload
    with pytest.raises(KeyError):
        EvidencePayload.from_dict({k: v for k, v in wire.items() if k != "metrics"})  # a missing field is a contract break


def test_a_snapshot_of_another_version_is_refused():
    import pytest

    from graph_query_engine import DualGraph

    with pytest.raises(ValueError, match="snapshot version"):
        DualGraph({**snapshot(), "version": 1})
