"""The Neo4j mirror: the snapshot published through a fake driver, full and as a diff, CO_OCCURS and the Cypher walk (docs/06_graph_and_storage.md §6.6)."""

from __future__ import annotations

import copy
import json

import pytest
from conftest import make_config, make_engine

from insight_contracts import LATENT_EDGES, EdgeType, GraphEdge
from insight_graph_service.core.neo4j_mirror import (
    EDGE_ENDPOINTS,
    LABELS,
    STALE_TYPES_QUERY,
    Mirror,
    edge_query,
    flatten_props,
    gone_edge_query,
    gone_node_query,
    load_cypher,
    node_query,
    stale_edge_query,
    stale_node_query,
)
from insight_graph_service.core.snapshot import co_occurrence_edges


class _Session:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        self.log.append((query, params))

    def execute_write(self, fn):
        fn(self)


class _Driver:
    def __init__(self):
        self.log, self.sessions = [], 0

    def session(self, database=None):
        self.sessions += 1
        return _Session(self.log)


def test_neo4j_mirror_equals_the_snapshot(hashed_engine):
    snap = hashed_engine.ws.load_graph()
    driver = _Driver()
    out = Mirror().publish(snap, hashed_engine.settings, driver=driver)
    queries = [q for q, _ in driver.log]
    assert any("CREATE CONSTRAINT pattern_id_unique" in q for q in queries)
    assert all("CREATE (" not in q for q in queries)  # idempotent: MERGE only
    assert out["nodes"]["Pattern"] == len(hashed_engine.graph().of_kind("Pattern")) and out["sync"] == "full"
    assert "MERGE (s)-[r:ACTIVATES]->(t)" in edge_query("ACTIVATES")
    # properties are replaced, not merged: a property that disappeared (or became None) is removed
    assert "SET n = row.props, n.id = row.id" in node_query("Pattern") and "SET r = row.props" in edge_query("RELATED_TO")
    # what the snapshot no longer holds is deleted: every label and every edge type, keyed by the snapshot ids
    stale = {q: p for q, p in driver.log if "NOT" in q}
    assert {q for q in stale if "DETACH DELETE" in q} == {stale_node_query(label) for label in LABELS}
    assert sorted(stale[stale_node_query("Pattern")]["ids"]) == sorted(n["id"] for n in hashed_engine.graph().of_kind("Pattern"))
    assert {q for q in stale if q.endswith("DELETE r")} == {stale_edge_query(t) for t in EDGE_ENDPOINTS} | {STALE_TYPES_QUERY}
    assert stale[STALE_TYPES_QUERY] == {"types": list(EDGE_ENDPOINTS), "labels": list(LABELS)}  # a type the schema dropped
    # an empty snapshot (a reset) clears the mirror
    empty = _Driver()
    Mirror().publish({"nodes": [], "edges": []}, hashed_engine.settings, driver=empty)
    assert all(p["ids"] == [] for q, p in empty.log if "DETACH DELETE" in q)
    flat = flatten_props({"a": 1, "b": [1, 2], "c": {"x": 1}, "d": None, "e": [{"k": 1}]})
    assert flat == {
        "a": 1,
        "b": [1, 2],
        "c_json": json.dumps({"x": 1}),
        "e_json": json.dumps([{"k": 1}]),
    }


def test_the_mirror_links_the_windows_of_a_trajectory(windowed):
    """EVOLVES_TO reaches Neo4j as a Pattern → Pattern relationship with its props (``shifts`` as JSON), one per edge."""
    engine = windowed[0]
    snap = engine.ws.load_graph()
    driver = _Driver()
    out = Mirror().publish(snap, engine.settings, driver=driver)
    rows = [row for q, p in driver.log if q == edge_query("EVOLVES_TO") for row in p["rows"]]
    evolves = [e for e in snap["edges"] if e["type"] == "EVOLVES_TO"]
    assert EDGE_ENDPOINTS["EVOLVES_TO"] == ("Pattern", "Pattern") and out["edges"]["EVOLVES_TO"] == len(rows) == len(evolves) > 0
    assert {(r["source"], r["target"]) for r in rows} == {(e["source"], e["target"]) for e in evolves}
    assert all({"trajectory", "transition", "from_window", "to_window", "shifts_json"} <= r["props"].keys() for r in rows)


def test_a_publish_after_the_first_sends_only_what_changed(hashed_engine, tmp_path, monkeypatch):
    """The diff with the last publish written: changed rows rewritten, gone ones deleted by id, no stale sweep; a
    failed publish and a reset are followed by a full sync."""
    settings, snap, mirror = hashed_engine.settings, hashed_engine.ws.load_graph(), Mirror()
    full = mirror.publish(snap, settings, driver=_Driver())
    again = _Driver()
    assert mirror.publish(snap, settings, driver=again) == {**full, "sync": "diff", "written": 0, "deleted": 0}  # the snapshot's counts
    assert again.sessions == 0  # nothing changed: no session at all

    nxt = copy.deepcopy(snap)
    patterns = [n for n in nxt["nodes"] if n["kind"] == "Pattern"]
    gone, changed = patterns[0]["id"], patterns[1]
    changed["props"]["weight"] = 0.123
    edge = next(e for e in nxt["edges"] if gone not in (e["source"], e["target"]))
    edge["props"] = {**edge["props"], "note": "x"}  # the same pair, new props
    nxt["nodes"] = [n for n in nxt["nodes"] if n["id"] != gone]
    dropped = [e for e in nxt["edges"] if gone in (e["source"], e["target"])]
    nxt["edges"] = [e for e in nxt["edges"] if e not in dropped]
    driver = _Driver()
    out = mirror.publish(nxt, settings, driver=driver)
    sent = {q: p for q, p in driver.log if "UNWIND" in q or "DELETE" in q}
    assert out["sync"] == "diff" and out["written"] == 2 and out["deleted"] == 1 + len(dropped)
    assert out["nodes"]["Pattern"] == len(patterns) - 1 and not [q for q, _ in driver.log if "CREATE" in q]  # no setup again
    assert [r["id"] for r in sent[node_query("Pattern")]["rows"]] == [changed["id"]]
    assert sent[edge_query(edge["type"])]["rows"][0]["props"]["note"] == "x"
    assert sent[gone_node_query("Pattern")] == {"ids": [gone]}
    deleted = [(t, r["source"], r["target"]) for t in EDGE_ENDPOINTS if (q := gone_edge_query(t)) in sent for r in sent[q]["rows"]]
    assert sorted(deleted) == sorted((e["type"], e["source"], e["target"]) for e in dropped)
    assert not [q for q in sent if q == STALE_TYPES_QUERY or "NOT" in q]

    class _Down(_Driver):
        def session(self, database=None):
            raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        mirror.publish(snap, settings, driver=_Down())
    assert mirror.publish(snap, settings, driver=_Driver())["sync"] == "full"
    engine = make_engine(make_config(tmp_path, neo4j_enabled=True))
    engine._mirror.written = mirror.written
    monkeypatch.setattr(engine._mirror, "publish", lambda s, c: Mirror.publish(engine._mirror, s, c, driver=_Driver()))
    assert engine.reset()["neo4j"]["sync"] == "full"


def test_co_occurrence_edges_from_memberships():
    """W[p, j] = a_j^2 / sum_k a_k over the non-weak alignments; weak memberships do not count; the pair weight is capped at 1."""

    def act(pid, aid, alignment, weak=False):
        return GraphEdge(pid, f"A-{aid}", EdgeType.ACTIVATES, alignment, {"weak": weak})

    edges = co_occurrence_edges([act("P1", 0, 0.9), act("P1", 1, 0.6), act("P2", 0, 0.8), act("P2", 1, 0.1, weak=True), act("P3", 1, 0.95)])
    assert [(e.source, e.target, e.type) for e in edges] == [("A-0", "A-1", EdgeType.CO_OCCURS)]
    assert edges[0].weight == pytest.approx((0.9**2 / 1.5) * (0.6**2 / 1.5)) and edges[0].props == {"kind": "co_occurrence", "shared": 1}
    many = [act(f"P{i}", j, 1.0) for i in range(5) for j in (0, 1)]  # five insights split evenly: 5 x 0.25
    assert [(e.weight, e.props["shared"]) for e in co_occurrence_edges(many)] == [(1.0, 5)]


def test_cypher_walks_the_latent_edge_set():
    query = load_cypher("queries/transversal")
    assert "[rel:RELATED_TO|CO_OCCURS]" in query and all(e.value in query for e in LATENT_EDGES)
