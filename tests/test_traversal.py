"""Traversal on a deterministic toy graph + query parsing + evidence (SDD 10, SDD 11)."""

from __future__ import annotations

import pytest

from ltir.config import load_config
from ltir.evidence import build_evidence
from ltir.graph import DualGraph
from ltir.models import Condition, EdgeType, GraphEdge, Insight, Shift
from ltir.query import SeedMatch, parse_query
from ltir.traversal import traverse


def pattern(pid, conds, target="margin", z=-1.5, w=0.8, dataset="ds1"):
    """Snapshot pattern node: props are an Insight journal record."""
    conditions = tuple(Condition(*c.split("=", 1)) for c in conds)
    ins = Insight(
        id=pid,
        dataset_id=dataset,
        batch_id="B1",
        conditions=conditions,
        expression=" AND ".join(conds),
        target=target,
        shifts=(Shift(target, z, 10.0, 12.0, 1.0),),
        support=100,
        support_fraction=0.1,
        baseline=12.0,
        local=10.0,
        effect_size=z,
        sd_score=1.0,
        sd_raw_score=1.0,
        emm_score=0.1,
        volume_utility=0.3,
        stability=0.9,
        integrated_index=1.0,
        p_value=1e-6,
        p_adjusted=1e-4,
        drivers=(),
        row_hash=pid,
        weight=w,
        provenance={
            "dataset_id": dataset,
            "filename": "toy.csv",
            "batch_id": "B1",
            "expression": " AND ".join(conds),
            "engine": "ltir/engines/eda/main_upd.py",
        },
    )
    props = ins.to_record()
    props["canonical"] = {"document": f"TARGET: {target}"}
    props["row_id"] = 0
    return {"id": pid, "kind": "Pattern", "label": f"{pid} {conds}", "props": props}


def edge(s, t, etype, w=1.0):
    return GraphEdge(s, t, EdgeType(etype), w).to_dict()


@pytest.fixture
def toy():
    """P1 -> A1 -RELATED_TO- A2 <- P3 ; P2 -> A1 ; P3 specialises P4 ; P5 weakly activates A1 ; A3 weakly related."""
    nodes = [
        pattern("P1", ["region=EU", "category=phones"]),
        pattern("P2", ["region=EU", "category=tablets"]),
        pattern("P3", ["region=US", "channel=online", "category=laptops"]),
        pattern("P4", ["region=US", "channel=online"], z=-1.0),
        pattern("P5", ["region=APAC", "category=tv"]),
        pattern("P6", ["region=LATAM", "category=tv"]),
        {"id": "A1", "kind": "Attractor", "label": "margin ↓", "props": {"n_patterns": 3, "distinct_scopes": 3, "signature": []}},
        {"id": "A2", "kind": "Attractor", "label": "margin ↓ · discount ↑", "props": {"n_patterns": 1, "distinct_scopes": 1, "signature": []}},
        {"id": "A3", "kind": "Attractor", "label": "unrelated", "props": {"n_patterns": 1, "distinct_scopes": 1, "signature": []}},
        {"id": "M:ds1:margin", "kind": "Metric", "label": "margin", "props": {"name": "margin", "global_median": 12.0, "global_mad": 2.0}},
        {"id": "D:ds1:region", "kind": "Dimension", "label": "region", "props": {"name": "region"}},
    ]
    edges = [
        edge("P1", "A1", "ACTIVATES", 0.9),
        edge("P2", "A1", "ACTIVATES", 0.8),
        edge("P5", "A1", "ACTIVATES", 0.2),  # below activation threshold
        edge("A1", "A2", "RELATED_TO", 0.7),
        edge("A1", "A3", "RELATED_TO", 0.1),  # below relation threshold
        edge("P6", "A3", "ACTIVATES", 0.9),
        edge("P3", "A2", "ACTIVATES", 0.95),
        edge("P3", "P4", "SPECIALIZES"),
        edge("P4", "P3", "GENERALIZES"),
    ]
    return DualGraph({"nodes": nodes, "edges": edges})


def run(graph, **over):
    cfg = load_config(**over)
    return traverse(graph, [SeedMatch("P1", 1.0, {})], cfg), cfg


def test_transversal_path_pattern_attractor_attractor_pattern(toy):
    res, _ = run(toy)
    got = {r.node_id: r for r in res.patterns}
    p3 = got["P3"]
    assert p3.route == "transversal" and p3.transversal_only  # scope-disjoint from the seed
    assert [(s.source, s.target, s.edge_type, s.hop) for s in p3.path] == [
        ("P1", "A1", "ACTIVATES", 1),
        ("A1", "A2", "RELATED_TO", 2),
        ("A2", "P3", "ACTIVATES", 3),
    ]
    assert p3.path[2].reverse and not p3.path[0].reverse
    assert [s.weight for s in p3.path] == [0.9, 0.7, 0.95]
    assert p3.score == pytest.approx(1.0 * 0.9 * 0.7 * 0.95 * 0.8)  # path product x insight_weight
    assert "RELATED_TO" in p3.rationale and "A2" in p3.rationale
    # same-anchor neighbour, then lattice expansion after descending
    assert [s.edge_type for s in got["P2"].path] == ["ACTIVATES", "ACTIVATES"] and not got["P2"].transversal_only
    assert [s.edge_type for s in got["P4"].path] == ["ACTIVATES", "RELATED_TO", "ACTIVATES", "SPECIALIZES"]
    # thresholds prune weak bridges
    assert "P5" not in got and "P6" not in got
    assert {a.node_id for a in res.attractors} == {"A1", "A2"}
    assert set(res.used_edges) >= {e.edge_id for e in p3.path}
    assert res.patterns[0].node_id == "P1" and res.patterns[0].route == "seed"


def test_latent_hop_budget_and_depth(toy):
    res, _ = run(toy, max_latent_hops=0)
    assert "P3" not in {r.node_id for r in res.patterns}
    res, _ = run(toy, traversal_max_depth=3)
    assert "P4" not in {r.node_id for r in res.patterns}  # would need 4 hops


def test_query_parsing_against_graph_vocabulary(toy):
    q = parse_query("Why is margin lower for tablets in the EU?", toy)
    assert q.targets == ["margin"] and q.direction == -1
    assert ("region", "EU") in q.conditions and ("category", "tablets") in q.conditions
    assert ("region", "US") not in parse_query("tell us about margins", toy).conditions  # 'us' is not 'US'
    assert parse_query("where does margin go up?", toy).direction == 1


def test_evidence_object_is_structured_and_traceable(toy):
    res, cfg = run(toy)
    q = parse_query("Why is margin lower for phones in the EU?", toy)
    ev = build_evidence(q, res, toy, cfg)
    assert ev.items[0].key == "P1" and ev.items[0].role == "seed" and ev.seed_patterns == ["P1"]
    p3 = next(i for i in ev.items if i.pattern_id == "P3")
    assert p3.transversal_only and p3.path and "-ACTIVATES(0.90)-> A1" in p3.path_text
    for it in ev.items:
        assert {"dataset_id", "batch_id", "expression", "pattern_id"} <= set(it.provenance)
        assert {"support", "shifts", "p_adjusted", "stability", "weight"} <= set(it.statistics)
        assert {"metric", "robust_z", "local_median", "global_median"} <= set(it.statistics["shifts"][0])
        assert "insight_weight" not in it.statistics
    prompt = ev.to_prompt()
    assert "[P1] role=seed" in prompt and "LATENT ANCHORS VISITED" in prompt and "margin: median 12" in prompt
    assert any("latent anchors" in n for n in ev.notes)
    assert ev.key_to_pattern["P1"] == "P1"
