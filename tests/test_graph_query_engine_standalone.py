"""graph_query_engine on its own: a snapshot and its vectors in, evidence and an LLM-ready payload out
(graph_query_engine/README.md). No engine, no workspace, no attractor_topology: a hand-written snapshot and an
encoder that only has the ``QueryEncoder`` shape."""

from __future__ import annotations

import json

import numpy as np
import pytest
from doubles import HashingEmbedder
from test_traversal import edge, pattern

from graph_query_engine import CommittedState, DualGraph, LatentFrame, QueryConfig, empty_payload, search
from insight_contracts import SNAPSHOT_VERSION, EvidencePayload

QUESTION = "Why is margin lower for phones in the US?"


class ShapeOnlyEncoder:
    """Anything with ``embedder`` and ``encode_query`` will do; here every question points at the margin patterns."""

    def __init__(self) -> None:
        self.embedder = HashingEmbedder()

    def encode_query(self, scope_text, target_text, components, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


VECTORS = {"P1": [1.0, 0.0, 0.0], "P2": [0.8, 0.6, 0.0]}  # P1 matches the question; P2 is reached through the anchor


def state_of(snap: dict, vectors: dict | None = None) -> CommittedState:
    """A hand-written snapshot's committed state: its graph and the vectors of the patterns it holds."""
    graph = DualGraph(snap)
    return CommittedState(graph, LatentFrame({p: np.array(v, dtype=float) for p, v in (vectors or VECTORS).items() if p in graph.insights}, {}))


def searched(state: CommittedState, question: str = QUESTION, config: QueryConfig | None = None, **kwargs):
    """The search over a committed state through the shape-only encoder."""
    return search(question, state, ShapeOnlyEncoder(), config or QueryConfig(), **kwargs)


def region(rid: str, **props) -> dict:
    """A Region node from its id ``R:<resolution>:<cell>``."""
    return {"id": rid, "kind": "Region", "label": rid, "props": {"resolution": int(rid.split(":")[1]), "h3": rid[-1], **props}}


def dataset(ds: str, label: str | None = None, **props) -> dict:
    return {"id": f"DS:{ds}", "kind": "Dataset", "label": label or ds, "props": {"dataset_id": ds, **props}}


def snapshot() -> dict:
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
    state = state_of(snapshot())
    found = searched(state)
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
    snap = snapshot()
    snap["nodes"] = [{**n, "props": {**n["props"], "provenance": {}}} if n["kind"] == "Pattern" else n for n in snap["nodes"]]
    payload = searched(state_of(snap)).payload()
    cited = {c["pattern_id"]: c for c in payload.citations.values()}
    assert cited["P1"] == {"pattern_id": "P1", "expression": "region=US AND category=phones", "dataset_id": "ds1", "filename": "", "batch_id": "B1"}
    assert payload.view["evidence"]["datasets"][0]["batch_id"] == "B1"


def test_a_partial_spatial_provenance_reads_without_the_missing_fields():
    snap = snapshot()
    spatial = {"resolution": 7, "cells": 40}  # an older record: no n_eff
    snap["nodes"] = [{**n, "props": {**n["props"], "provenance": {"spatial": spatial}}} if n["kind"] == "Pattern" else n for n in snap["nodes"]]
    prompt = searched(state_of(snap)).payload().evidence_prompt
    assert "place: 40 H3 cells (resolution 7)" in prompt and "effective n" not in prompt


def test_the_payload_contract_is_strict():
    payload = EvidencePayload("q?", evidence_prompt="QUESTION: q?", evidence_summary="- a=1 | m +1.00 sd | n=10 [P1]", metrics={"retrieval_s": 0.1})
    wire = payload.to_dict()
    assert EvidencePayload.from_dict(wire) == payload
    with pytest.raises(KeyError):
        EvidencePayload.from_dict({k: v for k, v in wire.items() if k != "metrics"})  # a missing field is a contract break


def test_a_snapshot_of_another_version_is_refused():
    with pytest.raises(ValueError, match="snapshot version"):
        DualGraph({**snapshot(), "version": 1})


def test_an_item_reached_only_by_location_is_marked():
    snap = snapshot()
    snap["edges"].append(edge("P1", "P2", "CO_LOCATED", 1.0, relation="same_region"))  # stronger than the walk through A-1
    found = searched(state_of(snap))
    item = next(it for it in found.evidence.items if it.pattern_id == "P2")
    prompt = found.payload().evidence_prompt
    assert item.located == "same_region" and "[reached by location only: same area, zones do not overlap]" in prompt
    assert any("reached only by location" in n for n in found.evidence.notes)


def test_a_location_step_after_the_latent_plane_wins_over_the_latent_mark():
    from graph_query_engine.evidence import LOCATION_TAG
    from graph_query_engine.traversal import LOCATED_RELATIONS

    snap = snapshot()
    snap["nodes"].append(pattern("P3", ["region=APAC", "category=laptops"]))
    snap["edges"].append(edge("P2", "P3", "CO_LOCATED", 1.0, relation="same_region"))  # P1 -> A-1 -> P2 -> P3
    found = searched(state_of(snap, VECTORS | {"P3": [0.0, 0.0, 1.0]}))
    item = next(it for it in found.evidence.items if it.pattern_id == "P3")
    assert [st["edge_type"] for st in item.path] == ["ACTIVATES", "ACTIVATES", "CO_LOCATED"]
    assert item.located == "same_region" and not item.transversal_only  # only the place links it to P2's phenomenon
    location, latent = (next(n for n in found.evidence.notes if mark in n) for mark in ("by location", "latent anchors"))
    assert item.key in location and item.key not in latent
    assert set(LOCATION_TAG) == LOCATED_RELATIONS  # every location-only relation has its tag
    assert "P3" not in found.payload().view["traversal"]["baselines"]["transversal_only"]  # the console reads the same mark


def _levels(
    coarse_z: float | None,
    planned: tuple[str, ...],
    *,
    upload: str | None = "up-1",
    coarse_regions: tuple[str, ...] = ("R:5:a",),
    coarse_first: bool = False,
) -> dict:
    """P1 at resolution 7 and, unless ``coarse_z`` is None, P7 at 5 of ``upload``: one anchor; P1 lies in R:5:a, P7 covers
    ``coarse_regions``."""

    def geo(pid, conds, res, ds, z):
        node = pattern(pid, conds, z=z, dataset=ds)
        node["props"]["provenance"]["spatial"] = {"resolution": res, "cells": 40}
        return node

    def level_of(ds, level):
        return dataset(ds, level=level, upload_levels=list(planned), **({"upload_id": upload} if upload else {}))

    signature = [{"component": "margin", "value": -1.0}]
    conds = [["region=US", "category=phones"], ["district=north"]][:: -1 if coarse_first else 1]  # the question names the first
    nodes = [
        geo("P1", conds[0], 7, "ds7", -1.5),
        level_of("ds7", "7"),
        {"id": "A-1", "kind": "Attractor", "label": "margin ↓", "props": {"n_patterns": 2, "distinct_scopes": 2, "signature": signature}},
        *(region(r) for r in ("R:5:a", "R:5:z", "R:3:b", "R:1:c")),
    ]
    edges = [
        edge("P1", "A-1", "ACTIVATES", 0.9),
        edge("P1", "R:5:a", "IN_REGION", 0.72),
        edge("R:5:a", "R:3:b", "PARENT_OF"),
        edge("R:5:z", "R:3:b", "PARENT_OF"),
        edge("R:3:b", "R:1:c", "PARENT_OF"),
    ]
    if coarse_z is not None:
        nodes += [geo("P7", conds[1], 5, "ds5", coarse_z), level_of("ds5", "5")]
        edges += [edge("P7", "A-1", "ACTIVATES", 0.8), *(edge("P7", r, "IN_REGION", 1.0) for r in coarse_regions)]
    return {"version": SNAPSHOT_VERSION, "nodes": nodes, "edges": edges}


def _levels_search(snap: dict, coarse_first: bool = False):
    vectors = {"P1": [1.0, 0.0, 0.0], "P7": [0.0, 1.0, 0.0]}
    if coarse_first:  # the coarser level matches the question, the finer one is reached through the anchor
        vectors = {"P1": vectors["P7"], "P7": vectors["P1"]}
    state = state_of(snap, vectors)
    return state.graph, searched(state)


def test_two_levels_of_one_upload_are_one_fact_with_a_scale():
    """The coarser level that shifts the same way is folded into the finer item (``also at``), labelled regional, and the
    chain runs up through the shared anchor and region to the regions above; a reversal is scale-dependent and stays a fact;
    a planned coarser level with nothing there is not_found_coarser; no coarser level planned prints no scale."""
    from graph_query_engine.vertical import picture, scale_chain, scale_label

    graph, found = _levels_search(_levels(-1.2, ("7", "5")))
    items = {it.pattern_id: it for it in found.evidence.items}
    prompt = found.payload().evidence_prompt
    p1, p7 = items["P1"], items["P7"]
    assert p1.scale == "regional" and p1.also == [p7.key] and p7.folded_into == p1.key and not p7.transversal_only
    assert f"also at res 5 [{p7.key}]" in prompt and f"[{p7.key}] role=" not in prompt  # one fact, still citable
    assert f"- res 7: {p1.key} -1.50 sd -> res 5: {p7.key} -1.20 sd -> region R:3:b" in prompt
    assert p7.key in found.payload().citations
    (chain,) = scale_chain(graph, ["P1"])  # what a zone from the map will call for the same pattern
    assert [s.pattern_id for s in chain.up] == ["P7"] and chain.regions == ["R:3:b", "R:1:c"]
    assert [scale_label(graph, p) for p in ("P1", "P7")] == [scale_chain(graph, [p])[0].scale for p in ("P1", "P7")]  # the map's label alone
    assert graph.facts("P7") is graph.facts("P7") and graph.memo  # what a step reads of a pattern: derived once per graph
    assert picture(graph, "R:5:a") == {"patterns": 2, "datasets": 2, "themes": 1}

    _, found = _levels_search(_levels(+1.2, ("7", "5")))
    p1 = next(it for it in found.evidence.items if it.pattern_id == "P1")
    assert p1.scale == "scale-dependent" and "[P2] role=" in found.payload().evidence_prompt  # a reversal is its own fact

    _, found = _levels_search(_levels(None, ("7", "5")))
    assert found.evidence.items[0].scale == "not_found_coarser" and any("not proof the effect is local" in n for n in found.evidence.notes)
    _, found = _levels_search(_levels(None, ("7",)))
    assert found.evidence.items[0].scale == "insufficient" and "scale:" not in found.payload().evidence_prompt


def test_the_scale_holds_for_a_broad_coarser_level_any_rank_and_no_upload(monkeypatch):
    """A coarser pattern over several regions still steps up from the one holding P; a coarser level ranked first keeps
    the group regional with its ``also at``; datasets of no upload never fold; the label does not depend on the budget."""
    import graph_query_engine.vertical as vertical

    _, found = _levels_search(_levels(-1.2, ("7", "5"), coarse_regions=("R:5:z", "R:5:a")))  # P7's finest regions: two
    assert next(it for it in found.evidence.items if it.pattern_id == "P1").scale == "regional"

    _, found = _levels_search(_levels(-1.2, ("7", "5"), coarse_first=True), coarse_first=True)
    items = {it.pattern_id: it for it in found.evidence.items}
    assert items["P7"].role == "seed" and items["P7"].scale == "regional" and items["P1"].folded_into == items["P7"].key
    assert f"also at res 7 [{items['P1'].key}]" in found.payload().evidence_prompt

    _, found = _levels_search(_levels(-1.2, ("7", "5"), upload=None))
    assert not any(it.folded_into for it in found.evidence.items) and "[P2] role=" in found.payload().evidence_prompt

    monkeypatch.setattr(vertical, "VERTICAL_MAX_TOTAL", 0)
    graph, _ = _levels_search(_levels(-1.2, ("7", "5")))
    (chain,) = vertical.scale_chain(graph, ["P1"])
    assert not chain.up and chain.scale == "regional"  # no step spent, the label still searched


def _held_by(edges: list[dict], rids: list[str]) -> list[dict]:
    """Region nodes counting their IN_REGION members, as the snapshot does."""
    return [region(r, patterns=sum(e["type"] == "IN_REGION" and e["target"] == r for e in edges)) for r in rids]


def test_the_console_draws_a_patterns_places_and_the_regions_its_scale_chain_names():
    """``shown_regions`` is one bound: a pattern's places at every level it lies at (its five strongest there, a geo edge
    by share, then its cells there) and the regions its scale chain names (with no coarser step, from its home region
    up), each holding an intact pattern; so every region the ZONES name, at whatever level, is drawn."""
    from graph_query_engine.vertical import places, rank_zones, scale_chain, shown_regions

    def geo(pid: str) -> dict:
        node = pattern(pid, [f"district={pid}"])
        node["props"]["provenance"]["spatial"] = {"resolution": 8, "cells": 40}
        return node

    edges = [edge("P1", r, "IN_REGION", 1.0, basis="cells", share=1.0, n=n) for r, n in (("R:7:a", 3), ("R:7:b", 9), ("R:5:c", 12))]
    edges += [edge("R:7:a", "R:5:c", "PARENT_OF"), edge("R:7:b", "R:5:c", "PARENT_OF"), edge("R:5:c", "R:3:d", "PARENT_OF")]
    rids = ["R:7:a", "R:7:b", "R:5:c", "R:3:d"]
    graph = DualGraph({"version": SNAPSHOT_VERSION, "nodes": [geo("P1"), *_held_by(edges, rids)], "edges": edges})
    assert places(graph, "P1") == [("R:7:b", 1.0), ("R:7:a", 1.0)]  # two regions at 7, one at 5: its finest; more cells first
    assert scale_chain(graph, ["P1"])[0].regions == ["R:7:a", "R:5:c", "R:3:d"]
    assert shown_regions(graph, ["P1"]) == {"R:7:a", "R:7:b", "R:5:c"}  # R:3:d holds no pattern
    at_seven = geo("P7")
    at_seven["props"]["provenance"]["spatial"]["resolution"] = 7  # its own level is a region level: its cells there are no places
    own = [edge("P7", r, "IN_REGION", 1.0, basis="cells", share=1.0, n=1) for r in ("R:7:a", "R:7:b")] + [
        edge("P7", "R:5:c", "IN_REGION", 1.0, basis="cells", share=1.0, n=2)
    ]
    assert places(DualGraph({"version": SNAPSHOT_VERSION, "nodes": [at_seven, *_held_by(own, rids)], "edges": own}), "P7") == [("R:5:c", 1.0)]
    forged = geo("P2")
    forged["props"]["integrity"] = "mismatch"
    edges += [
        edge("P2", "R:7:b", "IN_REGION", 1.0, basis="cells", share=1.0, n=1),
        edge("P2", "R:5:e", "IN_REGION", 1.0, basis="cells", share=1.0, n=1),
    ]
    tampered = DualGraph({"version": SNAPSHOT_VERSION, "nodes": [geo("P1"), forged, *_held_by(edges, [*rids, "R:5:e"])], "edges": edges})
    assert shown_regions(tampered, ["P1", "P2"]) == {"R:7:a", "R:7:b", "R:5:c"}  # R:5:e holds the tampered P2 alone
    six = [(f"R:5:{c}", n) for c, n in zip("abcdef", (6, 5, 4, 3, 2, 1), strict=True)]  # P: six regions at 5, two at 3
    zoned = [edge("P", r, "IN_REGION", 1.0, basis="cells", share=1.0, n=n) for r, n in six]
    zoned += [
        edge("P", "R:3:x", "IN_REGION", 1.0, basis="cells", share=1.0, n=5),
        edge("P", "R:3:y", "IN_REGION", 0.6, basis="cells", share=0.6, n=1),
    ]
    zoned += [edge("Q", r, "IN_REGION", 1.0, basis="cells", share=1.0, n=2) for r in ("R:3:z", "R:3:w")]
    zoned += [edge(f"R:5:{c}", "R:3:x", "PARENT_OF") for c in "abcde"] + [edge("R:5:f", "R:3:y", "PARENT_OF")]
    level = DualGraph(
        {
            "version": SNAPSHOT_VERSION,
            "nodes": [geo("P"), geo("Q"), *_held_by(zoned, [r for r, _ in six] + ["R:3:x", "R:3:y", "R:3:z", "R:3:w"])],
            "edges": zoned,
        }
    )
    zones = rank_zones(level, ["P", "Q"], ["P", "Q"])  # four regions at 3: the zones' level
    assert {z["resolution"] for z in zones} == {3} and "R:3:y" in {z["id"] for z in zones}
    assert places(level, "P", 5) == [(r, 1.0) for r, _ in six[:5]]  # R:5:f, its weakest at 5, the one under R:3:y, is no place
    assert {z["id"] for z in zones} <= shown_regions(level, ["P", "Q"])  # R:3:y too, though no place of P's at 5 lies under it


def test_a_reversal_above_a_coarser_host_keeps_the_scale_dependent_label():
    """Three levels of one upload, the middle one ranked first: its finer level folds in, but a reversal above it stays
    scale-dependent rather than being lifted to regional."""
    snap = _levels(-1.2, ("7", "5", "3"), coarse_first=True)
    top = pattern("P3", ["zone=east"], z=1.0, dataset="ds3")
    top["props"]["provenance"]["spatial"] = {"resolution": 3, "cells": 40}
    snap["nodes"] += [top, dataset("ds3", upload_id="up-1", level="3", upload_levels=["7", "5", "3"])]
    snap["edges"] += [edge("P3", "A-1", "ACTIVATES", 0.7), edge("P3", "R:3:b", "IN_REGION", 1.0), edge("P7", "R:3:b", "IN_REGION", 1.0)]
    _, found = _levels_search(snap, coarse_first=True)
    items = {it.pattern_id: it for it in found.evidence.items}
    assert items["P7"].scale == "scale-dependent" and items["P1"].folded_into == items["P7"].key
    assert "scale: scale-dependent" in found.payload().evidence_prompt


def _upload_graph(patterns: list[tuple[str, str, int, float, float]], in_region: list[tuple[str, str, float]]):
    """One upload's geo patterns, every one a member of A-1: ``patterns`` as (id, scope, resolution, z, alignment),
    ``in_region`` as (pattern, region, share)."""
    levels = sorted({str(res) for _, _, res, _, _ in patterns}, key=lambda lv: -int(lv))
    nodes = [{"id": "A-1", "kind": "Attractor", "label": "margin ↓", "props": {"n_patterns": len(patterns), "distinct_scopes": 2, "signature": []}}]
    for pid, scope, res, z, _ in patterns:
        nodes.append(node := pattern(pid, [scope], z=z, dataset=f"ds{res}"))
        node["props"]["provenance"]["spatial"] = {"resolution": res, "cells": 40}
    nodes += [dataset(f"ds{lv}", lv, upload_id="up-1", level=lv, upload_levels=levels) for lv in levels]
    nodes += [region(r) for r in sorted({r for _, r, _ in in_region})]
    edges = [edge(pid, "A-1", "ACTIVATES", w) for pid, _, _, _, w in patterns] + [edge(p, r, "IN_REGION", s) for p, r, s in in_region]
    return DualGraph({"version": SNAPSHOT_VERSION, "nodes": nodes, "edges": edges})


def test_a_step_takes_the_nearest_level_then_the_own_subgroup():
    """Coarser patterns of one upload compete by level before alignment: a reversal at the next level is scale-dependent
    even when a farther level corresponds more strongly, and the chain lists both levels in order; at one level a
    pattern's own subgroup comes before another pattern of the same region that aligns more strongly."""
    from graph_query_engine.vertical import scale_chain

    def three_levels(z5: float, z3: float):  # P at 7; its level-5 pattern aligns weakly, its level-3 pattern strongly
        patterns = [("P", "a=1", 7, -1.5, 0.9), ("Q5", "a=1", 5, z5, 0.4), ("Q3", "a=1", 3, z3, 0.95)]
        return _upload_graph(patterns, [("P", "R:5:x", 0.8), ("P", "R:3:y", 0.9), ("Q5", "R:5:x", 1.0), ("Q5", "R:3:y", 1.0), ("Q3", "R:3:y", 1.0)])

    (chain,) = scale_chain(three_levels(+1.0, -1.0), ["P"])
    assert [(s.pattern_id, s.relation) for s in chain.up] == [("Q5", "contrasts"), ("Q3", "contrasts")] and chain.scale == "scale-dependent"
    (chain,) = scale_chain(three_levels(-1.0, +1.0), ["P"])
    assert [s.pattern_id for s in chain.up] == ["Q5", "Q3"] and chain.scale == "regional"

    patterns = [
        ("A", "lisa=hot", 7, -1.5, 0.9),
        ("B", "ocean=bay", 7, -1.5, 0.85),
        ("A5", "lisa=hot", 5, -1.2, 0.95),
        ("B5", "ocean=bay", 5, -1.2, 0.5),
    ]
    graph = _upload_graph(patterns, [(p, "R:5:x", 0.9 if len(p) == 1 else 1.0) for p, *_ in patterns])
    assert {c.pattern_id: [s.pattern_id for s in c.up] for c in scale_chain(graph, ["A", "B"])} == {"A": ["A5"], "B": ["B5"]}


def test_the_vertical_budget_counts_distinct_patterns(monkeypatch):
    """``VERTICAL_MAX_TOTAL`` caps the distinct vertical patterns of a question: one already in a chain costs nothing
    again, a new one past the cap stays out of the chain, and the label still compares with it."""
    import graph_query_engine.vertical as vertical

    monkeypatch.setattr(vertical, "VERTICAL_MAX_TOTAL", 1)
    patterns = [("A", "lisa=hot", 7, -1.5, 0.9), ("B", "lisa=hot", 7, -1.5, 0.8), ("C", "ocean=bay", 7, -1.5, 0.8)]
    patterns += [("A5", "lisa=hot", 5, -1.2, 0.9), ("C5", "ocean=bay", 5, 1.2, 0.9)]
    graph = _upload_graph(patterns, [(p, "R:5:x", 0.9 if len(p) == 1 else 1.0) for p, *_ in patterns])
    chains = {c.pattern_id: c for c in vertical.scale_chain(graph, ["A", "B", "C"])}
    assert [s.pattern_id for s in chains["A"].up] == [s.pattern_id for s in chains["B"].up] == ["A5"]
    assert not chains["C"].up and chains["C"].scale == "scale-dependent"  # C5 would be a second pattern


def test_zones_rank_the_regions_holding_the_findings_the_asked_way():
    """Only findings that shift an asked target the asked way count: one shifting it the other way, or not at all, is
    neither scored nor listed, even in a region shared with answering ones; a geo finding lies in a region at a share of
    0.5 (D's 0.3 in R:5:x is no place) and adds its weight to its places in proportion to their shares; the level is the
    coarsest where the geo findings lie in three or more regions; a relationship question counts the correlation changes of its pair; without
    a target every geo finding counts; a question with scope gets none."""
    from types import SimpleNamespace

    from graph_query_engine.evidence import _zones
    from graph_query_engine.question import ParsedQuery

    def geo(pid, z, w, target="margin"):
        node = pattern(pid, [f"cell={pid}"], target=target, z=z, w=w)
        node["props"]["provenance"]["spatial"] = {"resolution": 7, "cells": 40}
        return node

    regions = [region(r) for r in ("R:5:x", "R:5:y", "R:5:z", "R:5:v", "R:3:w")]
    nodes = [
        geo("A", -1.0, 0.8),
        geo("B", -1.0, 0.6),
        geo("C", 1.0, 0.9),
        geo("D", -1.0, 0.5),
        geo("E", -1.0, 0.9, "discount"),
        geo("F", -1.0, 0.4),
        *regions,
    ]
    shares = (
        ("A", "R:5:x", 0.9),
        ("B", "R:5:x", 0.8),
        ("C", "R:5:x", 0.9),
        ("E", "R:5:y", 0.9),
        ("D", "R:5:z", 0.6),
        ("D", "R:5:x", 0.3),
        ("F", "R:5:v", 0.9),
    )
    edges = [edge(p, r, "IN_REGION", s) for p, r, s in shares] + [edge(p, "R:3:w", "IN_REGION", 1.0) for p in "ABCDEF"]
    graph = DualGraph({"version": SNAPSHOT_VERSION, "nodes": nodes, "edges": edges})
    items = [SimpleNamespace(pattern_id=p, folded_into="") for p in "ABCDEF"]
    zones = _zones(ParsedQuery("where is margin lower?", targets=["margin"], direction=-1), items, graph, QueryConfig())
    assert [(z["id"], z["holds"]) for z in zones] == [("R:5:x", ["A", "B"]), ("R:5:z", ["D"]), ("R:5:v", ["F"])]  # C, E and D's 0.3 out
    assert [z["score"] for z in zones] == [round(0.8 + 0.6, 4), 0.5, 0.4] and zones[0]["resolution"] == 5  # all its weight in its one place
    every = _zones(ParsedQuery("what is on the map?"), items, graph, QueryConfig())
    assert ["A", "B", "C"] == every[0]["holds"] and any(z["holds"] == ["E"] for z in every)
    graph.nodes["E"]["props"] |= {
        "phenomenon_type": "covariance",
        "covariance": {"pair": ["margin", "discount"], "local_corr": 0.1, "global_corr": -0.6, "delta": 0.7},
    }
    graph = DualGraph(graph.snapshot)  # E now a correlation change between margin and discount
    related = _zones(
        ParsedQuery("where does margin follow discount less?", targets=["margin", "discount"], covariance=True), items, graph, QueryConfig()
    )
    assert [z["holds"] for z in related] == [["E"]]  # the mean shifts of A-D do not answer a relationship question

    _, found = _levels_search(_levels(-1.2, ("7", "5")))
    assert not found.evidence.zones  # the question names phones and the US
    found = searched(state_of(_levels(-1.2, ("7", "5")), {"P1": [1.0, 0.0, 0.0], "P7": [0.8, 0.6, 0.0]}), "Where is margin lower?")
    assert [z["id"] for z in found.evidence.zones] == ["R:5:a"] and "- Z1 = R:5:a (res 5): contains P1" in found.payload().evidence_prompt


def test_a_zone_alone_seeds_from_inside_it_even_below_the_floor():
    """Without text a zone seeds only the patterns it holds: the best of them when none reaches SEED_MIN_SCORE, none
    when it holds none; the AREA line names in and out, never a pattern the zone did not weigh."""
    snap = snapshot()
    snap["nodes"][:2] = [pattern("P1", ["region=US", "category=phones"], w=0.2), pattern("P2", ["region=EU", "category=tablets"], w=0.2)]
    state = state_of(snap)
    found = searched(state, "", place_scores={"P2": 0.01, "P1": 0.0})
    assert [(s.pattern_id, s.matched.get("fallback")) for s in found.seeds] == [("P2", "place")]
    assert set(found.traversal.baselines) == {"structural_only", "transversal_only", "not_structurally_reachable"}
    keys = {it.pattern_id: it.key for it in found.evidence.items}
    assert (
        f"in it, with how closely each one's map cells match it (100% = the same cells): {keys['P2']} 1%; outside it: {keys['P1']}."
        in found.payload().evidence_prompt
    )
    prompt = found.payload().evidence_prompt
    assert f"[{keys['P2']}] role=seed | in the asked area | scope:" in prompt and f"[{keys['P1']}] role=" in prompt
    assert f"[{keys['P1']}] role={found.evidence.items[1].role} | outside the asked area" in prompt
    assert "(у виділеній зоні)" in found.payload().evidence_summary and "(поза виділеною зоною)" in found.payload().evidence_summary
    assert not searched(state, "", place_scores={"P1": 0.0}).seeds
    unweighed = searched(state, "", place_scores={"P2": 0.004}).payload().evidence_prompt  # P1 not weighed
    assert f"{keys['P2']} <1%" in unweighed  # in the zone, however little
    assert "outside it" not in next(line for line in unweighed.splitlines() if line.startswith("AREA:"))


def test_a_zone_keeps_a_finer_pattern_inside_it_among_the_seeds():
    """A lattice neighbour of a chosen seed is no duplicate when more of it lies in the asked zone; without a zone the
    diversity rule still skips it, and the walk reaches it as structural."""
    snap = snapshot()
    snap["nodes"][:2] = [
        pattern("O", ["zone=north"], w=0.95),
        pattern("I", ["zone=north", "shift=night"], w=0.3),
        pattern("Y", ["zone=south"], w=0.5),
    ]
    snap["edges"] = [edge("I", "O", "SPECIALIZES"), *(edge(p, "A-1", "ACTIVATES", 0.9) for p in "OIY")]
    state = state_of(snap, {p: [1.0, 0.0, 0.0] for p in "OIY"})

    def roles(place_scores):
        found = searched(state, place_scores=place_scores)
        return [s.pattern_id for s in found.seeds], {it.pattern_id: it.role for it in found.evidence.items}["I"]

    assert roles(None) == (["O", "Y"], "structural")
    assert roles({"I": 0.1, "O": 0.0, "Y": 0.0}) == (["O", "I", "Y"], "seed")


def test_a_tampered_pattern_takes_no_place_in_the_walk_or_a_region():
    """A tampered pattern the walk reaches takes no slot of its budget: the intact one below it is retrieved instead, the
    tampered one is named in a NOTE and in the evidence-only summary, and a region's picture does not count it."""
    from graph_query_engine.vertical import picture

    snap = snapshot()
    snap["nodes"].append(pattern("P3", ["region=EU", "category=laptops"], w=0.3))
    snap["nodes"].append(region("R:5:a"))
    snap["nodes"][1]["props"]["integrity"] = "mismatch"  # P2 (weight 0.8) ranks above P3 (0.3)
    snap["edges"] += [edge("P3", "A-1", "ACTIVATES", 0.8), edge("P2", "R:5:a", "IN_REGION", 0.9), edge("P3", "R:5:a", "IN_REGION", 0.9)]
    state = state_of(snap, VECTORS | {"P3": [0.8, 0.6, 0.0]})
    graph, found = state.graph, searched(state, config=QueryConfig(max_retrieved=1))
    assert [r.node_id for r in found.traversal.patterns] == ["P1", "P3"] and [r.node_id for r in found.traversal.excluded] == ["P2"]
    assert found.evidence.excluded == ["P2"] and any(n.startswith("P2 excluded: integrity check failed") for n in found.evidence.notes)
    assert "виключено, перевірку цілісності не пройдено (статистика не збігається з дайджестом запису): P2" in found.payload().evidence_summary
    assert picture(graph, "R:5:a")["patterns"] == 1


def test_two_links_to_one_anchor_are_one_entry_and_a_repeated_description_is_explained():
    """Anchors joined by both CO_OCCURS and RELATED_TO list each other once with both links; the second anchor of a
    description already shown names the first and how many members they share, or that it shares none."""
    snap = snapshot()
    snap["nodes"].append({"id": "A-2", "kind": "Attractor", "label": "margin ↓", "props": dict(snap["nodes"][2]["props"])})
    snap["edges"] = [
        edge("P1", "A-1", "ACTIVATES", 0.9),
        edge("P2", "A-2", "ACTIVATES", 0.8),
        edge("A-1", "A-2", "CO_OCCURS", 1.0, kind="co_occurrence", shared=2),
        edge("A-1", "A-2", "RELATED_TO", 0.9, kind="mutual_knn"),
    ]

    def anchors_and_prompt(edges):
        found = searched(state_of(snap | {"edges": edges}))
        return {a["id"]: a for a in found.evidence.attractors}, found.payload().evidence_prompt

    anchors, prompt = anchors_and_prompt(snap["edges"])
    assert anchors["A-1"]["related"] == [{"id": "A-2", "links": {"CO_OCCURS": 1.0, "RELATED_TO": 0.9}, "shared": 2}]
    assert anchors["A-2"].get("same_as") == "A-1" and "same_as" not in anchors["A-1"]
    assert "related: A-2 (2 shared members, 1.00; centroid 0.90)" in prompt
    assert '"margin down" (same description as A-1; shares 2 members with it)' in prompt
    _, prompt = anchors_and_prompt([e for e in snap["edges"] if e["type"] != "CO_OCCURS"])  # one description, no shared member
    assert '"margin down" (same description as A-1; a separate anchor with no shared members)' in prompt


def test_the_evidence_carries_each_pattern_credibility():
    """An evidence item copies the credibility and its basis from its Pattern node (the view; the prompt does not show it)."""
    snap = snapshot()
    snap["nodes"][0]["props"] |= {"credibility": 1, "credibility_basis": {"rule": "anchor", "patterns": ["P9"], "sources": ["b"]}}
    found = searched(state_of(snap))
    p1 = next(it for it in found.evidence.items if it.pattern_id == "P1")
    assert (p1.credibility, p1.credibility_basis["patterns"]) == (1, ["P9"]) and "credibility" not in found.payload().evidence_prompt


def test_the_evidence_shows_trust_once_the_graph_has_a_declared_source():
    """One undeclared upload, also at two levels, keeps the evidence as it was; a declared source or reliability adds to
    every item a trust line (reliability, credibility with its basis by key, integrity, which leaves the validation line),
    the header phrase, the summary's tag and a zone's weakest level."""
    from graph_query_engine.evidence import _credibility_text

    def found(dataset_props, *more):
        snap = snapshot()
        snap["nodes"][0]["props"] |= {"credibility": 1, "credibility_basis": {"rule": "place", "patterns": ["P2", "P9"], "sources": ["b"]}}
        snap["nodes"][0]["props"]["integrity"] = "ok"
        snap["nodes"][1]["props"] |= {"credibility": 6, "credibility_basis": {"rule": "place", "patterns": [], "sources": []}}
        snap["nodes"][1]["props"]["integrity"] = "unverified"
        for i, props in enumerate([dataset_props, *more], start=1):
            snap["nodes"].append(dataset(f"ds{i}", **props))
        return searched(state_of(snap))

    plain = found({"reliability": "F"})
    assert all(not it.trust for it in plain.evidence.items) and "trust" not in plain.payload().evidence_prompt
    levels = found({"reliability": "F", "upload_id": "up-1"}, {"reliability": "F", "upload_id": "up-1"})  # one upload at two levels
    assert all(not it.trust for it in levels.evidence.items) and "integrity unverified" in levels.payload().evidence_prompt
    assert all(it.trust for it in found({"reliability": "E"}).evidence.items)  # a rating alone is a declaration
    result = found({"source": "a", "reliability": "E"})
    items = {it.pattern_id: it for it in result.evidence.items}
    key = items["P2"].key
    assert items["P1"].trust == (
        f'source E (unreliable) | credibility 1 (confirmed by another source at the same place: "b"; [{key}], 1 finding not in this evidence)'
        " | integrity ok"
    )
    assert items["P2"].trust == "source E (unreliable) | credibility 6 (no independent source covers this) | integrity unverified"
    assert "integrity unverified" not in items["P2"].validation  # said once, on the trust line
    prompt, summary = result.payload().evidence_prompt, result.payload().evidence_summary
    assert "trust is given per item;" in prompt and f"     trust: {items['P1'].trust}" in prompt
    assert "(джерело E, підтверджено іншим джерелом)" in summary and "(джерело E, немає іншого джерела для перевірки)" in summary
    assert result.evidence._weakest({"holds": ["P1", "P2"]}) == "; weakest credibility 6"
    items["P1"].credibility = 9  # a snapshot edited on disk: listed, not a crash
    assert result.evidence._weakest({"holds": ["P1", "P2"]}) == "; weakest credibility 9"
    contra = {"rule": "anchor", "patterns": ["X"], "sources": ["b"], "confirmed_by": ["P1"]}
    assert (
        _credibility_text(5, contra, {"P1": "P1"})
        == 'credibility 5 (contradicted by another source: "b"; 1 finding not in this evidence; confirmed by [P1])'
    )
    assert (
        _credibility_text(2, {"rule": "anchor", "patterns": [], "sources": ["c"]}, {})
        == 'credibility 2 (not confirmed; another dataset has the same metric: "c")'
    )


def test_a_theme_is_the_strongest_walkable_membership():
    """``primary_anchor``: the theme an insight is shown and walked under skips a weak (coverage-only) membership, however
    strong; an insight with weak memberships only has no theme."""
    from graph_query_engine.graph import primary_anchor

    snap = snapshot()
    snap["nodes"].append({"id": "A-2", "kind": "Attractor", "label": "other", "props": {"n_patterns": 1}})
    strong_weak = edge("P1", "A-2", "ACTIVATES", 0.95)
    strong_weak["props"]["weak"] = True
    snap["edges"] = [strong_weak, edge("P1", "A-1", "ACTIVATES", 0.5), {**edge("P2", "A-1", "ACTIVATES", 0.9), "props": {"weak": True}}]
    graph = DualGraph(snap)
    assert primary_anchor(graph, "P1")["target"] == "A-1" and primary_anchor(graph, "P2") is None


def test_a_memoised_fact_is_built_once_however_many_readers_ask():
    """``DualGraph.memoised``: readers on several threads asking for one key at once share one build (a projection of a
    large upload takes seconds); a new commit's graph builds its own."""
    import threading
    import time

    graph, builds = DualGraph(snapshot()), []

    def build():
        builds.append(1)
        time.sleep(0.05)
        return len(builds)

    readers = [threading.Thread(target=graph.memoised, args=(("projection", "ds"), build)) for _ in range(8)]
    for r in readers:
        r.start()
    for r in readers:
        r.join()
    assert builds == [1] and graph.memoised(("projection", "ds"), build) == 1
    assert DualGraph(snapshot()).memoised(("projection", "ds"), build) == 2
