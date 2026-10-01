"""Deterministic structural plane (SDD 08)."""

from __future__ import annotations

import inspect

from ltir import structural
from ltir.config import load_config
from ltir.models import Condition, EdgeType, Insight, Shift


def ins(pid, scope, shifts, dataset="d"):
    conds = tuple(sorted(Condition(k, v) for k, v in scope.items()))
    sh = tuple(Shift(m, z, 0, 0, 1) for m, z in shifts)
    return Insight(
        id=pid,
        dataset_id=dataset,
        batch_id="b",
        conditions=conds,
        expression=pid,
        target=sh[0].metric,
        shifts=sh,
        support=100,
        support_fraction=0.1,
        baseline=0,
        local=0,
        effect_size=sh[0].robust_z,
        sd_score=1,
        sd_raw_score=1,
        emm_score=0,
        volume_utility=0.2,
        stability=1,
        p_value=0,
        p_adjusted=0,
        drivers=(),
        row_hash=pid,
    )


def edges_of(items, etype):
    return {(e.source, e.target) for e in structural.structural_edges(items, load_config()) if e.type == etype}


def test_specializes_uses_covering_relations_only():
    a = ins("A", {"r": "EU"}, [("m", 1.0)])
    b = ins("B", {"r": "EU", "c": "lap"}, [("m", 2.0)])
    c = ins("C", {"r": "EU", "c": "lap", "ch": "on"}, [("m", 3.0)])
    spec = edges_of([a, b, c], EdgeType.SPECIALIZES)
    assert spec == {("B", "A"), ("C", "B")}  # no transitive C->A
    assert edges_of([a, b, c], EdgeType.GENERALIZES) == {(t, s) for s, t in spec}  # exact inverse


def test_sibling_requires_same_parent_and_partition_attribute():
    a = ins("A", {"r": "EU", "c": "lap"}, [("m", 1.0)])
    b = ins("B", {"r": "EU", "c": "pho"}, [("m", 1.0)])
    c = ins("C", {"r": "US", "ch": "on"}, [("m", 1.0)])
    d = ins("D", {"r": "EU", "c": "tab"}, [("m", 1.0)], dataset="other")
    sib = edges_of([a, b, c, d], EdgeType.SIBLING)
    assert sib == {("A", "B")}
    e = next(e for e in structural.structural_edges([a, b], load_config()) if e.type == EdgeType.SIBLING)
    assert e.props["parent_scope"] == ["r=EU"] and e.props["partition_attribute"] == "c"


def test_contrasts_need_overlap_same_metric_and_opposite_shift():
    parent = ins("A", {"r": "EU", "c": "lap"}, [("margin", 1.1)])
    child = ins("B", {"r": "EU", "c": "lap", "ch": "ret"}, [("margin", -0.9)])
    same_dir = ins("C", {"r": "EU", "c": "pho"}, [("margin", 0.8)])
    disjoint = ins("D", {"r": "US", "c": "tv"}, [("margin", -2.0)])
    tiny = ins("E", {"r": "EU", "c": "tab"}, [("margin", -0.2)])
    items = [parent, child, same_dir, disjoint, tiny]
    con = edges_of(items, EdgeType.CONTRASTS)
    assert ("A", "B") in con  # specialisation reverses the effect
    assert ("B", "C") in con  # overlapping scope (r=EU), opposite margin shift
    assert not any("D" in pair for pair in con)  # no shared condition
    assert not any("E" in pair for pair in con)  # |z| below CONTRAST_MIN_SHIFT
    rel = {(e.source, e.target): e.props["relation"] for e in structural.structural_edges(items, load_config()) if e.type == EdgeType.CONTRASTS}
    assert rel[("A", "B")] == "specialization_reversal"


def test_contrast_relation_is_sibling_only_for_a_sibling_pair():
    a = ins("A", {"r": "EU", "c": "lap"}, [("m", 1.0)])
    sibling = ins("B", {"r": "EU", "c": "pho"}, [("m", -1.0)])  # one differing value of the same attribute
    other = ins("C", {"r": "EU", "ch": "on"}, [("m", -1.0)])  # same size, different attribute: not a sibling
    rel = {
        (e.source, e.target): e.props["relation"]
        for e in structural.structural_edges([a, sibling, other], load_config())
        if e.type == EdgeType.CONTRASTS
    }
    assert rel[("A", "B")] == "sibling" and rel[("A", "C")] == "overlap"


def test_structural_plane_never_uses_embeddings():
    assert list(inspect.signature(structural.structural_edges).parameters) == ["insights", "config"]
    src = inspect.getsource(structural)
    assert "import numpy" not in src and "encoder" not in src and "ontology" not in src
