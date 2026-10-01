"""Deterministic structural plane: SPECIALIZES / GENERALIZES / SIBLING / CONTRASTS (SDD 08).

Derived only from scope conditions and signed shifts — never from embeddings.
Relations are computed within one dataset (scopes of different schemas are
incomparable).
"""

from __future__ import annotations

from itertools import combinations

from ltir.config import Config
from ltir.models import EdgeType, GraphEdge, Insight


def _conds(ins: Insight) -> frozenset:
    return frozenset(ins.conditions)


def specialization_pairs(insights: list[Insight]) -> list[tuple[Insight, Insight]]:
    """(child, parent) covering pairs: child ⊃ parent with no present pattern strictly between."""
    pairs: list[tuple[Insight, Insight]] = []
    sets = {i.id: _conds(i) for i in insights}
    for child in insights:
        parents = [p for p in insights if p.id != child.id and sets[p.id] < sets[child.id]]
        for parent in parents:
            between = any(sets[parent.id] < sets[m.id] < sets[child.id] for m in parents)
            if not between:
                pairs.append((child, parent))
    return pairs


def _contrast(a: Insight, b: Insight, config: Config) -> tuple[str, float, float] | None:
    """Shared metric with opposite, material shifts (largest min(|z_a|, |z_b|))."""
    best = None
    for sa in a.shifts:
        sb = b.shift_for(sa.metric)
        if sb is None or sa.direction * sb.direction >= 0:
            continue
        strength = min(sa.magnitude, sb.magnitude)
        if strength >= config.contrast_min_shift and (best is None or strength > best[0]):
            best = (strength, sa.metric, sa.robust_z, sb.robust_z)
    return None if best is None else best[1:]


def structural_edges(insights: list[Insight], config: Config) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    by_dataset: dict[str, list[Insight]] = {}
    for ins in insights:
        by_dataset.setdefault(ins.dataset_id, []).append(ins)

    for group in by_dataset.values():
        spec = specialization_pairs(group)
        spec_keys = {(c.id, p.id) for c, p in spec}
        for child, parent in spec:
            added = sorted(c.expr for c in _conds(child) - _conds(parent))
            child_shift = child.shift_for(parent.target)
            props = {
                "added_conditions": added,
                "support_ratio": child.support / max(parent.support, 1),
                "metric": parent.target,
                "parent_z": parent.effect_size,
                "child_z": child_shift.robust_z if child_shift else None,
            }
            edges.append(GraphEdge(child.id, parent.id, EdgeType.SPECIALIZES, 1.0, props))
            edges.append(GraphEdge(parent.id, child.id, EdgeType.GENERALIZES, 1.0, dict(props)))

        for a, b in combinations(sorted(group, key=lambda i: i.id), 2):
            ca, cb = _conds(a), _conds(b)
            only_a, only_b = ca - cb, cb - ca
            # siblings: same parent scope, one differing value of the same attribute
            sibling = len(only_a) == len(only_b) == 1 and next(iter(only_a)).attribute == next(iter(only_b)).attribute
            if sibling:
                edges.append(
                    GraphEdge(
                        a.id,
                        b.id,
                        EdgeType.SIBLING,
                        1.0,
                        {
                            "parent_scope": sorted(c.expr for c in ca & cb),
                            "partition_attribute": next(iter(only_a)).attribute,
                            "values": [next(iter(only_a)).value, next(iter(only_b)).value],
                        },
                    )
                )
            overlap = len(ca & cb) / max(min(len(ca), len(cb)), 1)
            if overlap < config.contrast_min_overlap:
                continue
            found = _contrast(a, b, config)
            if found is None:
                continue
            metric, za, zb = found
            if (a.id, b.id) in spec_keys or (b.id, a.id) in spec_keys:
                relation = "specialization_reversal"
            elif sibling:
                relation = "sibling"
            else:
                relation = "overlap"
            edges.append(
                GraphEdge(
                    a.id,
                    b.id,
                    EdgeType.CONTRASTS,
                    float(overlap),
                    {"metric": metric, "z_source": za, "z_target": zb, "scope_overlap": overlap, "relation": relation},
                )
            )
    return edges
