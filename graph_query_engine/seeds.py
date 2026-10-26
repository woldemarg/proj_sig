"""Seed resolution (docs/07_question_answering.md §7.2): every pattern scored against the parsed question
and the query vector, then the diverse top seeds the transversal walk starts from."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from graph_query_engine.config import QueryConfig
from graph_query_engine.graph import DualGraph
from graph_query_engine.ports import QueryEncoder
from graph_query_engine.question import ParsedQuery
from insight_contracts import Insight
from insight_contracts.text import humanize

SEED_TOP_K = 3
SEED_MIN_SCORE = 0.25
SEED_RELATIVE_MIN = 0.75  # seeds must score >= this fraction of the best seed
SEED_PLACE_WEIGHT = 0.35  # a map zone: the weight of a pattern's overlap with it (docs/07_question_answering.md §7.2)


def trajectory(graph: DualGraph, pid: str) -> str | None:
    """The trajectory a time window's pattern is a node of (docs/02_discovery.md §2.10); None for any other pattern."""
    return (graph.insight(pid).provenance.get("temporal") or {}).get("trajectory")


def representatives(graph: DualGraph, scored: Iterable[tuple[str, float]]) -> dict[str, str]:
    """Trajectory key -> the pattern that stands for it among ``scored`` (id, score): its latest window when that one
    scores at least the seed floor of the trajectory's best, else its best (docs/07_question_answering.md §7.2)."""
    windows: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for pid, score in scored:
        if key := trajectory(graph, pid):
            windows[key].append((pid, score))
    out = {}
    for key, found in windows.items():
        best = max(found, key=lambda f: f[1])
        latest = max(found, key=lambda f: graph.insight(f[0]).provenance["temporal"]["window"])
        out[key] = latest[0] if latest[1] >= max(SEED_MIN_SCORE, SEED_RELATIVE_MIN * best[1]) else best[0]
    return out


@dataclass
class SeedMatch:
    pattern_id: str
    score: float
    matched: dict[str, Any]


def score_pattern(
    ins: Insight, query: ParsedQuery, qvec: np.ndarray | None, vec: np.ndarray | None, config: QueryConfig, place: float | None = None
) -> SeedMatch:
    """Seed score of one pattern: lexical match (target, scope, direction) + semantic cosine + weight, plus
    ``SEED_PLACE_WEIGHT`` × how much it and the asked map zone are one place (``place``; None: no zone); a zone without
    text: weight + place, 0 outside the zone."""
    pair = set(ins.covariance.get("pair", []))
    wanted_targets = set(query.targets)
    material = [s.robust_z for t in query.targets for s in ins.shifts if s.metric == t and config.thresholds.material(s)]
    if ins.target in wanted_targets:
        target = 1.0
    elif material:
        target = 0.7
    elif pair & wanted_targets:
        target = 0.6
    else:
        target = 0.0
    if query.covariance:  # relationship questions: the phenomenon slot scores covariance insights
        direction = 1.0 if ins.phenomenon_type == "covariance" and (not query.targets or pair & wanted_targets) else 0.0
    elif query.direction and material:
        direction = 1.0 if np.sign(material[0]) == query.direction else -0.5
    else:
        direction = 0.0
    wanted = set(query.conditions)
    exact = {(a, v) for a, v in wanted if a != "*"}
    wanted_attrs = {a for a, _ in exact}
    conds = {(c.attribute, c.value) for c in ins.conditions}
    hits = len(conds & exact) + sum(any(v == cv for _, cv in conds) for a, v in wanted if a == "*")  # a wildcard matches any column
    conflicts = len({a for a, v in conds if a in wanted_attrs and (a, v) not in exact})
    scope = (hits / len(wanted) if wanted else 0.0) - 0.5 * conflicts
    semantic = float(vec @ qvec) if vec is not None and qvec is not None else 0.0
    if query.is_lexical:
        score = 0.35 * target + 0.25 * scope + 0.15 * direction + 0.15 * semantic + 0.10 * ins.weight
    else:
        score = 0.7 * semantic + 0.3 * ins.weight
    if place is not None and query.text.strip():
        score += SEED_PLACE_WEIGHT * place
    elif place is not None:
        score = ins.weight + place if place else 0.0
    return SeedMatch(
        ins.id,
        float(score),
        {
            "target": target,
            "scope": round(scope, 3),
            "direction": direction,
            "semantic": round(semantic, 3),
            "weight": round(ins.weight, 3),
            "scope_conflicts": conflicts,
        }
        | ({"place": place} if place is not None else {}),
    )


def resolve_seeds(
    query: ParsedQuery,
    graph: DualGraph,
    encoder: QueryEncoder,
    pattern_vectors: dict[str, np.ndarray],
    config: QueryConfig,
    place_scores: dict[str, float] | None = None,
) -> list[SeedMatch]:
    """Score every Pattern against the parsed query (and the zone's ``place_scores``, pattern id -> overlap) and pick diverse top seeds."""
    scope_text = "; ".join(f"{a} = {v}" if a != "*" else v for a, v in query.conditions)
    target_text = "; ".join(humanize(t) for t in query.targets)
    alone = place_scores is not None and not query.text.strip()  # a zone without text: nothing to embed
    qvec = None if alone else encoder.encode_query(scope_text, target_text, query.components(), query.text)
    scored = sorted(
        (
            score_pattern(ins, query, qvec, pattern_vectors.get(pid), config, None if place_scores is None else place_scores.get(pid, 0.0))
            for pid, ins in graph.insights.items()
            if not graph.tampered(pid)  # a tampered pattern never starts a walk
        ),
        key=lambda s: -s.score,
    )
    seeds: list[SeedMatch] = []
    floor = max(SEED_MIN_SCORE, SEED_RELATIVE_MIN * scored[0].score) if scored else 1.0
    stands, by_id, taken = representatives(graph, ((s.pattern_id, s.score) for s in scored)), {s.pattern_id: s for s in scored}, set()
    for cand in scored:  # diverse seeds: skip direct lattice neighbours of chosen seeds, unless more of it lies in the zone
        if len(seeds) >= SEED_TOP_K or cand.score < floor:
            break
        if key := trajectory(graph, cand.pattern_id):  # a trajectory ranks by its best window and seeds its representative once
            if key in taken:
                continue
            taken.add(key)
            cand = by_id[stands[key]]
        place = cand.matched.get("place", 0.0)
        near = {o for s in seeds if place <= s.matched.get("place", 0.0) for _, o in graph.incident(s.pattern_id, ["SPECIALIZES"])}
        if cand.pattern_id not in near:
            seeds.append(cand)
    pool = [s for s in scored if s.matched.get("place")] if alone else scored  # a zone alone: only what lies in it
    if not seeds and pool:
        seeds = [max(pool, key=lambda s: s.score if alone else s.matched["semantic"])]
        seeds[0].matched["fallback"] = "place" if alone else "semantic"
    return seeds
