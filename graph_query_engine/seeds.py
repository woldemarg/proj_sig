"""Seed resolution (docs/07_question_answering.md §7.2): every pattern scored against the parsed question
and the query vector, then the diverse top seeds the transversal walk starts from."""

from __future__ import annotations

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


@dataclass
class SeedMatch:
    pattern_id: str
    score: float
    matched: dict[str, Any]


def score_pattern(
    ins: Insight, query: ParsedQuery, qvec: np.ndarray, vec: np.ndarray | None, config: QueryConfig, place: float | None = None
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
    semantic = float(vec @ qvec) if vec is not None else 0.0
    if query.is_lexical:
        score = 0.35 * target + 0.25 * scope + 0.15 * direction + 0.15 * semantic + 0.10 * ins.weight
    else:
        score = 0.7 * semantic + 0.3 * ins.weight
    if place is not None:
        score = (ins.weight + place if place else 0.0) if not query.text.strip() else score + SEED_PLACE_WEIGHT * place
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
    qvec = encoder.encode_query(scope_text, target_text, query.components(), query.text)
    scored = sorted(
        (
            score_pattern(ins, query, qvec, pattern_vectors.get(pid), config, None if place_scores is None else place_scores.get(pid, 0.0))
            for pid, ins in graph.insights.items()
            if graph.nodes[pid]["props"].get("integrity") != "mismatch"  # a tampered pattern never starts a walk
        ),
        key=lambda s: -s.score,
    )
    seeds: list[SeedMatch] = []
    floor = max(SEED_MIN_SCORE, SEED_RELATIVE_MIN * scored[0].score) if scored else 1.0
    for cand in scored:  # diverse seeds: skip direct lattice neighbours of chosen seeds
        if len(seeds) >= SEED_TOP_K or cand.score < floor:
            break
        near = {o for s in seeds for _, o in graph.incident(s.pattern_id, ["SPECIALIZES"])}
        if cand.pattern_id not in near:
            seeds.append(cand)
    if not seeds and scored and (place_scores is None or query.text.strip()):  # a zone alone that holds no pattern has no seed
        seeds = [max(scored, key=lambda s: s.matched["semantic"])]
        seeds[0].matched["fallback"] = "semantic"
    return seeds
