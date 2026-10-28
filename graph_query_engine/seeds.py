"""Seed resolution (docs/07_question_answering.md §7.2): every pattern scored against the parsed question
and the query vector, then the diverse top seeds the transversal walk starts from."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from graph_query_engine.config import QueryConfig
from graph_query_engine.graph import DualGraph
from graph_query_engine.ports import QueryEncoder
from graph_query_engine.question import ParsedQuery
from graph_query_engine.temporal import period_hub, recency, time_score
from insight_contracts import Insight
from insight_contracts.text import humanize

SEED_TOP_K = 3
SEED_MIN_SCORE = 0.25
SEED_RELATIVE_MIN = 0.75  # seeds must score >= this fraction of the best seed
SEED_PLACE_WEIGHT = 0.35  # a map zone: the weight of a pattern's overlap with it (docs/07_question_answering.md §7.2)
SEED_TIME_WEIGHT = 0.35  # a calendar period in the question: the weight of the pattern's measurement in it (same section)
SEED_RECENCY_WEIGHT = 0.35  # no period in the question: the weight of the pattern's freshness at the reference date (same section)


@dataclass
class SeedMatch:
    pattern_id: str
    score: float
    matched: dict[str, Any]


def score_pattern(
    ins: Insight,
    query: ParsedQuery,
    qvec: np.ndarray | None,
    vec: np.ndarray | None,
    config: QueryConfig,
    place: float | None = None,
    time: float | None = None,
    fresh: float | None = None,
) -> SeedMatch:
    """Seed score of one pattern: lexical match (target, scope, direction) + semantic cosine + weight, plus
    ``SEED_PLACE_WEIGHT`` × how much it and the asked map zone are one place (``place``; None: no zone); a zone without
    text: weight + place, 0 outside the zone. ``time``: how much the asked period it was measured in (``temporal.time_score``;
    None: no period asked, or a pattern without a series) adds ``SEED_TIME_WEIGHT`` × it; ``fresh``: its freshness at the
    reference date when no period is asked (``temporal.recency``; None: a period asked, or no core) adds
    ``SEED_RECENCY_WEIGHT`` × it — bonuses, never a penalty."""
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
    if time is not None:
        score += SEED_TIME_WEIGHT * time
    if fresh is not None:
        score += SEED_RECENCY_WEIGHT * fresh
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
        | ({"place": place} if place is not None else {})
        | ({"time": round(time, 3)} if time is not None else {})
        | ({"recency": round(fresh, 3)} if fresh is not None else {}),
    )


def resolve_seeds(
    query: ParsedQuery,
    graph: DualGraph,
    encoder: QueryEncoder,
    pattern_vectors: dict[str, np.ndarray],
    config: QueryConfig,
    place_scores: dict[str, float] | None = None,
    as_of: datetime | None = None,
) -> list[SeedMatch]:
    """Score every Pattern against the parsed query (and the zone's ``place_scores``, pattern id -> overlap; without a
    period, its freshness at ``as_of``, the reference date) and pick diverse top seeds."""
    scope_text = "; ".join(f"{a} = {v}" if a != "*" else v for a, v in query.conditions)
    target_text = "; ".join(humanize(t) for t in query.targets)
    alone = place_scores is not None and not query.text.strip()  # a zone without text: nothing to embed
    qvec = None if alone else encoder.encode_query(scope_text, target_text, query.components(), query.text)
    clock = as_of if query.when is None else None  # an asked period is its own window: no freshness then

    def fresh(pid: str) -> float | None:
        return recency(graph.trajectories.get(pid), clock, config.temporal_recency_half_life_days) if clock else None

    scored = sorted(
        (
            score_pattern(
                ins,
                query,
                qvec,
                pattern_vectors.get(pid),
                config,
                None if place_scores is None else place_scores.get(pid, 0.0),
                time_score(graph.trajectories[pid], query.when) if query.when and pid in graph.trajectories else None,
                fresh(pid),
            )
            for pid, ins in graph.insights.items()
            if not graph.tampered(pid)  # a tampered pattern never starts a walk
        ),
        key=lambda s: -s.score,
    )
    seeds: list[SeedMatch] = []
    base = {  # a period, or freshness, only reorders
        s.pattern_id: s.score - SEED_TIME_WEIGHT * s.matched.get("time", 0.0) - SEED_RECENCY_WEIGHT * s.matched.get("recency", 0.0) for s in scored
    }
    floor = max(SEED_MIN_SCORE, SEED_RELATIVE_MIN * max(base.values())) if scored else 1.0
    for cand in scored:  # diverse seeds: skip direct lattice neighbours of chosen seeds, unless more of it lies in the zone
        if len(seeds) >= SEED_TOP_K:
            break
        if base[cand.pattern_id] < floor:
            continue
        place = cand.matched.get("place", 0.0)
        near = {o for s in seeds if place <= s.matched.get("place", 0.0) for _, o in graph.incident(s.pattern_id, ["SPECIALIZES"])}
        if cand.pattern_id not in near:
            seeds.append(cand)
    pool = [s for s in scored if s.matched.get("place")] if alone else scored  # a zone alone: only what lies in it
    if not seeds and pool:
        members: set[str] = set()
        timed: list[SeedMatch] = []
        if query.when and not query.is_lexical:  # a period alone: its patterns, by the hub index (DualGraph.members)
            found = graph.members(query.when)
            members = set(found["active"]) | {pid for pid, _ in found["peaked"] + found["dropped"]}
            timed = [s for s in pool if s.pattern_id in members or s.matched.get("time")]
        if alone:
            seeds, how = [max(pool, key=lambda s: s.score)], "place"
        elif timed:  # nothing the catalog knows, a period it does: what lives or moved in it first, then what was measured there
            h = config.temporal_recency_half_life_days

            def rank(s: SeedMatch) -> tuple[bool, float, float, float]:
                return (
                    s.pattern_id in members,
                    s.matched.get("time", 0.0),
                    recency(graph.trajectories.get(s.pattern_id), as_of, h) or 0.0,
                    s.matched.get("weight", 0.0),
                )

            seeds, how = [max(timed, key=rank)], "period"
            if (hub := period_hub(query.text)) in graph.nodes:
                seeds[0].matched["hub"] = hub
        else:
            seeds, how = [max(pool, key=lambda s: s.matched["semantic"])], "semantic"
        seeds[0].matched["fallback"] = how
    return seeds
