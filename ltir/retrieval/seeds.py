"""Seed resolution (docs/07_question_answering.md §7.2): every pattern scored against the parsed question
and the query vector, then the diverse top seeds the transversal walk starts from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltir.analysis.canonical import humanize
from ltir.analysis.encoder import InsightEncoder
from ltir.analysis.graph import DualGraph
from ltir.config import Config
from ltir.models import Insight
from ltir.retrieval.question import ParsedQuery


@dataclass
class SeedMatch:
    pattern_id: str
    score: float
    matched: dict[str, Any]


def score_pattern(ins: Insight, query: ParsedQuery, qvec: np.ndarray, vec: np.ndarray | None, config: Config) -> SeedMatch:
    """Seed score of one pattern: lexical match (target, scope, direction) + semantic cosine + weight."""
    pair = set(ins.covariance.get("pair", []))
    wanted_targets = set(query.targets)
    material = [s.robust_z for t in query.targets for s in ins.shifts if s.metric == t and s.magnitude >= config.min_component_z]
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
        },
    )


def resolve_seeds(
    query: ParsedQuery,
    graph: DualGraph,
    encoder: InsightEncoder,
    pattern_vectors: dict[str, np.ndarray],
    config: Config,
) -> list[SeedMatch]:
    """Score every Pattern against the parsed query and pick diverse top seeds."""
    scope_text = "; ".join(f"{a} = {v}" if a != "*" else v for a, v in query.conditions)
    target_text = "; ".join(humanize(t) for t in query.targets)
    qvec = encoder.encode_query(scope_text, target_text, query.components(), query.text)
    scored = sorted(
        (score_pattern(ins, query, qvec, pattern_vectors.get(pid), config) for pid, ins in graph.insights.items()),
        key=lambda s: -s.score,
    )
    seeds: list[SeedMatch] = []
    floor = max(config.seed_min_score, config.seed_relative_min * scored[0].score) if scored else 1.0
    for cand in scored:  # diverse seeds: skip direct lattice neighbours of chosen seeds
        if len(seeds) >= config.seed_top_k or cand.score < floor:
            break
        near = {o for s in seeds for _, o in graph.incident(s.pattern_id, ["SPECIALIZES", "GENERALIZES"])}
        if cand.pattern_id not in near:
            seeds.append(cand)
    if not seeds and scored:
        seeds = [max(scored, key=lambda s: s.matched["semantic"])]
        seeds[0].matched["fallback"] = "semantic"
    return seeds
