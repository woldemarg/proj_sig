"""Query interpretation and seed resolution (SDD 10 §Query).

Deterministic parse against the graph vocabulary (metrics, dimension values)
plus a projection of the question into the insight space with the same
tripartite composition as patterns. No LLM is involved in retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltir.canonical import covariance_label, humanize
from ltir.config import Config
from ltir.encoder import InsightEncoder
from ltir.graph import DualGraph
from ltir.models import Insight

NEG_WORDS = {
    "lower",
    "low",
    "lowest",
    "decrease",
    "decreases",
    "decreased",
    "decreasing",
    "decline",
    "declines",
    "declining",
    "drop",
    "drops",
    "dropped",
    "down",
    "fall",
    "falls",
    "falling",
    "less",
    "reduced",
    "reduce",
    "reduction",
    "worse",
    "worst",
    "negative",
    "erosion",
    "eroded",
    "eroding",
    "compressed",
    "compression",
    "shrink",
    "shrinking",
    "below",
    "weak",
    "weaker",
    "poor",
    "smaller",
    "shorter",
    "fewer",
    "cheaper",
    "loss",
    "losses",
}
POS_WORDS = {
    "higher",
    "high",
    "highest",
    "increase",
    "increases",
    "increased",
    "increasing",
    "rise",
    "rises",
    "rising",
    "up",
    "more",
    "grow",
    "grows",
    "growth",
    "uplift",
    "better",
    "best",
    "positive",
    "above",
    "elevated",
    "spike",
    "surge",
    "larger",
    "longer",
    "bigger",
    "delay",
    "delays",
    "delayed",
    "slow",
    "slower",
    "stronger",
    "inflated",
}
GENERIC_METRIC_WORDS = {"median", "mean", "average", "avg", "total", "number", "num", "count", "percent", "pct", "rate", "value", "score", "index"}
COVARIANCE_WORDS = ("correl", "relationship", "relation", "coupl", "decoupl", "dependen", "covari", "linked", "associat")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _stem_match(a: str, b: str) -> bool:
    if a == b:
        return True
    k = min(len(a), len(b))
    return k >= 5 and a[:5] == b[:5] and abs(len(a) - len(b)) <= 3


@dataclass
class ParsedQuery:
    text: str
    targets: list[str] = field(default_factory=list)  # metric names
    direction: int = 0
    conditions: list[tuple[str, str]] = field(default_factory=list)  # (attribute, value)
    covariance: bool = False  # question is about a relationship between metrics

    @property
    def is_lexical(self) -> bool:
        return bool(self.targets or self.conditions)

    def components(self) -> tuple[tuple[str, float], ...]:
        if self.covariance and len(self.targets) >= 2:
            weakening = any(w in self.text.lower() for w in ("break", "weak", "decoupl", "lose", "loss", "disappear"))
            return ((covariance_label(self.targets[:2]), -2.0 if weakening else 2.0),)
        if not self.direction:
            # no direction word: no signed phenomenon components; the encoder then falls back
            # to the question text for the phenomenon block instead of assuming "higher"
            return ()
        return tuple((humanize(t), float(self.direction) * 2.0) for t in self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "conditions": [f"{a}={v}" for a, v in self.conditions]}


def parse_query(text: str, graph: DualGraph) -> ParsedQuery:
    raw_tokens = _TOKEN.findall(text)
    tokens = [t.lower() for t in raw_tokens]
    q = ParsedQuery(text=text)

    metrics = sorted({n["props"]["name"] for n in graph.of_kind("Metric")})
    parts_of = {m: [p for p in humanize(m).lower().split() if len(p) >= 3] for m in metrics}
    head_count: dict[str, int] = {}
    for parts in parts_of.values():
        if parts:
            head_count[parts[0]] = head_count.get(parts[0], 0) + 1
    matched: list[tuple[float, str]] = []
    for name, parts in parts_of.items():
        if not parts:
            continue
        hit = [any(_stem_match(p, t) for t in tokens) for p in parts]
        specific = sum(h for h, p in zip(hit, parts) if p not in GENERIC_METRIC_WORDS)
        # full name; or two words incl. a specific one ("house values" -> median_house_value);
        # or a distinctive head word ("returns" -> return_rate). "median" alone matches nothing.
        distinctive_head = hit[0] and parts[0] not in GENERIC_METRIC_WORDS and head_count[parts[0]] == 1
        if all(hit) or (sum(hit) >= 2 and specific >= 1) or distinctive_head:
            matched.append((sum(hit) / len(parts), name))
    if matched:  # best-covered metrics win (a full-name match beats partial ones)
        best = max(score for score, _ in matched)
        q.targets = sorted(name for score, name in matched if score == best)

    values: dict[tuple[str, str], None] = {}
    for ins in graph.insights.values():
        for cond in ins.conditions:
            values[(cond.attribute, cond.value)] = None
    for attr, value in values:
        short_upper = len(value) <= 4 and value.isupper()
        if short_upper:
            hit = value in raw_tokens
        else:
            vparts = value.lower().split()
            hit = all(any(_stem_match(v, t) for t in tokens) for v in vparts)
        if hit and (attr, value) not in q.conditions:
            q.conditions.append((attr, value))

    neg = sum(t in NEG_WORDS for t in tokens)
    pos = sum(t in POS_WORDS for t in tokens)
    q.direction = int(np.sign(pos - neg))
    q.covariance = any(t.startswith(COVARIANCE_WORDS) for t in tokens)
    if q.covariance:
        q.direction = 0  # "breaks down" / "weakens" describe the relationship, not a metric level
    return q


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
    wanted_attrs = {a for a, _ in wanted}
    conds = {(c.attribute, c.value) for c in ins.conditions}
    conflicts = len({a for a, v in conds if a in wanted_attrs and (a, v) not in wanted})
    scope = (len(conds & wanted) / len(wanted) if wanted else 0.0) - 0.5 * conflicts
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
    scope_text = "; ".join(f"{a} = {v}" for a, v in query.conditions)
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
