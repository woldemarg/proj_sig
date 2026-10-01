"""Transversal traversal engine (docs/07_question_answering.md §7.3).

Path grammar (regular), from each seed Pattern::

    P0 (structural){0,h}  --ACTIVATES-->  A (RELATED_TO){0,L}  --ACTIVATES^-1-->  P1 (structural){0,h}

i.e. optional structural drill-up/down at the seed, ascend into the latent plane,
move laterally across mutual-kNN attractors, descend into (possibly structurally
disjoint) patterns, then optional structural expansion there. Best-first search
maximises the product of edge factors (alignment, relation weight, structural
decay); node ranking multiplies by the pattern's Insight.weight.

Two baselines make the research hypothesis inspectable (docs/07_question_answering.md §7.6):
structural-only BFS and naive nearest-neighbour text retrieval.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ltir.config import Config
from ltir.graph import DualGraph
from ltir.models import STRUCTURAL_EDGES, UNDIRECTED_EDGES
from ltir.query import SeedMatch

STRUCTURAL = {e.value for e in STRUCTURAL_EDGES}
UNDIRECTED = {e.value for e in UNDIRECTED_EDGES}
SEED_SCORE_FLOOR = 1e-3


@dataclass
class PathStep:
    source: str
    target: str
    edge_type: str
    weight: float
    hop: int
    edge_id: str
    reverse: bool = False  # directed edge traversed against its stored direction (path text: <-TYPE(w)-)


@dataclass
class Retrieved:
    node_id: str
    kind: str
    score: float
    hop: int
    route: str  # seed | structural | transversal | latent
    seed_id: str
    path: list[PathStep]
    structural_distance: int | None = None  # lattice hops (incl. SIBLING) to nearest seed
    scope_overlap: float = 0.0  # max Jaccard of scope conditions with any seed
    transversal_only: bool = False  # reached via latent plane AND scope-disjoint from every seed


@dataclass
class TraversalResult:
    seeds: list[SeedMatch]
    patterns: list[Retrieved]
    attractors: list[Retrieved]
    used_edges: list[str]
    traversed_nodes: list[str]
    visited_count: int
    max_depth: int
    baselines: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": [asdict(s) for s in self.seeds],
            "patterns": [asdict(r) for r in self.patterns],
            "attractors": [asdict(r) for r in self.attractors],
            "used_edges": self.used_edges,
            "traversed_nodes": self.traversed_nodes,
            "visited_count": self.visited_count,
            "max_depth": self.max_depth,
            "baselines": self.baselines,
        }


def _conds(graph: DualGraph, pid: str) -> set[str]:
    return {c.expr for c in graph.insight(pid).conditions}


def structural_closure(graph: DualGraph, seed_ids: list[str], depth: int, edge_types: set[str] | None = None) -> dict[str, int]:
    """Patterns reachable via structural edges only (BFS -> hop count). Baseline 1 uses the
    traversal's lattice edge set; ``structural_distance`` uses every structural type."""
    types = edge_types or STRUCTURAL
    hops = {s: 0 for s in seed_ids}
    frontier = list(seed_ids)
    for d in range(1, depth + 1):
        nxt = []
        for node in frontier:
            for _, other in graph.incident(node, types):
                if other not in hops:
                    hops[other] = d
                    nxt.append(other)
        frontier = nxt
    return hops


def traverse(graph: DualGraph, seeds: list[SeedMatch], config: Config) -> TraversalResult:
    lattice = config.lattice_edges
    counter = itertools.count()
    heap: list[tuple] = []
    for s in seeds:
        # every edge factor is <= 1, so scores must start positive for best-first order to
        # mean "best"; a non-positive seed score (scope conflicts) is floored, not inverted
        heapq.heappush(heap, (-max(s.score, SEED_SCORE_FLOOR), next(counter), s.pattern_id, "P0", 0, 0, s.pattern_id, []))
    # Dijkstra state = (node, phase, structural hops used, latent hops used): a path that
    # reaches a node with a higher score but less remaining budget must not shadow one
    # with more budget, whose descendants may be unreachable otherwise.
    State = tuple[str, str, int, int]
    final: dict[State, tuple[float, str, list[PathStep]]] = {}
    seed_ids = {s.pattern_id for s in seeds}

    def push(score: float, node: str, phase: str, s_used: int, l_used: int, seed: str, path: list[PathStep]) -> None:
        if len(path) <= config.traversal_max_depth and (node, phase, s_used, l_used) not in final:
            heapq.heappush(heap, (-score, next(counter), node, phase, s_used, l_used, seed, path))

    while heap:
        neg, _, node, phase, s_cnt, l_cnt, seed, path = heapq.heappop(heap)
        if (node, phase, s_cnt, l_cnt) in final:
            continue
        score = -neg
        final[(node, phase, s_cnt, l_cnt)] = (score, seed, path)
        hop = len(path) + 1

        def step(e: dict[str, Any], other: str, w: float) -> list[PathStep]:
            reverse = e["target"] == node and e["source"] != node and e["type"] not in UNDIRECTED
            return path + [PathStep(node, other, e["type"], float(w), hop, e["id"], reverse=reverse)]

        if phase in ("P0", "P1"):
            if s_cnt < config.structural_hops:
                for e, other in graph.incident(node, lattice):
                    if e["type"] in ("SPECIALIZES", "GENERALIZES") and e["source"] != node:
                        continue  # both directions are materialised; follow out-edges only
                    if phase == "P1" and other in seed_ids:
                        continue
                    factor = config.structural_edge_decay * (e["weight"] if e["type"] == "CONTRASTS" else 1.0)
                    push(score * factor, other, phase, s_cnt + 1, l_cnt, seed, step(e, other, e["weight"]))
            if phase == "P0":
                for e, other in graph.incident(node, ["ACTIVATES"]):
                    if e["source"] == node and e["weight"] >= config.activation_threshold:
                        push(score * e["weight"], other, "A", 0, 0, seed, step(e, other, e["weight"]))
        elif phase == "A":
            if l_cnt < config.max_latent_hops:
                for e, other in graph.incident(node, ["RELATED_TO"]):
                    if e["weight"] >= config.relation_threshold:
                        push(score * e["weight"], other, "A", 0, l_cnt + 1, seed, step(e, other, e["weight"]))
            for e, other in graph.incident(node, ["ACTIVATES"]):
                if e["target"] == node and other not in seed_ids and e["weight"] >= config.activation_threshold:
                    push(score * e["weight"], other, "P1", 0, l_cnt, seed, step(e, other, e["weight"]))

    closure = structural_closure(graph, list(seed_ids), config.traversal_max_depth)
    seed_conds = [_conds(graph, s) for s in seed_ids]
    best: dict[str, Retrieved] = {}
    attractors: dict[str, Retrieved] = {}
    for (node, phase, _s, _l), (score, seed, path) in final.items():
        kind = graph.kind(node)
        if kind == "Attractor":
            r = Retrieved(node, kind, score, len(path), "latent", seed, path)
            if node not in attractors or score > attractors[node].score:
                attractors[node] = r
            continue
        weight = graph.insight(node).weight
        route = "seed" if node in seed_ids and not path else ("structural" if phase == "P0" else "transversal")
        ranked = score if route == "seed" else score * weight
        conds = _conds(graph, node)
        overlap = max((len(conds & c) / len(conds | c) for c in seed_conds), default=0.0)
        r = Retrieved(
            node,
            kind,
            ranked,
            len(path),
            route,
            seed,
            path,
            structural_distance=closure.get(node),
            scope_overlap=round(overlap, 3),
        )
        r.transversal_only = route == "transversal" and overlap == 0.0
        if node not in best or r.score > best[node].score:
            best[node] = r

    ordered = sorted(best.values(), key=lambda r: (r.route != "seed", -r.score))
    patterns = ordered[: len(seed_ids) + config.max_retrieved]
    used: dict[str, None] = {}
    nodes: dict[str, None] = {}
    for r in patterns:
        nodes[r.seed_id] = None
        for st in r.path:
            used[st.edge_id] = None
            nodes[st.source] = None
            nodes[st.target] = None
        nodes[r.node_id] = None
    visited_attr = [a for a in attractors.values() if a.node_id in nodes]
    return TraversalResult(
        seeds=seeds,
        patterns=patterns,
        attractors=sorted(visited_attr, key=lambda a: -a.score),
        used_edges=list(used),
        traversed_nodes=list(nodes),
        visited_count=len(final),
        max_depth=max((len(r.path) for r in patterns), default=0),
    )


def naive_nearest(question_vec: np.ndarray, doc_vectors: dict[str, np.ndarray], k: int) -> list[tuple[str, float]]:
    """Baseline 2: nearest canonical-document texts to the question text (plain vector RAG)."""
    scored = sorted(((pid, float(v @ question_vec)) for pid, v in doc_vectors.items()), key=lambda x: -x[1])
    return scored[:k]
