"""Retrieval for one question (docs/07_question_answering.md §7.1–7.4, §7.6).

question -> parse (+ literal grounding) -> seeds -> transversal walk (+ baselines) -> evidence.
The result is the structured analytical context: the chat verbalises it (``ltir.answering``),
any other caller can use the evidence object or its prompt text directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ltir.analysis.encoder import InsightEncoder
from ltir.analysis.graph import DualGraph
from ltir.config import Config
from ltir.models import LatentFrame
from ltir.retrieval.evidence import Evidence, build_evidence
from ltir.retrieval.question import LiteralCatalog, ParsedQuery, parse_query
from ltir.retrieval.seeds import SeedMatch, resolve_seeds
from ltir.retrieval.traversal import TraversalResult, naive_nearest, structural_closure, traverse

HIGHLIGHT_KEYS = ("seeds", "traversed", "anchors", "evidence", "edges", "transversal_only")


@dataclass
class CommittedState:
    """What retrieval reads, all from one commit: the graph, its vectors and — built on the first question
    (``Engine.prepared``) — its literal catalog. A commit replaces the whole object, never a part of it."""

    graph: DualGraph
    frame: LatentFrame
    catalog: LiteralCatalog | None = None


@dataclass
class SearchResult:
    question: str
    parsed: ParsedQuery
    seeds: list[SeedMatch]
    traversal: TraversalResult  # ``baselines``: what structural-only and naive text retrieval return (§7.6)
    evidence: Evidence
    seconds: float  # the search, evidence included (``Engine.search`` adds its own preparation)

    def highlight(self) -> dict[str, list[str]]:
        """Node and edge ids to mark in the graph and on the sphere (docs/08_interface.md §8.3)."""
        result, items = self.traversal, self.evidence.items
        return {
            "seeds": [s.pattern_id for s in self.seeds],
            "traversed": result.traversed_nodes,
            "anchors": [a.node_id for a in result.attractors],
            "evidence": [it.pattern_id for it in items],
            "edges": result.used_edges,
            "transversal_only": [it.pattern_id for it in items if it.transversal_only],
        }


def compute_baselines(
    graph: DualGraph, question: str, seeds: list[SeedMatch], result: TraversalResult, frame: LatentFrame, encoder: InsightEncoder, config: Config
) -> dict[str, Any]:
    """What structural-only and naive text retrieval would have returned (§7.6). The documents were embedded at
    ingest, so the question is the only text embedded here; a pattern without a document vector is left out."""
    closure = structural_closure(graph, [s.pattern_id for s in seeds], config.traversal_max_depth, config.lattice_edges)
    doc_vecs = {pid: frame.documents[pid] for pid in graph.insights if pid in frame.documents}
    naive = naive_nearest(encoder.embedder.embed_queries([question])[0], doc_vecs, config.max_retrieved)
    retrieved = {r.node_id for r in result.patterns}
    return {
        "structural_only": sorted(closure),
        "naive_nearest": [{"pattern_id": p, "cosine": round(c, 3)} for p, c in naive],
        "transversal_only": [r.node_id for r in result.patterns if r.transversal_only],
        "not_in_naive_topk": sorted(retrieved - {p for p, _ in naive}),
        "not_structurally_reachable": sorted(retrieved - set(closure)),
    }


def search(question: str, state: CommittedState, encoder: InsightEncoder, config: Config) -> SearchResult:
    """The evidence for ``question`` over one committed state."""
    start = time.perf_counter()
    graph = state.graph
    parsed = parse_query(question, graph, config, state.catalog)
    seeds = resolve_seeds(parsed, graph, encoder, state.frame.patterns, config)
    result = traverse(graph, seeds, config)
    result.baselines = compute_baselines(graph, question, seeds, result, state.frame, encoder, config)
    evidence = build_evidence(parsed, result, graph, config)
    return SearchResult(question, parsed, seeds, result, evidence, time.perf_counter() - start)
