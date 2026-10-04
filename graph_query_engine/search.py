"""Retrieval for one question (docs/07_question_answering.md §7.1–7.4, §7.6).

question -> parse (+ literal grounding) -> seeds -> transversal walk (+ baselines) -> evidence.
The result is the structured analytical context: ``SearchResult.payload()`` is what the narrator verbalises;
any other caller can use the evidence object or its prompt text directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from graph_query_engine.config import QueryConfig
from graph_query_engine.evidence import Evidence, build_evidence
from graph_query_engine.graph import DualGraph, LatentFrame
from graph_query_engine.ports import QueryEncoder
from graph_query_engine.question import LiteralCatalog, ParsedQuery, build_catalog, parse_query
from graph_query_engine.seeds import SeedMatch, resolve_seeds
from graph_query_engine.traversal import TraversalResult, naive_nearest, structural_closure, traverse
from insight_contracts.payload import CITATION_FIELDS, EvidencePayload

_HIGHLIGHT_KEYS = ("seeds", "traversed", "anchors", "evidence", "edges", "transversal_only")


@dataclass
class CommittedState:
    """What retrieval reads, all from one commit: the graph, its vectors and its literal catalog. ``catalog`` and the
    document vectors ``frame`` lacks are caches: filled once, on the first question (by the caller; the catalog by
    ``search`` if the caller did not), and never replaced. A commit replaces the whole object, never a part of it."""

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
    seconds: float  # the search, evidence included (a caller may add its own preparation)

    def payload(self) -> EvidencePayload:
        """The evidence for the narrator (prompt, summary, citation manifest, metrics) and the console (``view``)."""
        prompt = self.evidence.to_prompt()
        result = self.traversal
        return EvidencePayload(
            question=self.question,
            evidence_prompt=prompt,
            evidence_summary=self.evidence.summary(),
            # every field but filename is the insight's own; filename comes from the free-form provenance, if it has one
            citations={p["key"]: {k: p.get(k, "") for k in CITATION_FIELDS} for p in self.evidence.provenance},
            view={
                "evidence": self.evidence.to_dict(),
                "traversal": result.to_dict(),
                "highlight": self.highlight(),
            },
            metrics={
                "retrieval_s": round(self.seconds, 3),
                "seed_count": len(self.seeds),
                "traversal_depth": result.max_depth,
                "visited_states": result.visited_count,
                "retrieved_evidence": len(self.evidence.items),
                "anchors_visited": len(result.attractors),
            },
        )

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


def empty_payload(question: str) -> EvidencePayload:
    """The answer to any question while the graph holds no insight: an ordinary state, not an error."""
    return EvidencePayload(question, graph_empty=True, view={"evidence": {}, "traversal": {}, "highlight": {k: [] for k in _HIGHLIGHT_KEYS}})


def compute_baselines(
    graph: DualGraph, question: str, seeds: list[SeedMatch], result: TraversalResult, frame: LatentFrame, encoder: QueryEncoder, config: QueryConfig
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


def search(question: str, state: CommittedState, encoder: QueryEncoder, config: QueryConfig) -> SearchResult:
    """The evidence for ``question`` over one committed state."""
    start = time.perf_counter()
    graph = state.graph
    if state.catalog is None and graph.insights:  # a caller that did not prepare one: build it now, kept on the state
        state.catalog = build_catalog(graph, encoder.embedder)
    parsed = parse_query(question, graph, config, state.catalog)
    seeds = resolve_seeds(parsed, graph, encoder, state.frame.patterns, config)
    result = traverse(graph, seeds, config)
    result.baselines = compute_baselines(graph, question, seeds, result, state.frame, encoder, config)
    evidence = build_evidence(parsed, result, graph, config)
    return SearchResult(question, parsed, seeds, result, evidence, time.perf_counter() - start)
