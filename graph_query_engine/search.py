"""Retrieval for one question (docs/07_question_answering.md §7.1–7.4, §7.6).

question -> parse (+ literal grounding) -> seeds -> transversal walk (+ baselines) -> evidence.
The result is the structured analytical context: ``SearchResult.payload()`` is what the narrator verbalises;
any other caller can use the evidence object or its prompt text directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from graph_query_engine.config import QueryConfig
from graph_query_engine.evidence import Evidence, build_evidence
from graph_query_engine.graph import DualGraph, LatentFrame
from graph_query_engine.ports import QueryEncoder
from graph_query_engine.question import LiteralCatalog, ParsedQuery, build_catalog, parse_query
from graph_query_engine.seeds import SeedMatch, resolve_seeds
from graph_query_engine.traversal import LATTICE_EDGES, TraversalResult, structural_closure, traverse
from insight_contracts.payload import CITATION_FIELDS, EvidencePayload

_HIGHLIGHT_KEYS = ("seeds", "traversed", "anchors", "evidence", "edges", "transversal_only")


@dataclass
class CommittedState:
    """What retrieval reads, all from one commit: the graph, its vectors and its literal catalog. The graph service
    builds it complete at the commit; a library caller may leave ``catalog`` None and ``search`` builds it on first
    use. A commit replaces the whole object, never a part of it."""

    graph: DualGraph
    frame: LatentFrame
    catalog: LiteralCatalog | None = None


@dataclass
class SearchResult:
    question: str
    parsed: ParsedQuery
    seeds: list[SeedMatch]
    traversal: TraversalResult  # ``baselines``: what the structural lattice alone reaches (§7.6)
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


def compute_baselines(graph: DualGraph, seeds: list[SeedMatch], result: TraversalResult, config: QueryConfig) -> dict[str, Any]:
    """What the structural lattice alone would have reached from the seeds, and what the walk found beyond it (§7.6). The
    naive text baseline is a measurement, not part of an answer: ``scripts/experiment.py`` embeds the documents itself."""
    closure = structural_closure(graph, [s.pattern_id for s in seeds], config.traversal_max_depth, LATTICE_EDGES)
    retrieved = {r.node_id for r in result.patterns}
    return {
        "structural_only": sorted(closure),
        "transversal_only": [r.node_id for r in result.patterns if r.transversal_only],
        "not_structurally_reachable": sorted(retrieved - set(closure)),
    }


def search(
    question: str,
    state: CommittedState,
    encoder: QueryEncoder,
    config: QueryConfig,
    place_scores: dict[str, float] | None = None,
    as_of: datetime | None = None,
) -> SearchResult:
    """The evidence for ``question`` over one committed state; ``place_scores`` (pattern id -> its overlap with an asked
    map zone) weigh the seeds, and the question may then be empty; ``as_of`` is the reference date the findings' freshness
    counts from — the graph's latest data when not asked."""
    start = time.perf_counter()
    graph = state.graph
    if state.catalog is None and graph.insights:  # a caller that did not prepare one: build it now, kept on the state
        state.catalog = build_catalog(graph, encoder.embedder)
    parsed = parse_query(question, graph, config, state.catalog)
    clock = as_of or graph.as_of
    seeds = resolve_seeds(parsed, graph, encoder, state.frame.patterns, config, place_scores, clock)
    result = traverse(graph, seeds, config, None if parsed.when else clock)  # an asked period is its own window
    result.baselines = compute_baselines(graph, seeds, result, config)
    evidence = build_evidence(parsed, result, graph, config, place_scores, (clock, "asked" if as_of else "latest data") if clock else None)
    return SearchResult(question, parsed, seeds, result, evidence, time.perf_counter() - start)
