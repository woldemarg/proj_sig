"""Grounding, seeds, transversal search and evidence over the dual graph (docs/07_question_answering.md).

A question and a committed state (a snapshot's ``DualGraph`` plus its ``LatentFrame``) in, the evidence out —
as objects, or as the ``EvidencePayload`` an LLM prompt is built from. Any embedding model that satisfies
``ports.QueryEncoder`` works; depends only on ``insight_contracts`` and its numerical libraries.
"""

from graph_query_engine.config import QueryConfig
from graph_query_engine.evidence import Evidence, build_evidence
from graph_query_engine.graph import DualGraph, LatentFrame
from graph_query_engine.ports import Embedder, QueryEncoder
from graph_query_engine.question import LiteralCatalog, ParsedQuery, build_catalog, parse_query
from graph_query_engine.search import CommittedState, SearchResult, empty_payload, search
from graph_query_engine.seeds import SeedMatch, resolve_seeds
from graph_query_engine.traversal import TraversalResult, structural_closure, traverse

__all__ = [
    "CommittedState",
    "DualGraph",
    "Embedder",
    "Evidence",
    "LatentFrame",
    "LiteralCatalog",
    "ParsedQuery",
    "QueryConfig",
    "QueryEncoder",
    "SearchResult",
    "SeedMatch",
    "TraversalResult",
    "build_catalog",
    "build_evidence",
    "empty_payload",
    "parse_query",
    "resolve_seeds",
    "search",
    "structural_closure",
    "traverse",
]
