"""The dual graph as the query engine reads it: an adjacency index over a snapshot, plus the committed vectors
(docs/06_graph_and_storage.md §6.2). Snapshots are compiled by the graph service; this module only reads them."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from insight_contracts import SNAPSHOT_VERSION, Insight
from insight_contracts.text import describe_component


@dataclass(frozen=True)
class LatentFrame:
    """Committed vectors of the single insight frame: pattern id -> unit vector, attractor id -> centroid,
    plus the canonical-document embeddings the naive text baseline compares questions with.

    Frozen fields, but ``documents`` is a cache: a document vector the commit lacks is added in place, once, on the
    first question (by the caller); no entry is replaced."""

    patterns: dict[str, np.ndarray]
    attractors: dict[int, np.ndarray]
    documents: dict[str, np.ndarray]


def describe_components(signature: list[dict[str, Any]]) -> str:
    """Two strongest entries of a stored signature as prose (how the LLM prompt names an anchor)."""
    return " and ".join(describe_component(e["component"], e["value"]) for e in signature[:2])


class DualGraph:
    """Read-only adjacency index over a snapshot (used by traversal, UI, evidence).

    Pattern nodes carry the ``Insight.to_record()`` record. ``insight()`` is the object retrieval
    uses; ``canonical_document()`` is the text sitting beside that record.
    """

    def __init__(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("version", SNAPSHOT_VERSION) != SNAPSHOT_VERSION:
            raise ValueError(f"snapshot version {snapshot['version']}, this reader understands {SNAPSHOT_VERSION}")
        self.snapshot = snapshot
        self.nodes: dict[str, dict[str, Any]] = {n["id"]: n for n in snapshot.get("nodes", [])}
        self.edges: list[dict[str, Any]] = snapshot.get("edges", [])
        self.insights: dict[str, Insight] = {n["id"]: Insight.from_record(n["props"]) for n in self.nodes.values() if n["kind"] == "Pattern"}
        self.out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.inc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in self.edges:
            self.out[e["source"]].append(e)
            self.inc[e["target"]].append(e)

    def insight(self, node_id: str) -> Insight:
        return self.insights[node_id]

    def canonical_document(self, node_id: str) -> str:
        return self.nodes[node_id]["props"]["canonical"]["document"]

    @classmethod
    def empty(cls) -> DualGraph:
        return cls({"version": SNAPSHOT_VERSION, "nodes": [], "edges": []})

    def kind(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        return node["kind"] if node else None

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["kind"] == kind]

    def incident(self, node_id: str, types: Iterable[str]) -> list[tuple[dict[str, Any], str]]:
        """(edge, other endpoint) for edges of the given types touching node_id."""
        wanted = set(types)
        found = [(e, e["target"]) for e in self.out.get(node_id, []) if e["type"] in wanted]
        found += [(e, e["source"]) for e in self.inc.get(node_id, []) if e["type"] in wanted]
        return found
