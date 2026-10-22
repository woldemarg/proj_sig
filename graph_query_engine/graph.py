"""The dual graph as the query engine reads it: an adjacency index over a snapshot, plus the committed vectors
(docs/06_graph_and_storage.md §6.2). Snapshots are compiled by the graph service; this module only reads them."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from insight_contracts import SNAPSHOT_VERSION, Insight, PhenomenonThresholds, SpatialProvenance, dataset_node_id, dimension_node_id, wording
from insight_contracts.text import describe_component


@dataclass(frozen=True)
class LatentFrame:
    """Committed vectors of the single insight frame: pattern id -> unit vector, attractor id -> centroid."""

    patterns: dict[str, np.ndarray]
    attractors: dict[int, np.ndarray]


@dataclass(frozen=True)
class PatternFacts:
    """What the vertical zoom and the walk read of one pattern. A broad pattern at resolution 8 lies in thousands of regions, and
    a step meets every pattern of each of them: re-deriving these per meeting made one step take a minute."""

    resolution: int | None
    upload: str | None
    regions: dict[str, float]  # IN_REGION target -> share (read only)
    finest: frozenset[str]  # its regions at its finest level: its own resolution when that is a region level, many for a broad one
    anchors: dict[str, float]  # non-weak ACTIVATES target -> alignment (read only)


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
        self.memo: dict[tuple[str, str], Any] = {}  # facts readers derive from the snapshot, filled once: it never changes
        self._memo_lock = threading.RLock()

    def memoised(self, key: tuple[str, str], build: Callable[[], Any]) -> Any:
        """``build()`` kept under ``key``: computed once per graph even when readers on several threads ask at once (a
        projection takes seconds on a large upload). Reentrant, since one build may read another's facts."""
        # ponytail: one lock per graph serialises the first builds of different keys; per-key locks if readers ever queue
        if key in self.memo:
            return self.memo[key]
        with self._memo_lock:
            if key not in self.memo:
                self.memo[key] = build()
            return self.memo[key]

    def insight(self, node_id: str) -> Insight:
        return self.insights[node_id]

    def tampered(self, pattern_id: str) -> bool:
        """Whether the pattern's journal record fails its digest (docs/13_trust.md §13.4): never a seed, a walk slot or evidence."""
        return self.nodes[pattern_id]["props"].get("integrity") == "mismatch"

    def facts(self, pattern_id: str) -> PatternFacts:
        """``pattern_id``'s facts, derived once per graph (``memoised``)."""

        def build() -> PatternFacts:
            out, ins = self.out.get(pattern_id, []), self.insight(pattern_id)
            spatial = SpatialProvenance.from_dict(ins.provenance.get("spatial"))
            regions = {e["target"]: e["weight"] for e in out if e["type"] == "IN_REGION"}
            levels = {r: self.nodes[r]["props"]["resolution"] for r in regions}
            top = max(levels.values(), default=None)
            return PatternFacts(
                resolution=spatial.resolution if spatial else None,
                upload=self.nodes.get(dataset_node_id(ins.dataset_id), {"props": {}})["props"].get("upload_id"),
                regions=regions,
                finest=frozenset(r for r, lv in levels.items() if lv == top),
                anchors={e["target"]: e["weight"] for e in out if e["type"] == "ACTIVATES" and not e["props"].get("weak")},
            )

        return self.memoised(("facts", pattern_id), build)

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


def primary_anchor(graph: DualGraph, pattern_id: str) -> dict[str, Any] | None:
    """The pattern's strongest non-weak ACTIVATES edge: the theme it is shown and walked under (a weak membership counts
    for coverage only, docs/06_graph_and_storage.md §6.2); None when it has none."""
    acts = (e for e in graph.out.get(pattern_id, []) if e["type"] == "ACTIVATES" and not e["props"].get("weak"))
    return max(acts, key=lambda e: e["weight"], default=None)


def insight_reading(graph: DualGraph, pattern_id: str, thresholds: PhenomenonThresholds) -> dict[str, str]:
    """The insight in words, ``{"where", "what"}`` (``wording.reading``), with its dataset's declared glossary
    (``user_glossary`` of its Dataset node) and the band count of each quantile-band condition from its Dimension node:
    every surface that describes an insight reads it here."""
    ins = graph.insight(pattern_id)
    declared = graph.nodes.get(dataset_node_id(ins.dataset_id), {"props": {}})["props"].get("user_glossary", {})
    dims = (graph.nodes.get(dimension_node_id(ins.dataset_id, c.attribute)) for c in ins.conditions if c.attribute.endswith("_band"))
    # ponytail: a band column with missing values counts "missing" as a band; the band count is one too high then
    bands = {d["props"]["name"]: d["props"]["cardinality"] for d in dims if d and d["props"].get("cardinality")}
    return wording.reading(ins, thresholds, declared, bands)
