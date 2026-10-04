"""The dual graph's vocabulary and its snapshot contract (docs/06_graph_and_storage.md §6.1–6.2).

A snapshot is ``{"version": SNAPSHOT_VERSION, "nodes": [...], "edges": [...], "stats": {...}}``; the graph service
compiles it, the query engine and the console read it. Node ``props`` by kind, as the readers rely on them:

* ``Pattern``   the ``Insight.to_record()`` fields, plus ``row_id`` (its journal row), ``canonical`` (with ``document``
                and ``components``) and ``embedding`` (the vector contract)
* ``Attractor`` ``attractor_id``, ``mass``, ``n_patterns``, ``distinct_scopes``, ``signature`` [{component, value}],
                ``targets``, ``datasets``, ``dimensions``, ``dispersion``, ``centroid``
* ``Metric``    ``name``, ``dataset_id``, ``global_median``, ``global_mad``
* ``Dimension`` ``name``, ``dataset_id``, ``cardinality``, ``entropy``
* ``Dataset`` / ``Batch``  provenance (file, rows, batch sequence)

Edge ``props``: ``ACTIVATES`` carries ``alignment``, ``strength`` and ``weak`` (coverage only, never walked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SNAPSHOT_VERSION = 2  # Pattern props are the journal Insight record (not a projected schema)


class EdgeType(str, Enum):
    SPECIALIZES = "SPECIALIZES"
    GENERALIZES = "GENERALIZES"
    SIBLING = "SIBLING"
    CONTRASTS = "CONTRASTS"
    HAS_SCOPE = "HAS_SCOPE"
    TARGETS = "TARGETS"
    ACTIVATES = "ACTIVATES"
    RELATED_TO = "RELATED_TO"
    DISCOVERED_IN = "DISCOVERED_IN"
    OF_DATASET = "OF_DATASET"


STRUCTURAL_EDGES = {EdgeType.SPECIALIZES, EdgeType.GENERALIZES, EdgeType.SIBLING, EdgeType.CONTRASTS}
UNDIRECTED_EDGES = {EdgeType.SIBLING, EdgeType.CONTRASTS, EdgeType.RELATED_TO}
EDGE_PLANE = {
    EdgeType.SPECIALIZES: "structural",
    EdgeType.GENERALIZES: "structural",
    EdgeType.SIBLING: "structural",
    EdgeType.CONTRASTS: "structural",
    EdgeType.RELATED_TO: "latent",
    EdgeType.ACTIVATES: "bridge",
    EdgeType.HAS_SCOPE: "schema",
    EdgeType.TARGETS: "schema",
    EdgeType.DISCOVERED_IN: "provenance",
    EdgeType.OF_DATASET: "provenance",
}


@dataclass
class GraphNode:
    id: str
    kind: str  # Pattern | Attractor | Dimension | Metric | Dataset | Batch
    label: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.type.value}:{self.source}->{self.target}"

    @property
    def plane(self) -> str:
        return EDGE_PLANE[self.type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "plane": self.plane,
            "weight": self.weight,
            "props": self.props,
        }


def attractor_node_id(attractor_id: int) -> str:
    return f"A-{int(attractor_id)}"


def attractor_id_of(node_id: str) -> int:
    return int(node_id.split("-", 1)[1])


def dataset_node_id(dataset_id: str) -> str:
    return f"DS:{dataset_id}"


def batch_node_id(batch_id: str) -> str:
    return f"B:{batch_id}"


def metric_node_id(dataset_id: str, name: str) -> str:
    return f"M:{dataset_id}:{name}"


def dimension_node_id(dataset_id: str, name: str) -> str:
    return f"D:{dataset_id}:{name}"
