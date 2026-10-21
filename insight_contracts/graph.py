"""The dual graph's vocabulary and its snapshot contract (docs/06_graph_and_storage.md §6.1–6.2).

A snapshot is ``{"version": SNAPSHOT_VERSION, "nodes": [...], "edges": [...], "stats": {...}}``; the graph service
compiles it, the query engine and the console read it. Node ``props`` by kind, as the readers rely on them:

* ``Pattern``   the ``Insight.to_record()`` fields, plus ``row_id`` (its journal row), ``canonical`` (with ``document``
                and ``components``) and ``embedding`` (the vector contract)
* ``Attractor`` ``attractor_id``, ``mass``, ``n_patterns``, ``distinct_scopes``, ``signature`` [{component, value}],
                ``targets``, ``datasets``, ``dimensions``, ``dispersion``, ``centroid``
* ``Region``    ``resolution``, ``h3``, ``cells_covered`` (geo patterns' cells inside it), ``patterns`` (IN_REGION members
                of both bases)
* ``Metric``    ``name``, ``dataset_id``, ``global_median``, ``global_mad``, ``indicator`` / ``unit`` when declared
* ``Dimension`` ``name``, ``dataset_id``, ``cardinality``, ``entropy``
* ``Dataset`` / ``Batch``  provenance (file, rows, batch sequence)

Edge ``props``: ``ACTIVATES`` carries ``alignment``, ``strength`` and ``weak`` (coverage only, never walked);
``RELATED_TO`` carries ``kind: mutual_knn``; ``CO_OCCURS`` carries ``kind: co_occurrence`` and ``shared`` (member insights);
``CO_LOCATED`` (geo patterns of one resolution, any datasets) carries ``relation`` (overlaps | adjacent | same_region);
``IN_REGION`` carries ``basis``: ``cells`` for a geo pattern, with ``share`` (of the region's occupied cells it covers),
``events`` for a base (table-level) pattern, with ``rule`` (lift | share), ``share`` (of its located events), ``lift``,
``n``, ``q`` and ``placed_by`` (the geo level whose points place it); the two shares do not compare. ``PARENT_OF`` runs
from a region to its parent at the next coarser listed level. A geo ``Pattern`` also carries ``compactness``, a base one
``geo: partial`` when too few of its events are located, both derived at the commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SNAPSHOT_VERSION = 10  # the snapshot layout (docs/11_reference.md §11.5)


class EdgeType(str, Enum):
    SPECIALIZES = "SPECIALIZES"
    SIBLING = "SIBLING"
    CONTRASTS = "CONTRASTS"
    HAS_SCOPE = "HAS_SCOPE"
    TARGETS = "TARGETS"
    ACTIVATES = "ACTIVATES"
    RELATED_TO = "RELATED_TO"
    CO_OCCURS = "CO_OCCURS"
    CO_LOCATED = "CO_LOCATED"
    IN_REGION = "IN_REGION"
    PARENT_OF = "PARENT_OF"
    DISCOVERED_IN = "DISCOVERED_IN"
    OF_DATASET = "OF_DATASET"


STRUCTURAL_EDGES = {EdgeType.SPECIALIZES, EdgeType.SIBLING, EdgeType.CONTRASTS, EdgeType.CO_LOCATED}  # pattern <-> pattern
LATENT_EDGES = {EdgeType.RELATED_TO, EdgeType.CO_OCCURS}  # anchor <-> anchor: centroid neighbours, shared members
UNDIRECTED_EDGES = {EdgeType.SIBLING, EdgeType.CONTRASTS, EdgeType.CO_LOCATED, EdgeType.RELATED_TO, EdgeType.CO_OCCURS}
EDGE_PLANE = {
    EdgeType.SPECIALIZES: "structural",
    EdgeType.SIBLING: "structural",
    EdgeType.CONTRASTS: "structural",
    EdgeType.CO_LOCATED: "spatial",
    EdgeType.IN_REGION: "spatial",
    EdgeType.PARENT_OF: "spatial",
    EdgeType.RELATED_TO: "latent",
    EdgeType.CO_OCCURS: "latent",
    EdgeType.ACTIVATES: "bridge",
    EdgeType.HAS_SCOPE: "schema",
    EdgeType.TARGETS: "schema",
    EdgeType.DISCOVERED_IN: "provenance",
    EdgeType.OF_DATASET: "provenance",
}


@dataclass
class GraphNode:
    id: str
    kind: str  # Pattern | Attractor | Region | Dimension | Metric | Dataset | Batch
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


def region_node_id(resolution: int, cell: str) -> str:
    """A region is an H3 cell at a listed resolution, the same node for every dataset that reaches it."""
    return f"R:{resolution}:{cell}"


def dimension_node_id(dataset_id: str, name: str) -> str:
    return f"D:{dataset_id}:{name}"
