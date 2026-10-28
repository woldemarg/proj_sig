"""The dual graph's vocabulary and its snapshot contract (docs/06_graph_and_storage.md §6.1–6.2).

A snapshot is ``{"version": SNAPSHOT_VERSION, "created_at", "representation", "nodes": [...], "edges": [...],
"own_cells": {...}, "trajectories": {...}, "stats": {...}}``; the graph service compiles it, the query engine and the console read it. Node
``props`` by kind, as the readers rely on them:

* ``Pattern``   the ``Insight.to_record()`` fields, plus ``row_id`` (its journal row), ``canonical`` (with ``document``
                and ``components``) and ``embedding`` (the vector contract)
* ``Attractor`` ``attractor_id``, ``mass``, ``n_patterns``, ``distinct_scopes``, ``signature`` [{component, value}],
                ``targets``, ``datasets``, ``dimensions``, ``dispersion``, ``centroid``
* ``Region``    ``resolution``, ``h3``, ``cells_covered`` (geo patterns' cells inside it), ``patterns`` (IN_REGION members
                of both bases, and the geo patterns whose own cell it is)
* ``Metric``    ``name``, ``dataset_id``, ``global_median``, ``global_mad``, ``indicator`` / ``unit`` when declared
* ``Dimension`` ``name``, ``dataset_id``, ``cardinality``, ``entropy``
* ``Dataset`` / ``Batch``  provenance (file, rows, batch sequence); a dataset with a time column ``temporal``
                {column, grid, span}
* ``TemporalNode`` ``granularity``, ``period``, ``t_start``, ``t_end``: a calendar period, the same node for every
                dataset that has an event or a core end in it, and every period above such a one (its week, month,
                quarter: one chain, ``subgroup_miner.temporal.parent_period``); the bounds are the first dataset's calendar

Edge ``props``: ``ACTIVATES`` carries ``alignment``, ``strength`` and ``weak`` (coverage only, never walked);
``RELATED_TO`` carries ``kind: mutual_knn``; ``CO_OCCURS`` carries ``kind: co_occurrence`` and ``shared`` (member insights);
``CO_LOCATED`` (geo patterns of one resolution, any datasets) carries ``relation`` (overlaps | adjacent | same_region);
``IN_REGION`` carries ``basis``: ``cells`` for a geo pattern, with ``share`` (of the region's occupied cells it covers),
``events`` for a base (table-level) pattern, with ``rule`` (lift | share), ``share`` (of its located events), ``lift``,
``n``, ``q`` and ``placed_by`` (the geo level whose points place it); the two shares do not compare. ``PARENT_OF`` runs
from a region to its parent at the next coarser listed level. A geo ``Pattern`` also carries ``compactness``, a base one
``geo: partial`` when too few of its events are located, both derived at the commit. ``own_cells`` maps a geo pattern
whose resolution is a listed region level to its H3 cells: its regions at that level, which no IN_REGION repeats.
``PEAKED_AT`` / ``DROPPED_AT`` (the temporal plane, never walked) run from a pattern to the TemporalNode of a peak or a
dip of its series (docs/02_discovery.md §2.10): ``metric``, ``period``, ``z_raw``, ``z_smoothed``, ``prominence``, ``p``,
``metrics``, and when it has them ``carrier`` (the condition inside the pattern that carries the event: ``attribute``,
``value``, ``rows``, ``of``, ``shift``, ``p``) and ``echo_of`` (the broader pattern of the dataset with the same event).
``STARTS_AT`` / ``ENDS_AT`` (the temporal plane too) run from a pattern to the TemporalNode of the first / last bin of
its core — the shortest run of bins holding most of its rows, where it lives in time — at its dataset's grid, with
``period``, ``share`` (of its timed rows the core holds) and ``bins``; the one hub twice for a core of one bin.
``WITHIN`` runs from a TemporalNode to the period above it (a day's week, a week's month, a month's quarter), never
walked, read by the console and the mirror; who is active in a period is read from the cores, not from an edge.
``trajectories`` maps a pattern of a dataset with a time column to its ``column``, ``grid``, ``tz``, ``span``, ``metric``, ``series`` and
``smoothed`` (per metric, one entry per bin), ``events``, ``trend``, ``current``, ``bins_measured``, ``of``, ``gaps`` and
``core`` ({start, end, t_start, t_end, share, bins, first, last} | None); the pattern carries the short form as
``temporal_summary`` (its ``core`` without the bounds and indices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SNAPSHOT_VERSION = 14  # the snapshot layout (docs/11_reference.md §11.5)


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
    PEAKED_AT = "PEAKED_AT"
    DROPPED_AT = "DROPPED_AT"
    STARTS_AT = "STARTS_AT"
    ENDS_AT = "ENDS_AT"
    WITHIN = "WITHIN"


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
    EdgeType.PEAKED_AT: "temporal",
    EdgeType.DROPPED_AT: "temporal",
    EdgeType.STARTS_AT: "temporal",
    EdgeType.ENDS_AT: "temporal",
    EdgeType.WITHIN: "temporal",
}


@dataclass
class GraphNode:
    id: str
    kind: str  # Pattern | Attractor | Region | TemporalNode | Dimension | Metric | Dataset | Batch
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


TIME_GRIDS = ("day", "week", "month", "quarter")  # the calendar units an upload's time column is binned by


def temporal_node_id(grid: str, period: str) -> str:
    """A calendar period of a grid (``T:week:2026-W35``), the same node for every dataset that has an event or a core end
    in it, and for the periods above those (``WITHIN``)."""
    return f"T:{grid}:{period}"
