"""Shared kernel of SIG (docs/12_architecture.md): the insight, the graph vocabulary and the text an insight reads as.

Standard library only, so every package and service can depend on it without pulling a framework or a numerical
library: ``subgroup_miner`` produces insights, ``attractor_topology`` and ``graph_query_engine`` consume them, the
two services exchange them.
"""

from insight_contracts.graph import (
    LATENT_EDGES,
    SNAPSHOT_VERSION,
    STRUCTURAL_EDGES,
    UNDIRECTED_EDGES,
    EdgeType,
    GraphEdge,
    GraphNode,
    attractor_id_of,
    attractor_node_id,
    batch_node_id,
    dataset_node_id,
    dimension_node_id,
    metric_node_id,
    region_node_id,
)
from insight_contracts.insight import (
    Condition,
    Insight,
    PhenomenonThresholds,
    Rejection,
    Shift,
    SpatialProvenance,
    pattern_id,
    record_digest,
    stable_hash,
)
from insight_contracts.payload import EvidencePayload

__all__ = [
    "LATENT_EDGES",
    "SNAPSHOT_VERSION",
    "STRUCTURAL_EDGES",
    "UNDIRECTED_EDGES",
    "Condition",
    "EdgeType",
    "EvidencePayload",
    "GraphEdge",
    "GraphNode",
    "Insight",
    "PhenomenonThresholds",
    "Rejection",
    "Shift",
    "SpatialProvenance",
    "attractor_id_of",
    "attractor_node_id",
    "batch_node_id",
    "dataset_node_id",
    "dimension_node_id",
    "metric_node_id",
    "region_node_id",
    "pattern_id",
    "record_digest",
    "stable_hash",
]
