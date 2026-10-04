"""Subgroup and phenomenon discovery over one table (docs/02_discovery.md, docs/03_insights.md).

A table in, validated and weighted ``Insight`` records out — usable alone (``describe`` turns them into LLM
context) or with ``attractor_topology``. Depends only on ``insight_contracts`` and its numerical libraries.
"""

from subgroup_miner.config import MinerConfig
from subgroup_miner.describe import describe
from subgroup_miner.discovery import DiscoveryError, DiscoveryResult, build_insights, covers_of, run_discovery
from subgroup_miner.ingestion import IngestionError, LoadedDataset, load_dataset
from subgroup_miner.lattice import structural_edges
from subgroup_miner.selection import SelectionResult, select_insights

__all__ = [
    "DiscoveryError",
    "DiscoveryResult",
    "IngestionError",
    "LoadedDataset",
    "MinerConfig",
    "SelectionResult",
    "build_insights",
    "covers_of",
    "describe",
    "load_dataset",
    "run_discovery",
    "select_insights",
    "structural_edges",
]
