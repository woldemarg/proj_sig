"""Dense representation of insights and the living attractor space (docs/04_representation.md, docs/05_latent_anchors.md).

Insights in, a tripartite unit vector per insight and the attractors they activate out — the anchors persist in a
state folder and move with every batch. Usable alone (any source of ``Insight`` records) or after ``subgroup_miner``.
Depends only on ``insight_contracts`` and its numerical libraries.
"""

from attractor_topology.canonical import canonicalize
from attractor_topology.config import TopologyConfig
from attractor_topology.encoder import InsightEncoder, SentenceTransformerEmbedder, TextEmbedder
from attractor_topology.models import CANONICAL_VERSION, REPRESENTATION_VERSION, CanonicalInsight, EmbeddingSpec
from attractor_topology.ontology import Attractor, LatentOntology, OntologyUpdate

__all__ = [
    "CANONICAL_VERSION",
    "REPRESENTATION_VERSION",
    "Attractor",
    "CanonicalInsight",
    "EmbeddingSpec",
    "InsightEncoder",
    "LatentOntology",
    "OntologyUpdate",
    "SentenceTransformerEmbedder",
    "TextEmbedder",
    "TopologyConfig",
    "canonicalize",
]
