"""Graph query engine settings: grounding, seeds, traversal, evidence (docs/07_question_answering.md).
A host may fill the fields from the environment."""

from __future__ import annotations

from dataclasses import dataclass

from insight_contracts import PhenomenonThresholds


@dataclass(frozen=True)
class QueryConfig:
    thresholds: PhenomenonThresholds = PhenomenonThresholds()

    # dense literal grounding floor on catalog-centred cosines, embedder-specific (§7.1.1): Qwen3 0.30
    grounding_min_cosine: float = 0.30
    seed_top_k: int = 3
    seed_min_score: float = 0.25
    seed_relative_min: float = 0.75  # seeds must score >= this fraction of the best seed
    max_latent_hops: int = 1
    structural_hops: int = 1
    traversal_max_depth: int = 5
    max_retrieved: int = 12
    structural_edge_decay: float = 0.85
    traversal_structural_edges: str = "SPECIALIZES,GENERALIZES,CONTRASTS"  # SIBLING stays in the graph
    evidence_max_patterns: int = 10

    @property
    def lattice_edges(self) -> frozenset[str]:
        """Structural edge types the traversal follows (``traversal_structural_edges``)."""
        return frozenset(t.strip() for t in self.traversal_structural_edges.split(",") if t.strip())
