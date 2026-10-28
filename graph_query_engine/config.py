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
    max_latent_hops: int = 1
    traversal_max_depth: int = 5
    max_retrieved: int = 12
    # a finding's freshness halves every this many days from its core to the reference date (§7.2); 0 switches it off
    temporal_recency_half_life_days: int = 90
