"""Subgroup miner settings (docs/02_discovery.md, docs/03_insights.md). A host may fill the fields from the environment."""

from __future__ import annotations

from dataclasses import dataclass

from insight_contracts import PhenomenonThresholds


@dataclass(frozen=True)
class MinerConfig:
    thresholds: PhenomenonThresholds = PhenomenonThresholds()

    # ingestion (docs/02_discovery.md §2.1)
    min_rows: int = 50
    bin_columns: str = ""  # "col:q,col2:q" — derive quantile-band categorical dimensions from numerics
    categorical_columns: str = ""  # "Store,Holiday_Flag" — integer/float-coded columns to treat as categorical dimensions

    # spatial ingestion, only for an upload with a geo option (docs/02_discovery.md §2.9)
    geo_resolution: int = 7  # H3 resolution when the geo option names none
    min_cell_points: int = 5  # below it a cell keeps its counters but not its medians and shares (NaN)
    geo_share_max_levels: int = 8  # categoricals with at most this many levels also become per-level share metrics
    geo_mode_max_levels: int = 40  # categoricals with more levels are not aggregated
    geo_lisa_permutations: int = 99
    geo_colocation_min: float = 0.3  # CO_LOCATED edge: overlap of the one-ring-dilated extents

    # discovery (docs/02_discovery.md) — EDA defaults preserved
    compute_budget: int = 5000  # step3 compute_budget
    validation_budget: int = 50  # workflow: top-50 candidates to step4b
    min_search_dimensions: int = 3  # step2 min_categories backfill
    eda_random_seed: int = 42  # seeds step4b bootstrap
    redundancy_jaccard: float = 0.88  # pre-validation near-duplicate threshold (same primary metric and sign)

    # validity rules and evidence weight (docs/03_insights.md §3.2–3.3)
    min_support_rows: int = 30
    min_effect_z: float = 0.5
    min_stability: float = 0.5
    max_p_adjusted: float = 0.05
    weight_effect_ref: float = 1.5
    weight_confidence_ref: float = 6.0  # -log10(p_adj) that saturates confidence
    weight_exponents: tuple = (0.35, 0.25, 0.20, 0.10, 0.10)  # effect, stability, confidence, support, emm
    weight_floor: float = 0.05

    # the subgroup lattice (docs/06_graph_and_storage.md §6.1)
    contrast_min_overlap: float = 0.5
    contrast_min_shift: float = 0.5
