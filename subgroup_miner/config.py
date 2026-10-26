"""Subgroup miner settings (docs/02_discovery.md, docs/03_insights.md). A host may fill the fields from the environment."""

from __future__ import annotations

from dataclasses import dataclass

from insight_contracts import PhenomenonThresholds

WEIGHT_FACTORS = ("effect", "stability", "support", "emm")  # the insight weight's factors, in exponent order (docs/03_insights.md §3.3)


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
    weight_exponents: tuple[float, ...] = (0.35, 0.25, 0.10, 0.10)  # one per WEIGHT_FACTORS entry
    weight_floor: float = 0.05

    # trajectories through the time windows of one upload (docs/02_discovery.md §2.10)
    temporal_dormant_windows: int = 3  # evaluated-but-refused windows since the last admission before a trajectory ends
    temporal_unobserved_windows: int = 4  # windows in a row without an evaluation before tracking stops (not a finding)
    temporal_drift_excess: float = 0.5  # |excess shift| (reference-scale units) from which a transition is a drift

    # the subgroup lattice (docs/06_graph_and_storage.md §6.1)
    contrast_min_overlap: float = 0.5
    contrast_min_shift: float = 0.5

    def __post_init__(self) -> None:
        if len(self.weight_exponents) != len(WEIGHT_FACTORS):  # a five-entry setting from before the confidence factor went would shift every weight
            raise ValueError(f"WEIGHT_EXPONENTS needs one entry per factor {WEIGHT_FACTORS}, got {self.weight_exponents}")
