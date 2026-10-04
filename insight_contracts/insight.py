"""The insight: what subgroup mining finds and the attractor topology, the graph and the narrator speak about.

Standard library only. pandas objects never cross a package boundary: a finding leaves the miner as an
``Insight`` and travels as its ``to_record()`` dictionary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def stable_hash(*parts: Any, length: int = 12) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True, order=True)
class Condition:
    """One equality selector ``attribute == value`` of a subgroup scope."""

    attribute: str
    value: str

    @property
    def expr(self) -> str:
        return f"{self.attribute}={self.value}"


@dataclass(frozen=True)
class Shift:
    """Robust median shift of one metric inside the subgroup.

    ``robust_z`` is signed: ``sign(local_median - global_median)`` times the EDA
    ``robust_z_score`` (0.6745 * |Δmedian| / global MAD, capped at 10).
    """

    metric: str
    robust_z: float
    local_median: float
    global_median: float
    global_mad: float
    source: str = "eda"  # eda (step4 top shift) | covariance_pair (adapter, docs/02_discovery.md §2.4)

    @property
    def direction(self) -> int:
        return 1 if self.robust_z > 0 else (-1 if self.robust_z < 0 else 0)

    @property
    def magnitude(self) -> float:
        return abs(self.robust_z)


@dataclass
class Insight:
    """A statistically validated local insight (graph ``Pattern``)."""

    id: str
    dataset_id: str
    batch_id: str
    conditions: tuple[Condition, ...]
    expression: str  # exact EDA selector string (provenance)
    target: str  # primary metric = largest |shift|
    shifts: tuple[Shift, ...]  # EDA top shifts, primary first
    support: int
    support_fraction: float
    baseline: float  # global median of the target
    local: float  # subgroup median of the target
    effect_size: float  # signed robust z of the target
    sd_score: float  # EDA final_sd_score (bootstrap penalised)
    sd_raw_score: float  # EDA sd_aggregate_score (pass 1)
    emm_score: float  # EDA emm_stabilized_score / sqrt(m(m-1)): RMS correlation change per pair
    volume_utility: float
    # tests of the median shift; None for covariance insights (the correlation change has neither)
    stability: float | None  # final_sd / sd_raw = 1 - min(bootstrap CV, 0.9)
    p_value: float | None  # adapter: asymptotic median test on the primary metric
    p_adjusted: float | None  # Bonferroni over (distinct cohorts x metrics)
    drivers: tuple[str, ...]  # EDA root_cause_drivers (confounders)
    row_hash: str  # identity of the covered row set
    phenomenon_type: str = "shift"  # shift | covariance (correlation change, docs/03_insights.md §3.2)
    # strongest diverging correlation pair: {pair:[a,b], local_corr, global_corr, delta}
    covariance: dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()  # redundant expressions collapsed into this one
    weight: float = 0.0  # insight_weight (docs/03_insights.md §3.3)
    weight_factors: dict[str, float | None] = field(default_factory=dict)  # None = not measured
    provenance: dict[str, Any] = field(default_factory=dict)  # dataset_id, batch_id, filename, expression, …

    def shift_for(self, metric: str) -> Shift | None:
        return next((s for s in self.shifts if s.metric == metric), None)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["conditions"] = [asdict(c) for c in self.conditions]
        record["shifts"] = [asdict(s) for s in self.shifts]
        record["drivers"] = list(self.drivers)
        record["aliases"] = list(self.aliases)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Insight:
        data = dict(record)
        data["conditions"] = tuple(Condition(**c) for c in data["conditions"])
        data["shifts"] = tuple(Shift(**s) for s in data["shifts"])
        data["drivers"] = tuple(data.get("drivers", ()))
        data["aliases"] = tuple(data.get("aliases", ()))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def pattern_id(dataset_id: str, conditions: tuple[Condition, ...]) -> str:
    """Deterministic Pattern id: same dataset + same scope -> same id (idempotency)."""
    return "P-" + stable_hash(dataset_id, [c.expr for c in sorted(conditions)])


@dataclass(frozen=True)
class Rejection:
    """A candidate that did not become knowledge, with the rule that fired."""

    expression: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class PhenomenonThresholds:
    """What makes a measurement part of an insight's phenomenon. Mining, representation and retrieval must agree on
    these, so one object carries them to every package (one setting each: min_emm_score, weight_emm_ref, min_component_z)."""

    min_emm_score: float = 0.08  # RMS correlation change per metric pair (after reliability shrinkage) that counts
    weight_emm_ref: float = 0.08  # correlation change that saturates the EMM evidence (same scale as min_emm_score)
    min_component_z: float = 0.5  # |robust z| from which a secondary shift is part of the phenomenon

    def material(self, shift: Shift) -> bool:
        """A shift strong enough to be part of a phenomenon."""
        return shift.magnitude >= self.min_component_z

    def phenomenon_shifts(self, ins: Insight) -> list[Shift]:
        """Shifts that describe the phenomenon: the target always, others if material.

        Empty for a covariance insight: its median shift failed the shift test, so only the
        correlation change is cited (the shifts stay in the record as measurements).
        """
        if ins.phenomenon_type == "covariance":
            return []
        return [s for s in ins.shifts if s.metric == ins.target or self.material(s)]

    def has_material_covariance(self, ins: Insight) -> bool:
        """The correlation change is part of the phenomenon (covariance insight, or emm_score >= min_emm_score)."""
        return bool(ins.covariance) and (ins.phenomenon_type == "covariance" or ins.emm_score >= self.min_emm_score)
