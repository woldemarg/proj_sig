"""Canonical data contracts shared by every LTIR module (docs/03_insights.md, docs/04_representation.md).

Everything downstream of the discovery adapter speaks these types; pandas
objects never cross a module boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

# Bumped when stored vectors would no longer be comparable with new ones; a workspace
# built with another version is refused (``representation_mismatch``) instead of mixed.
CANONICAL_VERSION = "ltir-canon-4"  # closed-intent scopes; ASCII document; covariance insights cite only the correlation change
REPRESENTATION_VERSION = "ltir-rep-3"  # Qwen3-Embedding-0.6B (MRL 384, re-normalised); single uncentred frame


def utc_now() -> str:
    """Timestamps of records, snapshots and logs: ISO 8601, UTC, whole seconds."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def stable_hash(*parts: Any, length: int = 12) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


# Statistical insight (Pattern) model


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
    """A statistically validated, persisted local insight (graph ``Pattern``)."""

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
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def scope_expr(self) -> str:
        return " AND ".join(c.expr for c in self.conditions)

    @property
    def direction(self) -> int:
        return 1 if self.effect_size > 0 else (-1 if self.effect_size < 0 else 0)

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
    """A candidate that did not become persistent knowledge, with the rule that fired."""

    expression: str
    reason: str
    detail: str = ""


# Canonical representation + embeddings


@dataclass(frozen=True)
class CanonicalInsight:
    """Six-section canonical form; scope/target/phenomenon stay separately accessible."""

    insight_id: str
    version: str
    target: str
    scope: str  # embedding input: "attribute = value; ..."
    scope_sentence: str  # readable: "attribute is value and ..."
    phenomenon: str
    covariance: str
    confounders: str
    support: str
    # signed phenomenon components used by the encoder: (label, signed magnitude)
    components: tuple[tuple[str, float], ...]

    def document(self) -> str:
        """Readable Markdown rendering (humans, the naive text-NN baseline); never an embedding input."""
        return "\n".join(
            [
                f"### Subgroup finding {self.insight_id}",
                f"* Scope: {self.scope_sentence}",
                f"* Target metric: {self.target}",
                f"* Observed shift: {self.phenomenon}",
                f"* Metric relationships: {self.covariance}",
                f"* Confounders: {self.confounders}",
                f"* Validation: {self.support}",
            ]
        )


@dataclass(frozen=True)
class EmbeddingSpec:
    """Contract that makes a stored vector reproducible."""

    model_id: str
    model_revision: str  # pinned checkpoint revision ("" = unknown)
    truncate_dim: int  # Matryoshka truncation (0 = native width)
    query_instruction: str  # prefix for free question text ("" = none)
    block_dim: int
    dim: int
    dtype: str  # storage dtype of the vectors
    compute_dtype: str  # dtype the model ran in (its checkpoint dtype: bf16 for Qwen3, fp32 for MiniLM)
    normalization: str
    block_weights: tuple[float, float, float]
    emm_component_weight: float
    min_component_z: float  # which shifts become components
    min_emm_score: float  # whether the correlation change is a component
    weight_emm_ref: float  # scale of the correlation component
    canonical_version: str
    representation_version: str

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self), length=10)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fingerprint"] = self.fingerprint
        return out


# Graph


class EdgeType(str, Enum):
    SPECIALIZES = "SPECIALIZES"
    GENERALIZES = "GENERALIZES"
    SIBLING = "SIBLING"
    CONTRASTS = "CONTRASTS"
    HAS_SCOPE = "HAS_SCOPE"
    TARGETS = "TARGETS"
    ACTIVATES = "ACTIVATES"
    RELATED_TO = "RELATED_TO"
    DISCOVERED_IN = "DISCOVERED_IN"
    OF_DATASET = "OF_DATASET"


STRUCTURAL_EDGES = {EdgeType.SPECIALIZES, EdgeType.GENERALIZES, EdgeType.SIBLING, EdgeType.CONTRASTS}
UNDIRECTED_EDGES = {EdgeType.SIBLING, EdgeType.CONTRASTS, EdgeType.RELATED_TO}
EDGE_PLANE = {
    EdgeType.SPECIALIZES: "structural",
    EdgeType.GENERALIZES: "structural",
    EdgeType.SIBLING: "structural",
    EdgeType.CONTRASTS: "structural",
    EdgeType.RELATED_TO: "latent",
    EdgeType.ACTIVATES: "bridge",
    EdgeType.HAS_SCOPE: "schema",
    EdgeType.TARGETS: "schema",
    EdgeType.DISCOVERED_IN: "provenance",
    EdgeType.OF_DATASET: "provenance",
}


@dataclass
class GraphNode:
    id: str
    kind: str  # Pattern | Attractor | Dimension | Metric | Dataset | Batch
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


def activation_record(
    pattern: str,
    attractor: int,
    alignment: float,
    insight_weight: float,
    source: str,
    engine_weight: float,
    batch_id: str,
    *,
    row_id: int,
    weak: bool = False,
) -> dict[str, Any]:
    """Contract of one ACTIVATES edge (journal + graph)."""
    return {
        "pattern_id": pattern,
        "attractor_id": int(attractor),
        "alignment": float(alignment),
        "strength": float(alignment * insight_weight),
        "engine_weight": float(engine_weight),
        "source": source,  # cold_start | assign | omp | absorbed | nearest | reroute
        "weak": weak,  # rerouted below MIN_ACTIVATION_ALIGNMENT (coverage only)
        "batch_id": batch_id,
        "row_id": int(row_id),
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


def dimension_node_id(dataset_id: str, name: str) -> str:
    return f"D:{dataset_id}:{name}"


@dataclass(frozen=True)
class LatentFrame:
    """Committed vectors of the single insight frame: pattern id -> unit vector, attractor id -> centroid,
    plus the canonical-document embeddings the naive text baseline compares questions with."""

    patterns: dict[str, np.ndarray]
    attractors: dict[int, np.ndarray]
    documents: dict[str, np.ndarray]
