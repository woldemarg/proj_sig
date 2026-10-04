"""Representation types: the canonical form and the embedding contract (docs/04_representation.md).
The insight itself and the graph vocabulary live in ``insight_contracts``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from insight_contracts import stable_hash

# Bumped when stored vectors would no longer be comparable with new ones; a store that
# records another version must not mix its vectors with new ones.
CANONICAL_VERSION = "ltir-canon-4"  # closed-intent scopes; ASCII document; covariance insights cite only the correlation change
REPRESENTATION_VERSION = "ltir-rep-3"  # Qwen3-Embedding-0.6B (MRL 384, re-normalised); single uncentred frame


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
    compute_dtype: str  # dtype the model ran in (its checkpoint dtype: bf16 for Qwen3)
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
