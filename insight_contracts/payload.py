"""The evidence for one question, as the graph service hands it to the narrator (docs/12_architecture.md).

The narrator reads the prompt, the deterministic observation lines, the citation manifest and the metrics; ``view`` is what the
web console draws (evidence cards with the parse, traversal, highlight groups) and passes through the narrator
untouched, so the graph's internal structures never become the narrator's concern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

CITATION_FIELDS = ("pattern_id", "expression", "dataset_id", "filename", "batch_id")


@dataclass
class EvidencePayload:
    question: str
    graph_empty: bool = False  # an ordinary state: nothing ingested yet
    evidence_prompt: str = ""  # the LLM-ready evidence text
    evidence_summary: str = ""  # the cited observation lines; the narrator frames them as the evidence-only answer
    citations: dict[str, dict[str, Any]] = field(default_factory=dict)  # P# -> CITATION_FIELDS
    view: dict[str, Any] = field(default_factory=dict)  # for the console only: evidence, traversal, highlight
    metrics: dict[str, Any] = field(default_factory=dict)  # retrieval_s, seed_count, traversal_depth, ...

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidencePayload:
        """Strict: a payload missing a field is a contract break, not a default."""
        return cls(**{f.name: data[f.name] for f in fields(cls)})
