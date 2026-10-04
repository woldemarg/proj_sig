"""Canonical insight representation (docs/04_representation.md §4.1–4.2).

Two contracts live here and never mix:

* **embedding inputs** -- ``scope`` (``attribute = value; ...``), ``target`` and the signed
  ``components`` (labels are metric names or ``correlation between a and b``; the
  magnitude lives only in the coefficient, never in a label). They define the vector.
* **readable text** -- the Markdown ``document()``, written with the kernel's phrases
  (``insight_contracts.text``), which the LLM evidence prompt shares.

Structural predicates (scope) and statistical behaviour (phenomenon) are never mixed.
"""

from __future__ import annotations

from attractor_topology.config import TopologyConfig
from attractor_topology.models import CANONICAL_VERSION, CanonicalInsight
from insight_contracts import Insight
from insight_contracts.text import (
    covariance_change,
    covariance_label,
    describe_covariance,
    describe_scope,
    describe_shift,
    describe_validation,
    humanize,
)


def canonicalize(ins: Insight, config: TopologyConfig, dataset_rows: int | None = None) -> CanonicalInsight:
    """Embedding inputs (scope, target, components) + the readable sections of one insight."""
    shifts = config.thresholds.phenomenon_shifts(ins)
    components: list[tuple[str, float]] = [(humanize(s.metric), s.robust_z) for s in shifts]
    phrases = [describe_shift(s) for s in shifts]

    cov = ins.covariance
    if config.thresholds.has_material_covariance(ins):
        _, sign = covariance_change(cov)
        phrases.insert(0 if ins.phenomenon_type == "covariance" else len(phrases), describe_covariance(cov))
        components.append((covariance_label(cov["pair"]), sign * config.emm_component_weight * ins.emm_score / config.thresholds.weight_emm_ref))
    if ins.phenomenon_type == "covariance":
        phrases.append("no validated median shift")

    divergence = f"divergence score {ins.emm_score:.2f}"
    covariance = f"strongest change: {describe_covariance(cov)} ({divergence})" if cov else f"no correlation pair ({divergence})"
    total = f" of {dataset_rows:,}" if dataset_rows else ""
    support = f"{ins.support:,} rows ({ins.support_fraction:.1%}{total}); {describe_validation(ins, config.thresholds)}"
    return CanonicalInsight(
        insight_id=ins.id,
        version=CANONICAL_VERSION,
        target=humanize(ins.target),
        scope="; ".join(f"{c.attribute} = {c.value}" for c in ins.conditions),
        scope_sentence=describe_scope(ins.conditions),
        phenomenon="; ".join(phrases),
        covariance=covariance,
        confounders="; ".join(ins.drivers) if ins.drivers else "none detected",
        support=support,
        components=tuple(components),
    )
