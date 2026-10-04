"""Findings as LLM context: the miner used on its own to enrich a prompt (subgroup_miner/README.md)."""

from __future__ import annotations

from collections.abc import Sequence

from insight_contracts import Insight, PhenomenonThresholds
from insight_contracts.text import (
    describe_covariance,
    describe_scope,
    describe_shift,
    describe_validation,
)


def describe(insights: Sequence[Insight], thresholds: PhenomenonThresholds | None = None, *, limit: int = 20) -> str:
    """One numbered line per finding, heaviest first, in the phrases the graph's evidence uses (ASCII, shifts in
    robust sd, adjusted p in buckets) — ready to paste into an LLM prompt as statistical context."""
    thresholds = thresholds or PhenomenonThresholds()
    lines = []
    for n, ins in enumerate(sorted(insights, key=lambda i: (-i.weight, i.id))[:limit], 1):
        phenomenon = [describe_shift(s) for s in thresholds.phenomenon_shifts(ins)]
        if thresholds.has_material_covariance(ins):
            phenomenon.append(describe_covariance(ins.covariance))
        lines.append(
            f"{n}. {describe_scope(ins.conditions)} ({ins.support:,} rows, {ins.support_fraction:.1%}): "
            f"{'; '.join(phenomenon)}. Validation: {describe_validation(ins, thresholds)}; weight {ins.weight:.2f}."
        )
    return "\n".join(lines)
