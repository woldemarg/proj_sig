"""Canonical insight representation (docs/04_representation.md §4.1–4.2).

Two contracts live here and never mix:

* **embedding inputs** -- ``scope`` (``attribute = value; ...``), ``target`` and the signed
  ``components`` (labels are metric names or ``correlation between a and b``; the
  magnitude lives only in the coefficient, never in a label). They define the vector.
* **readable text** -- the Markdown ``document()`` and the phrase helpers below, shared
  with the LLM evidence prompt (``ltir/evidence.py``): ASCII only, rounded numbers,
  p-value buckets, shifts in robust standard deviations ("sd").

Structural predicates (scope) and statistical behaviour (phenomenon) are never mixed.
"""

from __future__ import annotations

from collections.abc import Iterable

from ltir.config import Config
from ltir.models import CANONICAL_VERSION, CanonicalInsight, Condition, Insight, Shift


def humanize(name: str) -> str:
    return name.replace("_", " ").strip()


def magnitude_word(z: float) -> str:
    z = abs(z)
    if z < 1.0:
        return "mild"
    if z < 2.0:
        return "moderate"
    if z < 3.0:
        return "strong"
    return "extreme"


def format_value(x: float) -> str:
    """Metric values in text: 4 significant digits below 1,000 (exponent notation below 1e-4), rounded integers with thousands separators above."""
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def format_p(p: float) -> str:
    """Significance in text: '< 0.001' | '< 0.01' | '< 0.05' | two decimals."""
    for cut in (0.001, 0.01, 0.05):
        if p < cut:
            return f"< {cut}"
    return f"{p:.2f}"


def describe_scope(conditions: Iterable[Condition]) -> str:
    """Prose scope: 'category is phones and region is US'; 'a is x, b is y, and c is z'."""
    parts = [f"{c.attribute} is {c.value}" for c in conditions]
    return " and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + f", and {parts[-1]}"


def describe_shift(s: Shift) -> str:
    """'discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall)'."""
    verb = "increase" if s.direction > 0 else "decrease"
    return (
        f"{humanize(s.metric)}: {magnitude_word(s.robust_z)} {verb}, {s.robust_z:+.2f} sd "
        f"(median {format_value(s.local_median)} vs {format_value(s.global_median)} overall)"
    )


def covariance_label(pair: list[str]) -> str:
    a, b = sorted(pair)
    return f"correlation between {humanize(a)} and {humanize(b)}"


def _covariance_change(cov: dict) -> tuple[str, int]:
    """Word + sign for the change of |corr|: strengthens (+1) / weakens or reverses (-1)."""
    before, after = cov["global_corr"], cov["local_corr"]
    if before * after < 0 and abs(before) > 0.1 and abs(after) > 0.1:
        return "reverses", -1
    return ("strengthens", 1) if abs(after) > abs(before) else ("weakens", -1)


def describe_covariance(cov: dict) -> str:
    """'correlation between delivery days and return rate weakens from +0.88 overall to +0.73 in the subgroup'."""
    word, _ = _covariance_change(cov)
    return f"{covariance_label(cov['pair'])} {word} from {cov['global_corr']:+.2f} overall to {cov['local_corr']:+.2f} in the subgroup"


def has_material_covariance(ins: Insight, config: Config) -> bool:
    """The correlation change is part of the phenomenon (covariance insight, or EMM >= MIN_EMM_SCORE)."""
    return bool(ins.covariance) and (ins.phenomenon_type == "covariance" or ins.emm_score >= config.min_emm_score)


def phenomenon_shifts(ins: Insight, config: Config) -> list[Shift]:
    """Shifts that describe the phenomenon: the target always, others if |z| >= MIN_COMPONENT_Z."""
    chosen = [s for s in ins.shifts if s.metric == ins.target or s.magnitude >= config.min_component_z]
    if ins.phenomenon_type == "covariance":
        chosen = [s for s in chosen if s.magnitude >= config.min_component_z]
    return chosen


def canonicalize(ins: Insight, config: Config, dataset_rows: int | None = None) -> CanonicalInsight:
    """Embedding inputs (scope, target, components) + the readable sections of one insight."""
    shifts = phenomenon_shifts(ins, config)
    components: list[tuple[str, float]] = [(humanize(s.metric), s.robust_z) for s in shifts]
    phrases = [describe_shift(s) for s in shifts]

    cov = ins.covariance
    if has_material_covariance(ins, config):
        _, sign = _covariance_change(cov)
        phrases.insert(0 if ins.phenomenon_type == "covariance" else len(phrases), describe_covariance(cov))
        components.append((covariance_label(cov["pair"]), sign * config.emm_component_weight * ins.emm_score / config.weight_emm_ref))
    if ins.phenomenon_type == "covariance" and not shifts:
        phrases.append("no material median shift")

    divergence = f"divergence score {ins.emm_score:.2f}"
    covariance = f"strongest change: {describe_covariance(cov)} ({divergence})" if cov else f"no correlation pair ({divergence})"
    total = f" of {dataset_rows:,}" if dataset_rows else ""
    support = (
        f"{ins.support:,} rows ({ins.support_fraction:.1%}{total}); bootstrap stability {ins.stability:.2f}; adjusted p {format_p(ins.p_adjusted)}"
    )
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


def headline(ins: Insight) -> str:
    """Compact ASCII label (graph node, Neo4j, UI tooltip): 'category=phones, region=US: discount +2.21 sd'."""
    scope = ", ".join(c.expr for c in ins.conditions)
    if ins.phenomenon_type == "covariance" and ins.covariance:
        word, _ = _covariance_change(ins.covariance)
        a, b = sorted(ins.covariance["pair"])
        return f"{scope}: corr({humanize(a)}, {humanize(b)}) {word}"
    return f"{scope}: {humanize(ins.target)} {ins.effect_size:+.2f} sd"
