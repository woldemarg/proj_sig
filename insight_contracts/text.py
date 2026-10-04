"""Insight -> readable text, shared by every package that talks about an insight (docs/04_representation.md §4.2).

English phrases in ASCII (only the data literals and the ↑/↓ of the compact graph labels may leave it), rounded
numbers, p-value buckets, shifts in robust standard deviations ("sd"). The canonical form
(``attractor_topology.canonical``), the evidence prompt (``graph_query_engine.evidence``) and the miner's LLM
context (``subgroup_miner.describe``) are built from these phrases, so the same finding reads the same everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable

from insight_contracts.insight import Condition, Insight, PhenomenonThresholds, Shift


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


def covariance_change(cov: dict) -> tuple[str, int]:
    """Word + sign for the change of |corr|: strengthens (+1) / weakens or reverses (-1)."""
    before, after = cov["global_corr"], cov["local_corr"]
    if before * after < 0 and abs(before) > 0.1 and abs(after) > 0.1:
        return "reverses", -1
    return ("strengthens", 1) if abs(after) > abs(before) else ("weakens", -1)


def describe_covariance(cov: dict) -> str:
    """'correlation between delivery days and return rate weakens from +0.88 overall to +0.73 in the subgroup'."""
    word, _ = covariance_change(cov)
    return f"{covariance_label(cov['pair'])} {word} from {cov['global_corr']:+.2f} overall to {cov['local_corr']:+.2f} in the subgroup"


def describe_validation(ins: Insight, thresholds: PhenomenonThresholds) -> str:
    """What validated the insight: '... stability 0.97; adjusted p < 0.001' or the correlation change."""
    if ins.phenomenon_type == "covariance":
        return f"correlation change (divergence score {ins.emm_score:.2f} >= {thresholds.min_emm_score:g}); no median test"
    return f"bootstrap stability {ins.stability:.2f}; adjusted p {format_p(ins.p_adjusted)}"


def describe_component(label: str, value: float) -> str:
    """One entry of an anchor's signature in prose (the LLM prompt): 'discount up', 'correlation between a and b weakens'."""
    if label.startswith("correlation between"):
        return f"{label} {'strengthens' if value > 0 else 'weakens'}"
    return f"{label} {'up' if value > 0 else 'down'}"


def component_label(label: str, value: float) -> str:
    """The same entry, compact (graph labels): 'discount ↑', 'corr(a~b) weakens'."""
    if label.startswith("correlation between"):
        pair = label.removeprefix("correlation between ").replace(" and ", "~")
        return f"corr({pair}) {'strengthens' if value > 0 else 'weakens'}"
    return f"{label} {'↑' if value > 0 else '↓'}"


def headline(ins: Insight) -> str:
    """Compact ASCII label (graph node, Neo4j, UI tooltip): 'category=phones, region=US: discount +2.21 sd'."""
    scope = ", ".join(c.expr for c in ins.conditions)
    if ins.phenomenon_type == "covariance" and ins.covariance:
        word, _ = covariance_change(ins.covariance)
        a, b = sorted(ins.covariance["pair"])
        return f"{scope}: corr({humanize(a)}, {humanize(b)}) {word}"
    return f"{scope}: {humanize(ins.target)} {ins.effect_size:+.2f} sd"
