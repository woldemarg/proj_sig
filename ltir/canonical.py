"""Canonical insight representation (SDD 05).

Serialises an ``Insight`` into six sections — TARGET, SCOPE, PHENOMENON,
COVARIANCE, CONFOUNDERS, SUPPORT — and derives the *signed phenomenon
components* the encoder composes (SDD 06). Structural predicates (scope) and
statistical behaviour (phenomenon) are never mixed in one string.
"""

from __future__ import annotations

from ltir.config import Config
from ltir.models import CANONICAL_VERSION, CanonicalInsight, Insight, Shift


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


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _shift_phrase(s: Shift) -> str:
    verb = "increase" if s.direction > 0 else "decrease"
    return (
        f"{humanize(s.metric)} {magnitude_word(s.robust_z)} {verb} "
        f"(robust z {s.robust_z:+.2f}; median {_fmt(s.local_median)} vs {_fmt(s.global_median)})"
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


def phenomenon_shifts(ins: Insight, config: Config) -> list[Shift]:
    """Shifts that describe the phenomenon: the target always, others if |z| >= MIN_COMPONENT_Z."""
    chosen = [s for s in ins.shifts if s.metric == ins.target or s.magnitude >= config.min_component_z]
    if ins.phenomenon_type == "covariance":
        chosen = [s for s in chosen if s.magnitude >= config.min_component_z]
    return chosen


def canonicalize(ins: Insight, config: Config, dataset_rows: int | None = None) -> CanonicalInsight:
    shifts = phenomenon_shifts(ins, config)
    components: list[tuple[str, float]] = [(humanize(s.metric), s.robust_z) for s in shifts]
    phrases = [_shift_phrase(s) for s in shifts]

    cov = ins.covariance
    if cov and (ins.phenomenon_type == "covariance" or ins.emm_score >= config.min_emm_score):
        word, sign = _covariance_change(cov)
        label = covariance_label(cov["pair"])
        phrases.insert(
            0 if ins.phenomenon_type == "covariance" else len(phrases),
            f"{label} {word} ({cov['global_corr']:+.2f} -> {cov['local_corr']:+.2f})",
        )
        components.append((label, sign * config.emm_component_weight * ins.emm_score / config.weight_emm_ref))
    if ins.phenomenon_type == "covariance" and not shifts:
        phrases.append("no material median shift")

    if cov:
        covariance = (
            f"stabilized correlation divergence {ins.emm_score:.3f}; strongest pair "
            f"{' ~ '.join(humanize(m) for m in cov['pair'])} {cov['global_corr']:+.2f} -> {cov['local_corr']:+.2f}"
        )
    else:
        covariance = f"stabilized correlation divergence {ins.emm_score:.3f}"
    total = f" of {dataset_rows:,}" if dataset_rows else ""
    support = f"{ins.support:,} rows ({ins.support_fraction:.1%}{total}); bootstrap stability {ins.stability:.2f}; adjusted p {ins.p_adjusted:.2g}"
    return CanonicalInsight(
        insight_id=ins.id,
        version=CANONICAL_VERSION,
        target=humanize(ins.target),
        scope="; ".join(f"{c.attribute} = {c.value}" for c in ins.conditions),
        phenomenon="; ".join(phrases),
        covariance=covariance,
        confounders="; ".join(ins.drivers) if ins.drivers else "none detected",
        support=support,
        components=tuple(components),
    )


def headline(ins: Insight) -> str:
    """One-line human description (UI labels, attractor labels, evidence)."""
    scope = ", ".join(f"{c.attribute}={c.value}" for c in ins.conditions)
    if ins.phenomenon_type == "covariance" and ins.covariance:
        word, _ = _covariance_change(ins.covariance)
        return f"{scope}: {covariance_label(ins.covariance['pair'])} {word}"
    arrow = "↑" if ins.direction > 0 else "↓"
    return f"{scope}: {humanize(ins.target)} {arrow} ({ins.effect_size:+.2f} z)"
