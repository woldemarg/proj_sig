"""Insight selection policy and insight_weight (SDD 04 §Selection, §Weight).

Rules run in a fixed order; every rejected candidate records the rule that
fired so pruning is observable. The weight is a weighted geometric mean of
five normalised evidence factors; it is consumed by the ontology (SDD 07) and
the traversal ranking (SDD 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ltir.config import Config
from ltir.models import Insight, Rejection

# volume_preference_score(p) = sqrt(p) * (1 - p) peaks at p = 1/3
VOLUME_UTILITY_MAX = math.sqrt(1.0 / 3.0) * (2.0 / 3.0)
WEIGHT_FACTORS = ("effect", "stability", "confidence", "support", "emm")


def weight_factors(insight: Insight, config: Config) -> dict[str, float]:
    """Map raw statistics to [0, 1] evidence factors (monotone, dataset-independent).

    For ``covariance`` insights the effect is the EMM divergence, and the median-test
    confidence / bootstrap stability (which measure the *shift*) are not measured by
    the EDA: they are reported as ``None`` and left out of the geometric mean (their
    exponents are renormalised over the measured factors), which is the only neutral
    treatment in a product.
    """
    emm_effect = 1.0 - math.exp(-max(insight.emm_score, 0.0) / config.weight_emm_ref)
    if insight.phenomenon_type == "covariance":
        effect, stability, confidence = emm_effect, None, None
    else:
        effect = 1.0 - math.exp(-abs(insight.effect_size) / config.weight_effect_ref)
        stability = float(np.clip(insight.stability, 0.0, 1.0))
        confidence = float(np.clip(-math.log10(max(insight.p_adjusted, 1e-300)) / config.weight_confidence_ref, 0.0, 1.0))
    return {
        "effect": float(effect),
        "stability": None if stability is None else float(stability),
        "confidence": None if confidence is None else float(confidence),
        "support": float(np.clip(insight.volume_utility / VOLUME_UTILITY_MAX, 0.0, 1.0)),
        # EMM is a bonus: no correlation change must not zero the weight -> [0.5, 1]
        "emm": 0.5 + 0.5 * emm_effect,
    }


def insight_weight(factors: dict[str, float | None], config: Config) -> float:
    """w = floor + (1 - floor) * prod_k f_k ** (a_k / sum a), over the measured factors (not None)."""
    exponents = {k: a for k, a in zip(WEIGHT_FACTORS, config.weight_exponents) if factors.get(k) is not None}
    total = sum(exponents.values()) or 1.0
    log_mean = sum((a / total) * math.log(max(factors[k], 1e-3)) for k, a in exponents.items())
    return float(config.weight_floor + (1.0 - config.weight_floor) * math.exp(log_mean))


def with_weight(insight: Insight, config: Config) -> Insight:
    factors = weight_factors(insight, config)
    return replace(insight, weight=insight_weight(factors, config), weight_factors=factors)


@dataclass
class SelectionResult:
    kept: list[Insight]
    rejections: list[Rejection]
    stats: dict[str, int] = field(default_factory=dict)


def _as_covariance(ins: Insight) -> Insight:
    """Retarget an EMM-only insight on the divergent pair metric with the larger |shift|."""
    shifts = [ins.shift_for(m) for m in ins.covariance["pair"]]
    target = sorted((s for s in shifts if s is not None), key=lambda s: (-s.magnitude, s.metric))[0]
    return replace(
        ins,
        phenomenon_type="covariance",
        target=target.metric,
        effect_size=target.robust_z,
        baseline=target.global_median,
        local=target.local_median,
    )


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 1.0


def select_insights(insights: list[Insight], covers: dict[str, np.ndarray], config: Config) -> SelectionResult:
    """Apply the SDD 04 rules R1..R7 to validated insights."""
    rejections: list[Rejection] = []
    survivors: list[Insight] = []
    for ins in insights:
        significant = abs(ins.effect_size) >= config.min_effect_z and ins.p_adjusted <= config.max_p_adjusted
        stable = ins.stability >= config.min_stability
        shift_ok = significant and stable  # R2 + R3: the bootstrap measures the shift score
        emm_ok = ins.emm_score >= config.min_emm_score and bool(ins.covariance)
        if not shift_ok and emm_ok:
            # the shift is absent, insignificant or unstable, but the correlation change holds:
            # the phenomenon is the divergent pair (never cited as a median shift)
            ins = _as_covariance(ins)
        ins = with_weight(ins, config)
        if ins.support < config.min_support_rows:  # R1
            rejections.append(Rejection(ins.expression, "min_support", f"support={ins.support}"))
        elif not (shift_ok or emm_ok):  # R2 / R3
            if significant:
                reason, detail = "unstable", f"stability={ins.stability:.2f}"
            elif abs(ins.effect_size) >= config.min_effect_z:
                reason, detail = "not_significant", f"|z|={abs(ins.effect_size):.2f} p_adj={ins.p_adjusted:.2g}"
            else:
                reason, detail = "weak_effect", f"|z|={abs(ins.effect_size):.2f} p_adj={ins.p_adjusted:.2g} emm={ins.emm_score:.2f}"
            rejections.append(Rejection(ins.expression, reason, detail))
        elif ins.weight < config.min_insight_weight:  # R4
            rejections.append(Rejection(ins.expression, "low_weight", f"weight={ins.weight:.2f}"))
        else:
            survivors.append(ins)

    # R5 cover equivalence: identical extents -> keep the closed pattern (maximal intent)
    groups: dict[str, list[Insight]] = {}
    for ins in survivors:
        groups.setdefault(ins.row_hash, []).append(ins)
    closed: list[Insight] = []
    for members in groups.values():
        members.sort(key=lambda i: (-len(i.conditions), -i.weight, i.expression))
        head, rest = members[0], members[1:]
        for other in rest:
            rejections.append(Rejection(other.expression, "cover_equivalent", f"same rows as {head.expression}"))
        closed.append(replace(head, aliases=head.aliases + tuple(o.expression for o in rest)))

    # R6 near duplicates: same target + direction and Jaccard(rows) >= threshold -> keep heavier
    closed.sort(key=lambda i: (-i.weight, i.expression))
    kept: list[Insight] = []
    for ins in closed:
        rows = np.unique(covers[ins.expression])
        dup = next(
            (
                k
                for k in kept
                if k.target == ins.target
                and k.direction == ins.direction
                and _jaccard(np.unique(covers[k.expression]), rows) >= config.redundancy_jaccard
            ),
            None,
        )
        if dup is None:
            kept.append(ins)
            continue
        rejections.append(Rejection(ins.expression, "near_duplicate", f"Jaccard>={config.redundancy_jaccard} with {dup.expression}"))
        idx = kept.index(dup)
        kept[idx] = replace(dup, aliases=dup.aliases + (ins.expression,))

    # R7 budget
    for ins in kept[config.max_insights_per_batch :]:
        rejections.append(Rejection(ins.expression, "budget", f"max_insights_per_batch={config.max_insights_per_batch}"))
    kept = kept[: config.max_insights_per_batch]

    stats: dict[str, int] = {"input": len(insights), "kept": len(kept)}
    for rej in rejections:
        stats[rej.reason] = stats.get(rej.reason, 0) + 1
    return SelectionResult(kept, rejections, stats)
