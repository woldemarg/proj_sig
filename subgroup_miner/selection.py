"""Validity rules and insight_weight (docs/03_insights.md §3.2–3.3).

The statistical rules (R1–R3) decide what a finding is; every rejected candidate records the rule that
fired so pruning is observable. How many findings a graph admits (R4 weight floor, R7 batch budget) is
the caller's admission, not the miner's. The weight is a weighted geometric mean of
five normalised evidence factors; it is consumed by the ontology (docs/05_latent_anchors.md §5.5) and
the traversal ranking (docs/07_question_answering.md §7.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from insight_contracts import Insight, Rejection
from subgroup_miner.config import MinerConfig

# volume_preference_score(p) = sqrt(p) * (1 - p) peaks at p = 1/3
VOLUME_UTILITY_MAX = math.sqrt(1.0 / 3.0) * (2.0 / 3.0)
WEIGHT_FACTORS = ("effect", "stability", "confidence", "support", "emm")


def weight_factors(insight: Insight, config: MinerConfig) -> dict[str, float]:
    """Map raw statistics to [0, 1] evidence factors (monotone, dataset-independent).

    For ``covariance`` insights the effect is the EMM divergence, and the median-test
    confidence / bootstrap stability (which measure the *shift*) are not measured by
    the EDA: they are reported as ``None`` and left out of the geometric mean (their
    exponents are renormalised over the measured factors), which is the only neutral
    treatment in a product.
    """
    emm_effect = 1.0 - math.exp(-max(insight.emm_score, 0.0) / config.thresholds.weight_emm_ref)
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


def insight_weight(factors: dict[str, float | None], config: MinerConfig) -> float:
    """w = floor + (1 - floor) * prod_k f_k ** (a_k / sum a), over the measured factors (not None)."""
    exponents = {k: a for k, a in zip(WEIGHT_FACTORS, config.weight_exponents) if factors.get(k) is not None}
    total = sum(exponents.values()) or 1.0
    log_mean = sum((a / total) * math.log(max(factors[k], 1e-3)) for k, a in exponents.items())
    return float(config.weight_floor + (1.0 - config.weight_floor) * math.exp(log_mean))


def with_weight(insight: Insight, config: MinerConfig) -> Insight:
    factors = weight_factors(insight, config)
    return replace(insight, weight=insight_weight(factors, config), weight_factors=factors)


@dataclass
class SelectionResult:
    kept: list[Insight]
    rejections: list[Rejection]


def _as_covariance(ins: Insight) -> Insight:
    """Retarget an EMM-only insight on the divergent pair metric with the larger |shift|.

    Its median shift failed the shift test, so the tests of that shift (bootstrap
    stability, median-test p-values) are dropped: they do not describe a correlation change.
    """
    shifts = [ins.shift_for(m) for m in ins.covariance["pair"]]
    target = sorted((s for s in shifts if s is not None), key=lambda s: (-s.magnitude, s.metric))[0]
    return replace(
        ins,
        phenomenon_type="covariance",
        target=target.metric,
        effect_size=target.robust_z,
        baseline=target.global_median,
        local=target.local_median,
        stability=None,
        p_value=None,
        p_adjusted=None,
    )


def select_insights(insights: list[Insight], config: MinerConfig) -> SelectionResult:
    """Apply the validity rules R1–R3 (docs/03_insights.md §3.2) to validated insights and weigh the survivors,
    heaviest first.

    R5 (identical extents) and R6 (near duplicates) run in discovery, before the
    validation budget is spent (``discovery.merge_identical_extents`` /
    ``prune_near_duplicates``), so every validated insight is a distinct cohort.
    A covariance insight (already selected once) has no shift tests: it is judged on the correlation change only.
    """
    rejections: list[Rejection] = []
    survivors: list[Insight] = []
    for ins in insights:
        covariance = ins.phenomenon_type == "covariance"  # its shift tests are None
        significant = not covariance and abs(ins.effect_size) >= config.min_effect_z and ins.p_adjusted <= config.max_p_adjusted
        stable = not covariance and ins.stability >= config.min_stability
        shift_ok = significant and stable  # R2 + R3: the bootstrap measures the shift score
        emm_ok = ins.emm_score >= config.thresholds.min_emm_score and bool(ins.covariance)
        if not shift_ok and emm_ok:
            # the shift is absent, insignificant or unstable, but the correlation change holds:
            # the phenomenon is the divergent pair (never cited as a median shift)
            ins = _as_covariance(ins)
        ins = with_weight(ins, config)
        if ins.support < config.min_support_rows:  # R1
            rejections.append(Rejection(ins.expression, "min_support", f"support={ins.support}"))
        elif not (shift_ok or emm_ok):  # R2 / R3
            if covariance:
                reason, detail = "weak_effect", f"emm={ins.emm_score:.2f}"
            elif significant:
                reason, detail = "unstable", f"stability={ins.stability:.2f}"
            elif abs(ins.effect_size) >= config.min_effect_z:
                reason, detail = "not_significant", f"|z|={abs(ins.effect_size):.2f} p_adj={ins.p_adjusted:.2g}"
            else:
                reason, detail = "weak_effect", f"|z|={abs(ins.effect_size):.2f} p_adj={ins.p_adjusted:.2g} emm={ins.emm_score:.2f}"
            rejections.append(Rejection(ins.expression, reason, detail))
        else:
            survivors.append(ins)
    return SelectionResult(sorted(survivors, key=lambda i: (-i.weight, i.expression)), rejections)
