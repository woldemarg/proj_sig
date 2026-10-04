"""Selection rules and insight_weight (docs/03_insights.md)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from conftest import toy_insight

from insight_contracts import Insight, Shift
from insight_graph_service.core.batch import admit_insights
from subgroup_miner.config import MinerConfig
from subgroup_miner.selection import insight_weight, select_insights, weight_factors, with_weight


def make_insight(expr="a=1", conds=(("a", "1"),), z=2.0, stability=0.9, p=1e-10, vu=0.25, emm=0.05, support=300, row_hash=None, target="m"):
    return toy_insight(
        sorted(conds),
        [Shift(target, z, 1.0, 0.0, 1.0)],
        id="P-" + expr,
        expression=expr,
        support=support,
        sd_score=abs(z) * stability,
        sd_raw_score=abs(z),
        emm_score=emm,
        volume_utility=vu,
        stability=stability,
        p_value=p,
        p_adjusted=p,
        row_hash=row_hash or expr,
        covariance={"pair": [target, "n"], "local_corr": 0.1, "global_corr": -0.5, "delta": 0.6},
    )


def test_weight_bounded_and_monotone():
    cfg = MinerConfig()
    base = make_insight()
    w = with_weight(base, cfg).weight
    assert cfg.weight_floor <= w <= 1.0
    assert with_weight(replace(base, effect_size=3.0), cfg).weight > w  # stronger effect
    assert with_weight(replace(base, stability=0.5), cfg).weight < w  # less stable
    assert with_weight(replace(base, p_adjusted=1e-2), cfg).weight < w  # less significant
    assert with_weight(replace(base, emm_score=0.5), cfg).weight > w  # EMM bonus
    f = weight_factors(base, cfg)
    assert set(f) == {"effect", "stability", "confidence", "support", "emm"} and all(0 <= v <= 1 for v in f.values())
    # formula: weighted geometric mean with floor
    expected = cfg.weight_floor + (1 - cfg.weight_floor) * np.prod([f[k] ** a for k, a in zip(f, cfg.weight_exponents)])
    assert insight_weight(f, cfg) == pytest.approx(expected)


def test_rules_record_reasons():
    cfg = MinerConfig()
    items = [
        make_insight("ok", (("a", "1"),)),
        make_insight("small", (("a", "2"),), support=5),
        make_insight("weak", (("a", "3"),), z=0.2, emm=0.0),
        make_insight("unsig", (("a", "4"),), p=0.5, emm=0.0),
        make_insight("unstable", (("a", "5"),), stability=0.2, emm=0.0),
    ]
    res = select_insights(items, cfg)
    reasons = {r.expression: r.reason for r in res.rejections}
    assert [i.expression for i in res.kept] == ["ok"]
    assert reasons == {"small": "min_support", "weak": "weak_effect", "unsig": "not_significant", "unstable": "unstable"}


def test_admission_keeps_the_heaviest_within_the_budget():
    """The graph's admission (R4 weight floor, R7 batch budget) runs on the miner's valid insights."""
    miner = MinerConfig()
    strong, weak = make_insight("strong", (("a", "1"),), z=3.0), make_insight("weak", (("a", "2"),), z=1.0)
    valid = select_insights([weak, strong], miner).kept
    kept, refused = admit_insights(valid, 0.2, 1)
    assert [i.expression for i in kept] == ["strong"] and [(r.expression, r.reason) for r in refused] == [("weak", "budget")]
    kept, refused = admit_insights(valid, 0.99, 200)
    assert kept == [] and {r.reason for r in refused} == {"low_weight"}


def test_emm_only_insight_becomes_covariance():
    cfg = MinerConfig()
    ins = make_insight("cov", z=0.1, emm=0.4, p=0.5)
    ins = replace(ins, shifts=(Shift("m", 0.1, 0, 0, 1), Shift("n", 0.3, 0, 0, 1)))
    res = select_insights([ins], cfg)
    kept = res.kept[0]
    assert kept.phenomenon_type == "covariance" and kept.target == "n"  # pair metric with larger |shift|
    assert kept.weight_factors["confidence"] is None and kept.weight_factors["stability"] is None  # unmeasured: left out
    # the shift test failed, so its stability and p-values do not describe the insight
    assert kept.stability is None and kept.p_value is None and kept.p_adjusted is None


def test_selection_is_idempotent_on_covariance_insights():
    """Selected insights selected again: a covariance insight (no shift tests) is judged on the correlation change."""
    cfg = MinerConfig()
    cov = replace(make_insight("cov", z=0.1, emm=0.4, p=0.5), shifts=(Shift("m", 0.1, 0, 0, 1), Shift("n", 0.3, 0, 0, 1)))
    kept = select_insights([cov, make_insight("ok", (("a", "2"),))], cfg).kept
    assert {i.phenomenon_type for i in kept} == {"covariance", "shift"}
    again = select_insights(kept, cfg)
    assert [i.id for i in again.kept] == [i.id for i in kept] and again.rejections == []
    faded = select_insights([replace(i, emm_score=0.0) for i in kept if i.phenomenon_type == "covariance"], cfg)
    assert faded.kept == [] and [(r.reason, r.detail) for r in faded.rejections] == [("weak_effect", "emm=0.00")]


def test_an_insight_record_round_trips():
    """``to_record`` is the journal form: through JSON and ``from_record`` it is the same insight, shift or covariance."""
    kept = select_insights([make_insight("ok"), make_insight("cov", (("a", "2"),), z=0.1, emm=0.4, p=0.5)], MinerConfig()).kept
    assert {i.phenomenon_type for i in kept} == {"shift", "covariance"}
    for ins in kept:
        assert Insight.from_record(json.loads(json.dumps(ins.to_record()))) == ins
