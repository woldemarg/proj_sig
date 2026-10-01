"""Selection rules and insight_weight (SDD 04)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ltir.config import load_config
from ltir.models import Condition, Insight, Shift
from ltir.quality import insight_weight, select_insights, weight_factors, with_weight


def make_insight(expr="a=1", conds=(("a", "1"),), z=2.0, stability=0.9, p=1e-10, vu=0.25, emm=0.05, support=300, row_hash=None, target="m"):
    conditions = tuple(sorted(Condition(k, v) for k, v in conds))
    return Insight(
        id="P-" + expr,
        dataset_id="d",
        batch_id="b",
        conditions=conditions,
        expression=expr,
        target=target,
        shifts=(Shift(target, z, 1.0, 0.0, 1.0),),
        support=support,
        support_fraction=0.1,
        baseline=0.0,
        local=1.0,
        effect_size=z,
        sd_score=abs(z) * stability,
        sd_raw_score=abs(z),
        emm_score=emm,
        volume_utility=vu,
        stability=stability,
        integrated_index=1.0,
        p_value=p,
        p_adjusted=p,
        drivers=(),
        row_hash=row_hash or expr,
        covariance={"pair": [target, "n"], "local_corr": 0.1, "global_corr": -0.5, "delta": 0.6},
    )


def test_weight_bounded_and_monotone():
    cfg = load_config()
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
    cfg = load_config()
    items = [
        make_insight("ok", (("a", "1"),)),
        make_insight("small", (("a", "2"),), support=5),
        make_insight("weak", (("a", "3"),), z=0.2, emm=0.0),
        make_insight("unsig", (("a", "4"),), p=0.5, emm=0.0),
        make_insight("unstable", (("a", "5"),), stability=0.2, emm=0.0),
    ]
    covers = {i.expression: np.arange(10) + k * 100 for k, i in enumerate(items)}
    res = select_insights(items, covers, cfg)
    reasons = {r.expression: r.reason for r in res.rejections}
    assert [i.expression for i in res.kept] == ["ok"]
    assert reasons == {"small": "min_support", "weak": "weak_effect", "unsig": "not_significant", "unstable": "unstable"}
    assert res.stats["kept"] == 1 and res.stats["min_support"] == 1


def test_cover_equivalence_keeps_closed_pattern():
    cfg = load_config()
    gen = make_insight("gen", (("a", "1"), ("b", "1")), row_hash="same")
    closed = make_insight("closed", (("a", "1"), ("b", "1"), ("c", "1")), row_hash="same")
    covers = {"gen": np.arange(50), "closed": np.arange(50)}
    res = select_insights([gen, closed], covers, cfg)
    assert [i.expression for i in res.kept] == ["closed"]
    assert res.kept[0].aliases == ("gen",)
    assert res.rejections[0].reason == "cover_equivalent"


def test_near_duplicate_collapse_same_target_and_direction():
    cfg = load_config()
    a = make_insight("a", (("a", "1"),), z=2.5)
    b = make_insight("b", (("b", "1"),), z=2.0)
    c = make_insight("c", (("c", "1"),), z=-2.0)  # opposite direction: not a duplicate
    covers = {"a": np.arange(100), "b": np.arange(97), "c": np.arange(99)}
    res = select_insights([a, b, c], covers, cfg)
    assert sorted(i.expression for i in res.kept) == ["a", "c"]
    assert next(i for i in res.kept if i.expression == "a").aliases == ("b",)


def test_emm_only_insight_becomes_covariance():
    cfg = load_config()
    ins = make_insight("cov", z=0.1, emm=0.4, p=0.5)
    ins = replace(ins, shifts=(Shift("m", 0.1, 0, 0, 1), Shift("n", 0.3, 0, 0, 1)))
    res = select_insights([ins], {"cov": np.arange(40)}, cfg)
    kept = res.kept[0]
    assert kept.phenomenon_type == "covariance" and kept.target == "n"  # pair metric with larger |shift|
    assert kept.weight_factors["confidence"] is None and kept.weight_factors["stability"] is None  # unmeasured: left out
