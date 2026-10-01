"""Statistical discovery -> canonical insight (SDD 03, SDD 04)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ltir.discovery import DiscoveryError, build_insights, covers_of, run_discovery
from ltir.models import Condition, pattern_id
from ltir.quality import select_insights


@pytest.fixture(scope="module")
def discovered(demo_df):
    from ltir.config import load_config

    cfg = load_config()
    res = run_discovery(demo_df, cfg)
    insights = build_insights(res, cfg, dataset_id="ds-test", batch_id="B-test", filename="retail.csv")
    return cfg, res, insights


def by_scope(insights, **scope):
    want = tuple(sorted(Condition(k, v) for k, v in scope.items()))
    return next(i for i in insights if i.conditions == want)


def test_profile_and_candidate_contract(discovered):
    cfg, res, insights = discovered
    assert res.profile.rows == 5000
    assert "order_id" not in res.profile.numerics  # integer identifier dropped by EDA step1
    assert set(res.profile.numerics) == {"discount", "margin", "delivery_days", "return_rate"}  # float targets kept
    assert res.profile.selected_dimensions == ["region", "category", "channel"]
    # rare levels enter the space (step 3 keeps the level crossing 95 % mass) but fail the size screen
    assert 0 < len(res.candidates) <= res.profile.search_space_size
    assert 0 < len(res.validated) <= cfg.validation_budget
    assert len(insights) == len(res.validated)
    assert res.n_tests == len(res.candidates) * len(res.profile.numerics)


def test_insight_fields_and_provenance(discovered):
    _, res, insights = discovered
    covers = covers_of(res)
    for ins in insights:
        assert ins.id == pattern_id("ds-test", ins.conditions)
        assert ins.dataset_id == "ds-test" and ins.batch_id == "B-test"
        assert ins.provenance["engine"] == "ltir/engines/eda/main_upd.py"
        assert ins.provenance["expression"] == ins.expression
        assert ins.support == len(covers[ins.expression])
        assert ins.target == ins.shifts[0].metric  # primary = largest |shift|
        assert abs(ins.effect_size) == pytest.approx(ins.shifts[0].magnitude)
        assert 0.1 - 1e-9 <= ins.stability <= 1.0
        assert ins.p_adjusted >= ins.p_value
        assert ins.row_hash and ins.covariance.get("pair")


def test_signed_shifts_recover_planted_directions(discovered):
    _, _, insights = discovered
    assert by_scope(insights, region="EU", category="laptops").effect_size > 0
    assert by_scope(insights, region="EU", category="laptops", channel="retail").shift_for("margin").robust_z < 0
    us_phones = by_scope(insights, region="US", category="phones")
    assert us_phones.target == "discount" and us_phones.effect_size > 0
    assert us_phones.shift_for("margin").robust_z < 0


def test_discovery_is_deterministic(demo_df, discovered):
    cfg, _, insights = discovered
    again = build_insights(run_discovery(demo_df, cfg), cfg, dataset_id="ds-test", batch_id="B-test", filename="retail.csv")
    assert [(i.id, round(i.sd_score, 10)) for i in again] == [(i.id, round(i.sd_score, 10)) for i in insights]


def test_selection_keeps_planted_phenomena(discovered):
    cfg, res, insights = discovered
    kept = {i.scope_expr: i for i in select_insights(insights, covers_of(res), cfg).kept}
    # local anomaly + stronger specialization
    parent = kept["category=laptops AND region=EU"]
    child = kept["category=laptops AND channel=online AND region=EU"]
    assert child.shift_for("margin").robust_z > parent.shift_for("margin").robust_z > 0
    # contrasting subgroup
    assert kept["category=laptops AND channel=retail AND region=EU"].shift_for("margin").robust_z < 0
    # recurring phenomenon in structurally disjoint scopes
    for scope in ("category=phones AND region=US", "category=tablets AND region=APAC"):
        s = kept[scope]
        assert s.shift_for("discount").robust_z > 1 and s.shift_for("margin").robust_z < -0.5
    # EMM-only correlation break becomes a covariance insight
    cov = kept["category=phones AND region=EU"]
    assert cov.phenomenon_type == "covariance" and set(cov.covariance["pair"]) == {"discount", "margin"}


def test_failure_modes():
    from ltir.config import load_config

    cfg = load_config()
    only_cats = pd.DataFrame({"a": ["x", "y"] * 50, "b": ["p", "q", "r", "s"] * 25})
    with pytest.raises(DiscoveryError) as exc:
        run_discovery(only_cats, cfg)
    assert exc.value.code == "no_numeric_targets"
    only_nums = pd.DataFrame({"a": np.arange(100) * 0.5, "b": np.sin(np.arange(100))})
    with pytest.raises(DiscoveryError) as exc:
        run_discovery(only_nums, cfg)
    assert exc.value.code == "invalid_schema"
