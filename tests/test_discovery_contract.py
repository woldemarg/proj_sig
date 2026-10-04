"""Statistical discovery -> canonical insight (docs/02_discovery.md, docs/03_insights.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insight_contracts import Condition, pattern_id
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import (
    Candidate,
    DiscoveryError,
    build_insights,
    closed_intent,
    covers_of,
    prune_near_duplicates,
    run_discovery,
)
from subgroup_miner.selection import select_insights


@pytest.fixture(scope="module")
def discovered(demo_df):
    cfg = MinerConfig()
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
        assert ins.provenance["engine"] == "subgroup_miner/vendor/eda/main_upd.py"
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
    kept = {" AND ".join(c.expr for c in i.conditions): i for i in select_insights(insights, cfg).kept}
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


def test_closed_intent_adds_implied_conditions():
    data = pd.DataFrame({"store": ["S1", "S1", "S2", "S2"], "type": ["A", "A", "A", "B"], "note": ["nan"] * 4})
    rows = np.array([0, 1])
    assert closed_intent(data, rows, ["store", "type", "note"]) == (Condition("store", "S1"), Condition("type", "A"))
    assert closed_intent(data, np.array([0, 2]), ["store", "type"]) == (Condition("type", "A"),)  # null-like levels never qualify


def implied_warehouse_frame(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    """EU ∧ laptops always ships from RTM, so `… AND warehouse == RTM` selects the same rows as EU ∧ laptops."""
    rng = np.random.RandomState(seed)
    region = rng.choice(["EU", "US", "APAC"], n)
    category = rng.choice(["laptops", "phones", "tablets"], n)
    warehouse = rng.choice(["RTM", "HAM", "PAR"], n)
    planted = (region == "EU") & (category == "laptops")
    warehouse[planted] = "RTM"
    margin = rng.normal(20.0, 3.0, n) + 8.0 * planted
    return pd.DataFrame({"region": region, "category": category, "warehouse": warehouse, "margin": margin, "discount": rng.normal(10.0, 2.0, n)})


def test_identical_extents_merge_before_validation():
    cfg = MinerConfig()
    res = run_discovery(implied_warehouse_frame(), cfg)
    merged = [r for r in res.rejections if r.reason == "cover_equivalent"]
    assert res.pass1_subgroups - len(res.candidates) == len(merged) >= 1
    cohort = next(c for c in res.candidates if {Condition("region", "EU"), Condition("category", "laptops")} <= set(c.conditions))
    assert Condition("warehouse", "RTM") in cohort.conditions  # implied condition is part of the scope
    assert cohort.aliases and all(r.expression in cohort.aliases for r in merged if "laptops" in r.expression and "EU" in r.expression)
    extents = [np.sort(c.row_indices).tobytes() for c in res.validated]
    assert len(extents) == len(set(extents))  # the bootstrap never sees the same cohort twice
    ins = next(i for i in build_insights(res, cfg, dataset_id="d", batch_id="b", filename="f.csv") if i.expression == cohort.expression)
    assert ins.id == pattern_id("d", cohort.conditions) and not any(d.startswith("[warehouse]") for d in ins.drivers)


def test_near_duplicates_are_pruned_in_rank_order():
    data = pd.DataFrame({"margin": np.r_[np.full(100, 30.0), np.full(100, 10.0)], "discount": np.r_[np.full(100, 5.0), np.full(100, 1.0)]})
    medians = {"margin": 20.0, "discount": 3.0}

    def cand(expr, rows, metric):
        return Candidate(expr, (), np.asarray(rows), len(rows), 0.2, [(metric, 2.0)], 2.0, 0.0)

    a, b = cand("a", range(100), "margin"), cand("b", range(95), "margin")  # Jaccard 0.95, same metric and sign
    c = cand("c", range(97), "discount")  # overlaps as much, but another primary metric
    d = cand("d", range(100, 200), "margin")  # disjoint rows
    kept, rejections = prune_near_duplicates([a, b, c, d], data, medians, 0.88)
    assert [k.expression for k in kept] == ["a", "c", "d"] and a.aliases == ["b"]
    assert [(r.expression, r.reason) for r in rejections] == [("b", "near_duplicate")]


def test_closed_intents_follow_extent_order(discovered):
    _, res, _ = discovered
    covers = covers_of(res)
    for a in res.validated:
        for b in res.validated:
            if set(covers[a.expression]) < set(covers[b.expression]):
                assert set(b.conditions) < set(a.conditions)  # Galois antitone: smaller extent, larger intent


def test_failure_modes():
    cfg = MinerConfig()
    only_cats = pd.DataFrame({"a": ["x", "y"] * 50, "b": ["p", "q", "r", "s"] * 25})
    with pytest.raises(DiscoveryError) as exc:
        run_discovery(only_cats, cfg)
    assert exc.value.code == "no_numeric_targets"
    only_nums = pd.DataFrame({"a": np.arange(100) * 0.5, "b": np.sin(np.arange(100))})
    with pytest.raises(DiscoveryError) as exc:
        run_discovery(only_nums, cfg)
    assert exc.value.code == "invalid_schema"
