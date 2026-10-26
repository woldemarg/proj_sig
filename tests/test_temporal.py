"""Time windows and trajectories (docs/02_discovery.md §2.10): the windows of an upload, known rules in
discovery, the fold of the observations and the transitions between windows."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_discovery_contract import implied_warehouse_frame

import subgroup_miner.ingestion as ingestion
from insight_contracts import UNDIRECTED_EDGES, Condition, EdgeType, GraphEdge, Insight, Shift
from insight_contracts.graph import LATENT_EDGES, STRUCTURAL_EDGES
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import DiscoveryError, KnownRule, build_insights, covers_of, run_discovery
from subgroup_miner.ingestion import IngestionError, forget_tables, load_dataset
from subgroup_miner.selection import select_insights
from subgroup_miner.temporal import (
    CLOSED,
    CONTINUATION,
    DORMANT,
    DRIFT,
    EVALUATED,
    NOT_EVALUATED,
    REAPPEARANCE,
    RECOVERY,
    TERMINATED,
    TTL_EXPIRED_UNOBSERVED,
    WINDOW_UNAVAILABLE,
    Observation,
    WindowFacts,
    fold_trajectories,
    observations,
    open_rules,
    reference_scale,
    series_id,
    stamp_insights,
    trajectory_key,
    transition,
    upload_windows,
)
from subgroup_miner.vendor.eda.main_upd import calculate_mad

CFG = MinerConfig()
SCOPE = (Condition("region", "x"), Condition("segment", "B"))


def planted_frame(seed: int = 1, n: int = 1500) -> pd.DataFrame:
    """B ∧ x shifts value, score and rate; noise and load never move: a rule that reports noise reports a shift the
    window's top three leave out."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "segment": rng.choice(["A", "B", "C"], n),
            "region": rng.choice(["x", "y", "z"], n),
            "channel": rng.choice(["p", "q"], n),
            **{m: rng.normal(0.0, 1.0, n) for m in ("value", "score", "rate", "load", "noise")},
        }
    )
    cohort = (frame.segment == "B") & (frame.region == "x")
    frame.loc[cohort, "value"] += 3.0
    frame.loc[cohort, "score"] += 2.0
    frame.loc[cohort, "rate"] += 1.5
    return frame


RULE = KnownRule("T-b-x", SCOPE, ("value", "noise"))


def test_a_known_rule_is_mined_outside_the_eda_pool_with_its_own_test_family():
    res = run_discovery(planted_frame(), CFG, known=[RULE])
    (cohort,) = res.tracked
    assert cohort.known == (RULE,) and cohort.expression == "region=='x' AND segment=='B'"
    extent = np.sort(cohort.row_indices).tobytes()
    assert all(np.sort(c.row_indices).tobytes() != extent for c in res.candidates)  # the EDA leaves its extent to it
    assert res.n_tests == len(res.candidates) * len(res.profile.numerics)  # the EDA's family only
    assert covers_of(res)[cohort.expression] is cohort.row_indices
    ins = next(i for i in build_insights(res, CFG, dataset_id="d", batch_id="b", filename="f") if i.expression == cohort.expression)
    m = len(res.profile.numerics)
    assert ins.provenance["multiple_testing_family"] == m and ins.p_adjusted == pytest.approx(min(1.0, ins.p_value * m))
    assert ins.target == "value" == ins.shifts[0].metric  # the primary is still the largest shift
    assert ins.shift_for("noise").source == "tracked"  # a metric of the trajectory the top three leave out


def test_known_rules_a_window_cannot_evaluate():
    absent = KnownRule("T-a", (Condition("missing", "v"), Condition("segment", "B")), ("value",))
    empty = KnownRule("T-e", (Condition("region", "nowhere"), Condition("segment", "B")), ("value",))
    res = run_discovery(planted_frame(), CFG, known=[absent, empty])
    assert res.tracked == [] and res.tracked_absent == {"T-a": ("EVALUATION_ERROR", None), "T-e": ("INSUFFICIENT_SAMPLE", 0)}


def test_a_known_rules_bootstrap_does_not_depend_on_the_eda_pool():
    wide = run_discovery(planted_frame(), CFG, known=[RULE])
    narrow = run_discovery(planted_frame(), replace(CFG, validation_budget=3), known=[RULE])
    assert len(narrow.validated) < len(wide.validated)
    assert narrow.tracked[0].final_sd_score == wide.tracked[0].final_sd_score


def test_a_window_whose_cohorts_are_all_known_is_still_mined():
    nothing = replace(CFG, compute_budget=0)  # no EDA search space at all
    with pytest.raises(DiscoveryError):
        run_discovery(planted_frame(), nothing)
    res = run_discovery(planted_frame(), nothing, known=[RULE])
    assert res.candidates == [] and [c.known for c in res.tracked] == [(RULE,)]


def test_the_shortest_selector_of_an_extent_opens_its_trajectory():
    frame = implied_warehouse_frame()
    res = run_discovery(frame, CFG)
    eu_laptops = (Condition("category", "laptops"), Condition("region", "EU"))
    cohort = next(c for c in res.candidates if set(eu_laptops) < set(c.conditions))
    assert Condition("warehouse", "RTM") in cohort.conditions and cohort.generator == eu_laptops
    rules = [KnownRule("T-1", eu_laptops, ("margin",)), KnownRule("T-2", (*eu_laptops, Condition("warehouse", "RTM")), ("margin",))]
    (both,) = run_discovery(frame, CFG, known=rules).tracked  # two rules, one extent: one cohort
    assert both.known == tuple(rules)


def test_insights_carry_their_trajectory_and_windows_record_what_they_saw():
    first = run_discovery(planted_frame(1), CFG)
    insights = build_insights(first, CFG, dataset_id="d0", batch_id="b0", filename="f")
    stamp_insights(insights, first, series="up-1", window=0)
    ins = next(i for i in insights if set(SCOPE) <= set(i.conditions) and len(i.conditions) == 2)
    temporal = ins.provenance["temporal"]
    assert temporal["origin"] == "eda" and temporal["trajectory"] == trajectory_key("up-1", SCOPE)
    assert temporal["metrics"] == [s.metric for s in ins.shifts] and temporal["window"] == 0
    rule = KnownRule(temporal["trajectory"], SCOPE, tuple(temporal["metrics"]))
    gone = KnownRule("T-gone", (Condition("region", "nowhere"), Condition("segment", "B")), ("value",))
    second = run_discovery(planted_frame(2), CFG, known=[rule, gone])
    later = build_insights(second, CFG, dataset_id="d1", batch_id="b1", filename="f")
    stamp_insights(later, second, series="up-1", window=1)
    tracked = next(i for i in later if i.expression == "region=='x' AND segment=='B'")
    assert tracked.provenance["temporal"] | {"window": 0, "origin": "eda"} == temporal  # the same trajectory, rule and metrics
    selection = select_insights(later, CFG)
    seen = observations([rule, gone], second, later, selection.kept, selection.rejections)
    assert seen[rule.key].active and seen[rule.key].pattern_id == tracked.id
    assert seen["T-gone"] == Observation(NOT_EVALUATED, reason="INSUFFICIENT_SAMPLE", support=0)
    refused = observations([rule], second, later, [], [])  # admitted nowhere: evaluated, inactive
    assert refused[rule.key].status == EVALUATED and not refused[rule.key].active and refused[rule.key].pattern_id is None


def events_csv(tmp_path, rows: int = 600, days: int = 20):
    """``rows`` events over ``days`` days with a band-able metric, one row exactly on a window edge and one without a time."""
    rng = np.random.default_rng(5)
    stamps = pd.Timestamp("2031-03-03 06:00", tz="UTC") + pd.to_timedelta(rng.uniform(0, days * 86_400, rows), unit="s")
    frame = pd.DataFrame(
        {
            "when": stamps.strftime("%Y-%m-%d %H:%M:%S"),
            "segment": rng.choice(["A", "B"], rows),
            "value": rng.normal(0.0, 1.0, rows),
            "flat": 1.0,
            "mark": np.arange(rows, dtype=float),
        }
    )
    frame.loc[0, "when"], frame.loc[1, "when"] = "2031-03-10 00:00:00", "not a time"  # the edge W0 | W1; no window
    path = tmp_path / "events.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    return path, frame


def test_an_upload_is_cut_into_half_open_windows(tmp_path):
    forget_tables()
    path, frame = events_csv(tmp_path)
    windows, scale = upload_windows(path, CFG, time="when", window="7D", max_windows=52)
    assert [(w["index"], w["of"], w["t_start"][:10]) for w in windows] == [(0, 3, "2031-03-03"), (1, 3, "2031-03-10"), (2, 3, "2031-03-17")]
    assert sum(w["rows"] for w in windows) == len(frame) - 1  # the row without a time is in none
    assert scale["flat"] is None and scale["value"] == pytest.approx(calculate_mad(frame["value"]) / 0.6745)
    loads = [load_dataset(path, CFG, window=w) for w in windows]
    assert [len(d.frame) for d in loads] == [w["rows"] for w in windows]
    assert "when" not in loads[0].frame and 0.0 in set(loads[1].frame["mark"]) and 0.0 not in set(loads[0].frame["mark"])
    assert len({d.dataset_id for d in loads} | {load_dataset(path, CFG).dataset_id}) == 4
    assert any("1 rows without a readable time" in w for w in loads[0].warnings)
    assert series_id("up-1", windows[0]) == series_id("up-1", upload_windows(path, CFG, time="when", window="1W", max_windows=52)[0][0])
    other = upload_windows(path, CFG, time="when", window="14D", max_windows=52)[0]
    assert series_id("up-1", other[0]) != series_id("up-1", windows[0])  # another cut of the file: other trajectories
    assert load_dataset(path, CFG, window=other[0]).dataset_id != loads[0].dataset_id  # a window is its bounds, not its index


def test_window_options_are_refused(tmp_path):
    path, _ = events_csv(tmp_path)
    for spec, column, code in [
        ("1M", "when", "invalid_options"),
        ("3", "when", "invalid_options"),
        ("7D", "nope", "invalid_options"),
        ("7D", "value", "invalid_options"),
        ("1h", "when", "invalid_options"),
    ]:
        with pytest.raises(IngestionError) as exc:
            upload_windows(path, CFG, time=column, window=spec, max_windows=52)
        assert exc.value.code == code, (spec, column)
    small = upload_windows(path, CFG, time="when", window="1D", max_windows=52)[0]
    with pytest.raises(IngestionError) as exc:
        load_dataset(path, CFG, window=small[0])
    assert exc.value.code == "invalid_schema"  # fewer than MIN_ROWS rows in the window


def test_an_uploads_windows_share_one_prepared_table(tmp_path, monkeypatch):
    forget_tables()
    path, frame = events_csv(tmp_path)
    calls = []
    monkeypatch.setattr(ingestion, "apply_bins", lambda df, bins: calls.append(1) or apply_bins(df, bins))
    windows, _ = upload_windows(path, CFG, time="when", window="7D", max_windows=52, bins="value:4")
    bands = [load_dataset(path, CFG, window=w, bins="value:4").frame for w in windows]
    assert len(calls) == 1  # one band cut for the whole upload, not one per window
    whole = dict(zip(frame["mark"], "q" + (pd.qcut(frame["value"], 4).cat.codes + 1).astype(str)))  # quartiles of the whole upload
    for band in bands:
        assert band["value_band"].tolist() == [whole[m] for m in band["mark"]]  # every window reads the upload's quartiles, not its own


apply_bins = ingestion.apply_bins


def node(pid: str, key: str = "T-1", *, local: float = 2.0, base: float = 0.0, ptype: str = "shift") -> Insight:
    shift = Shift("value", 2.0, local, base, 0.6745)
    return Insight(
        id=pid,
        dataset_id="ds",
        batch_id="b",
        conditions=SCOPE,
        expression="e",
        target="value",
        shifts=(shift,),
        support=100,
        support_fraction=0.1,
        baseline=base,
        local=local,
        effect_size=2.0,
        sd_score=1.0,
        sd_raw_score=1.0,
        emm_score=0.0,
        volume_utility=0.3,
        stability=None if ptype == "covariance" else 0.9,
        p_value=None if ptype == "covariance" else 1e-4,
        p_adjusted=None if ptype == "covariance" else 1e-3,
        drivers=(),
        row_hash="h",
        phenomenon_type=ptype,
        provenance={
            "temporal": {"trajectory": key, "predicate": [{"attribute": c.attribute, "value": c.value} for c in SCOPE], "metrics": ["value"]}
        },
    )


REFUSED = Observation(EVALUATED, rejection="weak_effect")


def facts(*nodes: Insight, refused: str | None = None) -> WindowFacts:
    return WindowFacts({refused: REFUSED} if refused else {}, list(nodes))


def fold(windows: dict[int, WindowFacts], upto: int, scale=None):
    return fold_trajectories(windows, upto, scale or {"value": 1.0}, CFG)


def test_three_refusals_terminate_a_trajectory_and_unobserved_windows_do_not_reset_them():
    t = fold({0: facts(node("P0")), 1: facts(refused="T-1"), 3: facts(refused="T-1"), 4: facts(refused="T-1")}, 5)["T-1"]  # 2: unavailable
    assert (t.state, t.closed_reason) == (CLOSED, TERMINATED)
    assert [(w["index"], w["status"], w["reason"]) for w in t.windows][1:3] == [(1, EVALUATED, None), (2, NOT_EVALUATED, WINDOW_UNAVAILABLE)]
    assert open_rules({"T-1": t}) == []


def test_four_unobserved_windows_close_a_trajectory_for_want_of_data():
    t = fold({0: facts(node("P0"))}, 4)["T-1"]  # windows 1-3 missing: unavailable
    assert t.state != CLOSED
    t = fold({0: facts(node("P0"))}, 5)["T-1"]
    assert (t.state, t.closed_reason) == (CLOSED, TTL_EXPIRED_UNOBSERVED)


def test_a_dormant_trajectory_recovers_and_a_closed_one_reappears_as_a_new_episode():
    windows = {0: facts(node("P0")), 1: facts(refused="T-1"), 2: facts(node("P2")), 3: facts(refused="T-1")}
    windows |= {4: facts(refused="T-1"), 5: facts(refused="T-1"), 6: facts(), 7: facts(node("P7"))}
    t = fold(windows, 8)["T-1"]
    assert [(a, b, p["transition"], p["gap"], p["episode"]) for a, b, p in t.edges] == [
        ("P0", "P2", RECOVERY, 1, 1),
        ("P2", "P7", REAPPEARANCE, 4, 2),
    ]
    assert t.state == "ACTIVE" and t.episode == 2
    assert fold(windows, 6)["T-1"].state == CLOSED  # terminated before it reappeared
    assert fold(windows, 2)["T-1"].state == DORMANT


def test_transitions_measure_the_excess_over_the_population():
    a, b = node("Pa", local=2.0, base=0.0), node("Pb", local=4.0, base=1.0)
    kind, shifts, metric = transition(a, b, {"value": 2.0}, ["value"], CFG)
    assert shifts == {"value": {"subgroup": 1.0, "baseline": 0.5, "excess": 0.5}} and (kind, metric) == (DRIFT, "value")  # at the threshold
    assert transition(a, node("Pc", local=4.0, base=2.0), {"value": 2.0}, ["value"], CFG)[0] == CONTINUATION  # the population moved as much
    assert transition(a, b, {"value": None}, ["value"], CFG) == (CONTINUATION, {}, None)  # no scale, no numbers
    cov = node("Pd", local=9.0, base=0.0, ptype="covariance")
    assert transition(a, cov, {"value": 1.0}, ["value"], CFG)[0] == CONTINUATION  # a covariance node's medians are not its phenomenon
    t = fold({0: facts(node("P0")), 1: facts(cov)}, 2)["T-1"]
    assert t.windows[1]["p_adjusted"] is None and t.edges[0][2]["transition"] == CONTINUATION


def test_evolves_to_is_a_temporal_edge_no_walk_follows():
    edge = GraphEdge("P-a", "P-b", EdgeType.EVOLVES_TO, 1.0, {}).to_dict()
    assert edge["plane"] == "temporal" and edge["id"] == "EVOLVES_TO:P-a->P-b"
    assert EdgeType.EVOLVES_TO not in STRUCTURAL_EDGES | LATENT_EDGES | UNDIRECTED_EDGES


def test_reference_scale_reads_the_whole_upload():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "flat": 3.0, "text": list("vwxyz")})
    assert reference_scale(frame) == {"a": pytest.approx(1.0 / 0.6745), "flat": None}
