"""Time windows and trajectories (docs/02_discovery.md §2.10): the windows of an upload, known rules in
discovery, the fold of the observations and the transitions between windows; an upload's windows as batches of the
graph service, their records and the temporal plane of the snapshot (docs/06_graph_and_storage.md §6.2)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_config, make_engine, weekly_events
from test_discovery_contract import implied_warehouse_frame

import subgroup_miner.ingestion as ingestion
from insight_contracts import UNDIRECTED_EDGES, Condition, EdgeType, GraphEdge, Insight, Shift
from insight_contracts.graph import LATENT_EDGES, STRUCTURAL_EDGES
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.snapshot import compute_metadata_integrity
from subgroup_miner import discovery as discovery_module
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
    observed,
    open_rules,
    reference_scale,
    series_id,
    stamp_insights,
    trajectory_key,
    transition,
    upload_windows,
)
from subgroup_miner.vendor.eda.main_upd import calculate_mad

discovery_eda = discovery_module.eda

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
    renamed = run_discovery(planted_frame(), CFG, known=[replace(RULE, key="T-another-upload")])  # the same rule in another series
    assert renamed.tracked[0].final_sd_score == wide.tracked[0].final_sd_score


def test_a_window_whose_cohorts_are_all_known_is_still_mined():
    nothing = replace(CFG, compute_budget=0)  # no EDA search space at all
    with pytest.raises(DiscoveryError):
        run_discovery(planted_frame(), nothing)
    res = run_discovery(planted_frame(), nothing, known=[RULE])
    assert res.candidates == [] and [c.known for c in res.tracked] == [(RULE,)]


def test_a_window_reports_its_known_rules_when_nothing_passes_pass_one():
    """``no_candidates`` is for a window asked to follow nothing: one with known rules records why each was not evaluated
    — no search space and a rule on a missing column, or a window too thin for the search and the rule alike."""
    missing = KnownRule("T-a", (Condition("missing", "v"), Condition("segment", "B")), ("value",))
    res = run_discovery(planted_frame(), replace(CFG, compute_budget=0), known=[missing])
    assert (res.candidates, res.tracked, res.tracked_absent) == ([], [], {"T-a": ("EVALUATION_ERROR", None)})
    thin = planted_frame(1, n=80)  # every cohort under the pass-1 floor, the rule's six rows too
    with pytest.raises(DiscoveryError, match="pass-1"):
        run_discovery(thin, CFG)
    res = run_discovery(thin, CFG, known=[RULE])
    assert res.tracked_absent == {RULE.key: ("INSUFFICIENT_SAMPLE", int(((thin.segment == "B") & (thin.region == "x")).sum()))}
    seen = observations(res, [], [], [])
    assert seen[RULE.key] == Observation(NOT_EVALUATED, reason="INSUFFICIENT_SAMPLE", support=6)


def test_known_rules_cost_no_second_pass_one_without_a_rule(monkeypatch):
    calls = []
    real = discovery_eda.step4_evaluate_micro_slices
    monkeypatch.setattr(discovery_eda, "step4_evaluate_micro_slices", lambda *a: calls.append(len(a[1])) or real(*a))
    run_discovery(planted_frame(), CFG)
    assert len(calls) == 1  # the search's alone
    run_discovery(planted_frame(), CFG, known=[RULE])
    assert calls[1:] == [calls[0], 1]  # then the known rule's


def test_a_trajectory_metric_the_window_profile_drops_is_still_reported():
    """A metric constant in a window leaves its profile's metrics but stays a column: the rule reports it from the window's
    own median and MAD, and its move reads as a drift."""
    before = planted_frame(1)
    before.loc[(before.segment == "B") & (before.region == "x"), "noise"] += 2.0
    flat = planted_frame(2).assign(noise=1.0)

    def tracked(frame):
        res = run_discovery(frame, CFG, known=[RULE])
        return res, next(i for i in build_insights(res, CFG, dataset_id="d", batch_id="b", filename="f") if i.expression == res.tracked[0].expression)

    res, after = tracked(flat)
    assert "noise" not in res.profile.numerics
    noise = after.shift_for("noise")
    assert noise.source == "tracked" and noise.local_median == noise.global_median == 1.0
    kind, shifts, metric = transition(tracked(before)[1], after, {"value": 1.0, "noise": 1.0}, ["value", "noise"], CFG)
    assert (kind, metric) == (DRIFT, "noise") and shifts["noise"]["excess"] < -1.5


def test_the_shortest_selector_of_an_extent_opens_its_trajectory():
    frame = implied_warehouse_frame()
    res = run_discovery(frame, CFG)
    eu_laptops = (Condition("category", "laptops"), Condition("region", "EU"))
    cohort = next(c for c in res.candidates if set(eu_laptops) < set(c.conditions))
    assert Condition("warehouse", "RTM") in cohort.conditions and cohort.generator == eu_laptops
    rules = [KnownRule("T-1", eu_laptops, ("margin",)), KnownRule("T-2", (*eu_laptops, Condition("warehouse", "RTM")), ("margin",))]
    res = run_discovery(frame, CFG, known=rules)
    (both,) = res.tracked  # two rules, one extent: one cohort
    assert both.known == tuple(rules)
    insights = build_insights(res, CFG, dataset_id="d", batch_id="b", filename="f")
    stamp_insights(insights, res, series="up-1", window=1)
    ins = next(i for i in insights if i.expression == both.expression)
    seen = observations(res, insights, [ins], [])
    assert ins.provenance["temporal"]["trajectory"] == "T-1" and seen["T-1"] == seen["T-2"] and seen["T-2"].pattern_id == ins.id


def test_one_node_observed_by_two_trajectories_links_both():
    """Two rules whose extents meet in a window: one node, named in its provenance by the first, linked to both by the
    window's observations — the fold's record of which trajectories it serves."""
    later = node("P1", "T-1")
    seen = WindowFacts({"T-1": observed(later, True), "T-2": observed(later, True)}, [later])
    t = fold({0: facts(node("P0", "T-1"), node("Q0", "T-2")), 1: seen}, 2)
    assert [(a, b) for k in ("T-1", "T-2") for a, b, _ in t[k].edges] == [("P0", "P1"), ("Q0", "P1")]


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
    seen = observations(second, later, selection.kept, selection.rejections)
    assert seen[rule.key].active and seen[rule.key].pattern_id == tracked.id
    assert seen["T-gone"] == Observation(NOT_EVALUATED, reason="INSUFFICIENT_SAMPLE", support=0)
    refused = observations(second, later, [], [])  # admitted nowhere: evaluated, inactive
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
    assert loads[0].frame.index.equals(pd.RangeIndex(len(loads[0].frame)))  # its covers are rows of the window, not of the file
    respelled = {**windows[0], "t_start": "2031-03-03 00:00:00", "t_end": "2031-03-10T02:00:00+02:00"}  # the same instants
    again = load_dataset(path, CFG, window=respelled)
    assert again.dataset_id == loads[0].dataset_id and again.frame.equals(loads[0].frame)


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


def test_a_window_map_keyed_by_strings_folds_as_by_integers():
    windows = {0: facts(node("P0")), 1: facts(refused="T-1"), 2: facts(node("P2"))}
    assert fold({str(k): v for k, v in windows.items()}, 3)["T-1"] == fold(windows, 3)["T-1"]  # as JSON keys them


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


PHONES_US = [{"attribute": "category", "value": "phones"}, {"attribute": "region", "value": "US"}]


def upload_windows_of(engine, csv: Path) -> list[dict]:
    """``csv`` uploaded with ``time=when``, ``window=7D``, each window processed in order."""
    with csv.open("rb") as f:
        return [engine.process(r["batch_id"]) for r in engine.upload(csv.name, f, time="when", window="7D")]


def phones_us(snapshot: dict) -> tuple[str, dict]:
    return next((k, t) for k, t in snapshot["trajectories"].items() if t["predicate"] == PHONES_US)


def evolves(snapshot: dict, key: str) -> list[tuple]:
    return [
        (e["props"]["from_window"], e["props"]["to_window"], e["props"]["transition"])
        for e in snapshot["edges"]
        if e["type"] == "EVOLVES_TO" and e["props"]["trajectory"] == key
    ]


def test_an_upload_with_a_time_column_is_a_batch_per_window_and_a_trajectory_through_them(windowed):
    engine, records, served = windowed
    outcome = [(r["window"]["index"], r["status"], (r.get("error") or {}).get("code")) for r in records]
    assert outcome == [
        (0, "READY", None),
        (1, "READY", None),
        (2, "FAILED", "no_viable_insights"),
        (3, "FAILED", "invalid_schema"),
        (4, "READY", None),
    ]
    assert len({r["upload_id"] for r in records}) == 1 and len({r["dataset_id"] for r in records if r["dataset_id"]}) == 4
    assert records[2]["temporal"]["observations"] and records[3]["temporal"]["observations"] is None  # mined and refused; never mined
    assert {compute_metadata_integrity(r) for r in records} == {"ok"}  # a failed window is signed as a READY one
    snapshot = engine.ws.load_graph()
    key, t = phones_us(snapshot)
    seen = [(w["index"], w["status"], w["active"], w["reason"] or w["rejection"]) for w in t["windows"]]
    assert seen == [
        (0, "EVALUATED", True, None),
        (1, "EVALUATED", True, None),
        (2, "EVALUATED", False, "weak_effect"),  # the shuffled week: dormant
        (3, "NOT_EVALUATED", False, "WINDOW_UNAVAILABLE"),  # ten rows: nothing observed
        (4, "EVALUATED", True, None),
    ]
    assert t["state"] == "ACTIVE" and t["series"] == records[0]["upload_id"] + ":when:P7DT0H0M0S"
    assert evolves(snapshot, key) == [(0, 1, "CONTINUATION"), (1, 4, "RECOVERY")]
    after_refusal = served[2].trajectories[key]  # W2 committed nothing, yet readers saw its refusal before W3 ran
    assert after_refusal["state"] == "DORMANT" and [w["index"] for w in after_refusal["windows"]] == [0, 1, 2]
    graph = engine.graph()
    nodes = [graph.nodes[w["pattern_id"]]["props"] for w in t["windows"] if w["active"]]
    assert [n["provenance"]["temporal"]["origin"] for n in nodes] == ["eda", "tracked", "tracked"]  # found once, then followed by its rule
    assert all(n["credibility"] != 1 for n in nodes)  # the windows of one upload are one source: they confirm nothing
    assert {graph.nodes[f"DS:{r['dataset_id']}"]["props"]["window"]["index"] for r in records if r["status"] == "READY"} == {0, 1, 4}


def test_a_windows_node_reads_its_trajectory_reproduces_and_no_walk_follows_it(windowed, monkeypatch):
    from graph_query_engine.seeds import SeedMatch
    from graph_query_engine.traversal import traverse
    from insight_graph_service.server.views import node_details

    engine, records, _ = windowed
    key, t = phones_us(engine.ws.load_graph())
    tracked = t["windows"][1]["pattern_id"]
    details = node_details(engine.graph(), tracked, engine.settings.thresholds)
    assert details["trajectory"]["key"] == key and [w["index"] for w in details["trajectory"]["windows"]] == [0, 1, 2, 3, 4]
    assert details["neighbor_counts"]["EVOLVES_TO"] == 1 and details["neighbor_counts"]["EVOLVES_TO (in)"] == 1
    checked = engine.verify_pattern(tracked)  # a known rule's family is the window's metrics; its tracked shifts recomputed too
    assert checked["status"] == "reproduced", checked["diff"]
    eda = next(
        r["id"] for r in engine.ws.patterns() if r["dataset_id"] == records[1]["dataset_id"] and r["provenance"]["temporal"]["origin"] == "eda"
    )
    assert engine.verify_pattern(eda)["status"] == "reproduced"  # the window's other family
    walked = traverse(engine.graph(), [SeedMatch(tracked, 1.0, {})], engine.settings.query)
    assert walked.patterns and walked.used_edges and not any(e.startswith("EVOLVES_TO:") for e in walked.used_edges)
    journal = engine.ws.patterns()
    record = next(r for r in journal if r["id"] == tracked)
    forged = {**record, "p_adjusted": record["p_value"], "provenance": {**record["provenance"], "multiple_testing_family": 1}}
    monkeypatch.setattr(engine.ws, "patterns", lambda: [forged if r["id"] == tracked else r for r in journal])
    assert "multiple_testing_family" in {d["field"] for d in engine.verify_pattern(tracked)["diff"]}  # forged with its p: not the window's


def test_the_same_file_uploaded_again_and_still_queued_hides_no_window(windowed):
    from insight_graph_service.core.temporal import derive_temporal

    engine, records, _ = windowed
    twin = [{**r, "batch_id": f"twin-{i}", "status": "UPLOADED", "dataset_id": None} for i, r in enumerate(records)]
    alone, beside = (derive_temporal(rs, engine.ws.patterns(), engine.settings.miner).series for rs in (records, [*records, *twin]))
    assert beside == alone and alone[records[0]["upload_id"] + ":when:P7DT0H0M0S"]


def test_a_thin_window_records_why_its_known_rules_were_not_evaluated(tmp_path):
    """A window too thin for the search and for its known rules alike fails with nothing to mine, its observations on
    its record (``INSUFFICIENT_SAMPLE`` with the rows, not an unavailable window), and readers see them at once."""
    from insight_graph_service.core.demo import generate_retail_dataset

    frame = generate_retail_dataset(5000, seed=7)
    thin = frame.sample(60, random_state=3).assign(when="2031-03-12 12:00:00")  # W1: sixty rows, every cohort under pass 1
    frame = frame.sample(1500, random_state=1).assign(when="2031-03-05 12:00:00")
    path = tmp_path / "events.csv"
    pd.concat([frame, thin]).to_csv(path, index=False, lineterminator="\n")
    engine = make_engine(make_config(tmp_path / "ws"))
    first, second = upload_windows_of(engine, path)
    assert first["status"] == "READY" and (second["status"], second["error"]["code"]) == ("FAILED", "no_viable_insights")
    key, _ = phones_us(engine.ws.load_graph())
    seen = second["temporal"]["observations"][key]
    assert (seen["status"], seen["reason"]) == ("NOT_EVALUATED", "INSUFFICIENT_SAMPLE") and seen["support"] > 0
    assert [(w["index"], w["reason"]) for w in engine.graph().trajectories[key]["windows"]] == [(0, None), (1, "INSUFFICIENT_SAMPLE")]


def test_time_options_are_refused_before_the_file_is_kept(cfg, tmp_path):
    engine = make_engine(replace(cfg, max_windows=3))
    csv = weekly_events(tmp_path)
    for options, said in [
        ({"time": "when"}, "go together"),
        ({"window": "7D"}, "go together"),
        ({"time": "when", "window": "7D", "geo": "auto"}, "go together"),
        ({"time": "when", "window": "7D"}, "MAX_WINDOWS=3"),
        ({"time": "nope", "window": "7D"}, "not in dataset"),
    ]:
        with csv.open("rb") as f, pytest.raises(PipelineError, match=said) as refused:
            engine.upload(csv.name, f, **options)
        assert refused.value.code == "invalid_options"
    assert not engine.ws.list_batches() and not (engine.ws.uploads_dir.exists() and any(engine.ws.uploads_dir.iterdir()))


def test_a_window_whose_record_fails_its_digest_is_read_as_unavailable(cfg, tmp_path):
    engine = make_engine(cfg)
    records = upload_windows_of(engine, weekly_events(tmp_path))
    forged = engine.ws.load_batch(records[1]["batch_id"])
    forged["temporal"]["observations"] = {}  # by hand, past the API: the window seems to have seen none of its rules
    engine.ws.save_batch(forged)
    engine = make_engine(cfg)  # a restart rebuilds the snapshot
    snapshot = engine.ws.load_graph()
    key, t = phones_us(snapshot)
    assert [w["reason"] for w in t["windows"]][:2] == [None, "WINDOW_UNAVAILABLE"]
    assert evolves(snapshot, key) == [(0, 4, "RECOVERY")]  # the forged window links nothing; W2 refused the rule, W4 recovers it


def test_the_upload_form_takes_a_time_column_and_names_each_window(engine, tmp_path):
    from fastapi.testclient import TestClient

    from insight_graph_service.server.app import create_app

    csv = weekly_events(tmp_path)
    with TestClient(create_app(engine)) as client, csv.open("rb") as f:
        accepted = client.post("/api/upload", files={"file": ("events.csv", f, "text/csv")}, data={"time": "when", "window": "7D"})
        assert accepted.status_code == 202 and list(accepted.json()["levels"]) == ["W0", "W1", "W2", "W3", "W4"]
        refused = client.post("/api/upload", files={"file": ("events.csv", b"a,b\n1,2\n", "text/csv")}, data={"time": "when"})
        assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"


QUESTION = "Why is margin lower for phones in the US?"


def test_a_trajectory_is_one_seed_and_one_card_its_latest_window_unless_it_scores_far_below_its_best(windowed):
    """A trajectory's representative: its latest admitted window when that one scores at least the seed floor of the
    trajectory's best (max(SEED_MIN_SCORE, 75 %)), else its best; a question seeds it once and shows it once."""
    from graph_query_engine.seeds import SEED_MIN_SCORE, SEED_RELATIVE_MIN
    from graph_query_engine.temporal import representatives, trajectory

    engine, _, _ = windowed
    graph = engine.graph()
    key, t = phones_us(engine.ws.load_graph())
    w0, w1, w4 = (w["pattern_id"] for w in t["windows"] if w["active"])
    floors = (SEED_MIN_SCORE, SEED_RELATIVE_MIN)
    assert representatives(graph, [(w0, 1.0), (w1, 0.9), (w4, SEED_RELATIVE_MIN)], *floors) == {key: w4}  # the latest, at the floor
    assert representatives(graph, [(w0, 1.0), (w1, 0.9), (w4, SEED_RELATIVE_MIN - 0.01)], *floors) == {key: w0}  # under it: the best
    assert representatives(graph, [(w1, 0.3), (w4, SEED_MIN_SCORE - 0.01)], *floors) == {key: w1}  # under the absolute floor too
    found = engine.search(QUESTION)
    seeded = [trajectory(graph, s.pattern_id) for s in found.seeds]
    shown = [trajectory(graph, r.node_id) for r in found.traversal.patterns]
    assert key in seeded and len(seeded) == len(set(seeded))  # phones in the US seeds the walk, once
    assert len(shown) == len(set(shown)) and None not in shown  # every trajectory is one card


def test_a_window_card_names_its_window_and_trajectory_and_the_prompt_says_so(windowed, monkeypatch):
    """The evidence of a time window's insight: its time line (window, state, episode, active windows, last transition,
    latest and peak — a tampered window none of them), its dataset's window in DATASETS and in METRIC BASELINES, one
    NOTE, and the evidence-only answer's window and state in the console's words."""
    from graph_query_engine.temporal import time_facts
    from insight_contracts import wording

    engine, _, _ = windowed
    key, t = phones_us(engine.ws.load_graph())
    payload = engine.evidence(QUESTION)
    items = payload.view["evidence"]["items"]
    phones = next(it for it in items if it["time"]["trajectory"] == key)
    assert all(it["time"] for it in items)  # every insight here is a window's
    peak = max((w for w in t["windows"] if w["active"]), key=lambda w: abs(w["effect_size"]))
    line = f"     time: window W4 [2031-03-31, 2031-04-07) | trajectory ACTIVE (episode 1, active in 3/5 windows; last transition: RECOVERY; latest: W4, peak: W{peak['index']} {peak['effect_size']:+.2f} sd)"
    assert phones["time"]["window"]["index"] == 4 and line in payload.evidence_prompt.splitlines()
    datasets = payload.evidence_prompt.split("DATASETS:")[1].split("\n\n")[0]
    assert all(" window=W" in row for row in datasets.strip().splitlines())
    baselines = payload.evidence_prompt.split("METRIC BASELINES (whole dataset):")[1].split("\n\n")[0]
    assert all(" in window W" in row for row in baselines.strip().splitlines())
    assert sum("come from time windows" in n for n in payload.view["evidence"]["notes"]) == 1
    tag = f"({wording.name('time_window')} W4 [2031-03-31, 2031-04-07); {wording.name('trajectory')}: {wording.name('trajectory_ACTIVE')})"
    assert tag in payload.evidence_summary
    graph, (w0, w1, w4) = engine.graph(), (w["pattern_id"] for w in t["windows"] if w["active"])
    monkeypatch.setitem(graph.nodes[w4]["props"], "integrity", "mismatch")  # a window no answer shows is no admitted one
    assert {k: time_facts(graph, w1)[k] for k in ("active", "latest", "transition")} == {"active": 2, "latest": 1, "transition": "CONTINUATION"}
    for pid in (w0, w1):
        monkeypatch.setitem(graph.nodes[pid]["props"], "integrity", "mismatch")
    assert time_facts(graph, w1) == {}  # every admitted window tampered: no time line, no crash


def test_the_narrator_reads_the_windows_of_an_ended_and_a_living_trajectory(windowed, monkeypatch):
    """Rule 13 is in the system prompt with the console's words for states and transitions; an answer over an ended and
    a living trajectory's cards is grounded, its prompt holds both states and the NOTE, and those words are no jargon."""
    from conftest import FakeLLM

    from evidence_narrator_service.narration import SYSTEM_PROMPT, answer, jargon

    engine, _, _ = windowed
    graph = engine.graph()
    keys = list(dict.fromkeys(it["time"]["trajectory"] for it in engine.evidence(QUESTION).view["evidence"]["items"]))
    ended = keys[1]
    monkeypatch.setitem(graph.trajectories, ended, {**graph.trajectories[ended], "state": "CLOSED", "closed_reason": "TERMINATED"})
    payload = engine.evidence(QUESTION)
    qa = answer(payload, FakeLLM())
    assert "13. TIME:" in SYSTEM_PROMPT and qa.answer_mode == "llm" and qa.citations["grounded"]
    assert "trajectory TERMINATED" in qa.prompt and "trajectory ACTIVE" in qa.prompt and "NOTE: " in qa.prompt
    assert jargon("Траєкторія активна, інша завершена; дрейф, відновлення, повторна поява, без спостереження.", qa.prompt) == {}
