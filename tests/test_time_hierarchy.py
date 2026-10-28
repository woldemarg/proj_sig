"""The calendar above the time hubs and where an insight lives in time (docs/02_discovery.md §2.10,
docs/06_graph_and_storage.md §6.1): a pattern's core, the period above a bin, the patterns of a period, and what the
console is sent of them; and the answer's reference date: the findings' freshness and age, a period's hub
(docs/07_question_answering.md §7.2–7.4)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from conftest import make_config, make_engine, weekly_retail
from fastapi.testclient import TestClient
from test_temporal import entry, phones_us, restarted, utc

from graph_query_engine import seeds
from graph_query_engine.config import QueryConfig
from graph_query_engine.graph import DualGraph
from graph_query_engine.seeds import SEED_RECENCY_WEIGHT, SEED_TIME_WEIGHT, SeedMatch
from graph_query_engine.temporal import core_age, parse_as_of, period_interval, recency, time_facts, time_line
from graph_query_engine.traversal import RECENCY_RANK_WEIGHT, traverse
from insight_contracts import SNAPSHOT_VERSION
from insight_graph_service.core.snapshot import metadata_digest
from insight_graph_service.server.app import create_app
from insight_graph_service.server.views import cytoscape_elements, node_details
from subgroup_miner.config import MinerConfig
from subgroup_miner.temporal import calendar_bins, core_span, dynamics, parent_period, period_bounds

CFG = MinerConfig()
QUESTION = "Why is the discount higher for phones in the US?"


def test_the_core_is_the_shortest_run_of_bins_holding_most_of_the_rows_the_earliest_on_a_tie():
    assert core_span([100] * 6) == (0, 4) and core_span([0, 0, 100, 0, 0]) == (2, 2) and core_span([5, 95]) == (1, 1)
    assert core_span([10, 30, 30, 10, 10, 10]) == (0, 3)  # two runs of four hold 80: the earliest
    assert core_span([0, 0, 0]) is None and core_span([]) is None
    bins = [{"period": f"2031-W{w}", "t_start": f"s{w}", "t_end": f"e{w}"} for w in (10, 11, 12)]
    assert dynamics(entry([1.0] * 3), bins, ["m"], CFG)["core"] is None  # an entry without its rows
    core = dynamics(entry([1.0] * 3) | {"rows": [10, 80, 10]}, bins, ["m"], CFG)["core"]
    assert core == {"start": "2031-W11", "end": "2031-W11", "t_start": "s11", "t_end": "e11", "share": 0.8, "bins": 1, "first": 1, "last": 1}


def test_a_parent_period_is_the_week_of_a_day_the_thursdays_month_of_a_week_and_the_quarter_of_a_month():
    assert parent_period("2031-03-31", "day") == ("week", "2031-W14")  # a Monday whose week runs into April: its month is April's, through its week
    assert parent_period("2031-W14", "week") == ("month", "2031-04") and parent_period("2031-W13", "week") == ("month", "2031-03")
    assert parent_period("2026-W53", "week") == ("month", "2026-12") and parent_period("2027-W01", "week") == ("month", "2027-01")
    assert parent_period("2031-04", "month") == ("quarter", "2031-Q2") and parent_period("2031-Q2", "quarter") is None
    assert period_bounds("2031-W14", "week") == ("2031-03-31T00:00:00+00:00", "2031-04-07T00:00:00+00:00")
    assert period_bounds("2031-Q2", "quarter") == ("2031-04-01T00:00:00+00:00", "2031-07-01T00:00:00+00:00")
    assert period_bounds("2031-03", "month", "Europe/Kyiv") == ("2031-03-01T00:00:00+02:00", "2031-04-01T00:00:00+03:00")  # across the clock change
    assert period_bounds("2031-03-24", "day") == tuple(calendar_bins(utc("2031-03-24 10:00"), "day")[1][0][k] for k in ("t_start", "t_end"))


def test_a_period_names_the_patterns_active_in_it_or_with_an_event_inside_it_on_their_own_calendar():
    def traj(periods, grid, core, events, tz=None):  # a trajectory as the compile stores it: its bins' bounds, its core and events by bin
        bins = [list(period_bounds(p, grid, tz)) for p in periods]
        return {
            "tz": tz,
            "bins": bins,
            "core": core and {"t_start": bins[core[0]][0], "t_end": bins[core[1]][1]},
            "events": [{"kind": k, "bin": b, "period": periods[b]} for k, b in events],
        }

    weeks, days = [f"2031-W{w}" for w in range(10, 15)], [f"2031-03-{d}" if d < 32 else f"2031-04-0{d - 31}" for d in range(24, 38)]
    trajectories = {
        "P1": traj(weeks, "week", (0, 2), [("PEAKED_AT", 4)]),  # W10–W12, a peak in W14
        "P2": traj(days, "day", (0, 13), [("DROPPED_AT", 2)], "Europe/Kyiv"),  # 24 March – 6 April on Kyiv's days, a dip on the 26th
        "P3": traj(weeks, "week", None, []),
    }
    graph = DualGraph({"version": SNAPSHOT_VERSION, "nodes": [], "edges": [], "trajectories": trajectories})
    assert graph.members(period_interval("2031-W14")) == {"active": ["P2"], "peaked": [("P1", "2031-W14")], "dropped": []}
    assert graph.members(period_interval("2031-W11"))["active"] == ["P1"]
    assert graph.members(period_interval("2031-W12"))["active"] == ["P1"]  # P2's week starts at Kyiv's midnight, two hours before UTC's
    assert graph.members(period_interval("2031-03-26")) == {"active": ["P2"], "peaked": [], "dropped": [("P2", "2031-03-26")]}
    month = graph.members(period_interval("2031-03"))  # W14 runs into April: its peak is no event of March
    assert month == {"active": ["P1", "P2"], "peaked": [], "dropped": [("P2", "2031-03-26")]}
    assert graph.members(period_interval("2031-Q3")) == {"active": [], "peaked": [], "dropped": []} and graph.as_of is None


def test_the_graph_payload_carries_the_periods_levels_the_cores_and_the_reference_date(timed):
    engine, record = timed
    graph = engine.graph()
    view = cytoscape_elements(graph, record["dataset_id"], False)
    hubs = {el["data"]["id"]: el["data"] for el in view["nodes"] if el["data"]["kind"] == "TemporalNode"}
    assert {"T:week:2031-W13", "T:month:2031-03", "T:quarter:2031-Q1"} <= hubs.keys()  # the periods above the dataset's survive its filter
    assert hubs["T:month:2031-03"] == hubs["T:month:2031-03"] | {"granularity": "month", "period": "2031-03", "t_end": "2031-04-01T00:00:00+00:00"}
    cores = [el["data"]["core"] for el in view["nodes"] if el["data"]["kind"] == "Pattern"]
    assert cores and all(c["t_start"] < c["t_end"] and c["bins"] >= 1 for c in cores)
    assert view["stats"]["as_of"] == record["temporal"]["span"][1]
    week = node_details(graph, "T:week:2031-W13", CFG.thresholds)["members"]
    pid = phones_us(engine.ws.load_graph())
    assert pid in {m["id"] for m in week["peaked"]} and all(m["period"] == "2031-W13" for m in week["peaked"])
    peaked = {m["id"] for m in week["peaked"]}
    assert peaked <= {m["id"] for m in week["active"]} and "period" not in week["active"][0]  # a peak in a week of its core
    month = node_details(graph, "T:month:2031-03", CFG.thresholds)["members"]  # the month holds the week's events, named by their week
    assert peaked <= {m["id"] for m in month["peaked"]} and all(m["period"].startswith("2031-W") for m in month["peaked"])
    assert [m["weight"] for m in month["active"]] == sorted((m["weight"] for m in month["active"]), reverse=True)


def test_the_zone_is_signed_in_the_time_record_and_the_calendar_is_read_from_there(tmp_path):
    """The zone the bins are named on is part of what the record signs of its time column: an edit of the record's
    unsigned copy of it changes neither the trajectories' calendar nor the periods above the hubs, and an edit of the
    signed one is a mismatch (docs/06_graph_and_storage.md §6.1, docs/13_trust.md §13.4)."""
    engine = make_engine(make_config(tmp_path / "ws"))
    with weekly_retail(tmp_path).open("rb") as f:
        record = engine.process(engine.upload("weekly.csv", f, time="when", grid="week", tz="Europe/Kyiv")[0]["batch_id"])
    assert record["temporal"]["tz"] == record["tz"] == "Europe/Kyiv"
    engine.ws.save_batch({**record, "tz": None})  # the unsigned copy edited: the digest stands, the calendar is the signed one
    snap = restarted(engine)
    assert next(n["props"] for n in snap["nodes"] if n["kind"] == "Dataset")["integrity"] == "ok"
    assert {t["tz"] for t in snap["trajectories"].values()} == {"Europe/Kyiv"}
    assert next(n["props"]["t_start"] for n in snap["nodes"] if n["id"] == "T:month:2031-03") == "2031-03-01T00:00:00+02:00"
    assert metadata_digest({**record, "temporal": {**record["temporal"], "tz": None}}) != record["metadata_digest"]


# the answer's reference date (docs/07_question_answering.md §7.2–7.4)


def test_freshness_halves_every_half_life_from_the_core_and_a_reference_date_reads_as_iso():
    traj = {"core": {"t_start": "2031-03-03T00:00:00+00:00", "t_end": "2031-04-07T00:00:00+00:00"}}
    inside, after, before = parse_as_of("2031-03-20"), parse_as_of("2031-07-06"), parse_as_of("2030-12-03")  # 90 days past each end
    assert [core_age(traj["core"], t) for t in (inside, after, before)] == [
        {"side": "inside", "days": 0.0},
        {"side": "after", "days": 90.0},
        {"side": "before", "days": 90.0},
    ]
    assert core_age(traj["core"], parse_as_of("2031-04-07")) == {"side": "after", "days": 0.0}  # the core's end, which it excludes: ended
    assert recency(traj, inside, 90) == 1.0 and recency(traj, after, 90) == pytest.approx(0.5) == recency(traj, before, 90)  # the future as the past
    assert recency(traj, parse_as_of("2031-04-07"), 90) == 1.0  # a distance of 0 either way
    assert (
        recency(traj, None, 90) is None
        and recency(traj, inside, 0) is None
        and recency({"core": None}, inside, 90) is None
        and recency(None, inside, 90) is None
    )
    assert parse_as_of("2031-04-01T12:00+03:00").utcoffset().total_seconds() == 3 * 3600 and parse_as_of(" 2031-04-01 ").tzinfo is not None
    for bad in ("nope", "31.03.2031", "2031-13-01", "2031-02-29"):
        with pytest.raises(ValueError):
            parse_as_of(bad)


def test_a_period_alone_seeds_its_own_patterns_before_any_the_leftover_words_reached(monkeypatch):
    """A question with a period and nothing the catalog knows is about the period: what lives or moved in it seeds, even
    over a pattern its other words score above the floor; with nothing in it, the words' seed stands; with a metric or
    a condition named, the period only reorders (docs/07_question_answering.md §7.2)."""
    fixed = {
        "other": SeedMatch("other", 0.43, {"semantic": 0.43, "weight": 0.5}),  # the leftover words' best, over the floor, nothing in the week
        "event": SeedMatch("event", 0.185 + SEED_TIME_WEIGHT, {"semantic": 0.1, "weight": 0.3, "time": 1.0}),  # under it, a peak in the week
    }
    monkeypatch.setattr(seeds, "score_pattern", lambda ins, *args: fixed[ins])
    held = {"active": [], "peaked": [("event", "2031-W13")], "dropped": []}
    graph = SimpleNamespace(
        insights={k: k for k in fixed}, trajectories={}, tampered=lambda pid: False, incident=lambda pid, kinds: [], members=lambda asked: held
    )
    query = SimpleNamespace(
        conditions=[], targets=[], text="what happened in 2031-W13?", when=period_interval("2031-W13"), is_lexical=False, components=lambda: []
    )
    ask = lambda: [
        (s.pattern_id, s.matched.get("fallback"))
        for s in seeds.resolve_seeds(query, graph, SimpleNamespace(encode_query=lambda *args: None), {}, QueryConfig())
    ]  # noqa: E731
    assert ask() == [("event", "period")]
    held["peaked"] = []
    assert ask() == [("other", None)]  # nothing lives or moved in the week: the words' seed stands
    query.is_lexical, held["peaked"] = True, [("event", "2031-W13")]
    assert ask() == [("other", None)]  # a metric or a condition named: the period reorders, never lifts one over the floor


def test_the_time_line_gives_the_core_its_age_and_an_event_outside_it():
    core = {
        "start": "2031-W10",
        "end": "2031-W12",
        "t_start": "2031-03-03T00:00:00+00:00",
        "t_end": "2031-03-24T00:00:00+00:00",
        "share": 0.83,
        "bins": 3,
        "first": 0,
        "last": 2,
    }
    peak = {"kind": "PEAKED_AT", "period": "2031-W14", "metric": "m", "z_smoothed": 2.0, "prominence": 1.0, "bin": 4}
    traj = {"grid": "week", "span": ["2031-03-03", "2031-04-13"], "metric": "m", "trend": "spike", "current": 0.1, "bins_measured": 6, "of": 6}
    facts = time_facts(traj | {"gaps": [], "core": core, "events": [peak]}, 0.5)
    assert facts["core"] == core and facts["events"][0]["outside"] is True  # the one core, as the trajectory holds it
    assert "| core 2031-W10–2031-W12 (3 bins, 83% of its timed rows) | events: peak 2031-W14 (m, +2.00 sd, outside its core) |" in time_line(facts)
    for asked, words in (
        ("2031-04-13", "ended 20 days before"),
        ("2031-03-10", "the reference date inside it"),
        ("2031-02-20", "starts 11 days after"),
        ("2031-03-24", "ended less than a day before"),  # the core's last instant is not in it, and 0 days is no day
        ("2031-03-24T06:00", "ended less than a day before"),
        ("2031-03-25T06:00", "ended 1 day before"),
        ("2031-02-21T06:00", "starts 10 days after"),  # 9.75 days: the nearest day, never a day for every started one
    ):
        assert f"timed rows), {words}" in time_line(facts | {"age": core_age(core, parse_as_of(asked))})
    alone = time_facts(traj | {"gaps": [], "core": None, "events": [peak]}, 0.5)
    assert "| core: none (no timed rows) |" in time_line(alone) and "outside" not in alone["events"][0]


def test_the_reference_date_is_the_latest_data_and_a_question_may_set_its_own(timed):
    engine, record = timed
    assert engine.graph().as_of == datetime.fromisoformat(record["temporal"]["span"][1])
    latest = engine.search(QUESTION).evidence
    assert latest.as_of == {"date": "2031-04-13", "source": "latest data", "recency": True}
    prompt = latest.to_prompt()
    assert (
        "\nAS OF: 2031-04-13 (latest data)\nUNITS:" in prompt
        and "fresher findings (the closer their core to the AS OF date, half-life 90 days)" in prompt
    )
    later = engine.search(QUESTION, as_of=parse_as_of("2031-06-01")).evidence
    item = next(it for it in later.items if it.pattern_id == phones_us(engine.ws.load_graph()))
    assert later.as_of["source"] == "asked" and "AS OF: 2031-06-01 (asked)" in later.to_prompt()
    assert (
        item.time["age"]["side"] == "after"
        and "days before the reference date" in time_line(item.time)
        and f"     time: {time_line(item.time)}" in later.to_prompt()
    )
    timed_week = engine.search(QUESTION.rstrip("?") + " in 2031-W13?").evidence
    assert timed_week.as_of["recency"] is False and "fresher findings" not in timed_week.to_prompt()  # an asked period is its own window
    with TestClient(create_app(engine)) as client:
        refused = client.post("/api/search", json={"question": QUESTION, "as_of": "31.03.2031"})
        assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
        asked = client.post("/api/search", json={"question": QUESTION, "as_of": "2031-03-25"})
        assert asked.status_code == 200 and "AS OF: 2031-03-25 (asked)" in asked.json()["evidence_prompt"]


def test_without_a_period_fresher_findings_rank_first_and_none_is_dropped(timed, hashed_engine, monkeypatch):
    engine, _ = timed
    assert all("recency" in s.matched for s in engine.search(QUESTION).seeds)
    assert not any("recency" in s.matched for s in engine.search(QUESTION.rstrip("?") + " in 2031-W13?").seeds)
    graph, config, far = engine.graph(), engine.settings.query, parse_as_of("2032-01-01")
    start = engine.search(QUESTION).seeds
    before = {r.node_id: r.score for r in traverse(graph, start, config).patterns}
    reached = [r for r in traverse(graph, start, config, far).patterns if r.route != "seed" and r.node_id in before]
    assert reached and all(  # the walk's paths are the same; a reached finding's rank is scaled by its freshness
        r.score
        == pytest.approx(before[r.node_id] * (1 - RECENCY_RANK_WEIGHT + RECENCY_RANK_WEIGHT * recency(graph.trajectories[r.node_id], far, 90)))
        for r in reached
    )
    plain = hashed_engine.graph()  # no time column: no core, no change
    found = hashed_engine.search("Why is margin lower for phones in the US?").seeds
    assert [(r.node_id, r.score) for r in traverse(plain, found, config, far).patterns] == [
        (r.node_id, r.score) for r in traverse(plain, found, config).patterns
    ]
    fixed = {"fresh": SeedMatch("fresh", 0.6 + SEED_RECENCY_WEIGHT, {"recency": 1.0}), "stale": SeedMatch("stale", 0.6, {"recency": 0.0})}
    monkeypatch.setattr(seeds, "score_pattern", lambda ins, *args: fixed[ins])
    stub = SimpleNamespace(insights={k: k for k in fixed}, trajectories={}, tampered=lambda pid: False, incident=lambda pid, kinds: [])
    query = SimpleNamespace(conditions=[], targets=[], text="margin", when=None, components=lambda: [])
    picked = seeds.resolve_seeds(query, stub, SimpleNamespace(encode_query=lambda *args: None), {}, QueryConfig(), None, far)
    assert [s.pattern_id for s in picked] == ["fresh", "stale"]  # the floor is taken without the bonus: 0.75 · 0.6, not 0.75 · 0.95
