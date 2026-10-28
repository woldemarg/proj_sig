"""The calendar above the time hubs and where an insight lives in time (docs/02_discovery.md §2.10,
docs/06_graph_and_storage.md §6.1): a pattern's core, the period above a bin, the patterns of a period, and what the
console is sent of them."""

from __future__ import annotations

from test_temporal import entry, phones_us, utc

from graph_query_engine.graph import DualGraph
from graph_query_engine.temporal import period_interval
from insight_contracts import SNAPSHOT_VERSION
from insight_graph_service.server.views import cytoscape_elements, node_details
from subgroup_miner.config import MinerConfig
from subgroup_miner.temporal import calendar_bins, core_span, dynamics, parent_period, period_bounds

CFG = MinerConfig()


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
    def traj(core, events, tz=None):
        return {"tz": tz, "core": core and {"t_start": core[0], "t_end": core[1]}, "events": [{"kind": k, "period": p} for k, p in events]}

    trajectories = {
        "P1": traj(("2031-03-03T00:00:00+00:00", "2031-03-24T00:00:00+00:00"), [("PEAKED_AT", "2031-W14")]),  # W10–W12, a peak in W14
        "P2": traj(("2031-03-24T00:00:00+02:00", "2031-04-07T00:00:00+03:00"), [("DROPPED_AT", "2031-03-26")], "Europe/Kyiv"),  # W13–W14, Kyiv's
        "P3": traj(None, []),
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
