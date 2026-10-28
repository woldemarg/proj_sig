"""Time in one upload: the time column apart, the calendar grid, each insight's series, its smoothing and its events
(docs/02_discovery.md §2.10), and the temporal plane the graph service derives from them (docs/06_graph_and_storage.md §6.2)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import EPOCH, geo_frame, make_config, make_engine, upload_levels, weekly_retail
from fastapi.testclient import TestClient

from graph_query_engine import seeds
from graph_query_engine.config import QueryConfig
from graph_query_engine.question import parse_query
from graph_query_engine.seeds import SEED_TIME_WEIGHT, SeedMatch, score_pattern
from graph_query_engine.temporal import period_interval, time_facts, time_line, time_score
from insight_contracts import LATENT_EDGES, STRUCTURAL_EDGES, UNDIRECTED_EDGES, EdgeType, GraphEdge, wording
from insight_graph_service.core.demo import generate_retail_dataset
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.snapshot import compute_metadata_integrity, metadata_digest
from insight_graph_service.server.app import create_app
from insight_graph_service.server.views import cytoscape_elements, node_details
from subgroup_miner.config import MinerConfig
from subgroup_miner.ingestion import IngestionError, load_dataset, parse_tz, read_times, resolve_grid
from subgroup_miner.spatial import build_cells, point_profile
from subgroup_miner.temporal import (
    DROPPED_AT,
    FALLING,
    FLAT,
    PEAKED_AT,
    RISING,
    SPIKE,
    annotate,
    calendar_bins,
    carriers,
    detect_events,
    dynamics,
    echoes,
    extent_points,
    fit_grid,
    period_bounds,
    reference_scale,
    rts_smooth,
)
from subgroup_miner.vendor.eda.main_upd import calculate_mad

CFG = MinerConfig()


def utc(*stamps: str) -> pd.Series:
    return pd.to_datetime(pd.Series(stamps), utc=True, format="mixed")


def test_iso_weeks_cross_the_year_by_their_own_calendar():
    """2026 has a week 53; its days in January 2027 belong to it, and the next Monday opens 2027-W01."""
    codes, bins = calendar_bins(utc("2026-12-28 10:00", "2027-01-03 23:59", "2027-01-04 00:00", None), "week")
    assert [b["period"] for b in bins] == ["2026-W53", "2027-W01"]
    assert codes.tolist() == [0, 0, 1, -1] and [b["rows"] for b in bins] == [2, 1]
    assert bins[0]["t_start"] == "2026-12-28T00:00:00+00:00" and bins[0]["t_end"] == bins[1]["t_start"] == "2027-01-04T00:00:00+00:00"


def test_months_quarters_and_days_name_their_bins_and_keep_empty_ones():
    times = utc("2026-08-14", "2026-10-02")
    assert [b["period"] for b in calendar_bins(times, "month")[1]] == ["2026-08", "2026-09", "2026-10"]
    assert [b["rows"] for b in calendar_bins(times, "month")[1]] == [1, 0, 1]
    assert [b["period"] for b in calendar_bins(times, "quarter")[1]] == ["2026-Q3", "2026-Q4"]
    assert [b["period"] for b in calendar_bins(utc("2026-08-14 23:00", "2026-08-15 01:00"), "day")[1]] == ["2026-08-14", "2026-08-15"]


def test_a_bin_the_upload_covers_in_part_is_partial():
    bins = calendar_bins(utc("2031-03-04 08:00", "2031-03-10", "2031-03-20 12:00"), "week")[1]
    assert [b["partial"] for b in bins] == [True, False, True]  # begins on a Tuesday; ends on a Thursday
    assert [b["covered"] for b in bins] == [pytest.approx(6 / 7, abs=1e-3), 1.0, pytest.approx(4 / 7, abs=1e-3)]
    assert [b["partial"] for b in calendar_bins(utc("2031-03-03", "2031-03-16 23:00"), "week")[1]] == [False, False]


def table(seed: int = 3, weeks: int = 6, rows: int = 3000) -> tuple[pd.DataFrame, pd.Series]:
    """A frame with a cohort ``g == 'a'`` whose ``m`` is +1 every week and +3 more in week 3, and its times, spread over each week."""
    rng = np.random.default_rng(seed)
    week = rng.integers(0, weeks, rows)
    group = rng.choice(["a", "b", "c"], rows)
    frame = pd.DataFrame({"g": group, "m": rng.normal(0, 1, rows) + (group == "a") * (1 + 3 * (week == 3)), "flat": np.ones(rows)})
    times = pd.Series(pd.Timestamp("2031-03-03", tz="UTC") + pd.to_timedelta(week * 7 + rng.integers(0, 7, rows), unit="D"))
    return frame, times


def series_of(frame, codes, bins, covers, metrics, min_rows=CFG.temporal_min_rows_bin, tie_share=CFG.temporal_tie_share):
    return annotate(frame, codes, bins, covers, metrics, reference_scale(frame), min_rows, tie_share)


def test_a_series_is_robust_z_against_the_rest_of_its_bin_and_counts_every_covered_row():
    frame, times = table()
    times.iloc[:40] = pd.NaT  # rows without a time: mined, in no bin
    codes, bins = calendar_bins(times, "week")
    rows = np.flatnonzero(frame.g == "a")
    out = series_of(frame, codes, bins, {"P": rows}, {"P": ["m", "flat"]})["P"]
    assert sum(out["rows"]) + out["untimed"] == len(rows) and out["untimed"] == int(times.iloc[rows].isna().sum()) > 0
    assert "flat" not in out["series"]  # a constant metric has no scale, so no series
    z = [e["z"] for e in out["series"]["m"]]
    steady = [v for i, v in enumerate(z) if i != 3]
    assert z[3] - max(steady) > 1.5 and np.ptp(steady) < 0.5 and min(steady) > 0.3  # the bin's population moves too
    assert all(0 < e["se"] < 0.3 and e["n"] >= CFG.temporal_min_rows_bin for e in out["series"]["m"])
    scale = reference_scale(frame)["m"]
    rest = (codes == 1) & (frame.g != "a")  # the bin's rows outside the insight: the reference
    own = (codes == 1) & (frame.g == "a")
    assert z[1] == pytest.approx((frame.m[own].median() - frame.m[rest].median()) / scale)


def test_a_cohort_that_is_most_of_its_bin_still_differs_from_the_rest():
    """Against the whole bin a cohort of nine rows in ten would be its own reference and read as nothing."""
    rng = np.random.default_rng(4)
    big = rng.random(3000) < 0.9
    frame = pd.DataFrame({"m": rng.normal(0, 1, 3000) + big})
    codes, bins = calendar_bins(
        pd.Series(pd.Timestamp("2031-03-03", tz="UTC") + pd.to_timedelta(rng.integers(0, 4, 3000) * 7 + rng.integers(0, 7, 3000), unit="D")), "week"
    )
    z = [e["z"] for e in series_of(frame, codes, bins, {"P": np.flatnonzero(big)}, {"P": ["m"]})["P"]["series"]["m"]]
    scale = reference_scale(frame)["m"]
    assert min(z) > 0.6 / scale and max(z) < 1.4 / scale  # about the planted +1, in reference units


def test_a_tie_heavy_metric_is_levelled_by_its_mean():
    """A metric most of whose values are one value: its bin median jumps between support values and reads as events;
    its level is the mean and the error s / √n. A continuous metric keeps the median."""
    rng = np.random.default_rng(2)
    k = np.where(rng.random(2000) < 0.6, 2.0, rng.integers(0, 5, 2000).astype(float))  # 60 % at the median, 2
    frame = pd.DataFrame({"k": k, "m": rng.normal(0, 1, 2000)})
    codes, bins = calendar_bins(
        pd.Series(pd.Timestamp("2031-03-03", tz="UTC") + pd.to_timedelta(rng.integers(0, 3, 2000) * 7 + rng.integers(0, 7, 2000), unit="D")), "week"
    )
    rows = np.flatnonzero(rng.random(2000) < 0.5)
    out = series_of(frame, codes, bins, {"P": rows}, {"P": ["k", "m"]})["P"]["series"]
    scale = reference_scale(frame)
    own, rest = np.zeros(2000, bool), np.ones(2000, bool)
    own[rows], rest[rows] = True, False
    s, r = frame[own & (codes == 0)], frame[rest & (codes == 0)]
    assert out["k"][0]["z"] == pytest.approx((s.k.mean() - r.k.mean()) / scale["k"])
    assert out["k"][0]["se"] == pytest.approx(np.hypot(s.k.std(ddof=0) / np.sqrt(len(s)), r.k.std(ddof=0) / np.sqrt(len(r))) / scale["k"])
    assert out["m"][0]["z"] == pytest.approx((s.m.median() - r.m.median()) / scale["m"])
    assert series_of(frame, codes, bins, {"P": rows}, {"P": ["k"]}, tie_share=0.7)["P"]["series"]["k"][0]["z"] == pytest.approx(
        (s.k.median() - r.k.median()) / scale["k"]
    )  # above the share, the median again


def test_a_short_edge_bin_is_unmeasured():
    """An upload that starts on a Saturday: its first week holds two days of it and is no level to read."""
    frame, _ = table()
    rng = np.random.default_rng(6)
    weekend = pd.Timestamp("2031-03-08", tz="UTC") + pd.to_timedelta(rng.integers(0, 6, len(frame)) * 7 + rng.integers(0, 2, len(frame)), unit="D")
    codes, bins = calendar_bins(pd.Series(weekend), "week")  # Saturdays and Sundays only: the first week holds two days of the upload
    assert bins[0]["covered"] == pytest.approx(2 / 7, abs=1e-3) and bins[0]["rows"] > 30 and bins[-1]["covered"] == 1.0
    out = series_of(frame, codes, bins, {"P": np.flatnonzero(frame.g == "a")}, {"P": ["m"]})["P"]
    assert out["series"]["m"][0] is None and out["series"]["m"][1] is not None and out["rows"][0] > 0  # counted, not levelled


@pytest.mark.parametrize(
    ("first", "last", "rows", "unit"),
    [
        ("2031-03-01", "2031-03-31", 30_000, "day"),  # a dense month: 31 days of ~1,000 rows
        ("2031-03-01", "2031-03-31", 600, "week"),  # a thin month: its days would hold 20 rows, its weeks 120
        ("2026-03-02", "2026-03-11", 5_000, "day"),  # ten days
        ("2026-01-01", "2026-12-31", 100_000, "week"),  # a year: 53 weeks of ~1,900 rows
        ("2026-01-01", "2026-12-31", 10_000, "month"),  # a thin year: 12 months of ~830 rows
        ("2024-01-01", "2026-12-31", 50_000, "month"),  # three years
        ("2020-01-01", "2026-12-31", 15_000, "quarter"),  # seven thin years
    ],
)
def test_the_grid_balances_the_bins_of_the_span_against_the_rows_a_bin_holds(first, last, rows, unit):
    times = pd.Series(pd.date_range(first, last, periods=rows, tz="UTC"))
    assert resolve_grid(times, "auto") == unit and resolve_grid(times, "month") == "month"
    stray = pd.concat([times, utc("1970-01-01")], ignore_index=True)  # one placeholder date: the span is the 1st–99th percentile
    assert resolve_grid(stray, "auto") == unit


def test_auto_coarsens_the_grid_until_the_median_insight_is_measured():
    """The span's grid weighs the upload's rows; an insight holds a few of them. Eight months of 22,000 rows make 35
    weeks of ~630 rows, but an insight of 3 % holds ~19 a week — under the floor — and ~90 a month."""
    times = pd.Series(pd.date_range("2031-01-01", "2031-08-31 23:00", periods=22_000, tz="UTC"))
    assert resolve_grid(times, "auto") == "week"
    small = {f"P{i}": np.arange(i, 22_000, 30) for i in range(5)}
    assert fit_grid(times, "week", small, 30, 3) == "month"
    assert fit_grid(times, "week", {**small, "Q": np.arange(0, 22_000, 3), "R": np.arange(1, 22_000, 3)}, 30, 3) == "month"  # the median's
    assert fit_grid(times, "week", {"P": np.arange(0, 22_000, 3)}, 30, 3) == "week"  # a third: ~210 a week
    tiny = {"P": np.arange(0, 22_000, 2_000)}  # 11 rows: no unit fits; the coarsest giving the bins the upload needs
    assert fit_grid(times, "week", tiny, 30, 3) == "quarter" and fit_grid(times, "week", tiny, 30, 4) == "month"


def test_a_carrier_names_the_part_of_an_insight_that_moved_in_its_events_bin():
    """An insight g = a with an event in week 3: rows k = x of it raised then carry it; the whole insight raised, or
    k = x raised in every week (a gap it always has), has no carrier."""
    rng = np.random.default_rng(4)
    week = rng.integers(0, 6, 6_000)
    frame = pd.DataFrame({"g": rng.choice(["a", "b"], 6_000), "k": rng.choice(["x", "y", "z"], 6_000), "m": rng.normal(0, 1, 6_000)})
    rows = np.flatnonzero(frame.g == "a")
    part = (frame.g == "a") & (frame.k == "x")

    def carried(m: pd.Series, kind: str = PEAKED_AT) -> dict:
        shifted = frame.assign(m=m)
        return carriers(shifted, week, {"P": rows}, {"P": [{"metric": "m", "bin": 3, "kind": kind}]}, reference_scale(shifted), CFG, ["g", "k"])

    (found,) = carried(frame.m + 3 * (part & (week == 3)))["P"]
    assert (found["attribute"], found["value"], found["bin"], found["metric"]) == ("k", "x", 3, "m")
    assert found["rows"] == (part & (week == 3)).sum() and found["of"] == ((frame.g == "a") & (week == 3)).sum()
    assert found["shift"] > 2 and found["p"] < 1e-6
    assert carried(frame.m - 3 * (part & (week == 3)), DROPPED_AT)["P"][0]["shift"] < -2  # a dip's carrier moved down
    assert carried(frame.m - 3 * (part & (week == 3))) == {}  # a carrier moves the way its event does
    assert carried(frame.m + 3 * ((frame.g == "a") & (week == 3))) == {}
    assert carried(frame.m + 3 * ((frame.g == "a") & (week == 3)) + 1.5 * (part & (week == 3))) == {}  # the rest moved too: broad
    assert carried(frame.m + 3 * part) == {}
    assert carriers(frame, week, {"P": rows}, {"P": [{"metric": "m", "bin": 3, "kind": PEAKED_AT}]}, reference_scale(frame), CFG, []) == {}


def test_an_event_a_broader_insight_has_too_is_its_echo():
    def fired(*bins: int, kind: str = PEAKED_AT) -> dict:
        return {"events": [{"metric": "m", "bin": b, "kind": kind} for b in bins]}

    def scope(support: int, **conditions: str) -> dict:
        return {"conditions": [{"attribute": a, "value": v} for a, v in conditions.items()], "support": support}

    trajectories = {"P": fired(3), "Q": fired(3, 5), "R": fired(3), "S": fired(3), "D": fired(3, kind=DROPPED_AT)}
    patterns = {
        "P": scope(900, g="a"),
        "Q": scope(300, g="a", k="x"),
        "R": scope(1_200, k="x"),  # broader than Q too, and larger than P
        "S": scope(900, g="b"),
        "D": scope(100, g="a", k="x", h="1"),  # a dip: no peak echoes it
    }
    echoes(trajectories, patterns)
    assert [e.get("echo_of") for e in trajectories["Q"]["events"]] == ["R", None]  # the broadest; week 5 is its own
    assert all("echo_of" not in e for pid in "PRSD" for e in trajectories[pid]["events"])


def test_a_bin_too_thin_for_the_insight_or_for_the_upload_is_unmeasured():
    frame, times = table()
    codes, bins = calendar_bins(times, "week")
    rows = np.flatnonzero(frame.g == "a")
    thin = np.concatenate([rows[codes[rows] != 2], rows[codes[rows] == 2][:5]])  # five of the insight's rows in week 2
    assert series_of(frame, codes, bins, {"P": thin}, {"P": ["m"]})["P"]["series"]["m"][2] is None
    times = times.mask((codes == 4) & (np.arange(len(codes)) % 40 > 0))  # the upload's week 4 under the floor
    codes, bins = calendar_bins(times, "week")
    every = series_of(frame, codes, bins, {"P": rows, "Q": np.flatnonzero(frame.g == "b")}, {"P": ["m"], "Q": ["m"]})
    assert bins[4]["rows"] < 30 and every["P"]["series"]["m"][4] is None and every["Q"]["series"]["m"][4] is None
    assert every["P"]["series"]["m"][5] is not None


def test_a_bin_with_no_spread_takes_the_uploads_spread():
    """A bin all of whose values tie has no spread of its own: the upload's stands in — never a zero error."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"k": np.where(np.arange(600) < 300, 2.0, rng.integers(0, 5, 600).astype(float))})
    times = pd.Series(pd.Timestamp("2031-03-03", tz="UTC") + pd.to_timedelta(np.arange(600) // 300 * 7, unit="D"))
    codes, bins = calendar_bins(times, "week")
    entry = series_of(frame, codes, bins, {"P": np.arange(0, 600, 2)}, {"P": ["k"]})["P"]["series"]["k"]
    assert entry[0]["se"] > 0 and entry[0]["z"] == pytest.approx(0.0)


def test_the_smoother_follows_a_constant_series_and_only_predicts_across_a_gap():
    x, p = rts_smooth([1.0, 1.0, None, 1.0, 1.0], [0.01] * 5, 0.1)
    assert x == pytest.approx([1.0] * 5) and p[2] > max(p[1], p[3])
    assert rts_smooth([None, 2.0, None], [None, 0.01, None], 0.1) == ([None, 2.0, None], [None, 0.01, None])


def test_a_thin_last_bin_is_pulled_to_its_neighbours_and_marks_no_event():
    y = [0.0, 0.1, -0.1, 0.0, 0.05, 3.0]
    x, p = rts_smooth(y, [0.01] * 5 + [4.0], 0.1)
    assert x[-1] < 1.0
    assert detect_events(x, p, range(6), 0.05, 0.5) == []


def test_a_monotone_series_has_no_event_and_a_planted_interior_peak_has_one():
    y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x, p = rts_smooth(y, [0.01] * 6, 0.1)
    assert detect_events(x, p, range(6), 0.05, 0.5) == []
    y = [0.0, 0.1, 3.0, 0.0, -0.1, 0.0]
    x, p = rts_smooth(y, [0.01] * 6, 0.1)
    events = detect_events(x, p, range(6), 0.05, 0.5)
    assert [(e["bin"], e["kind"]) for e in events] == [(2, PEAKED_AT)] and events[0]["p"] < 0.05 and events[0]["prominence"] > 2
    x, p = rts_smooth([-v for v in y], [0.01] * 6, 0.1)
    assert [(e["bin"], e["kind"]) for e in detect_events(x, p, range(6), 0.05, 0.5)] == [(2, DROPPED_AT)]


def test_a_hump_of_several_bins_is_one_event_at_its_top_and_an_edge_never_is():
    y = [0.0, 0.1, 2.0, 2.2, 2.1, 0.0, 0.1]
    x, p = rts_smooth(y, [0.01] * 7, 0.1)
    assert [(e["bin"], e["kind"]) for e in detect_events(x, p, range(7), 0.05, 0.5)] == [(3, PEAKED_AT)]
    x, p = rts_smooth([3.0, 0.0, 0.1, 0.0, 3.0], [0.01] * 5, 0.1)
    assert [(e["bin"], e["kind"]) for e in detect_events(x, p, range(5), 0.05, 0.5)] == [(2, DROPPED_AT)]  # a basin
    x, p = rts_smooth([0.0, 0.1, 0.0, 0.1, 3.0], [0.01] * 5, 0.1)
    assert detect_events(x, p, range(5), 0.05, 0.5) == []  # a jump in the last bin: what follows it is unknown


def test_a_peak_must_clear_the_least_prominence_and_the_sampling_noise():
    y = [0.0, 0.0, 0.4, 0.0, 0.0]
    x, p = rts_smooth(y, [0.001] * 5, 0.1)
    assert detect_events(x, p, range(5), 0.05, 0.5) == []  # significant, too small to matter
    y = [0.0, 0.0, 2.0, 0.0, 0.0]
    x, p = rts_smooth(y, [1.0] * 5, 0.1)
    assert detect_events(x, p, range(5), 0.05, 0.5) == []  # large, within the noise of a thin bin


def entry(zs: list[float | None], se: float = 0.1) -> dict:
    return {"series": {"m": [None if z is None else {"z": z, "n": 100, "se": se} for z in zs]}}


def test_dynamics_names_the_trend_the_current_level_the_events_and_the_gaps():
    bins = calendar_bins(utc("2031-03-03", "2031-04-27"), "week")[1]  # W10 … W17
    rising = dynamics(entry([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]), bins, ["m"], CFG)
    assert rising["trend"] == RISING and rising["events"] == [] and rising["current"] == pytest.approx(3.5, abs=0.1)
    assert rising["bins"] == [[b["t_start"], b["t_end"]] for b in bins]  # the bins' bounds, which the readers index by bin
    assert dynamics(entry([3.0, 2.0, 1.0, 0.0, -1.0, -1.5, -2.0, -2.5]), bins, ["m"], CFG)["trend"] == FALLING
    dip = dynamics(entry([2.0, 2.0, 2.0, None, None, 2.0, 2.0, 2.0]), bins, ["m"], CFG)
    assert dip["trend"] == FLAT and dip["gaps"] == [["2031-W13", "2031-W14"]] and dip["bins_measured"] == 6 and dip["of"] == 8
    spike = dynamics(entry([1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0]), bins, ["m"], CFG)
    assert spike["trend"] == SPIKE and [(e["period"], e["kind"], e["metric"]) for e in spike["events"]] == [("2031-W13", PEAKED_AT, "m")]
    assert spike["events"][0]["z_raw"] == 4.0 and spike["series"]["m"][3] == {"period": "2031-W13", "z": 4.0, "se": 0.1}
    unmeasured = dynamics(entry([None] * 8), bins, ["m"], CFG)
    assert (unmeasured["trend"], unmeasured["current"], unmeasured["bins_measured"], unmeasured["events"]) == (FLAT, None, 0, [])


def write(tmp_path, frame: pd.DataFrame):
    path = tmp_path / "events.csv"
    frame.to_csv(path, index=False)
    return path


def test_the_time_column_is_read_apart_and_its_grid_names_another_analysis(tmp_path):
    frame, times = table(rows=600)
    frame.insert(0, "when", times.dt.strftime("%Y-%m-%d %H:%M:%S"))
    frame.loc[3, "when"] = "not a time"
    path = write(tmp_path, frame)
    plain = load_dataset(path, CFG)
    weekly = load_dataset(path, CFG, time="when", grid="week")
    monthly = load_dataset(path, CFG, time="when", grid="Month")
    by_span = load_dataset(path, CFG, time="when")  # 600 rows over six weeks: days would hold 14 rows each, weeks 100
    assert "when" not in weekly.frame.columns and weekly.grid == "week" and monthly.grid == "month" and by_span.grid == "week"
    assert weekly.time.isna().sum() == 1 and len(weekly.time) == len(weekly.frame) and str(weekly.time.dt.tz) == "UTC"
    # auto's unit is fitted after admission (temporal.fit_grid): auto is another analysis than the unit it starts from
    assert len({plain.dataset_id, weekly.dataset_id, monthly.dataset_id, by_span.dataset_id}) == 4
    assert any("without a readable time" in w for w in weekly.warnings)
    assert plain.time is None and plain.grid is None


def test_a_time_column_reads_in_one_format_never_a_date_per_value():
    frame = pd.DataFrame({"when": ["02/03/2026", "09/03/2026", "16/03/2026", "23/03/2026", "2026-03-30"], "m": range(5)})
    times = read_times(frame, "when")  # day first: the only reading of 16/03; another format is no date
    assert times.dt.strftime("%Y-%m-%d").tolist()[:4] == ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23"] and times.isna().tolist()[4]
    iso = read_times(pd.DataFrame({"when": ["2026-03-02", "2026-03-02 10:00:00", "2026-03-02T10:00:00+03:00", "soon"]}), "when")
    assert iso.dt.strftime("%Y-%m-%d %H:%M").tolist()[:3] == ["2026-03-02 00:00", "2026-03-02 10:00", "2026-03-02 07:00"] and iso.isna().tolist()[3]


def test_a_geo_levels_cells_count_the_days_the_series_read():
    """One reading of a date column: the days a cell counts and the series of its patterns read the same dates — the
    day before the month where the column reads so, a value in another format NaT in both, never a day of its own."""
    frame = geo_frame().head(600)
    frame["when"] = [f"{1 + i % 28:02d}-{1 + i % 3:02d}-2026" for i in range(600)]  # 15-02-2026 reads one way only
    frame.loc[5, "when"] = "2026-07-08"  # ISO among them: not the column's format
    read = read_times(frame, "when")
    assert read.isna().tolist()[5] and (read.dt.day.tolist()[14], read.dt.month.tolist()[14]) == (15, 3)
    days = point_profile(frame, "latitude,longitude").times["when"]
    expected = read.dt.tz_localize(None).to_numpy().astype("datetime64[us]")
    assert np.array_equal(np.isnat(days), np.isnat(expected)) and np.array_equal(days[~np.isnat(days)], expected[~np.isnat(expected)])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"time": "missing"}, "not in dataset"),
        ({"time": "m"}, "holds numbers"),
        ({"time": "g"}, "reads as a time"),
        ({"time": "when", "grid": "7D"}, "a calendar unit"),
    ],
)
def test_time_options_are_refused(tmp_path, options, message):
    frame, times = table(rows=600)
    frame.insert(0, "when", times.dt.strftime("%Y-%m-%d"))
    with pytest.raises(IngestionError, match=message) as err:
        load_dataset(write(tmp_path, frame), CFG, **options)
    assert err.value.code == "invalid_options"


# the graph service: the series signed at the commit, the temporal plane derived at every compile


def phones_us(snapshot: dict) -> str:
    return next(
        n["id"]
        for n in snapshot["nodes"]
        if n["kind"] == "Pattern" and {(c["attribute"], c["value"]) for c in n["props"]["conditions"]} == {("category", "phones"), ("region", "US")}
    )


def test_a_time_column_makes_each_insight_a_trajectory_signed_with_its_batch(timed):
    engine, record = timed
    signed = record["temporal"]
    assert (signed["column"], signed["grid"], signed["rows_without_time"]) == ("when", "week", 10)
    assert [b["period"] for b in signed["bins"]] == [f"2031-W{w}" for w in range(10, 16)] and "when" not in record["profile"]["numerics"]
    path = engine.ws.datasets_dir / record["dataset_id"] / "temporal.json"
    assert signed["series_digest"] == hashlib.sha256(path.read_bytes()).hexdigest() and compute_metadata_integrity(record) == "ok"
    snap = engine.ws.load_graph()
    nodes = {n["id"]: n for n in snap["nodes"]}
    patterns = {n["id"]: n["props"] for n in snap["nodes"] if n["kind"] == "Pattern"}
    assert set(snap["trajectories"]) == set(patterns)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert all(sum(stored[pid]["rows"]) + stored[pid]["untimed"] == p["support"] for pid, p in patterns.items())
    dataset = next(n["props"] for n in snap["nodes"] if n["kind"] == "Dataset")
    read = pd.to_datetime(pd.read_csv(record["source_path"])["when"], errors="coerce", utc=True)
    assert dataset["temporal"] == {"column": "when", "grid": "week", "tz": None, "span": [read.min().isoformat(), read.max().isoformat()]}
    assert dataset["temporal"]["span"][0][:10] == "2031-03-03" and dataset["temporal"]["span"][1][:10] == "2031-04-13"  # the data's, not the bins'
    pid = phones_us(snap)
    trajectory = snap["trajectories"][pid]
    assert [(e["kind"], e["period"]) for e in trajectory["events"] if e["metric"] == "discount"] == [("PEAKED_AT", "2031-W13")]
    assert (
        trajectory["column"] == "when"
        and trajectory["metric"] == patterns[pid]["target"]
        and trajectory["of"] == 6
        and set(trajectory["series"]) == {s["metric"] for s in patterns[pid]["shifts"]}
    )
    core = trajectory["core"]  # where the insight lives: the shortest run of weeks holding 80 % of its timed rows
    assert patterns[pid]["temporal_summary"] == {k: trajectory[k] for k in ("metric", "trend", "current", "bins_measured", "of", "core")} | {
        "events": len(trajectory["events"])
    }
    assert trajectory["bins"] == [[b["t_start"], b["t_end"]] for b in signed["bins"]]
    rows = stored[pid]["rows"]
    assert core["share"] >= 0.8 and core["bins"] == core["last"] - core["first"] + 1 <= 5 and core["start"] == signed["bins"][core["first"]]["period"]
    assert sum(rows[core["first"] : core["last"] + 1]) / sum(rows) == pytest.approx(core["share"], abs=1e-3)
    assert (core["t_start"], core["t_end"]) == (signed["bins"][core["first"]]["t_start"], signed["bins"][core["last"]]["t_end"])
    hub = nodes["T:week:2031-W13"]
    assert hub["kind"] == "TemporalNode" and hub["props"] == {
        "granularity": "week",
        "period": "2031-W13",
        "t_start": "2031-03-24T00:00:00+00:00",
        "t_end": "2031-03-31T00:00:00+00:00",
    }
    plane = [e for e in snap["edges"] if e["plane"] == "temporal"]
    hubs = {n["id"] for n in snap["nodes"] if n["kind"] == "TemporalNode"}
    assert {e["target"] for e in plane} == hubs  # a hub with an event or a core's end in it, or above such a one
    assert {e["type"] for e in plane} <= {"PEAKED_AT", "DROPPED_AT", "STARTS_AT", "ENDS_AT", "WITHIN"}
    assert all(e["source"] in patterns or (e["type"] == "WITHIN" and e["source"] in hubs) for e in plane)
    ends = {e["type"]: e for e in plane if e["source"] == pid and e["type"] in ("STARTS_AT", "ENDS_AT")}
    assert (ends["STARTS_AT"]["target"], ends["ENDS_AT"]["target"]) == (f"T:week:{core['start']}", f"T:week:{core['end']}")
    assert ends["ENDS_AT"]["props"] == {"period": core["end"], "share": core["share"], "bins": core["bins"]}
    within = {(e["source"], e["target"]) for e in plane if e["type"] == "WITHIN"}
    assert {("T:week:2031-W13", "T:month:2031-03"), ("T:month:2031-03", "T:quarter:2031-Q1"), ("T:month:2031-04", "T:quarter:2031-Q2")} <= within
    assert any(s in ("T:week:2031-W14", "T:week:2031-W15") and t == "T:month:2031-04" for s, t in within)  # a week in its Thursday's month
    assert len(within) == len({s for s, _ in within})  # one period above each
    assert nodes["T:month:2031-04"]["props"] == {
        "granularity": "month",
        "period": "2031-04",
        "t_start": "2031-04-01T00:00:00+00:00",
        "t_end": "2031-05-01T00:00:00+00:00",
    }
    assert snap["stats"]["as_of"] == dataset["temporal"]["span"][1]  # the latest data: the reference date


def test_the_events_are_a_plane_no_walk_follows(timed):
    engine, _ = timed
    plane = {EdgeType.PEAKED_AT, EdgeType.DROPPED_AT, EdgeType.STARTS_AT, EdgeType.ENDS_AT, EdgeType.WITHIN}
    assert plane.isdisjoint(STRUCTURAL_EDGES | LATENT_EDGES | UNDIRECTED_EDGES) and {GraphEdge("a", "b", t).plane for t in plane} == {"temporal"}
    found = engine.search("Why is the discount higher for phones in the US?")
    assert found.evidence.items and not [e for e in found.traversal.used_edges if e.split(":", 1)[0] in {t.value for t in plane}]


def test_metrics_peaking_in_one_period_are_one_edge_and_a_filtered_view_keeps_its_own_periods(tmp_path):
    engine = make_engine(make_config(tmp_path / "ws"))
    path = weekly_retail(tmp_path, spiked=("discount", "margin"))
    with path.open("rb") as f:
        timed = engine.process(engine.upload(path.name, f, time="when", grid="week")[0]["batch_id"])
    plain = engine.ingest_file(path)
    snap = engine.ws.load_graph()
    ids = [e["id"] for e in snap["edges"]]
    assert len(ids) == len(set(ids))
    peaks = [e for e in snap["edges"] if e["source"] == phones_us(snap) and e["target"] == "T:week:2031-W13"]
    assert len(peaks) == 1 and sorted(peaks[0]["props"]["metrics"]) == ["discount", "margin"]
    assert peaks[0]["props"]["metric"] == peaks[0]["props"]["metrics"][0]  # the most prominent of them
    hubs = {n["id"] for n in snap["nodes"] if n["kind"] == "TemporalNode"}
    shown = {
        ds: {el["data"]["id"] for el in cytoscape_elements(engine.graph(), ds, False)["nodes"]} for ds in (timed["dataset_id"], plain["dataset_id"])
    }
    assert {"T:month:2031-03", "T:quarter:2031-Q1"} <= hubs <= shown[timed["dataset_id"]] and not hubs & shown[plain["dataset_id"]]
    starts = {el["data"]["id"]: el["data"]["t_start"] for el in cytoscape_elements(engine.graph())["nodes"] if el["data"]["kind"] == "TemporalNode"}
    assert starts.keys() == hubs and all(starts[h] == next(n["props"]["t_start"] for n in snap["nodes"] if n["id"] == h) for h in hubs)


def test_an_event_inside_a_broader_insight_is_carried_there_and_its_narrower_copy_is_an_echo(tmp_path):
    """Phones in the US raise their discount in week 13 through the online channel only: the event of phones ∧ US is
    carried by channel = online (signed in the series at the commit); phones ∧ US ∧ online has it too, as an echo."""
    engine = make_engine(make_config(tmp_path / "ws"))
    path = weekly_retail(tmp_path, rows=10_000, channel="online")
    with path.open("rb") as f:
        record = engine.process(engine.upload(path.name, f, time="when", grid="week")[0]["batch_id"])
    snap = engine.ws.load_graph()
    scopes = {frozenset((c["attribute"], c["value"]) for c in n["props"]["conditions"]): n["id"] for n in snap["nodes"] if n["kind"] == "Pattern"}
    broad = scopes[frozenset({("category", "phones"), ("region", "US")})]
    narrow = scopes[frozenset({("category", "phones"), ("region", "US"), ("channel", "online")})]

    def week_13(pid: str) -> dict:
        (event,) = [e for e in snap["trajectories"][pid]["events"] if (e["metric"], e["period"]) == ("discount", "2031-W13")]
        return event

    carrier = week_13(broad)["carrier"]
    assert (carrier["attribute"], carrier["value"]) == ("channel", "online") and carrier["shift"] > 2 and carrier["rows"] < carrier["of"]
    assert "echo_of" not in week_13(broad) and week_13(narrow)["echo_of"] == broad and "carrier" not in week_13(narrow)
    edges = {(e["source"], e["target"]): e["props"] for e in snap["edges"] if e["plane"] == "temporal"}
    assert edges[(broad, "T:week:2031-W13")]["carrier"] == carrier and edges[(narrow, "T:week:2031-W13")]["echo_of"] == broad
    stored = json.loads((engine.ws.datasets_dir / record["dataset_id"] / "temporal.json").read_text(encoding="utf-8"))
    assert [(c["attribute"], c["value"]) for c in stored[broad]["carriers"]] == [("channel", "online")]  # signed with the series
    facts = time_facts(snap["trajectories"][narrow], CFG.thresholds.temporal_fade_z)
    assert not [e for e in facts["events"] if e["period"] == "2031-W13"] and facts["echoes"][0]["echo_of"] == broad


def test_an_auto_grid_is_fitted_to_the_insights_and_the_record_keeps_auto(tmp_path):
    """Eight months of 12,000 orders: the span asks for weeks, the insights' rows for months."""
    frame = generate_retail_dataset(12_000, seed=7)
    frame.insert(0, "when", pd.date_range("2031-01-01", "2031-08-31 23:00", periods=len(frame)).strftime("%Y-%m-%d %H:%M:%S"))
    path = tmp_path / "months.csv"
    frame.to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    done = {}
    for grid in ("auto", "week"):
        with path.open("rb") as f:
            done[grid] = engine.process(engine.upload(path.name, f, time="when", grid=grid)[0]["batch_id"])
    assert [done[g]["status"] for g in done] == ["READY", "READY"] and done["auto"]["dataset_id"] != done["week"]["dataset_id"]
    assert (done["auto"]["grid"], done["auto"]["temporal"]["grid"], done["week"]["temporal"]["grid"]) == ("auto", "month", "week")
    snap = engine.ws.load_graph()
    of = {n["id"]: n["props"]["dataset_id"] for n in snap["nodes"] if n["kind"] == "Pattern"}
    measured = {g: np.median([t["bins_measured"] / t["of"] for p, t in snap["trajectories"].items() if of[p] == done[g]["dataset_id"]]) for g in done}
    assert measured["auto"] > 0.5 > measured["week"]
    one = next(p for p, ds in of.items() if ds == done["auto"]["dataset_id"])
    assert engine.verify_pattern(one)["status"] == "reproduced"  # re-read with its options as recorded: auto names the dataset


def restarted(engine):
    """The same workspace opened again: its snapshot compiled anew from the journals, the records and the files."""
    return make_engine(engine.settings).ws.load_graph()


def test_an_edited_time_record_or_series_file_gives_its_dataset_no_temporal_plane(tmp_path, caplog):
    engine = make_engine(make_config(tmp_path / "ws"))
    with weekly_retail(tmp_path).open("rb") as f:
        record = engine.process(engine.upload("weekly.csv", f, time="when", grid="week")[0]["batch_id"])
    assert restarted(engine)["trajectories"]
    path = engine.ws.datasets_dir / record["dataset_id"] / "temporal.json"
    original = path.read_bytes()
    for damaged in (original.replace(b'"z":', b'"z": ', 1), original[:-10]):  # other bytes; a file that no longer reads
        path.write_bytes(damaged)
        again = make_engine(engine.settings)
        snap = again.ws.load_graph()
        assert again.problem is None and not snap["trajectories"] and not any(n["kind"] == "TemporalNode" for n in snap["nodes"])
        assert "temporal" not in next(n["props"] for n in snap["nodes"] if n["kind"] == "Dataset")
    caplog.clear()  # the damaged files above logged the same line
    path.unlink()  # a missing file: no plane either, and the log names it
    again = make_engine(engine.settings)
    assert again.problem is None and not again.ws.load_graph()["trajectories"]
    assert f"{record['dataset_id']}: its series file is missing or not the one its record signs" in caplog.text
    path.write_bytes(original)
    edited = {**record, "temporal": {**record["temporal"], "grid": "month"}}
    engine.ws.save_batch(edited)
    snap = restarted(engine)
    assert not snap["trajectories"] and next(n["props"]["integrity"] for n in snap["nodes"] if n["kind"] == "Dataset") == "mismatch"


def test_a_time_record_of_the_chain_model_compiles_without_a_temporal_plane(tmp_path, caplog):
    engine = make_engine(make_config(tmp_path / "ws"))
    with weekly_retail(tmp_path).open("rb") as f:
        record = engine.process(engine.upload("weekly.csv", f, time="when", grid="week")[0]["batch_id"])
    window = {"index": 0, "of": 1, "column": "when", "size": "P7D", "t_start": "2031-03-03T00:00:00+00:00", "t_end": "2031-03-10T00:00:00+00:00"}
    chained = {**record, "window": window, "temporal": {"reference_scale": record["temporal"]["reference_scale"], "observations": {}}}
    engine.ws.save_batch({**chained, "metadata_digest": metadata_digest(chained)})  # signed as the chain model signed it
    snap = restarted(engine)
    assert not snap["trajectories"] and any(n["kind"] == "Pattern" for n in snap["nodes"])
    assert next(n["props"]["integrity"] for n in snap["nodes"] if n["kind"] == "Dataset") == "ok"  # still verifiable: its window is signed
    assert "no series_digest" in caplog.text


def test_time_options_are_refused_before_the_file_is_kept(cfg, tmp_path):
    engine = make_engine(cfg)
    path = weekly_retail(tmp_path)
    (tmp_path / "short").mkdir()
    two_weeks = weekly_retail(tmp_path / "short", weeks=2)
    for options, message in [
        ({"grid": "week"}, "grid or tz without time"),
        ({"tz": "Europe/Kyiv"}, "grid or tz without time"),
        ({"time": "when", "grid": "7D"}, "a calendar unit"),
        ({"time": "nope"}, "not in dataset"),
        ({"time": "discount"}, "holds numbers"),
    ]:
        with path.open("rb") as f, pytest.raises(PipelineError, match=message) as err:
            engine.upload(path.name, f, **options)
        assert err.value.code == "invalid_options"
    with two_weeks.open("rb") as f, pytest.raises(PipelineError, match="TEMPORAL_MIN_BINS=3"):
        engine.upload(two_weeks.name, f, time="when", grid="week")
    sentinel = tmp_path / "sentinel.csv"  # one placeholder date: 416,000 weeks, refused before a bin is built
    sentinel.write_text(path.read_text(encoding="utf-8").replace("unknown", "9999-12-31 00:00:00", 1), encoding="utf-8")
    with sentinel.open("rb") as f, pytest.raises(PipelineError, match="TEMPORAL_MAX_BINS=1000"):
        engine.upload(sentinel.name, f, time="when", grid="week")
    stray = tmp_path / "stray.csv"  # a year of rows and one epoch date: the grid is the year's (weeks), which the stray spans too widely
    year = generate_retail_dataset(20_000, seed=3)
    year.insert(0, "when", pd.date_range("2031-01-01", "2031-12-31", periods=len(year)).strftime("%Y-%m-%d %H:%M:%S"))
    year.loc[0, "when"] = "1970-01-01 00:00:00"
    year.to_csv(stray, index=False)
    with stray.open("rb") as f, pytest.raises(PipelineError, match="TEMPORAL_MAX_BINS=1000"):
        engine.upload(stray.name, f, time="when")
    assert not list(engine.ws.uploads_dir.glob("*")) and not engine.ws.list_batches()


def test_the_same_file_with_its_time_apart_is_another_dataset_and_its_deletion_takes_the_series(timed, tmp_path):
    engine = make_engine(make_config(tmp_path / "ws"))
    path = weekly_retail(tmp_path)
    with path.open("rb") as f:
        weekly = engine.process(engine.upload(path.name, f, time="when", grid="week")[0]["batch_id"])
    with path.open("rb") as f:
        daily = engine.process(engine.upload(path.name, f, time="when", grid="day")[0]["batch_id"])
    with path.open("rb") as f:
        again = engine.process(engine.upload(path.name, f, time="when", grid="week")[0]["batch_id"])
    plain = engine.ingest_file(path)
    assert [r["status"] for r in (weekly, daily, again, plain)] == ["READY", "READY", "SKIPPED", "READY"]
    assert len({weekly["dataset_id"], daily["dataset_id"], plain["dataset_id"]}) == 3 and weekly["dataset_id"] == timed[1]["dataset_id"]
    assert daily["temporal"]["grid"] == "day" and len(daily["temporal"]["bins"]) == 42 and "temporal" not in plain
    engine.delete_dataset(weekly["dataset_id"])
    snap = engine.ws.load_graph()
    assert not (engine.ws.datasets_dir / weekly["dataset_id"] / "temporal.json").exists()
    assert {t["grid"] for t in snap["trajectories"].values()} == {"day"}


def test_the_upload_form_takes_a_time_column_and_its_grid(engine, tmp_path):
    path = weekly_retail(tmp_path)
    with TestClient(create_app(engine)) as client, path.open("rb") as f:
        accepted = client.post("/api/upload", files={"file": (path.name, f, "text/csv")}, data={"time": "when", "grid": "day"})
        assert accepted.status_code == 202 and (accepted.json()["time"], accepted.json()["grid"]) == ("when", "day")
        refused = client.post("/api/upload", files={"file": (path.name, f, "text/csv")}, data={"grid": "week"})
    assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"


# geo + time: a cell pattern's series are those of its points (docs/02_discovery.md §2.10)


def geo_timed(folder: Path) -> Path:
    """``geo_frame`` with a dimension the table level finds a shift on (kind=a ∧ channel=web: load +3), ten points
    without a readable time and 300 rows without a point (their load far higher: no cell's population may see them)."""
    frame = geo_frame()
    frame["kind"] = np.random.default_rng(5).choice(["a", "b", "c"], len(frame))
    frame.loc[(frame["kind"] == "a") & (frame["channel"] == "web"), "load"] += 3
    unlocated = frame.sample(300, random_state=2).assign(latitude=np.nan, longitude=np.nan, load=lambda f: f["load"] + 20)
    frame = pd.concat([frame, unlocated], ignore_index=True)
    frame["when"] = frame["when"].astype(str)
    frame.loc[:9, "when"] = "unknown"
    path = folder / "points.csv"
    frame.to_csv(path, index=False)
    return path


def test_a_geo_patterns_points_are_those_of_its_cells():
    import h3

    frame = geo_frame().head(800)
    _, ctx, _ = build_cells(frame, "latitude,longitude", 7, min_cell_points=5)
    rows = {"P": np.array([0, 3, 5]), "Q": np.array([1])}
    placed = [h3.latlng_to_cell(a, b, 7) for a, b in zip(frame["latitude"], frame["longitude"])]
    brute = {pid: [i for i, cell in enumerate(placed) if cell in set(ctx.cells[r])] for pid, r in rows.items()}
    assert {pid: points.tolist() for pid, points in extent_points(ctx.point_cell, rows).items()} == brute


def test_a_geo_upload_with_a_time_column_gives_its_cell_patterns_the_series_of_their_points(tmp_path, monkeypatch):
    import h3

    import insight_graph_service.core.engine as engine_module

    engine = make_engine(make_config(tmp_path / "ws"))
    annotate = engine_module.annotate_batch

    def without_placing(*args):  # the commit's annotation reads the stored point → cell map: it never places a point again
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(h3, "latlng_to_cell", lambda *_: pytest.fail("a point placed again at the commit"))
            return annotate(*args)

    monkeypatch.setattr(engine_module, "annotate_batch", without_placing)
    path = geo_timed(tmp_path)
    cells, table = upload_levels(engine, path, geo="latitude,longitude@7", table=True, time="when", grid="week")
    assert (cells["level"], cells["status"], table["level"], table["status"]) == ("7", "READY", "table", "READY")
    weeks = [f"2019-W{w}" for w in range(10, 15)]
    assert [b["period"] for b in cells["temporal"]["bins"]] == [b["period"] for b in table["temporal"]["bins"]] == weeks  # one calendar
    assert cells["temporal"]["rows_without_time"] == table["temporal"]["rows_without_time"] == 10
    assert sum(b["rows"] for b in cells["temporal"]["bins"]) == 6000 - 10  # the located points; the table holds the rest too
    assert sum(b["rows"] for b in table["temporal"]["bins"]) == 6300 - 10
    plain = load_dataset(path, CFG, geo="latitude,longitude@7")  # time adds a layer: the cells are those of the upload without it
    assert load_dataset(path, CFG, geo="latitude,longitude@7", time="when", grid="week").frame.equals(plain.frame)
    assert "when" not in table["profile"]["numerics"] + table["profile"]["categoricals"]  # the table never mines the column itself
    stored = engine.ws.load_geo(cells["dataset_id"])
    raw = pd.read_csv(path)
    raw["t"] = pd.to_datetime(raw["when"], errors="coerce", utc=True, format="ISO8601")
    snap = engine.ws.load_graph()
    props = {n["id"]: n["props"] for n in snap["nodes"] if n["kind"] == "Pattern"}
    geo = [pid for pid, p in props.items() if p["dataset_id"] == cells["dataset_id"]]
    series = json.loads((engine.ws.datasets_dir / cells["dataset_id"] / "temporal.json").read_text(encoding="utf-8"))
    for pid in geo:  # a median has a series, a share or a counter none; the trend reads the target when it has one
        shifts = [s["metric"] for s in props[pid]["shifts"]]
        assert set(series[pid]["series"]) == {m for m in shifts if m.endswith("_median")}
        assert snap["trajectories"][pid]["metric"] == (props[pid]["target"] if props[pid]["target"] in series[pid]["series"] else "load_median")
        assert props[pid]["temporal_summary"]["metric"] == snap["trajectories"][pid]["metric"]
    assert any(not props[pid]["target"].endswith("_median") for pid in geo)  # a target without a series: the trend falls to the median
    covers = engine.ws.load_covers(cells["dataset_id"])
    assert all(sum(series[pid]["rows"]) + series[pid]["untimed"] == np.isin(stored["point_cell"], covers[pid]).sum() for pid in geo)
    # one bin of a cell pattern again from the file with h3 alone: its points in the week against the week's other located points
    located = raw[raw["latitude"].notna()].copy()
    located["cell"] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(located["latitude"], located["longitude"])]
    pid = next(p for p in geo if any(series[p]["series"]["load_median"]))
    t = next(i for i, e in enumerate(series[pid]["series"]["load_median"]) if e)
    bin_ = cells["temporal"]["bins"][t]
    week = located[(located["t"] >= pd.Timestamp(bin_["t_start"])) & (located["t"] < pd.Timestamp(bin_["t_end"]))]
    inside = week["cell"].isin(set(stored["cells"][covers[pid]]))
    own, rest = week[inside]["load"], week[~inside]["load"]
    s_ref = calculate_mad(located["load"]) / 0.6745
    assert series[pid]["series"]["load_median"][t]["z"] == pytest.approx((own.median() - rest.median()) / s_ref)
    assert series[pid]["series"]["load_median"][t]["n"] == own.notna().sum()
    tables = [pid for pid, p in props.items() if p["dataset_id"] == table["dataset_id"]]
    assert tables and all(snap["trajectories"][pid]["metric"] == props[pid]["target"] for pid in tables)  # a table's trend is its target's


# the readers: the time line in the evidence, a calendar period in the question (docs/07_question_answering.md §7.2, §7.4)


@pytest.mark.parametrize(
    ("text", "interval"),
    [
        ("what happened in 2031-W13?", ("2031-03-24", "2031-03-31")),
        ("2026-W53 then", ("2026-12-28", "2027-01-04")),
        ("in Q1 2031", ("2031-01-01", "2031-04-01")),
        ("in q2 2031 or 2031-w13", ("2031-04-01", "2031-07-01")),  # as typed, in any case
        ("in 2031-Q4", ("2031-10-01", "2032-01-01")),
        ("on 2031-03-24", ("2031-03-24", "2031-03-25")),
        ("during 2031-12", ("2031-12-01", "2032-01-01")),
        ("in 2031-W54 or 2031-13", None),
        ("2031-W54, that is 2031-W13", ("2031-03-24", "2031-03-31")),  # the first that names a period
        ("on 9999-12-31", None),  # its next day runs past the calendar
        ("2031-W13 or 2031-W14", ("2031-03-24", "2031-03-31")),  # two periods: the first
        ("2019-2020, v2031-03, 2031-03-24-1", None),
        ("in March", None),
    ],
)
def test_a_calendar_period_reads_as_its_utc_interval(text, interval):
    found = period_interval(text)
    assert (found and tuple(d.strftime("%Y-%m-%d") for d in found)) == interval or found is interval is None
    assert found is None or found[0].tzinfo is not None


def test_every_timed_insight_carries_its_time_line_with_or_without_a_date(timed):
    engine, _ = timed
    found = engine.search("Why is the discount higher for phones in the US?")
    pid = phones_us(engine.ws.load_graph())
    item = next(it for it in found.evidence.items if it.pattern_id == pid)
    assert item.time["trend"] and item.time["events"][0]["period"] == "2031-W13" and item.time["bins_measured"] == 6
    prompt, summary = found.evidence.to_prompt(), found.evidence.summary()
    assert f"[{item.key}]" in prompt and f"     time: {time_line(item.time)}" in prompt and "| events: peak 2031-W13 (discount, " in prompt
    assert "| span [2031-03-03, 2031-04-13] | trend spike «сплеск» (discount) |" in prompt  # the data's days; the word the answer should use
    assert "have a time line: each finding holds over its dataset's whole span" in prompt
    assert item.time["words"] == wording.time_words(item.time) and f"({item.time['words']})" in summary
    assert re.search(r"динаміка: сплеск; зараз: помітно вище \(\+2\.\d\d sd\); пік у періоді 2031-W13", summary)
    assert all(it.time for it in found.evidence.items)  # one dataset, with a time column: every item has its line


def test_a_period_in_the_question_raises_the_patterns_measured_in_it_and_never_lowers_one(timed):
    engine, _ = timed
    graph = engine.graph()
    pid = phones_us(graph.snapshot)
    week = period_interval("2031-W13")
    assert time_score(graph.trajectories[pid], week) == 1.0  # measured all week, its peak in it
    assert time_score(graph.trajectories[pid], period_interval("2031-W40")) == 0.0  # outside the upload
    assert time_score(graph.trajectories[pid], period_interval("2031-04")) == pytest.approx(13 / 60)  # its bins cover 1–13 April, no event
    steady = next(t for t in graph.trajectories.values() if not t["events"] and t["bins_measured"] == t["of"])
    assert time_score(steady, week) == 0.5  # present all week, nothing happened: half of one with an event
    ins, config = graph.insight(pid), engine.settings.query
    query = parse_query("discount for phones in the US in 2031-W13", graph)
    assert query.when == week
    asked, plain = score_pattern(ins, query, None, None, config, time=1.0), score_pattern(ins, query, None, None, config)
    assert asked.score - plain.score == pytest.approx(SEED_TIME_WEIGHT) and asked.matched["time"] == 1.0 and "time" not in plain.matched
    seeds = engine.search("Why is the discount higher for phones in the US in 2031-W13?").seeds
    assert seeds[0].pattern_id == pid and seeds[0].matched["time"] == 1.0


def test_a_coarse_event_is_no_peak_of_each_finer_period_and_a_period_is_read_on_the_datasets_calendar():
    def trajectory(periods: list[str], grid: str, event: str | None, tz: str | None = None) -> dict:
        series = {"m": [{"period": p, "z": 0.0, "se": 0.1} for p in periods]}
        bins = [list(period_bounds(p, grid, tz)) for p in periods]  # as the compile stores them: the readers never parse a label
        return {"metric": "m", "tz": tz, "bins": bins, "series": series, "events": [{"bin": periods.index(event), "metric": "m"}] if event else []}

    monthly = trajectory([f"2031-{m:02d}" for m in range(1, 7)], "month", "2031-03")
    assert time_score(monthly, period_interval("2031-W13")) == 0.5  # measured through the week; the event is the month's
    assert time_score(monthly, period_interval("2031-03")) == 1.0 and time_score(monthly, period_interval("2031-Q1")) == 1.0
    weekly = trajectory([f"2031-W{w:02d}" for w in range(9, 15)], "week", "2031-W13")  # 24 February … 6 April
    assert time_score(weekly, period_interval("2031-03")) == 1.0  # the week's peak lies in the month, measured throughout
    assert time_score(weekly, period_interval("2031-W13")) == 1.0 and time_score(weekly, period_interval("2031-W10")) == 0.5
    day = trajectory(["2031-03-24"], "day", None, "America/Los_Angeles")
    assert time_score(day, period_interval("2031-03-24")) == 0.5  # its day on its own calendar, not 16 of 24 UTC hours
    assert time_score(day, period_interval("2031-03-25")) == 0.0


def test_a_series_never_measured_has_no_trend_and_another_metrics_event_is_named():
    head = {"grid": "week", "span": ["2031-03-03T00:00:00+00:00", "2031-04-14T00:00:00+00:00"], "metric": "margin", "of": 6, "gaps": []}
    never = time_facts(head | {"trend": FLAT, "current": None, "bins_measured": 0, "events": []}, 0.5)
    assert (
        never["trend"] is None
        and never["words"] == "динаміка: не виміряно"
        and "| trend not measured «не виміряно» (margin) | current not measured |" in time_line(never)
    )
    peak = {"kind": "PEAKED_AT", "period": "2031-W13", "metric": "discount", "z_smoothed": 5.0, "prominence": 4.0}
    other = time_facts(head | {"trend": FLAT, "current": 0.1, "bins_measured": 6, "events": [peak]}, 0.5)
    assert other["words"] == "динаміка: без змін; зараз: нижче порога (+0.10 sd); пік у періоді 2031-W13 (discount)"


def test_an_events_carrier_is_named_and_an_echo_told_once_as_the_broader_findings():
    head = {"grid": "week", "span": ["2031-03-03T00:00:00+00:00", "2031-04-14T00:00:00+00:00"], "metric": "discount", "of": 6, "gaps": []}
    carrier = {"attribute": "channel", "value": "online", "rows": 78, "of": 153, "shift": 3.1, "p": 1e-9}
    peak = {"kind": "PEAKED_AT", "period": "2031-W13", "metric": "discount", "z_smoothed": 1.2, "prominence": 1.0, "carrier": carrier}
    dip = {"kind": "DROPPED_AT", "period": "2031-W11", "metric": "discount", "z_smoothed": -0.9, "prominence": 0.8, "echo_of": "P-broad"}
    facts = time_facts(head | {"trend": SPIKE, "current": 0.1, "bins_measured": 6, "events": [peak, dip]}, 0.5)
    assert facts["events"] == [{k: peak[k] for k in ("kind", "period", "metric", "z_smoothed", "carrier")}]
    assert facts["echoes"] == [{"kind": "DROPPED_AT", "period": "2031-W11", "metric": "discount", "echo_of": "P-broad"}]
    line = time_line(facts)
    assert "| events: peak 2031-W13 (discount, +1.20 sd, in channel = online: 78 of its 153 rows, +3.10 sd against the rest)" in line
    assert "| the same as a broader finding's: dip 2031-W11 (discount) |" in line
    assert facts["words"].endswith("пік у періоді 2031-W13, зосереджено в: channel — online")
    echoed = time_facts(head | {"trend": SPIKE, "current": 0.1, "bins_measured": 6, "events": [dip]}, 0.5)
    assert (
        echoed["events"] == []
        and echoed["words"].endswith("спад у періоді 2031-W11 (як у ширшої знахідки)")
        and "| events: none |" in time_line(echoed)
    )


def test_a_periods_bonus_reorders_the_seeds_and_never_drops_one(monkeypatch):
    fixed = {"timed": SeedMatch("timed", 0.6 + SEED_TIME_WEIGHT, {"time": 1.0}), "plain": SeedMatch("plain", 0.6, {})}
    monkeypatch.setattr(seeds, "score_pattern", lambda ins, *args: fixed[ins])
    graph = SimpleNamespace(insights={k: k for k in fixed}, trajectories={}, tampered=lambda pid: False, incident=lambda pid, kinds: [])
    query = SimpleNamespace(conditions=[], targets=[], text="margin in 2031-W13", when=None, components=lambda: [])
    found = seeds.resolve_seeds(query, graph, SimpleNamespace(encode_query=lambda *args: None), {}, None)
    assert [s.pattern_id for s in found] == ["timed", "plain"]  # the floor is taken without the bonus: 0.75 · 0.6, not 0.75 · 0.95


def test_a_period_alone_starts_from_the_pattern_with_the_most_of_it(monkeypatch):
    """Under the floor with nothing lexical, a question naming a period enters through its hub: what lives or moved in it
    (``DualGraph.members``) first, then the pattern with the highest time score — as a zone without text falls back to
    what lies in it; with words the catalog knows, to the semantic best."""
    fixed = {
        "steady": SeedMatch("steady", 0.2, {"semantic": 0.2, "time": 0.5}),
        "peaked": SeedMatch("peaked", 0.1 + SEED_TIME_WEIGHT, {"semantic": 0.1, "time": 1.0}),
        "member": SeedMatch("member", 0.05, {"semantic": 0.05, "weight": 0.4}),  # active in the week, no series measured there
    }
    monkeypatch.setattr(seeds, "score_pattern", lambda ins, *args: fixed[ins])
    held = {"active": [], "peaked": [], "dropped": []}
    graph = SimpleNamespace(
        insights={k: k for k in fixed},
        trajectories={},
        nodes={},
        tampered=lambda pid: False,
        incident=lambda pid, kinds: [],
        members=lambda asked: held,
    )
    query = SimpleNamespace(conditions=[], targets=[], text="2031-W13", when=period_interval("2031-W13"), is_lexical=False, components=lambda: [])
    ask = lambda: seeds.resolve_seeds(query, graph, SimpleNamespace(encode_query=lambda *args: None), {}, QueryConfig())  # noqa: E731
    found = ask()
    assert [s.pattern_id for s in found] == ["peaked"] and found[0].matched["fallback"] == "period" and "hub" not in found[0].matched
    held["active"], graph.nodes = ["member"], {"T:week:2031-W13": {}}
    found = ask()  # the hub's member first, though measured nowhere; the hub named once the graph holds it
    assert [s.pattern_id for s in found] == ["member"] and found[0].matched["hub"] == "T:week:2031-W13"
    query.is_lexical = True
    found = ask()
    assert [s.pattern_id for s in found] == ["steady"] and found[0].matched["fallback"] == "semantic"


def test_the_drawer_gets_the_curve_of_its_findings_metric_only(timed):
    engine, _ = timed
    graph = engine.graph()
    pid = phones_us(graph.snapshot)
    shown = node_details(graph, pid, engine.settings.thresholds)["trajectory"]
    assert set(shown["series"]) == set(shown["smoothed"]) == {shown["metric"]} and len(graph.trajectories[pid]["series"]) > 1
    assert shown["events"] == graph.trajectories[pid]["events"]


def test_no_readable_time_is_no_bin():
    codes, bins = calendar_bins(pd.Series([pd.NaT] * 3, dtype="datetime64[us, UTC]"), "week")
    assert codes.tolist() == [-1, -1, -1] and bins == []


def test_a_geo_level_checks_the_times_of_its_located_points(tmp_path):
    frame = geo_frame()
    frame["when"] = "2019-03-04"  # every located point in one week
    unlocated = frame.sample(300, random_state=2).assign(latitude=np.nan, longitude=np.nan)
    unlocated["when"] = pd.date_range("2019-03-04", periods=300, freq="D").strftime("%Y-%m-%d")  # the rows without a point: 43 weeks
    path = tmp_path / "points.csv"
    pd.concat([frame, unlocated], ignore_index=True).to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    with path.open("rb") as f:
        (level,) = engine.upload(path.name, f, geo="latitude,longitude@7", time="when", grid="week")  # the file's rows fill the weeks
    record = engine.process(level["batch_id"])  # the series would be of the located points: one week, refused before the search
    assert record["status"] == "FAILED" and record["error"]["code"] == "invalid_options" and record["failed_stage"] == "VALIDATING"
    assert "over its located points: 0 week bins measurable" in record["error"]["message"]  # one day of points: no week half covered


def test_a_geo_levels_auto_grid_is_the_unit_of_its_located_points(tmp_path):
    """Six weeks of located points, and rows without a point dated over three years: the file's rows ask for months,
    the points for days — a geo level's series are of its points, so its ``auto`` unit is theirs; the table level's,
    of every row, is the file's (coarsened again for its insights, most of whose rows lie in the points' six weeks)."""
    frame = geo_frame()
    frame["kind"] = np.random.default_rng(5).choice(["a", "b", "c"], len(frame))
    frame.loc[(frame["kind"] == "a") & (frame["channel"] == "web"), "load"] += 3  # a shift the table level finds
    frame["when"] = (EPOCH + (np.arange(len(frame)) % 42).astype("timedelta64[D]")).astype(str)
    unlocated = frame.sample(300, random_state=2).assign(latitude=np.nan, longitude=np.nan)
    unlocated["when"] = pd.date_range("2018-03-04", periods=300, freq="4D").strftime("%Y-%m-%d")
    path = tmp_path / "points.csv"
    pd.concat([frame, unlocated], ignore_index=True).to_csv(path, index=False)
    assert resolve_grid(read_times(pd.read_csv(path), "when"), "auto") == "month"  # the file's rows: the points fill two
    engine = make_engine(make_config(tmp_path / "ws"))
    cells, table = upload_levels(engine, path, geo="latitude,longitude@7", table=True, time="when")
    assert (cells["status"], cells["grid"]) == ("READY", "auto") and cells["temporal"]["grid"] in ("day", "week")
    assert table["status"] == "READY" and table["temporal"]["grid"] in ("month", "quarter")


def test_a_geo_pattern_without_a_series_has_no_time_line(tmp_path):
    frame = geo_frame().drop(columns="load")  # the cells hold counters, shares and modes only: no metric has a series
    frame["when"] = frame["when"].astype(str)
    path = tmp_path / "points.csv"
    frame.to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    (cells,) = upload_levels(engine, path, geo="latitude,longitude@7", time="when", grid="week")
    series = json.loads((engine.ws.datasets_dir / cells["dataset_id"] / "temporal.json").read_text(encoding="utf-8"))
    snap = engine.ws.load_graph()
    assert cells["status"] == "READY" and series and all(not entry["series"] for entry in series.values())
    assert not snap["trajectories"] and not any("temporal_summary" in n["props"] for n in snap["nodes"])  # never «not measured»


def test_a_time_zone_reads_wall_times_in_it_and_bins_on_its_calendar():
    """A time written without a zone is a wall time of ``tz``; one with an offset keeps it; the days are that zone's."""
    frame = pd.DataFrame({"when": ["2031-03-02 01:30", "2031-03-02T01:30:00+00:00", "2031-07-02 01:30"]})
    kyiv = read_times(frame, "when", "Europe/Kyiv")
    assert kyiv.dt.strftime("%Y-%m-%d %H:%M").tolist() == ["2031-03-01 23:30", "2031-03-02 01:30", "2031-07-01 22:30"]  # +2 in winter, +3 in summer
    assert read_times(frame, "when").dt.strftime("%H:%M").tolist() == ["01:30", "01:30", "01:30"]  # no zone: UTC
    typed = pd.DataFrame({"when": pd.to_datetime(["2031-03-02 01:30"])})  # a parquet's timestamp: typed, without a zone
    assert read_times(typed, "when", "Europe/Kyiv").dt.strftime("%Y-%m-%d %H:%M").tolist() == ["2031-03-01 23:30"]
    _, utc_days = calendar_bins(kyiv[:1], "day")
    _, local_days = calendar_bins(kyiv[:1], "day", tz="Europe/Kyiv")
    assert utc_days[0]["period"] == "2031-03-01" and local_days[0]["period"] == "2031-03-02"
    assert local_days[0]["t_start"] == "2031-03-02T00:00:00+02:00" and local_days[0]["t_end"] == "2031-03-03T00:00:00+02:00"
    assert parse_tz("") is None and parse_tz(" Europe/Kyiv ") == "Europe/Kyiv" and parse_tz("utc") is None  # UTC: no zone
    for name in ("Kyiv/Europe", "Europe", "europe/kyiv"):  # a folder of the zone database, a name as it is not written
        with pytest.raises(IngestionError, match="IANA name") as err:
            parse_tz(name)
        assert err.value.code == "invalid_options"
    hm = lambda f, tz: read_times(pd.DataFrame({"when": f}), "when", tz).dt.strftime("%Y-%m-%d %H:%M").tolist()  # noqa: E731
    assert hm(["02-03-2031", "03-03-2031"], "America/New_York") == ["2031-03-02 05:00", "2031-03-03 05:00"]  # a date's -2031 no offset
    assert hm(["2031-03-02T10:00:00+03", "2031-03-02T10:00-05"], "America/New_York") == ["2031-03-02 07:00", "2031-03-02 15:00"]
    assert hm(["2031-10-26 03:30"], "Europe/Kyiv") == ["2031-10-26 00:30"]  # the repeated hour: its first, never no time
    _, havana = calendar_bins(pd.Series(pd.to_datetime(["2031-11-02T04:30Z"])), "day", tz="America/Havana")
    assert havana[0]["t_start"] == "2031-11-02T00:00:00-04:00"  # a midnight that repeats: the day starts at its first


def test_a_time_zone_names_another_analysis_and_its_upload_bins_on_its_calendar(cfg, tmp_path):
    frame, times = table(rows=600)
    frame.insert(0, "when", times.dt.strftime("%Y-%m-%d %H:%M:%S"))
    path = write(tmp_path, frame)
    utc = load_dataset(path, CFG, time="when", grid="week")
    kyiv = load_dataset(path, CFG, time="when", grid="week", tz="Europe/Kyiv")
    assert kyiv.tz == "Europe/Kyiv" and utc.tz is None and kyiv.dataset_id != utc.dataset_id
    engine = make_engine(cfg)
    with path.open("rb") as f, pytest.raises(PipelineError, match="IANA name"):
        engine.upload(path.name, f, time="when", tz="Mars/Olympus")
    with weekly_retail(tmp_path).open("rb") as f:
        (level,) = engine.upload("weekly.csv", f, time="when", grid="week", tz="Europe/Kyiv")
    record = engine.process(level["batch_id"])
    assert record["status"] == "READY" and record["tz"] == "Europe/Kyiv"
    assert all(b["t_start"].endswith(("+02:00", "+03:00")) for b in record["temporal"]["bins"])  # the zone's Mondays


def test_a_parquet_upload_keeps_its_timestamps_typed(tmp_path):
    """A parquet file's time column is a timestamp already: it is read as one, in the zone the upload names."""
    frame, times = table(rows=600)
    frame.insert(0, "when", times.dt.tz_localize(None))
    path = tmp_path / "events.parquet"
    frame.to_parquet(path)
    loaded = load_dataset(path, CFG, time="when", grid="week", tz="Europe/Kyiv")
    assert str(loaded.time.dt.tz) == "UTC" and (loaded.time == (times.dt.tz_localize(None).dt.tz_localize("Europe/Kyiv").dt.tz_convert("UTC"))).all()
