"""Time in one upload: the time column apart, the calendar grid, each insight's series, its smoothing and its events
(docs/02_discovery.md §2.10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subgroup_miner.config import MinerConfig
from subgroup_miner.ingestion import IngestionError, load_dataset, read_times
from subgroup_miner.temporal import (
    DROPPED_AT,
    FALLING,
    FLAT,
    PEAKED_AT,
    RISING,
    SPIKE,
    annotate,
    calendar_bins,
    detect_events,
    dynamics,
    reference_scale,
    rts_smooth,
)

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
    assert [b["partial"] for b in calendar_bins(utc("2031-03-03", "2031-03-16 23:00"), "week")[1]] == [False, False]


def table(seed: int = 3, weeks: int = 6, rows: int = 3000) -> tuple[pd.DataFrame, pd.Series]:
    """A frame with a cohort ``g == 'a'`` whose ``m`` is +1 every week and +3 more in week 3, and its times."""
    rng = np.random.default_rng(seed)
    week = rng.integers(0, weeks, rows)
    group = rng.choice(["a", "b", "c"], rows)
    frame = pd.DataFrame({"g": group, "m": rng.normal(0, 1, rows) + (group == "a") * (1 + 3 * (week == 3)), "flat": np.ones(rows)})
    times = pd.Series(pd.Timestamp("2031-03-03", tz="UTC") + pd.to_timedelta(week * 7 + 2, unit="D"))
    return frame, times


def test_a_series_is_robust_z_against_its_bins_population_and_counts_every_covered_row():
    frame, times = table()
    times.iloc[:40] = pd.NaT  # rows without a time: mined, in no bin
    codes, bins = calendar_bins(times, "week")
    rows = np.flatnonzero(frame.g == "a")
    out = annotate(frame, codes, bins, {"P": rows}, {"P": ["m", "flat"]}, reference_scale(frame), CFG.temporal_min_rows_bin)["P"]
    assert sum(out["rows"]) + out["untimed"] == len(rows) and out["untimed"] == int(times.iloc[rows].isna().sum()) > 0
    assert "flat" not in out["series"]  # a constant metric has no scale, so no series
    z = [e["z"] for e in out["series"]["m"]]
    steady = [v for i, v in enumerate(z) if i != 3]
    assert z[3] - max(steady) > 1.5 and np.ptp(steady) < 0.5 and min(steady) > 0.3  # the bin's population moves too
    assert all(0 < e["se"] < 0.3 and e["n"] >= CFG.temporal_min_rows_bin for e in out["series"]["m"])


def test_a_bin_too_thin_for_the_insight_or_for_the_upload_is_unmeasured():
    frame, times = table()
    codes, bins = calendar_bins(times, "week")
    scale = reference_scale(frame)
    rows = np.flatnonzero(frame.g == "a")
    thin = np.concatenate([rows[codes[rows] != 2], rows[codes[rows] == 2][:5]])  # five of the insight's rows in week 2
    assert annotate(frame, codes, bins, {"P": thin}, {"P": ["m"]}, scale, 30)["P"]["series"]["m"][2] is None
    times = times.mask((codes == 4) & (np.arange(len(codes)) % 40 > 0))  # the upload's week 4 under the floor
    codes, bins = calendar_bins(times, "week")
    every = annotate(frame, codes, bins, {"P": rows, "Q": np.flatnonzero(frame.g == "b")}, {"P": ["m"], "Q": ["m"]}, scale, 30)
    assert bins[4]["rows"] < 30 and every["P"]["series"]["m"][4] is None and every["Q"]["series"]["m"][4] is None
    assert every["P"]["series"]["m"][5] is not None


def test_a_bin_with_no_spread_takes_the_uploads_mad():
    """A discrete metric whose bin values tie: the bin's MAD is 0, the upload's stands in — never a zero error."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"k": np.where(np.arange(600) < 300, 2.0, rng.integers(0, 5, 600).astype(float))})
    times = pd.Series(pd.Timestamp("2031-03-03", tz="UTC") + pd.to_timedelta(np.arange(600) // 300 * 7, unit="D"))
    codes, bins = calendar_bins(times, "week")
    entry = annotate(frame, codes, bins, {"P": np.arange(0, 600, 2)}, {"P": ["k"]}, reference_scale(frame), 30)["P"]["series"]["k"]
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
    bins = [{"period": f"2031-W{w:02d}"} for w in range(10, 18)]
    rising = dynamics(entry([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]), bins, "m", CFG)
    assert rising["trend"] == RISING and rising["events"] == [] and rising["current"] == pytest.approx(3.5, abs=0.1)
    assert dynamics(entry([3.0, 2.0, 1.0, 0.0, -1.0, -1.5, -2.0, -2.5]), bins, "m", CFG)["trend"] == FALLING
    dip = dynamics(entry([2.0, 2.0, 2.0, None, None, 2.0, 2.0, 2.0]), bins, "m", CFG)
    assert dip["trend"] == FLAT and dip["gaps"] == [["2031-W13", "2031-W14"]] and dip["bins_measured"] == 6 and dip["of"] == 8
    spike = dynamics(entry([1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0]), bins, "m", CFG)
    assert spike["trend"] == SPIKE and [(e["period"], e["kind"], e["metric"]) for e in spike["events"]] == [("2031-W13", PEAKED_AT, "m")]
    assert spike["events"][0]["z_raw"] == 4.0 and spike["series"]["m"][3] == {"period": "2031-W13", "z": 4.0, "se": 0.1}
    unmeasured = dynamics(entry([None] * 8), bins, "m", CFG)
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
    weekly = load_dataset(path, CFG, time="when")
    monthly = load_dataset(path, CFG, time="when", grid="Month")
    assert "when" not in weekly.frame.columns and weekly.grid == "week" and monthly.grid == "month"
    assert weekly.time.isna().sum() == 1 and len(weekly.time) == len(weekly.frame) and str(weekly.time.dt.tz) == "UTC"
    assert len({plain.dataset_id, weekly.dataset_id, monthly.dataset_id}) == 3
    assert any("without a readable time" in w for w in weekly.warnings)
    assert plain.time is None and plain.grid is None


def test_a_time_column_reads_in_one_format_never_a_date_per_value():
    frame = pd.DataFrame({"when": ["02/03/2026", "09/03/2026", "16/03/2026", "23/03/2026", "2026-03-30"], "m": range(5)})
    times = read_times(frame, "when")  # day first: the only reading of 16/03; another format is no date
    assert times.dt.strftime("%Y-%m-%d").tolist()[:4] == ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23"] and times.isna().tolist()[4]
    iso = read_times(pd.DataFrame({"when": ["2026-03-02", "2026-03-02 10:00:00", "2026-03-02T10:00:00+03:00", "soon"]}), "when")
    assert iso.dt.strftime("%Y-%m-%d %H:%M").tolist()[:3] == ["2026-03-02 00:00", "2026-03-02 10:00", "2026-03-02 07:00"] and iso.isna().tolist()[3]


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
