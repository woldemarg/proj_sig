"""Time in one upload: the time column apart, the calendar grid, each insight's series, its smoothing and its events
(docs/02_discovery.md §2.10), and the temporal plane the graph service derives from them (docs/06_graph_and_storage.md §6.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import geo_frame, make_config, make_engine, upload_levels, weekly_retail
from fastapi.testclient import TestClient

from graph_query_engine import seeds
from graph_query_engine.question import parse_query
from graph_query_engine.seeds import SEED_TIME_WEIGHT, SeedMatch, score_pattern
from graph_query_engine.temporal import period_interval, time_facts, time_line, time_score
from insight_contracts import LATENT_EDGES, STRUCTURAL_EDGES, UNDIRECTED_EDGES, EdgeType, GraphEdge, wording
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.snapshot import compute_metadata_integrity, metadata_digest
from insight_graph_service.server.app import create_app
from insight_graph_service.server.views import cytoscape_elements, node_details
from subgroup_miner.config import MinerConfig
from subgroup_miner.ingestion import IngestionError, load_dataset, read_times
from subgroup_miner.spatial import build_cells
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
    extent_points,
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
    rising = dynamics(entry([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]), bins, ["m"], CFG)
    assert rising["trend"] == RISING and rising["events"] == [] and rising["current"] == pytest.approx(3.5, abs=0.1)
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
    assert dataset["temporal"] == {"column": "when", "grid": "week", "span": ["2031-03-03T00:00:00+00:00", "2031-04-14T00:00:00+00:00"]}
    pid = phones_us(snap)
    trajectory = snap["trajectories"][pid]
    assert [(e["kind"], e["period"]) for e in trajectory["events"] if e["metric"] == "discount"] == [("PEAKED_AT", "2031-W13")]
    assert (
        trajectory["column"] == "when"
        and trajectory["metric"] == patterns[pid]["target"]
        and trajectory["of"] == 6
        and set(trajectory["series"]) == {s["metric"] for s in patterns[pid]["shifts"]}
    )
    assert patterns[pid]["temporal_summary"] == {k: trajectory[k] for k in ("metric", "trend", "current", "bins_measured", "of")} | {
        "events": len(trajectory["events"])
    }
    hub = nodes["T:week:2031-W13"]
    assert hub["kind"] == "TemporalNode" and hub["props"] == {
        "granularity": "week",
        "period": "2031-W13",
        "t_start": "2031-03-24T00:00:00+00:00",
        "t_end": "2031-03-31T00:00:00+00:00",
    }
    events = [e for e in snap["edges"] if e["plane"] == "temporal"]
    assert {e["target"] for e in events} == {n["id"] for n in snap["nodes"] if n["kind"] == "TemporalNode"}  # a hub only with an event
    assert all(e["type"] in ("PEAKED_AT", "DROPPED_AT") and e["source"] in patterns for e in events)


def test_the_events_are_a_plane_no_walk_follows(timed):
    engine, _ = timed
    events = {EdgeType.PEAKED_AT, EdgeType.DROPPED_AT}
    assert events.isdisjoint(STRUCTURAL_EDGES | LATENT_EDGES | UNDIRECTED_EDGES) and {GraphEdge("a", "b", t).plane for t in events} == {"temporal"}
    found = engine.search("Why is the discount higher for phones in the US?")
    assert found.evidence.items and not [e for e in found.traversal.used_edges if e.split(":", 1)[0] in {t.value for t in events}]


def test_metrics_peaking_in_one_period_are_one_edge_and_a_filtered_view_keeps_its_own_periods(tmp_path):
    engine = make_engine(make_config(tmp_path / "ws"))
    path = weekly_retail(tmp_path, spiked=("discount", "margin"))
    with path.open("rb") as f:
        timed = engine.process(engine.upload(path.name, f, time="when")[0]["batch_id"])
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
    assert hubs <= shown[timed["dataset_id"]] and not hubs & shown[plain["dataset_id"]]


def restarted(engine):
    """The same workspace opened again: its snapshot compiled anew from the journals, the records and the files."""
    return make_engine(engine.settings).ws.load_graph()


def test_an_edited_time_record_or_series_file_gives_its_dataset_no_temporal_plane(tmp_path):
    engine = make_engine(make_config(tmp_path / "ws"))
    with weekly_retail(tmp_path).open("rb") as f:
        record = engine.process(engine.upload("weekly.csv", f, time="when")[0]["batch_id"])
    assert restarted(engine)["trajectories"]
    path = engine.ws.datasets_dir / record["dataset_id"] / "temporal.json"
    original = path.read_bytes()
    for damaged in (original.replace(b'"z":', b'"z": ', 1), original[:-10]):  # other bytes; a file that no longer reads
        path.write_bytes(damaged)
        again = make_engine(engine.settings)
        snap = again.ws.load_graph()
        assert again.problem is None and not snap["trajectories"] and not any(n["kind"] == "TemporalNode" for n in snap["nodes"])
        assert "temporal" not in next(n["props"] for n in snap["nodes"] if n["kind"] == "Dataset")
    path.write_bytes(original)
    edited = {**record, "temporal": {**record["temporal"], "grid": "month"}}
    engine.ws.save_batch(edited)
    snap = restarted(engine)
    assert not snap["trajectories"] and next(n["props"]["integrity"] for n in snap["nodes"] if n["kind"] == "Dataset") == "mismatch"


def test_a_time_record_of_the_chain_model_compiles_without_a_temporal_plane(tmp_path, caplog):
    engine = make_engine(make_config(tmp_path / "ws"))
    with weekly_retail(tmp_path).open("rb") as f:
        record = engine.process(engine.upload("weekly.csv", f, time="when")[0]["batch_id"])
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
        ({"grid": "week"}, "grid without time"),
        ({"time": "when", "grid": "7D"}, "a calendar unit"),
        ({"time": "nope"}, "not in dataset"),
        ({"time": "discount"}, "holds numbers"),
    ]:
        with path.open("rb") as f, pytest.raises(PipelineError, match=message) as err:
            engine.upload(path.name, f, **options)
        assert err.value.code == "invalid_options"
    with two_weeks.open("rb") as f, pytest.raises(PipelineError, match="TEMPORAL_MIN_BINS=3"):
        engine.upload(two_weeks.name, f, time="when")
    sentinel = tmp_path / "sentinel.csv"  # one placeholder date: 416,000 weeks, refused before a bin is built
    sentinel.write_text(path.read_text(encoding="utf-8").replace("unknown", "9999-12-31 00:00:00", 1), encoding="utf-8")
    with sentinel.open("rb") as f, pytest.raises(PipelineError, match="TEMPORAL_MAX_BINS=1000"):
        engine.upload(sentinel.name, f, time="when")
    assert not list(engine.ws.uploads_dir.glob("*")) and not engine.ws.list_batches()


def test_the_same_file_with_its_time_apart_is_another_dataset_and_its_deletion_takes_the_series(timed, tmp_path):
    engine = make_engine(make_config(tmp_path / "ws"))
    path = weekly_retail(tmp_path)
    with path.open("rb") as f:
        weekly = engine.process(engine.upload(path.name, f, time="when")[0]["batch_id"])
    with path.open("rb") as f:
        daily = engine.process(engine.upload(path.name, f, time="when", grid="day")[0]["batch_id"])
    with path.open("rb") as f:
        again = engine.process(engine.upload(path.name, f, time="when")[0]["batch_id"])
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

    engine = make_engine(make_config(tmp_path / "ws"))
    annotate = engine._annotate

    def without_placing(*args):  # the commit's annotation reads the stored point → cell map: it never places a point again
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(h3, "latlng_to_cell", lambda *_: pytest.fail("a point placed again at the commit"))
            return annotate(*args)

    monkeypatch.setattr(engine, "_annotate", without_placing)
    path = geo_timed(tmp_path)
    cells, table = upload_levels(engine, path, geo="latitude,longitude@7", table=True, time="when")
    assert (cells["level"], cells["status"], table["level"], table["status"]) == ("7", "READY", "table", "READY")
    weeks = [f"2019-W{w}" for w in range(10, 15)]
    assert [b["period"] for b in cells["temporal"]["bins"]] == [b["period"] for b in table["temporal"]["bins"]] == weeks  # one calendar
    assert cells["temporal"]["rows_without_time"] == table["temporal"]["rows_without_time"] == 10
    assert sum(b["rows"] for b in cells["temporal"]["bins"]) == 6000 - 10  # the located points; the table holds the rest too
    assert sum(b["rows"] for b in table["temporal"]["bins"]) == 6300 - 10
    assert not any(c.startswith("when") for c in cells["profile"]["numerics"] + cells["profile"]["categoricals"])  # time is never mined
    stored = engine.ws.load_geo(cells["dataset_id"])
    raw = pd.read_csv(path)
    raw["t"] = pd.to_datetime(raw["when"], errors="coerce", utc=True, format="ISO8601")
    assert stored["point_ts"].tolist() == raw["t"].dt.as_unit("us").array.asi8.tolist() and stored["point_ts"][0] == np.iinfo(np.int64).min
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
    # one bin of a cell pattern again from the file with h3 alone: its points in the week against every located point there
    located = raw[raw["latitude"].notna()].copy()
    located["cell"] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(located["latitude"], located["longitude"])]
    pid = next(p for p in geo if any(series[p]["series"]["load_median"]))
    t = next(i for i, e in enumerate(series[pid]["series"]["load_median"]) if e)
    bin_ = cells["temporal"]["bins"][t]
    week = located[(located["t"] >= pd.Timestamp(bin_["t_start"])) & (located["t"] < pd.Timestamp(bin_["t_end"]))]
    own = week[week["cell"].isin(set(stored["cells"][covers[pid]]))]["load"]
    s_ref = calculate_mad(located["load"]) / 0.6745
    assert series[pid]["series"]["load_median"][t]["z"] == pytest.approx((own.median() - week["load"].median()) / s_ref)
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
    assert "have a time line: each finding holds over its dataset's whole span" in prompt
    assert item.time["words"] == wording.time_words(item.time) and f"({item.time['words']})" in summary
    assert "динаміка: сплеск; зараз: помітно вище (+2.21 sd); пік у періоді 2031-W13" in summary
    assert all(it.time for it in found.evidence.items)  # one dataset, with a time column: every item has its line


def test_a_period_in_the_question_raises_the_patterns_measured_in_it_and_never_lowers_one(timed):
    engine, _ = timed
    graph = engine.graph()
    pid = phones_us(graph.snapshot)
    week = period_interval("2031-W13")
    assert time_score(graph.trajectories[pid], week) == 1.0  # measured all week, its peak in it
    assert time_score(graph.trajectories[pid], period_interval("2031-W40")) == 0.0  # outside the upload
    assert time_score(graph.trajectories[pid], period_interval("2031-04")) == pytest.approx(13 / 30)  # its bins cover 1–13 April, no event
    ins, config = graph.insight(pid), engine.settings.query
    query = parse_query("discount for phones in the US in 2031-W13", graph)
    assert query.when == week
    asked, plain = score_pattern(ins, query, None, None, config, time=1.0), score_pattern(ins, query, None, None, config)
    assert asked.score - plain.score == pytest.approx(SEED_TIME_WEIGHT) and asked.matched["time"] == 1.0 and "time" not in plain.matched
    seeds = engine.search("Why is the discount higher for phones in the US in 2031-W13?").seeds
    assert seeds[0].pattern_id == pid and seeds[0].matched["time"] == 1.0


def test_a_series_never_measured_has_no_trend_and_another_metrics_event_is_named():
    head = {"grid": "week", "span": ["2031-03-03T00:00:00+00:00", "2031-04-14T00:00:00+00:00"], "metric": "margin", "of": 6, "gaps": []}
    never = time_facts(head | {"trend": FLAT, "current": None, "bins_measured": 0, "events": []}, 0.5)
    assert (
        never["trend"] is None
        and never["words"] == "динаміка: не виміряно"
        and "| trend not measured (margin) | current not measured |" in time_line(never)
    )
    peak = {"kind": "PEAKED_AT", "period": "2031-W13", "metric": "discount", "z_smoothed": 5.0, "prominence": 4.0}
    other = time_facts(head | {"trend": FLAT, "current": 0.1, "bins_measured": 6, "events": [peak]}, 0.5)
    assert other["words"] == "динаміка: без змін; зараз: нижче порога (+0.10 sd); пік у періоді 2031-W13 (discount)"


def test_a_periods_bonus_reorders_the_seeds_and_never_drops_one(monkeypatch):
    fixed = {"timed": SeedMatch("timed", 0.6 + SEED_TIME_WEIGHT, {"time": 1.0}), "plain": SeedMatch("plain", 0.6, {})}
    monkeypatch.setattr(seeds, "score_pattern", lambda ins, *args: fixed[ins])
    graph = SimpleNamespace(insights={k: k for k in fixed}, trajectories={}, tampered=lambda pid: False, incident=lambda pid, kinds: [])
    query = SimpleNamespace(conditions=[], targets=[], text="margin in 2031-W13", when=None, components=lambda: [])
    found = seeds.resolve_seeds(query, graph, SimpleNamespace(encode_query=lambda *args: None), {}, None)
    assert [s.pattern_id for s in found] == ["timed", "plain"]  # the floor is taken without the bonus: 0.75 · 0.6, not 0.75 · 0.95


def test_the_drawer_gets_the_curve_of_its_findings_metric_only(timed):
    engine, _ = timed
    graph = engine.graph()
    pid = phones_us(graph.snapshot)
    shown = node_details(graph, pid, engine.settings.thresholds)["trajectory"]
    assert set(shown["series"]) == set(shown["smoothed"]) == {shown["metric"]} and len(graph.trajectories[pid]["series"]) > 1
    assert shown["events"] == graph.trajectories[pid]["events"]
