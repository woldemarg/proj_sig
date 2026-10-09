"""The geo option: point rows -> H3 cell table, neighbourhood conditions, n_eff correction (docs/02_discovery.md §2.9)."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import make_config, make_engine

from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import _median_test
from subgroup_miner.ingestion import IngestionError, load_dataset
from subgroup_miner.spatial import build_cells, effective_n_factor, parse_geo, parse_points

HOT = (48.5, 35.25)  # the planted cluster: many points, mostly at night, late in the period


def geo_frame(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_bg, n_hot = 3000, 3000
    lat = np.concatenate([rng.uniform(48.25, 48.75, n_bg), rng.normal(HOT[0], 0.05, n_hot)])
    lon = np.concatenate([rng.uniform(35.0, 35.5, n_bg), rng.normal(HOT[1], 0.07, n_hot)])
    hot = np.r_[np.zeros(n_bg, bool), np.ones(n_hot, bool)]
    start = np.datetime64("2026-08-01")
    days = np.where(hot, rng.integers(15, 31, len(hot)), rng.integers(0, 31, len(hot)))
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(len(hot))],
            "latitude": lat,
            "longitude": lon,
            "when": start + days.astype("timedelta64[D]"),
            "shift": np.where(rng.random(len(hot)) < np.where(hot, 0.8, 0.2), "night", "day"),
            "sector": np.where(lat > 48.5, "north", "south"),
            "load": rng.normal(10, 2, len(hot)) + 4 * hot,
        }
    )


def cells_of(frame: pd.DataFrame, **kw):
    opts = dict(min_cell_points=5, share_max_levels=8, mode_max_levels=40, lisa_permutations=99, seed=42) | kw
    return build_cells(frame, "auto", 7, **opts)


def test_parse_geo_and_points():
    assert parse_geo("auto", 7) == ("auto", 7)
    assert parse_geo(" lat,lon @ 8", 7) == ("lat,lon", 8)
    with pytest.raises(ValueError):
        parse_geo("auto@99", 7)
    lat, lon = parse_points(pd.Series(['{"coordinates":[33.77,47.04],"type":"Point"}', "POINT (30.5 50.45)", "nowhere", None]))
    assert lat[:2].tolist() == [47.04, 50.45] and lon[:2].tolist() == [33.77, 30.5]
    assert np.isnan(lat[2:]).all()


def test_cell_table_roles_follow_dtypes():
    frame = geo_frame()
    table, ctx, warnings = cells_of(frame)
    assert ctx.location == "latitude,longitude" and len(table) == len(ctx.cells)
    assert table["points"].sum() == len(frame)  # every point lands in exactly one cell
    assert (ctx.point_cell >= 0).all()
    assert {"when_active_days", "when_late_share", "load_median", "share_shift_night", "sector_mode", "lisa_points"} <= set(table.columns)
    assert "event_id" not in table.columns and any("event_id" in w for w in warnings)  # an identifier is not aggregated
    sparse = table["points"] < 5
    assert sparse.any() and table.loc[sparse, "load_median"].isna().all()  # intensive metrics need MIN_CELL_POINTS
    assert table.loc[sparse, "points"].notna().all()  # counters are kept for every cell
    assert table["sector_mode"].isin(["north", "south"]).all()  # a place-like column becomes a mode condition


def test_lag_excludes_the_cell_and_counts_absent_neighbours_as_zero():
    table, ctx, _ = cells_of(geo_frame())
    pos = {c: i for i, c in enumerate(ctx.cells)}
    row = int(table["points"].idxmax())
    ring = h3.grid_ring(ctx.cells[row], 1)
    expected = sum(table["points"].iloc[pos[c]] for c in ring if c in pos) / len(ring)
    assert table["neighbours_points"].iloc[row] == pytest.approx(expected)
    assert ctx.derived_from["lisa_points"] == ["points", "when_active_days"]


def test_lisa_marks_the_planted_cluster_hot():
    table, ctx, _ = cells_of(geo_frame())
    centre = ctx.cells.tolist().index(h3.latlng_to_cell(*HOT, 7))
    assert table["lisa_points"].iloc[centre] == "hot spot"
    assert (table["lisa_points"] == "hot spot").sum() < len(table) / 4
    assert ctx.moran["points"] > 0.2  # the cluster is spatially autocorrelated


def test_effective_n_widens_the_median_test():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, 200)
    assert effective_n_factor(-0.4) == 1.0 and effective_n_factor(0.5) == pytest.approx(1 / 3)
    assert _median_test(values, 0.0, 0.67, effective_n_factor(0.5)) > _median_test(values, 0.0, 0.67)


def test_geo_option_is_strict_and_changes_the_dataset_id(tmp_path):
    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    plain = load_dataset(path, MinerConfig())
    cells = load_dataset(path, MinerConfig(), geo="auto")
    assert plain.spatial is None and plain.geo is None and len(plain.frame) == 6000
    assert cells.spatial is not None and cells.geo == "auto@7" and cells.dataset_id != plain.dataset_id
    assert "neighbours_points_band" in cells.frame.columns and "neighbours_points" not in cells.frame.columns
    with pytest.raises(IngestionError) as err:
        load_dataset(path, MinerConfig(), geo="nowhere")
    assert err.value.code == "invalid_options"


def test_geo_batch_end_to_end(tmp_path):
    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    record = engine.process(engine.submit(path, geo="auto")["batch_id"])
    assert record["status"] == "READY", record.get("error")
    geo = record["profile"]["geo"]
    assert geo["resolution"] == 7 and geo["sac"]["significant_corrected"] <= geo["sac"]["significant_iid"]
    patterns = [p for p in engine.ws.patterns() if p["dataset_id"] == record["dataset_id"]]
    assert all({"cells", "compactness", "n_eff", "colocated"} <= set(p["provenance"]["spatial"]) for p in patterns)
    for p in patterns:  # a neighbourhood condition never explains the counters it is computed from
        if any(c["attribute"] in ("lisa_points", "neighbours_points_band") for c in p["conditions"]):
            assert p["target"] not in ("points", "when_active_days")
    hot = [p for p in patterns if {"attribute": "lisa_points", "value": "hot spot"} in p["conditions"] and p["effect_size"] > 0]
    assert any(p["target"] in ("share_shift_night", "load_median", "when_late_share") for p in hot)  # the planted effects
    cells = engine.geo_cells(record["dataset_id"])
    assert len(cells["features"]) == geo["cells"] and {p["id"] for p in cells["patterns"]} == {p["id"] for p in patterns}
    one = engine.geo_pattern(record["dataset_id"], patterns[0]["id"])
    assert len(one["cells"]) == patterns[0]["support"] and 0 < len(one["points"]) <= one["n_points"]
    assert engine.geo_datasets()[0]["dataset_id"] == record["dataset_id"]
