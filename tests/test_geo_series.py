"""A geo pattern's counts and shares over the calendar grid (the cross-layer experiments of 2026-11-04, R-10): a layer
link's insight is a count or a share near the layer, and it has a time line as a median has (docs/02_discovery.md
§2.10)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import geo_frame, make_config, make_engine, upload_levels

from subgroup_miner.config import MinerConfig
from subgroup_miner.ingestion import load_dataset
from subgroup_miner.vendor.eda.main_upd import calculate_mad

CFG = MinerConfig()


def test_a_geo_patterns_counts_and_shares_have_their_own_time_line(tmp_path):
    """A cell pattern on a count (``points``, a per-type counter) or a share has the series of its points (the cross-layer
    experiments of 2026-11-04, R-10: a layer link's insight is a count or a share): a share is the pattern's share of the
    type in each bin against the rest's, in the share's spread over the cells (the unit it was admitted in); a count its
    events per cell against the rest's per cell, a log rate ratio with ½ added, in the spread of log(1 + count) over the
    cells — one bin of each again from the file with h3 alone."""
    import h3

    frame = geo_frame().drop(columns="load")  # the cells hold counts, shares and modes only
    frame["when"] = frame["when"].astype(str)
    path = tmp_path / "points.csv"
    frame.to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    (cells,) = upload_levels(engine, path, geo="latitude,longitude@7", time="when", grid="week", count_values="channel=phone")
    series = json.loads((engine.ws.datasets_dir / cells["dataset_id"] / "temporal.json").read_text(encoding="utf-8"))
    snap = engine.ws.load_graph()
    measured = {m for entry in series.values() for m, s in entry["series"].items() if any(s)}
    assert cells["status"] == "READY" and any(m.startswith("share_") for m in measured) and measured & {"points", "count_channel_phone"}
    assert snap["trajectories"] and all(tr["metric"] in series[pid]["series"] for pid, tr in snap["trajectories"].items())
    table = load_dataset(path, CFG, geo="latitude,longitude@7", count_values=["channel=phone"]).frame
    stored, covers = engine.ws.load_geo(cells["dataset_id"]), engine.ws.load_covers(cells["dataset_id"])
    points = frame.assign(t=pd.to_datetime(frame["when"], utc=True, format="ISO8601"))
    points["cell"] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(points["latitude"], points["longitude"])]
    for metric in sorted(measured):
        pid = next(p for p, e in series.items() if any(e["series"].get(metric) or []))
        t = next(i for i, e in enumerate(series[pid]["series"][metric]) if e)
        bin_ = cells["temporal"]["bins"][t]
        week = points[(points["t"] >= pd.Timestamp(bin_["t_start"])) & (points["t"] < pd.Timestamp(bin_["t_end"]))]
        inside = week["cell"].isin(set(stored["cells"][covers[pid]]))
        if metric == "points":
            kind = pd.Series(True, index=week.index)
        else:
            _, column, value = metric.split("_", 2)
            kind = week[column].eq(value)
        if metric.startswith("share_"):
            expected = (kind[inside].mean() - kind[~inside].mean()) / (calculate_mad(table[metric].dropna()) / 0.6745)
        else:
            logged = np.log1p(table[metric].to_numpy(float))
            mad = np.median(np.abs(logged - np.median(logged)))
            k_s = len(covers[pid])
            ratio = np.log((kind[inside].sum() + 0.5) / k_s) - np.log((kind[~inside].sum() + 0.5) / (len(table) - k_s))
            expected = ratio / (mad / 0.6745 if mad else logged.std())
        assert series[pid]["series"][metric][t]["z"] == pytest.approx(expected), metric
