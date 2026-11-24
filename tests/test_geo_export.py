"""The graph service's geo exports (docs/08_interface.md §8.8): a geo dataset's rows with a ``row_id`` and their cells
(``GET /api/geo/{dataset}/rows``), which of them drawn zones hold (``POST /api/geo/{dataset}/zone_rows``) and every
cell of each zone (``POST /api/geo/{dataset}/zone_cells``)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from conftest import make_config, make_engine, upload_levels
from fastapi.testclient import TestClient

from insight_graph_service.server.app import create_app
from subgroup_miner.spatial_plane import point_cells, zone_places


def test_a_datasets_rows_go_once_and_each_zone_by_its_row_ids(tmp_path, points_csv):
    """The rows carry a ``row_id`` and the cell columns; the rows each zone holds join them on it: a polygon that is the
    cells' outline holds exactly its cells' points, one cutting a cell splits that cell's points by its edge; another
    level's cells place the rows alike; a bad polygon, a repeated or unsafe id, a level no dataset has and an unknown
    dataset are refused."""
    engine = make_engine(make_config(tmp_path / "ws", weight_by_events=False))
    assert [r["status"] for r in upload_levels(engine, points_csv, geo="auto@8,7")] == ["READY", "READY"]
    seed = engine.search("Where is load higher?").seeds[0].pattern_id
    ds = engine.graph().insight(seed).dataset_id
    zone = [str(c) for c in engine.ws.load_geo(ds)["cells"][engine.ws.load_covers(ds)[seed]]]
    with TestClient(create_app(engine)) as client:
        # the dataset's rows with their cells, and the rows each drawn zone holds: what an analysis outside SIG reads
        res, shape = h3.get_resolution(zone[0]), h3.h3shape_to_geo(h3.cells_to_h3shape(zone))
        got = client.get(f"/api/geo/{ds}/rows")
        table, rows = json.loads(got.headers["x-rows"]), pd.read_csv(io.BytesIO(got.content), keep_default_na=False)
        assert got.headers["content-type"].startswith("text/csv") and table["resolution"] == table["dataset_resolution"] == res
        assert table["columns"] == ["row_id", f"h3_r{res}", f"h3_r{res - 1}", f"h3_r{res - 2}"] and list(rows.columns)[0] == "row_id"
        assert list(rows["row_id"]) == list(range(table["total"])) and table["upload_id"].startswith("up-") and table["where"] is None
        assert (rows[f"h3_r{res - 1}"] == rows[f"h3_r{res}"].map(lambda c: h3.cell_to_parent(c, res - 1))).all() and "time" in table
        busy = rows[rows[f"h3_r{res}"] == rows.loc[rows[f"h3_r{res}"].isin(zone), f"h3_r{res}"].mode()[0]]  # a zone cell's points
        mid, (s_, n_) = busy["longitude"].median(), (busy["latitude"].min() - 1e-6, busy["latitude"].max() + 1e-6)
        west = busy["longitude"].min() - 1e-6  # a box over the cell's western half: the edge cuts through the cell
        cut = {"type": "Polygon", "coordinates": [[[west, s_], [mid, s_], [mid, n_], [west, n_], [west, s_]]]}
        got = client.post(f"/api/geo/{ds}/zone_rows", json={"zones": [{"id": "Z1", "polygon": shape}, {"id": "Z2", "polygon": cut}]})
        placed, members = json.loads(got.headers["x-zones"]), pd.read_csv(io.BytesIO(got.content), keep_default_na=False)
        z1 = placed["zones"]["Z1"]
        assert (
            placed["resolution"] == res
            and z1["zone_cells"] == len(zone)
            and placed["total"] == len(rows) == z1["inside"] + z1["adjacent"] + z1["far"] + z1["unlocated"]
        )
        assert {k: placed[k] for k in ("columns", "upload_id", "filename", "time", "where")} == {
            k: table[k] for k in ("columns", "upload_id", "filename", "time", "where")
        }
        assert z1["inside"] > 0 and z1["far"] > 0 and set(members["aoi_zone"]) <= {"inside", "adjacent"}  # a row not listed is far
        one = members[members["zone_id"] == "Z1"].merge(rows, on="row_id")
        assert len(one) == z1["inside"] + z1["adjacent"] and (one.loc[one["aoi_zone"] == "inside", f"h3_r{res}"].isin(zone)).all()
        assert set(rows.loc[rows[f"h3_r{res}"].isin(zone), "row_id"]) == set(
            one.loc[one["aoi_zone"] == "inside", "row_id"]
        )  # the outline: in it is in them
        near = set().union(*(h3.grid_disk(c, 1) for c in zone))
        assert one[f"h3_r{res}"].isin(near).all() and not rows.loc[~rows["row_id"].isin(one["row_id"]), f"h3_r{res}"].isin(near).any()
        two = members[members["zone_id"] == "Z2"].merge(rows, on="row_id")  # a polygon cutting a cell: its points split by the edge
        halved = set(two.loc[two["aoi_zone"] == "inside", "row_id"]) & set(busy["row_id"])
        assert (
            0 < len(halved) < len(busy)
            and halved == set(busy.loc[busy["longitude"] < mid, "row_id"])
            and placed["zones"]["Z2"]["inside"] >= len(halved)
        )
        for bad in ({"type": "Point", "coordinates": [0, 0]}, {"type": "Polygon", "coordinates": [[]]}):
            assert client.post(f"/api/geo/{ds}/zone_rows", json={"zones": [{"id": "Z", "polygon": bad}]}).json()["code"] == "invalid_options"
        twice = {"zones": [{"id": "Z", "polygon": shape}, {"id": "Z", "polygon": shape}]}
        assert client.post(f"/api/geo/{ds}/zone_rows", json=twice).json()["code"] == "invalid_options"
        # every cell of each zone, with rows or without, the cells zone_rows counts: what a zone's per-cell figures divide by
        lone = h3.cell_to_boundary(h3.latlng_to_cell(-60.0, 100.0, res))  # one cell far from every point: no row in it
        empty = {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in (*lone, lone[0])]]}
        asked = [{"id": "Z1", "polygon": shape}, {"id": "Z3", "polygon": empty}]
        got = client.post(f"/api/geo/{ds}/zone_cells", json={"zones": asked})
        cells = pd.read_csv(io.BytesIO(got.content))
        counted = json.loads(client.post(f"/api/geo/{ds}/zone_rows", json={"zones": asked}).headers["x-zones"])["zones"]
        assert got.headers["content-type"].startswith("text/csv") and list(cells.columns) == ["zone_id", f"h3_r{res}"]
        assert set(cells.loc[cells["zone_id"] == "Z1", f"h3_r{res}"]) == set(zone) and counted["Z3"]["inside"] == 0
        assert cells["zone_id"].value_counts().to_dict() == {z: c["zone_cells"] for z, c in counted.items()} == {"Z1": len(zone), "Z3": 1}
        for refused in (twice, {"zones": [{"id": "Z", "polygon": {"type": "Point", "coordinates": [0, 0]}}]}, {"zones": asked, "resolution": 12}):
            assert client.post(f"/api/geo/{ds}/zone_cells", json=refused).json()["code"] == "invalid_options"
        assert client.post(f"/api/geo/{ds}/zone_rows", json={"zones": [{"id": "a,b", "polygon": shape}]}).status_code == 422
        other = ({7, 8} - {res}).pop()  # the upload's other level: another table's cells, the same places
        at = json.loads(
            client.post(f"/api/geo/{ds}/zone_rows", json={"zones": [{"id": "Z1", "polygon": shape}], "resolution": other}).headers["x-zones"]
        )
        assert at["resolution"] == other and at["zones"]["Z1"]["inside"] == z1["inside"]
        listed = client.post(f"/api/geo/{ds}/zone_cells", json={"zones": asked, "resolution": other}).text.splitlines()
        assert listed[0] == f"zone_id,h3_r{other}" and {h3.get_resolution(line.split(",")[1]) for line in listed[1:]} == {other}
        assert json.loads(client.get(f"/api/geo/{ds}/rows", params={"resolution": other}).headers["x-rows"])["columns"][1] == f"h3_r{other}"
        assert client.get(f"/api/geo/{ds}/rows", params={"resolution": 12}).json()["code"] == "invalid_options"  # no dataset's level
        assert client.get("/api/geo/ds-unknown/rows").json()["code"] == "unknown_dataset"
        # a layer link run's layer: its own file's points as an aligned run read them (outside its window: no cell), placed by the same zones
        bars = busy[["longitude", "latitude"]].head(4).assign(t=["2031-03-02", "2031-03-30", "2031-04-02", "2031-02-01"])
        bars.to_csv(tmp_path / "bars.csv", index=False)
        record = engine.ws.ready_batch_for(ds)
        span = {"time": "t", "window": ["2031-03-01T00:00:00+00:00", "2031-03-31T00:00:00+00:00"]}  # as a run stores its target's span
        engine.ws.save_batch(record | {"context_ring": 1, "context": [{"name": "bars", "path": str(tmp_path / "bars.csv"), "geo": "auto"} | span]})
        got = client.get(f"/api/geo/{ds}/rows", params={"layer": "bars"})
        layer, points = json.loads(got.headers["x-rows"]), pd.read_csv(io.BytesIO(got.content), keep_default_na=False)
        assert (layer["layer"], layer["filename"], layer["total"], layer["ring"], layer["window"]) == ("bars", "bars.csv", 4, 1, span["window"])
        assert list(points[f"h3_r{res}"] != "") == [True, True, False, False]  # the run counted the window's points alone
        read = pd.read_csv(io.BytesIO(got.content))[f"h3_r{res}"]  # as an ADS kernel reads it, pandas' defaults: no point is NaN, not ""
        assert list(read.isna()) == [False, False, True, True] and (read != "").all()  # so the task's recipe names NaN
        placed = json.loads(
            client.post(f"/api/geo/{ds}/zone_rows", json={"zones": [{"id": "Z1", "polygon": shape}], "layer": "bars"}).headers["x-zones"]
        )
        assert placed["layer"] == "bars" and (placed["zones"]["Z1"]["inside"], placed["zones"]["Z1"]["unlocated"]) == (2, 2)
        assert client.get(f"/api/geo/{ds}/rows", params={"layer": "nope"}).json()["code"] == "unknown_dataset"
        bare = client.post(f"/api/geo/{ds}/zone_rows", json={"zones": []})  # no zone: the table's counts alone
        assert bare.text.strip() == "zone_id,row_id,aoi_zone" and json.loads(bare.headers["x-zones"])["zones"] == {}
        # the stored file is no longer the upload the insights were mined from: no row of another table goes out under this id
        stored = Path(engine.ws.ready_batch_for(ds)["source_path"])
        stored.write_bytes(stored.read_bytes() + b"\n")
        exports = (client.get(f"/api/geo/{ds}/rows"), *(client.post(f"/api/geo/{ds}/{e}", json={"zones": []}) for e in ("zone_rows", "zone_cells")))
        for changed in exports:
            assert (changed.status_code, changed.json()["code"]) == (409, "source_changed")


def test_a_row_without_a_point_is_in_no_place():
    """A row without a valid point is neither in a zone nor far from it: the exports leave it out of every comparison group."""
    lat, lon = np.array([0.5, np.nan, 5.0]), np.array([0.5, 0.5, 5.0])
    square = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    place, cells = zone_places(lat, lon, point_cells(lat, lon, 5), 5, square)
    assert list(place) == ["inside", "unlocated", "far"] and cells > 0
