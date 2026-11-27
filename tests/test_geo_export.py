"""The graph service's geo exports (docs/08_interface.md §8.8): a geo dataset's rows with a ``row_id`` and their cells
(``GET /api/geo/{dataset}/rows``), every cell of each drawn zone and of the ring around it
(``POST /api/geo/{dataset}/zone_cells``) and how many of the rows each of those cell lists holds
(``POST /api/geo/{dataset}/zone_counts``)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import h3
import pandas as pd
from conftest import make_config, make_engine, upload_levels
from fastapi.testclient import TestClient

from insight_graph_service.server.app import create_app


def test_a_datasets_rows_go_once_and_a_zone_is_its_cells(tmp_path, points_csv):
    """The rows carry a ``row_id`` and the cell columns. A drawn zone is its cells, those whose centres its polygon holds,
    and a row is in it when its cell is one of them: a polygon that is the cells' outline holds exactly their rows, one
    smaller than a cell is that whole cell; the ring of cells around a zone is a list of its own; a cell without rows is
    listed too; another level's cells place the rows by that level; a bad polygon, a repeated id or one that is another
    zone's ring list, an unsafe id, a level no dataset has and an unknown dataset are refused."""
    engine = make_engine(make_config(tmp_path / "ws", weight_by_events=False))
    assert [r["status"] for r in upload_levels(engine, points_csv, geo="auto@8,7")] == ["READY", "READY"]
    seed = engine.search("Where is load higher?").seeds[0].pattern_id
    ds = engine.graph().insight(seed).dataset_id
    zone = [str(c) for c in engine.ws.load_geo(ds)["cells"][engine.ws.load_covers(ds)[seed]]]
    with TestClient(create_app(engine)) as client:
        # the dataset's rows with their cells: what an analysis outside SIG reads
        res, shape = h3.get_resolution(zone[0]), h3.h3shape_to_geo(h3.cells_to_h3shape(zone))
        cell = f"h3_r{res}"
        got = client.get(f"/api/geo/{ds}/rows")
        table, rows = json.loads(got.headers["x-rows"]), pd.read_csv(io.BytesIO(got.content), keep_default_na=False)
        assert got.headers["content-type"].startswith("text/csv") and table["resolution"] == table["dataset_resolution"] == res
        assert table["columns"] == ["row_id", cell, f"h3_r{res - 1}", f"h3_r{res - 2}"] and list(rows.columns)[0] == "row_id"
        assert list(rows["row_id"]) == list(range(table["total"])) and table["upload_id"].startswith("up-") and table["where"] is None
        assert (rows[f"h3_r{res - 1}"] == rows[cell].map(lambda c: h3.cell_to_parent(c, res - 1))).all() and "time" in table

        def held(cells: set[str], frame: pd.DataFrame = rows, column: str = cell) -> int:
            return int(frame[column].isin(cells).sum())

        # a zone is its cells: the outline's are the cells it was drawn around, the ring the cells around them, theirs left out
        busy = rows.loc[rows[cell].isin(zone), cell].mode()[0]  # a zone cell with points
        (la, lo), (va, vo) = h3.cell_to_latlng(busy), h3.cell_to_boundary(busy)[0]
        (ma, mo), d = ((la + va) / 2, (lo + vo) / 2), max(abs(va - la), abs(vo - lo)) / 20  # inside the cell, off its centre
        sliver = {"type": "Polygon", "coordinates": [[[mo - d, ma - d], [mo + d, ma - d], [mo + d, ma + d], [mo - d, ma + d], [mo - d, ma - d]]]}
        lone = h3.cell_to_boundary(h3.latlng_to_cell(-60.0, 100.0, res))  # one cell far from every point: no row in it
        empty = {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in (*lone, lone[0])]]}
        asked = [{"id": "Z1", "polygon": shape}, {"id": "Z2", "polygon": sliver}, {"id": "Z3", "polygon": empty}]
        got = client.post(f"/api/geo/{ds}/zone_cells", json={"zones": asked})
        listed = pd.read_csv(io.BytesIO(got.content))
        lists = {z: set(g[cell]) for z, g in listed.groupby("zone_id")}
        near = set().union(*(h3.grid_disk(c, 1) for c in zone))
        assert got.headers["content-type"].startswith("text/csv") and list(listed.columns) == ["zone_id", cell]
        assert list(lists) == ["Z1", "Z1_ring", "Z2", "Z2_ring", "Z3", "Z3_ring"] and not listed.duplicated().any()
        assert lists["Z1"] == set(zone) and lists["Z1_ring"] == near - set(zone)
        assert lists["Z2"] == {busy} and lists["Z2_ring"] == set(h3.grid_ring(busy, 1))  # smaller than a cell: the cell under it, whole
        assert len(lists["Z3"]) == 1 and len(lists["Z3_ring"]) == 6  # cells without rows are listed too
        # a drawn zone is its polygon alone: a member another kind of zone is read by changes nothing
        circled = [{"id": "Z1", "polygon": shape | {"center": [0.0, 0.0], "radius_m": 10}}]
        again = client.post(f"/api/geo/{ds}/zone_cells", json={"zones": circled}).text.splitlines()[1:]
        assert {line.split(",")[1] for line in again if line.startswith("Z1,")} == set(zone)

        # how many rows each list holds: a row is in a zone when its cell is listed for it, never by its point against the polygon
        counted = client.post(f"/api/geo/{ds}/zone_counts", json={"zones": asked}).json()
        assert counted["resolution"] == res and counted["total"] == len(rows) and counted["unlocated"] == held({""}) == 0
        assert {k: counted[k] for k in ("columns", "upload_id", "filename", "time", "where")} == {
            k: table[k] for k in ("columns", "upload_id", "filename", "time", "where")
        }
        assert counted["zones"] == {
            z: {"cells": len(lists[z]), "rows": held(lists[z]), "ring": {"cells": len(lists[f"{z}_ring"]), "rows": held(lists[f"{z}_ring"])}}
            for z in ("Z1", "Z2", "Z3")
        }
        z1 = counted["zones"]["Z1"]
        assert (
            0 < z1["rows"] < len(rows)
            and z1["ring"]["rows"] > 0
            and counted["zones"]["Z3"] == {"cells": 1, "rows": 0, "ring": {"cells": 6, "rows": 0}}
        )
        assert counted["zones"]["Z2"]["rows"] == held({busy}) > 0  # every row of the cell, though the polygon covers a sliver of it

        # refused on both exports: a shape that is no drawn zone, an id twice, an id that is another zone's ring list
        multi = {"type": "MultiPolygon", "coordinates": [shape["coordinates"]]}  # the outline as a one-part MultiPolygon
        shapes = [
            {"zones": [{"id": "Z", "polygon": p}]}
            for p in ({"type": "Point", "coordinates": [0, 0]}, {"type": "Polygon", "coordinates": [[]]}, multi)
        ]
        twice = {"zones": [{"id": "Z", "polygon": shape}, {"id": "Z", "polygon": shape}]}
        ring_named = {"zones": [{"id": "Z", "polygon": shape}, {"id": "Z_ring", "polygon": empty}]}
        for export in ("zone_cells", "zone_counts"):
            for refused in (*shapes, twice, ring_named, {"zones": asked, "resolution": 12}):  # 12: no dataset's level
                assert client.post(f"/api/geo/{ds}/{export}", json=refused).json()["code"] == "invalid_options", (export, refused)
            assert client.post(f"/api/geo/{ds}/{export}", json={"zones": [{"id": "a,b", "polygon": shape}]}).status_code == 422
            assert client.post(f"/api/geo/ds-unknown/{export}", json={"zones": []}).json()["code"] == "unknown_dataset"

        # the upload's other level: the zone read by that level's cells, the rows by their cell of that level
        other = ({7, 8} - {res}).pop()
        lines = client.post(f"/api/geo/{ds}/zone_cells", json={"zones": asked[:1], "resolution": other}).text.splitlines()
        coarse = {line.split(",")[1] for line in lines[1:] if line.startswith("Z1,")}
        assert lines[0] == f"zone_id,h3_r{other}" and {h3.get_resolution(c) for c in coarse} == {other}
        at = client.post(f"/api/geo/{ds}/zone_counts", json={"zones": asked[:1], "resolution": other}).json()
        relevelled = pd.read_csv(io.BytesIO(client.get(f"/api/geo/{ds}/rows", params={"resolution": other}).content), keep_default_na=False)
        assert at["resolution"] == other and at["columns"][1] == f"h3_r{other}"
        assert at["zones"]["Z1"]["cells"] == len(coarse) and at["zones"]["Z1"]["rows"] == held(coarse, relevelled, f"h3_r{other}")
        assert client.get(f"/api/geo/{ds}/rows", params={"resolution": 12}).json()["code"] == "invalid_options"
        assert client.get("/api/geo/ds-unknown/rows").json()["code"] == "unknown_dataset"

        # a layer link run's layer: its own file's points as an aligned run read them (outside its window: no cell), counted by the same lists
        points = rows.loc[rows[cell] == busy, ["longitude", "latitude"]].head(4)
        points.assign(t=["2031-03-02", "2031-03-30", "2031-04-02", "2031-02-01"]).to_csv(tmp_path / "bars.csv", index=False)
        record = engine.ws.ready_batch_for(ds)
        span = {"time": "t", "window": ["2031-03-01T00:00:00+00:00", "2031-03-31T00:00:00+00:00"]}  # as a run stores its target's span
        engine.ws.save_batch(record | {"context_ring": 1, "context": [{"name": "bars", "path": str(tmp_path / "bars.csv"), "geo": "auto"} | span]})
        got = client.get(f"/api/geo/{ds}/rows", params={"layer": "bars"})
        layer, bars = json.loads(got.headers["x-rows"]), pd.read_csv(io.BytesIO(got.content), keep_default_na=False)
        assert (layer["layer"], layer["filename"], layer["total"], layer["ring"], layer["window"]) == ("bars", "bars.csv", 4, 1, span["window"])
        assert list(bars[cell] != "") == [True, True, False, False]  # the run counted the window's points alone
        read = pd.read_csv(io.BytesIO(got.content))[cell]  # as an ADS kernel reads it, pandas' defaults: no point is NaN, not ""
        assert list(read.isna()) == [False, False, True, True] and (read != "").all()  # so the task names NaN
        placed = client.post(f"/api/geo/{ds}/zone_counts", json={"zones": asked[:1], "layer": "bars"}).json()
        assert placed["layer"] == "bars" and (placed["zones"]["Z1"]["rows"], placed["unlocated"], placed["total"]) == (2, 2, 4)  # no cell: in no zone
        assert client.get(f"/api/geo/{ds}/rows", params={"layer": "nope"}).json()["code"] == "unknown_dataset"
        bare = client.post(f"/api/geo/{ds}/zone_counts", json={"zones": []}).json()  # no zone: what the table is, alone
        assert bare["zones"] == {} and bare["total"] == len(rows)
        assert client.post(f"/api/geo/{ds}/zone_cells", json={"zones": []}).text.strip() == f"zone_id,{cell}"

        # the stored file is no longer the upload the insights were mined from: no row of another table goes out under this id
        stored = Path(engine.ws.ready_batch_for(ds)["source_path"])
        stored.write_bytes(stored.read_bytes() + b"\n")
        exports = (client.get(f"/api/geo/{ds}/rows"), *(client.post(f"/api/geo/{ds}/{e}", json={"zones": []}) for e in ("zone_counts", "zone_cells")))
        for changed in exports:
            assert (changed.status_code, changed.json()["code"]) == (409, "source_changed")
