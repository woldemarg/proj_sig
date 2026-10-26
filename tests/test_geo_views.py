"""The map's reads over the graph (insight_graph_service/core/geo_views.py, docs/08_interface.md §8.7): the regions,
the window and what it lists, and every insight in words."""

from __future__ import annotations

import h3
import numpy as np
import pytest
from conftest import geo_frame, upload_levels

from insight_graph_service.core.engine import PipelineError


def test_the_window_lists_the_insights_characteristic_of_its_regions(engine, tmp_path):
    """docs/08_interface.md §8.7: an insight is characteristic of a region when its located events in the region's cells
    reach ``link_min_events`` and ``(x + a) / (E + a) ≥ 1``, ``E`` the number its map-wide share puts there (a base insight
    with unlocated events counts its located ones); a base insight the graph leaves unplaced never is. A finer region
    also lists what the coarser one holding its cells lists, when it holds one of its events, so its children together
    list everything a region does. A view lists the
    rows of the regions holding its cells, each result once: one region alone its overlay ``n``, a wider view the rows of
    their union, ranked by how many of them an insight is characteristic of, then by lift. A level that is not a region
    level of the map is refused; without one the finest is read."""
    from insight_graph_service.core.geo_views import _projection, _region_sets, _rows, window_view
    from subgroup_miner.spatial import cell_centres, link_min_events

    frame = geo_frame()
    frame.loc[frame.sample(frac=0.1, random_state=1).index, ["latitude", "longitude"]] = np.nan  # events no map locates
    web = frame.index[frame["channel"] == "web"]
    frame.loc[web[::2], ["latitude", "longitude"]] = np.nan  # half the web events: an insight on them is partial
    frame.to_csv(path := tmp_path / "points.csv", index=False)
    fine, _, table = upload_levels(engine, path, geo="auto@8,7", table=True)
    ds, graph = fine["dataset_id"], engine.graph()
    geo = engine.ws.load_geo(ds, ("cells", "point_cell"))
    point_cell, covers, centres = geo["point_cell"], engine.ws.load_covers(table["dataset_id"]), cell_centres(geo["cells"])
    sets = _region_sets(engine.ws, graph, ds)
    assert set(sets) <= {1, 3, 5, 7} and all(any(s.listed) for s in sets.values())  # the levels coarser than the cells with a hint
    at = sets[5]
    partial = {pid for pid in covers if graph.nodes[pid]["props"].get("geo") == "partial"}
    assert partial and not partial & {pid for region in at.listed for pid in region}
    pid, r = next((p, r) for r, region in enumerate(at.own) for p in region if p in covers and (point_cell[covers[p]] < 0).any())
    located, mine = point_cell[point_cell >= 0], point_cell[covers[pid]]
    mine, inside = mine[mine >= 0], np.flatnonzero(at.idx == r)
    x, here = int(np.isin(mine, inside).sum()), int(np.isin(located, inside).sum())
    assert x >= link_min_events(len(mine)) and at.listed[r][pid] == pytest.approx((x + 30) / (len(mine) * here / len(located) + 30))  # a = 30
    levels = sorted(sets)
    for up, down in zip(levels, levels[1:], strict=False):  # a zoom never loses what the coarser region lists
        above, below = sets[up], sets[down]
        for r, region in enumerate(above.listed):
            kids = np.unique(below.idx[above.idx == r])
            assert set(region) <= set().union(*(below.listed[k] for k in kids))
    drawn = {pid: set(p.events.tolist()) for pid, p in _projection(engine.ws, graph, ds)["patterns"].items()}
    for s in sets.values():  # a region shows every insight it lists: each is drawn by a cell of it
        for r, region in enumerate(s.listed):
            assert all(drawn[pid] & set(np.flatnonzero(s.idx == r).tolist()) for pid in region)
    finest = sets[levels[-1]]
    assert any(set(listed) - own for listed, own in zip(finest.listed, finest.own, strict=True))  # some only from above
    overlay = {f["properties"]["id"]: f["properties"] for f in engine.geo_overlay(ds)["features"]}
    for props in [p for p in overlay.values() if p["resolution"] == 5][:12]:  # a view of one region lists its n
        lon, lat = centres[props["rows"][0]]
        alone = engine.geo_window(ds, f"{lon - 1e-7},{lat - 1e-7},{lon + 1e-7},{lat + 1e-7}", 5)
        assert alone["regions"] == 1 and alone["insights"] == props["n"]
    split = float(np.median(centres[:, 0]))
    for bbox in (f"-180,-90,{split},90", f"{split},-90,180,90", "-180,-90,180,90"):
        window = engine.geo_window(ds, bbox, 5)
        rows = sorted((i for t in window["themes"] for i in t["insights"]), key=lambda i: i["rank"])
        focus = np.unique(at.idx[window_view(centres, bbox)])
        expected = _rows(graph, ds, {p: 0 for r in focus for p in at.listed[r]})
        assert window["regions"] == len(focus) and [i["rank"] for i in rows] == list(range(window["insights"]))
        assert {i["id"] for i in rows} == set(expected) and len(rows) == len(expected)  # each result once
        assert not {i["id"] for i in rows} & {a["id"] for i in rows for a in i["also"]}
        keys = [(-i["regions"], -i["lift"]) for i in rows]
        assert keys == sorted(keys) and all(i["lift"] >= 1 and i["dataset_id"] == ds for i in rows)
        own = [i["rank"] for i in rows if not i["wider"]]  # a row listed only as a coarser region's after the place's own
        assert all(i["regions"] == 0 for i in rows if i["wider"]) and (
            not own or max(own) < min([i["rank"] for i in rows if i["wider"]], default=len(rows))
        )
    assert engine.geo_window(ds, "-180,-90,180,90")["level"] == 7
    with pytest.raises(PipelineError, match="region level"):
        engine.geo_window(ds, "-180,-90,180,90", 8)


def test_a_map_counts_the_union_of_its_datasets(engine, points_csv, tmp_path):
    """Over several geo datasets (``map_set``) a region's ``n`` is the sum of each dataset's rows there, their insights
    being of different uploads, and a view of it lists their rows, each with its dataset; a level of the map's own upload
    adds nothing, since the map counts it already, and a dataset that is not a geo dataset is refused."""
    from insight_graph_service.core.geo_views import _region_sets, map_set
    from subgroup_miner.spatial import cell_centres

    fine, coarse = upload_levels(engine, points_csv, geo="auto@8,7")
    geo_frame(seed=4).to_csv(other_path := tmp_path / "other.csv", index=False)
    (other,) = upload_levels(engine, other_path, geo="auto@8")
    a, b = fine["dataset_id"], other["dataset_id"]
    assert map_set(engine.ws, a, (coarse["dataset_id"], b)) == (a, b) and map_set(engine.ws, a, ()) == (a,)
    with pytest.raises(PipelineError, match="not geo datasets"):
        map_set(engine.ws, a, ("ds-nope",))
    alone = {f["properties"]["id"]: f["properties"]["n"] for f in engine.geo_overlay(a)["features"]}
    theirs = {f["properties"]["id"]: f["properties"]["n"] for f in engine.geo_overlay(b)["features"]}
    both = engine.geo_overlay(a, (b,))["features"]
    assert {f["properties"]["id"] for f in both} == set(alone) | set(theirs)
    assert all(f["properties"]["n"] == alone.get(f["properties"]["id"], 0) + theirs.get(f["properties"]["id"], 0) for f in both)
    shared = next(p for f in both if alone.get((p := f["properties"])["id"]) and theirs.get(p["id"]))  # a region where both list rows
    lon, lat = cell_centres(engine.ws.load_geo(a, ("cells",))["cells"])[shared["rows"][0]]
    window = engine.geo_window(a, f"{lon - 1e-7},{lat - 1e-7},{lon + 1e-7},{lat + 1e-7}", shared["resolution"], (b,))
    rows = [i for t in window["themes"] for i in t["insights"]]
    assert window["insights"] == len(rows) == shared["n"] and {i["dataset_id"] for i in rows} == {a, b}
    c = coarse["dataset_id"]  # a coarser map with a finer upload: the map's own levels read both
    levels = set(_region_sets(engine.ws, engine.graph(), c))
    assert {f["properties"]["resolution"] for f in engine.geo_overlay(c, (b,))["features"]} <= levels
    assert engine.geo_window(c, "-180,-90,180,90", None, (b,))["level"] == max(levels)


def test_one_result_of_two_levels_lists_once_at_the_maps_level(engine, points_csv):
    """A geo insight whose scale chain restates another level of its upload (``restating``) lists once, the other in its
    ``also``: hosts are the open map's own insights, then the finest; a level several hosts restate goes to the one whose
    subgroup is closest (``affinity``); hosts follow which levels a view lists, never its numbers."""
    from graph_query_engine.vertical import affinity
    from insight_graph_service.core.geo_views import _fold_levels, _restates

    fine, coarse, _ = upload_levels(engine, points_csv, geo="auto@8,7", table=True)
    graph = engine.graph()
    contested = 0
    for ds in (fine, coarse):
        listed = {i["id"]: i for t in engine.geo_window(ds["dataset_id"], "-180,-90,180,90")["themes"] for i in t["insights"]}
        rank = sorted(
            (p for p in listed if graph.facts(p).resolution is not None),
            key=lambda p: (graph.insight(p).dataset_id != ds["dataset_id"], -graph.facts(p).resolution, p),
        )
        for pid, row in listed.items():
            for a in row["also"]:
                assert a["id"] not in listed and a["id"] in _restates(graph, pid)
                claimants = [h for h in rank if a["id"] in _restates(graph, h)]
                contested += len(claimants) > 1
                assert pid == max(claimants, key=lambda h: (affinity(graph, h, a["id"]), -rank.index(h)))
            if row["also"]:  # the host is the map's own insight, else finer than what it folds
                assert graph.insight(pid).dataset_id == ds["dataset_id"] or graph.facts(pid).resolution > max(
                    graph.facts(a["id"]).resolution for a in row["also"]
                )
    assert contested  # the case that needs the rule: one level restated by several hosts
    # a panned view keeps its hosts: they follow which levels it lists, never its numbers or their order
    listed = {i["id"]: i["lift"] for t in engine.geo_window(coarse["dataset_id"], "-180,-90,180,90")["themes"] for i in t["insights"]}
    listed |= {a["id"]: 1.0 for t in engine.geo_window(coarse["dataset_id"], "-180,-90,180,90")["themes"] for i in t["insights"] for a in i["also"]}
    hosts = _fold_levels(graph, coarse["dataset_id"], listed)
    assert hosts == _fold_levels(graph, coarse["dataset_id"], {q: -v for q, v in reversed(listed.items())}) and any(hosts.values())


def test_a_map_without_a_located_point_lists_nothing(engine, points_csv):
    """A projection whose cells hold no event (never a READY level's, which refuses a file without a valid point) lists an
    empty window instead of dividing by its total."""
    from unittest import mock

    from insight_graph_service.core import geo_views

    (fine,) = upload_levels(engine, points_csv, geo="auto@7")
    ds = fine["dataset_id"]
    proj = dict(geo_views._projection(engine.ws, engine.graph(), ds))
    proj["per_cell"] = np.zeros_like(proj["per_cell"])
    with mock.patch.object(geo_views, "_projection", return_value=proj):
        window = engine.geo_window(ds, "-180,-90,180,90")
    assert window["themes"] == [] and window["insights"] == 0 and window["points"] == 0 and window["cells"] == len(proj["per_cell"])


def test_a_window_across_the_antimeridian_holds_the_cells_of_both_sides():
    """``west > east`` is a view across ±180°: the cells on both sides of it, and none elsewhere; a bbox out of range
    is refused."""
    from insight_graph_service.core.geo_views import window_view
    from subgroup_miner.spatial import cell_centres

    lons = (179.6, 179.9, -179.9, -179.6, 10.0)
    centres = cell_centres(np.array([h3.latlng_to_cell(-17.0, x, 6) for x in lons]))
    assert window_view(centres, "179.5,-18,-179.5,-16").tolist() == [True, True, True, True, False]
    assert window_view(centres, "179.5,-18,180,-16").tolist() == [True, True, False, False, False]
    assert window_view(centres, "-180,-18,180,-16").all()
    for bad in ("190,0,200,1", "0,5,1,4", "nan,0,1,1"):
        with pytest.raises(PipelineError, match="west,south"):
            window_view(centres, bad)
    from insight_graph_service.core.geo_views import lon_span  # the cells file's bbox: west > east across ±180° (M10-F05)

    assert lon_span(np.array([179.2, 179.6, -179.5])) == (179.2, -179.5) and lon_span(np.array([-114.3, -124.3])) == (-124.3, -114.3)


def test_every_map_read_says_an_insight_in_the_same_ukrainian_words(engine, points_csv):
    """The panel, the overlay the map card reads and the other levels' insights carry one reading per insight, the one the
    window's rows read from the map; the Dataset node's derived glossary stays English, for the prompt, and never reaches
    a reading."""
    from insight_contracts import dataset_node_id
    from insight_graph_service.server.views import node_details

    fine, coarse, table = upload_levels(engine, points_csv, geo="auto@8,7", table=True)
    graph, thresholds = engine.graph(), engine.settings.thresholds
    overlay = engine.geo_overlay(fine["dataset_id"])["patterns"]
    projected = {p["id"]: p["reading"] for p in engine.geo_projected(fine["dataset_id"])}
    window = engine.geo_window(fine["dataset_id"], "-180,-90,180,90")
    rows = [i["id"] for t in window["themes"] for i in t["insights"]]
    readings = {pid: t["reading"] for pid, t in overlay.items()} | projected
    assert table["dataset_id"] in {graph.insight(p).dataset_id for p in overlay} and coarse["dataset_id"] in {
        graph.insight(p).dataset_id for p in projected
    }
    assert rows and set(rows) <= set(readings)  # every row is an insight the map holds a reading for
    assert all(node_details(graph, p, thresholds)["reading"] == r for p, r in readings.items())
    glossary = graph.nodes[dataset_node_id(fine["dataset_id"])]["props"]["glossary"]
    words = " ".join(f"{r['where']} {r['what']}" for r in readings.values())
    assert glossary and not [g for g in glossary.values() if g in words] and "подій комірки" in words and "сусідні комірки" in words
    assert all(t["reading"].startswith("Повторювана картина") if t["id"] else t["label"] == "без теми" for t in window["themes"])
