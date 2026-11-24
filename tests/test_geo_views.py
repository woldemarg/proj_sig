"""The map's reads over the graph (insight_graph_service/core/geo_views.py, docs/08_interface.md §8.7): the regions,
the window and what it lists, and every insight in words."""

from __future__ import annotations

import h3
import numpy as np
import pytest
from conftest import HOT, geo_frame, upload_levels
from fastapi.testclient import TestClient

from insight_graph_service.core.engine import PipelineError
from insight_graph_service.server.app import create_app


def test_the_period_bar_counts_the_window_and_a_patterns_points_in_its_days(engine, tmp_path):
    """With the bar's days, the window's points and a pattern's points are those in them; the rows stay the whole span's."""
    from graph_query_engine.temporal import parse_period

    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    (level,) = upload_levels(engine, path, geo="auto@8", time="when")
    ds, week = level["dataset_id"], parse_period("2019-03-04", "2019-03-10")
    counts = engine.maps().period_counts(ds, week)["counts"]
    whole, part = engine.maps().window(ds, "-180,-90,180,90"), engine.maps().window(ds, "-180,-90,180,90", period=week)
    assert whole["period_points"] is None and part["period_points"] == sum(counts) < part["points"] == whole["points"]
    assert part["themes"] == whole["themes"]
    pid = next(iter(engine.ws.load_covers(ds)))
    full, cut = engine.maps().pattern(ds, pid), engine.maps().pattern(ds, pid, period=week)
    assert cut["n_points"] == sum(counts[r] for r in full["rows"]) < full["n_points"] and cut["rows"] == full["rows"]
    assert cut["in_period"] and not full["in_period"]


def test_the_insights_idle_in_the_days_are_every_level_the_map_lists(engine, tmp_path):
    """/counts names the insights the map lists that do not act in the days (``DualGraph.members``), those of the
    upload's other levels too, which the console's graph may not hold; one with neither a core nor an event never is."""
    from graph_query_engine.temporal import parse_period

    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    levels = upload_levels(engine, path, geo="auto@8,7", time="when", grid="week", tz="Europe/Kyiv")
    ds = next(r["dataset_id"] for r in levels if (r.get("profile") or {}).get("geo", {}).get("resolution") == 8)
    graph, late = engine.graph(), parse_period("2019-04-02", "2019-04-04")
    listed = {p for f in engine.maps().overlay(ds)["features"] for ids in f["properties"]["ids"].values() for p in ids}
    timed = {p for p in listed if (t := graph.trajectories.get(p)) and (t.get("core") or t["events"])}
    found = graph.members(late)
    acting = {*found["active"], *(p for p, _ in found["peaked"]), *(p for p, _ in found["dropped"])}
    idle = engine.maps().period_counts(ds, late)["idle"]
    assert set(idle) == timed - acting and {graph.insight(p).dataset_id for p in idle} - {ds}  # another level's too


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
    from subgroup_miner.spatial_plane import cell_centres, link_min_events

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
    overlay = {f["properties"]["id"]: f["properties"] for f in engine.maps().overlay(ds)["features"]}
    for props in [p for p in overlay.values() if p["resolution"] == 5][:12]:  # a view of one region lists its n
        lon, lat = centres[props["rows"][0]]
        alone = engine.maps().window(ds, f"{lon - 1e-7},{lat - 1e-7},{lon + 1e-7},{lat + 1e-7}", 5)
        assert alone["regions"] == 1 and alone["insights"] == props["n"]
    split = float(np.median(centres[:, 0]))
    for bbox in (f"-180,-90,{split},90", f"{split},-90,180,90", "-180,-90,180,90"):
        window = engine.maps().window(ds, bbox, 5)
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
    assert engine.maps().window(ds, "-180,-90,180,90")["level"] == 7
    with pytest.raises(PipelineError, match="region level"):
        engine.maps().window(ds, "-180,-90,180,90", 8)


def test_a_map_counts_the_union_of_its_datasets(engine, points_csv, tmp_path):
    """Over several geo datasets (``map_set``) a region's ``n`` is the sum of each dataset's rows there, their insights
    being of different uploads, and a view of it lists their rows, each with its dataset; a level of the map's own upload
    adds nothing, since the map counts it already, and a dataset that is not a geo dataset is refused. The overlay splits
    each count by dataset and carries the trust of every insight another dataset lists, with the dataset that draws it,
    whose ``/patterns`` gives its cells' rings."""
    from insight_graph_service.core.geo_views import _projection, _region_sets, map_set
    from subgroup_miner.spatial_plane import cell_centres

    fine, coarse = upload_levels(engine, points_csv, geo="auto@8,7")
    geo_frame(seed=4).to_csv(other_path := tmp_path / "other.csv", index=False)
    (other,) = upload_levels(engine, other_path, geo="auto@8")
    a, b = fine["dataset_id"], other["dataset_id"]
    assert map_set(engine.ws, a, (coarse["dataset_id"], b)) == (a, b) and map_set(engine.ws, a, ()) == (a,)
    with pytest.raises(PipelineError, match="not geo datasets"):
        map_set(engine.ws, a, ("ds-nope",))
    alone = {f["properties"]["id"]: f["properties"]["n"] for f in engine.maps().overlay(a)["features"]}
    theirs = {f["properties"]["id"]: f["properties"]["n"] for f in engine.maps().overlay(b)["features"]}
    both = engine.maps().overlay(a, (b,))["features"]
    assert {f["properties"]["id"] for f in both} == set(alone) | set(theirs)
    assert all(f["properties"]["n"] == alone.get(f["properties"]["id"], 0) + theirs.get(f["properties"]["id"], 0) for f in both)
    shared = next(p for f in both if alone.get((p := f["properties"])["id"]) and theirs.get(p["id"]))  # a region where both list rows
    lon, lat = cell_centres(engine.ws.load_geo(a, ("cells",))["cells"])[shared["rows"][0]]
    window = engine.maps().window(a, f"{lon - 1e-7},{lat - 1e-7},{lon + 1e-7},{lat + 1e-7}", shared["resolution"], (b,))
    rows = [i for t in window["themes"] for i in t["insights"]]
    assert window["insights"] == len(rows) == shared["n"] and {i["dataset_id"] for i in rows} == {a, b}
    c = coarse["dataset_id"]  # a coarser map with a finer upload: the map's own levels read both
    levels = set(_region_sets(engine.ws, engine.graph(), c))
    coarser = engine.maps().overlay(c, (b,))["features"]
    assert {f["properties"]["resolution"] for f in coarser} <= levels
    assert all(
        any(f["properties"]["ids"].get(b) for f in coarser if f["properties"]["resolution"] == lv) for lv in levels
    )  # the finer file at every level
    assert engine.maps().window(c, "-180,-90,180,90", None, (b,))["level"] == max(levels)
    overlay = engine.maps().overlay(a, (b,))  # what «Усі датасети» reads: each region's count per dataset, every listed insight's trust
    assert all(sum(map(len, f["properties"]["ids"].values())) == f["properties"]["n"] for f in overlay["features"])
    away = {pid: t["map_dataset"] for pid, t in overlay["patterns"].items() if "map_dataset" in t}
    assert set(away) == set(_projection(engine.ws, engine.graph(), b)["patterns"]) and set(away.values()) == {b}
    drawn = engine.maps().pattern(b, next(iter(away)), rings=True)  # a map holding none of its cells draws it by their rings
    assert (
        "rings" not in engine.maps().pattern(b, next(iter(away)))
        and len(drawn["rings"]) == len(drawn["cells"]) > 0
        and drawn["level"] == "res 8"
        and {"effect_size", "support", "weight"} <= set(drawn)
    )
    assert {d["upload_id"] for d in engine.maps().datasets()} == {fine["upload_id"], other["upload_id"]}


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
        listed = {i["id"]: i for t in engine.maps().window(ds["dataset_id"], "-180,-90,180,90")["themes"] for i in t["insights"]}
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
    listed = {i["id"]: i["lift"] for t in engine.maps().window(coarse["dataset_id"], "-180,-90,180,90")["themes"] for i in t["insights"]}
    listed |= {
        a["id"]: 1.0 for t in engine.maps().window(coarse["dataset_id"], "-180,-90,180,90")["themes"] for i in t["insights"] for a in i["also"]
    }
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
        window = engine.maps().window(ds, "-180,-90,180,90")
    assert window["themes"] == [] and window["insights"] == 0 and window["points"] == 0 and window["cells"] == len(proj["per_cell"])


def test_a_window_across_the_antimeridian_holds_the_cells_of_both_sides():
    """``west > east`` is a view across ±180°: the cells on both sides of it, and none elsewhere; a bbox out of range
    is refused."""
    from insight_graph_service.core.geo_views import window_view
    from subgroup_miner.spatial_plane import cell_centres

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
    overlay = engine.maps().overlay(fine["dataset_id"])["patterns"]
    projected = {p["id"]: p["reading"] for p in engine.maps().projected(fine["dataset_id"])}
    window = engine.maps().window(fine["dataset_id"], "-180,-90,180,90")
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


def test_a_zone_tells_how_each_insight_lies_in_it():
    """Over located events (docs/07_question_answering.md §7.2): an insight wholly in the zone is inside, one partly in it
    but denser there beyond chance concentrated, one spread at the map's rate map-wide, one with too few events there
    few, one rarer there than on the map elsewhere, one without an event there outside; the ratio shown is the raw one
    the fit reads; a zone holding the whole map makes every insight map-wide."""
    from subgroup_miner.spatial_plane import cell_centres, points_in_zone, zone_measures

    centre = h3.latlng_to_cell(*HOT, 7)
    cells = np.array(sorted(h3.grid_disk(centre, 5)))
    point_cell = np.repeat(np.arange(len(cells)), 10)  # 910 events, 10 per cell; the zone's 7 cells hold 70
    lonlat = cell_centres(cells)[point_cell]
    ring = {k: np.flatnonzero(np.isin(cells, list(h3.grid_ring(centre, k)))) for k in range(6)}
    disk = np.flatnonzero(np.isin(cells, list(h3.grid_disk(centre, 1))))
    covers = {
        "local": disk,
        "dense": np.concatenate([disk, ring[3]]),
        "spread": np.arange(len(cells)),
        "few": np.concatenate([ring[0], ring[2], ring[3], ring[4]]),
        "rare": np.concatenate([disk[:3], ring[3], ring[4], ring[5]]),
        "out": ring[5],
    }
    zone = frozenset(h3.grid_disk(centre, 1))
    places = zone_measures(cells, point_cell, points_in_zone(lonlat, zone), covers, zone, by_points=False)
    fits = {"local": "inside", "dense": "concentrated", "spread": "map-wide", "few": "few", "rare": "elsewhere", "out": "outside"}
    assert {k: p.fit for k, p in places.items()} == fits
    local, dense, spread = places["local"], places["dense"], places["spread"]
    assert (local.share, local.cover, local.ratio, local.lift) == (1.0, 1.0, 13.0, round(100 / (70 * 70 / 910 + 30), 2))  # prior 30
    assert dense.ratio == round(70 / (250 * 70 / 910), 2) > dense.lift  # the fit and the number shown read the raw ratio
    assert spread.share == round(70 / 910, 4) and spread.ratio == spread.lift == 1.0 and places["out"].share == places["out"].cover == 0.0
    whole = frozenset(str(c) for c in cells)
    assert {p.fit for p in zone_measures(cells, point_cell, np.ones(910, dtype=bool), covers, whole, by_points=False).values()} == {"map-wide"}


def test_a_zones_fit_needs_more_than_chance_and_reads_one_place_on_every_level():
    """Five events where 3.3 are expected are no concentration (binomial tail 0.24), fifteen where 5 are are; an insight
    wholly in a zone of three quarters of the map is inside, one spread over it map-wide; and an event is in the zone or
    not alike on a coarse level and a fine one, so a coarse cell holding the zone counts only its points in it."""
    from subgroup_miner.spatial_plane import points_in_zone, zone_fit, zone_measures

    assert zone_fit(5, 100, 1000, 30000) == "map-wide" and zone_fit(15, 150, 1000, 30000) == "concentrated"
    assert zone_fit(100, 100, 750, 1000) == "inside" and zone_fit(75, 100, 750, 1000) == zone_fit(85, 100, 750, 1000) == "map-wide"
    assert zone_fit(4, 6, 50, 1000) == "few"  # two thirds of a small insight's events in the zone: too few to tell

    centre = h3.latlng_to_cell(*HOT, 7)
    fine = np.array(sorted(c for p in h3.grid_disk(centre, 1) for c in h3.cell_to_children(p, 9)))
    lonlat = np.array([h3.cell_to_latlng(str(c))[::-1] for c in fine])  # one event at each fine cell's centre
    coarse = np.array(sorted({h3.latlng_to_cell(lat, lon, 7) for lon, lat in lonlat}))
    on_coarse = np.searchsorted(coarse, [h3.latlng_to_cell(lat, lon, 7) for lon, lat in lonlat])
    mine = np.flatnonzero(coarse[on_coarse] == centre)  # the centre coarse cell's events
    zone = frozenset(h3.grid_disk(h3.cell_to_center_child(centre, 9), 1))
    in_zone = points_in_zone(lonlat, zone)
    seen = [
        zone_measures(coarse, on_coarse, in_zone, {"P": np.unique(on_coarse[mine])}, zone, by_points=False)["P"],
        zone_measures(fine, np.arange(len(fine)), in_zone, {"P": mine}, zone, by_points=False)["P"],
        zone_measures(coarse, on_coarse, in_zone, {"P": mine}, zone, by_points=True)["P"],  # a table level placed by the coarse one
    ]
    assert in_zone.sum() == 7 and len({(p.share, p.cover, p.ratio, p.fit) for p in seen}) == 1 and seen[0].cover == 1.0


def test_the_places_of_findings_in_drawn_zones_are_a_questions(engine, points_csv):
    """``POST /api/geo/places`` places each asked finding in each drawn zone as a question with that zone does, an id it
    does not weigh left out; two zones of one id are refused, and so is a workspace the engine cannot use."""
    from dataclasses import asdict

    upload_levels(engine, points_csv, geo="auto@8,7", table=True)
    ids = sorted(engine.graph().insights)
    hot = dict(h3.cells_to_geo(list(h3.grid_disk(h3.latlng_to_cell(*HOT, 8), 3))))
    with TestClient(create_app(engine)) as client:
        found = client.post("/api/geo/places", json={"zones": [{"id": "Z1", "polygon": hot}], "pattern_ids": [*ids, "P-none"]})
        assert found.status_code == 200
        asked = engine.place_scores(engine.committed(), geometry=hot) or {}
        assert found.json() == {"Z1": {pid: asdict(p) for pid, p in asked.items()}} and set(found.json()["Z1"]) <= set(ids)
        assert {p["fit"] for p in found.json()["Z1"].values()} & {"inside", "concentrated"}  # the planted cluster lies in it
        twice = client.post("/api/geo/places", json={"zones": [{"id": "Z1", "polygon": hot}] * 2, "pattern_ids": ids})
        assert twice.status_code == 400 and twice.json()["code"] == "invalid_options"
        engine.problem = "workspace was built with another representation"
        refused = client.post("/api/geo/places", json={"zones": [{"id": "Z1", "polygon": hot}], "pattern_ids": ids})
        assert refused.status_code == 409 and refused.json()["code"] == "workspace_degraded"
