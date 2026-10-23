"""The map's reads over the graph (insight_graph_service/core/geo_views.py, docs/08_interface.md §8.7): the window and
what it lists, its themes grouped, and every insight in words."""

from __future__ import annotations

import h3
import numpy as np
import pytest
from conftest import HOT, geo_frame, upload_levels

from insight_graph_service.core.engine import PipelineError


def test_the_window_lists_what_is_characteristic_of_the_view(engine, tmp_path):
    """docs/08_interface.md §8.7: an insight counts all its located events, over the map's located points, so the whole
    map gives every insight lift 1, a base one with unlocated events too; a part of the map lists only insights with
    enough events there and lift at least 1, by ``(x + a) / (E + a)``; a view of too few events lists nothing; a base insight the graph leaves unplaced
    is not listed."""
    from subgroup_miner.spatial import GEO_LINK_MIN_EVENTS, cell_centres, link_min_events

    frame = geo_frame()
    frame.loc[frame.sample(frac=0.1, random_state=1).index, ["latitude", "longitude"]] = np.nan  # events no map locates
    web = frame.index[frame["channel"] == "web"]
    frame.loc[web[::2], ["latitude", "longitude"]] = np.nan  # half the web events: an insight on them is partial
    frame.to_csv(path := tmp_path / "points.csv", index=False)
    fine, _, table = upload_levels(engine, path, geo="auto@8,7", table=True)
    geo = engine.ws.load_geo(fine["dataset_id"], ("cells", "point_cell"))
    point_cell, covers = geo["point_cell"], engine.ws.load_covers(table["dataset_id"])
    whole = {i["id"]: i for t in engine.geo_window(fine["dataset_id"], "-180,-90,180,90")["themes"] for i in t["insights"]}
    unlocated = [pid for pid in covers if pid in whole and (point_cell[covers[pid]] < 0).any()]
    assert unlocated and {i["lift"] for i in whole.values()} == {1.0}
    partial = {pid for pid in covers if engine.graph().nodes[pid]["props"].get("geo") == "partial"}
    assert partial and not partial & set(whole)  # the graph does not place them: the window does not list them
    graph = engine.graph()
    placed = {e["source"] for e in graph.snapshot["edges"] if e["type"] == "IN_REGION" and e["source"] in covers}
    assert placed and placed <= set(whole)  # every insight the graph places lists on the whole map: one floor for both
    lon = cell_centres(geo["cells"])[:, 0]
    split = float(np.median(lon))
    half = engine.geo_window(fine["dataset_id"], f"-180,-90,{split},90")
    listed = {i["id"]: i for t in half["themes"] for i in t["insights"]}
    shown = set(whole) | {a["id"] for i in whole.values() for a in i["also"]}  # a level folded there stands alone without its host
    assert listed and set(listed) < shown and all(i["lift"] >= 1 for i in listed.values())
    assert all(i["points"] >= GEO_LINK_MIN_EVENTS or i["points"] >= 5 and i["share"] >= 0.05 for i in listed.values())  # link_min_events
    pid = next(p for p in listed if p in covers)  # a base insight's lift from its events alone
    located = point_cell[point_cell >= 0]
    mine = point_cell[covers[pid]]
    mine = mine[mine >= 0]
    x, here = int((lon[mine] <= split).sum()), int((lon[located] <= split).sum())
    expected = len(mine) * here / len(located)
    assert listed[pid]["points"] == x >= link_min_events(len(mine)) and listed[pid]["lift"] == round(
        (x + 30) / (expected + 30), 1
    )  # a = 30; one decimal, as its words read it
    counts = np.bincount(located, minlength=len(geo["cells"]))
    quietest = np.flatnonzero(counts == counts[counts > 0].min())[0]  # one cell's view: fewer events than any insight needs
    w, s = cell_centres(geo["cells"][[quietest]])[0] - 1e-6
    few = engine.geo_window(fine["dataset_id"], f"{w},{s},{w + 2e-6},{s + 2e-6}")
    assert few["cells"] == 1 and 0 < few["points"] < 5 and few["themes"] == []


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


def test_the_window_leads_with_the_levels_of_its_extent(engine, points_csv):
    """Over more than ``WINDOW_DETAIL_REGIONS`` regions the table and coarser levels lead every theme, closer the map's
    own; then by lift; themes follow their leading row, the insights without a theme last."""
    fine, _, _ = upload_levels(engine, points_csv, geo="auto@8,7", table=True)

    def tiers(bbox: str) -> tuple[str, list[list[tuple[bool, float]]]]:
        window = engine.geo_window(fine["dataset_id"], bbox)

        def coarse(i: dict) -> bool:
            return i["level"] in ("table", "res 7")

        rows = [[(coarse(i) != (window["first"] == "coarse"), -i["lift"]) for i in t["insights"]] for t in window["themes"]]
        assert [t["id"] is None for t in window["themes"]] == sorted(t["id"] is None for t in window["themes"])
        named = [r for r, t in zip(rows, window["themes"]) if t["id"] is not None]
        assert all(r == sorted(r) for r in rows) and [r[0] for r in named] == sorted(r[0] for r in named)
        return window["first"], rows

    wide, rows = tiers("-180,-90,180,90")
    assert wide == "coarse" and {k[0] for r in rows for k in r} == {False, True}  # both kinds listed: the order is tested
    lat, lon = HOT
    close, rows = tiers(f"{lon - 0.04},{lat - 0.03},{lon + 0.04},{lat + 0.03}")
    assert close == "map" and rows


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


def test_related_themes_group_by_one_hop_in_the_windows_order():
    """Over eight themes a theme joins the earlier kept theme it has the strongest direct RELATED_TO to (the earlier on a
    tie); one linked only to a member stays on its own, as does one sharing a place (CO_LOCATED) and none; the list keeps
    every theme in order, and eight themes group nothing."""
    from graph_query_engine.graph import DualGraph
    from insight_contracts import SNAPSHOT_VERSION
    from insight_graph_service.core.geo_views import WINDOW_THEMES_UNGROUPED, group_themes

    ids = [f"A-{k}" for k in range(9)]
    links = [("RELATED_TO", "A-1", "A-0", 0.5), ("RELATED_TO", "A-2", "A-1", 0.9), ("RELATED_TO", "A-3", "A-0", 0.4),
             ("RELATED_TO", "A-2", "A-3", 0.6), ("RELATED_TO", "A-4", "A-0", 0.5), ("RELATED_TO", "A-2", "A-4", 0.5),
             ("CO_LOCATED", "A-0", "A-5", 1.0)]  # fmt: skip
    edges = [{"id": f"e{n}", "type": t, "source": a, "target": b, "weight": w, "props": {}} for n, (t, a, b, w) in enumerate(links)]
    graph = DualGraph({"version": SNAPSHOT_VERSION, "nodes": [{"id": a, "kind": "Attractor", "label": a, "props": {}} for a in ids], "edges": edges})

    def window(keys):
        return [{"id": k, "members": []} for k in keys]

    grouped = group_themes(graph, window([*ids, None]))
    assert [t["id"] for t in grouped] == [*ids, None] and {t["id"]: t["members"] for t in grouped if t["members"]} == {
        "A-0": ["A-1", "A-4"],  # A-4 ties A-0 and A-2 at 0.5: the earlier
        "A-2": ["A-3"],  # A-3 links A-0 at 0.4, A-2 at 0.6; A-2 links only A-1, a member: kept
    }
    assert not any(t["members"] for t in group_themes(graph, window(ids[:WINDOW_THEMES_UNGROUPED])))


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
