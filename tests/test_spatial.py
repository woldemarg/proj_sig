"""The geo option: point rows -> H3 cell table, neighbourhood conditions, n_eff correction (docs/02_discovery.md §2.9)."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from unittest import mock

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import HOT, geo_frame, make_config, make_engine, upload_levels
from fastapi.testclient import TestClient

import subgroup_miner.vendor.eda.main_upd as eda
from insight_contracts import Condition, SpatialProvenance
from insight_contracts.insight import SPATIAL_DIGEST_FIELDS
from insight_graph_service.core.engine import PipelineError
from insight_graph_service.core.settings import PROJECT_ROOT
from insight_graph_service.server.app import create_app
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import Candidate, _median_test, _validation_frame, merge_identical_extents
from subgroup_miner.ingestion import IngestionError, auto_levels, load_dataset
from subgroup_miner.spatial import (
    SpatialError,
    build_cells,
    cells_inside,
    effective_n_factor,
    footprint,
    locate_points,
    parse_geo,
    parse_points,
    select_levels,
    subgroup_moran,
)


def cells_of(frame: pd.DataFrame, **kw):
    opts = dict(min_cell_points=5) | kw
    return build_cells(frame, "auto", 7, **opts)


def test_parse_geo_and_points():
    assert parse_geo("auto", 7) == ("auto", 7)
    assert parse_geo(" lat,lon @ 8", 7) == ("lat,lon", 8)
    with pytest.raises(ValueError):
        parse_geo("auto@99", 7)
    lat, lon = parse_points(pd.Series(['{"coordinates":[-118.25,34.05],"type":"Point"}', "POINT (-122.42 37.77)", "nowhere", None]))
    assert lat[:2].tolist() == [34.05, 37.77] and lon[:2].tolist() == [-118.25, -122.42]
    assert np.isnan(lat[2:]).all()
    frame = geo_frame()  # columns named longitude then latitude: the names give the roles, in range either way
    swapped, ordered = locate_points(frame, "longitude,latitude"), locate_points(frame, "latitude,longitude")
    assert np.array_equal(swapped[0], frame["latitude"]) and all(np.array_equal(a, b) for a, b in zip(swapped, ordered))


def test_cell_table_roles_follow_dtypes():
    frame = geo_frame()
    table, ctx, warnings = cells_of(frame)
    assert ctx.location == "latitude,longitude" and len(table) == len(ctx.cells)
    assert table["points"].sum() == len(frame)  # every point lands in exactly one cell
    assert (ctx.point_cell >= 0).all()
    assert {"when_active_days", "when_late_share", "load_median", "share_channel_phone", "sector_mode", "lisa_points"} <= set(table.columns)
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


def test_subgroup_moran_sees_a_block_not_scattered_cells():
    table, ctx, _ = cells_of(geo_frame())
    pos = {c: i for i, c in enumerate(ctx.cells)}
    centre = ctx.cells[int(table["points"].idxmax())]
    block = np.array([pos[c] for c in h3.grid_disk(centre, 2) if c in pos])  # a solid patch of busy cells
    assert subgroup_moran(table["points"].to_numpy(float)[block], block, ctx) > 0.0
    scattered = []
    for i in range(len(ctx.cells)):  # no two of them neighbours
        if not np.isin(ctx.nbr_idx[ctx.nbr_ptr[i] : ctx.nbr_ptr[i + 1]], scattered).any():
            scattered.append(i)
    scattered = np.array(scattered)
    assert subgroup_moran(table["points"].to_numpy(float)[scattered], scattered, ctx) == 0.0


def test_closure_strips_the_shifts_a_closed_condition_restates():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        {"sector": ["n"] * 4 + ["s"] * 4, "lisa_points": ["hot spot"] * 4 + ["no cluster"] * 4, "points": rng.random(8), "load_median": rng.random(8)}
    )
    ranks = data[["points", "load_median"]].rank().to_numpy()
    profile = {
        "data_safe": data,
        "categoricals": ["sector", "lisa_points"],
        "numerics": ["points", "load_median"],
        "ranks": ranks,
        "global_corr": eda.rank_correlation(ranks),
    }
    cand = Candidate(
        "sector=='n'", (Condition("sector", "n"),), np.arange(4), 4, 0.5, [("points", 3.0), ("load_median", 1.0)], 4.0, 0.5, emm_reliability=1.0
    )
    (merged,), _ = merge_identical_extents([cand], profile, {"lisa_points": ["points"]})
    assert merged.conditions == (Condition("lisa_points", "hot spot"), Condition("sector", "n"))  # the closure adds the implied condition
    assert merged.restated == {"points"} and merged.top_shifts == [("load_median", 1.0)] and merged.sd_aggregate_score == 1.0
    assert merged.emm_stabilized_score == 0.0  # one metric left: no pair may carry a correlation change


def test_block_bootstrap_skips_draws_without_values():
    data = pd.DataFrame({"load": np.r_[np.full(5, 30.0), np.full(8, np.nan)], "sector": ["n"] * 13})  # one block with values, one without
    cand = Candidate("sector=='n'", (Condition("sector", "n"),), np.arange(13), 13, 0.5, [("load", 2.0)], 2.0, 0.0)
    np.random.seed(0)
    blocks = np.r_[np.zeros(5, dtype=int), np.ones(8, dtype=int)]
    out = eda.step4b_deep_validation(data, _validation_frame([cand]), ["load"], ["sector"], {"load": 10.0}, {"load": 5.0}, blocks)
    assert out["final_sd_score"][0] == pytest.approx(2.0)  # every draw with a value finds the same median: stable, never NaN


def test_validation_never_reports_a_restated_metric_as_a_hidden_shift():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {"points": np.r_[rng.normal(50, 1, 20), rng.normal(10, 1, 80)], "load": rng.normal(0, 1, 100), "sector": ["n"] * 20 + ["s"] * 80}
    )
    medians = {m: float(data[m].median()) for m in ("points", "load")}
    mads = {m: float(eda.calculate_mad(data[m])) for m in ("points", "load")}

    def drivers(restated: frozenset[str]) -> list[str]:
        cand = Candidate("sector=='n'", (Condition("sector", "n"),), np.arange(20), 20, 0.5, [("load", 0.1)], 0.1, 0.0, restated=restated)
        out = eda.step4b_deep_validation(data, _validation_frame([cand]), ["points", "load"], ["sector"], medians, mads, np.arange(100))
        return list(out["root_cause_drivers"][0])

    assert any(d.startswith("[points] hidden shift") for d in drivers(frozenset()))  # a plain cohort: the counter is a hidden shift
    assert not any("points" in d for d in drivers(frozenset({"points"})))  # a cohort whose condition is computed from it: never


def test_geo_option_is_strict_and_changes_the_dataset_id(points_csv):
    plain = load_dataset(points_csv, MinerConfig())
    cells = load_dataset(points_csv, MinerConfig(), geo="auto")
    assert plain.spatial is None and plain.geo is None and len(plain.frame) == 6000
    assert cells.spatial is not None and cells.geo == "auto@7" and cells.dataset_id != plain.dataset_id
    assert "neighbours_points_band" in cells.frame.columns and "neighbours_points" not in cells.frame.columns
    with pytest.raises(IngestionError) as err:
        load_dataset(points_csv, MinerConfig(), geo="nowhere")
    assert err.value.code == "invalid_options"


def test_geo_batch_end_to_end(engine, points_csv):
    record = engine.process(engine.submit(points_csv, geo="auto")["batch_id"])
    assert record["status"] == "READY", record.get("error")
    geo = record["profile"]["geo"]
    assert geo["resolution"] == 7 and geo["sac"]["significant_corrected"] <= geo["sac"]["significant_iid"]
    patterns = [p for p in engine.ws.patterns() if p["dataset_id"] == record["dataset_id"]]
    assert all(set(p["provenance"]["spatial"]) == {"resolution", "cells", "moran_i", "n_eff", "p_adjusted_iid"} for p in patterns)  # ingest only
    graph = engine.graph()
    assert all(0 <= graph.nodes[p["id"]]["props"]["compactness"] <= 1 for p in patterns)  # derived at the commit
    regions = graph.of_kind("Region")
    assert regions and {r["props"]["resolution"] for r in regions} == {7, 5, 3, 1}  # own level included
    assert {e["type"] for e in graph.edges} >= {"IN_REGION", "PARENT_OF"}
    for p in patterns:  # a neighbourhood condition never explains the counters it is computed from
        if any(c["attribute"] in ("lisa_points", "neighbours_points_band") for c in p["conditions"]):
            assert p["target"] not in ("points", "when_active_days")
    hot = [p for p in patterns if {"attribute": "lisa_points", "value": "hot spot"} in p["conditions"] and p["effect_size"] > 0]
    assert any(p["target"] in ("share_channel_phone", "load_median", "when_late_share") for p in hot)  # the planted effects
    cells = json.loads(engine.geo_cells_file(record["dataset_id"]).read_text(encoding="utf-8"))  # written with the batch
    assert len(cells["features"]) == geo["cells"] and {p["id"] for p in cells["patterns"]} == {p["id"] for p in patterns}
    one = engine.geo_pattern(record["dataset_id"], patterns[0]["id"])
    assert len(one["cells"]) == patterns[0]["support"] and 0 < len(one["points"]) <= one["n_points"]
    assert engine.geo_datasets()[0]["dataset_id"] == record["dataset_id"]
    engine.ws.save_batch({**record, "status": "FAILED"})  # a batch that failed after its files were written serves no map
    with pytest.raises(PipelineError):
        engine.geo_cells_file(record["dataset_id"])


def test_geo_evidence_speaks_plain_language(engine, points_csv):
    record = engine.process(engine.submit(points_csv, geo="auto")["batch_id"])
    glossary = record["profile"]["geo"]["glossary"]
    assert glossary["lisa_points=hot spot"].startswith("a hot spot") and "busiest" in glossary["neighbours_points_band=q4"]
    prompt = engine.evidence("Що відрізняє зони hot spot від решти?").evidence_prompt
    assert "PLAIN LANGUAGE" in prompt
    scopes = [line for line in prompt.splitlines() if line.startswith("[P")]
    assert scopes and not any("neighbours_points_band" in line or "lisa_points" in line for line in scopes)


def test_spatial_provenance_is_the_five_ingest_fields():
    stored = {"resolution": 7, "cells": 40, "moran_i": 0.2, "n_eff": 31.5, "p_adjusted_iid": 0.01, "compactness": 0.5, "colocated": []}
    spatial = SpatialProvenance.from_dict(stored)
    assert spatial.to_dict() == {k: stored[k] for k in ("resolution", "cells", "moran_i", "n_eff", "p_adjusted_iid")}  # derived keys stay out
    assert SpatialProvenance.from_dict(spatial.to_dict()) == spatial
    assert SpatialProvenance.from_dict({"resolution": 7, "cells": 40}).moran_i is None  # an older record reads
    assert SpatialProvenance.from_dict(None) is None and SpatialProvenance.from_dict({}) is None
    assert tuple(SpatialProvenance.__dataclass_fields__) == SPATIAL_DIGEST_FIELDS  # widening either is a decision about stored digests


def _geo_dataset(name, cells, covers):
    from subgroup_miner.spatial import GeoDataset, _neighbour_graph

    cells = np.array(sorted(set(cells)), dtype=object)
    ptr, idx, _ = _neighbour_graph(cells)
    pos = {c: i for i, c in enumerate(cells)}
    return GeoDataset(name, cells, ptr, idx, {p: np.array(sorted({pos[c] for c in cs})) for p, cs in covers.items()})


def test_derive_spatial_relations_regions_and_the_partner_cap():
    from subgroup_miner.spatial import derive_spatial

    c = h3.latlng_to_cell(36.5, -119.2, 8)
    touching = h3.grid_ring(c, 7)[0]  # two radius-3 disks seven cells apart touch without overlapping
    region = h3.cell_to_parent(c, 5)
    kids = set(h3.cell_to_children(region, 8))
    inside = sorted(k for k in kids if set(h3.grid_disk(k, 2)) <= kids)
    x = inside[0]
    y = max(inside, key=lambda k: h3.grid_distance(x, k))  # far apart in one resolution-5 region
    near, far = h3.grid_disk(x, 2), h3.grid_disk(y, 2)
    coarse = [h3.cell_to_parent(k, 7) for k in near]
    elsewhere = h3.latlng_to_cell(39.5, -122.5, 8)
    crowd = {f"P-q{i}": h3.grid_disk(elsewhere, 2) for i in range(5)}
    shifted = h3.grid_ring(elsewhere, 2)[0]
    lone = h3.latlng_to_cell(34.0, -116.5, 8)
    apart = h3.grid_ring(lone, 8)[0]  # radius-4 disks eight cells apart share their midpoint cells only
    spatial = derive_spatial(
        [
            _geo_dataset("one", h3.grid_disk(c, 15), {"P-a": h3.grid_disk(c, 3), "P-b": h3.grid_disk(c, 3), "P-c": h3.grid_disk(touching, 3)}),
            _geo_dataset("two", near + far, {"P-x": near, "P-y": far}),
            _geo_dataset("coarse", coarse, {"P-z": coarse}),  # resolution 7
            _geo_dataset("crowd", h3.grid_disk(elsewhere, 4), {"P-p": h3.grid_disk(elsewhere, 2), **crowd}),
            _geo_dataset("other", h3.grid_disk(elsewhere, 6), {f"P-r{i}": h3.grid_disk(shifted, 2) for i in range(4)}),
            _geo_dataset("dot", [lone], {"P-d": [lone]}),
            _geo_dataset("ring", h3.grid_ring(lone, 1), {"P-e": h3.grid_ring(lone, 1)}),
            _geo_dataset("lens", h3.grid_disk(lone, 14), {"P-f": h3.grid_disk(lone, 4), "P-g": h3.grid_disk(apart, 4)}),
        ],
        (7, 5, 3, 1),
    )
    relation = {(a, b): rel for a, b, rel, _ in spatial.co_located}
    assert relation[("P-a", "P-b")] == "overlaps" and relation[("P-a", "P-c")] == "adjacent" and relation[("P-x", "P-y")] == "same_region"
    weight = {(a, b): w for a, b, _, w in spatial.co_located}
    assert relation[("P-d", "P-e")] == "adjacent" and weight[("P-d", "P-e")] == 1.0  # a cell and its ring: the whole edge, not 6
    assert relation[("P-f", "P-g")] == "adjacent"  # shared cells below the overlap threshold still touch
    assert all(0 < w <= 1 for w in weight.values())
    crowded = {"P-p", *crowd}
    own = [pair for pair in relation if set(pair) <= crowded]
    assert len(own) < 15 and all(sum(p in pair for pair in own) >= 3 for p in crowded)  # each keeps three of five
    assert any(pair[0] in crowded and pair[1].startswith("P-r") for pair in relation)  # other datasets are ranked apart
    assert not any("P-z" in pair for pair in relation)  # CO_LOCATED joins one resolution only
    shared = f"R:5:{region}"
    members = {p for p, r, _ in spatial.in_region if r == shared}
    assert {"P-x", "P-y", "P-z"} <= members  # one region for datasets of resolution 8 and 7
    assert spatial.regions[shared]["resolution"] == 5 and (shared, f"R:3:{h3.cell_to_parent(region, 3)}") in spatial.parent_of
    assert all(0 < share <= 1 and isinstance(share, float) for _, _, share in spatial.in_region)
    assert region in spatial.occupied["coarse"][5] and spatial.occupied["one"][8].isdisjoint(spatial.occupied["dot"][8])
    centre = {lv: frozenset(h3.latlng_to_cell(*h3.cell_to_latlng(k), lv) for k in coarse) for lv in (5, 3, 1)}  # where each cell lies
    assert spatial.places["P-z"] == {7: frozenset(coarse), **centre}
    fine = "8729aab49ffffff"  # its centre lies in another resolution-5 cell than its H3 parent
    holder = h3.latlng_to_cell(*h3.cell_to_latlng(fine), 5)
    assert holder != h3.cell_to_parent(fine, 5)
    edge = derive_spatial([_geo_dataset("fine", [fine], {"P-fine": [fine]}), _geo_dataset("coarse", [holder], {"P-holder": [holder]})], (7, 5, 3, 1))
    assert edge.places["P-fine"][5] == edge.places["P-holder"][5] == edge.occupied["fine"][5] == {holder}  # one place for credibility
    assert set(derive_spatial([_geo_dataset("bare", near, {})], (7, 5, 3, 1)).occupied["bare"]) == {8, 7, 5, 3, 1}  # no pattern needed
    from subgroup_miner.spatial import region_features

    drawn = region_features(np.array(near + far), [shared, f"R:8:{x}"])  # the console's region outlines, with the cell rows inside
    by_centre = [i for i, c in enumerate(near + far) if h3.latlng_to_cell(*h3.cell_to_latlng(c), 5) == region]  # as events are placed
    assert [f["properties"]["rows"] for f in drawn] == [by_centre, [near.index(x)]] and len(by_centre) < len(near + far)
    assert drawn[0]["geometry"]["coordinates"][0][0] == drawn[0]["geometry"]["coordinates"][0][-1]


def test_derive_spatial_compares_the_thresholds_before_rounding(monkeypatch):
    """An overlap or a region share of exactly 2/3 stays below a threshold of 0.66667, although it is stored as 0.6667."""
    import subgroup_miner.spatial as spatial

    monkeypatch.setattr(spatial, "GEO_COLOCATION_MIN", 0.66667)
    monkeypatch.setattr(spatial, "GEO_REGION_MIN_SHARE", 0.66667)
    apart = h3.grid_ring(h3.latlng_to_cell(36.5, -119.2, 8), 10)[::3][:4]  # no two neighbours: the dilation adds nothing
    home = h3.latlng_to_cell(39.5, -122.5, 7)
    kids, away = sorted(h3.cell_to_children(home, 8))[:3], h3.cell_to_center_child(h3.grid_ring(home, 2)[0], 8)
    out = spatial.derive_spatial(
        [
            _geo_dataset("pairs", apart, {"P-a": apart[:3], "P-b": [*apart[:2], apart[3]]}),  # |A ∩ B| / min = 2/3
            _geo_dataset("share", [*kids, away], {"P-s": [*kids[:2], away]}),  # two of its three cells in a region of three
        ],
        (7, 5, 3, 1),
    )
    assert {(a, b): rel for a, b, rel, _ in out.co_located}[("P-a", "P-b")] == "adjacent"  # not overlaps
    assert not any(p == "P-s" and r == f"R:7:{home}" for p, r, _ in out.in_region)


def test_a_geo_dataset_ingested_before_the_spatial_plane_gets_it_at_the_rebuild(cfg, points_csv):
    """The spatial plane is derived from geo.npz and covers.npz: a snapshot of an older version is rebuilt with it,
    no upload again."""
    engine = make_engine(cfg)
    assert engine.process(engine.submit(points_csv, geo="auto")["batch_id"])["status"] == "READY"
    old = engine.ws.load_graph()
    old["nodes"] = [n for n in old["nodes"] if n["kind"] != "Region"]
    old["edges"] = [e for e in old["edges"] if e["type"] not in ("IN_REGION", "PARENT_OF", "CO_LOCATED")]
    engine.ws.save_graph({**old, "version": 4})
    graph = make_engine(cfg).graph()
    assert graph.of_kind("Region") and {e["props"]["relation"] for e in graph.edges if e["type"] == "CO_LOCATED"} >= {"overlaps"}


def test_footprint_keeps_the_busiest_cells_up_to_the_share():
    point_cell = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, -1])  # the last point has no cell
    rows, counts = footprint(point_cell, np.arange(11), 0.8)
    assert rows.tolist() == [0, 1] and counts.tolist() == [5, 3]  # 8 of the 10 located points
    assert footprint(point_cell, np.array([10]))[0].tolist() == []


def test_one_upload_makes_one_dataset_per_level(engine, points_csv):
    """``geo=<loc>@<r1>,<r2>`` and ``table`` make ordinary datasets under one ``upload_id``: each geo level with the id
    of its own upload, the table level without its point columns; a level is deleted alone or the whole upload at once."""
    from pathlib import Path

    from graph_query_engine.vertical import picture, scale_chain

    done = upload_levels(engine, points_csv, geo="auto@8,7", bins="points:4", table=True)
    assert [(r["level"], r["status"]) for r in done] == [("8", "READY"), ("7", "READY"), ("table", "READY")]
    assert len({r["upload_id"] for r in done}) == len({r["source_path"] for r in done}) == 1  # one file, one upload id
    separate = [
        load_dataset(points_csv, engine.settings.miner, geo=g, bins=b, locator=loc).dataset_id
        for g, b, loc in (("auto@8", "points:4", None), ("auto", "points:4", None), (None, "", "auto"))
    ]
    assert [r["dataset_id"] for r in done] == separate  # the table level skips the cell column it lacks
    first = {r["dataset_id"]: next(p["id"] for p in engine.ws.patterns() if p["dataset_id"] == r["dataset_id"]) for r in done}
    assert all(engine.verify_pattern(pid)["status"] == "reproduced" for pid in first.values())  # each level from the shared file
    assert done[2]["dataset_id"] != load_dataset(points_csv, engine.settings.miner, bins="").dataset_id  # not a plain upload
    assert not {"latitude", "longitude"} & set(done[2]["profile"]["numerics"])  # its point columns left out
    patterns = engine.graph().of_kind("Pattern")  # a mixed snapshot: every level carries the rule that gave it; one upload, one source
    rule = {n["props"]["dataset_id"]: n["props"]["credibility_basis"]["rule"] for n in patterns}
    assert {n["props"]["credibility"] for n in patterns} == {6} and rule == {
        done[0]["dataset_id"]: "place",
        done[1]["dataset_id"]: "place",
        done[2]["dataset_id"]: "anchor",
    }
    assert any("points" in w for w in done[2]["warnings"]) and any("left out of the table level" in w for w in done[2]["warnings"])
    with pytest.raises(IngestionError, match="pointz"):  # a name in neither the file nor the cells fails the geo levels
        load_dataset(points_csv, engine.settings.miner, geo="auto@8", bins="pointz:4", strict=False)
    graph, table = engine.graph(), done[2]["dataset_id"]
    assert graph.nodes[f"DS:{table}"]["props"]["level"] == "table"
    placed = [e for e in graph.edges if e["type"] == "IN_REGION" and graph.insight(e["source"]).dataset_id == table]
    assert placed and all(e["props"]["basis"] == "events" and e["weight"] == e["props"]["share"] for e in placed)  # by its events
    assert {e["props"]["placed_by"] for e in placed} == {done[0]["dataset_id"]}  # the finest level of its upload
    assert all(e["props"]["basis"] == "cells" for e in graph.edges if e["type"] == "IN_REGION" and e not in placed)
    assert not any(
        table in {graph.insight(e["source"]).dataset_id, graph.insight(e["target"]).dataset_id} for e in graph.edges if e["type"] == "CO_LOCATED"
    )
    geo_ids = [pid for pid in graph.insights if graph.facts(pid).resolution is not None]
    steps = {s.pattern_id for c in scale_chain(graph, geo_ids) for s in [*c.up, *c.down]}
    assert all(graph.facts(e["source"]).resolution is None for e in placed) and not steps & {e["source"] for e in placed}  # never a step
    region = placed[0]["target"]  # a region's members are of both bases, as picture and its patterns count them
    members = [e for e in graph.inc[region] if e["type"] == "IN_REGION" and not graph.tampered(e["source"])]
    assert picture(graph, region)["patterns"] == graph.nodes[region]["props"]["patterns"] == len(members)
    assert {e["props"]["basis"] for e in members} >= {"events"}
    # the map as a window on the graph: every other level's insights on the finest level's cells, a coarser geo one by
    # every cell inside its own, a base one by the footprint of its events
    fine, table_ids = done[0]["dataset_id"], {p["id"] for p in engine.ws.patterns() if p["dataset_id"] == table}
    projected = {p["id"]: p for p in engine.geo_projected(fine)}
    assert table_ids <= set(projected) and {p["level"] for p in projected.values()} == {"res 7", "table"}
    assert all(projected[pid]["rows"] and projected[pid]["spatial"]["footprint_share"] >= 0.8 for pid in table_ids)
    fine_cells, coarse_cells = (engine.ws.load_geo(r["dataset_id"], ("cells",))["cells"] for r in done[:2])
    for pid, rows in engine.ws.load_covers(done[1]["dataset_id"]).items():
        assert projected[pid]["rows"] == np.flatnonzero(cells_inside(fine_cells, coarse_cells[rows].astype(str))).tolist()
    everything = engine.geo_window(fine, "-180,-90,180,90")  # the whole map: every insight at its own share, lift 1
    listed = {i["id"]: i for t in everything["themes"] for i in t["insights"]}
    assert table_ids & set(listed) and {i["lift"] for i in listed.values()} == {1.0}
    assert engine.geo_window(fine, "0,0,1,1")["themes"] == []  # nothing there
    overlay = engine.geo_overlay(fine)
    with mock.patch.object(engine.ws, "load_geo", side_effect=AssertionError("a repeated view reads no file")):  # once per commit
        assert engine.geo_window(fine, "-180,-90,180,90") == everything and engine.geo_overlay(fine) is overlay
    pid = next(iter(table_ids & set(listed)))
    drawn = engine.geo_pattern(fine, pid)  # drawn by its footprint, counted by all its events
    assert drawn["n_points"] == projected[pid]["spatial"]["footprint_points"] < listed[pid]["points"] and set(drawn["rows"]) == set(
        projected[pid]["rows"]
    )
    with pytest.raises(PipelineError, match="west,south"):
        engine.geo_window(fine, "nowhere")
    plain = engine.ingest_file(points_csv)  # the same file as a plain table: as many rows as the map has points, but no upload
    plain_ids = {p["id"] for p in engine.ws.patterns() if p["dataset_id"] == plain["dataset_id"]}
    assert plain["status"] == "READY" and plain_ids and not plain_ids & {p["id"] for p in engine.geo_projected(fine)}
    assert not any(e["source"] in plain_ids for e in engine.graph().edges if e["type"] == "IN_REGION")  # nor is it placed
    engine.delete_dataset(plain["dataset_id"])
    upload_id = done[0]["upload_id"]
    with TestClient(create_app(engine)) as client:
        listed = client.get("/api/batches", params={"upload_id": upload_id}).json()
        assert sorted(b["batch_id"] for b in listed) == sorted(r["batch_id"] for r in done)
        with points_csv.open("rb") as f:  # the same upload again: every level SKIPPED, and the 202 names each level's batch
            data = {"geo": "auto@8,7", "bins": "points:4", "table": True}
            accepted = client.post("/api/upload", files={"file": ("points.csv", f, "text/csv")}, data=data).json()
        assert list(accepted["levels"]) == ["8", "7", "table"] and accepted["batch_id"] == accepted["levels"]["8"]
        for _ in range(600):  # one worker thread: the table level runs last
            if client.get(f"/api/batches/{accepted['levels']['table']}").json()["status"] in ("SKIPPED", "READY", "FAILED"):
                break
            time.sleep(0.1)
        again = client.get("/api/batches", params={"upload_id": upload_id}).json()
        assert {b["batch_id"] for b in again} - {r["batch_id"] for r in done} == set(accepted["levels"].values())
        assert all(b["status"] == "SKIPPED" for b in again if b["batch_id"] in accepted["levels"].values())
        client.delete(f"/api/datasets/{done[1]['dataset_id']}")
        assert engine.ws.ready_batch_for(done[0]["dataset_id"]) and Path(done[0]["source_path"]).is_file()  # the other levels stay
        assert client.delete(f"/api/uploads/{upload_id}").json()["datasets"] == sorted([done[0]["dataset_id"], table])
        assert not engine.ws.list_batches() and not Path(done[0]["source_path"]).exists()
        assert client.delete(f"/api/uploads/{upload_id}").status_code == 404
        for geo, table_level in (("auto@8,6", False), ("auto@7,7", False), ("auto@8,,7", False), ("auto@7,", False), ("", True)):
            with points_csv.open("rb") as f:
                refused = client.post("/api/upload", files={"file": ("p.csv", f, "text/csv")}, data={"geo": geo, "table": table_level})
            assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
    assert not list(engine.ws.uploads_dir.glob("*"))  # refused before the file is kept
    with points_csv.open("rb") as f:  # geo levels alone apply their options strictly, as one upload does
        assert not any("strict" in r for r in engine.upload("points.csv", f, geo="auto@8,7", bins="points:4"))


def test_a_base_insight_is_placed_in_the_answer_the_map_and_its_panel(engine, points_csv):
    """After the commit places a table level (docs/07_question_answering.md §7.2, §7.4): a zone drawn on a base insight's
    footprint seeds it with a place; the ZONES of a question on it name regions it is placed in, and ZONES over geo and
    base insights add each one's weight by its share of its place keys; every region the ZONES or a SCALE CHAIN name is
    on the map; its panel lists its places, and a broad insight shows its 25 strongest regions."""
    from graph_query_engine.vertical import places, rank_zones
    from insight_graph_service.server.views import IN_REGION_SHOWN, node_details

    done = upload_levels(engine, points_csv, geo="auto@8,7", table=True)
    fine, table, graph = done[0]["dataset_id"], done[2]["dataset_id"], engine.graph()
    base = [pid for pid, ins in graph.insights.items() if ins.dataset_id == table]
    pid = max(base, key=lambda p: graph.insight(p).weight)
    cells = engine.ws.load_geo(fine, ("cells",))["cells"]
    footprint_cells = [str(cells[r]) for r in {p["id"]: p for p in engine.geo_projected(fine)}[pid]["rows"]]
    assert engine.place_scores(engine.committed(), cells=footprint_cells)[pid] > 0
    assert any(s.pattern_id == pid and s.matched["place"] > 0 for s in engine.search("", cells=footprint_cells).seeds)
    asked = TestClient(create_app(engine)).post("/api/search", json={"question": "", "cells": footprint_cells}).json()  # as the card asks
    assert "AREA:" in asked["evidence_prompt"] and pid in asked["view"]["highlight"]["seeds"]
    found = engine.search("Where is load median higher?")
    zones = found.evidence.zones
    assert zones and all(set(z["holds"]) <= set(base) for z in zones)
    assert all(z["id"] in {r for r, _ in places(graph, h, z["resolution"])} for z in zones for h in z["holds"])  # placed there
    drawn = {f["properties"]["id"] for f in engine.geo_overlay(fine)["features"]}
    for question in ("Where is load median higher?", "What is on the map?", "Where is load higher?", "Where is the night share higher?"):
        asked = engine.search(question).evidence  # every region the ZONES or a SCALE CHAIN name is on the map
        assert {z["id"] for z in asked.zones} | {r["id"] for c in asked.scale_chains for r in c["regions"]} <= drawn
    placed = {e["source"] for e in graph.edges if e["type"] == "IN_REGION" and e["props"].get("basis") == "events"}
    assert {p for p in engine.place_scores(engine.committed(), cells=footprint_cells) if p in base} <= placed  # an unplaced one has none
    geo = [p for p in graph.insights if graph.facts(p).resolution is not None]
    mixed = rank_zones(graph, geo + base, geo + base)
    assert any(set(z["holds"]) & set(geo) and set(z["holds"]) & set(base) for z in mixed)  # one region, both bases
    for z in mixed:  # each insight's weight times its share of its place keys at the level
        shares = [(graph.insight(h).weight, dict(places(graph, h, z["resolution"]))) for h in z["holds"]]
        assert z["score"] == pytest.approx(sum(w * keys[z["id"]] / sum(keys.values()) for w, keys in shares), abs=1e-3)
    panel = node_details(graph, pid)
    assert panel["scale_chain"] is None and panel["places"] and all({"id", "share", "lift", "n"} <= set(p) for p in panel["places"])
    broad = max(graph.insights, key=lambda p: len(graph.facts(p).regions))
    wide = node_details(graph, broad)
    assert len(wide["neighbors"]["IN_REGION"]) == IN_REGION_SHOWN < wide["neighbor_counts"]["IN_REGION"] == len(graph.facts(broad).regions)
    held = [x["weight"] for x in node_details(graph, zones[0]["id"])["neighbors"]["IN_REGION (in)"]]
    assert held == sorted(held, reverse=True)  # a region's insights by weight: a lift and a cell share do not compare
    finer = make_engine(make_config(points_csv.parent / "ws8", geo_region_levels="8,7,5,3,1"))  # regions at 8 too
    levels = upload_levels(finer, points_csv, geo="auto@8,7", table=True)
    every = {n["id"] for n in finer.graph().of_kind("Region")}
    assert any(r.startswith("R:8:") for r in every)  # regions finer than the map of 7
    with mock.patch("insight_graph_service.core.geo_views.shown_regions", return_value=every):  # were they shown
        drawn = finer.geo_overlay(levels[1]["dataset_id"])["features"]
    assert drawn and all(f["properties"]["resolution"] <= 7 for f in drawn)  # none inside one of its cells


def test_a_table_level_is_placed_by_its_events(engine, points_csv):
    """A table level's IN_REGION come from the finest geo level of its upload and point columns: deleting that level
    places it by the next, deleting both leaves it unplaced, and another pair of coordinates of the same file never
    places it. On a wide map a random subgroup is placed nowhere where the level places by lift,
    and in one region where it places by share (the coarsest level); a pattern with 30 % of its rows unlocated is
    partial and unplaced."""
    from subgroup_miner.spatial import TableDataset, place_table

    done = upload_levels(engine, points_csv, geo="auto@8,7", bins="points:4", table=True)
    table = done[2]["dataset_id"]

    def placed_by() -> set[str]:
        graph = engine.graph()
        return {e["props"]["placed_by"] for e in graph.edges if e["type"] == "IN_REGION" and graph.insight(e["source"]).dataset_id == table}

    assert placed_by() == {done[0]["dataset_id"]}
    engine.delete_dataset(done[0]["dataset_id"])
    assert placed_by() == {done[1]["dataset_id"]}
    engine.delete_dataset(done[1]["dataset_id"])
    assert placed_by() == set()
    trips = geo_frame().assign(dest_lat=lambda f: f["latitude"] + 2.0, dest_lon=lambda f: f["longitude"] - 1.0)  # another pair
    trips.to_csv(trips_csv := points_csv.with_name("trips.csv"), index=False)
    origin = upload_levels(engine, trips_csv, geo="latitude,longitude@7", table=True)
    table = origin[1]["dataset_id"]
    destination = upload_levels(engine, trips_csv, geo="dest_lat,dest_lon@8")  # the same bytes, a finer map of other points
    assert destination[0]["upload_id"] == origin[0]["upload_id"] and placed_by() == {origin[0]["dataset_id"]}
    finer = upload_levels(engine, trips_csv, geo="auto@8")  # the same points by another name, finer: it places the table now
    assert placed_by() == {finer[0]["dataset_id"]}
    engine.delete_dataset(finer[0]["dataset_id"])
    engine.delete_dataset(origin[0]["dataset_id"])
    assert placed_by() == set()  # never by the destinations
    rng, n = np.random.default_rng(0), 40_000
    lat, lon = 34 + rng.random(n) * 4, -122 + rng.random(n) * 6  # 925 regions at 5, 27 at 3, 3 at 1
    cells, point_cell = np.unique([h3.latlng_to_cell(a, b, 6) for a, b in zip(lat, lon, strict=True)], return_inverse=True)
    random = {f"P-{f}": rng.choice(n, int(f * n), replace=False) for f in (0.05, 0.1, 0.2, 0.3, 0.5)}
    edges, partial = place_table(TableDataset("ds-t", "ds-g", cells, point_cell, random), (7, 5, 3, 1))
    placed = Counter((p, rid.split(":")[1]) for p, rid, _, _ in edges)  # never by lift (5 and 3), at 1 by share alone
    assert not partial and {lv for _, lv in placed} <= {"1"} and max(placed.values(), default=0) <= 1
    assert all(props["rule"] == "share" and props["lift"] >= 1 for *_, props in edges)
    blank = point_cell.copy()
    blank[: int(0.3 * n)] = -1
    edges, partial = place_table(
        TableDataset("ds-t", "ds-g", cells, blank, {"P-part": np.arange(n), "P-whole": np.arange(int(0.3 * n), n)}), (7, 5, 3, 1)
    )
    assert partial == {"P-part"} and {p for p, *_ in edges} == {"P-whole"}


def test_the_scale_reads_the_levels_the_upload_has(engine, points_csv):
    """The coarser levels a scale label compares with are the levels the upload's batches stand for now: one added later
    by an upload of the same file counts, one deleted alone does not."""
    from graph_query_engine.vertical import scale_chain

    def upload(geo: str) -> list[dict]:
        return upload_levels(engine, points_csv, geo=geo)

    def levels_and_scales() -> tuple[list[str], set[str]]:
        graph = engine.graph()
        fine = [p for p in graph.insights if graph.facts(p).resolution == 8]
        return graph.nodes[f"DS:{fine_ds}"]["props"]["upload_levels"], {c.scale for c in scale_chain(graph, fine)}

    fine_ds = upload("auto@8")[0]["dataset_id"]
    skipped, coarse = upload("auto@8,7")  # the same file: level 8 is SKIPPED, level 7 joins its upload
    assert (skipped["status"], coarse["status"]) == ("SKIPPED", "READY")
    levels, scales = levels_and_scales()
    assert levels == ["8", "7"] and "insufficient" not in scales
    engine.delete_dataset(coarse["dataset_id"])
    assert levels_and_scales() == (["8"], {"insufficient"})  # no coarser level left: nothing was "not found" there


def _committed_view(engine, *, structural: bool = False) -> tuple[list[str], list[str]]:
    """The graph's nodes and edges as text, each batch id read as its dataset's and the digests and times that follow
    from the batch left out: what two histories of the same uploads must share; ``structural`` leaves out the themes,
    which learn from the order the batches came in."""
    batches = {b["batch_id"]: f"batch-of-{b['dataset_id']}" for b in engine.ws.list_batches() if b["status"] == "READY"}

    def text(item) -> str:
        out = re.sub(
            r'"(record_digest|created_at|committed_at|ingested_at|batch_seq)": ("[^"]*"|\d+)',
            r'"\1": "-"',
            json.dumps(item, sort_keys=True, default=str),
        )
        return re.sub(r"B\d{8}T\d{6}-[0-9a-f]{6}", lambda m: batches.get(m.group(0), "batch-not-ready"), out)

    graph = engine.graph()
    themes = {nid for nid, n in graph.nodes.items() if structural and n["kind"] == "Attractor"}
    nodes = [text([nid, n["kind"], n["props"]]) for nid, n in graph.nodes.items() if nid not in themes]
    return sorted(nodes), sorted(text(e) for e in graph.snapshot["edges"] if not {e["source"], e["target"]} & themes)


def test_the_geo_levels_first_then_the_table_is_the_upload_of_all_levels(tmp_path, points_csv):
    """One upload of every level and the geo levels followed by the same file with ``table``: the second upload's geo
    levels are SKIPPED, its table placed by their events, and the two graphs are one (docs/02_discovery.md §2.9)."""
    together, later = make_engine(make_config(tmp_path / "together")), make_engine(make_config(tmp_path / "later"))
    all_levels = upload_levels(together, points_csv, geo="auto@8,7", bins="points:4", table=True)
    geo_first = upload_levels(later, points_csv, geo="auto@8,7", bins="points:4")
    with_table = upload_levels(later, points_csv, geo="auto@8,7", bins="points:4", table=True)
    assert [(d["level"], d["status"]) for d in with_table] == [("8", "SKIPPED"), ("7", "SKIPPED"), ("table", "READY")]
    assert [d["dataset_id"] for d in all_levels] == [d["dataset_id"] for d in geo_first + with_table[2:]]
    assert any(e["type"] == "IN_REGION" and e["props"]["basis"] == "events" for e in later.graph().snapshot["edges"])
    assert _committed_view(together) == _committed_view(later)
    later.delete_dataset(with_table[2]["dataset_id"])  # a stack whose table level was deleted gets it back the same way
    again = upload_levels(later, points_csv, geo="auto@8,7", bins="points:4", table=True)
    assert [d["status"] for d in again] == ["SKIPPED", "SKIPPED", "READY"]
    assert _committed_view(together, structural=True) == _committed_view(later, structural=True)


def test_a_failed_coarse_level_leaves_the_table_placed_by_the_others(engine, points_csv):
    """A coarse level too sparse to mine fails on its own: the table level after it is READY and placed by the levels
    that are."""
    fine, coarse, table = upload_levels(engine, points_csv, geo="auto@8,5", table=True)
    assert (fine["status"], coarse["status"], table["status"]) == ("READY", "FAILED", "READY")
    assert coarse["error"]["code"] == "invalid_schema"
    graph = engine.graph()
    base = {pid for pid, ins in graph.insights.items() if ins.dataset_id == table["dataset_id"]}
    placed = {e["props"]["placed_by"] for e in graph.snapshot["edges"] if e["type"] == "IN_REGION" and e["source"] in base}
    assert placed == {fine["dataset_id"]}


def test_a_failed_table_level_leaves_the_geo_levels_as_they_are(engine, points_csv, monkeypatch):
    """The table level failing after the geo levels committed: they stay READY with their insights and regions."""
    from insight_graph_service.core.engine import Engine

    discover = Engine._discover

    def failing(self, record, loaded):
        if record["level"] == "table":
            raise PipelineError("no_candidates", "the table level found nothing")
        return discover(self, record, loaded)

    monkeypatch.setattr(Engine, "_discover", failing)
    fine, coarse, table = upload_levels(engine, points_csv, geo="auto@8,7", table=True)
    assert (fine["status"], coarse["status"], table["status"]) == ("READY", "READY", "FAILED")
    graph = engine.graph()
    assert {ins.dataset_id for ins in graph.insights.values()} == {fine["dataset_id"], coarse["dataset_id"]}
    assert any(e["type"] == "IN_REGION" for e in graph.snapshot["edges"])


def test_auto_picks_the_levels_from_the_points(points_csv, engine, tmp_path, monkeypatch):
    """``@auto``: the finest resolution with enough cells, enough of them dense and few enough at its region level (its
    own below the finest region level), then the coarser region levels; the same file the same levels; a denser file
    a finer level; an upload with ``@auto`` is the upload of its levels."""
    cfg, levels = MinerConfig(), (7, 5, 3, 1)
    california = PROJECT_ROOT / "data" / "demo" / "california_housing.csv"
    picked = auto_levels(california, "auto", cfg, levels)
    assert picked["chosen"] == [5] and picked == auto_levels(california, "auto", cfg, levels)
    rng = np.random.default_rng(0)
    finest = {
        n: select_levels(rng.uniform(36, 37, n), rng.uniform(-120, -119.17, n), levels, min_rows=50, min_cell_points=5)["chosen"][0]
        for n in (2_000, 50_000)
    }
    assert finest == {2_000: 6, 50_000: 7}  # the denser file the finer level; 8 has 10,000 cells of its own
    with pytest.raises(SpatialError, match="dense share"):  # many cells, none dense: the refusal names that
        select_levels(rng.uniform(-80, 80, 3_000), rng.uniform(-179, 179, 3_000), levels, min_rows=50, min_cell_points=5)
    big = np.random.default_rng(1)  # a large export: level 7 dense but 13,500 cells, level 5 only ~310
    picked = select_levels(35 + big.random(80_000) * 3, -121 + big.random(80_000) * 2.5, levels, min_rows=50, min_cell_points=5)
    assert picked["chosen"] == [6] and "GEO_AUTO_MAX_CELLS" in picked["reasons"]["7"] and "GEO_AUTO_MIN_CELLS" in picked["reasons"]["5"]
    box = 35 + big.random(80_000) * 3, -121 + big.random(80_000) * 2.5
    assert select_levels(*box, (5, 3, 1), min_rows=50, min_cell_points=5)["chosen"] == [6]  # finer than every region level: its own cells
    with monkeypatch.context() as m:  # a cap of 300: 6 counts the cells of 5, 4 those of 3
        m.setattr("subgroup_miner.spatial.GEO_AUTO_MAX_CELLS", 300)
        capped = select_levels(*box, levels, min_rows=50, min_cell_points=5)
    assert capped["chosen"] == [4] and capped["capped"] == [7, 6, 5] and capped["reasons"]["6"].endswith("cells at 5 > GEO_AUTO_MAX_CELLS")
    wide = select_levels(34 + big.random(40_000) * 4, -122 + big.random(40_000) * 6, levels, min_rows=50, min_cell_points=5)
    assert wide["candidates"]["6"]["cells"] > 5_000 >= wide["candidates"]["6"]["cap_cells"] == wide["candidates"]["5"]["cells"]
    assert wide["chosen"] == [6, 5] and wide["capped"] == []  # the wide box: 6 counts its IN_REGION regions at 5
    world = select_levels(big.uniform(-80, 80, 250_000), big.uniform(-179, 179, 250_000), levels, min_rows=50, min_cell_points=5)
    assert world["chosen"] == [3] and world["candidates"]["3"]["cells"] > 5_000  # every level over the cap: the coarsest that fits
    rng = np.random.default_rng(48)  # parents of the resolution-10 cells counted 50 cells at 5, the points' own cells 49
    n, size = int(rng.integers(2000, 6000)), float(rng.uniform(0.6, 1.2))
    lat0, lon0 = rng.uniform(32.6, 40.7), rng.uniform(-124.3, -115.4)  # inside CALIFORNIA
    lat, lon = lat0 + rng.random(n) * size, lon0 + rng.random(n) * size
    picked = select_levels(lat, lon, levels, min_rows=50, min_cell_points=5)
    for res, candidate in picked["candidates"].items():  # each count is the cell table build_cells makes at that level
        assert candidate["cells"] == len({h3.latlng_to_cell(a, b, int(res)) for a, b in zip(lat, lon)})
    assert 5 not in picked["chosen"] and picked["candidates"]["5"]["cells"] == 49

    with points_csv.open("rb") as f:
        record, table = engine.upload("points.csv", f, geo="auto@auto", table=True)
    assert record["profile"]["geo"]["levels"]["chosen"] == [7]  # where the selection stays, whatever the batch's end
    done = engine.process(record["batch_id"])
    assert done["level"] == "7" and done["profile"]["geo"]["levels"]["chosen"] == [7] and "geo_levels" not in done
    assert table["level"] == "table" and not table["profile"]  # the selection is the geo levels' alone
    assert done["dataset_id"] == load_dataset(points_csv, engine.settings.miner, geo="auto@7").dataset_id
    pd.DataFrame({"lat": [36.5] * 60, "lon": [-119.2] * 60, "x": range(60)}).to_csv(tiny := tmp_path / "tiny.csv", index=False)
    with tiny.open("rb") as f, pytest.raises(PipelineError) as exc:
        engine.upload("tiny.csv", f, geo="auto@auto")  # one cell at every resolution
    assert exc.value.code == "invalid_options" and len(list(engine.ws.uploads_dir.glob("*"))) == 1  # not kept


def test_an_auto_level_with_nothing_names_the_level_the_cap_left_out(engine, tmp_path, monkeypatch):
    """``@auto``'s finest level that finds nothing while finer ones were left out by ``GEO_AUTO_MAX_CELLS`` alone: the
    batch's error names the nearest of them as the explicit option to try."""
    rng = np.random.default_rng(0)
    lat, lon = 36 + rng.random(40_000) * 0.5, -120 + rng.random(40_000) * 0.5
    frame = pd.DataFrame({"latitude": lat, "longitude": lon, "sector": np.where(lat > 36.25, "north", "south"), "load": rng.normal(10, 2, 40_000)})
    frame.to_csv(path := tmp_path / "noise.csv", index=False)
    monkeypatch.setattr("subgroup_miner.spatial.GEO_AUTO_MAX_CELLS", 400)  # 8 (3,200 cells) and 7 (500) over it, 6 under
    with path.open("rb") as f:
        (record,) = engine.upload("noise.csv", f, geo="latitude,longitude@auto")
    assert record["profile"]["geo"]["levels"]["capped"] == [8, 7]
    done = engine.process(record["batch_id"])
    assert done["level"] == "6" and done["error"]["code"] in {"no_candidates", "no_viable_insights"}
    assert done["error"]["message"].endswith("@auto left latitude,longitude@7 out for GEO_AUTO_MAX_CELLS alone, try it")


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
    assert listed[pid]["points"] == x >= link_min_events(len(mine)) and listed[pid]["lift"] == round((x + 30) / (expected + 30), 2)  # a = 30
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


def test_a_geometry_becomes_the_cells_of_a_zone():
    """A polygon equal to a set of cells is that set, as a Feature or a FeatureCollection too, whatever their own ``bbox``
    member says; a circle and a box take the cells whose centres they hold, a shape smaller than a cell the cell under it;
    anything else, a box across 180° and a point out of range too, is refused. A zone cell of another resolution is placed
    by its centre, also where the centre lies outside its H3 parent."""
    from subgroup_miner.spatial import geometry_to_cells, parse_zone, zone_overlap

    centre = h3.latlng_to_cell(*HOT, 7)
    disk = set(h3.grid_disk(centre, 2))
    shape = h3.h3shape_to_geo(h3.cells_to_h3shape(list(disk)))
    feature = {"type": "Feature", "properties": {}, "geometry": shape}
    assert (
        geometry_to_cells(shape, 7)
        == geometry_to_cells(feature, 7)
        == geometry_to_cells({"type": "FeatureCollection", "features": [feature]}, 7)
        == geometry_to_cells(feature | {"bbox": [0, 0, 0.001, 0.001]}, 7)  # GeoJSON's own bbox is metadata, not the shape
        == geometry_to_cells({"type": "FeatureCollection", "bbox": [0, 0, 1, 1], "features": [feature]}, 7)
        == disk
    )
    lat, lon = h3.cell_to_latlng(centre)
    edge = h3.average_hexagon_edge_length(7, "m")
    assert geometry_to_cells({"center": [lon, lat], "radius_m": 2 * edge}, 7) == set(h3.grid_disk(centre, 1))  # neighbours' centres √3 edges away
    assert geometry_to_cells({"center": [lon, lat], "radius_m": 1.0}, 7) == {centre}
    assert geometry_to_cells({"bbox": [lon - 1e-4, lat - 1e-4, lon + 1e-4, lat + 1e-4]}, 7) == {centre}
    for bad in (
        {"type": "Point", "coordinates": [lon, lat]},
        {"center": [lon]},
        {"center": [lon, lat], "radius_m": float("inf")},
        {"bbox": "x"},
        {"bbox": [170, -10, -170, 10]},  # across 180°: two boxes, not the rest of the world
        {"center": [181, lat], "radius_m": 10.0},
        {"center": [lon, 91], "radius_m": 10.0},
        [lon, lat],
    ):
        with pytest.raises(SpatialError):
            geometry_to_cells(bad, 7)
    for bad in ([], ["nope"], [centre, 7]):
        with pytest.raises(SpatialError):
            parse_zone(bad)

    cells = np.array(sorted(h3.grid_disk(centre, 3)))
    inner = np.flatnonzero(np.isin(cells, list(h3.grid_disk(centre, 1))))
    covers = {"A": inner, "B": np.flatnonzero(np.isin(cells, list(h3.grid_ring(centre, 3))))}
    assert zone_overlap(cells, covers, frozenset(h3.grid_disk(centre, 1))) == {"A": 1.0, "B": 0.0}  # A is the zone, B outside it
    assert zone_overlap(cells, covers, frozenset(h3.cell_to_children(centre, 9)[:1]))["A"] == round(1 / 7**0.5, 4)  # a finer cell: its parent
    coarse = h3.latlng_to_cell(*h3.cell_to_latlng(centre), 6)
    below = {str(c) for c in cells if h3.latlng_to_cell(*h3.cell_to_latlng(str(c)), 6) == coarse}  # a coarser cell: the centres it holds
    hit = len(below & set(h3.grid_disk(centre, 1)))
    assert zone_overlap(cells, covers, frozenset([coarse]))["A"] == round(hit / (7 * len(below)) ** 0.5, 4)
    fine = "8729aab49ffffff"  # its centre lies in another resolution-5 cell than its H3 parent
    holder, parent = h3.latlng_to_cell(*h3.cell_to_latlng(fine), 5), h3.cell_to_parent(fine, 5)
    assert holder != parent
    assert zone_overlap(np.array([holder, parent]), {"holder": [0], "parent": [1]}, frozenset([fine])) == {"holder": 1.0, "parent": 0.0}
    assert zone_overlap(np.array([fine]), {"fine": [0]}, frozenset([holder])) == {"fine": 1.0}


def test_a_zone_from_the_map_is_the_same_question(engine, points_csv):
    """A zone weighs the seeds of the ordinary search: the question's seed stays first and gets the same scale chain, a
    zone alone seeds from inside it, a polygon equal to the cells seeds the same, ``resolution`` keeps the coarser levels."""
    assert [r["status"] for r in upload_levels(engine, points_csv, geo="auto@8,7")] == ["READY", "READY"]
    question = "Where is load higher?"
    text = engine.search(question)
    assert {z["resolution"] for z in text.evidence.zones} == {5}  # where geo findings lie, at a share of 0.5: not the 3 of the whole map
    seed = text.seeds[0].pattern_id
    ds = engine.graph().insight(seed).dataset_id
    zone = [str(c) for c in engine.ws.load_geo(ds)["cells"][engine.ws.load_covers(ds)[seed]]]

    zoned = engine.search(question, cells=zone)
    assert zoned.seeds[0].pattern_id == seed and zoned.seeds[0].matched["place"] == 1.0 and "place" not in text.seeds[0].matched
    chains = [[c for c in found.evidence.scale_chains if c["pattern_id"] == seed] for found in (text, zoned)]
    assert chains[0] and chains[0] == chains[1]  # the same SCALE CHAIN as the text question
    assert "AREA:" in zoned.payload().evidence_prompt and "AREA:" not in text.payload().evidence_prompt
    assert text.evidence.zones and not zoned.evidence.zones  # the asked area is the question's place: no ZONES beside it
    unweighed = engine.search(question, cells=zone, resolution=6).evidence  # no geo dataset at 6 or coarser
    assert unweighed.area is None and unweighed.zones == text.evidence.zones and any("weighs no geo dataset" in n for n in unweighed.notes)

    place = engine.place_scores(engine.committed(), cells=zone)
    alone = engine.search("", cells=zone)
    assert alone.seeds and all(place.get(s.pattern_id, 0) > 0 for s in alone.seeds)
    assert alone.payload().evidence_prompt.startswith("QUESTION: (no text")
    polygon = engine.search("", geometry=h3.h3shape_to_geo(h3.cells_to_h3shape(zone)))
    assert [(s.pattern_id, s.score) for s in polygon.seeds] == [(s.pattern_id, s.score) for s in alone.seeds]
    graph = engine.graph()
    coarse = engine.place_scores(engine.committed(), cells=zone, resolution=7)
    assert coarse and {graph.facts(p).resolution for p in coarse} == {7} and {graph.facts(p).resolution for p in place} == {7, 8}
    area = engine.search(question, cells=zone, resolution=7).evidence
    assert {it.key for it in area.items if graph.facts(it.pattern_id).resolution == 7} == set(area.area)  # level 8: neither in nor out

    with TestClient(create_app(engine)) as client:
        assert client.post("/api/search", json={"cells": zone}).json()["evidence_prompt"] == alone.payload().evidence_prompt
        refused = client.post("/api/search", json={"cells": ["nope"]})
        assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
        assert client.post("/api/search", json={"cells": zone, "geometry": {"bbox": [0, 0, 1, 1]}}).status_code == 400
        assert client.post("/api/search", json={"question": " "}).status_code == 400


def test_the_graph_exports_as_geosparql(engine, points_csv):
    """``GET /api/export/geosparql.ttl``: Turtle that rdflib parses and the GeoSPARQL 1.1 validator accepts; a SPARQL
    query for the insights inside a region returns its IN_REGION members; CO_LOCATED keeps its relation; the trust axes
    and the provenance chain are there."""
    import rdflib
    from pyshacl import validate

    done = upload_levels(engine, points_csv, geo="auto@8,7", source="sensor net", reliability="B")
    assert [r["status"] for r in done] == ["READY", "READY"]
    response = TestClient(create_app(engine=engine)).get("/api/export/geosparql.ttl")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/turtle")
    data = rdflib.Graph().parse(data=response.text, format="turtle")
    shapes = rdflib.Graph().parse(PROJECT_ROOT / "tests" / "vendor" / "geosparql_validator.ttl", format="turtle")
    conforms, _, report = validate(data, shacl_graph=shapes)
    assert conforms, report

    graph = engine.graph()
    edges = [e for e in graph.edges if e["type"] == "IN_REGION"]
    region = max({e["target"] for e in edges}, key=lambda r: sum(e["target"] == r for e in edges))
    rows = data.query(f"PREFIX sig: <urn:sig:vocab#> SELECT ?p WHERE {{ ?p a sig:Insight ; sig:inRegion <urn:sig:{region}> }}")
    assert {str(r.p) for r in rows} == {f"urn:sig:{e['source']}" for e in edges if e["target"] == region}
    features = (
        "PREFIX geo: <http://www.opengis.net/ont/geosparql#> SELECT (COUNT(?f) AS ?n) WHERE { ?f a geo:Feature ; geo:hasGeometry/geo:asWKT ?w }"
    )
    assert int(next(iter(data.query(features))).n) == sum(1 for n in graph.of_kind("Pattern")) + sum(1 for _ in graph.of_kind("Region"))
    sig = rdflib.Namespace("urn:sig:vocab#")
    relation = {"overlaps": sig.extentsOverlap, "adjacent": sig.extentsTouch, "same_region": sig.sameRegion}
    co = [e for e in graph.edges if e["type"] == "CO_LOCATED"]
    assert co and all(
        (rdflib.URIRef(f"urn:sig:{e['source']}"), relation[e["props"]["relation"]], rdflib.URIRef(f"urn:sig:{e['target']}")) in data for e in co
    )
    pid = next(n["id"] for n in graph.of_kind("Pattern"))
    insight, dataset = rdflib.URIRef(f"urn:sig:{pid}"), rdflib.URIRef(f"urn:sig:DS:{graph.insight(pid).dataset_id}")
    prov = rdflib.Namespace("http://www.w3.org/ns/prov#")
    assert (insight, prov.wasDerivedFrom, dataset) in data and (dataset, sig.sourceReliability, rdflib.Literal("B")) in data
    assert data.value(insight, sig.credibility).toPython() == graph.nodes[pid]["props"]["credibility"]
    assert data.value(dataset, prov.wasDerivedFrom) == rdflib.URIRef(f"urn:sig:{done[0]['upload_id']}")
    geo = rdflib.Namespace("http://www.opengis.net/ont/geosparql#")
    wkt = str(data.value(data.value(insight, geo.hasGeometry), geo.asWKT))
    body = wkt.removeprefix("MULTIPOLYGON (((").removesuffix(")))")  # polygons split by ")), ((", their rings by "), ("
    rings = [[[tuple(map(float, xy.split()))[::-1] for xy in r.split(", ")] for r in poly.split("), (")] for poly in body.split(")), ((")]
    shape = h3.LatLngMultiPoly(*(h3.LatLngPoly(*poly) for poly in rings))
    ds = graph.insight(pid).dataset_id
    cells = {str(c) for c in engine.ws.load_geo(ds)["cells"][engine.ws.load_covers(ds)[pid]]}
    assert set(h3.h3shape_to_cells(shape, h3.get_resolution(next(iter(cells))))) == cells  # the WKT outline holds exactly its cells
