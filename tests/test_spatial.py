"""The geo option: point rows -> H3 cell table, neighbourhood conditions, n_eff correction (docs/02_discovery.md §2.9)."""

from __future__ import annotations

import json
import time

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import make_config, make_engine

import subgroup_miner.vendor.eda.main_upd as eda
from insight_contracts import Condition, SpatialProvenance
from insight_contracts.insight import SPATIAL_DIGEST_FIELDS
from insight_graph_service.core.engine import PipelineError
from insight_graph_service.core.settings import PROJECT_ROOT
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import Candidate, _median_test, _validation_frame, merge_identical_extents
from subgroup_miner.ingestion import IngestionError, auto_levels, load_dataset
from subgroup_miner.spatial import (
    SpatialError,
    build_cells,
    effective_n_factor,
    locate_points,
    parse_geo,
    parse_points,
    select_levels,
    subgroup_moran,
)

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
    opts = dict(min_cell_points=5) | kw
    return build_cells(frame, "auto", 7, **opts)


def test_parse_geo_and_points():
    assert parse_geo("auto", 7) == ("auto", 7)
    assert parse_geo(" lat,lon @ 8", 7) == ("lat,lon", 8)
    with pytest.raises(ValueError):
        parse_geo("auto@99", 7)
    lat, lon = parse_points(pd.Series(['{"coordinates":[33.77,47.04],"type":"Point"}', "POINT (30.5 50.45)", "nowhere", None]))
    assert lat[:2].tolist() == [47.04, 50.45] and lon[:2].tolist() == [33.77, 30.5]
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


def test_subgroup_moran_sees_a_block_not_scattered_cells():
    table, ctx, _ = cells_of(geo_frame())
    pos = {c: i for i, c in enumerate(ctx.cells)}
    centre = ctx.cells[int(table["points"].idxmax())]
    block = np.array([pos[c] for c in h3.grid_disk(centre, 2) if c in pos])  # a solid patch of busy cells
    assert subgroup_moran(table["points"].to_numpy(float)[block], block, ctx) > 0.0
    scattered = []
    for i in range(len(ctx.cells)):  # no two of them neighbours
        if not np.isin(ctx.neighbours(i), scattered).any():
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
    assert any(p["target"] in ("share_shift_night", "load_median", "when_late_share") for p in hot)  # the planted effects
    cells = json.loads(engine.geo_cells_file(record["dataset_id"]).read_text(encoding="utf-8"))  # written with the batch
    assert len(cells["features"]) == geo["cells"] and {p["id"] for p in cells["patterns"]} == {p["id"] for p in patterns}
    one = engine.geo_pattern(record["dataset_id"], patterns[0]["id"])
    assert len(one["cells"]) == patterns[0]["support"] and 0 < len(one["points"]) <= one["n_points"]
    assert engine.geo_datasets()[0]["dataset_id"] == record["dataset_id"]
    engine.ws.save_batch({**record, "status": "FAILED"})  # a batch that failed after its files were written serves no map
    with pytest.raises(PipelineError):
        engine.geo_cells_file(record["dataset_id"])


def test_geo_evidence_speaks_plain_language(tmp_path):
    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    record = engine.process(engine.submit(path, geo="auto")["batch_id"])
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

    c = h3.latlng_to_cell(48.5, 35.3, 8)
    touching = h3.grid_ring(c, 7)[0]  # two radius-3 disks seven cells apart touch without overlapping
    region = h3.cell_to_parent(c, 5)
    kids = set(h3.cell_to_children(region, 8))
    inside = sorted(k for k in kids if set(h3.grid_disk(k, 2)) <= kids)
    x = inside[0]
    y = max(inside, key=lambda k: h3.grid_distance(x, k))  # far apart in one resolution-5 region
    near, far = h3.grid_disk(x, 2), h3.grid_disk(y, 2)
    coarse = [h3.cell_to_parent(k, 7) for k in near]
    elsewhere = h3.latlng_to_cell(40.0, 30.0, 8)
    crowd = {f"P-q{i}": h3.grid_disk(elsewhere, 2) for i in range(5)}
    shifted = h3.grid_ring(elsewhere, 2)[0]
    lone = h3.latlng_to_cell(45.0, 20.0, 8)
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
    assert spatial.places["P-z"] == {7: frozenset(coarse), **{lv: frozenset(h3.cell_to_parent(k, lv) for k in coarse) for lv in (5, 3, 1)}}
    assert set(derive_spatial([_geo_dataset("bare", near, {})], (7, 5, 3, 1)).occupied["bare"]) == {8, 7, 5, 3, 1}  # no pattern needed
    from subgroup_miner.spatial import region_features

    drawn = region_features(np.array(near + far), [shared, f"R:8:{x}"])  # the console's region outlines, with the cell rows inside
    assert [f["properties"]["rows"] for f in drawn] == [list(range(len(near + far))), [near.index(x)]]
    assert drawn[0]["geometry"]["coordinates"][0][0] == drawn[0]["geometry"]["coordinates"][0][-1]


def test_derive_spatial_compares_the_thresholds_before_rounding(monkeypatch):
    """An overlap or a region share of exactly 2/3 stays below a threshold of 0.66667, although it is stored as 0.6667."""
    import subgroup_miner.spatial as spatial

    monkeypatch.setattr(spatial, "GEO_COLOCATION_MIN", 0.66667)
    monkeypatch.setattr(spatial, "GEO_REGION_MIN_SHARE", 0.66667)
    apart = h3.grid_ring(h3.latlng_to_cell(48.5, 35.3, 8), 10)[::3][:4]  # no two neighbours: the dilation adds nothing
    home = h3.latlng_to_cell(40.0, 30.0, 7)
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


def test_a_geo_dataset_ingested_before_the_spatial_plane_gets_it_at_the_rebuild(tmp_path):
    """The spatial plane is derived from geo.npz and covers.npz: a snapshot of an older version is rebuilt with it,
    no upload again."""
    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    cfg = make_config(tmp_path / "ws")
    engine = make_engine(cfg)
    assert engine.process(engine.submit(path, geo="auto")["batch_id"])["status"] == "READY"
    old = engine.ws.load_graph()
    old["nodes"] = [n for n in old["nodes"] if n["kind"] != "Region"]
    old["edges"] = [e for e in old["edges"] if e["type"] not in ("IN_REGION", "PARENT_OF", "CO_LOCATED")]
    engine.ws.save_graph({**old, "version": 4})
    graph = make_engine(cfg).graph()
    assert graph.of_kind("Region") and {e["props"]["relation"] for e in graph.edges if e["type"] == "CO_LOCATED"} >= {"overlaps"}


def test_one_upload_makes_one_dataset_per_level(tmp_path):
    """``geo=<loc>@<r1>,<r2>`` and ``table`` make ordinary datasets under one ``upload_id``: each geo level with the id
    of its own upload, the table level without its point columns; a level is deleted alone or the whole upload at once."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from insight_graph_service.server.app import create_app

    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    with path.open("rb") as f:
        done = [engine.process(r["batch_id"]) for r in engine.upload("points.csv", f, geo="auto@8,7", bins="points:4", table=True)]
    assert [(r["level"], r["status"]) for r in done] == [("8", "READY"), ("7", "READY"), ("table", "READY")]
    assert len({r["upload_id"] for r in done}) == len({r["source_path"] for r in done}) == 1  # one file, one upload id
    separate = [
        load_dataset(path, engine.settings.miner, geo=g, bins=b, locator=loc).dataset_id
        for g, b, loc in (("auto@8", "points:4", None), ("auto", "points:4", None), (None, "", "auto"))
    ]
    assert [r["dataset_id"] for r in done] == separate  # the table level skips the cell column it lacks
    first = {r["dataset_id"]: next(p["id"] for p in engine.ws.patterns() if p["dataset_id"] == r["dataset_id"]) for r in done}
    assert all(engine.verify_pattern(pid)["status"] == "reproduced" for pid in first.values())  # each level from the shared file
    assert done[2]["dataset_id"] != load_dataset(path, engine.settings.miner, bins="").dataset_id  # not a plain upload
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
        load_dataset(path, engine.settings.miner, geo="auto@8", bins="pointz:4", strict=False)
    graph, table = engine.graph(), done[2]["dataset_id"]
    assert graph.nodes[f"DS:{table}"]["props"]["level"] == "table"
    assert not any(graph.insight(e["source"]).dataset_id == table for e in graph.edges if e["type"] in ("IN_REGION", "CO_LOCATED"))
    upload_id = done[0]["upload_id"]
    with TestClient(create_app(engine)) as client:
        listed = client.get("/api/batches", params={"upload_id": upload_id}).json()
        assert sorted(b["batch_id"] for b in listed) == sorted(r["batch_id"] for r in done)
        with path.open("rb") as f:  # the same upload again: every level SKIPPED, and the 202 names each level's batch
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
            with path.open("rb") as f:
                refused = client.post("/api/upload", files={"file": ("p.csv", f, "text/csv")}, data={"geo": geo, "table": table_level})
            assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
    assert not list(engine.ws.uploads_dir.glob("*"))  # refused before the file is kept
    with path.open("rb") as f:  # geo levels alone apply their options strictly, as one upload does
        assert not any("strict" in r for r in engine.upload("points.csv", f, geo="auto@8,7", bins="points:4"))


def test_the_scale_reads_the_levels_the_upload_has(tmp_path):
    """The coarser levels a scale label compares with are the levels the upload's batches stand for now: one added later
    by an upload of the same file counts, one deleted alone does not."""
    from graph_query_engine.vertical import resolution, scale_chain

    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))

    def upload(geo: str) -> list[dict]:
        with path.open("rb") as f:
            return [engine.process(r["batch_id"]) for r in engine.upload("points.csv", f, geo=geo)]

    def levels_and_scales() -> tuple[list[str], set[str]]:
        graph = engine.graph()
        fine = [p for p in graph.insights if resolution(graph, p) == 8]
        return graph.nodes[f"DS:{fine_ds}"]["props"]["upload_levels"], {c.scale for c in scale_chain(graph, fine)}

    fine_ds = upload("auto@8")[0]["dataset_id"]
    skipped, coarse = upload("auto@8,7")  # the same file: level 8 is SKIPPED, level 7 joins its upload
    assert (skipped["status"], coarse["status"]) == ("SKIPPED", "READY")
    levels, scales = levels_and_scales()
    assert levels == ["8", "7"] and "insufficient" not in scales
    engine.delete_dataset(coarse["dataset_id"])
    assert levels_and_scales() == (["8"], {"insufficient"})  # no coarser level left: nothing was "not found" there


def test_auto_picks_the_levels_from_the_points(tmp_path):
    """``@auto``: the finest resolution with enough cells, enough of them dense, then the coarser region levels; the
    same file the same levels; a denser file a finer level; an upload with ``@auto`` is the upload of its levels."""
    cfg, levels = MinerConfig(), (7, 5, 3, 1)
    california = PROJECT_ROOT / "data" / "demo" / "california_housing.csv"
    picked = auto_levels(california, "auto", cfg, levels)
    assert picked["chosen"] == [5] and picked == auto_levels(california, "auto", cfg, levels)
    rng = np.random.default_rng(0)
    finest = {
        n: select_levels(rng.uniform(48, 49, n), rng.uniform(35, 36, n), levels, min_rows=50, min_cell_points=5)["chosen"][0] for n in (2_000, 50_000)
    }
    assert finest[50_000] > finest[2_000]  # the denser file gets the finer level
    with pytest.raises(SpatialError, match="dense share"):  # many cells, none dense: the refusal names that
        select_levels(rng.uniform(-80, 80, 3_000), rng.uniform(-179, 179, 3_000), levels, min_rows=50, min_cell_points=5)
    rng = np.random.default_rng(71)  # parents of the resolution-10 cells counted 50 cells at 5, the points' own cells 49
    n, size = int(rng.integers(2000, 6000)), float(rng.uniform(0.6, 1.2))
    lat0, lon0 = rng.uniform(-60, 60), rng.uniform(-170, 170)
    lat, lon = lat0 + rng.random(n) * size, lon0 + rng.random(n) * size
    picked = select_levels(lat, lon, levels, min_rows=50, min_cell_points=5)
    for res, candidate in picked["candidates"].items():  # each count is the cell table build_cells makes at that level
        assert candidate["cells"] == len({h3.latlng_to_cell(a, b, int(res)) for a, b in zip(lat, lon)})
    assert 5 not in picked["chosen"] and picked["candidates"]["5"]["cells"] == 49

    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    with path.open("rb") as f:
        record, table = engine.upload("points.csv", f, geo="auto@auto", table=True)
    assert record["profile"]["geo"]["levels"]["chosen"] == [7]  # where the selection stays, whatever the batch's end
    done = engine.process(record["batch_id"])
    assert done["level"] == "7" and done["profile"]["geo"]["levels"]["chosen"] == [7] and "geo_levels" not in done
    assert table["level"] == "table" and not table["profile"]  # the selection is the geo levels' alone
    assert done["dataset_id"] == load_dataset(path, engine.settings.miner, geo="auto@7").dataset_id
    pd.DataFrame({"lat": [48.5] * 60, "lon": [35.3] * 60, "x": range(60)}).to_csv(tiny := tmp_path / "tiny.csv", index=False)
    with tiny.open("rb") as f, pytest.raises(PipelineError) as exc:
        engine.upload("tiny.csv", f, geo="auto@auto")  # one cell at every resolution
    assert exc.value.code == "invalid_options" and len(list(engine.ws.uploads_dir.glob("*"))) == 1  # not kept


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
    fine = "871e4985bffffff"  # its centre lies in another resolution-5 cell than its H3 parent
    holder, parent = h3.latlng_to_cell(*h3.cell_to_latlng(fine), 5), h3.cell_to_parent(fine, 5)
    assert holder != parent
    assert zone_overlap(np.array([holder, parent]), {"holder": [0], "parent": [1]}, frozenset([fine])) == {"holder": 1.0, "parent": 0.0}
    assert zone_overlap(np.array([fine]), {"fine": [0]}, frozenset([holder])) == {"fine": 1.0}


def test_a_zone_from_the_map_is_the_same_question(tmp_path):
    """A zone weighs the seeds of the ordinary search: the question's seed stays first and gets the same scale chain, a
    zone alone seeds from inside it, a polygon equal to the cells seeds the same, ``resolution`` keeps the coarser levels."""
    from fastapi.testclient import TestClient

    from graph_query_engine.vertical import resolution
    from insight_graph_service.server.app import create_app

    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    with path.open("rb") as f:
        assert [engine.process(r["batch_id"])["status"] for r in engine.upload("points.csv", f, geo="auto@8,7")] == ["READY", "READY"]
    question = "Where is load higher?"
    text = engine.search(question)
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
    assert coarse and {resolution(graph, p) for p in coarse} == {7} and {resolution(graph, p) for p in place} == {7, 8}
    area = engine.search(question, cells=zone, resolution=7).evidence
    assert {it.key for it in area.items if resolution(graph, it.pattern_id) == 7} == set(area.area)  # level 8: neither in nor out

    with TestClient(create_app(engine)) as client:
        assert client.post("/api/search", json={"cells": zone}).json()["evidence_prompt"] == alone.payload().evidence_prompt
        refused = client.post("/api/search", json={"cells": ["nope"]})
        assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
        assert client.post("/api/search", json={"cells": zone, "geometry": {"bbox": [0, 0, 1, 1]}}).status_code == 400
        assert client.post("/api/search", json={"question": " "}).status_code == 400


def test_the_graph_exports_as_geosparql(tmp_path):
    """``GET /api/export/geosparql.ttl``: Turtle that rdflib parses and the GeoSPARQL 1.1 validator accepts; a SPARQL
    query for the insights inside a region returns its IN_REGION members; CO_LOCATED keeps its relation; the trust axes
    and the provenance chain are there."""
    import rdflib
    from fastapi.testclient import TestClient
    from pyshacl import validate

    from insight_graph_service.server.app import create_app

    path = tmp_path / "points.csv"
    geo_frame().to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path / "ws"))
    with path.open("rb") as f:
        done = [engine.process(r["batch_id"]) for r in engine.upload("points.csv", f, geo="auto@8,7", source="sensor net", reliability="B")]
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
