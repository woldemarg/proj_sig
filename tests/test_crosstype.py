"""Cross-type spatial context: a second point type as one forced band of the target's cell table, and the permutation null
of a cross-type cohort (docs/02_discovery.md §2.9; docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import toy_insight, wait_batch
from fastapi.testclient import TestClient

from graph_query_engine.evidence import Evidence, _datasets
from graph_query_engine.graph import declared_words, insight_reading
from insight_contracts import Condition, Shift, SpatialProvenance
from insight_contracts.graph import dataset_node_id
from insight_contracts.insight import SPATIAL_DIGEST_FIELDS
from insight_contracts.wording import column_words, condition_words, disk_words, short_conditions, source_column, value_words
from insight_graph_service.core import uploads
from insight_graph_service.core.workspace import TERMINAL
from insight_graph_service.server.app import create_app
from subgroup_miner.config import MinerConfig
from subgroup_miner.context import NONE_NEARBY, ContextSource, deal, disk_sums
from subgroup_miner.discovery import build_insights, recompute_statistics, run_discovery
from subgroup_miner.ingestion import IngestionError, dataset_fingerprint, load_dataset, read_context
from subgroup_miner.selection import select_insights
from subgroup_miner.spatial import build_cells

RES = 7
ORIGIN = h3.latlng_to_cell(36.5, -119.25, RES)
BAND = "context_schools_band"


def points_at(cells, per_cell: int, **columns) -> pd.DataFrame:
    """``per_cell`` rows at the centre of each cell, plus constant ``columns``."""
    lat, lon = zip(*[h3.cell_to_latlng(c) for c in cells for _ in range(per_cell)])
    return pd.DataFrame({"lat": lat, "lon": lon, **columns})


def layer_at(name: str, cells) -> ContextSource:
    pts = points_at(cells, 1)
    return ContextSource(name, pts["lat"].to_numpy(float), pts["lon"].to_numpy(float))


def test_the_band_sums_the_disk_including_cells_without_target_points():
    a = ORIGIN
    b = h3.grid_ring(a, 1)[0]
    e = next(c for c in h3.grid_ring(b, 1) if c not in h3.grid_disk(a, 1))  # a neighbour of b, not of a; no target points
    far = h3.grid_ring(a, 5)[0]
    frame = points_at([a, b, far], 10, load=1.0)
    table, ctx, warnings = build_cells(frame, "lat,lon", RES, min_cell_points=5, context=[layer_at("schools", [a, a, a, e])], context_ring=1)
    assert dict(zip(ctx.cells, disk_sums(ctx.context["schools"]))) == {a: 3.0, b: 4.0, far: 0.0}  # e counts for b, though e has no target point
    band = dict(zip(ctx.cells, table[BAND]))
    assert band[far] == NONE_NEARBY and NONE_NEARBY not in (band[a], band[b]) and band[a] != band[b]
    assert BAND in ctx.conditions and ctx.derived_from[BAND] == []  # forced; restates no metric of the target
    assert {BAND, f"{BAND}={NONE_NEARBY}", f"{BAND}=q1"} <= set(ctx.glossary)
    assert [c for c in table.columns if "schools" in c] == [BAND] and not any("schools" in w for w in warnings)  # one column, no lag of it


def test_a_layer_with_nothing_in_any_disk_is_left_out_with_a_warning():
    frame = points_at([ORIGIN, h3.grid_ring(ORIGIN, 1)[0]], 10, load=1.0)
    table, ctx, warnings = build_cells(frame, "lat,lon", RES, min_cell_points=5, context=[layer_at("shops", [h3.grid_ring(ORIGIN, 6)[0]])])
    assert "context_shops_band" not in table.columns and not ctx.context and not [c for c in ctx.conditions if c.startswith("context_")]
    assert any("shops" in w for w in warnings)


def test_a_deal_keeps_every_parents_total():
    cells = list(h3.grid_disk(ORIGIN, 3))
    rng = np.random.default_rng(0)
    _, ctx, _ = build_cells(points_at(cells, 6, load=1.0), "lat,lon", RES, min_cell_points=5, context=[layer_at("x", list(rng.choice(cells, 20)))])
    layer = ctx.context["x"]
    drawn = layer.counts[deal(layer, np.random.default_rng(1))]
    assert drawn.sum() == layer.counts.sum() and not np.array_equal(drawn, layer.counts)
    for parent in np.unique(layer.parents):
        assert drawn[layer.parents == parent].sum() == layer.counts[layer.parents == parent].sum()


def test_the_fingerprint_names_the_layers_whatever_their_order():
    digest = hashlib.sha256(b"rows")
    plain = dataset_fingerprint(digest, {}, geo="lat,lon@7")
    ab = dataset_fingerprint(digest, {}, geo="lat,lon@7", context=["schools:0123456789ab:auto", "shops:ba9876543210:auto"])
    ba = dataset_fingerprint(digest, {}, geo="lat,lon@7", context=["shops:ba9876543210:auto", "schools:0123456789ab:auto"])
    assert plain != ab == ba
    layers = ["schools:0123456789ab:auto", "shops:ba9876543210:auto"]
    ring = dataset_fingerprint(digest, {}, geo="lat,lon@7", context=layers, context_ring=0)
    assert ring != ab == dataset_fingerprint(digest, {}, geo="lat,lon@7", context=layers, context_ring=1)
    assert ab == "ds-22ba51b5b5c8"  # the default disk keeps the id an upload had before the option


def schools_of(rng) -> tuple[list[str], list[str], set[str]]:
    """The 271 cells around ``ORIGIN``, a dozen schools two rings apart or more among them and the schools' disks."""
    cells = list(h3.grid_disk(ORIGIN, 9))
    schools: list[str] = []
    for c in rng.permutation(cells):
        if len(schools) < 12 and h3.grid_distance(c, ORIGIN) <= 7 and all(h3.grid_distance(c, s) > 3 for s in schools):
            schools.append(str(c))
    return cells, schools, {n for s in schools for n in h3.grid_disk(s, 1)}


def planted(tmp_path):
    """A uniform target and a dozen schools two rings apart or more; in each school's disk (the cell and its six
    neighbours) the target is three times as dense and ``load`` four units higher — the dependence the band reads."""
    rng = np.random.default_rng(5)
    cells, schools, near = schools_of(rng)
    parts = []
    for c in cells:
        n, lift = (24, 4.0) if c in near else (8, 0.0)
        lat, lon = h3.cell_to_latlng(c)
        parts.append(pd.DataFrame({"lat": lat + rng.normal(0, 1e-4, n), "lon": lon + rng.normal(0, 1e-4, n), "load": rng.normal(10, 2, n) + lift}))
    frame = pd.concat(parts, ignore_index=True)
    frame["kind"] = rng.choice(["a", "b"], len(frame))
    target, context = tmp_path / "target.csv", tmp_path / "schools.csv"
    frame.to_csv(target, index=False)
    points_at(schools, 1).to_csv(context, index=False)
    return target, context


def mined(loaded, config):
    result = run_discovery(loaded.frame, config, spatial=loaded.spatial)
    return build_insights(result, config, dataset_id="ds", batch_id="b", filename="t.csv"), result


def cross(insights):
    return [i for i in insights if any(c.attribute == BAND and c.value != NONE_NEARBY for c in i.conditions)]


def test_load_dataset_reads_a_context_file_and_refuses_one_without_points(tmp_path):
    target, context = planted(tmp_path)
    plain = load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}")
    loaded = load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=[("schools", context, "auto")])
    assert BAND in loaded.frame.columns and "schools" in loaded.spatial.context
    assert loaded.dataset_id != plain.dataset_id and loaded.frame.drop(columns=BAND).equals(plain.frame)  # the target's table is unchanged
    blank = tmp_path / "blank.csv"
    pd.DataFrame({"name": ["a", "b"], "size": [1, 2]}).to_csv(blank, index=False)
    with pytest.raises(IngestionError, match="context layer 'blank'") as err:
        load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=[("blank", blank, "auto")])
    assert err.value.code == "invalid_options"
    with pytest.raises(IngestionError, match="need the geo option"):
        load_dataset(target, MinerConfig(), context=[("schools", context, "auto")])


def test_the_disk_radius_is_the_uploads_own_and_names_the_dataset(tmp_path):
    """The cell alone by default (``DEFAULT_RING``), the cell and its six neighbours when asked: two readings of the same
    files, two datasets."""
    target, context = planted(tmp_path)
    layer = [("schools", context, "auto")]
    cell = load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=layer)
    disk = load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=layer, context_ring=1)
    assert cell.dataset_id == load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=layer, context_ring=0).dataset_id
    assert cell.dataset_id != disk.dataset_id  # another reading of the same files: another dataset
    schools = {h3.latlng_to_cell(lat, lon, RES) for lat, lon in pd.read_csv(context)[["lat", "lon"]].itertuples(index=False)}
    near = {n for s in schools for n in h3.grid_disk(s, 1)}
    for loaded, counted in ((cell, schools), (disk, near)):  # the cell alone, or the cell and its six neighbours
        assert set(loaded.spatial.cells[loaded.frame[BAND] != NONE_NEARBY]) == counted & set(loaded.spatial.cells)
    for bad in (3, -1, "x"):
        with pytest.raises(IngestionError, match="context_ring") as err:
            load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=layer, context_ring=bad)
        assert err.value.code == "invalid_options"


def test_a_planted_dependence_beats_the_dealt_layer_and_the_recomputation_draws_the_same(tmp_path):
    target, context = planted(tmp_path)
    config = MinerConfig(context_permutations=99)
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], context_ring=1)
    insights, result = mined(loaded, config)
    found = [i for i in cross(insights) if i.target in ("points", "load_median") and i.effect_size > 0]
    family = sum(any(c.attribute == BAND for c in i.conditions) for i in insights)  # the upload's context cohorts
    assert {i.provenance["spatial"]["context_family"] for i in cross(insights)} == {family}
    assert found and min(i.provenance["spatial"]["p_context"] for i in found) * family <= 0.05  # passes the adjusted gate
    lead = min(found, key=lambda i: i.provenance["spatial"]["p_context"])
    fresh = recompute_statistics(
        loaded.frame,
        [(c.attribute, c.value) for c in lead.conditions],
        lead.expression,
        [s.metric for s in lead.shifts],
        loaded.spatial,
        result.n_tests,
        config,
        family,  # as verify reads it from the record
    )
    assert {"p_context", "context_family"} <= set(fresh["spatial"])
    assert fresh["spatial"] == {k: lead.provenance["spatial"][k] for k in fresh["spatial"]}  # drawn again, the same
    others = [i for i in insights if not any(c.attribute == BAND for c in i.conditions)]
    assert others and all("p_context" not in i.provenance["spatial"] for i in others)  # no context condition: no test, no key


def test_without_draws_there_is_no_context_test(tmp_path):
    target, context = planted(tmp_path)
    config = MinerConfig(context_permutations=0)
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], context_ring=1)
    insights, _ = mined(loaded, config)
    assert cross(insights) and all("p_context" not in i.provenance["spatial"] for i in insights)
    assert all(r.reason != "context_confounded" for r in select_insights(insights, config).rejections)


def confounded(tmp_path, seed: int):
    """Both types drawn from one smooth density (a city centre), independently of each other given it; ``load`` rises
    with the density too (as injuries downtown), so "more schools nearby, higher load" holds on the map and means nothing."""
    rng = np.random.default_rng(seed)
    cells = list(h3.grid_disk(ORIGIN, 9))
    density = np.array([0.2 + np.exp(-(h3.grid_distance(c, ORIGIN) ** 2) / 32) for c in cells])
    parts = []
    for c, n, d in zip(cells, rng.poisson(40 * density), density):
        lat, lon = h3.cell_to_latlng(c)
        parts.append(pd.DataFrame({"lat": lat + rng.normal(0, 1e-4, n), "lon": lon + rng.normal(0, 1e-4, n), "load": rng.normal(8 + 4 * d, 2, n)}))
    frame = pd.concat(parts, ignore_index=True)
    frame["kind"] = rng.choice(["a", "b"], len(frame))
    schools = [c for c, n in zip(cells, rng.poisson(60 * density / density.sum())) for _ in range(n)]
    target, context = tmp_path / f"target{seed}.csv", tmp_path / f"schools{seed}.csv"
    frame.to_csv(target, index=False)
    points_at(schools, 1).to_csv(context, index=False)
    return target, context


def test_a_shared_density_passes_the_median_test_and_fails_the_dealt_layer(tmp_path):
    config = MinerConfig(context_permutations=99)
    marginal = tested = 0
    for seed in range(8):
        target, context = confounded(tmp_path, seed)
        loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], context_ring=1)
        higher = [
            i for i in cross(mined(loaded, config)[0]) if i.target == "load_median" and i.effect_size > 0 and i.p_adjusted <= config.max_p_adjusted
        ]
        marginal += bool(higher)
        tested += any(i.provenance["spatial"]["p_context"] <= config.max_p_adjusted for i in higher)
    assert marginal >= 6  # the median test alone reads "more schools nearby, higher load" off the shared centre
    assert tested <= 1  # the layer dealt within its parent cells explains it


def test_the_layers_a_scope_names_are_dealt_together_so_a_layer_given_twice_asks_once(tmp_path):
    """Dealt each on its own, two layers in the same places would part in every draw: the two-layer scope's draws shrink
    and weaken, and a shift the layer's arrangement explains reads as a link (the Chicago 'L' stations and liquor
    stores, spec §14.6). One permutation within the parents for every layer keeps them where they are to each other."""
    target, context = confounded(tmp_path, 1)  # a shared density: draws as strong as the cohort, p between the extremes
    config = MinerConfig(context_permutations=99)
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto"), ("shops", context, "auto")])

    def p_context(conditions):
        fresh = recompute_statistics(loaded.frame, conditions, "the same seed", ["load_median"], loaded.spatial, 1, config, 1)
        return fresh["spatial"]["p_context"]

    for value in sorted(set(loaded.frame[BAND])):
        assert p_context([(BAND, value), ("context_shops_band", value)]) == p_context([(BAND, value)])


def test_a_refused_cohort_stops_drawing_once_no_draw_left_could_pass_the_gate(tmp_path, monkeypatch):
    import subgroup_miner.discovery as discovery

    config = MinerConfig(context_permutations=99)
    target, context = confounded(tmp_path, 0)
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], context_ring=1)
    insights, result = mined(loaded, config)
    weak = max(cross(insights), key=lambda i: i.provenance["spatial"]["p_context"])
    assert weak.provenance["spatial"]["p_context"] >= 0.2  # refused at any family
    deals, band_of = [], discovery.dealt_band
    monkeypatch.setattr(discovery, "dealt_band", lambda layer, perm: deals.append(1) or band_of(layer, perm))

    def drawn(family):
        deals.clear()
        conditions = [(c.attribute, c.value) for c in weak.conditions]
        fresh = recompute_statistics(loaded.frame, conditions, weak.expression, [weak.target], loaded.spatial, result.n_tests, config, family)
        return len(deals), fresh["spatial"]["p_context"]

    planned = 40 * 2 * 20  # forty cohorts to adjust over: the draws at which the least p can still pass
    dealt, p = drawn(40)
    assert min(1.0, p * 40) > config.max_p_adjusted  # refused, as the whole run would refuse it
    assert dealt < planned  # stopped once no draw left could bring it under the gate


def context_insight(p_context, family=1, **fields):
    spatial = {"resolution": 7, "cells": 40, "moran_i": 0.1, "n_eff": 36.0, "p_adjusted_iid": 1e-6, "p_context": p_context, "context_family": family}
    values = dict(id="P-x", expression=f"{BAND}=='q4'", support=300, sd_score=1.8, sd_raw_score=2.0, emm_score=0.01, volume_utility=0.3)
    values |= dict(stability=0.9, p_value=1e-8, p_adjusted=1e-6, row_hash="x", provenance={"spatial": spatial})
    return toy_insight([(BAND, "q4")], [Shift("points", 2.0, 12.0, 8.0, 2.0)], **values | fields)


def test_selection_refuses_a_shift_the_dealt_layer_reproduces_and_any_correlation_change_in_a_context_scope():
    config = MinerConfig()
    refused = select_insights([context_insight(0.3)], config)
    assert not refused.kept and [r.reason for r in refused.rejections] == ["context_confounded"] and "0.3" in refused.rejections[0].detail
    assert select_insights([context_insight(0.01)], config).kept and select_insights([context_insight(None)], config).kept
    many = select_insights([context_insight(0.01, family=10)], config)  # 0.01 × 10 cohorts: adjusted as p_adjusted is
    assert not many.kept and [r.reason for r in many.rejections] == ["context_family"] and "10 cohorts" in many.rejections[0].detail
    pair = {"pair": ["points", "load_median"], "local_corr": 0.6, "global_corr": -0.1, "delta": 0.7}
    unstable = context_insight(0.01, stability=0.2, emm_score=0.3, covariance=pair)  # no shift; a strong correlation change
    assert not select_insights([unstable], config).kept  # never retargeted to a covariance insight
    plain = context_insight(None, stability=0.2, emm_score=0.3, covariance=pair)
    assert [i.phenomenon_type for i in select_insights([plain], config).kept] == ["covariance"]  # the rule binds only a tested scope


def test_spatial_provenance_reads_records_without_p_context():
    old = {"resolution": 7, "cells": 40, "moran_i": 0.2, "n_eff": 31.5, "p_adjusted_iid": 0.01}
    spatial = SpatialProvenance.from_dict(old)
    assert spatial.p_context is None and spatial.to_dict() == old  # a record written before M17 reads and writes back the same
    assert SPATIAL_DIGEST_FIELDS == ("resolution", "cells", "moran_i", "n_eff", "p_adjusted_iid", "p_context", "context_family", "baseline")
    tested = old | {"p_context": 0.02, "context_family": 12}
    assert SpatialProvenance.from_dict(tested).to_dict() == tested


@pytest.mark.parametrize("ring", [0, 1, 2])
def test_context_conditions_read_the_uploads_disk_in_plain_ukrainian_and_hold_no_bare_literal(ring):
    q4, none = Condition(BAND, "q4"), Condition(BAND, NONE_NEARBY)
    disk = disk_words(ring)  # the cell alone, with its six neighbours, or within two rings: never one phrase for all
    assert disk == ["у самій комірці", "у комірці та шести сусідніх", "у комірці та сусідніх у межах двох кілець"][ring]
    phrase = condition_words(q4, {}, True, {}, ring)
    assert "schools" in phrase and disk in phrase and "четвертій чверті" in phrase
    assert condition_words(none, {}, True, {}, ring) == f"немає schools {disk}" and disk in column_words(BAND, {}, True, ring)
    assert source_column(BAND, {"schools", "lat"}) is None  # a declared column named like the layer is not its source
    assert " " in NONE_NEARBY  # two words: a bare word would ground onto unrelated question words (question._values)


def post_upload(client, target, *layers, geo=f"lat,lon@{RES}", **data):
    """``POST /api/upload`` of ``target`` with ``layers`` as context files (each ``(filename, path)``)."""
    files = [("file", ("target.csv", target.read_bytes(), "text/csv")), *(("context", (n, p.read_bytes(), "text/csv")) for n, p in layers)]
    return client.post("/api/upload", files=files, data={**({"geo": geo} if geo else {}), **data})


def test_an_upload_with_a_context_file_mines_its_band_and_verifies(engine, tmp_path):
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        accepted = post_upload(client, target, ("Chicago Schools.csv", context), context_ring="1")  # the plant's disk
        assert accepted.status_code == 202, accepted.text
        record = wait_batch(client, accepted.json()["batch_id"], timeout=300)
        overlay = client.get(f"/api/geo/{record['dataset_id']}/overlay").json()["patterns"]  # what the map card reads
    assert record["status"] == "READY" and [(c["name"], c["geo"]) for c in record["context"]] == [("chicago_schools", "auto")]
    column = "context_chicago_schools_band"
    scoped = [p for p in engine.ws.patterns() if any(c["attribute"] == column for c in p["conditions"])]
    assert scoped and all("p_context" in p["provenance"]["spatial"] for p in scoped)
    # the verdict reads as the adjusted share and its family; an insight without a layer has none
    graph, spatial = engine.graph(), scoped[0]["provenance"]["spatial"]
    verdict = value_words(graph.insight(scoped[0]["id"]))["p_context"]
    assert (
        verdict.startswith(f"{min(1.0, spatial['p_context'] * spatial['context_family']):.2g} (")
        and overlay[scoped[0]["id"]]["numbers"]["p_context"] == verdict
    )
    assert all("p_context" not in value_words(graph.insight(p["id"])) for p in engine.ws.patterns() if p not in scoped)
    # the band's disk is read off the dataset: the record's ring, carried by its node
    props = graph.nodes[dataset_node_id(record["dataset_id"])]["props"]
    assert props["context_ring"] == 1 and disk_words(1) in insight_reading(graph, scoped[0]["id"], engine.settings.thresholds)["where"]
    props["context_ring"] = 0
    assert disk_words(0) in insight_reading(graph, scoped[0]["id"], engine.settings.thresholds)["where"]
    assert any(disk_words(0) in w for w in short_conditions(graph.insight(scoped[0]["id"]), *declared_words(graph, scoped[0]["id"])))  # IN VIEW
    props["context_ring"] = 1
    assert engine.verify_pattern(scoped[0]["id"])["status"] == "reproduced"  # both files read again, p_context drawn the same
    Path(record["context"][0]["path"]).unlink()
    assert engine.verify_pattern(scoped[0]["id"])["status"] == "source_missing"


def test_context_files_are_refused_before_any_batch_and_leave_with_their_upload(engine, tmp_path):
    target, context = planted(tmp_path)
    blank = tmp_path / "blank.csv"
    pd.DataFrame({"name": ["a", "b"], "size": [1, 2]}).to_csv(blank, index=False)
    with TestClient(create_app(engine)) as client:
        refused = [
            post_upload(client, target, *[("s.csv", context)] * 3),
            post_upload(client, target, ("blank.csv", blank)),
            post_upload(client, target, ("s.csv", context), geo=None),
            post_upload(client, target, ("s.csv", context), context_ring="3"),
            post_upload(client, target, context_ring="0"),  # a disk without a layer to count over it
        ]
        assert [r.status_code for r in refused] == [400] * 5 and all(r.json()["code"] == "invalid_options" for r in refused)
        assert not engine.ws.list_batches() and not list(engine.ws.uploads_dir.glob("*"))  # nothing kept
        accepted = post_upload(client, target, ("schools.csv", context), ("schools.csv", context), table="true")
        records = [wait_batch(client, b, timeout=300) for b in accepted.json()["levels"].values()]
    geo, table = (next(r for r in records if r["level"] == level) for level in (str(RES), "table"))
    assert [c["name"] for c in geo["context"]] == ["schools", "schools_2"] and "context" not in table  # a table level has no cells
    kept = [Path(c["path"]) for c in geo["context"]]
    assert all(p.is_file() for p in kept)
    engine.delete_upload(geo["upload_id"])
    assert not any(p.exists() for p in kept)


def test_an_upload_reads_its_layers_on_its_own_disk(engine, tmp_path):
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        sent = [post_upload(client, target, ("schools.csv", context), **ring) for ring in ({"context_ring": "1"}, {})]
        records = [wait_batch(client, r.json()["batch_id"], timeout=300) for r in sent]
    layer = [("schools", Path(records[1]["context"][0]["path"]), "auto")]
    expected = [load_dataset(target, engine.settings.miner, geo=f"lat,lon@{RES}", context=layer, context_ring=r).dataset_id for r in (1, 0)]
    assert [r.get("context_ring") for r in records] == [1, 0]  # the second asked for none: the cell alone, the default
    # each read on its own disk, as a re-check reads it (the cell alone keeps no insight of this disk-wide plant)
    assert [r["dataset_id"] for r in records] == expected and [r["status"] for r in records] == ["READY", "FAILED"]


def upload_file(client, name: str, path: Path, geo: str, **data) -> dict:
    """``POST /api/upload`` of one file with its geo option, once every batch it made has finished; a context file is
    counted over the planted fixture's disk (the cell and its six neighbours) unless the call names one."""
    data = ({"context_ring": "1"} if "context" in data or "context_demo" in data else {}) | data
    response = client.post("/api/upload", files=[("file", (name, path.read_bytes(), "text/csv"))], data={"geo": geo, **data})
    assert response.status_code == 202, response.text
    body = response.json()
    for batch in body.get("levels", {}).values():
        wait_batch(client, batch, timeout=300)
    return body


def cross_run(client, target: str, *layers: str, **options):
    """``POST /api/crosstype`` of ``target`` with ``layers``; the response, its batches finished when accepted. The planted
    target is read at r7, coarser than the product's gate, so the runs say ``allow_coarse`` as the scripts do, and its
    dependence fills a school's disk, so they count the layer over the cell and its six neighbours (``ring`` 1) unless
    the call names a disk."""
    response = client.post("/api/crosstype", json={"target": target, "layers": list(layers), "allow_coarse": True, "ring": 1, **options})
    if response.status_code == 202:
        for batch in response.json()["levels"].values():
            wait_batch(client, batch, timeout=300)
    return response


def test_a_layer_only_upload_is_stored_with_its_points_and_never_mined(engine, tmp_path):
    """Another type's points meant only as a layer (the schools: too few to mine) are stored and located once, one
    ``LAYER`` record of their own upload, no batch run: no failed cards, nothing in the graph."""
    _, context = planted(tmp_path)
    blank = tmp_path / "blank.csv"
    pd.DataFrame({"name": ["a", "b"], "size": [1, 2]}).to_csv(blank, index=False)
    with TestClient(create_app(engine)) as client:
        layer = upload_file(client, "schools.csv", context, "lat,lon", layer="true")
        refused = [
            client.post("/api/upload", files=[("file", ("schools.csv", context.read_bytes(), "text/csv"))], data=data)
            for data in (
                {"layer": "true"},
                {"layer": "true", "geo": "lat,lon", "table": "true"},
                {"layer": "true", "geo": "lat,lon", "grid": "month"},  # a layer may declare its time column; a grid, or a zone without it, not
                {"layer": "true", "geo": "lat,lon", "tz": "Europe/Kyiv"},
            )
        ]
        blank_refused = client.post(
            "/api/upload", files=[("file", ("blank.csv", blank.read_bytes(), "text/csv"))], data={"layer": "true", "geo": "auto"}
        )
    assert [r.status_code for r in [*refused, blank_refused]] == [400] * 5 and all(r.json()["code"] == "invalid_options" for r in refused)
    (record,) = engine.ws.list_batches()
    assert (record["status"], record["points"], record["dataset_id"], record["upload_id"]) == ("LAYER", 12, None, layer["upload_id"])
    assert len(list(engine.ws.uploads_dir.glob("*"))) == 1 and not engine.ws.patterns()  # the blank file not kept; nothing mined
    engine.delete_dataset(record["batch_id"])
    assert not engine.ws.list_batches() and not list(engine.ws.uploads_dir.glob("*"))


def test_two_stored_uploads_cross_at_the_targets_levels_only(engine, tmp_path):
    """A target read at two levels, a layer-only upload and a mined one: a cross-type run reads the target's finest level
    — a mined layer's own levels say nothing about the run, its points are counted on the target's grid — or the levels
    named, as batches of the target's upload with its options and trust, a copy of the layer's stored file as its
    context, and only the insights on the layer's band (the rest are the target's own). One layer a run: two are refused
    (a second layer is a second run)."""
    target, context = planted(tmp_path)
    shops = tmp_path / "shops.csv"
    pd.read_csv(context).head(10).to_csv(shops, index=False)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES + 1},{RES}", source="test source")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        c = upload_file(client, "shops.csv", shops, f"lat,lon@{RES},{RES - 2}")["upload_id"]  # too few points to mine: still levels
        unknown = "up-000000000000"
        refused = [
            cross_run(client, a, unknown),
            cross_run(client, unknown, b),
            cross_run(client, b, a),  # a layer-only upload has no level to read the target's metrics on
            cross_run(client, a, a),
            cross_run(client, a, c, c),
            cross_run(client, a, b, c),  # two layers: a second layer is a second run
            cross_run(client, a, b, c, b),
            cross_run(client, a),
            cross_run(client, a, b, ring="3"),
            cross_run(client, a, b, levels=["9"]),
            cross_run(client, a, b, levels=["x"]),
        ]
        assert [r.status_code for r in refused] == [400] * len(refused), [r.text for r in refused]
        assert all(r.json()["code"] == "invalid_options" for r in refused)
        assert list(cross_run(client, a, b).json()["levels"]) == [str(RES + 1)]  # a layer only: the target's finest level
        assert list(cross_run(client, a, c).json()["levels"]) == [str(RES + 1)]  # a mined layer read at r9 too: still the finest alone
        crossed = cross_run(client, a, b, levels=[str(RES)], ring="1")
    record = engine.ws.load_batch(crossed.json()["batch_id"])
    plain = next(r for r in engine.ws.list_batches() if r["upload_id"] == a and r["level"] == str(RES) and "cross" not in r)
    stored = next(r for r in engine.ws.list_batches() if r["upload_id"] == b)
    assert record["status"] == "READY" and (record["upload_id"], record["level"], record["source"], record["cross"]) == (
        a,
        str(RES),
        "test source",
        [b],
    )
    assert [c["name"] for c in record["context"]] == ["schools"] and record["context"][0]["path"] != stored["source_path"]
    assert record["source_path"] == plain["source_path"] and record["dataset_id"] != plain["dataset_id"]
    patterns = [p for p in engine.ws.patterns() if p["dataset_id"] == record["dataset_id"]]
    assert patterns and all(any(c["attribute"] == BAND for c in p["conditions"]) for p in patterns)  # the target's own stay its own
    rejections = json.loads((engine.ws.dataset_dir(record["dataset_id"]) / "rejections.json").read_text(encoding="utf-8"))
    assert any(r["reason"] == "outside_cross_type" for r in rejections)
    node = engine.graph().nodes[dataset_node_id(record["dataset_id"])]["props"]
    assert (node["cross"], node["cross_layers"]) == ([b], ["schools"])  # the graph names the run with its layer
    (evidenced,) = _datasets(
        [SimpleNamespace(provenance={"dataset_id": record["dataset_id"], "batch_id": record["batch_id"]}, scope=[])], engine.graph()
    )
    assert evidenced["cross_layers"] == ["schools"]  # what the prompt names the run by
    with TestClient(create_app(engine)) as client:
        served = client.get(f"/api/datasets/{record['dataset_id']}/rejections")
        assert served.status_code == 200 and any(r["reason"] == "outside_cross_type" for r in served.json())
        (engine.ws.datasets_dir / "ds.x").mkdir()
        (engine.ws.datasets_dir / "ds.x" / "rejections.json").write_text("[]", encoding="utf-8")  # a file a path would reach
        assert [client.get(f"/api/datasets/{x}/rejections").status_code for x in ("ds-000000000000", "ds.x")] == [404, 404]  # an id, never a path
        assert next(x for x in client.get("/api/batches").json() if x["upload_id"] == b)["layer_of"] == 2  # crossed at its default level and at r7
        engine.delete_upload(b)  # the layer's own upload gone: the run reads its copy, marked stale
        listed = {x["batch_id"]: x for x in client.get("/api/batches").json()}
    assert listed[record["batch_id"]]["stale"] == {"layers": [b], "reason": "deleted"} and "stale" not in listed[plain["batch_id"]]
    assert "stale" not in engine.ws.load_batch(record["batch_id"])  # read off the registry, never written
    assert engine.verify_pattern(patterns[0]["id"])["status"] == "reproduced"


def test_a_layer_upload_may_declare_its_time_and_a_run_records_each_layers_rule(engine, tmp_path):
    """A layer-only upload with a time column keeps its span; a cross-type run writes each layer's time rule on its
    context entry — ``events`` for a layer with a time column, ``static`` without — and warns once per event layer that
    it is read as static over its whole span, naming both spans and the share of the target's span it covers; a target
    without a time column is crossed in space only, without one."""
    target, context = planted(tmp_path)
    frame, untimed = pd.read_csv(target), tmp_path / "untimed.csv"
    frame.to_csv(untimed, index=False)
    frame["when"] = pd.date_range("2025-01-01", "2025-12-31", periods=len(frame)).strftime("%Y-%m-%d")
    frame.to_csv(target, index=False)
    timed, schools = tmp_path / "patrols.csv", pd.read_csv(context)
    schools["when"] = pd.date_range("2025-07-01", "2026-06-30", periods=len(schools)).strftime("%Y-%m-%d")  # half past the target
    schools.to_csv(timed, index=False)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", time="when")["upload_id"]
        b = upload_file(client, "patrols.csv", timed, "lat,lon", layer="true", time="when")["upload_id"]
        c = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        crossed, static = cross_run(client, a, b), cross_run(client, a, c)
        assert crossed.status_code == static.status_code == 202, crossed.text + static.text
        spatial = cross_run(client, upload_file(client, "untimed.csv", untimed, f"lat,lon@{RES}")["upload_id"], b)
    layer = next(r for r in engine.ws.list_batches() if r["upload_id"] == b)
    assert layer["status"] == "LAYER" and layer["time"] == "when" and [s[:10] for s in layer["span"]] == ["2025-07-01", "2026-06-30"]
    assert "span" not in next(r for r in engine.ws.list_batches() if r["upload_id"] == c)
    record, plain = (engine.ws.load_batch(r.json()["batch_id"]) for r in (crossed, static))
    assert [(e["name"], e["time_rule"]) for e in record["context"] + plain["context"]] == [("patrols", "events"), ("schools", "static")]
    assert record["context"][0]["span"] == layer["span"] and "span" not in plain["context"][0]
    assert not [w for w in plain["warnings"] if "static" in w]  # a static layer: no span to compare
    (warning,) = [w for w in record["warnings"] if "static" in w]
    assert not [w for w in engine.ws.load_batch(spatial.json()["batch_id"])["warnings"] if "static" in w]
    assert warning.startswith("patrols:") and "50%" in warning and layer["span"][0] in warning
    assert uploads.overlap_share(["2025-01-01T00:00:00+00:00", "2025-12-31T00:00:00+00:00"], layer["span"]) == pytest.approx(0.5, abs=0.01)
    assert uploads.overlap_share(["2025-03-01T00:00:00+00:00"] * 2, layer["span"]) == 0.0  # a target of one instant: covered or not


def test_align_span_counts_only_the_layers_events_within_the_targets_span(engine, tmp_path):
    """``align=span``: an event layer counts only its events within the target's span (its own reading's time record):
    every school twice, once inside the target's year and once after it, counts each school once; the run is its own
    dataset and ``verify`` reads the same window. A layer without time is counted whole, with a warning; an unknown mode,
    a target without time and no layer with time are refused."""
    target, context = planted(tmp_path)
    frame = pd.read_csv(target)
    frame.to_csv(untimed := tmp_path / "untimed.csv", index=False)
    frame["when"] = pd.date_range("2025-01-01", "2025-12-31", periods=len(frame)).strftime("%Y-%m-%d")
    frame.to_csv(target, index=False)
    schools = pd.read_csv(context)
    pd.concat([schools.assign(when="2025-06-15"), schools.assign(when="2026-06-15")]).to_csv(patrols := tmp_path / "patrols.csv", index=False)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", time="when")["upload_id"]
        b = upload_file(client, "patrols.csv", patrols, "lat,lon", layer="true", time="when")["upload_id"]
        c = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        d = upload_file(client, "untimed.csv", untimed, f"lat,lon@{RES}")["upload_id"]
        refused = [cross_run(client, a, b, align="days"), cross_run(client, d, b, align="span"), cross_run(client, a, c, align="span")]
        assert [r.status_code for r in refused] == [400] * 3 and all(r.json()["code"] == "invalid_options" for r in refused), [
            r.text for r in refused
        ]
        whole, aligned = (engine.ws.load_batch(cross_run(client, a, b, **o).json()["batch_id"]) for o in ({}, {"align": "span"}))
    span = next(r for r in engine.ws.list_batches() if r["upload_id"] == a and "cross" not in r)["temporal"]["span"]
    (entry,) = aligned["context"]
    assert entry["window"] == span and "window" not in whole["context"][0] and aligned["dataset_id"] != whole["dataset_id"]
    counted = read_context("patrols", Path(entry["path"]), "lat,lon", ("when", None, *span))[0]
    assert np.isfinite(counted.lat).sum() == len(schools)  # each school once: its later visit lies past the target's span
    assert np.isfinite(read_context("patrols", Path(entry["path"]), "lat,lon")[0].lat).sum() == 2 * len(schools)
    assert not [w for w in aligned["warnings"] if w.startswith("patrols:")]  # aligned: not read as static
    patterns = [p for p in engine.ws.patterns() if p["dataset_id"] == aligned["dataset_id"]]
    assert aligned["status"] == "READY" and patterns and engine.verify_pattern(patterns[0]["id"])["status"] == "reproduced"


def test_the_prompt_names_a_runs_layers():
    """A cross-type run's dataset line in the prompt names its layers; a dataset without them reads as before."""
    ev = Evidence.__new__(Evidence)  # the dataset lines read only the datasets
    ev.datasets = [
        {"dataset_id": "d", "filename": "crimes.csv", "batch_id": "b", "rows": 9, "cross_layers": ["bars", "stations"]},
        {"dataset_id": "e", "filename": "crimes.csv", "batch_id": "c", "rows": 9},
    ]
    assert ev._dataset_lines()[2:] == ["- d file=crimes.csv layers=bars,stations batch=b rows=9", "- e file=crimes.csv batch=c rows=9"]


def test_cross_state_marks_a_run_whose_layer_is_gone_and_counts_a_layers_runs():
    """Read off the registry, never stored: a run naming an upload no record holds is stale — superseded when a stored
    version names it, else deleted —, and every record of an upload a run reads as a layer counts those runs."""
    run = lambda bid, *layers: {"batch_id": bid, "upload_id": "up-t", "cross": list(layers)}  # noqa: E731
    batches = [
        {"batch_id": "t", "upload_id": "up-t"},
        {"batch_id": "l", "upload_id": "up-l"},
        {"batch_id": "v2", "upload_id": "up-m2", "supersedes": "up-m"},
        run("r1", "up-l"),
        run("r2", "up-l", "up-gone"),
        run("r3", "up-m"),
    ]
    state = uploads.cross_state(batches)
    assert state["r2"]["stale"] == {"layers": ["up-gone"], "reason": "deleted"} and state["r3"]["stale"] == {
        "layers": ["up-m"],
        "reason": "superseded",
    }
    assert "r1" not in state and "t" not in state and state["l"] == {"layer_of": 2}
    assert uploads.cross_state(batches) == state and "stale" not in batches[3]  # a function of the registry: nothing written


def test_cross_runs_inherits_only_the_keys_the_target_has_and_warns_on_a_coarse_level(engine, tmp_path):
    """A run of one layer reads the target's finest level on one disk and keeps the target's per-type counters and
    baseline only where its record holds them; an empty ``count_values`` asks for none. A level coarser than
    ``LINK_MIN_LEVEL`` is refused — the console's gate is the API's — unless ``allow_coarse`` (scripts and eval), then
    named in the run's warnings; the registry offers a link only on an upload's finest mined level not coarser than it
    (``linkable``: the console's *+ контекстний шар*) — READY, or none of its findings admitted alone —, never on a
    run."""
    target, context = planted(tmp_path)
    plain = tmp_path / "plain.csv"
    pd.read_csv(target).iloc[:-1].to_csv(plain, index=False)  # other bytes: another upload
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES + 1},{RES}", count_values="kind=a", baseline="zone")["upload_id"]
        p = upload_file(client, "plain.csv", plain, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        asked = [((a, b), {}), ((a, b), {"levels": [RES], "count_values": ""}), ((p, b), {})]
        kept, coarse, bare = (engine.ws.load_batch(cross_run(client, *pair, **options).json()["batch_id"]) for pair, options in asked)
        refused = cross_run(client, p, b, allow_coarse=False)  # the product's gate: r7 is coarser than r8
        assert refused.status_code == 400 and f"coarser than r{uploads.LINK_MIN_LEVEL}" in refused.json()["detail"], refused.text
    assert ([c["value"] for c in kept["count_values"]], kept["baseline"], kept["level"], kept["context_ring"], kept["cross"]) == (
        ["a"],
        "zone",
        str(RES + 1),
        1,
        [b],
    )
    assert "count_values" not in coarse and coarse["baseline"] == "zone" and "count_values" not in bare and "baseline" not in bare
    words = f"coarser than r{uploads.LINK_MIN_LEVEL}"
    assert [any(words in w for w in r.get("warnings", ())) for r in (kept, coarse, bare)] == [False, True, True]
    level = lambda bid, upload, r, status="READY", **x: {"batch_id": bid, "upload_id": upload, "geo": f"lat,lon@{r}", "status": status, **x}  # noqa: E731
    registry = [
        *(level("a9", "up-a", 9), level("a7", "up-a", 7), level("run", "up-a", 9, cross=["up-l"]), level("k7", "up-k", 7)),
        *(level("q9", "up-q", 9, "FAILED", error={"code": "no_viable_insights"}), level("n9", "up-n", 9, "FAILED", error={"code": "no_candidates"})),
    ]
    assert sorted(b for b, s in uploads.cross_state(registry).items() if s.get("linkable")) == ["a9", "q9"]


def test_a_run_asked_from_a_reading_takes_that_record_never_a_repeats_or_the_levels_latest(engine, tmp_path):
    """The card's action asks from a reading: ``target`` as that reading's dataset id reads its record (its baseline,
    its trust), not the level's latest; asked by the upload id, the level's latest *mined* record is read — a later
    repeat (``SKIPPED``) with other trust donates nothing (the e2e review's F-16)."""
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        first = upload_file(client, "target.csv", target, f"lat,lon@{RES}", source="first reading")
        zoned = upload_file(client, "target.csv", target, f"lat,lon@{RES}", baseline="zone", source="zoned reading")
        a, b = first["upload_id"], upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        first, zoned = (engine.ws.load_batch(r["batch_id"]) for r in (first, zoned))  # processed: their dataset ids
        assert zoned["upload_id"] == a and zoned["dataset_id"] != first["dataset_id"]  # one file, two readings
        repeat = engine.ws.load_batch(upload_file(client, "target.csv", target, f"lat,lon@{RES}", baseline="zone", source="a repeat")["batch_id"])
        assert repeat["status"] == "SKIPPED"
        by_upload = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
        by_reading = engine.ws.load_batch(cross_run(client, first["dataset_id"], b).json()["batch_id"])
        unknown = cross_run(client, "ds-000000000000", b)
    assert (by_upload["baseline"], by_upload["source"]) == ("zone", "zoned reading")  # the level's latest mined reading, not the repeat
    assert "baseline" not in by_reading and by_reading["source"] == "first reading" and by_reading["upload_id"] == a
    assert unknown.status_code == 400


def test_runs_follow_their_target_and_outlive_their_layer(engine, tmp_path):
    """The incremental operations around cross-type runs (the M18 specification §13): a third upload leaves the insights
    already there as they were; deleting one run leaves its target, its layer and the other run whole; a new version of
    a mined layer leaves the run that read it whole and reads it stale (superseded); a new version of the target takes
    its runs with it."""
    target, context = planted(tmp_path)
    frame = pd.read_csv(target)
    frame.assign(load=frame["load"] + 1).to_csv(mined := tmp_path / "mined.csv", index=False)  # other bytes, the same points
    frame.assign(load=frame["load"] + 2).to_csv(mined_v2 := tmp_path / "mined_v2.csv", index=False)
    frame.assign(load=frame["load"] + 3).to_csv(third := tmp_path / "third.csv", index=False)
    frame.assign(load=frame["load"] + 4).to_csv(target_v2 := tmp_path / "target_v2.csv", index=False)
    geo = f"lat,lon@{RES}"
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, geo)["upload_id"]
        layer = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        m = upload_file(client, "mined.csv", mined, geo)["upload_id"]
        r1, r2 = (engine.ws.load_batch(cross_run(client, a, x).json()["batch_id"]) for x in (layer, m))
        own = {p["id"] for p in engine.ws.patterns()}
        upload_file(client, "third.csv", third, geo)
        assert own <= {p["id"] for p in engine.ws.patterns()}  # a third upload adds, takes nothing
        assert client.delete(f"/api/datasets/{r1['dataset_id']}").status_code == 200
        kept = {b["batch_id"] for b in engine.ws.list_batches()}
        assert r1["batch_id"] not in kept and r2["batch_id"] in kept and {b["upload_id"] for b in engine.ws.list_batches()} >= {a, layer, m}
        upload_file(client, "mined_v2.csv", mined_v2, geo, supersedes=m)  # the mined layer's new version
        listed = {x["batch_id"]: x for x in client.get("/api/batches").json()}
        assert listed[r2["batch_id"]]["stale"] == {"layers": [m], "reason": "superseded"}
        upload_file(client, "target_v2.csv", target_v2, geo, supersedes=a)  # the target's new version
    assert not [b for b in engine.ws.list_batches() if b.get("upload_id") == a]  # its runs went with it


def test_a_shipped_dataset_serves_as_a_context_layer(engine, tmp_path, monkeypatch):
    import insight_graph_service.server.app as app_module

    target, context = planted(tmp_path)
    entry = {"key": "schools", "file": context.name, "context_only": True, "options": {"geo": "lat,lon"}}
    monkeypatch.setattr(app_module, "DEMO_DIR", tmp_path)
    monkeypatch.setattr(app_module, "demo_datasets", lambda: [entry])
    with TestClient(create_app(engine)) as client:
        accepted = post_upload(client, target, context_demo="schools", context_geo="lat,lon", context_ring="1")
        record = wait_batch(client, accepted.json()["batch_id"], timeout=300)
        unknown = post_upload(client, target, context_demo="nowhere")
        kept = client.post("/api/upload", data={"demo": "schools", "layer": "true", "geo": "lat,lon"})  # what the console sends
    assert record["status"] == "READY" and [(c["name"], c["geo"]) for c in record["context"]] == [("schools", "lat,lon")]
    assert unknown.status_code == 400 and kept.status_code == 202, kept.text
    (layer,) = [r for r in engine.ws.list_batches() if r["status"] == "LAYER"]
    assert (layer["filename"], layer["points"]) == (context.name, 12)  # a shipped layer kept on its own, no file of the user's


@pytest.mark.browser
def test_the_console_sends_two_shipped_layers_their_locator_and_disk(cfg, tmp_path, monkeypatch):
    sync_api = pytest.importorskip("playwright.sync_api")
    from conftest import make_engine, serve, stack
    from dev import chromium

    import insight_graph_service.server.app as app_module

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    target, context = planted(tmp_path)
    (tmp_path / "shops.csv").write_bytes(context.read_bytes())
    entries = [{"key": k, "file": f"{k}.csv", "title": k, "context_only": True, "options": {"geo": "lat,lon"}} for k in ("schools", "shops")]
    monkeypatch.setattr(app_module, "DEMO_DIR", tmp_path)
    monkeypatch.setattr(app_module, "demo_datasets", lambda: entries)
    engine = make_engine(cfg)
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # the developer mode: the graph and the technical fields
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("document.querySelectorAll('#context-sample option[value]').length >= 2", timeout=20000)
        page.set_input_files("#file", str(target))
        page.evaluate("document.querySelector('#geo').closest('details').open = true")
        page.fill("#geo", f"lat,lon@{RES}")
        page.select_option("#context-sample", ["schools", "shops"])
        page.fill("#context-geo", "lat,lon")
        page.select_option("#context-ring", "0")
        page.click("#upload-btn")
        deadline = time.time() + 300
        while not (done := [b for b in engine.ws.list_batches() if b.get("context") and b["status"] in {"READY", "FAILED"}]):
            assert time.time() < deadline, "the upload did not finish"
            time.sleep(0.5)
    (record,) = done
    assert [(c["name"], c["geo"]) for c in record["context"]] == [("schools", "lat,lon"), ("shops", "lat,lon")] and record["context_ring"] == 0
    assert not errors, errors


def test_a_run_that_admits_nothing_still_counts_its_refusals_by_reason(engine, tmp_path):
    """A layer whose points lie in no target cell's disk leaves one band and is left out, so every finding of the run is
    the target's own (``outside_cross_type``): the run fails as ``no_viable_insights`` and its record counts the refusals
    by reason, as a READY record does."""
    target, context = planted(tmp_path)
    far = tmp_path / "far.csv"
    pd.read_csv(context).assign(lat=lambda f: f["lat"] + 5).to_csv(far, index=False)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "far.csv", far, "lat,lon", layer="true")["upload_id"]
        run = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
    assert (run["status"], run["error"]["code"]) == ("FAILED", "no_viable_insights")
    pruned = run["metrics"]["pruned"]
    assert pruned["outside_cross_type"] > 0 and run["metrics"]["pruned_total"] == sum(pruned.values())


@pytest.mark.browser
def test_the_console_shows_the_layers_verdict_the_reasons_it_refused_and_the_warnings(cfg, tmp_path):
    """A cross-type insight's drawer and map card read the layer's verdict (``p_context`` times its family) and its
    scope the band's disk; an insight without a layer has no such row; the run's card counts its refusals by reason and
    a card shows its record's warnings."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import httpx
    from conftest import make_engine, serve, stack
    from dev import chromium

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    target, context = planted(tmp_path)
    engine = make_engine(cfg)
    port, app = stack(engine)
    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as client, sync_api.sync_playwright() as p:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", indicators="ghost:count")["upload_id"]  # a metric it lacks: a warning
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        run = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
        far = tmp_path / "far.csv"
        pd.read_csv(context).assign(lat=lambda f: f["lat"] + 5).to_csv(far, index=False)
        empty = engine.ws.load_batch(
            cross_run(client, a, upload_file(client, "far.csv", far, "lat,lon", layer="true")["upload_id"]).json()["batch_id"]
        )
        own = next(r for r in engine.ws.list_batches() if r["upload_id"] == a and "cross" not in r)
        crossed = next(x["id"] for x in engine.ws.patterns() if x["dataset_id"] == run["dataset_id"])
        plain = next(x["id"] for x in engine.ws.patterns() if x["dataset_id"] == own["dataset_id"] and "p_context" not in x["provenance"]["spatial"])
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # the developer mode: the graph and the technical fields
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("typeof S !== 'undefined' && S.nodes.length > 0", timeout=60000)
        card = page.locator(f".ds[data-ds='{run['dataset_id']}']")
        refused = card.locator(".term", has_text="без умови шару")
        assert refused.get_attribute("title") and f"без умови шару: {run['metrics']['pruned']['outside_cross_type']}" in card.inner_text()
        warned = page.locator(f".ds[data-ds='{own['dataset_id']}']")  # the target's own level dropped the indicator it lacks
        assert "indicators: no metric named" in warned.inner_text()  # the record's warning on its card
        nothing = page.locator(".ds", has_text="+ far")  # a run that admitted none: its refusals by reason all the same
        assert f"без умови шару: {empty['metrics']['pruned']['outside_cross_type']}" in nothing.inner_text()
        page.evaluate(f"inspect('{crossed}')")
        drawer = page.text_content("#drawer")  # its title (the scope) and its folded evidence too
        assert "нуль шару" in drawer and disk_words(1) in drawer and page.locator("#inspector .term", has_text="нуль шару").get_attribute("title")
        page.evaluate(f"inspect('{plain}')")
        assert "нуль шару" not in page.text_content("#drawer")  # no layer, no verdict
        page.evaluate(f"inspect('{crossed}')")
        assert "target.csv · + schools" in page.text_content("#drawer")  # its provenance names the run with its layer
        assert "шар у 1 прогоні" in page.locator(".ds", has_text="schools.csv · шар").inner_text()  # a layer's runs
        assert card.locator("a[href$='/rejections']").count() == 1  # the refusals one click away
        card.locator(".ds-refused summary").click()  # and the layer's refused cohorts in words on the card
        page.wait_for_function(
            f"!document.querySelector(\".ds[data-ds='{run['dataset_id']}'] .ds-refused .hint\").textContent.startsWith('…')", timeout=20000
        )
        client.delete(f"/api/datasets/{next(r['batch_id'] for r in engine.ws.list_batches() if r['upload_id'] == b)}")
        page.evaluate("refreshBatches()")
        page.wait_for_function(f"document.querySelector(\".ds[data-ds='{run['dataset_id']}']\").textContent.includes('шар видалено')", timeout=20000)
        page.select_option("#dataset-filter", run["dataset_id"])
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function(f"G.ds === '{run['dataset_id']}' && G.map.loaded() && G.data", timeout=60000)
        page.evaluate(f"GeoMap.focusPattern('{crossed}')")
        page.wait_for_function("!$('map-card').hidden", timeout=10000)
        assert "нуль шару" in page.inner_text("#map-card")
        # a run with a layer that admitted none is the layer's verdict on its own too, not only beside a READY level of its upload
        for gone in (own["dataset_id"], run["dataset_id"]):
            client.delete(f"/api/datasets/{gone}")
        page.evaluate("refreshBatches()")
        page.wait_for_function("document.querySelectorAll('#batches .ds').length === 2", timeout=20000)  # the far run and the far layer
        alone = nothing.inner_text()
        assert "Без знахідок" in alone and "Знахідок з умовою на шар немає" in alone and "Помилка" not in alone and "Спробуйте" not in alone
    assert not errors, errors


@pytest.mark.browser
def test_the_console_keeps_a_shipped_layer_with_its_time_and_shows_its_run(cfg, tmp_path, monkeypatch):
    """Every shipped set is listed and its role proposed when picked — events for a set with points, a table without,
    *Лише шар* alone for a set meant only as a layer —; the one kept as a layer goes through the form with its time
    column sent; the form's *Лише рядки* is sent and refused for a value no row holds. A run of the kept layer with a stored upload (``POST /api/crosstype``)
    joins the target's group as ``+ <layer>``; the console holds no control that runs one."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import httpx
    from conftest import make_engine, serve, stack
    from dev import chromium

    import insight_graph_service.server.app as app_module

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    target, context = planted(tmp_path)
    frame = pd.read_csv(target)
    frame["when"] = pd.date_range("2025-01-01", "2025-12-31", periods=len(frame)).strftime("%Y-%m-%d")  # timed: a span
    frame.to_csv(target, index=False)
    pd.read_csv(context).assign(when="2025-06-15").to_csv(tmp_path / "patrols.csv", index=False)  # a layer with a time column
    entries = [
        {"key": "plain", "file": "plain.csv", "title": "plain"},  # no points: never a layer
        {"key": "target", "file": "target.csv", "title": "target", "options": {"geo": f"lat,lon@{RES}"}},
        {"key": "schools", "file": "schools.csv", "title": "schools", "context_only": True, "options": {"geo": "lat,lon"}},
        {"key": "patrols", "file": "patrols.csv", "title": "patrols", "context_only": True, "options": {"geo": "lat,lon", "time": "when"}},
    ]
    monkeypatch.setattr(app_module, "DEMO_DIR", tmp_path)
    monkeypatch.setattr(app_module, "demo_datasets", lambda: entries)
    offered = "[...document.querySelectorAll('#sample option')].map((o) => o.value).filter(Boolean)"
    engine = make_engine(cfg)
    port, app = stack(engine)
    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as client, sync_api.sync_playwright() as p:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", time="when")["upload_id"]
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # the developer mode: the graph and the technical fields
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function(f"{offered}.length === 4", timeout=20000)
        assert page.evaluate(offered) == ["plain", "target", "schools", "patrols"]  # every shipped set, its role proposed
        assert page.locator("#cross-panel, #cross-btn").count() == 0  # runs are queued through the API alone
        page.select_option("#sample", "target")
        assert page.is_checked("#role-events") and page.text_content("#upload-btn") == "Аналізувати"
        page.evaluate("document.querySelector('#where').closest('details').open = true")
        page.fill("#where", "kind=zzz")  # its own reading under a filter: a value no row holds, refused with the values listed
        page.click("#upload-btn")
        page.wait_for_function("document.getElementById('upload-msg').textContent.includes('the values of kind')", timeout=20000)
        page.fill("#where", "")
        page.select_option("#sample", "plain")  # no points: a table, no geo sent
        assert page.is_checked("#role-table") and page.is_disabled("#geo")  # held, not sent
        page.select_option("#sample", "schools")  # meant only as a layer: the other roles closed
        assert page.is_checked("#role-layer") and page.is_disabled("#role-events") and page.is_disabled("#role-table")
        assert page.text_content("#upload-btn") == "Зберегти шар"
        for key in ("schools", "patrols"):  # the shipped schools and the timed patrols, kept as layers through the form
            page.select_option("#sample", key)
            page.click("#upload-btn")
            page.wait_for_function(f"document.getElementById('batches').textContent.includes('{key}.csv · шар')", timeout=20000)
        layer, patrols = sorted((r for r in engine.ws.list_batches() if r["status"] == "LAYER"), key=lambda r: r["filename"] != "schools.csv")
        assert (patrols.get("time"), "span" in patrols) == ("when", True)  # the form sent the layer's time column
        assert page.is_disabled("#role-events") and not page.is_checked("#role-layer") and page.input_value("#sample") == ""  # the form reset
        run = engine.ws.load_batch(cross_run(client, a, layer["upload_id"]).json()["batch_id"])
        page.evaluate("refreshBatches()")
        page.wait_for_function("document.getElementById('batches').textContent.includes('+ schools')", timeout=20000)
    assert (run["status"], run["level"], run["upload_id"], run["cross"]) == ("READY", str(RES), a, [layer["upload_id"]])
    assert not errors, errors


@pytest.mark.browser
def test_the_shipped_demonstration_runs_from_the_console_and_shows_the_nulls_verdict(cfg):
    """The M18 acceptance case (docs/extra_funcs/spatial_analytics_multi_type/cross_dataset_analysis_architecture.md §12)
    from an empty workspace, on the shipped tables: the crimes picked in the console and read with their manifest's
    options, the bars and the 'L' stations kept as layers from the same list, each crossed with the crimes from their
    card's *+ контекстний шар* at its defaults — the crimes' finest level, the cell alone, shares per event. The verdict is
    the one measured on these defaults (the cross-layer experiments of 2026-11-05): the stations' run admits nothing and its
    card counts the cohorts the stations' arrangement reproduces (the fraud share M17–M19 admitted at the old disk was a
    small-cell artefact, gone per event); the bars' admits a finding, whose drawer reads the layer's verdict."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import httpx
    from conftest import make_engine, serve, stack
    from dev import chromium

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    port, app = stack(engine)

    def finished(**match) -> dict:
        deadline = time.time() + 300
        while not (done := [b for b in engine.ws.list_batches() if all(b.get(k) == v for k, v in match.items()) and b["status"] in TERMINAL]):
            assert time.time() < deadline, f"no finished batch with {match}"
            time.sleep(0.5)
        return done[-1]

    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as client, sync_api.sync_playwright() as p:
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # the developer mode: the graph and the technical fields
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("document.querySelector('#sample option[value=chicago_crimes]')", timeout=20000)
        page.select_option("#sample", "chicago_crimes")  # the options it is meant to be read with fill the form
        # the crimes have no number of their own beyond the point: no base level to tick, so the defaults cannot fail;
        # a set that has one gets it back, and a file of one's own frees the box again
        assert page.is_disabled("#table") and not page.is_checked("#table")
        page.select_option("#sample", "chicago_crashes")
        assert page.is_enabled("#table") and page.is_checked("#table")
        page.select_option("#sample", "chicago_crimes")
        assert page.is_disabled("#table") and not page.is_checked("#table")
        page.click("#upload-btn")
        crimes = finished(filename="chicago_crimes.csv")
        assert (crimes["status"], crimes["level"]) == ("READY", "9")
        for key in ("chicago_bars", "chicago_l_stations"):  # kept as layers from the shipped list: the role proposed
            page.select_option("#sample", key)
            assert page.is_checked("#role-layer")
            page.click("#upload-btn")
            page.wait_for_function(f"document.getElementById('batches').textContent.includes('{key}.csv · шар')", timeout=20000)
        layers = {b["filename"]: b["upload_id"] for b in engine.ws.list_batches() if b["status"] == "LAYER"}
        runs = {}
        for name in ("chicago_bars.csv", "chicago_l_stations.csv"):  # from the crimes' card: their finest level, the default disk
            page.click(".ds-link")
            page.select_option("#link-layer", layers[name])
            page.click("#link-go")
            runs[name] = finished(cross=[layers[name]])
            assert (runs[name]["level"], runs[name]["context_ring"]) == ("9", 0)  # the cell alone: the default disk
        page.evaluate("refreshBatches()")
        bars, stations = runs["chicago_bars.csv"], runs["chicago_l_stations.csv"]
        # the stations: nothing admitted, the cohorts their arrangement within parent cells reproduces counted on the card
        assert (stations["status"], stations["error"]["code"]) == ("FAILED", "no_viable_insights")
        assert stations["metrics"]["pruned"]["context_confounded"] > 0
        page.wait_for_function("document.getElementById('batches').textContent.includes('+ chicago_l_stations')", timeout=20000)
        card = page.locator(".ds", has_text="+ chicago_l_stations").inner_text()
        assert f"пояснюється фоном зони: {stations['metrics']['pruned']['context_confounded']}" in card
        # the bars: a finding, its drawer reading the layer's verdict
        found = [x["id"] for x in engine.ws.patterns() if x["dataset_id"] == bars["dataset_id"]]
        assert bars["status"] == "READY" and found
        page.evaluate(f"inspect('{found[0]}')")
        page.wait_for_selector("#drawer .link-card", timeout=10000)
        assert "нуль шару" in page.text_content("#drawer")
        # the literature's frequency question as the walkthrough asks it (docs 10 §10.2): one class of the crimes as a
        # reading of its own, crossed with the bars over the cell alone; the verdict is the data's, never an error
        page.select_option("#sample", "chicago_crimes")
        page.evaluate("document.querySelector('#where').closest('details').open = true")
        page.fill("#where", "crime=robbery")
        page.click("#upload-btn")
        robbery = finished(filename="chicago_crimes.csv", where="crime=robbery")
        page.wait_for_function("document.getElementById('batches').textContent.includes('· crime=robbery')", timeout=20000)
        cross_run(client, robbery["upload_id"], layers["chicago_bars.csv"], ring=0)
        frequency = finished(cross=[layers["chicago_bars.csv"]], upload_id=robbery["upload_id"])
        page.evaluate("refreshBatches()")
        page.wait_for_function("document.getElementById('batches').textContent.includes('crime=robbery · + chicago_bars')", timeout=20000)
        card = page.locator(".ds", has_text="crime=robbery · + chicago_bars").inner_text()
        assert frequency["status"] in TERMINAL and "Помилка" not in card and "Спробуйте" not in card
    assert not errors, errors
