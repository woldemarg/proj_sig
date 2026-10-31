"""Cross-type spatial context: a second point type as one forced band of the target's cell table, and the permutation null
of a cross-type cohort (docs/02_discovery.md §2.9; docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import toy_insight, wait_batch
from fastapi.testclient import TestClient

from insight_contracts import Condition, Shift, SpatialProvenance
from insight_contracts.insight import SPATIAL_DIGEST_FIELDS
from insight_contracts.wording import column_words, condition_words, source_column
from insight_graph_service.server.app import create_app
from subgroup_miner.config import MinerConfig
from subgroup_miner.context import NONE_NEARBY, ContextSource, deal, disk_sums
from subgroup_miner.discovery import build_insights, recompute_statistics, run_discovery
from subgroup_miner.ingestion import IngestionError, dataset_fingerprint, load_dataset
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
    table, ctx, warnings = build_cells(frame, "lat,lon", RES, min_cell_points=5, context=[layer_at("schools", [a, a, a, e])])
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


def planted(tmp_path, seed: int = 5):
    """A uniform target and a dozen schools two rings apart or more; in each school's disk (the cell and its six
    neighbours) the target is three times as dense and ``load`` four units higher — the dependence the band reads."""
    rng = np.random.default_rng(seed)
    cells = list(h3.grid_disk(ORIGIN, 9))  # 271 cells
    schools: list[str] = []
    for c in rng.permutation(cells):
        if len(schools) < 12 and h3.grid_distance(c, ORIGIN) <= 7 and all(h3.grid_distance(c, s) > 3 for s in schools):
            schools.append(str(c))
    near = {n for s in schools for n in h3.grid_disk(s, 1)}
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
    target, context = planted(tmp_path)
    layer = [("schools", context, "auto")]
    disk = load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=layer)
    cell = load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=layer, context_ring=0)
    assert disk.dataset_id == load_dataset(target, MinerConfig(), geo=f"lat,lon@{RES}", context=layer, context_ring=1).dataset_id
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
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")])
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
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")])
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
        loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")])
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
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")])
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
    assert not many.kept and [r.reason for r in many.rejections] == ["context_confounded"] and "10 cohorts" in many.rejections[0].detail
    pair = {"pair": ["points", "load_median"], "local_corr": 0.6, "global_corr": -0.1, "delta": 0.7}
    unstable = context_insight(0.01, stability=0.2, emm_score=0.3, covariance=pair)  # no shift; a strong correlation change
    assert not select_insights([unstable], config).kept  # never retargeted to a covariance insight
    plain = context_insight(None, stability=0.2, emm_score=0.3, covariance=pair)
    assert [i.phenomenon_type for i in select_insights([plain], config).kept] == ["covariance"]  # the rule binds only a tested scope


def test_spatial_provenance_reads_records_without_p_context():
    old = {"resolution": 7, "cells": 40, "moran_i": 0.2, "n_eff": 31.5, "p_adjusted_iid": 0.01}
    spatial = SpatialProvenance.from_dict(old)
    assert spatial.p_context is None and spatial.to_dict() == old  # a record written before M17 reads and writes back the same
    assert SPATIAL_DIGEST_FIELDS == ("resolution", "cells", "moran_i", "n_eff", "p_adjusted_iid", "p_context", "context_family")
    tested = old | {"p_context": 0.02, "context_family": 12}
    assert SpatialProvenance.from_dict(tested).to_dict() == tested


def test_context_conditions_read_in_plain_ukrainian_and_hold_no_bare_literal():
    q4, none = Condition(BAND, "q4"), Condition(BAND, NONE_NEARBY)
    assert "schools" in condition_words(q4, {}, True, {}) and "четвертій чверті" in condition_words(q4, {}, True, {})
    assert condition_words(none, {}, True, {}) == "поруч немає schools"
    assert "сусідніх" in column_words(BAND, {}, True)
    assert source_column(BAND, {"schools", "lat"}) is None  # a declared column named like the layer is not its source
    assert " " in NONE_NEARBY  # two words: a bare word would ground onto unrelated question words (question._values)


def post_upload(client, target, *layers, geo=f"lat,lon@{RES}", **data):
    """``POST /api/upload`` of ``target`` with ``layers`` as context files (each ``(filename, path)``)."""
    files = [("file", ("target.csv", target.read_bytes(), "text/csv")), *(("context", (n, p.read_bytes(), "text/csv")) for n, p in layers)]
    return client.post("/api/upload", files=files, data={**({"geo": geo} if geo else {}), **data})


def test_an_upload_with_a_context_file_mines_its_band_and_verifies(engine, tmp_path):
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        accepted = post_upload(client, target, ("Chicago Schools.csv", context))
        assert accepted.status_code == 202, accepted.text
        record = wait_batch(client, accepted.json()["batch_id"], timeout=300)
    assert record["status"] == "READY" and [(c["name"], c["geo"]) for c in record["context"]] == [("chicago_schools", "auto")]
    column = "context_chicago_schools_band"
    scoped = [p for p in engine.ws.patterns() if any(c["attribute"] == column for c in p["conditions"])]
    assert scoped and all("p_context" in p["provenance"]["spatial"] for p in scoped)
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
        sent = [post_upload(client, target, ("schools.csv", context), **ring) for ring in ({}, {"context_ring": "0"})]
        records = [wait_batch(client, r.json()["batch_id"], timeout=300) for r in sent]
    layer = [("schools", Path(records[1]["context"][0]["path"]), "auto")]
    expected = [load_dataset(target, engine.settings.miner, geo=f"lat,lon@{RES}", context=layer, context_ring=r).dataset_id for r in (1, 0)]
    assert [r.get("context_ring") for r in records] == [1, 0]
    # each read on its own disk, as a re-check reads it (the cell alone keeps no insight of this disk-wide plant)
    assert [r["dataset_id"] for r in records] == expected and [r["status"] for r in records] == ["READY", "FAILED"]


def upload_file(client, name: str, path: Path, geo: str, **data) -> dict:
    """``POST /api/upload`` of one file with its geo option, once every batch it made has finished."""
    response = client.post("/api/upload", files=[("file", (name, path.read_bytes(), "text/csv"))], data={"geo": geo, **data})
    assert response.status_code == 202, response.text
    body = response.json()
    for batch in body.get("levels", {}).values():
        wait_batch(client, batch, timeout=300)
    return body


def cross_run(client, target: str, *layers: str, **options):
    """``POST /api/crosstype`` of ``target`` with ``layers``; the response, its batches finished when accepted."""
    response = client.post("/api/crosstype", json={"target": target, "layers": list(layers), **options})
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
            for data in ({"layer": "true"}, {"layer": "true", "geo": "lat,lon", "table": "true"}, {"layer": "true", "geo": "lat,lon", "time": "when"})
        ]
        blank_refused = client.post(
            "/api/upload", files=[("file", ("blank.csv", blank.read_bytes(), "text/csv"))], data={"layer": "true", "geo": "auto"}
        )
    assert [r.status_code for r in [*refused, blank_refused]] == [400] * 4 and all(r.json()["code"] == "invalid_options" for r in refused)
    (record,) = engine.ws.list_batches()
    assert (record["status"], record["points"], record["dataset_id"], record["upload_id"]) == ("LAYER", 12, None, layer["upload_id"])
    assert len(list(engine.ws.uploads_dir.glob("*"))) == 1 and not engine.ws.patterns()  # the blank file not kept; nothing mined
    engine.delete_dataset(record["batch_id"])
    assert not engine.ws.list_batches() and not list(engine.ws.uploads_dir.glob("*"))


def test_two_stored_uploads_cross_at_the_targets_finest_level_and_the_levels_both_were_read_at(engine, tmp_path):
    """A target read at two levels, a layer-only upload and a mined one: a cross-type run reads the target's finest level
    and the levels a mined layer has too, as batches of the target's upload with its options and trust, a copy of each
    layer's stored file as its context, and only the insights on a layer's band (the rest are the target's own)."""
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
            cross_run(client, a, b, c, b),
            cross_run(client, a),
            cross_run(client, a, b, ring="3"),
            cross_run(client, a, b, levels=["9"]),
            cross_run(client, a, b, levels=["x"]),
        ]
        assert [r.status_code for r in refused] == [400] * len(refused), [r.text for r in refused]
        assert all(r.json()["code"] == "invalid_options" for r in refused)
        assert list(cross_run(client, a, b).json()["levels"]) == [str(RES + 1)]  # a layer only: the target's finest level
        assert list(cross_run(client, a, c).json()["levels"]) == [str(RES + 1), str(RES)]  # and the level both were read at
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
    engine.delete_upload(b)  # the layer's own upload gone: the run reads its copy
    assert engine.verify_pattern(patterns[0]["id"])["status"] == "reproduced"


def test_a_shipped_dataset_serves_as_a_context_layer(engine, tmp_path, monkeypatch):
    import insight_graph_service.server.app as app_module

    target, context = planted(tmp_path)
    entry = {"key": "schools", "file": context.name, "context_only": True, "options": {"geo": "lat,lon"}}
    monkeypatch.setattr(app_module, "DEMO_DIR", tmp_path)
    monkeypatch.setattr(app_module, "demo_datasets", lambda: [entry])
    with TestClient(create_app(engine)) as client:
        accepted = post_upload(client, target, context_demo="schools", context_geo="lat,lon")
        record = wait_batch(client, accepted.json()["batch_id"], timeout=300)
        unknown = post_upload(client, target, context_demo="nowhere")
    assert record["status"] == "READY" and [(c["name"], c["geo"]) for c in record["context"]] == [("schools", "lat,lon")]
    assert unknown.status_code == 400


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


@pytest.mark.browser
def test_the_console_keeps_a_layer_and_crosses_it_with_a_stored_upload_at_the_levels_it_ticks(cfg, tmp_path):
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
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("document.querySelectorAll('#cross-target option').length === 1", timeout=20000)
        page.set_input_files("#file", str(context))  # the schools, kept as a layer through the form
        page.evaluate("document.querySelector('#geo').closest('details').open = true")
        page.fill("#geo", "lat,lon")
        page.check("#layer-only")
        page.click("#upload-btn")
        page.wait_for_function("document.querySelectorAll('#cross-layers option').length === 2", timeout=20000)
        (layer,) = [r for r in engine.ws.list_batches() if r["status"] == "LAYER"]
        page.evaluate("document.querySelector('#cross-panel').open = true")
        page.select_option("#cross-target", a)
        page.select_option("#cross-layers", [layer["upload_id"]])
        assert page.is_checked(f"#cross-levels input[value='{RES}']") and page.is_disabled("#cross-both")  # a layer is never a target
        page.click("#cross-btn")
        page.wait_for_function("document.getElementById('batches').textContent.includes('+ schools')", timeout=20000)
        deadline = time.time() + 300
        while not (done := [r for r in engine.ws.list_batches() if r.get("cross") and r["status"] in {"READY", "FAILED"}]):
            assert time.time() < deadline, "the cross-type run did not finish"
            time.sleep(0.5)
    (record,) = done
    assert (record["status"], record["level"], record["upload_id"], record["cross"]) == ("READY", str(RES), a, [layer["upload_id"]])
    assert not errors, errors
