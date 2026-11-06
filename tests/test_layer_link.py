"""The layer link's cell metrics (docs/02_discovery.md §2.9; the concept's §4 and §9 in
docs/extra_funcs/spatial_analytics_multi_type/cross_layer_analysis_concept.md): the per-type counters asked for (*більше
подій*), the shares confirmed per event (*інший склад*), the zone baseline — what a run of a base layer with a context layer
reads, and the layer's null on them."""

from __future__ import annotations

import math
import re
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import geo_frame, make_config, make_engine
from fastapi.testclient import TestClient
from test_crosstype import BAND, ORIGIN, RES, cross, cross_run, mined, planted, points_at, schools_of, upload_file
from test_spatial import cells_of

from insight_contracts import PhenomenonThresholds, Shift, wording
from insight_contracts.graph import dataset_node_id
from insight_graph_service.core.engine import PipelineError
from insight_graph_service.server.app import create_app
from subgroup_miner.config import MinerConfig
from subgroup_miner.context import NONE_NEARBY
from subgroup_miner.discovery import _event_weights, _median_z, _per_event, _weighted_median, _weighted_stats, recompute_statistics, run_discovery
from subgroup_miner.ingestion import load_dataset, parse_ring

ROOT = Path(__file__).resolve().parent.parent


def mixed(tmp_path):
    """The schools of ``planted`` with the target's mix planted instead of its density: one density everywhere, uneven
    cells (8 to 40 events), eight in ten events of kind a in a school's disk against three in ten elsewhere, and a
    north/south half as a condition beside the band."""
    rng = np.random.default_rng(5)
    cells, schools, near = schools_of(rng)
    north = h3.cell_to_latlng(ORIGIN)[0]
    parts = []
    for c in cells:
        n, lat_lon = int(rng.integers(8, 41)), h3.cell_to_latlng(c)
        kind = np.where(rng.random(n) < (0.8 if c in near else 0.3), "a", "b")
        half = "north" if lat_lon[0] > north else "south"
        parts.append(
            pd.DataFrame(
                {
                    "lat": lat_lon[0] + rng.normal(0, 1e-4, n),
                    "lon": lat_lon[1] + rng.normal(0, 1e-4, n),
                    "load": rng.normal(10, 2, n),
                    "kind": kind,
                    "half": half,
                }
            )
        )
    target, context = tmp_path / "target.csv", tmp_path / "schools.csv"
    pd.concat(parts, ignore_index=True).to_csv(target, index=False)
    points_at(schools, 1).to_csv(context, index=False)
    return target, context


def test_a_per_type_counter_passes_the_layer_null_like_points(engine, tmp_path):
    """A run asks for a per-type counter its target was read without (``count_values``, checked against the target's
    rows): three times as many a's in a school's disk, the counter ``count_kind_a`` is a metric of the run's cells, the
    cohort near the schools raises it past the dealt layer as it raises ``points`` — the same null, read through the
    run's own reading —, and mining admits it as a primary with the layer's band in its scope: a neighbourhood
    condition restates ``points``, never a type's own count (docs/02_discovery.md §2.9). The run's Dataset node names
    the counters asked and what a count is per."""
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        refused = cross_run(client, a, b, count_values="kind=z")
        assert refused.status_code == 400 and "no row holds" in refused.json()["detail"]
        run = engine.ws.load_batch(cross_run(client, a, b, count_values="kind=a").json()["batch_id"])
    assert run["status"] == "READY" and [(c["value"], c["status"]) for c in run["count_values"]] == [("a", "ok")]
    loaded = engine._load(run)  # the run's file, its layer and its counter, as its processing read them
    near = [(BAND, "q1")]
    for metric in ("count_kind_a", "points"):
        fresh = recompute_statistics(loaded.frame, near, f"{BAND}=='q1'", [metric], loaded.spatial, 10, engine.settings.miner, 4)
        assert fresh["shifts"][0]["robust_z"] > 1 and fresh["spatial"]["p_context"] * 4 <= 0.05, (metric, fresh["spatial"])
    assert "count_kind_a" not in loaded.spatial.derived_from["lisa_points"]  # a hot spot restates every event's count, not a type's
    found = [p for p in engine.ws.patterns() if p["dataset_id"] == run["dataset_id"] and p["target"] == "count_kind_a"]
    assert found and all(any(c["attribute"] == BAND for c in p["conditions"]) for p in found)  # F admitted, on the layer's band
    assert all(p["provenance"]["spatial"]["p_context"] * p["provenance"]["spatial"]["context_family"] <= 0.05 for p in found)
    props = engine.graph().nodes[dataset_node_id(run["dataset_id"])]["props"]
    assert props["count_values"] == run["count_values"] and props["exposure"]["kind"] == "area"


def test_the_layer_null_reads_a_share_cohort_with_the_same_weights(tmp_path):
    """``WEIGHT_BY_EVENTS`` on: mined per cell, a share's insight is confirmed per event — its local and baseline the
    events' weighted medians, its p and its layer's ``p_context`` drawn with the same weights, its ``n_eff`` Kish's size
    times the Moran factor —; the recomputation reproduces every number, and the dataset id names the weighting."""
    target, context = mixed(tmp_path)
    on = MinerConfig(context_permutations=99, weight_by_events=True)
    loaded = load_dataset(target, on, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], context_ring=1)
    assert (
        loaded.dataset_id
        != load_dataset(
            target,
            MinerConfig(context_permutations=99, weight_by_events=False),
            geo=f"lat,lon@{RES}",
            context=[("schools", context, "auto")],
            context_ring=1,
        ).dataset_id
    )
    table = tmp_path / "table.csv"
    pd.DataFrame({"x": np.random.default_rng(3).normal(size=300), "k": np.resize(["a", "b", "c"], 300)}).to_csv(table, index=False)
    assert (
        load_dataset(table, on).dataset_id == load_dataset(table, MinerConfig(weight_by_events=False)).dataset_id
    )  # no cells, no shares: the weights apply to nothing
    insights, result = mined(loaded, on)
    assert result.profile.weight_by_events
    family = sum(any(c.attribute == BAND for c in i.conditions) for i in insights)
    (lead, *_) = [i for i in cross(insights) if i.target.startswith("share_") and "p_context" in i.provenance["spatial"]]
    conditions = [(c.attribute, c.value) for c in lead.conditions]
    fresh = recompute_statistics(
        loaded.frame, conditions, lead.expression, [s.metric for s in lead.shifts], loaded.spatial, result.n_tests, on, family
    )
    events, share = loaded.frame["points"].to_numpy(float), loaded.frame[lead.target].to_numpy(float)
    rows = fresh["rows"][~np.isnan(share[fresh["rows"]])]
    assert lead.local == _weighted_median(share[rows], events[rows]) != np.median(share[rows])  # per event, not per cell
    assert (fresh["shifts"][0]["local_median"], fresh["shifts"][0]["global_median"], fresh["p_value"]) == (lead.local, lead.baseline, lead.p_value)
    assert fresh["spatial"] == {k: lead.provenance["spatial"][k] for k in fresh["spatial"]}  # p_context and n_eff drawn again, the same
    assert lead.provenance["spatial"]["n_eff"] < len(rows)  # Kish's size of uneven cells, times the Moran factor


def test_a_chosen_value_becomes_a_counter_the_bands_restate_and_no_band_of_its_own():
    """``count_values`` makes each chosen value a per-type counter of every cell (``count_<column>_<value>``: the cell's
    events of that value, divided by nothing): the mode of its own column restates it, the neighbourhood bands and
    ``lisa_points`` do not (they restate every event's count), and it adds no neighbourhood band of its own; the
    counted value's share is written too, whatever its rarity; ``exposure`` names the occupied cells' mean area. Without
    the option nothing changes; cells far apart in latitude differ in area, and say so (docs/02_discovery.md §2.9)."""
    frame = geo_frame()
    plain, plain_ctx, _ = cells_of(frame)
    table, ctx, warnings = cells_of(frame, count_values=["channel=phone", "sector=north"])
    phone = (frame["channel"] == "phone").to_numpy()
    expected = np.bincount(ctx.point_cell[phone], minlength=len(table)).astype(float)
    assert np.array_equal(table["count_channel_phone"].to_numpy(), expected)  # every cell, zero where the value has no event
    assert (table["count_sector_north"] <= table["points"]).all() and table["count_sector_north"].sum() == (frame["sector"] == "north").sum()
    counters = {"count_channel_phone", "count_sector_north"}
    assert not counters & set(ctx.derived_from["neighbours_points_band"]) and not counters & set(ctx.derived_from["lisa_points"])
    assert "count_sector_north" in ctx.derived_from["sector_mode"]  # the cell's most frequent sector restates its own count
    assert set(table.columns) - set(plain.columns) == counters  # no neighbours_count_*: the family grows by the counters alone
    web = cells_of(frame, count_values=["channel=web"])[0]  # the commoner level of a binary column, counted: its share written too
    assert "share_channel_web" in web.columns and "share_channel_web" not in plain.columns and "share_channel_phone" in web.columns
    occupied = round(float(np.mean([h3.cell_area(c, "km^2") for c in ctx.cells])), 6)
    assert ctx.exposure == {"kind": "area", "cell_km2": occupied} and plain_ctx.exposure == {} and np.array_equal(ctx.events, table["points"])
    assert not any("areas differ" in w for w in warnings)  # one region's cells: one area
    far = frame.assign(latitude=np.where(frame["sector"] == "north", 55.0, 0.5) + frame["latitude"] % 0.01)
    assert any("areas differ by" in w for w in cells_of(far, count_values=["channel=phone"])[2])  # 0.5° and 55° north


def test_counted_values_are_checked_against_the_rows_and_a_rare_one_is_a_status(tmp_path):
    """An upload's ``count_values`` are read against its rows: each value with its rows and a status — ``ok``, or
    ``sparse`` below ``MIN_ROWS × MIN_CELL_POINTS`` rows (no counter: too few events for a frequency; the level is the
    one asked) —, on the geo levels only; a column the file lacks, a value no row holds (its column's values listed) and
    a layer-only upload's or a plain table's counters are refused; the dataset id names the counters. The rows that
    count are the located ones — what the cells will hold —, so a value held mostly by rows without a point is sparse
    (the e2e review's F-20)."""
    frame = geo_frame()
    frame.loc[frame.index[:300], "channel"] = "fax"  # 300 rows, 30 of them with a point: below 50 × 5 on the cells
    frame.loc[frame.index[30:300], "latitude"] = np.nan
    path = tmp_path / "points.csv"
    frame.to_csv(path, index=False)
    engine = make_engine(make_config(tmp_path))
    with path.open("rb") as stream:
        records = engine.upload(path.name, stream, geo="latitude,longitude@7", table=True, count_values="channel=phone, channel=fax")
    geo, plain = (engine.process(r["batch_id"]) for r in sorted(records, key=lambda r: r["level"] == "table"))
    asked = [(c["value"], c["rows"], c["status"]) for c in geo["count_values"]]
    assert asked == [("phone", int((frame["channel"] == "phone").sum()), "ok"), ("fax", 30, "sparse")] and geo["level"] == "7"
    assert "count_values" not in plain  # a table level counts nothing
    columns = engine._load(geo).frame.columns
    assert "count_channel_phone" in columns and "count_channel_fax" not in columns
    from insight_graph_service.core.snapshot import metadata_digest

    assert (
        metadata_digest(geo | {"count_values": []}) != metadata_digest(geo) != metadata_digest(geo | {"baseline": "zone"})
    )  # signed: the card reads them
    with path.open("rb") as stream:
        again = engine.upload(path.name, stream, geo="latitude,longitude@7")[0]
    assert again["status"] == "UPLOADED" and engine.process(again["batch_id"])["dataset_id"] != geo["dataset_id"]  # no counters, another id
    for bad, why in (("nope=a", "no column"), ("channel=pager", "phone"), ("load=1", "not a category"), ("channel", "one column=value")):
        with path.open("rb") as stream, pytest.raises(PipelineError, match=why):
            engine.upload(path.name, stream, geo="latitude,longitude@7", count_values=bad)
    with path.open("rb") as stream, pytest.raises(PipelineError, match="count_values"):
        engine.upload(path.name, stream, geo="latitude,longitude", layer=True, count_values="channel=phone")
    with path.open("rb") as stream, pytest.raises(PipelineError, match="without geo"):
        engine.upload(path.name, stream, count_values="channel=phone")  # a table has no cells to count into


def test_event_weights_are_the_counts_as_counted_whatever_the_baseline(tmp_path):
    """``WEIGHT_BY_EVENTS`` with ``baseline=zone``: the zone replaces ``points`` with signed residuals, but a share's test
    weighs the cells by their events as counted (``SpatialContext.events``): every weight at least one, the confirmed
    shift and its z finite, the same weights the recomputation reads (the e2e review's F-01)."""
    target, context = mixed(tmp_path)
    on = MinerConfig(context_permutations=49, weight_by_events=True)
    loaded = load_dataset(target, on, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], baseline="zone")
    frame, ctx = loaded.frame, loaded.spatial
    assert (frame["points"] <= 0).any() and (ctx.events >= 1).all() and ctx.events.sum() == (ctx.point_cell >= 0).sum()
    weights = _event_weights("share_kind_a", on, ctx)
    assert weights is ctx.events and _event_weights("load_median", on, ctx) is None and _event_weights("share_kind_a", on, None) is None
    rows = np.flatnonzero(frame["points"].to_numpy() <= 0)  # a silent cohort: every cell at or under its zone's median
    assert len(rows) >= 2
    shift, own = _per_event(frame, rows, Shift("share_kind_a", 1.0, 0.5, 0.4, 0.1), on, ctx)
    assert np.isfinite(shift.local_median) and own is not None and (own >= 1).all() and len(own) == len(rows)
    assert np.isfinite(_median_z(frame["share_kind_a"].to_numpy(float)[rows], 0.4, 0.1, 0.5, own))


def test_the_weighted_median_follows_the_events_not_the_cells():
    """With each cell weighing its events, a share's median is the events' typical value: one busy cell outweighs two
    quiet ones. Equal weights read as the plain test (an odd count); unequal ones shrink the sample to Kish's effective
    size, and the Moran factor multiplies it on top — two corrections, neither in place of the other (§2.9)."""
    assert _weighted_median(np.array([0.1, 0.1, 0.9]), np.array([1.0, 1.0, 100.0])) == 0.9  # the plain median is 0.1
    rng = np.random.default_rng(1)
    sample, w = rng.normal(0.5, 0.1, 31), rng.uniform(1, 50, 31)
    assert _median_z(sample, 0.4, 0.1, 0.5, np.ones(31)) == pytest.approx(_median_z(sample, 0.4, 0.1, 0.5))
    median, mad = _weighted_stats(sample, w)
    kish = w.sum() ** 2 / (w**2).sum()
    expected = (median - 0.4) / (1.2533 * 1.4826 * mad / math.sqrt(kish * 0.5))
    assert kish < 31 and _median_z(sample, 0.4, 0.1, 0.5, w) == pytest.approx(expected)
    assert (
        _median_z(sample, 0.4, 0.1, 0.02, w) == 0.0 and _median_z(sample[:2], 0.4, 0.1, 0.3) == 0.0
    )  # under one effective value: no test, not a floor
    assert _weighted_stats(np.array([0.2, 0.2, 0.2, 0.6, np.nan]), np.ones(5)) == (
        0.2,
        pytest.approx(0.0845, abs=1e-3),
    )  # NaN out; the mean deviation


def test_the_zone_baseline_uses_the_nulls_parent_and_replaces_the_columns():
    """The zone is the null's parent cell; counters and shares are replaced by their distance from its median (§2.9)."""
    frame = geo_frame()
    plain, ctx_map, _ = cells_of(frame)
    zoned, ctx, _ = cells_of(frame, baseline="zone")
    assert list(zoned.columns) == list(plain.columns) and (ctx.baseline, ctx_map.baseline) == ("zone", None)
    parents = pd.Series([h3.cell_to_parent(c, 6) for c in ctx.cells])
    assert (pd.factorize(parents)[0] == ctx.blocks).all()  # the zone is the null's and the bootstrap's parent cell
    logged = np.log1p(plain["points"])
    assert np.allclose(zoned["points"], logged - logged.groupby(parents).transform("median"))
    shares = plain["share_channel_phone"]
    assert np.allclose(zoned["share_channel_phone"], shares - shares.groupby(parents).transform("median"), equal_nan=True)
    assert zoned["share_channel_phone"].isna().equals(shares.isna()) and zoned["neighbours_points"].equals(plain["neighbours_points"])


def test_the_zone_baseline_removes_the_difference_between_zones():
    """Two halves of a map at different levels: against the map the north half's cells read six units higher; against
    their zones the difference between the halves is gone, while each cell keeps its distance from its zone's median."""
    frame = geo_frame()
    frame["load"] = frame["load"] + np.where(frame["sector"] == "north", 6.0, 0.0)
    plain, ctx, _ = cells_of(frame)
    zoned = cells_of(frame, baseline="zone")[0]
    north = plain["sector_mode"] == "north"
    gap = lambda t: t.loc[north, "load_median"].median() - t.loc[~north, "load_median"].median()  # noqa: E731
    assert gap(plain) > 5 and abs(gap(zoned)) < 1


def test_the_maps_hover_counts_events_as_counted_under_a_zone_baseline(engine, tmp_path):
    """A zoned level's ``points`` column holds signed residuals; the map's cell properties count the events as counted,
    so the hover never reads a negative number of events (the e2e review's F-15)."""
    from insight_graph_service.core.geo_views import geo_cells

    target, _ = planted(tmp_path)
    with target.open("rb") as stream:
        (record,) = [engine.process(r["batch_id"]) for r in engine.upload("target.csv", stream, geo=f"lat,lon@{RES}", baseline="zone")]
    loaded = engine._load(record)
    counts = [f["properties"]["points"] for f in geo_cells(engine.ws, record["dataset_id"], loaded)["features"]]
    assert (loaded.frame["points"] <= 0).any() and min(counts) >= 1 and sum(counts) == int((loaded.spatial.point_cell >= 0).sum())


def test_a_planted_dependence_survives_the_zone_baseline(engine, tmp_path):
    """A zone baseline takes out the differences between parent cells only: the schools' dependence, which varies within
    them, still beats the dealt layer. The run asks for the baseline (or keeps its target level's), names it on its
    record, its insights' provenance and its Dataset node, and ``verify`` reads it the same (docs/02_discovery.md §2.9)."""
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        assert cross_run(client, a, b, baseline="region").json()["code"] == "invalid_options"
        run = engine.ws.load_batch(cross_run(client, a, b, baseline="zone").json()["batch_id"])
        plain = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
    assert run["baseline"] == "zone" and "baseline" not in plain and run["dataset_id"] != plain["dataset_id"]
    patterns = [p for p in engine.ws.patterns() if p["dataset_id"] == run["dataset_id"]]
    found = [p for p in patterns if p["effect_size"] > 0 and any(c["attribute"] == BAND and c["value"] != NONE_NEARBY for c in p["conditions"])]
    family = found[0]["provenance"]["spatial"]["context_family"]
    assert any(p["provenance"]["spatial"]["p_context"] * family <= 0.05 for p in found)  # the dependence within the zones
    assert all(p["provenance"]["spatial"]["baseline"] == "zone" for p in patterns)
    assert engine.graph().nodes[dataset_node_id(run["dataset_id"])]["props"]["baseline"] == "zone"
    assert engine.verify_pattern(found[0]["id"])["status"] == "reproduced"


def test_a_runs_insight_names_its_disk_and_the_other_links_of_its_base_in_the_same_places(engine, tmp_path):
    """The drawer's facts for a layer link's card (``GET /api/nodes``): the run's disk and what a count is per in words,
    and each ``CO_LOCATED`` row of another run of the same base layer named with that run's layers; none for a plain
    upload's insight."""
    from insight_contracts import wording

    target, context = planted(tmp_path)
    patrols = tmp_path / "patrols.csv"
    pd.read_csv(context).assign(unit=1).to_csv(patrols, index=False)  # the same points, other bytes: another layer
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", count_values="kind=a")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        c = upload_file(client, "patrols.csv", patrols, "lat,lon", layer="true")["upload_id"]
        first, second = (engine.ws.load_batch(cross_run(client, a, layer).json()["batch_id"]) for layer in (b, c))
        found = [x["id"] for x in engine.ws.patterns() if x["dataset_id"] == first["dataset_id"]]
        details = [client.get(f"/api/nodes/{pid}").json() for pid in found]
        plain = next(x["id"] for x in engine.ws.patterns() if x["dataset_id"] not in (first["dataset_id"], second["dataset_id"]))
        assert "link" not in client.get(f"/api/nodes/{plain}").json()
    assert first["status"] == second["status"] == "READY" and details
    words = {"disk": wording.disk_words(1), "exposure": wording.exposure_words(details[0]["dataset"]["exposure"])}
    assert all(d["link"].items() >= words.items() and "pair" in d["link"] for d in details)  # the pair: link_pair's own test
    rows = [it for d in details for key in ("CO_LOCATED", "CO_LOCATED (in)") for it in d["neighbors"].get(key, [])]
    named = [it for it in rows if "layers" in it]
    assert named and all(it["layers"] == ["patrols"] for it in named)
    assert all(engine.graph().nodes[it["id"]]["props"]["dataset_id"] == second["dataset_id"] for it in named)


def test_a_pair_is_the_same_insights_other_question_never_a_second_insight(engine, tmp_path):
    """``views.link_pair``: one cohort is one insight with one tested primary, so its other question is the complementary
    shift of the same insight — a count's share, a share's count, of the same value — with its medians, untested on its
    own; none without that shift or for a metric of neither kind. On ``mixed`` with the a's counted, the count insight
    carries its share and the share insight its count (the e2e review's F-31)."""
    from types import SimpleNamespace as N

    from insight_graph_service.server.views import link_pair

    def ins(target, *shifts):
        return N(target=target, shifts=[N(metric=m, robust_z=z, local_median=lo, global_median=hi) for m, z, lo, hi in shifts])

    insights = {
        "c": ins("count_kind_a", ("count_kind_a", 2.0, 3.0, 1.0), ("share_kind_a", 0.5, 0.12, 0.07)),
        "s": ins("share_kind_a", ("share_kind_a", 1.0, 0.12, 0.07)),
        "p": ins("points", ("points", 1.0, 3.0, 1.0), ("share_kind_a", 0.5, 0.12, 0.07)),
    }
    graph = N(insight=insights.__getitem__)
    assert link_pair(graph, "c") == {"target": "share_kind_a", "effect": 0.5, "local": 0.12, "baseline": 0.07}
    assert link_pair(graph, "s") is None and link_pair(graph, "p") is None
    target, context = mixed(tmp_path)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        run = engine.ws.load_batch(cross_run(client, a, b, count_values="kind=a").json()["batch_id"])
        pairs = {
            p["target"]: client.get(f"/api/nodes/{p['id']}").json()["link"]["pair"]
            for p in engine.ws.patterns()
            if p["dataset_id"] == run["dataset_id"]
        }
    assert "share_kind_a" in pairs and pairs["share_kind_a"]["target"] == "count_kind_a", sorted(pairs)  # the share's cohort carries its count
    assert pairs["share_kind_a"]["local"] > pairs["share_kind_a"]["baseline"] >= 1
    assert all(p["target"] in {"count_kind_a", "share_kind_a"} for t, p in pairs.items() if p and t in ("count_kind_a", "share_kind_a"))


def rare(tmp_path):
    """The schools of ``planted`` with a rare type's frequency planted: eight events in every cell, of kinds b and c
    evenly but one in a hundred of kind r away from the schools and six in a hundred in a school's disk — so a cell's count of r is 0 in most cells
    near a school too (its median 0 everywhere), while its mean there is six times the map's."""
    rng = np.random.default_rng(11)
    cells, schools, near = schools_of(rng)
    parts = []
    for c in cells:
        lat, lon = h3.cell_to_latlng(c)
        kind = np.where(rng.random(8) < (0.06 if c in near else 0.01), "r", rng.choice(["b", "c"], 8))  # three kinds: shares, as a type column
        parts.append(
            pd.DataFrame({"lat": lat + rng.normal(0, 1e-4, 8), "lon": lon + rng.normal(0, 1e-4, 8), "load": rng.normal(10, 2, 8), "kind": kind})
        )
    target, context = tmp_path / "target.csv", tmp_path / "schools.csv"
    pd.concat(parts, ignore_index=True).to_csv(target, index=False)
    points_at(schools, 1).to_csv(context, index=False)
    return target, context


def test_a_rare_types_count_is_read_by_its_mean_where_its_median_is_zero(tmp_path):
    """A counted type whose map median is 0 (more than half the cells hold none) says nothing through a median: the
    cohort near the schools holds six times its rate and still a median 0. Such a counter is read by its mean —
    in the screen, the confirmation and the layer's null alike —, the insight names it among its ``means``, and its
    words say *у середньому*; a counter with a positive median stays a median (docs/02_discovery.md §2.9)."""
    target, context = rare(tmp_path)
    config = MinerConfig(context_permutations=99)
    loaded = load_dataset(
        target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], context_ring=1, count_values=["kind=r", "kind=b"]
    )
    counts = loaded.frame["count_kind_r"]
    assert counts.median() == 0 and counts[loaded.frame[BAND] != NONE_NEARBY].median() == 0  # the median sees nothing
    insights, result = mined(loaded, config)
    assert result.profile.means == ["count_kind_r"]  # count_kind_b's median is positive: it stays a median
    (found, *_) = [i for i in cross(insights) if i.target == "count_kind_r"]
    assert found.local > 2 * found.baseline and found.provenance["means"] == ["count_kind_r"]  # the map holds the near cells too
    family = sum(any(c.attribute == BAND for c in i.conditions) for i in insights)
    fresh = recompute_statistics(
        loaded.frame,
        [(c.attribute, c.value) for c in found.conditions],
        found.expression,
        ["count_kind_r"],
        loaded.spatial,
        result.n_tests,
        config,
        family,
    )
    assert (fresh["shifts"][0]["local_median"], fresh["p_value"]) == (found.local, found.p_value)  # a re-check reads the same mean
    assert found.provenance["spatial"]["p_context"] * family <= 0.05  # past the layer's null, read by the same mean
    assert "у середньому" in wording.reading(found, PhenomenonThresholds())["what"]


def test_the_layers_null_draws_from_its_points_not_its_name(tmp_path):
    """The null's draws are seeded from the scope with each layer named by its content (the file's hash), so the same
    points under another file name give the same ``p_context``: a renamed copy is the same evidence."""
    target, context = mixed(tmp_path)
    config = MinerConfig(context_permutations=99)
    renamed = tmp_path / "clinics.csv"
    renamed.write_bytes(context.read_bytes())
    drawn = []
    for name, path in (("schools", context), ("clinics", renamed)):
        loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[(name, path, "auto")], context_ring=1)
        band = f"context_{name}_band"
        fresh = recompute_statistics(
            loaded.frame, [(band, "q1")], f"{band}=='q1'", ["load_median"], loaded.spatial, 10, config
        )  # family 1: a wider family stops at its first hits
        drawn.append(fresh["spatial"]["p_context"])
    assert drawn[0] == drawn[1] and 0.05 < drawn[0] < 1  # a mid-range p: the draws decide it, and they are the same


def test_event_weights_are_on_and_the_link_reads_the_cell_alone_by_default():
    """Two defaults the cross-layer experiments measured (review 2026-11-04, X-1 and X-2): a share is confirmed per
    event unless switched off, and a layer is counted in the cell itself unless a wider disk is asked — at the old
    default disk the layer's within-parent null kept 88 % of the cells near a point near it and refused every
    same-cell effect. The console offers the cell alone first."""
    assert MinerConfig().weight_by_events and parse_ring(None) == parse_ring("") == 0
    page = (ROOT / "sig_web_console" / "static" / "index.html").read_text(encoding="utf-8")
    assert re.search(r'id="link-ring"><option value="0">', page)


def test_an_asked_counter_is_mined_even_where_its_share_ranks_with_it(tmp_path):
    """A rare type is 0 in most cells by count and by share, so the two rank together (0.99 on the Chicago crimes'
    robbery): the profile's redundancy rule kept the share and dropped the counter the run asked for, silently. The
    counters a run asks for and their shares are its question: never dropped as redundant."""
    target, context = rare(tmp_path)
    config = MinerConfig(context_permutations=0)
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")], context_ring=1, count_values=["kind=r"])
    frame = loaded.frame
    assert frame[["count_kind_r", "share_kind_r"]].rank().corr().iloc[0, 1] > 0.95  # the redundancy rule's threshold
    result = run_discovery(frame, config, spatial=loaded.spatial)
    assert {"count_kind_r", "share_kind_r"} <= set(result.profile.numerics)


def test_the_prompts_band_words_follow_the_layers_band_count():
    """The English of a layer's bands (the evidence prompt's glossary) names the bands the layer has, as the console's
    Ukrainian does: the shipped bars at r9, counted in the cell alone, give two bands (most cells with a bar hold one),
    so «q2» is the top band — «the most», never «fewer than the middle» (on 2026-11-06 the narrator read the finding
    «more thefts where the bars are» as the opposite)."""
    from subgroup_miner.context import ContextLayer, band_words, glossary

    def layer(edges):
        return ContextLayer("bars", np.zeros(1), np.zeros(1, int), np.zeros(2, int), np.zeros(0, int), np.asarray(edges, float))

    two = glossary(layer([1, 3, 43]), 0)
    assert two["context_bars_band=q2"] == "the cell itself holds the most bars points (of the cells that have any)"
    assert "the fewest" in two["context_bars_band=q1"] and "q3" not in str(two) and "2 bands" in two["context_bars_band"]
    assert two["context_bars_band=none nearby"] == "no bars point in the cell itself"  # the disk in words, as ``wording.disk_words``
    four = glossary(layer([1, 2, 3, 5, 43]), 1)
    assert four["context_bars_band=q2"] == "the cell and its six neighbours hold fewer than the middle bars points (of the cells that have any)"
    assert "quarters" in four["context_bars_band"]
    for k in (1, 2, 3, 4):  # the top band is the top band in both languages
        assert band_words(k, k) in ("the most", "some") and wording.layer_band_words(k, k + 1) in ("найбільше", "є")
