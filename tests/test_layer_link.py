"""The layer link's cell metrics (docs/02_discovery.md §2.9; the concept's §4 and §9 in
docs/extra_funcs/spatial_analytics_multi_type/cross_layer_analysis_concept.md): the per-type counters asked for (*більше
подій*), the shares confirmed per event (*інший склад*), the zone baseline — what a run of a base layer with a context layer
reads, and the layer's null on them."""

from __future__ import annotations

import math

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import geo_frame, make_config, make_engine
from fastapi.testclient import TestClient
from test_crosstype import BAND, ORIGIN, RES, cross, cross_run, mined, planted, points_at, schools_of, upload_file
from test_spatial import cells_of

from insight_contracts.graph import dataset_node_id
from insight_graph_service.core.engine import PipelineError
from insight_graph_service.server.app import create_app
from subgroup_miner.config import MinerConfig
from subgroup_miner.context import NONE_NEARBY
from subgroup_miner.discovery import _median_z, _weighted_median, _weighted_stats, recompute_statistics
from subgroup_miner.ingestion import load_dataset


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
    rows): three times as many a's in a school's disk, the counter ``count_kind_a`` is a metric of the run's cells, and
    the cohort near the schools raises it past the dealt layer as it raises ``points`` — the same null, read through the
    run's own reading. Whether mining admits a counter depends on the data's other conditions: here every cohort beside
    the band holds a neighbourhood condition, which restates the counters. The run's Dataset node names the counters
    asked and what a count is per (docs/02_discovery.md §2.9)."""
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
    assert "count_kind_a" in loaded.spatial.derived_from["lisa_points"]  # the neighbourhood conditions restate it
    props = engine.graph().nodes[dataset_node_id(run["dataset_id"])]["props"]
    assert props["count_values"] == run["count_values"] and props["exposure"]["kind"] == "area"


def test_the_layer_null_reads_a_share_cohort_with_the_same_weights(tmp_path):
    """``WEIGHT_BY_EVENTS`` on: mined per cell, a share's insight is confirmed per event — its local and baseline the
    events' weighted medians, its p and its layer's ``p_context`` drawn with the same weights, its ``n_eff`` Kish's size
    times the Moran factor —; the recomputation reproduces every number, and the dataset id names the weighting."""
    target, context = mixed(tmp_path)
    on = MinerConfig(context_permutations=99, weight_by_events=True)
    loaded = load_dataset(target, on, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")])
    assert (
        loaded.dataset_id
        != load_dataset(target, MinerConfig(context_permutations=99), geo=f"lat,lon@{RES}", context=[("schools", context, "auto")]).dataset_id
    )
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
    events of that value, divided by nothing): the neighbourhood bands and ``lisa_points`` restate it, the mode of its
    own column too, and it adds no neighbourhood band of its own; ``exposure`` names the level's cell area. Without the
    option nothing changes; cells far apart in latitude differ in area, and say so (docs/02_discovery.md §2.9)."""
    frame = geo_frame()
    plain, plain_ctx, _ = cells_of(frame)
    table, ctx, warnings = cells_of(frame, count_values=["channel=phone", "sector=north"])
    phone = (frame["channel"] == "phone").to_numpy()
    expected = np.bincount(ctx.point_cell[phone], minlength=len(table)).astype(float)
    assert np.array_equal(table["count_channel_phone"].to_numpy(), expected)  # every cell, zero where the value has no event
    assert (table["count_sector_north"] <= table["points"]).all() and table["count_sector_north"].sum() == (frame["sector"] == "north").sum()
    counters = {"count_channel_phone", "count_sector_north"}
    assert counters <= set(ctx.derived_from["neighbours_points_band"]) and counters <= set(ctx.derived_from["lisa_points"])
    assert "count_sector_north" in ctx.derived_from["sector_mode"]  # the cell's most frequent sector restates its own count
    assert set(table.columns) - set(plain.columns) == counters  # no neighbours_count_*: the family grows by the counters alone
    assert ctx.exposure == {"kind": "area", "cell_km2": round(h3.average_hexagon_area(7, "km^2"), 6)} and plain_ctx.exposure == {}
    assert not any("areas differ" in w for w in warnings)  # one region's cells: one area
    far = frame.assign(latitude=np.where(frame["sector"] == "north", 55.0, 0.5) + frame["latitude"] % 0.01)
    assert any("areas differ by" in w for w in cells_of(far, count_values=["channel=phone"])[2])  # 0.5° and 55° north


def test_counted_values_are_checked_against_the_rows_and_a_rare_one_is_a_status(tmp_path):
    """An upload's ``count_values`` are read against its rows: each value with its rows and a status — ``ok``, or
    ``sparse`` below ``MIN_ROWS × MIN_CELL_POINTS`` rows (no counter: too few events for a frequency; the level is the
    one asked) —, on the geo levels only; a column the file lacks, a value no row holds (its column's values listed) and
    a layer-only upload's or a plain table's counters are refused; the dataset id names the counters."""
    frame = geo_frame()
    frame.loc[frame.index[:30], "channel"] = "fax"  # 30 rows: below 50 × 5
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


def test_a_pair_is_the_runs_insight_with_the_same_conditions_on_the_same_values_other_question():
    """``views.link_pair`` (X-08): a count's share and a share's count of one run with the same conditions; another scope,
    another value, another run or a metric of neither kind is no pair."""
    from types import SimpleNamespace as N

    from insight_graph_service.server.views import link_pair

    def ins(target, dataset="ds-r", *conds):
        return N(target=target, dataset_id=dataset, conditions=[N(expr=c) for c in conds or ("band=q4",)], effect_size=0.5)

    insights = {"c": ins("count_kind_a"), "s": ins("share_kind_a"), "p": ins("points")}
    graph = N(insights=insights, insight=insights.__getitem__)
    assert link_pair(graph, "c") == {"id": "s", "target": "share_kind_a", "effect": 0.5} and link_pair(graph, "s")["id"] == "c"
    assert link_pair(graph, "p") is None
    insights |= {"s": ins("share_kind_a", "ds-r", "band=q4", "hour=night")}
    assert link_pair(graph, "c") is None  # another scope
    insights |= {"s": ins("share_kind_b")}
    assert link_pair(graph, "c") is None  # another value
    insights |= {"s": ins("share_kind_a", "ds-other")}
    assert link_pair(graph, "c") is None  # another run
