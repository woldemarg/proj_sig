"""Cross-type spatial context: a second point type as one forced band of the target's cell table, and the permutation null
of a cross-type cohort (docs/02_discovery.md §2.9; docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md)."""

from __future__ import annotations

import hashlib

import h3
import numpy as np
import pandas as pd
import pytest
from conftest import toy_insight

from insight_contracts import Condition, Shift, SpatialProvenance
from insight_contracts.insight import SPATIAL_DIGEST_FIELDS
from insight_contracts.wording import column_words, condition_words, source_column
from subgroup_miner.config import MinerConfig
from subgroup_miner.context import NONE_NEARBY, ContextSource, dealt_counts, disk_sums
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
    drawn = dealt_counts(layer, np.random.default_rng(1))
    assert drawn.sum() == layer.counts.sum() and not np.array_equal(drawn, layer.counts)
    for parent in np.unique(layer.parents):
        assert drawn[layer.parents == parent].sum() == layer.counts[layer.parents == parent].sum()


def test_the_fingerprint_names_the_layers_whatever_their_order():
    digest = hashlib.sha256(b"rows")
    plain = dataset_fingerprint(digest, {}, geo="lat,lon@7")
    ab = dataset_fingerprint(digest, {}, geo="lat,lon@7", context=["schools:0123456789ab:auto", "shops:ba9876543210:auto"])
    ba = dataset_fingerprint(digest, {}, geo="lat,lon@7", context=["shops:ba9876543210:auto", "schools:0123456789ab:auto"])
    assert plain != ab == ba


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


def test_a_planted_dependence_beats_the_dealt_layer_and_the_recomputation_draws_the_same(tmp_path):
    target, context = planted(tmp_path)
    config = MinerConfig(context_permutations=99)
    loaded = load_dataset(target, config, geo=f"lat,lon@{RES}", context=[("schools", context, "auto")])
    insights, result = mined(loaded, config)
    found = [i for i in cross(insights) if i.target in ("points", "load_median") and i.effect_size > 0]
    assert found and min(i.provenance["spatial"]["p_context"] for i in found) <= 0.05
    lead = min(found, key=lambda i: i.provenance["spatial"]["p_context"])
    fresh = recompute_statistics(
        loaded.frame,
        [(c.attribute, c.value) for c in lead.conditions],
        lead.expression,
        [s.metric for s in lead.shifts],
        loaded.spatial,
        result.n_tests,
        config,
    )
    assert fresh["spatial"] == {k: lead.provenance["spatial"][k] for k in fresh["spatial"]}  # p_context drawn again, the same
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


def context_insight(p_context, **fields):
    spatial = {"resolution": 7, "cells": 40, "moran_i": 0.1, "n_eff": 36.0, "p_adjusted_iid": 1e-6, "p_context": p_context}
    values = dict(id="P-x", expression=f"{BAND}=='q4'", support=300, sd_score=1.8, sd_raw_score=2.0, emm_score=0.01, volume_utility=0.3)
    values |= dict(stability=0.9, p_value=1e-8, p_adjusted=1e-6, row_hash="x", provenance={"spatial": spatial})
    return toy_insight([(BAND, "q4")], [Shift("points", 2.0, 12.0, 8.0, 2.0)], **values | fields)


def test_selection_refuses_a_shift_the_dealt_layer_reproduces_and_any_correlation_change_in_a_context_scope():
    config = MinerConfig()
    refused = select_insights([context_insight(0.3)], config)
    assert not refused.kept and [r.reason for r in refused.rejections] == ["context_confounded"] and "0.3" in refused.rejections[0].detail
    assert select_insights([context_insight(0.01)], config).kept and select_insights([context_insight(None)], config).kept
    pair = {"pair": ["points", "load_median"], "local_corr": 0.6, "global_corr": -0.1, "delta": 0.7}
    unstable = context_insight(0.01, stability=0.2, emm_score=0.3, covariance=pair)  # no shift; a strong correlation change
    assert not select_insights([unstable], config).kept  # never retargeted to a covariance insight
    plain = context_insight(None, stability=0.2, emm_score=0.3, covariance=pair)
    assert [i.phenomenon_type for i in select_insights([plain], config).kept] == ["covariance"]  # the rule binds only a tested scope


def test_spatial_provenance_reads_records_without_p_context():
    old = {"resolution": 7, "cells": 40, "moran_i": 0.2, "n_eff": 31.5, "p_adjusted_iid": 0.01}
    spatial = SpatialProvenance.from_dict(old)
    assert spatial.p_context is None and spatial.to_dict() == old  # a record written before M17 reads and writes back the same
    assert SPATIAL_DIGEST_FIELDS == ("resolution", "cells", "moran_i", "n_eff", "p_adjusted_iid", "p_context")
    assert SpatialProvenance.from_dict(old | {"p_context": 0.02}).to_dict() == old | {"p_context": 0.02}


def test_context_conditions_read_in_plain_ukrainian_and_hold_no_bare_literal():
    q4, none = Condition(BAND, "q4"), Condition(BAND, NONE_NEARBY)
    assert "schools" in condition_words(q4, {}, True, {}) and "четвертій чверті" in condition_words(q4, {}, True, {})
    assert condition_words(none, {}, True, {}) == "поруч немає schools"
    assert "сусідніх" in column_words(BAND, {}, True)
    assert source_column(BAND, {"schools", "lat"}) is None  # a declared column named like the layer is not its source
    assert " " in NONE_NEARBY  # two words: a bare word would ground onto unrelated question words (question._values)
