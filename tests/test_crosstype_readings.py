"""Readings of one file under a row filter (``where``: one column, one value) — Scenario D of the M18 specification
(docs/extra_funcs/spatial_analytics_multi_type/cross_dataset_analysis_architecture.md §3, §11; docs/02_discovery.md §2.9):
its own reading over the same stored bytes, one filter for a target and for a layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_crosstype import RES, cross_run, planted, upload_file

from insight_graph_service.server.app import create_app
from subgroup_miner.config import MinerConfig
from subgroup_miner.ingestion import IngestionError, forget_tables, load_dataset, read_context


def twice(tmp_path):
    """The planted target written twice in one file, as kind ``a`` and as kind ``b`` (each reading mines as the plant
    does), the schools beside it."""
    target, context = planted(tmp_path)
    frame = pd.read_csv(target)
    pd.concat([frame.assign(kind="a"), frame.assign(kind="b", load=frame["load"] * 2)]).to_csv(both := tmp_path / "both.csv", index=False)
    return both, context, len(frame)


def test_a_filtered_reading_counts_its_rows_only_and_is_another_dataset(tmp_path):
    """``where`` keeps the rows holding the value and drops the column it reads (constant now); the levels of one upload
    share one point profile, so a filtered reading keeps its own beside the whole file's; a filter as written or trimmed
    is the same reading; a column the file lacks, a number, a value no row holds or a spec without ``=`` is refused,
    the column's values listed with their counts."""
    both, _, n = twice(tmp_path)
    config, geo = MinerConfig(), f"lat,lon@{RES}"
    whole = load_dataset(both, config, geo=geo, shared=True)
    only_a = load_dataset(both, config, geo=geo, shared=True, where="kind=a")  # after the whole file's profile is kept
    forget_tables()
    assert whole.frame["points"].sum() == 2 * n and only_a.frame["points"].sum() == n
    assert not [c for c in only_a.frame.columns if "kind" in c] and only_a.dataset_id != whole.dataset_id
    assert load_dataset(both, config, geo=geo, where=" kind = a ").dataset_id == only_a.dataset_id
    for spec, words in (("kind=c", f"a ({n})"), ("load=3", "not a category"), ("nowhere=1", "no column"), ("kind", "column=value")):
        with pytest.raises(IngestionError) as refused:
            load_dataset(both, config, geo=geo, where=spec)
        assert refused.value.code == "invalid_options" and words in str(refused.value), (spec, str(refused.value))


def test_an_upload_with_where_is_its_own_reading_over_the_stored_file(engine, tmp_path):
    """An upload under a filter is its own upload id (bytes and filter) over the file already stored — no second copy —,
    READY on its rows, verified the same; an unknown value is refused before any batch, the values listed; a layer is
    stored whole (a filter is the run's)."""
    both, _, n = twice(tmp_path)
    with TestClient(create_app(engine)) as client:
        whole = upload_file(client, "both.csv", both, f"lat,lon@{RES}")
        only_a = upload_file(client, "both.csv", both, f"lat,lon@{RES}", where="kind=a")
        refused = [
            client.post("/api/upload", files=[("file", ("both.csv", both.read_bytes(), "text/csv"))], data=data)
            for data in ({"geo": f"lat,lon@{RES}", "where": "kind=zzz"}, {"geo": "lat,lon", "layer": "true", "where": "kind=a"})
        ]
    assert [r.status_code for r in refused] == [400, 400] and all(r.json()["code"] == "invalid_options" for r in refused)
    assert f"a ({n})" in refused[0].json()["detail"] and f"b ({n})" in refused[0].json()["detail"]
    reading = next(r for r in engine.ws.list_batches() if r["upload_id"] == only_a["upload_id"])
    plain = next(r for r in engine.ws.list_batches() if r["upload_id"] == whole["upload_id"])
    assert reading["status"] == "READY" and reading["where"] == "kind=a" and only_a["upload_id"] != whole["upload_id"]
    assert reading["source_path"] == plain["source_path"] and len(list(engine.ws.uploads_dir.glob("*"))) == 1  # one stored file
    patterns = [p for p in engine.ws.patterns() if p["dataset_id"] == reading["dataset_id"]]
    assert patterns and engine.verify_pattern(patterns[0]["id"])["status"] == "reproduced"


def test_one_file_crosses_itself_under_two_filters_read_in_place(engine, tmp_path):
    """One file holding the target's rows (kind ``a``) and the schools (kind ``s``): the target read under one filter and
    the file, kept whole as a layer, under another (``layer_where``) — the run reads the layer's rows in place, no copy,
    named ``<stem>_<value>``, and finds the plant; a run of a reading with itself, and a filter for a layer not in the
    run or for one already filtered, are refused."""
    target, context = planted(tmp_path)
    frame = pd.read_csv(target)
    schools = pd.read_csv(context).assign(load=0.0, kind="s")
    pd.concat([frame.assign(kind="a"), schools]).to_csv(mixed := tmp_path / "mixed.csv", index=False)
    with TestClient(create_app(engine)) as client:
        kept = upload_file(client, "mixed.csv", mixed, "lat,lon", layer="true")["upload_id"]
        a = upload_file(client, "mixed.csv", mixed, f"lat,lon@{RES}", where="kind=a")["upload_id"]
        other = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        refused = [
            cross_run(client, a, a),
            cross_run(client, a, kept, layer_where={"up-000000000000": "kind=s"}),
            cross_run(client, other, a, layer_where={a: "kind=s"}),  # a reading filtered already
        ]
        run = cross_run(client, a, kept, layer_where={kept: "kind=s"})
    assert [r.status_code for r in refused] == [400] * 3 and all(r.json()["code"] == "invalid_options" for r in refused)
    record = engine.ws.load_batch(run.json()["batch_id"])
    (entry,) = record["context"]
    assert (entry["name"], entry["where"], entry["path"]) == ("mixed_s", "kind=s", record["source_path"])  # read in place
    assert len(list(engine.ws.uploads_dir.glob("*mixed.csv"))) == 1 and record["status"] == "READY"  # one copy of the file
    located = read_context("mixed_s", Path(entry["path"]), "lat,lon", where="kind=s")[0]
    assert np.isfinite(located.lat).sum() == len(schools)  # the layer's rows only, not the target's beside them
    layer = [("mixed_s", Path(entry["path"]), "lat,lon", None, "kind=s")]
    direct = load_dataset(
        Path(record["source_path"]),
        engine.settings.miner,
        filename=record["filename"],
        geo=record["geo"],
        where="kind=a",
        context=layer,
        context_ring=record["context_ring"],
    )
    assert record["dataset_id"] == direct.dataset_id  # both filters, the run's own load included
    assert any(
        c["attribute"] == "context_mixed_s_band" for p in engine.ws.patterns() if p["dataset_id"] == record["dataset_id"] for c in p["conditions"]
    )
