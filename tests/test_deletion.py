"""Deletion and versions: a dataset deleted with its knowledge, an interrupted deletion undone, a new version replacing the
old in one transaction (docs/06_graph_and_storage.md §6.8)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from conftest import Crash, geo_frame, make_config, make_engine, upload_levels

from graph_query_engine.graph import DualGraph


def test_delete_dataset_removes_its_knowledge_and_orphan_anchors(cfg, tmp_path, demo_df, demo_csv):
    """Deleting a dataset removes its patterns, vectors, memberships and artefacts; an anchor survives
    only with a remaining member or a RELATED_TO link; the journal, state and snapshot stay consistent."""
    engine = make_engine(cfg)
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    other = tmp_path / "other.csv"  # same rows, other column names: its own metrics, dimensions and anchors
    demo_df.rename(columns=lambda c: f"x_{c}").to_csv(other, index=False)
    rec = engine.ingest_file(other)
    assert rec["status"] == "READY", rec.get("error")
    ds = rec["dataset_id"]
    g = engine.graph()
    before_members = Counter(e["target"] for e in g.edges if e["type"] == "ACTIVATES" and g.nodes[e["source"]]["props"]["dataset_id"] != ds)
    linked = {a for e in g.edges if e["type"] == "RELATED_TO" for a in (e["source"], e["target"])}
    anchors_before = {n["id"] for n in g.of_kind("Attractor")}
    assert any(t.startswith("x_") for t in engine.committed().catalog.texts)  # the literal cache now holds both datasets' literals

    out = engine.delete_dataset(ds)
    catalog = engine.committed().catalog  # the deletion's commit rewrites the cache
    cached = engine.ws.load_literals(engine.encoder.spec.fingerprint)
    assert set(cached) == set(catalog.texts) and not any(t.startswith("x_") for t in cached)  # its literals are gone
    assert out["patterns_removed"] == rec["metrics"]["validated_insights"] and out["batches"] == [rec["batch_id"]]
    g = engine.graph()
    assert all(n["props"]["dataset_id"] != ds for n in g.of_kind("Pattern"))
    assert not (cfg.workspace_dir / "datasets" / ds).exists() and not (cfg.workspace_dir / "journal" / "blocks" / f"{rec['batch_id']}.npz").exists()
    assert all(b["dataset_id"] != ds for b in engine.ws.list_batches())
    # the orphan rule, anchor by anchor
    anchors_after = {n["id"] for n in g.of_kind("Attractor")}
    for a in anchors_before:
        assert (a in anchors_after) == bool(before_members.get(a) or a in linked), a
    assert set(out["anchors_removed"]) == {int(a[2:]) for a in anchors_before - anchors_after}
    # journal, frame and ontology agree; the snapshot equals a fresh rebuild; questions still work
    rows = engine.ws.patterns()
    assert [r["row_id"] for r in rows] == list(range(len(rows))) == list(range(len(engine.ws.vectors())))
    assert engine.ontology().store.next_chunk_id == len(rows) and set(engine.frame().patterns) == {r["id"] for r in rows}
    assert all(a["row_id"] < len(rows) for a in engine.ws.activations())
    ids = {(n["id"], n["kind"]) for n in g.nodes.values()}
    assert {(n["id"], n["kind"]) for n in DualGraph(engine._derive(engine.ontology())[0]).nodes.values()} == ids
    assert engine.evidence("Why is margin lower for phones in the US?").view["evidence"]["items"]
    assert engine.ingest_file(other)["status"] == "READY"  # not a duplicate any more
    # deleting the last dataset empties the knowledge base, and ingestion starts over cleanly
    for b in {b["dataset_id"] for b in engine.ws.list_batches()}:
        engine.delete_dataset(b)
    assert not engine.graph().of_kind("Pattern") and not engine.ontology().attractor_ids and len(engine.ws.vectors()) == 0
    assert engine.ingest_file(demo_csv)["status"] == "READY"


def test_a_failed_ready_save_rolls_back_the_snapshot_too(cfg, tmp_path, demo_df, demo_csv, monkeypatch):
    """The READY record is the commit marker. If writing it fails (a full disk, a permission error), the batch rolls
    back completely: journal, state and the snapshot it had already written."""
    engine = make_engine(cfg)
    save = engine.ws.save_batch

    def flaky(record):
        if record["status"] == "READY":
            raise PermissionError(13, "Access is denied")
        save(record)

    monkeypatch.setattr(engine.ws, "save_batch", flaky)
    assert engine.ingest_file(demo_csv)["status"] == "FAILED"
    assert engine.ws.load_graph() is None and not engine.ws.patterns()  # first batch: no snapshot left behind
    monkeypatch.setattr(engine.ws, "save_batch", save)
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    before = {n["id"] for n in engine.ws.load_graph()["nodes"]}
    other = tmp_path / "other.csv"
    demo_df.rename(columns=lambda c: f"x_{c}").to_csv(other, index=False)
    monkeypatch.setattr(engine.ws, "save_batch", flaky)
    assert engine.ingest_file(other)["status"] == "FAILED"
    assert {n["id"] for n in engine.ws.load_graph()["nodes"]} == before  # the previous snapshot is back
    fresh = make_engine(cfg, recover=False)
    assert set(fresh.frame().patterns) == {n["id"] for n in fresh.graph().of_kind("Pattern")}


def test_an_interrupted_deletion_is_undone(cfg, tmp_path, demo_df, demo_csv, monkeypatch):
    """Deletion is all or nothing: an error rolls it back at once; a crash leaves its marker, and the next
    writer start rolls it back — journal, ontology, snapshot, batch records and dataset folder alike."""
    engine = make_engine(cfg)
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    other = tmp_path / "other.csv"
    demo_df.rename(columns=lambda c: f"x_{c}").to_csv(other, index=False)
    rec = engine.ingest_file(other)
    ds = rec["dataset_id"]

    def state():
        ont = engine.ontology()
        return (
            len(engine.ws.patterns()),
            len(engine.ws.vectors()),
            ont.store.next_chunk_id,
            ont.attractor_ids,
            sorted(n["id"] for n in engine.ws.load_graph()["nodes"]),
            sorted(b["batch_id"] for b in engine.ws.list_batches()),
            (cfg.workspace_dir / "datasets" / ds / "profile.json").is_file(),
        )

    before = state()
    for failure in (RuntimeError("disk full"), Crash()):
        monkeypatch.setattr(engine.ws, "save_graph", lambda snapshot, failure=failure: (_ for _ in ()).throw(failure))
        with pytest.raises(type(failure)):
            engine.delete_dataset(ds)
        monkeypatch.undo()
        if isinstance(failure, Crash):
            assert engine.ws.pending_path.exists() and state() != before  # half done, marker left
            engine = make_engine(cfg)  # the next writer start; the crashed process reads nothing more
        assert state() == before and not engine.ws.pending_path.exists()
    assert engine.delete_dataset(ds)["patterns_removed"] == rec["metrics"]["validated_insights"]


def test_a_new_version_supersedes_every_level_of_the_old_in_one_transaction(engine, tmp_path, monkeypatch):
    """``supersedes=<upload_id>``: the last level of the new version to be READY removes every level of the old in its own
    commit, and a commit that fails rolls the removal back with it. The pattern ids are those of the new version loaded
    alone, and the anchors the new version joins keep their ids."""
    v1, v2 = tmp_path / "points_v1.csv", tmp_path / "points_v2.csv"
    geo_frame(3).to_csv(v1, index=False)
    geo_frame(4).to_csv(v2, index=False)
    alone = make_engine(make_config(tmp_path / "alone"))
    assert [r["status"] for r in upload_levels(alone, v2, geo="auto@8,7")] == ["READY", "READY"]
    old = upload_levels(engine, v1, geo="auto@8,7")
    old_ids, old_upload = {r["dataset_id"] for r in old}, old[0]["upload_id"]
    journal = [p["id"] for p in engine.ws.patterns()]

    derive = engine._derive

    def crash(ontology, pending=None):
        if pending and "superseded" in pending:
            raise RuntimeError("crash after the removal, before the commit point")
        return derive(ontology, pending)

    monkeypatch.setattr(engine, "_derive", crash)
    failed = upload_levels(engine, v2, geo="auto@8,7", supersedes=old_upload)
    assert [r["status"] for r in failed] == ["READY", "FAILED"] and not {"superseded"} & {*failed[0], *failed[1]}  # 8 waits for 7
    assert all(engine.ws.ready_batch_for(ds) for ds in old_ids) and Path(old[0]["source_path"]).is_file()  # rolled back
    assert [p["id"] for p in engine.ws.patterns() if p["dataset_id"] in old_ids] == journal
    monkeypatch.undo()
    engine.delete_upload(failed[0]["upload_id"])

    before = {a["attractor_id"] for a in engine.ws.activations() if not a.get("weak")}  # every member is a row of v1
    assert {p["dataset_id"] for p in engine.ws.patterns()} == old_ids
    new = upload_levels(engine, v2, geo="auto@8,7", supersedes=old_upload)
    assert [r["status"] for r in new] == ["READY", "READY"] and new[1]["superseded"]["datasets"] == sorted(old_ids)
    assert {p["id"] for p in engine.ws.patterns()} == {p["id"] for p in alone.ws.patterns()}  # as if v2 came alone
    assert not any(engine.ws.load_batch(r["batch_id"]) for r in old) and not Path(old[0]["source_path"]).exists()
    rows = {p["row_id"] for p in engine.ws.patterns()}  # all of v2's now
    joined = {a["attractor_id"] for a in engine.ws.activations() if a["row_id"] in rows and not a.get("weak")}
    assert joined & before  # anchors only v1's rows held before kept their ids as v2's
    assert {n["props"].get("supersedes") for n in engine.graph().of_kind("Dataset")} == {old_upload}
    rows = engine.ws.patterns()  # journal, frame and ontology agree, and the snapshot equals a fresh rebuild
    assert [r["row_id"] for r in rows] == list(range(len(rows))) == list(range(len(engine.ws.vectors())))
    assert engine.ontology().store.next_chunk_id == len(rows) and all(a["row_id"] < len(rows) for a in engine.ws.activations())
    rebuilt = DualGraph(engine._derive(engine.ontology())[0])
    assert {(n["id"], n["kind"]) for n in rebuilt.nodes.values()} == {(n["id"], n["kind"]) for n in engine.graph().nodes.values()}


def test_patch_supersedes_deletes_the_old_version_without_mining(engine, tmp_path):
    """``PATCH supersedes`` on a READY dataset removes what it names at once: no new batch, the same pattern ids; a
    dataset cannot replace itself or its own upload, an unknown id or the very file it replaces is refused."""
    from fastapi.testclient import TestClient

    from insight_graph_service.server.app import create_app

    v1, v2 = tmp_path / "points_v1.csv", tmp_path / "points_v2.csv"
    geo_frame(3).to_csv(v1, index=False)
    geo_frame(4).to_csv(v2, index=False)
    with v1.open("rb") as f:
        queued = engine.upload(v1.name, f, geo="auto@8,7")
    old, new = [engine.process(queued[0]["batch_id"])], upload_levels(engine, v2, geo="auto@8,7")
    ds, mine = new[0]["dataset_id"], {p["id"] for p in engine.ws.patterns() if p["dataset_id"] in {r["dataset_id"] for r in new}}
    with TestClient(create_app(engine)) as client:
        for bad in (ds, new[0]["upload_id"], new[1]["dataset_id"], "ds-nope"):  # itself, its upload, its other level
            refused = client.patch(f"/api/datasets/{ds}", json={"supersedes": bad})
            assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
        busy = client.patch(f"/api/datasets/{ds}", json={"supersedes": old[0]["upload_id"]})  # its level 7 still queued
        assert busy.status_code == 409 and busy.json()["code"] == "busy" and engine.ws.load_batch(queued[1]["batch_id"])
        old.append(engine.process(queued[1]["batch_id"]))
        for target, geo in ((old[0]["upload_id"], ""), (old[0]["dataset_id"], "auto@6")):  # the same bytes: its upload, or a level of it
            with v1.open("rb") as f:
                same = client.post("/api/upload", files={"file": ("points_v1.csv", f, "text/csv")}, data={"geo": geo, "supersedes": target})
            assert same.status_code == 400 and len(list(engine.ws.uploads_dir.iterdir())) == 2  # not kept
        done = client.patch(f"/api/datasets/{ds}", json={"supersedes": old[0]["upload_id"]}).json()
    assert done["supersedes"] == old[0]["upload_id"] and done["superseded"]["datasets"] == sorted(r["dataset_id"] for r in old)
    assert {p["id"] for p in engine.ws.patterns()} == mine and len(engine.ws.list_batches()) == 2  # no mining, no new batch


def test_a_coarser_level_with_nothing_to_mine_still_completes_a_new_version(engine, tmp_path):
    """A last level that fails for its data (here too few cells at resolution 5) applies ``supersedes`` on its own."""
    v1, v2 = tmp_path / "points_v1.csv", tmp_path / "points_v2.csv"
    geo_frame(3).to_csv(v1, index=False)
    geo_frame(4).to_csv(v2, index=False)
    old = upload_levels(engine, v1, geo="auto@8,7")
    with v2.open("rb") as f:
        new = [engine.process(r["batch_id"]) for r in engine.upload(v2.name, f, geo="auto@8,5", supersedes=old[0]["upload_id"])]
    assert [(r["status"], (r["error"] or {}).get("code")) for r in new] == [("READY", None), ("FAILED", "invalid_schema")]
    assert new[1]["superseded"]["datasets"] == sorted(r["dataset_id"] for r in old) and "neo4j" not in new[1]["superseded"]
    assert {p["dataset_id"] for p in engine.ws.patterns()} == {new[0]["dataset_id"]}


def test_a_new_version_leaves_an_old_one_still_queued(engine, tmp_path):
    """The commit of a new version whose target still has a queued batch commits, and the old version stays (busy)."""
    v1, v2 = tmp_path / "points_v1.csv", tmp_path / "points_v2.csv"
    geo_frame(3).to_csv(v1, index=False)
    geo_frame(4).to_csv(v2, index=False)
    with v1.open("rb") as f:
        queued = engine.upload(v1.name, f, geo="auto@8,7")
    old = engine.process(queued[0]["batch_id"])
    with v2.open("rb") as f:
        (new,) = [engine.process(r["batch_id"]) for r in engine.upload(v2.name, f, geo="auto@8", supersedes=old["upload_id"])]
    assert new["status"] == "READY" and "superseded" not in new and any("still to run" in w for w in new["warnings"])
    assert engine.ws.ready_batch_for(old["dataset_id"]) and engine.ws.load_batch(queued[1]["batch_id"])["status"] == "UPLOADED"
