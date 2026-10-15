"""Persistence: write, reload, idempotency, rollback, recovery, graph consistency (docs/06_graph_and_storage.md)."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from conftest import make_config, make_engine

from graph_query_engine.graph import DualGraph
from insight_contracts import LATENT_EDGES, EdgeType, GraphEdge, record_digest
from insight_graph_service.core import engine as engine_mod
from insight_graph_service.core.neo4j_mirror import (
    EDGE_ENDPOINTS,
    LABELS,
    edge_query,
    flatten_props,
    load_cypher,
    node_query,
    publish_snapshot,
    stale_edge_query,
    stale_node_query,
)
from insight_graph_service.core.snapshot import co_occurrence_edges


@pytest.fixture
def engine(tmp_path):
    return make_engine(make_config(tmp_path / "ws"))


def test_write_reload_and_idempotency(engine, demo_csv, tmp_path):
    rec = engine.ingest_file(demo_csv)
    assert rec["status"] == "READY"
    ws = engine.ws
    n = rec["metrics"]["validated_insights"]
    assert len(ws.patterns()) == n == ws.vectors().shape[0] == len({a["pattern_id"] for a in ws.activations()})
    for rel in (
        "state/concepts.npz",
        "state/state.json",
        "state/representation.json",
        "graph/snapshot.json",
        f"datasets/{rec['dataset_id']}/covers.npz",
        f"datasets/{rec['dataset_id']}/profile.json",
    ):
        assert (ws.root / rel).is_file(), rel

    # reload from disk in a fresh engine -> identical graph; rebuild reproduces it
    again = make_engine(make_config(tmp_path / "ws"))

    def ids(snap):
        return sorted(n["id"] for n in snap["nodes"]), sorted(e["id"] for e in snap["edges"])

    assert ids(again.graph().snapshot) == ids(engine.graph().snapshot)
    assert ids(again._derive(again.ontology())[0]) == ids(engine.graph().snapshot)
    assert again.ontology().store.next_chunk_id == n

    # idempotent re-ingestion of identical content
    dup = again.ingest_file(demo_csv)
    assert dup["status"] == "SKIPPED" and dup["duplicate_of"] == rec["batch_id"]
    assert len(again.ws.patterns()) == n


def test_graph_consistency(hashed_engine):
    g = hashed_engine.graph()
    kinds = {nid: n["kind"] for nid, n in g.nodes.items()}
    counts = Counter(e["type"] for e in g.edges)
    for e in g.edges:
        if e["type"] == "ACTIVATES":
            assert kinds[e["source"]] == "Pattern" and kinds[e["target"]] == "Attractor"
        if e["type"] in {"RELATED_TO", "CO_OCCURS"}:
            assert kinds[e["source"]] == kinds[e["target"]] == "Attractor" and e["source"] != e["target"] and 0 < e["weight"] <= 1
        if e["type"] in {"SPECIALIZES", "GENERALIZES", "SIBLING", "CONTRASTS"}:
            assert kinds[e["source"]] == kinds[e["target"]] == "Pattern"
    activated = {e["source"] for e in g.edges if e["type"] == "ACTIVATES"}
    assert activated == {nid for nid, k in kinds.items() if k == "Pattern"}
    # one predicate for coverage and retrieval: a membership below the ontology's floor is weak (not walked)
    floor = hashed_engine.settings.topology.min_activation_alignment
    assert all(e["props"]["weak"] or e["weight"] >= floor for e in g.edges if e["type"] == "ACTIVATES")
    assert {e["target"] for e in g.edges if e["type"] == "ACTIVATES"} == {nid for nid, k in kinds.items() if k == "Attractor"}
    spec = {(e["source"], e["target"]) for e in g.edges if e["type"] == "SPECIALIZES"}
    gen = {(e["target"], e["source"]) for e in g.edges if e["type"] == "GENERALIZES"}
    assert spec == gen and counts["SPECIALIZES"] > 0 and counts["CONTRASTS"] > 0 and counts["RELATED_TO"] > 0
    # mutual k-NN keeps the latent plane sparse
    assert counts["RELATED_TO"] <= hashed_engine.settings.topology.related_to_peer_count * len(g.of_kind("Attractor")) / 2
    # a co-occurrence link needs a member insight on both sides
    members = {e["target"] for e in g.edges if e["type"] == "ACTIVATES" and not e["props"]["weak"]}
    assert all({e["source"], e["target"]} <= members for e in g.edges if e["type"] == "CO_OCCURS")
    # Pattern nodes are the journal record, rehydrated as Insight (one weight, one condition shape).
    journal = {r["id"]: r for r in hashed_engine.ws.patterns()}
    assert set(g.insights) == set(journal)
    for pid, ins in g.insights.items():
        props = g.nodes[pid]["props"]
        assert props["weight"] == ins.weight == journal[pid]["weight"]
        assert props["conditions"] == journal[pid]["conditions"]
        assert props["shifts"] == journal[pid]["shifts"]
        assert "insight_weight" not in props
        assert ins.conditions and ins.conditions[0].expr


def test_failed_batch_rolls_back(engine, demo_csv, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated graph failure")

    monkeypatch.setattr(engine_mod, "build_snapshot", boom)
    rec = engine.ingest_file(demo_csv)
    assert rec["status"] == "FAILED" and rec["error"]["code"] == "internal_error"
    assert rec["failed_stage"] == "PERSISTING"
    assert engine.ws.patterns() == [] and engine.ws.vectors().size == 0
    assert engine.ontology().store.is_empty and not (engine.ws.root / "checkpoint").exists()
    monkeypatch.undo()
    ok = engine.ingest_file(demo_csv)
    assert ok["status"] == "READY"  # nothing half-written blocks the retry


def test_committed_batch_stays_ready_when_publish_save_fails(engine, demo_csv, monkeypatch):
    real = engine.ws.save_batch

    def save(record):
        if record.get("status") == "READY" and "neo4j" in record:
            raise OSError("disk full")
        real(record)

    monkeypatch.setattr(engine.ws, "save_batch", save)
    with pytest.raises(OSError, match="disk full"):
        engine.ingest_file(demo_csv)
    saved = engine.ws.list_batches()
    assert len(saved) == 1 and saved[0]["status"] == "READY"
    assert engine.ws.patterns()


def test_representation_mismatch_is_refused(engine, demo_csv, tmp_path):
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    other = make_engine(make_config(tmp_path / "ws", block_weights=(1.0, 1.0, 1.0)))
    csv2 = tmp_path / "copy.csv"
    csv2.write_text(demo_csv.read_text().replace("\n", "\n", 1) + "\n")  # different bytes -> new dataset id
    rec = other.ingest_file(csv2)
    assert rec["status"] == "FAILED" and rec["error"]["code"] == "representation_mismatch"


class Crash(BaseException):  # the process dies: no except-handler runs
    pass


def test_interrupted_batch_is_recovered(engine, demo_csv, tmp_path, monkeypatch):
    """A batch that dies after writing its journal leaves the transaction marker; the next writer start rolls
    it back and fails the batch, and the same upload then commits."""
    monkeypatch.setattr(engine_mod, "build_snapshot", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        engine.ingest_file(demo_csv)
    monkeypatch.undo()
    assert engine.ws.patterns() and engine.ws.pending_path.exists()  # half written
    fresh = make_engine(make_config(tmp_path / "ws"))
    (after,) = fresh.ws.list_batches()
    assert after["status"] == "FAILED" and after["error"]["code"] == "interrupted"
    assert not fresh.ws.patterns() and not fresh.ws.pending_path.exists() and fresh.ontology().store.is_empty
    assert fresh.ingest_file(demo_csv)["status"] == "READY"


def test_a_failed_rollback_blocks_writes_until_recovered(engine, demo_csv, tmp_path, monkeypatch):
    """A rollback that fails keeps its marker and checkpoint: no later write may overwrite them, and the next
    writer start finishes the rollback."""
    monkeypatch.setattr(engine_mod, "build_snapshot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("graph failure")))
    monkeypatch.setattr(engine.ws, "rollback", lambda cp: (_ for _ in ()).throw(OSError("disk gone")))
    rec = engine.ingest_file(demo_csv)
    assert rec["status"] == "FAILED" and rec["error"]["message"] == "graph failure" and "rollback failed" in rec["warnings"][-1]
    monkeypatch.undo()
    assert engine.ws.pending_path.exists() and engine.ws.patterns()  # the evidence stays
    assert engine.ingest_file(demo_csv)["error"]["code"] == "rollback_pending"
    fresh = make_engine(make_config(tmp_path / "ws"))
    assert not fresh.ws.patterns() and not fresh.ws.pending_path.exists()
    assert fresh.ingest_file(demo_csv)["status"] == "READY"


def test_reset_is_refused_while_a_batch_runs(engine, demo_csv, monkeypatch):
    """A reset during a batch answers ``busy`` at once instead of waiting for the batch lock (the web request
    would hang for the whole batch); readers keep the committed state meanwhile and the batch still commits."""
    import threading

    started, release = threading.Event(), threading.Event()
    discover = engine_mod.run_discovery

    def held(*args, **kwargs):
        started.set()
        release.wait(timeout=30)  # bounded: a reset that waited would see the batch finish and fail the assertion
        return discover(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "run_discovery", held)
    batch = engine.submit(demo_csv)
    worker = threading.Thread(target=engine.process, args=(batch["batch_id"],))
    worker.start()
    try:
        assert started.wait(timeout=30)
        with pytest.raises(engine_mod.PipelineError, match="is running") as refused:
            engine.reset()
        assert refused.value.code == "busy" and not engine.graph().insights  # the committed (empty) state, read without a lock
    finally:
        release.set()
        worker.join(timeout=120)
    assert engine.ws.load_batch(batch["batch_id"])["status"] == "READY" and engine.graph().insights
    engine.reset()
    assert not engine.graph().insights and not engine.frame().patterns and not engine.ws.list_batches()


def test_delete_dataset_removes_its_knowledge_and_orphan_anchors(tmp_path, demo_df, demo_csv):
    """Deleting a dataset removes its patterns, vectors, memberships and artefacts; an anchor survives
    only with a remaining member or a RELATED_TO link; the journal, state and snapshot stay consistent."""
    cfg = make_config(tmp_path / "ws")
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


def test_a_failed_ready_save_rolls_back_the_snapshot_too(tmp_path, demo_df, demo_csv, monkeypatch):
    """The READY record is the commit marker. If writing it fails (a full disk, a permission error), the batch rolls
    back completely: journal, state and the snapshot it had already written."""
    cfg = make_config(tmp_path / "ws")
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


def test_an_interrupted_deletion_is_undone(tmp_path, demo_df, demo_csv, monkeypatch):
    """Deletion is all or nothing: an error rolls it back at once; a crash leaves its marker, and the next
    writer start rolls it back — journal, ontology, snapshot, batch records and dataset folder alike."""
    cfg = make_config(tmp_path / "ws")
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


def test_a_second_writer_process_is_refused(tmp_path, demo_csv):
    """One writer per workspace: a second writer started while the service holds the lock is refused
    and leaves the service's queued upload alone; readers stay allowed."""
    import os
    import subprocess
    import sys

    from insight_graph_service.core.settings import PROJECT_ROOT
    from insight_graph_service.core.workspace import WorkspaceBusy

    cfg = make_config(tmp_path / "ws")
    queued = make_engine(cfg, recover=False).submit(demo_csv)  # an upload waiting in the web app's queue
    holder = subprocess.Popen(  # the web app: another process holding the writer lock
        [
            sys.executable,
            "-c",
            "import sys; from insight_graph_service.core.workspace import acquire_writer_lock; acquire_writer_lock(sys.argv[1]); print('locked', flush=True); sys.stdin.read()",
            str(cfg.workspace_dir),
        ],
        cwd=PROJECT_ROOT,
        env=os.environ,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(WorkspaceBusy, match="in use by another writer process"):
            make_engine(cfg)
        reader = make_engine(cfg, recover=False)
        assert reader.ws.load_batch(queued["batch_id"])["status"] == "UPLOADED"
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)
    # the holder is gone: the next writer gets the lock and recovers the orphaned upload
    make_engine(cfg)
    assert make_engine(cfg, recover=False).ws.load_batch(queued["batch_id"])["error"]["code"] == "interrupted"


def test_an_outdated_workspace_starts_degraded_and_reset_recovers(tmp_path, demo_csv):
    """Journals of another canonical version cannot be served: the engine starts on the empty state with the
    reason, refuses writes and questions, never touches the Neo4j mirror, and a reset makes it usable again."""
    from insight_graph_service.core.workspace import atomic_write_json

    cfg = make_config(tmp_path / "ws")
    old = make_engine(cfg)
    assert old.ingest_file(demo_csv)["status"] == "READY"
    atomic_write_json(old.ws.state_dir / "representation.json", {**old.ws.representation(), "canonical_version": "ltir-canon-0"})
    engine = make_engine(cfg)
    assert "ltir-canon-0" in engine.problem and not engine.graph().insights
    for write in (lambda: engine.submit(demo_csv), lambda: engine.evidence("margin?"), lambda: engine.delete_dataset("x")):
        with pytest.raises(engine_mod.PipelineError, match="POST /api/reset") as refused:
            write()
        assert refused.value.code == "workspace_degraded"
    published = []
    engine.sync_neo4j = lambda snapshot=None: published.append(snapshot) or {"status": "ok"}
    assert engine.startup_sync() == {"status": "skipped"} and published == []  # a degraded start never wipes the mirror
    engine.reset()
    assert engine.problem is None and engine.ingest_file(demo_csv)["status"] == "READY"
    assert engine.startup_sync()["status"] == "ok" and published[-1]["stats"]["patterns"] > 0  # a healthy start syncs


def test_a_changed_journal_number_is_caught_and_a_record_without_digest_is_unverified(tmp_path, demo_csv):
    """A p-value edited in the journal no longer matches its digest: after the rebuild the pattern is ``mismatch`` and
    left out of the evidence. A record without a digest (journaled before digests) reads ``unverified``, not degraded."""
    cfg = make_config(tmp_path / "ws")
    first = make_engine(cfg)
    assert first.ingest_file(demo_csv)["status"] == "READY"
    question = "Why is margin lower for phones in the US?"
    forged, old = (it.pattern_id for it in first.search(question).evidence.items[:2])
    rows = first.ws.patterns()
    assert {first.graph().nodes[r["id"]]["props"]["integrity"] for r in rows} == {"ok"}
    for field, value in (("emm_score", 9.9), ("support_fraction", 0.9), ("drivers", ["x"]), ("phenomenon_type", "covariance"), ("effect_size", -1.0)):
        assert record_digest({**rows[0], field: value}) != rows[0]["record_digest"], field  # every number the prompt shows is covered
    assert record_digest({**rows[0], "row_id": 999, "provenance": {}}) == rows[0]["record_digest"]  # what a deletion rewrites is not
    assert record_digest({**rows[0], "retired_field": 1}) != rows[0]["record_digest"]  # the record's own keys, not the class's
    for r in rows:
        if r["id"] == forged:
            r["p_adjusted"] = 0.5
        if r["id"] == old:
            del r["record_digest"]
    first.ws.journal.chunks_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    first.ws.save_graph({**first.ws.load_graph(), "version": 3})  # a snapshot of the previous version: rebuilt at start

    engine = make_engine(cfg)
    assert engine.problem is None
    props = {pid: engine.graph().nodes[pid]["props"]["integrity"] for pid in (forged, old)}
    assert props == {forged: "mismatch", old: "unverified"}
    found = engine.search(question)
    evidence = found.evidence
    assert forged not in {it.pattern_id for it in evidence.items} | {s.pattern_id for s in found.seeds}
    assert any(n.startswith(f"{forged} excluded: integrity check failed") for n in evidence.notes)
    assert next(it for it in evidence.items if it.pattern_id == old).validation.endswith("integrity unverified")
    assert "integrity unverified" in engine.evidence(question).evidence_prompt


def test_a_failed_recovery_starts_degraded_and_reset_still_works(tmp_path, demo_csv, monkeypatch):
    """A rollback that cannot finish leaves the queued uploads failed (no process runs them), so the reset that
    recovers is never refused as busy."""
    from insight_graph_service.core.workspace import Workspace

    cfg = make_config(tmp_path / "ws")
    queued = make_engine(cfg).submit(demo_csv)  # left UPLOADED by a process that died
    monkeypatch.setattr(Workspace, "recover", lambda self: (_ for _ in ()).throw(OSError("disk gone")))
    engine = make_engine(cfg)
    assert "disk gone" in engine.problem and engine.startup_sync() == {"status": "skipped"}
    assert engine.ws.load_batch(queued["batch_id"])["error"]["code"] == "interrupted"
    with pytest.raises(engine_mod.PipelineError, match="workspace_degraded|POST /api/reset"):
        engine.submit(demo_csv)
    monkeypatch.undo()
    engine.reset()
    assert engine.problem is None and engine.ingest_file(demo_csv)["status"] == "READY"


def test_an_unreadable_snapshot_starts_degraded(tmp_path, demo_csv):
    cfg = make_config(tmp_path / "ws")
    assert make_engine(cfg).ingest_file(demo_csv)["status"] == "READY"
    (cfg.workspace_dir / "graph" / "snapshot.json").write_text("{not json", encoding="utf-8")
    engine = make_engine(cfg)
    assert engine.problem and not engine.graph().insights  # started: the reset stays reachable
    engine.reset()
    assert engine.problem is None and not list((tmp_path).glob("ws.deleting-*"))  # renamed aside, then deleted


def test_refusals_over_http(hashed_engine, tmp_path, demo_csv):
    """The graph service's status mapping: 413 too large, 404 unknown dataset or batch, a column id with '/' resolves."""
    from fastapi.testclient import TestClient

    from insight_graph_service.server.app import create_app

    with TestClient(create_app(hashed_engine)) as client:
        assert client.delete("/api/datasets/nope").json()["code"] == "unknown_dataset"
        assert client.get("/api/batches/..%5Cregistry").status_code == 404
        column = next(n["id"] for n in hashed_engine.graph().nodes.values() if n["kind"] == "Metric")
        assert client.get(f"/api/nodes/{column}").status_code == 200
    small = make_engine(make_config(tmp_path / "small", max_upload_mb=0))
    with TestClient(create_app(small)) as client, open(demo_csv, "rb") as f:
        refused = client.post("/api/upload", files={"file": ("demo.csv", f, "text/csv")})
    assert refused.status_code == 413 and refused.json()["code"] == "file_too_large"


def test_dataset_source_and_reliability(engine, demo_csv):
    """Source and reliability are upload options outside the dataset id; PATCH changes them on the batch record and
    recompiles the snapshot (no new batch, the same pattern ids); the prompt names them, quoted, only once declared."""
    from fastapi.testclient import TestClient

    from insight_graph_service.server.app import create_app
    from subgroup_miner.ingestion import load_dataset

    question = "Why is margin lower for phones in the US?"
    record = engine.process(engine.submit(demo_csv, source=" field\nreports ", reliability="b")["batch_id"])
    assert record["status"] == "READY" and (record["source"], record["reliability"]) == ("field reports", "B")
    ds = record["dataset_id"]
    assert load_dataset(demo_csv, engine.settings.miner).dataset_id == ds  # the options are not part of the dataset id
    patterns = sorted(p["id"] for p in engine.ws.patterns())
    assert 'source="field reports" reliability=B (usually reliable)' in engine.evidence(question).evidence_prompt
    with TestClient(create_app(engine)) as client:
        changed = client.patch(f"/api/datasets/{ds}", json={"reliability": "e"}).json()
        assert (changed["source"], changed["reliability"]) == ("field reports", "E")
        assert client.patch(f"/api/datasets/{ds}", json={"reliability": "Z"}).json()["code"] == "invalid_options"
        assert client.patch("/api/datasets/nope", json={"source": "x"}).status_code == 404
        stored = sorted(engine.ws.uploads_dir.glob("*"))
        with open(demo_csv, "rb") as f:
            refused = client.post("/api/upload", files={"file": ("demo.csv", f, "text/csv")}, data={"reliability": "G"})
        assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
        assert sorted(engine.ws.uploads_dir.glob("*")) == stored  # refused before the file is kept
    assert len(engine.ws.list_batches()) == 1 and sorted(p["id"] for p in engine.ws.patterns()) == patterns
    props = engine.graph().nodes[f"DS:{ds}"]["props"]
    assert (props["source"], props["reliability"]) == ("field reports", "E")
    assert "reliability=E (unreliable)" in engine.evidence(question).evidence_prompt
    again = engine.process(engine.submit(demo_csv, source="other")["batch_id"])
    assert again["status"] == "SKIPPED" and any("PATCH /api/datasets/" in w for w in again["warnings"])  # a re-upload changes nothing
    engine.update_dataset_metadata(ds, source="", reliability="")  # undeclared again: no source in the prompt
    assert "source=" not in engine.evidence(question).evidence_prompt and engine.graph().nodes[f"DS:{ds}"]["props"]["reliability"] == "F"


def test_indicator_registry(engine, demo_csv):
    """What each metric measures: an upload option outside the dataset id (an unknown metric only warns), on the Metric
    nodes and, quoted, in the prompt's baselines; PATCH replaces it strictly, without a new batch or new pattern ids."""
    from fastapi.testclient import TestClient

    from insight_graph_service.server.app import create_app

    question = "Why is margin lower for phones in the US?"
    record = engine.process(engine.submit(demo_csv, indicators="margin: gross margin :%, discount:discount_rate, nosuch:x")["batch_id"])
    assert record["status"] == "READY" and any("nosuch" in w for w in record["warnings"]) and "nosuch" not in record["indicators"]
    again = engine.process(engine.submit(demo_csv, indicators="margin:other")["batch_id"])
    assert any("PATCH /api/datasets/" in w for w in again["warnings"])  # a re-upload never changes the registry
    ds, patterns = record["dataset_id"], sorted(p["id"] for p in engine.ws.patterns())
    margin = engine.graph().nodes[f"M:{ds}:margin"]["props"]
    assert (margin["indicator"], margin["unit"]) == ("gross margin", "%") and "unit" not in engine.graph().nodes[f"M:{ds}:discount"]["props"]
    assert '- margin (indicator "gross margin", unit "%"): median' in engine.evidence(question).evidence_prompt
    with TestClient(create_app(engine)) as client:
        assert client.patch(f"/api/datasets/{ds}", json={"indicators": "margin:net_margin"}).json()["indicators"] == {
            "margin": {"indicator": "net_margin"}
        }
        assert client.patch(f"/api/datasets/{ds}", json={"indicators": "nosuch:x"}).json()["code"] == "invalid_options"
        assert client.patch(f"/api/datasets/{ds}", json={"indicators": "margin"}).status_code == 400
    assert [b["status"] for b in engine.ws.list_batches()].count("READY") == 1 and sorted(p["id"] for p in engine.ws.patterns()) == patterns
    assert "indicator" not in engine.graph().nodes[f"M:{ds}:discount"]["props"]  # PATCH replaces the registry
    engine.update_dataset_metadata(ds, indicators="")
    assert "(indicator" not in engine.evidence(question).evidence_prompt


def test_ingestion_failure_states(engine, tmp_path):
    bad = tmp_path / "data.xlsx"
    bad.write_bytes(b"not a table")
    assert engine.ingest_file(bad)["error"]["code"] == "unsupported_file"
    tiny = tmp_path / "tiny.csv"
    tiny.write_text("a,b\nx,1\ny,2\n")
    assert engine.ingest_file(tiny)["error"]["code"] == "invalid_schema"


def test_column_options_are_strict_when_explicit_and_lenient_as_defaults(tmp_path):
    from subgroup_miner.ingestion import IngestionError, load_dataset

    table = tmp_path / "t.csv"
    table.write_text("group,value\n" + "".join(f"{'ab'[i % 2]},{i}.5\n" for i in range(60)))
    cfg = make_config(tmp_path / "ws", categorical_columns="Store", bin_columns="price:4").miner
    loaded = load_dataset(table, cfg)  # workspace defaults: absent columns are skipped with a warning
    assert loaded.categories == [] and loaded.bins == {}
    assert any("Store" in w and "price" in w for w in loaded.warnings)
    for option in ({"categories": "Store"}, {"bins": "price:4"}):  # an explicit option must name real columns
        with pytest.raises(IngestionError) as exc:
            load_dataset(table, cfg, **option)
        assert exc.value.code == "invalid_options"


class _Session:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        self.log.append((query, params))

    def execute_write(self, fn):
        fn(self)


class _Driver:
    def __init__(self):
        self.log = []

    def session(self, database=None):
        return _Session(self.log)


def test_neo4j_mirror_equals_the_snapshot(hashed_engine):
    snap = hashed_engine.ws.load_graph()
    driver = _Driver()
    out = publish_snapshot(snap, hashed_engine.settings, driver=driver)
    queries = [q for q, _ in driver.log]
    assert any("CREATE CONSTRAINT pattern_id_unique" in q for q in queries)
    assert all("CREATE (" not in q for q in queries)  # idempotent: MERGE only
    assert out["nodes"]["Pattern"] == len(hashed_engine.graph().of_kind("Pattern"))
    assert "MERGE (s)-[r:ACTIVATES]->(t)" in edge_query("ACTIVATES")
    # properties are replaced, not merged: a property that disappeared (or became None) is removed
    assert "SET n = row.props, n.id = row.id" in node_query("Pattern") and "SET r = row.props" in edge_query("RELATED_TO")
    # what the snapshot no longer holds is deleted: every label and every edge type, keyed by the snapshot ids
    stale = {q: p for q, p in driver.log if "NOT" in q}
    assert {q for q in stale if "DETACH DELETE" in q} == {stale_node_query(label) for label in LABELS}
    assert sorted(stale[stale_node_query("Pattern")]["ids"]) == sorted(n["id"] for n in hashed_engine.graph().of_kind("Pattern"))
    assert {q for q in stale if q.endswith("DELETE r")} == {stale_edge_query(t) for t in EDGE_ENDPOINTS}
    # an empty snapshot (a reset) clears the mirror
    empty = _Driver()
    publish_snapshot({"nodes": [], "edges": []}, hashed_engine.settings, driver=empty)
    assert all(p["ids"] == [] for q, p in empty.log if "DETACH DELETE" in q)
    flat = flatten_props({"a": 1, "b": [1, 2], "c": {"x": 1}, "d": None, "e": [{"k": 1}]})
    assert flat == {
        "a": 1,
        "b": [1, 2],
        "c_json": json.dumps({"x": 1}),
        "e_json": json.dumps([{"k": 1}]),
    }


def test_co_occurrence_edges_from_memberships():
    """W[p, j] = a_j^2 / sum_k a_k over the non-weak alignments; weak memberships do not count; the pair weight is capped at 1."""

    def act(pid, aid, alignment, weak=False):
        return GraphEdge(pid, f"A-{aid}", EdgeType.ACTIVATES, alignment, {"weak": weak})

    edges = co_occurrence_edges([act("P1", 0, 0.9), act("P1", 1, 0.6), act("P2", 0, 0.8), act("P2", 1, 0.1, weak=True), act("P3", 1, 0.95)])
    assert [(e.source, e.target, e.type) for e in edges] == [("A-0", "A-1", EdgeType.CO_OCCURS)]
    assert edges[0].weight == pytest.approx((0.9**2 / 1.5) * (0.6**2 / 1.5)) and edges[0].props == {"kind": "co_occurrence", "shared": 1}
    many = [act(f"P{i}", j, 1.0) for i in range(5) for j in (0, 1)]  # five insights split evenly: 5 x 0.25
    assert [(e.weight, e.props["shared"]) for e in co_occurrence_edges(many)] == [(1.0, 5)]


def test_cypher_walks_the_latent_edge_set():
    query = load_cypher("queries/transversal")
    assert "[rel:RELATED_TO|CO_OCCURS]" in query and all(e.value in query for e in LATENT_EDGES)
