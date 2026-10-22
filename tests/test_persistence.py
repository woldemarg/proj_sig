"""Persistence: write, reload, idempotency, rollback, recovery, graph consistency (docs/06_graph_and_storage.md)."""

from __future__ import annotations

import json
import time
from collections import Counter

import pytest
from conftest import Crash, make_config, make_engine, upload_levels
from fastapi.testclient import TestClient

from insight_graph_service.core import engine as engine_mod
from insight_graph_service.server.app import create_app


def test_write_reload_and_idempotency(engine, demo_csv):
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
    again = make_engine(engine.settings)

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
        if e["type"] in {"SPECIALIZES", "SIBLING", "CONTRASTS"}:
            assert kinds[e["source"]] == kinds[e["target"]] == "Pattern"
    activated = {e["source"] for e in g.edges if e["type"] == "ACTIVATES"}
    assert activated == {nid for nid, k in kinds.items() if k == "Pattern"}
    # one predicate for coverage and retrieval: a membership below the ontology's floor is weak (not walked)
    floor = hashed_engine.settings.topology.min_activation_alignment
    assert all(e["props"]["weak"] or e["weight"] >= floor for e in g.edges if e["type"] == "ACTIVATES")
    assert {e["target"] for e in g.edges if e["type"] == "ACTIVATES"} == {nid for nid, k in kinds.items() if k == "Attractor"}
    assert "GENERALIZES" not in counts and counts["SPECIALIZES"] > 0 and counts["CONTRASTS"] > 0 and counts["RELATED_TO"] > 0
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


def test_a_transaction_marker_that_cannot_be_removed(engine, demo_csv, monkeypatch):
    """Removing a transaction's marker can fail (a file another process holds open). A batch committed at its READY record:
    it stays READY and is published, and the next writer start closes the marker. A deletion commits when its marker goes:
    it fails, readers keep the dataset, and the next writer start rolls it back."""

    def locked(self, cp):
        raise PermissionError("pending.json is held open")

    monkeypatch.setattr(type(engine.ws), "_close", locked)
    rec = engine.ingest_file(demo_csv)
    assert rec["status"] == "READY" and engine.graph().insights
    assert engine.ws.pending_path.exists()  # waits for the next writer start
    monkeypatch.undo()
    again = make_engine(engine.settings)
    assert again.problem is None and not again.ws.pending_path.exists()
    assert [b["status"] for b in again.ws.list_batches()] == ["READY"] and again.graph().insights
    monkeypatch.setattr(type(again.ws), "_close", locked)
    with pytest.raises(PermissionError):
        again.delete_dataset(rec["dataset_id"])
    assert again.graph().insights and again.ws.pending_path.exists()  # not published; the marker waits for the next start
    monkeypatch.undo()
    last = make_engine(engine.settings)
    assert last.problem is None and not last.ws.pending_path.exists()
    assert [b["status"] for b in last.ws.list_batches()] == ["READY"] and last.graph().insights  # rolled back


def test_a_batch_reads_ready_once_readers_see_its_graph(engine, demo_csv, monkeypatch):
    """READY is on disk a moment before readers switch to the new graph (the commit point, then ``_publish``): a poll in
    that moment reads PERSISTING, so a client that acts on READY — the console's one reload — never reads the old graph."""
    seen, publish = [], engine._publish

    def spy(state, snapshot):  # the moment between the commit point and the switch
        bid = engine.ws.list_batches()[-1]["batch_id"]  # the newest
        polls = [b["status"] for b in client.get("/api/batches").json()] + [client.get(f"/api/batches/{bid}").json()["status"]]
        seen.append((engine.ws.load_batch(bid)["status"], polls, len(engine.graph().insights)))
        return publish(state, snapshot)

    monkeypatch.setattr(engine, "_publish", spy)
    with TestClient(create_app(engine)) as client:
        rec = engine.ingest_file(demo_csv)
        after = client.get(f"/api/batches/{rec['batch_id']}").json()["status"]
    assert seen == [("READY", ["PERSISTING", "PERSISTING"], 0)]  # READY on disk, readers still on the empty graph
    assert after == "READY" and engine.graph().insights


def test_representation_mismatch_is_refused(engine, demo_csv, tmp_path):
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    other = make_engine(make_config(tmp_path / "ws", block_weights=(1.0, 1.0, 1.0)))
    csv2 = tmp_path / "copy.csv"
    csv2.write_text(demo_csv.read_text().replace("\n", "\n", 1) + "\n")  # different bytes -> new dataset id
    rec = other.ingest_file(csv2)
    assert rec["status"] == "FAILED" and rec["error"]["code"] == "representation_mismatch"
    literals = other.ws.literals_path.read_bytes()
    assert make_engine(make_config(tmp_path / "ws", block_weights=(1.0, 1.0, 1.0))).committed().catalog is None  # no cache built or saved
    assert other.ws.literals_path.read_bytes() == literals


def test_a_model_failure_at_start_leaves_the_workspace_usable(engine, demo_csv):
    """The literal catalog is built when the state is loaded; if the model fails there, the engine still starts, not degraded."""
    from doubles import HashingEmbedder

    class Broken(HashingEmbedder):
        def embed(self, texts):
            raise RuntimeError("CUDA out of memory")

    assert engine.ingest_file(demo_csv)["status"] == "READY"
    engine.ws.literals_path.unlink()  # the start must embed the literals
    broken = make_engine(engine.settings, embedder=Broken())
    assert broken.problem is None and broken.graph().insights and broken.committed().catalog is None


def test_a_catalog_failure_at_the_commit_is_a_batch_warning(engine, demo_csv, monkeypatch):
    monkeypatch.setattr(engine_mod, "build_catalog", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model gone")))
    record = engine.ingest_file(demo_csv)
    assert record["status"] == "READY" and any("literal catalog not built" in w for w in record["warnings"])
    assert engine.committed().catalog is None and engine.search("Why is margin lower for phones in the US?").evidence.items  # built on the question


def test_interrupted_batch_is_recovered(engine, demo_csv, monkeypatch):
    """A batch that dies after writing its journal leaves the transaction marker; the next writer start rolls
    it back and fails the batch, and the same upload then commits."""
    monkeypatch.setattr(engine_mod, "build_snapshot", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        engine.ingest_file(demo_csv)
    monkeypatch.undo()
    assert engine.ws.patterns() and engine.ws.pending_path.exists()  # half written
    fresh = make_engine(engine.settings)
    (after,) = fresh.ws.list_batches()
    assert after["status"] == "FAILED" and after["error"]["code"] == "interrupted"
    assert not fresh.ws.patterns() and not fresh.ws.pending_path.exists() and fresh.ontology().store.is_empty
    assert fresh.ingest_file(demo_csv)["status"] == "READY"


def test_a_failed_rollback_blocks_writes_until_recovered(engine, demo_csv, monkeypatch):
    """A rollback that fails keeps its marker and checkpoint: no later write may overwrite them, and the next
    writer start finishes the rollback."""
    monkeypatch.setattr(engine_mod, "build_snapshot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("graph failure")))
    monkeypatch.setattr(engine.ws, "rollback", lambda cp: (_ for _ in ()).throw(OSError("disk gone")))
    rec = engine.ingest_file(demo_csv)
    assert rec["status"] == "FAILED" and rec["error"]["message"] == "graph failure" and "rollback failed" in rec["warnings"][-1]
    monkeypatch.undo()
    assert engine.ws.pending_path.exists() and engine.ws.patterns()  # the evidence stays
    assert engine.ingest_file(demo_csv)["error"]["code"] == "rollback_pending"
    fresh = make_engine(engine.settings)
    assert not fresh.ws.patterns() and not fresh.ws.pending_path.exists()
    assert fresh.ingest_file(demo_csv)["status"] == "READY"


def test_a_commit_gives_readers_a_new_graph_and_the_old_one_keeps_what_it_derived(engine, demo_csv, demo_df, tmp_path):
    """Read views are memoised on the committed graph: the next commit's graph derives them afresh, with the new
    dataset, while a reader that still holds the previous graph keeps its own."""
    from insight_graph_service.server.views import cytoscape_elements

    engine.ingest_file(demo_csv)
    old = engine.graph()
    before = cytoscape_elements(old)
    shuffled = tmp_path / "shuffled.csv"
    demo_df.sample(frac=1, random_state=1).to_csv(shuffled, index=False)
    added = engine.ingest_file(shuffled)["dataset_id"]
    new = engine.graph()
    after = cytoscape_elements(new)
    assert new is not old and len(after["nodes"]) > len(before["nodes"])
    assert any(n["data"].get("dataset") == added for n in after["nodes"]) and cytoscape_elements(old) is before


def test_writes_are_refused_while_a_batch_runs(engine, demo_csv, demo_df, tmp_path, monkeypatch):
    """A reset, a deletion and a PATCH of another dataset during a batch answer ``busy`` at once instead of waiting
    for the batch lock (the web request would hang for the whole batch); readers keep the committed state meanwhile
    and the batch still commits."""
    import threading

    shuffled = tmp_path / "shuffled.csv"
    demo_df.sample(frac=1, random_state=1).to_csv(shuffled, index=False)
    ready = engine.ingest_file(shuffled)["dataset_id"]
    before = len(engine.graph().insights)

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
        for write in (engine.reset, lambda: engine.delete_dataset(ready), lambda: engine.update_dataset_metadata(ready, source="x")):
            t0 = time.perf_counter()
            with pytest.raises(engine_mod.PipelineError, match="is running") as refused:
                write()
            assert refused.value.code == "busy" and time.perf_counter() - t0 < 1
        assert len(engine.graph().insights) == before  # the committed state, read without a lock
    finally:
        release.set()
        worker.join(timeout=120)
    assert engine.ws.load_batch(batch["batch_id"])["status"] == "READY" and engine.ws.ready_batch_for(ready)
    engine.reset()
    assert not engine.graph().insights and not engine.frame().patterns and not engine.ws.list_batches()


def test_a_second_writer_process_is_refused(cfg, demo_csv):
    """One writer per workspace: a second writer started while the service holds the lock is refused
    and leaves the service's queued upload alone; readers stay allowed."""
    import os
    import subprocess
    import sys

    from insight_graph_service.core.settings import PROJECT_ROOT
    from insight_graph_service.core.workspace import WorkspaceBusy

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


def test_an_outdated_workspace_starts_degraded_and_reset_recovers(cfg, demo_csv):
    """Journals of another canonical version cannot be served: the engine starts on the empty state with the
    reason, refuses writes and questions, never touches the Neo4j mirror, and a reset makes it usable again."""
    from insight_graph_service.core.workspace import atomic_write_json

    old = make_engine(cfg)
    assert old.ingest_file(demo_csv)["status"] == "READY"
    atomic_write_json(old.ws.state_dir / "representation.json", {**old.ws.representation(), "canonical_version": "ltir-canon-0"})
    engine = make_engine(cfg)
    assert "ltir-canon-0" in engine.problem and not engine.graph().insights
    reads = (engine.export_geosparql, engine.geo_datasets, lambda: engine.geo_overlay("x"), lambda: engine.geo_cells_pack("x"))  # the map too
    for write in (lambda: engine.submit(demo_csv), lambda: engine.evidence("margin?"), lambda: engine.delete_dataset("x"), *reads):
        with pytest.raises(engine_mod.PipelineError, match="POST /api/reset") as refused:
            write()
        assert refused.value.code == "workspace_degraded"
    published = []
    engine.sync_neo4j = lambda snapshot=None: published.append(snapshot) or {"status": "ok"}
    assert engine.startup_sync() == {"status": "skipped"} and published == []  # a degraded start never wipes the mirror
    engine.reset()
    assert engine.problem is None and engine.ingest_file(demo_csv)["status"] == "READY"
    assert engine.startup_sync()["status"] == "ok" and published[-1]["stats"]["patterns"] > 0  # a healthy start syncs


def test_a_failed_recovery_starts_degraded_and_reset_still_works(cfg, demo_csv, monkeypatch):
    """A rollback that cannot finish leaves the queued uploads failed (no process runs them), so the reset that
    recovers is never refused as busy."""
    from insight_graph_service.core.workspace import Workspace

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


def test_an_unreadable_snapshot_is_rebuilt_at_start(cfg, demo_csv):
    """The snapshot is derived: a start rebuilds it from the journals whatever the file holds."""
    first = make_engine(cfg)
    assert first.ingest_file(demo_csv)["status"] == "READY"
    (cfg.workspace_dir / "graph" / "snapshot.json").write_text("{not json", encoding="utf-8")
    engine = make_engine(cfg)
    assert engine.problem is None and set(engine.graph().insights) == set(first.graph().insights) and engine.ws.load_graph()["nodes"]


def test_base_places_are_rebuilt_identically_at_start(cfg, points_csv):
    """A start derives the snapshot from the journals and the stored files, whatever the snapshot file holds: a table
    level's IN_REGION by its events (never journaled) come back with the same props."""
    first = make_engine(cfg)
    upload_levels(first, points_csv, geo="auto@8,7", table=True)

    def placed(graph) -> list[tuple[str, str, float, str]]:
        edges = [e for e in graph.edges if e["type"] == "IN_REGION" and e["props"]["basis"] == "events"]
        return sorted((e["source"], e["target"], e["weight"], json.dumps(e["props"], sort_keys=True)) for e in edges)

    before = placed(first.graph())
    path = cfg.workspace_dir / "graph" / "snapshot.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**old, "version": old["version"] - 1, "edges": []}), encoding="utf-8")
    again = make_engine(cfg)
    assert again.problem is None and before and placed(again.graph()) == before


def test_refusals_over_http(hashed_engine, tmp_path, demo_csv):
    """The graph service's status mapping: 413 too large, 404 unknown dataset or batch, a column id with '/' resolves."""
    with TestClient(create_app(hashed_engine)) as client:
        assert client.delete("/api/datasets/nope").json()["code"] == "unknown_dataset"
        assert client.get("/api/batches/..%5Cregistry").status_code == 404
        column = next(n["id"] for n in hashed_engine.graph().nodes.values() if n["kind"] == "Metric")
        assert client.get(f"/api/nodes/{column}").status_code == 200
    small = make_engine(make_config(tmp_path / "small", max_upload_mb=0))
    with TestClient(create_app(small)) as client, open(demo_csv, "rb") as f:
        refused = client.post("/api/upload", files={"file": ("demo.csv", f, "text/csv")})
    assert refused.status_code == 413 and refused.json()["code"] == "file_too_large"


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


def test_the_c_parser_reads_a_file_as_pandas_python_parser_did(demo_csv, points_csv, tmp_path):
    """``read_file`` gives the frame pandas' Python parser gave — columns, types and values, one type per column from all
    its rows — and the dataset id the bytes always gave."""
    import hashlib

    import pandas as pd

    from insight_graph_service.core.settings import PROJECT_ROOT
    from subgroup_miner import ingestion

    semicolons = tmp_path / "semi.txt"
    semicolons.write_text('city;value;note\n"Kyiv";0.1234567890123456789;"a; b"\nLviv;1e-310;c\n', encoding="utf-8")
    mixed = tmp_path / "mixed.csv"  # a code column numeric-looking for its first 300,000 rows
    mixed.write_text("code,value\n" + "".join(f"{i % 900:05d},{i}\n" for i in range(300_000)) + "x00042,1\n")
    for path in (demo_csv, points_csv, PROJECT_ROOT / "data" / "demo" / "california_housing.csv", semicolons, mixed):
        pd.testing.assert_frame_equal(ingestion.read_file(path)[0], pd.read_csv(path, sep=None, engine="python"), check_exact=True)
    assert (
        ingestion.dataset_fingerprint(ingestion.read_file(points_csv)[1], {})
        == "ds-" + hashlib.sha256(points_csv.read_bytes() + b"[]").hexdigest()[:12]
    )


def test_a_file_read_otherwise_than_before_gets_its_own_dataset_id(tmp_path):
    """A BOM is dropped from the first name and a TSV whose chunks typed a column apart is read whole; such a file's id
    mixes in ``READING_FIX`` (its old dataset reads ``source_changed``, a new upload is no duplicate), any other keeps its
    id, and a TSV whose text columns hold one type each is parsed once."""
    import hashlib
    from unittest import mock

    import pandas as pd

    from subgroup_miner import ingestion

    def plain_id(path, fix=b""):
        return "ds-" + hashlib.sha256(path.read_bytes() + fix + b"[]").hexdigest()[:12]

    def read(path):
        frame, digest = ingestion.read_file(path)
        return frame, ingestion.dataset_fingerprint(digest, {})

    for i, header in enumerate(["region,value", '"region",value', ",value", "a,a", '"region";"value"']):  # plain, quoted, empty, repeated, ;
        bare = tmp_path / f"bare{i}.csv"
        bare.write_text(f"{header}\nEU,1\nUS,2\n".replace(",", ";" if ";" in header else ","), encoding="utf-8")
        bom = tmp_path / f"bom{i}.csv"
        bom.write_bytes(b"\xef\xbb\xbf" + bare.read_bytes())
        frame, ds = read(bom)
        pd.testing.assert_frame_equal(frame, pd.read_csv(bare, sep=None, engine="python"), check_exact=True)  # as without the BOM
        assert ds == plain_id(bom, ingestion.READING_FIX) != plain_id(bom) and read(bare)[1] == plain_id(bare)
    mixed = tmp_path / "mixed.tsv"  # a code column numeric-looking for its first 300,000 rows
    mixed.write_text("code\tvalue\n" + "".join(f"{i}\t{i}\n" for i in range(300_000)) + "x42\t1\n")
    with mock.patch.object(pd, "read_csv", wraps=pd.read_csv) as parse:
        frame, ds = read(mixed)
    assert frame["code"].map(type).eq(str).all() and ds == plain_id(mixed, ingestion.READING_FIX) and parse.call_count == 2
    unsigned = tmp_path / "unsigned.tsv"  # an int64 chunk, then a uint64 one: their concatenation is a lossy float64, no warning
    unsigned.write_text("code\tvalue\n" + "".join(f"c{i % 9}\t{i}\n" for i in range(300_000)) + f"c0\t{2**63}\n")
    frame, ds = read(unsigned)
    assert frame["value"].dtype == "uint64" and ds == plain_id(unsigned, ingestion.READING_FIX)
    clean = tmp_path / "clean.tsv"
    clean.write_bytes(b"\xef\xbb\xbfregion\tflag\tvalue\nEU\tTrue\t1\nUS\t\t2\n")  # a BOM the C parser always dropped, an object column
    with mock.patch.object(pd, "read_csv", wraps=pd.read_csv) as parse:
        frame, ds = read(clean)
    assert list(frame.columns) == ["region", "flag", "value"] and ds == plain_id(clean) and parse.call_count == 1  # one type per column: once


def test_an_upload_parses_its_file_once_and_a_recheck_reads_it_anew(cfg, points_csv, demo_csv):
    """The ``@auto`` choice, each level and each dataset id of one upload read one parse, gone once no batch waits and
    after a refused upload; a re-check reads the file as it is now: an edit of the same length under the old mtime is
    ``source_changed``, a file gone ``unsupported_file``."""
    import os
    from pathlib import Path
    from unittest import mock

    import pandas as pd

    from subgroup_miner import ingestion

    engine = make_engine(cfg)
    ingestion.forget_tables()
    with mock.patch.object(ingestion.pd, "read_csv", wraps=pd.read_csv) as parse:
        levels = upload_levels(engine, points_csv, geo="auto@8,7", table=True)
    assert len(levels) == 3 and all(r["status"] == "READY" for r in levels)
    assert parse.call_count == 1 and ingestion._kept.cache_info().currsize == 0  # one parse, then gone
    with demo_csv.open("rb") as f, pytest.raises(engine_mod.PipelineError):
        engine.upload("demo.csv", f, geo="nowhere@auto")  # no such point columns: refused
    assert ingestion._kept.cache_info().currsize == 0
    table = next(r for r in levels if r["level"] == "table")
    pid = next(p for p, ins in engine.graph().insights.items() if ins.dataset_id == table["dataset_id"])
    assert engine.verify_pattern(pid)["status"] == "reproduced"
    stored = Path(table["source_path"])
    st, raw = stored.stat(), stored.read_bytes()
    stored.write_bytes(raw.replace(b"4", b"5", 1))  # the same length, put back under the old mtime
    os.utime(stored, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert engine.verify_pattern(pid)["status"] == "source_changed"
    with pytest.raises(ingestion.IngestionError, match="not found") as gone:
        ingestion.read_file(stored.with_name("gone.csv"))
    assert gone.value.code == "unsupported_file"


def test_a_commit_derives_only_the_new_datasets_spatial_part(cfg, points_csv):
    """Each geo dataset's part of the spatial plane is derived once, at its own commit, and a table's placement once:
    the snapshot equals one derived afresh, is written compact, hands out copies the memo does not share, and a
    deletion drops what it held from the memo, a reset all of it."""
    from unittest import mock

    from subgroup_miner import spatial

    engine = make_engine(cfg)
    with (
        mock.patch.object(spatial, "_dataset_part", wraps=spatial._dataset_part) as part,
        mock.patch.object(spatial, "place_table", wraps=spatial.place_table) as place,
    ):
        levels = upload_levels(engine, points_csv, geo="auto@8,7", table=True)
    assert [r["status"] for r in levels] == ["READY"] * 3 and part.call_count == 2 and place.call_count == 1

    def derived() -> dict:  # the snapshot but its build time
        return {k: v for k, v in engine._derive(engine.ontology())[0].items() if k != "created_at"}

    kept = {k: v for k, v in engine.graph().snapshot.items() if k != "created_at"}
    for e in engine._derive(engine.ontology())[0]["edges"]:  # what a commit hands out is a copy: the memo stays as built
        e["props"]["touched"] = True
    assert derived() == kept
    engine._spatial_memo.clear()
    assert derived() == kept and "\n" not in engine.ws.graph_path.read_text(encoding="utf-8")
    table = next(r["dataset_id"] for r in levels if r["level"] == "table")
    engine.delete_dataset(table)
    assert not any(table in key for key in engine._spatial_memo) and len(engine._spatial_memo) == 2
    engine.reset()
    assert not engine._spatial_memo
