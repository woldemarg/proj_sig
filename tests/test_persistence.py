"""Persistence: write, reload, idempotency, rollback, recovery, graph consistency (SDD 09)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from conftest import FakeLLM, make_config

from ltir import pipeline as pipeline_mod
from ltir.neo4j_sink import edge_query, flatten_props, publish_snapshot
from ltir.pipeline import Engine


@pytest.fixture
def engine(tmp_path):
    return Engine(make_config(tmp_path / "ws"), llm=FakeLLM())


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
    again = Engine(make_config(tmp_path / "ws"), llm=FakeLLM())

    def ids(snap):
        return sorted(n["id"] for n in snap["nodes"]), sorted(e["id"] for e in snap["edges"])

    assert ids(again.graph().snapshot) == ids(engine.graph().snapshot)
    assert ids(again.rebuild_graph()) == ids(engine.graph().snapshot)
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
        if e["type"] == "RELATED_TO":
            assert kinds[e["source"]] == kinds[e["target"]] == "Attractor" and e["source"] != e["target"]
        if e["type"] in {"SPECIALIZES", "GENERALIZES", "SIBLING", "CONTRASTS"}:
            assert kinds[e["source"]] == kinds[e["target"]] == "Pattern"
    activated = {e["source"] for e in g.edges if e["type"] == "ACTIVATES"}
    assert activated == {nid for nid, k in kinds.items() if k == "Pattern"}
    assert {e["target"] for e in g.edges if e["type"] == "ACTIVATES"} == {nid for nid, k in kinds.items() if k == "Attractor"}
    spec = {(e["source"], e["target"]) for e in g.edges if e["type"] == "SPECIALIZES"}
    gen = {(e["target"], e["source"]) for e in g.edges if e["type"] == "GENERALIZES"}
    assert spec == gen and counts["SPECIALIZES"] > 0 and counts["CONTRASTS"] > 0 and counts["RELATED_TO"] > 0
    # mutual k-NN keeps the latent plane sparse
    assert counts["RELATED_TO"] <= hashed_engine.config.related_to_peer_count * len(g.of_kind("Attractor")) / 2
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

    monkeypatch.setattr(pipeline_mod, "build_snapshot", boom)
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
    other = Engine(make_config(tmp_path / "ws", block_weights=(1.0, 1.0, 1.0)), llm=FakeLLM())
    csv2 = tmp_path / "copy.csv"
    csv2.write_text(demo_csv.read_text().replace("\n", "\n", 1) + "\n")  # different bytes -> new dataset id
    rec = other.ingest_file(csv2)
    assert rec["status"] == "FAILED" and rec["error"]["code"] == "representation_mismatch"


def test_interrupted_batch_is_recovered(engine, demo_csv, tmp_path):
    rec = engine.submit(demo_csv)
    cp = engine.ws.checkpoint()
    rec.update(status="UPDATING_ONTOLOGY", checkpoint=cp)
    engine.ws.save_batch(rec)
    fresh = Engine(make_config(tmp_path / "ws"), llm=FakeLLM())
    after = fresh.ws.load_batch(rec["batch_id"])
    assert after["status"] == "FAILED" and after["error"]["code"] == "interrupted"


def test_migrate_rebuilds_an_outdated_workspace(tmp_path, demo_csv):
    from ltir.migrate import migrate_workspace
    from ltir.store import RepresentationMismatch, atomic_write_json

    cfg = make_config(tmp_path / "ws")
    old = Engine(cfg, llm=FakeLLM())
    rec = old.ingest_file(demo_csv)
    rep_path = old.ws.state_dir / "representation.json"
    atomic_write_json(rep_path, {**old.ws.representation(), "canonical_version": "ltir-canon-0"})  # an older build
    with pytest.raises(RepresentationMismatch):
        old.rebuild_graph()
    report = migrate_workspace(cfg)
    assert [b["status"] for b in report["batches"]] == ["READY"] and report["batches"][0]["dataset_id"] == rec["dataset_id"]
    fresh = Engine(cfg, llm=FakeLLM())
    assert fresh.ws.representation()["canonical_version"] != "ltir-canon-0" and fresh.rebuild_graph()["stats"]["patterns"] > 0
    backup = Path(report["backup"])
    assert json.loads((backup / "state" / "representation.json").read_text(encoding="utf-8"))["canonical_version"] == "ltir-canon-0"


def test_ingestion_failure_states(engine, tmp_path):
    bad = tmp_path / "data.xlsx"
    bad.write_bytes(b"not a table")
    assert engine.ingest_file(bad)["error"]["code"] == "unsupported_file"
    tiny = tmp_path / "tiny.csv"
    tiny.write_text("a,b\nx,1\ny,2\n")
    assert engine.ingest_file(tiny)["error"]["code"] == "invalid_schema"


def test_column_options_are_strict_when_explicit_and_lenient_as_defaults(tmp_path):
    from ltir.ingestion import IngestionError, load_dataset

    table = tmp_path / "t.csv"
    table.write_text("group,value\n" + "".join(f"{'ab'[i % 2]},{i}.5\n" for i in range(60)))
    cfg = make_config(tmp_path / "ws", categorical_columns="Store", bin_columns="price:4")
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


def test_neo4j_publish_is_merge_based(hashed_engine):
    snap = hashed_engine.ws.load_graph()
    driver = _Driver()
    out = publish_snapshot(snap, hashed_engine.config, driver=driver)
    queries = [q for q, _ in driver.log]
    assert any("CREATE CONSTRAINT pattern_id_unique" in q for q in queries)
    assert all("CREATE (" not in q for q in queries)  # idempotent: MERGE only
    assert out["nodes"]["Pattern"] == len(hashed_engine.graph().of_kind("Pattern"))
    assert "MERGE (s)-[r:ACTIVATES]->(t)" in edge_query("ACTIVATES")
    flat = flatten_props({"a": 1, "b": [1, 2], "c": {"x": 1}, "d": None, "e": [{"k": 1}]})
    assert flat == {
        "a": 1,
        "b": [1, 2],
        "c_json": json.dumps({"x": 1}),
        "e_json": json.dumps([{"k": 1}]),
    }
