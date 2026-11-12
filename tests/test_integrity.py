"""Integrity and trust: a pattern recomputed from its stored upload, edited records caught, the declared source, reliability
and indicators, credibility by anchor and by place (docs/06_graph_and_storage.md §6.2, §6.7, docs/13_trust.md)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_engine, rewrite_patterns, stale_snapshot, upload_levels
from fastapi.testclient import TestClient

from insight_contracts import EdgeType, GraphEdge, record_digest
from insight_graph_service.core.engine import PipelineError
from insight_graph_service.server.app import create_app
from insight_graph_service.server.views import node_details


def test_a_pattern_is_recomputed_from_its_stored_upload(cfg, tmp_path, demo_csv):
    """``verify_pattern`` (docs/06_graph_and_storage.md §6.7): every demo pattern recomputes as journaled; forged with
    their digests recomputed, so that they pass the integrity check, a p-value, a test family, a dropped p, a selector and a
    target each differ, field by field, as does a shift order whose first shift is not the target, its p recomputed for
    that first shift; a missing covers.npz changes nothing (the conditions decide the rows) and is named as not checked; a
    dropped shift is not caught (the metrics are not re-mined: ``not_checked``); a changed upload is ``source_changed``, a
    removed one ``source_missing``; an unknown id is refused."""
    import shutil

    from subgroup_miner.discovery import recompute_statistics

    own = tmp_path / "demo.csv"  # ingest_file keeps the path it is given: this test changes and removes the file
    shutil.copy(demo_csv, own)
    engine = make_engine(cfg)
    ds = engine.ingest_file(own)["dataset_id"]
    rows = engine.ws.patterns()
    assert {engine.verify_pattern(r["id"])["status"] for r in rows} == {"reproduced"}  # covariance insights included
    tested = [r for r in rows if r["p_value"] is not None and len(r["shifts"]) > 1]
    relabelled = next(r for r in rows if r["phenomenon_type"] == "covariance")
    relabelled["target"] = next(m for m in relabelled["covariance"]["pair"] if m != relabelled["target"])  # the weaker of its pair
    relabelled["record_digest"] = record_digest(relabelled)
    forged, family, dropped, selector, target, shift, reordered = tested[:7]
    loaded = engine._load(engine.ws.ready_batch_for(ds))
    reordered["shifts"][:2] = reordered["shifts"][1::-1]  # its target second: its p now the test of another metric
    again = recompute_statistics(
        loaded.frame,
        [(c["attribute"], c["value"]) for c in reordered["conditions"]],
        reordered["expression"],
        [s["metric"] for s in reordered["shifts"]],
        loaded.spatial,
        reordered["provenance"]["multiple_testing_family"],
        engine.settings.miner,
    )
    reordered["p_value"], reordered["p_adjusted"] = again["p_value"], again["p_adjusted"]
    forged["p_value"], forged["p_adjusted"] = 0.0123, 0.0123 * forged["provenance"]["multiple_testing_family"]  # a demo p may be 0.0
    family["provenance"]["multiple_testing_family"] //= 100
    family["p_adjusted"] = min(1.0, family["p_value"] * family["provenance"]["multiple_testing_family"])
    dropped["p_value"] = dropped["p_adjusted"] = None  # the console would say "not tested"
    selector["expression"] = "region=='MOON'"
    target["target"] = "no_such_metric"
    shift["shifts"] = shift["shifts"][:1]
    for r in (forged, family, dropped, selector, target, shift, reordered):
        r["record_digest"] = record_digest(r)  # a writer who recomputes the digest
    rewrite_patterns(engine, rows)
    stale_snapshot(engine)
    engine = make_engine(cfg)
    assert engine.graph().nodes[forged["id"]]["props"]["integrity"] == "ok"
    checked = engine.verify_pattern(forged["id"])
    assert checked["status"] == "differs" and {d["field"] for d in checked["diff"]} == {"p_value", "p_adjusted"}
    assert next(d for d in checked["diff"] if d["field"] == "p_value")["journal"] == forged["p_value"]
    caught = {r["id"]: {d["field"] for d in engine.verify_pattern(r["id"])["diff"]} for r in (family, dropped, selector, target, reordered)}
    assert caught == {
        family["id"]: {"multiple_testing_family"},
        dropped["id"]: {"p_value"},
        selector["id"]: {"expression"},
        target["id"]: {"target"},
        reordered["id"]: {"target"},
    }
    assert engine.verify_pattern(shift["id"])["status"] == "reproduced" and "the metrics shifted (mining is not re-run)" in checked["not_checked"]
    assert "target" in {d["field"] for d in engine.verify_pattern(relabelled["id"])["diff"]}  # selection picks the stronger metric
    (engine.ws.datasets_dir / ds / "covers.npz").unlink()
    uncovered = engine.verify_pattern(next(r["id"] for r in rows if r not in [*tested[:7], relabelled]))
    assert uncovered["status"] == "reproduced" and "its rows in covers.npz (none stored)" in uncovered["not_checked"]
    other = next(r["id"] for r in rows if r["id"] != forged["id"])
    source = Path(engine.ws.ready_batch_for(ds)["source_path"])
    source.write_bytes(source.read_bytes() + b"\n")  # one byte more
    assert engine.verify_pattern(other)["status"] == "source_changed"
    source.unlink()
    assert engine.verify_pattern(other)["status"] == "source_missing"
    with pytest.raises(PipelineError) as refused:
        engine.verify_pattern("P-nope")
    assert refused.value.code == "unknown_pattern"


def test_a_changed_batch_record_is_caught_and_its_metadata_ignored(cfg, demo_csv):
    """The READY record signs what the graph reads of its dataset, as stored, and a PATCH signs it again (``ok``); a rating
    edited in the record is ``mismatch`` after the rebuild: its source, rating and registry are ignored, the evidence says
    so on the trust line, in the summary and in a NOTE, and a PATCH refuses to sign it; a record without a digest is
    ``unverified``, not degraded, and its trust lines say so."""
    import numpy as np

    from insight_graph_service.core.snapshot import metadata_digest
    from insight_graph_service.server.views import cytoscape_elements

    engine = make_engine(cfg)
    ds = engine.ingest_file(demo_csv)["dataset_id"]
    record = engine.ws.ready_batch_for(ds)
    assert record["metadata_digest"] == metadata_digest(record)
    engine.update_dataset_metadata(ds, source="Field reports", reliability="E", indicators="margin:gross margin")
    assert engine.graph().nodes[f"DS:{ds}"]["props"]["integrity"] == "ok"
    for field, value in (("source", "x"), ("filename", "y.csv"), ("profile", {}), ("indicators", {})):
        assert metadata_digest({**engine.ws.ready_batch_for(ds), field: value}) != engine.ws.ready_batch_for(ds)["metadata_digest"], field

    forged = engine.ws.ready_batch_for(ds)
    engine.ws.save_batch({**forged, "reliability": "A"})  # by hand, past the API; the snapshot on disk still says ok
    engine = make_engine(cfg)  # an ordinary restart rebuilds the snapshot
    props = engine.graph().nodes[f"DS:{ds}"]["props"]
    assert props["integrity"] == "mismatch" and "source" not in props and props["reliability"] == "F"
    assert "indicator" not in engine.graph().nodes[f"M:{ds}:margin"]["props"]  # the registry is ignored too
    assert cytoscape_elements(engine.graph())["datasets"][ds] == {"reliability": "F", "integrity": "mismatch"}  # what the rail's card shows
    found = engine.search("Why is margin lower for phones in the US?")
    assert all(it.trust.startswith("source unknown (its dataset's metadata failed the integrity check)") for it in found.evidence.items)
    assert any(ds in n and "its metadata no longer matches its digest" in n for n in found.evidence.notes)
    assert "- довіра всіх свідчень: джерело невідоме" in found.payload().evidence_summary  # every item shares it: one line
    assert f"- метадані датасету {ds} не пройшли перевірку цілісності" in found.payload().evidence_summary  # as the NOTE says
    with pytest.raises(PipelineError) as refused:
        engine.update_dataset_metadata(ds, indicators="")  # any PATCH would sign the forged rating as genuine
    assert refused.value.code == "metadata_mismatch"

    record = engine.ws.ready_batch_for(ds)
    engine.ws.save_batch({k: v for k, v in record.items() if k != "metadata_digest"})  # journaled before digests
    stale_snapshot(engine)
    engine = make_engine(cfg)
    assert engine.problem is None and engine.graph().nodes[f"DS:{ds}"]["props"]["integrity"] == "unverified"
    trust = engine.search("Why is margin lower for phones in the US?").evidence.items[0].trust
    assert trust.startswith("source A (completely reliable), metadata unverified")  # seen, though not refused
    assert metadata_digest({"profile": {"n": np.int64(3), 7: "x"}}) == metadata_digest({"profile": {"n": "3", "7": "x"}})  # as stored


def test_a_changed_batch_record_places_no_table_level(cfg, points_csv):
    """A table level whose batch record fails its metadata digest (its upload id, its point columns or its row count may
    be forged) is not placed by its events: its declarations are ignored like a forged rating. The unsigned locator of
    the options moves nothing: the signed profile names the point columns."""
    engine = make_engine(cfg)
    done = upload_levels(engine, points_csv, geo="auto@7", table=True)

    def base_edges(graph) -> int:
        return sum(e["type"] == "IN_REGION" and e["props"]["basis"] == "events" for e in graph.edges)

    placed = base_edges(engine.graph())
    table = engine.ws.ready_batch_for(done[1]["dataset_id"])
    assert placed and table["profile"]["points"] == "latitude,longitude"
    engine.ws.save_batch({**table, "locator": "nowhere"})  # outside the digest, and read by nothing once signed
    assert base_edges(make_engine(cfg).graph()) == placed
    engine.ws.save_batch({**table, "upload_id": "up-000000000000"})  # past the API
    assert base_edges(make_engine(cfg).graph()) == 0


def test_a_changed_journal_number_is_caught_and_a_record_without_digest_is_unverified(cfg, demo_csv):
    """A p-value edited in the journal no longer matches its digest: after the rebuild the pattern is ``mismatch`` and
    left out of the evidence. A record without a digest (journaled before digests) reads ``unverified``, not degraded."""
    first = make_engine(cfg)
    assert first.ingest_file(demo_csv)["status"] == "READY"
    question = "Why is margin lower for phones in the US?"
    forged, old = (it.pattern_id for it in first.search(question).evidence.items[:2])
    rows = first.ws.patterns()
    assert {first.graph().nodes[r["id"]]["props"]["integrity"] for r in rows} == {"ok"}
    for field, value in (("emm_score", 9.9), ("support_fraction", 0.9), ("drivers", ["x"]), ("phenomenon_type", "covariance"), ("effect_size", -1.0)):
        assert record_digest({**rows[0], field: value}) != rows[0]["record_digest"], field  # every number the prompt shows is covered
    for field, value in (
        ("canonical", {**rows[0]["canonical"], "components": [["fraud rate", 9.0]]}),
        ("embedding", {**rows[0]["embedding"], "dim": 1}),
    ):
        assert record_digest({**rows[0], field: value}) != rows[0]["record_digest"], field  # written once: the anchor labels read the components
    assert record_digest({**rows[0], "row_id": 999, "provenance": {}}) == rows[0]["record_digest"]  # what a deletion rewrites is not
    assert record_digest({**rows[0], "retired_field": 1}) != rows[0]["record_digest"]  # the record's own keys, not the class's
    for r in rows:
        if r["id"] == forged:
            r["p_adjusted"] = 0.5
        if r["id"] == old:
            del r["record_digest"]
    rewrite_patterns(first, rows)
    snapshot = first.ws.load_graph()  # its trust edited too: a start serves the journal's, never the file's
    for node in snapshot["nodes"]:
        if node["id"] == forged:
            node["props"] |= {"integrity": "ok", "credibility": 1}
    first.ws.save_graph(snapshot)

    engine = make_engine(cfg)
    assert engine.problem is None
    props = {pid: engine.graph().nodes[pid]["props"]["integrity"] for pid in (forged, old)}
    assert props == {forged: "mismatch", old: "unverified"}
    found = engine.search(question)
    evidence = found.evidence
    assert forged not in {it.pattern_id for it in evidence.items} | {s.pattern_id for s in found.seeds}
    assert any(n.startswith(f"{forged} excluded: integrity check failed") for n in evidence.notes)
    assert forged in evidence.excluded and forged not in {r.node_id for r in found.traversal.patterns}  # no slot of the budget
    assert f"виключено, перевірку цілісності не пройдено (статистика не збігається з дайджестом запису): {forged}" in found.payload().evidence_summary
    assert engine.graph().nodes[forged]["props"]["credibility"] == first.graph().nodes[forged]["props"]["credibility"]  # recomputed
    graph = engine.graph()
    for anchor in {e["target"] for e in graph.edges if e["type"] == "ACTIVATES" and e["source"] == forged}:
        members = {e["source"] for e in graph.edges if e["type"] == "ACTIVATES" and e["target"] == anchor}
        assert graph.nodes[anchor]["props"]["n_patterns"] == len(members - {forged})  # nor does it describe its anchors
    assert next(it for it in evidence.items if it.pattern_id == old).validation.endswith("integrity unverified")
    assert "integrity unverified" in engine.evidence(question).evidence_prompt


def test_a_changed_geo_record_is_not_drawn_on_the_map(cfg, points_csv):
    """A geo insight whose journal record fails its digest is not drawn: its points are refused, as the projection and the
    window already skip it, while an intact one of the same map is served (M9AC-F02); its neighbours' panels mark it."""
    first = make_engine(cfg)
    ds = first.process(first.submit(points_csv, geo="auto")["batch_id"])["dataset_id"]
    rows = first.ws.patterns()
    forged, intact = next(
        (e["source"], e["target"]) for e in first.graph().edges if e["type"] in ("CONTRASTS", "SPECIALIZES", "SIBLING", "CO_LOCATED")
    )
    next(r for r in rows if r["id"] == forged)["p_adjusted"] = 0.5
    rewrite_patterns(first, rows)
    engine = make_engine(cfg)
    assert engine.graph().nodes[forged]["props"]["integrity"] == "mismatch"
    with pytest.raises(PipelineError, match="no intact pattern"):
        engine.geo_pattern(ds, forged)
    assert engine.geo_pattern(ds, intact)["n_points"] > 0
    assert forged not in {i["id"] for t in engine.geo_window(ds, "-180,-90,180,90")["themes"] for i in t["insights"]}
    assert engine.geo_overlay(ds)["patterns"][forged]["integrity"] == "mismatch"  # what the console's map reads to leave it out
    listed = [
        it for group in node_details(engine.graph(), intact, engine.settings.thresholds)["neighbors"].values() for it in group if it["id"] == forged
    ]
    assert listed and {it["integrity"] for it in listed} == {"mismatch"}  # its neighbour's Walk on leaves it out (M9AC-F07)


def test_dataset_source_and_reliability(engine, demo_csv):
    """Source and reliability are upload options outside the dataset id; PATCH changes them on the batch record and
    recompiles the snapshot (no new batch, the same pattern ids); the prompt names them, quoted, only once declared."""
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


def test_credibility_by_place_phase_b():
    """Phase B (docs/06_graph_and_storage.md §6.2), for a pattern with a place, its cells compared at the finest resolution
    both datasets have: another declared source confirms (1) where one place lies mostly inside the other and contradicts (5, in any
    subgroup) only over a comparable place; elsewhere it confirms nothing, and the place is 2 when another dataset holds
    events over at least GEO_REGION_MIN_SHARE of it, else 6. The levels of one upload are one source; no registry is
    nominal 2; units, when both give one, must agree."""
    from conftest import toy_insight

    from insight_contracts import PhenomenonThresholds, Shift
    from insight_graph_service.core.snapshot import credibility
    from subgroup_miner.spatial_plane import SpatialStructure

    def ins(pid, ds, z, scope=(("lisa", "high"),)):
        return toy_insight(list(scope), [Shift("load", z, 10.0, 12.0, 1.0)], id=pid, dataset_id=ds, batch_id=f"b-{ds}")

    def member(pid, anchor="A-1"):
        return GraphEdge(pid, anchor, EdgeType.ACTIVATES, 0.9, {"weak": False})

    def batch(ds, source, unit=None, **more):
        entry = {"indicator": "Load", **({"unit": unit} if unit else {})}
        return {"dataset_id": ds, "status": "READY", "source": source, "indicators": {"load": entry}, **more}

    batches = {f"b-{ds}": batch(ds, src) for ds, src in (("a", "a"), ("b", "b"), ("c", "c"))}
    batches |= {"b-l5": batch("l5", "a", upload_id="up-1"), "b-l7": batch("l7", "a", upload_id="up-1")}
    batches |= {"b-n": {"dataset_id": "n", "status": "READY", "source": "n"}, "b-usd": batch("usd", "usd", "USD")}
    here = [f"R:7:h{i}" for i in range(6)] + ["R:5:H"]

    def at(ids):  # "R:<res>:<cell>" ids -> {res: cells}
        out = {}
        for i in ids:
            out.setdefault(int(i.split(":")[1]), set()).add(i)
        return {res: frozenset(cells) for res, cells in out.items()}

    def levels(insights, places, occupied, edges=None):
        spatial = SpatialStructure(places={p: at(ids) for p, ids in places.items()}, occupied={d: at(ids) for d, ids in occupied.items()})
        out = credibility(insights, batches, edges or [member(i.id) for i in insights], set(), PhenomenonThresholds(), spatial)
        return {pid: t["credibility"] for pid, t in out.items()}, out

    a, b = ins("A", "a", -1.5), ins("B", "b", -1.2, scope=(("mode", "x"),))
    lv, out = levels([a, b], {"A": here, "B": here[:2]}, {"a": set(here), "b": set(here[:2])})
    assert lv == {"A": 1, "B": 1}
    assert out["A"]["credibility_basis"] == {"rule": "place", "patterns": ["B"], "sources": ["b"], "comparability": "indicator"}
    lv, out = levels([a, b], {"A": here, "B": ["R:7:far", "R:5:H"]}, {"b": set(here)})  # b's pattern is elsewhere, its events here
    assert lv["A"] == 2 and out["A"]["credibility_basis"] == {"rule": "place", "patterns": [], "sources": ["b"], "comparability": "nominal"}
    assert levels([a, b], {"A": here, "B": ["R:7:far", "R:5:H"]}, {"b": {"R:7:far"}})[0]["A"] == 6  # phase A would say 1
    assert levels([a, b], {"A": here, "B": ["R:7:far", "R:5:H"]}, {"b": {"R:7:h0"}})[0]["A"] == 6  # 1 of 6 cells: not covered
    assert levels([a, b], {"A": here, "B": ["R:5:H"]}, {"b": {"R:5:H"}})[0]["A"] == 1  # b is coarser: compared at level 5
    spot = [f"R:8:f{i}" for i in range(7)] + ["R:7:X"]
    sparse = ["R:8:f0", *(f"R:8:s{i}" for i in range(48)), "R:7:X", *(f"R:7:y{i}" for i in range(48))]  # one cell of the spot
    assert levels([a, b], {"A": spot, "B": sparse}, {"b": set(sparse)})[0]["A"] == 6  # one shared region at 7 is not the same place
    lv, _ = levels([a, ins("C", "c", +1.4, scope=(("mode", "y"),))], {"A": here, "C": here}, {"c": set(here)}, [member("A"), member("C", "A-2")])
    assert lv == {"A": 5, "C": 5}  # the same place the other way, another subgroup: a contradiction
    wide = here + [f"R:7:w{i}" for i in range(30)]
    lv, _ = levels([a, ins("C", "c", +1.4, scope=(("mode", "y"),))], {"A": here, "C": wide}, {"c": set(wide)}, [member("A"), member("C", "A-2")])
    assert lv == {"A": 2, "C": 2}  # a wide place's average says nothing against a spot inside it
    assert levels([a, ins("W", "c", -1.2)], {"A": here, "W": wide}, {"c": set(wide)})[0]["A"] == 1  # but its phenomenon covers the spot
    assert levels([ins("L7", "l7", -1.5), ins("L5", "l5", -1.2)], {"L7": here, "L5": here[6:]}, {"l7": set(here), "l5": {"R:5:H"}})[0] == {
        "L7": 6,
        "L5": 6,
    }  # one upload: neither a confirmation nor a cover
    lv, out = levels([a, ins("N", "n", -1.2)], {"A": here, "N": here}, {"n": set(here)})
    assert lv["A"] == 2 and out["A"]["credibility_basis"]["patterns"] == ["N"]  # no registry: no 1
    assert levels([a, ins("U", "usd", -1.2)], {"A": here, "U": here}, {"usd": set(here)})[0]["A"] == 1  # a unit on one side only
    batches["b-a"] = batch("a", "a", "UAH")
    assert levels([a, ins("U", "usd", -1.2)], {"A": here, "U": here}, {"usd": set(here)})[0]["A"] == 2  # UAH is not USD
    lv, out = levels([a, ins("T", "b", -1.2)], {"T": here}, {"b": set(here)})  # a pattern without a place keeps phase A
    assert lv == {"A": 1, "T": 6} and out["A"]["credibility_basis"]["rule"] == "anchor"


def test_credibility_from_independent_sources_phase_a():
    """Phase A (docs/06_graph_and_storage.md §6.2): another declared source sharing a non-weak anchor and the indicator
    the same way confirms (1); one shifting that indicator materially the other way in a shared condition contradicts
    (5, over 1) in the same subgroup (one's conditions contain the other's); a source of another dataset with the same metric name only is 2, nothing 6. Undeclared sources never
    give 1 or 5, one source in two datasets (any case) and the levels of one upload are not independent, and the level
    reads no p, stability or weight."""
    from dataclasses import replace

    from conftest import toy_insight

    from insight_contracts import PhenomenonThresholds, Shift
    from insight_graph_service.core.snapshot import credibility
    from subgroup_miner.spatial_plane import SpatialStructure

    def ins(pid, ds, z, scope=(("zone", "north"),), metric="margin"):
        return toy_insight(list(scope), [Shift(metric, z, 10.0, 12.0, 1.0)], id=pid, dataset_id=ds, batch_id=f"b-{ds}")

    def member(pid, anchor, weak=False):
        return GraphEdge(pid, anchor, EdgeType.ACTIVATES, 0.9, {"weak": weak})

    registry = {"margin": {"indicator": "gross margin"}}
    batches = {
        "b-a": {"source": "Field reports", "indicators": registry},
        "b-b": {"source": "Insurer", "indicators": registry},
        "b-c": {"source": "insurer", "indicators": registry},  # the same source as b, another case
        "b-u": {},  # undeclared
        "b-n": {"source": "Census"},  # declared, no registry
        "b-l7": {"source": "Satellite", "indicators": registry, "upload_id": "up-1"},
        "b-l5": {"source": "Other name", "indicators": registry, "upload_id": "up-1"},
    }
    thresholds = PhenomenonThresholds()

    def levels(insights, edges, tampered=frozenset()):
        out = credibility(insights, batches, edges, set(tampered), thresholds, SpatialStructure())
        return {pid: t["credibility"] for pid, t in out.items()}, out

    a, b = ins("A", "a", -1.5), ins("B", "b", -1.2, scope=(("zone", "south"),))
    lv, out = levels([a, b], [member("A", "A-1"), member("B", "A-1")])
    assert lv == {"A": 1, "B": 1} and out["A"]["credibility_basis"] == {
        "rule": "anchor",
        "patterns": ["B"],
        "sources": ["Insurer"],
        "comparability": "indicator",
    }
    assert levels([a], [member("A", "A-1")])[0] == {"A": 6}  # one source
    assert levels([a, b], [member("A", "A-1"), member("B", "A-1", weak=True)])[0] == {"A": 2, "B": 2}  # a weak membership confirms nothing
    assert levels([a, b], [member("A", "A-1"), member("B", "A-1")], tampered={"B"})[0]["A"] == 6  # a tampered pattern confirms nothing
    contra = ins("C", "b", +1.4)  # the same indicator the other way, in the shared condition zone=north
    lv, out = levels([a, b, contra], [member("A", "A-1"), member("B", "A-1"), member("C", "A-2")])
    assert lv["A"] == 5 and out["A"]["credibility_basis"]["patterns"] == ["C"] and out["A"]["credibility_basis"]["confirmed_by"] == ["B"]
    assert levels([a, ins("C", "b", +1.4, scope=(("zone", "east"),))], [member("A", "A-1"), member("C", "A-2")])[0]["A"] == 2  # another subgroup
    wider = ins("W", "b", +1.4, scope=(("zone", "north"), ("night", "yes")))  # contains a's condition: the same subgroup, narrowed
    overlap = ins("O", "b", +1.4, scope=(("zone", "east"), ("night", "yes")))
    a2 = ins("A2", "a", -1.5, scope=(("zone", "east"), ("rain", "yes")))  # shares zone=east with O, neither contains the other
    assert levels([a, wider], [member("A", "A-1"), member("W", "A-2")])[0]["A"] == 5
    assert levels([a2, overlap], [member("A2", "A-1"), member("O", "A-2")])[0]["A2"] == 2
    assert levels([a, ins("U", "u", -1.2)], [member("A", "A-1"), member("U", "A-1")])[0] == {"A": 2, "U": 2}  # undeclared: never 1
    assert levels([a, ins("U", "u", +1.4)], [member("A", "A-1"), member("U", "A-2")])[0] == {"A": 2, "U": 2}  # undeclared: never 5
    assert levels([b, ins("D", "c", -1.2)], [member("B", "A-1"), member("D", "A-1")])[0] == {"B": 6, "D": 6}  # Insurer = insurer
    lv, out = levels([a, ins("N", "n", -1.2)], [member("A", "A-1"), member("N", "A-1")])
    assert lv == {"A": 2, "N": 2} and out["A"]["credibility_basis"]["comparability"] == "nominal"  # no registry: no 1
    assert levels([ins("L7", "l7", -1.5), ins("L5", "l5", -1.2)], [member("L7", "A-1"), member("L5", "A-1")])[0] == {"L7": 6, "L5": 6}  # one upload
    batches |= {"b-g": {"source": "INSURER", "indicators": {"margin": {"indicator": "Gross Margin"}}}}  # another case of b's source and indicator
    lv, out = levels([a, b, ins("G", "g", -1.1)], [member("A", "A-1"), member("B", "A-1"), member("G", "A-1")])
    assert lv["A"] == 1 and out["A"]["credibility_basis"]["sources"] == ["Insurer"]  # one source listed once
    assert levels([a, ins("G", "g", -1.1)], [member("A", "A-1"), member("G", "A-1")])[0]["A"] == 1  # Gross Margin = gross margin
    batches |= {"b-v2": {"source": "Revised", "indicators": registry, "supersedes": "a"}}  # a new version of dataset a
    assert levels([a, ins("V", "v2", -1.4)], [member("A", "A-1"), member("V", "A-1")])[0] == {"A": 6, "V": 6}  # not independent
    stronger = replace(a, p_value=0.04, p_adjusted=0.049, stability=0.51, weight=0.21)
    assert levels([stronger, b], [member("A", "A-1"), member("B", "A-1")])[0] == {"A": 1, "B": 1}  # p, stability, weight: unread


def test_a_changed_declared_language_is_caught_and_ignored(cfg, demo_csv):
    """The glossary and the metadata document are signed with the record (``BATCH_DIGEST_FIELDS``): either changed by hand
    is a ``metadata_mismatch`` whose Dataset node declares neither, whose prompt has no PLAIN LANGUAGE nor DATASET
    CONTEXT, and which no PATCH may sign again."""
    from test_uploads import DOC, QUESTION

    from insight_graph_service.core.snapshot import metadata_digest

    engine = make_engine(cfg)
    ds = engine.process(engine.submit(demo_csv, glossary='{"category": "товарна категорія"}', metadata=DOC)["batch_id"])["dataset_id"]
    prompt = engine.evidence(QUESTION).evidence_prompt
    assert "PLAIN LANGUAGE" in prompt and "DATASET CONTEXT" in prompt
    record = engine.ws.ready_batch_for(ds)
    for field in ("user_glossary", "metadata_doc"):
        assert metadata_digest({**record, field: {"forged": "x"}}) != record["metadata_digest"], field
    engine.ws.save_batch({**record, "user_glossary": {"category": "forged"}})  # by hand, past the API
    engine = make_engine(cfg)  # an ordinary restart rebuilds the snapshot
    props = engine.graph().nodes[f"DS:{ds}"]["props"]
    assert props["integrity"] == "mismatch" and "user_glossary" not in props and "metadata_doc" not in props
    prompt = engine.evidence(QUESTION).evidence_prompt
    assert "PLAIN LANGUAGE" not in prompt and "DATASET CONTEXT" not in prompt and "forged" not in prompt
    with pytest.raises(PipelineError) as refused:
        engine.update_dataset_metadata(ds, glossary="")
    assert refused.value.code == "metadata_mismatch"
