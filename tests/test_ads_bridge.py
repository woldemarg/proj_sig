"""The ADS bridge (docs/12_architecture.md): a conversation's questions about zones through the graph, ADS and the final
report, against one stub that plays the narrator, the graph service's exports, the ADS orchestrator and its LLM gateway."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import httpx
import pytest
from conftest import make_engine, serve, stack, upload_levels
from dev import chromium
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from ads_bridge_service import task
from ads_bridge_service.app import _PERIODS, create_app, error_code
from ads_bridge_service.settings import AdsSettings
from ads_bridge_service.task import build_metadata_md, name_tables
from insight_contracts import ZONE_FITS, ZONE_OWN
from insight_contracts.wording import PERIOD

POLYGON = {"type": "Polygon", "coordinates": [[[30.0, 50.0], [30.2, 50.0], [30.2, 50.1], [30.0, 50.1], [30.0, 50.0]]]}
OTHER = {"type": "Polygon", "coordinates": [[[31.0, 50.0], [31.2, 50.0], [31.2, 50.1], [31.0, 50.1], [31.0, 50.0]]]}
SID = "11111111-2222-4333-8444-555555555555"
TABLES = [{"dataset_id": "ds-a", "filename": "events.csv", "resolution": 8}, {"dataset_id": "ds-b", "filename": "bars.csv", "resolution": 5}]
FINAL = "## Результати аналізу\nу зоні 3 події\n## Методологія\nполігон"


def item(key: str, credibility: int, dataset: str = "ds-a", resolution: int | None = 8, **more) -> dict[str, Any]:
    return {
        "provenance": {"dataset_id": dataset, "spatial": {"resolution": resolution} if resolution else None},
        "key": key,
        "pattern_id": f"P-{key}",
        "scope": ["lisa_points=hot spot"],
        "scope_text": "lisa points is hot spot",
        "shift_text": ["points: strong increase, +2.1 sd (median 9 vs 3 overall)"],
        "validation": "adjusted p 0.001",
        "trust": "source F",
        "credibility": credibility,
        **more,
    }


def place(fit: str, share: float = 0.0, cover: float = 0.0, lift: float = 1.0) -> dict[str, Any]:
    return {"overlap": 0.5, "share": share, "cover": cover, "ratio": lift, "lift": lift, "fit": fit}


PLACES = {  # how each finding lies in the two test zones; P-P4, a table level the map does not place, in neither
    json.dumps(POLYGON): {
        "P-P1": place("inside", 0.9, 0.3, 4.2),
        "P-P5": place("concentrated", 0.4, 0.1, 2.6),
        "P-P2": place("map-wide", 0.1, 0.05, 1.0),
        "P-Q1": place("outside"),
        "P-P6": place("outside"),
    },
    json.dumps(OTHER): {
        "P-P1": place("concentrated", 0.3, 0.2, 2.0),
        "P-Q1": place("inside", 0.95, 0.5, 6.0),
        "P-P2": place("elsewhere", 0.01, 0.0, 0.2),
        "P-P5": place("outside"),
    },
}


def upstream(
    seen: dict[str, Any],
    polls: int = 1,
    report: str = "ADS: 3 events ![plot](/workspace/a.png)",
    final: str | None = FINAL,
    flaky: int = 0,
    hold: threading.Event | None = None,
) -> FastAPI:
    """``polls``: status reads until the job ends; ``flaky``: 502s before the first of them; ``final`` None: the gateway is
    down; ``hold``: the gateway answers once it is set."""
    app = FastAPI()
    left = {"polls": polls, "flaky": flaky}

    @app.post("/api/chat/query")
    def chat(body: dict[str, Any]):
        seen.setdefault("chat", []).append(body)
        files = {"P1": "events.csv", "P2": "other.csv", "P3": "events.csv", "P4": "events.csv", "P5": "events.csv"}
        provenance = [{"key": k, "filename": f} for k, f in files.items()]
        if body.get("geometry") == OTHER:  # another zone's evidence: its own P1 is another finding than the first zone's
            items = [item("P1", 6, pattern_id="P-Q1"), item("P2", 6, pattern_id="P-P1")]
        elif body["question"]:  # P3: P1 again at another level, folded into it; P4: the base file's table level, a dataset of its own
            items = [item("P2", 1, "ds-x"), item("P3", 6, "ds-a5", 5, folded_into="P1"), item("P4", 6, "ds-t", None), item("P1", 6), item("P6", 1)]
        else:  # the zone alone finds the finding concentrated in it, which the text did not seed
            items = [item("P5", 6), item("P1", 6)]
        return {"view": {"evidence": {"items": items, "provenance": provenance}}}

    @app.post("/api/geo/places")
    def places(body: dict[str, Any]):
        seen.setdefault("places", []).append(body)
        out = {z["id"]: PLACES[json.dumps(z["polygon"])] for z in body["zones"]}
        return {z: {pid: p for pid, p in here.items() if pid in body["pattern_ids"]} for z, here in out.items()}

    def table(dataset_id: str) -> dict[str, Any] | None:
        if dataset_id not in ("ds-a", "ds-b"):
            return None
        base = dataset_id == "ds-a"  # the base: analysed at 8, with a time column; the layer: at 5, a filtered reading
        return {"dataset_resolution": 8 if base else 5, "time": "seen_at" if base else None, "where": None if base else "kind=bar", "total": 5}

    def unknown(dataset_id: str) -> JSONResponse:
        return JSONResponse({"detail": f"{dataset_id} is not a geo dataset", "code": "unknown_dataset"}, status_code=404)

    @app.post("/api/geo/{dataset_id}/zone_counts")
    def zone_counts(dataset_id: str, body: dict[str, Any]):
        seen.setdefault("zone_counts", []).append((dataset_id, body))
        if (meta := table(dataset_id)) is None:
            return unknown(dataset_id)
        if body.get("layer"):  # a layer of an aligned run over one ring
            meta |= {"layer": body["layer"], "ring": 1, "window": ["2031-03-01", "2031-03-31"]}
        zones = {z["id"]: {"cells": 7, "rows": 3} for z in body["zones"]}
        at = body["resolution"]  # the export's level, as asked: the graph service names it, not the table's own
        return meta | {"resolution": at, "columns": ["row_id", *(f"h3_r{r}" for r in (at, at - 1, at - 2))], "unlocated": 2, "zones": zones}

    @app.post("/api/geo/{dataset_id}/zone_cells")
    def zone_cells(dataset_id: str, body: dict[str, Any]):
        seen.setdefault("zone_cells", []).append((dataset_id, body))
        if table(dataset_id) is None:
            return unknown(dataset_id)
        head = "zone_id,row_id" if seen.get("cells") == "broken" else f"zone_id,h3_r{body['resolution']}"  # another export's CSV
        listed = "".join(f"{z['id']},88{n}f\n" for z in body["zones"] for n in (1, 2))  # each zone's two cells, one without rows
        return Response(f"{head}\n{listed}", media_type="text/csv")

    @app.get("/api/geo/{dataset_id}/rows")
    def rows(dataset_id: str, resolution: int, layer: str | None = None):
        seen.setdefault("rows", []).append((dataset_id, resolution) if layer is None else (dataset_id, resolution, layer))
        return Response(f"rows of {dataset_id}".encode(), media_type="text/csv") if table(dataset_id) else unknown(dataset_id)

    @app.get("/api/datasets/{dataset_id}/metadata")
    def metadata(dataset_id: str) -> Any:
        seen.setdefault("metadata", []).append(dataset_id)
        if seen.get("docs") == "down":
            return JSONResponse({"detail": "graph down"}, status_code=500)
        return seen.get("docs", {}).get(dataset_id, {})

    @app.get("/health")
    def health():
        return {}

    @app.post("/sessions")
    def session():
        seen["sessions"] = seen.get("sessions", 0) + 1
        return {"session_id": SID}

    @app.post("/sessions/{sid}/datasets")
    async def dataset(sid: str, file: UploadFile):
        seen.setdefault("uploads", []).append(file.filename)
        seen.setdefault("files", {})[file.filename] = (await file.read()).decode("utf-8")
        return {"session_id": sid}

    @app.get("/sessions/{sid}/sources")
    def sources(sid: str):
        files = [{"name": n, "path": f"/workspace/{sid}/{n}", "source_kind": "uploaded_input"} for n in seen.get("files", {})]
        return {"files": [*files, {"name": "report.md", "path": "/x", "source_kind": "generated_artifact"}]}

    @app.post("/sessions/{sid}/sources/delete")
    def delete(sid: str, body: dict[str, Any]):
        seen.setdefault("deleted", []).append(body["path"])
        seen.get("files", {}).pop(body["path"].rsplit("/", 1)[-1], None)  # gone from the session's sources
        return {"deleted": True}

    @app.post("/sessions/{sid}/jobs")
    def job(sid: str, body: dict[str, Any]):
        if seen.get("busy"):
            return JSONResponse({"detail": "running"}, status_code=409)
        seen.setdefault("queries", []).append(body["user_query"])
        seen["query"] = body["user_query"]
        left["polls"] = polls
        return {"job_id": "j1"}

    @app.post("/sessions/{sid}/jobs/cancel")
    def cancel(sid: str):
        seen["cancelled"] = sid
        return {}

    @app.get("/sessions/{sid}/status")
    def status(sid: str):
        if seen.get("busy"):
            return JSONResponse({}, status_code=202)
        if left["flaky"] > 0:
            left["flaky"] -= 1
            return JSONResponse({"detail": "bad gateway"}, status_code=502)
        left["polls"] -= 1
        if left["polls"] > 0:
            return JSONResponse({}, status_code=202)
        return {"outcome": "completed", "result": {"last_execute_response": {"report": report}}}

    @app.post("/sessions/{sid}/close")
    def close(sid: str):
        seen["closed"] = sid
        return {}

    @app.post("/api/reset")
    def reset():
        seen["reset"] = True
        return {}

    @app.post("/v1/chat/completions")
    async def llm(req: Request):
        if final is None:
            return JSONResponse({"detail": "gateway down"}, status_code=503)
        seen["prompt"] = (await req.json())["messages"][0]["content"]
        while hold and not hold.is_set():
            await asyncio.sleep(0.01)
        return {"choices": [{"message": {"content": final}}]}

    return app


def finished(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(200):
        job: dict[str, Any] = client.get(f"/api/ads/jobs/{job_id}").json()
        if job["stage"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(job)


def settings(url: str, **over: Any) -> AdsSettings:
    values: dict[str, Any] = {"ads_url": url, "ads_llm_url": url, "ads_graph_url": url, "ads_narrator_url": url, "ads_poll_s": 0.01}
    return AdsSettings(**values | over)


def test_a_conversation_asks_ads_in_one_session_and_sends_a_table_once():
    seen: dict[str, Any] = {
        "docs": {
            "ds-a": {
                "dataset": {"description": "planted events"},
                "columns": {"seen_at": {"title_uk": "коли", "assistant_context": "the event's time"}, "kind": {}},
            }
        }
    }
    ask = {"question": "Скільки подій у зоні?", "tables": TABLES, "zones": [{"id": "Z1", "polygon": POLYGON}], "focus": ["Z1"]}
    files = ("events_dsa_r8.csv", "bars_dsb_r8.csv")  # the name, the dataset and the base table's resolution: one file per session
    with serve(upstream(seen, polls=3, flaky=2)) as url, TestClient(create_app(settings(url))) as client:  # two bad status reads, read again
        assert client.get("/api/ads/health").json()["ok"]
        parts = client.post("/api/ads/preview", json=ask | {"question": "Q?"}).json()["parts"]
        assert [p["kind"] for p in parts] == ["primer", "frame", "user", "data", "insights", "guidance"] and parts[2]["text"] == "Q?"
        assert parts[1]["text"] == "Analyse the area of interest (AOI) Z1: " and "(after the ask)" in parts[4]["text"]
        assert f"df_events_dsa_r8 ({files[0]}: events.csv)" in parts[3]["text"] and f"df_bars_dsb_r8 ({files[1]}: bars.csv" in parts[3]["text"]

        started = client.post("/api/ads/jobs", json=ask)
        assert started.status_code == 202
        job = finished(client, started.json()["job_id"])
        assert job["stage"] == "done" and set(job["timings"]) == {"search", "ads", "joint"} and job["session_id"] == SID, job
        # the graph is asked with the polygon the rows are read against, and no model, for the text and for the zone alone
        assert seen["chat"] == [
            {"question": ask["question"], "geometry": POLYGON, "use_llm": False, "period_only": True},
            {"question": "", "geometry": POLYGON, "use_llm": False, "period_only": True},
        ]
        assert seen["places"] == [{"zones": [{"id": "Z1", "polygon": POLYGON}], "pattern_ids": ["P-P1", "P-P2", "P-P4", "P-P5", "P-P6"]}]
        # by fit, then lift: the zone's own first, the map-wide one after them, the unplaced table level last; P6, outside, left out
        assert [(i["pattern_id"], (i["places"]["Z1"] or {}).get("fit")) for i in job["insights"]] == [
            ("P-P1", "inside"),
            ("P-P5", "concentrated"),
            ("P-P2", "map-wide"),
            ("P-P4", None),
        ]
        # every table's rows and counts at the base table's resolution: one cell column tells a row's zone in every table
        assert [(d, b["resolution"]) for d, b in seen["zone_counts"]] == [("ds-a", 8), ("ds-b", 8)] and seen["rows"] == [("ds-a", 8), ("ds-b", 8)]
        # the zones go as cell lists alone, once, from the base table: no row is stamped with its zone
        assert seen["uploads"] == [*files, "zone_cells.csv", "zones.md", "metadata.md"] and seen["files"][files[0]] == "rows of ds-a"
        assert seen["zone_cells"] == [("ds-a", {"zones": ask["zones"], "resolution": 8})]
        assert seen["files"]["zone_cells.csv"].splitlines() == ["zone_id,h3_r8", "Z1,881f", "Z1,882f"]
        assert seen["files"]["metadata.md"].splitlines()[2:] == [
            f"## {files[0]} (events.csv)",
            "planted events",
            "- seen_at (коли): the event's time",
            "- kind",
        ]
        assert f"{files[0]} (events.csv)" in seen["files"]["zones.md"] and "events.csv: O1" in seen["files"]["zones.md"]
        assert (
            "- Zones of this conversation: Z1;" in seen["files"]["zones.md"]
            and "SIG analysed this table at H3 resolution 5" in seen["files"]["zones.md"]
        )
        query = seen["query"]
        assert query == job["ads_query"] and ask["question"] in query
        assert "Zone lists (main table rows): Z1: cells=7, rows=3. Rows without a cell: 2." in query
        assert "O1 · Z1: inside (90% of its events, 30% of the zone's events, x4.2 the map's rate)" in query
        assert "O2 · Z1: concentrated (40% of its events" in query and "established: points: strong increase" in query
        assert "O3 · Z1: map-wide (10% of its events, 5% of the zone's events, x1.0 the map's rate)" in query
        assert "file: other.csv (another dataset of the graph" in query and query.count("another dataset of the graph") == 1
        assert "O4 · Z1: place not measured" in query and "level: the table level of the base table's file" in query and query.count("  level: ") == 1
        assert "`map-wide` (in the zone at about the map's rate" in query and "Shared (" not in query  # one zone: nothing to share
        assert "bars.csv, only its rows where kind=bar, 5 rows" in query and "Time: seen_at in df_events_dsa_r8" in query
        assert "h3_r8, h3_r7, h3_r6 are H3 cell ids" in query
        # the task describes the zones' file and teaches no method: a row is in a zone by its listed cell, and the file lists zones alone
        assert "df_zone_cells (zone_cells.csv) lists the cells of every zone of this conversation, one row per cell: zone_id, h3_r8" in query
        assert "A row of this question's tables is in a zone when its h3_r8 is listed for that zone" in query
        assert "The zones are areas of the map, not its bounds: the map is every row of the tables, in a zone or in none." in query
        assert "- zone_cells.csv: zone_id, h3_r8;" in seen["files"]["zones.md"]
        for gone in ("grid_disk", "cell_area", "Comparison groups", "adjacent", "aoi_zone", "df_zones", "zones.csv", "polygon", "left-join", "_ring"):
            assert gone not in query and gone not in seen["files"]["zones.md"], gone  # no stamp, no polygon, no recipe
        assert job["ads_report"] == "ADS: 3 events plot" and "closed" not in seen  # no link into ADS's own workspace; the session stays
        assert job["joint_report"] == FINAL and "ADS: 3 events plot" in seen["prompt"] and "Словник термінів" in seen["prompt"]
        # the final report's methodology: one rule, by cell, for the tables and for the graph's findings
        assert "рядок у зоні, коли його комірка H3 рівня 8 є серед комірок зони" in seen["prompt"] and "(комірок у зоні: Z1 7)" in seen["prompt"]
        assert "events.csv: 3 з 5 рядків у зоні" in seen["prompt"] and "найдрібнішого гео-рівня" in seen["prompt"]
        assert "point-in-polygon" not in seen["prompt"] and "adjacent" not in seen["prompt"]
        # what the grid does to a small zone is said to the reader of the report
        assert "комірок під його вершинами" in seen["prompt"] and "зону з кількох комірок вона окреслює грубо" in seen["prompt"]

        # the next question of the conversation: the same session, the tables it holds not sent again, the zones replaced
        two = ask | {
            "session_id": SID,
            "zones": [*ask["zones"], {"id": "Z2", "polygon": OTHER}],
            "focus": ["Z2", "Z1"],
            "question": "Чим Z2 відрізняється?",
        }
        seen["docs"] = {}  # the document taken off between the questions: the copy ADS holds is replaced, not kept
        seen["files"]["zones.csv"] = "row_id,zone_id\n"  # a conversation opened before a zone was its cells: row stamps of another rule
        job = finished(client, client.post("/api/ads/jobs", json=two).json()["job_id"])
        assert job["stage"] == "done" and job["session_id"] == SID and seen["sessions"] == 1 and "closed" not in seen
        assert seen["uploads"][5:] == ["zone_cells.csv", "zones.md", "metadata.md"] and len(seen["rows"]) == 2  # the tables went once
        assert "No table of this question has a document" in seen["files"]["metadata.md"]
        assert "- Zones of this conversation: Z1, Z2;" in seen["files"]["zones.md"]
        # every zone of the conversation, and nothing else: no list of the cells around a zone
        assert {line.split(",")[0] for line in seen["files"]["zone_cells.csv"].splitlines()[1:]} == {"Z1", "Z2"}
        # the stamps this question does not send leave the session: the task names one rule, and one file holds it
        assert seen.pop("deleted") == [f"/workspace/{SID}/zones.csv"] and "zones.csv" not in seen["files"]
        query = seen["query"]
        assert query.count("Compare the areas of interest Z2, Z1: ") == 1
        # each zone's own first (5 each), every finding placed in both zones; Z2's P1 is not Z1's P1
        assert [i["pattern_id"] for i in job["insights"]] == ["P-Q1", "P-P1", "P-P5", "P-P2", "P-P4"]
        assert "O1 · Z2: inside (95% of its events, 50% of the zone's events, x6.0 the map's rate); Z1: outside" in query
        assert "O2 · Z2: concentrated (30% of its events, 20% of the zone's events, x2.0 the map's rate); Z1: inside (90%" in query
        assert "O4 · Z2: elsewhere (1% of its events" in query and "O5 · Z2: place not measured; Z1: place not measured" in query
        summary = "Shared (inside or concentrated in several zones): O2 (Z2, Z1).\nOne zone only: O1 (Z2), O3 (Z1).\nMap-wide (as everywhere): O4."
        assert summary in query
        lone = ask | {"session_id": SID, "zones": two["zones"]}  # Z2 is the conversation's, the question is about Z1 alone
        assert (
            "The conversation's other zones (Z2) are listed in df_zone_cells too"
            in finished(client, client.post("/api/ads/jobs", json=lone).json()["job_id"])["ads_query"]
        )
        # the counts are asked for the question's zone alone, the cell lists for every zone of the conversation
        assert [z["id"] for z in seen["zone_counts"][-1][1]["zones"]] == ["Z1"]
        assert [z["id"] for z in seen["zone_cells"][-1][1]["zones"]] == ["Z1", "Z2"]
        seen["busy"], sent, asked = True, len(seen["uploads"]), len(seen["chat"])  # the conversation's last question still runs in ADS
        assert finished(client, client.post("/api/ads/jobs", json=two).json()["job_id"])["error"]["code"] == "ads_busy"
        assert (len(seen["uploads"]), len(seen["chat"])) == (sent, asked)  # refused before the graph is asked or a file replaced

        # the graph reads the period bar alone (period_only): a period in the text is ADS's, the bar's goes beside it, and
        # the words go as written, a token that only looks like a period (2019-20) a literal the graph keeps
        seen["chat"].clear()
        seen["busy"] = False
        bare = ask | {"question": "Період: 2026-08-09 — 2026-08-17 (включно)\n2026-W33"}  # periods alone: the zone is asked once, alone
        assert finished(client, client.post("/api/ads/jobs", json=bare).json()["job_id"])["stage"] == "done"
        assert [c["question"] for c in seen["chat"]] == [""]
        seen["chat"].clear()
        dated = ask | {
            "question": "Період: 2026-08-09 — 2026-08-17 (включно)\nСкільки подій у 2026-W33 чи Q3 2026 за сезон 2019-20?",
            "period": ["2026-08-09T00:00", "2026-08-18T00:00"],
        }
        job = finished(client, client.post("/api/ads/jobs", json=dated).json()["job_id"])
        sent = seen["chat"][0]
        assert sent["period"] == dated["period"] and sent["period_only"] is True
        assert sent["question"].split() == "Скільки подій у 2026-W33 чи Q3 2026 за сезон 2019-20?".split()  # the header line alone goes
        assert dated["question"] in job["ads_query"] and "A period the question names bounds every figure" in job["ads_query"]

        # the conversation's end: its tables deleted from ADS, the session closed
        assert client.post(f"/api/ads/sessions/{SID}/close").json() == {"session_id": SID, "kept": [], "closed": True}
        assert seen["closed"] == SID and sorted(seen["deleted"]) == sorted(f"/workspace/{SID}/{f}" for f in (*files, "zone_cells.csv"))
        assert client.post("/api/ads/sessions/..%2Freset/close").status_code in (404, 422) and "reset" not in seen

        assert client.post("/api/ads/jobs", json=ask | {"question": " "}).status_code == 400
        assert client.post("/api/ads/jobs", json=ask | {"question": "Період: 2026-08-09 — 2026-08-17 (включно)\n "}).status_code == 400
        assert client.post("/api/ads/jobs", json=ask | {"zones": []}).status_code == 422
        assert client.post("/api/ads/jobs", json=ask | {"focus": ["Z9"]}).status_code == 422  # a zone the conversation lacks
        assert client.post("/api/ads/jobs", json=ask | {"focus": [f"Z{n}" for n in range(6)]}).status_code == 422  # five at most
        assert client.post("/api/ads/jobs", json=ask | {"session_id": "../x"}).status_code == 422
        assert client.post("/api/ads/jobs", json=ask | {"tables": []}).status_code == 422
        assert client.post("/api/ads/jobs", json=ask | {"tables": TABLES[:1] * 2}).status_code == 422  # a table twice: its rows twice
        assert client.post("/api/ads/jobs", json=ask | {"focus": ["Z1", "Z1"]}).status_code == 422
        many = [{"id": f"Z{n}", "polygon": POLYGON} for n in range(51)]
        assert client.post("/api/ads/jobs", json=ask | {"zones": many}).status_code == 422
        escaping = ask | {"tables": [{"dataset_id": "../reset?x=", "filename": "x.csv", "resolution": 8}]}  # an id is one path segment, never a path
        assert finished(client, client.post("/api/ads/jobs", json=escaping).json()["job_id"])["error"]["code"] == "sig_failed" and "reset" not in seen
        assert client.get("/api/ads/jobs/nope").status_code == 404


def test_a_layer_goes_as_a_context_table():
    """A layer link run's layer goes to ADS as a context table of its points, its rows and zones read by name and tied
    to the insights' band of it; a layer is never the base table."""
    seen: dict[str, Any] = {}
    bars = {"dataset_id": "ds-a", "filename": "bars", "resolution": 8, "layer": "bars"}
    ask = {"question": "Де більше подій біля барів?", "tables": [TABLES[0], bars], "zones": [{"id": "Z1", "polygon": POLYGON}], "focus": ["Z1"]}
    with serve(upstream(seen)) as url, TestClient(create_app(settings(url))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["stage"] == "done" and seen["rows"] == [("ds-a", 8), ("ds-a", 8, "bars")]
        assert [b.get("layer") for _, b in seen["zone_counts"]] == [None, "bars"]
        assert seen["uploads"] == ["events_dsa_r8.csv", "bars_dsa_layer_r8.csv", "zone_cells.csv", "zones.md", "metadata.md"]
        # the zones' cells are the base table's, never a layer's: the layer's rows fall in a zone by the same cell column
        assert [d for d, _ in seen["zone_cells"]] == ["ds-a"]
        query = job["ads_query"]
        assert "df_bars_dsa_layer_r8 (bars_dsa_layer_r8.csv: bars" in query
        assert "context_bars_band counts per cell with its six neighbours, only its points within 2031-03-01..2031-03-31" in query
        assert "is a context layer of the main table" in query and "bars_dsa_layer_r8.csv (bars): a context layer" in seen["files"]["zones.md"]
        assert (
            "an empty h3_r8 (NaN once pandas reads the CSV) has no cell and is in no zone" in query and "NaN in pandas" in seen["files"]["zones.md"]
        )
        assert "the run's H3 resolution" not in query  # the layer's cells at the run's own level: nothing to say
        # a layer its run counted on another grid than the base table's: the preview and the job's task both say so
        note = "SIG counted them on the run's H3 resolution 8 grid; their h3 columns here are the main table's resolution 5"
        coarse = {"question": "Q?", "tables": [TABLES[1], bars]}
        assert note in client.post("/api/ads/preview", json=coarse).json()["parts"][3]["text"]
        sent = finished(client, client.post("/api/ads/jobs", json=ask | coarse).json()["job_id"])
        assert sent["stage"] == "done" and note in sent["ads_query"]  # the graph's answer names the export's level, never the layer's own
        same = name_tables(
            [
                {"dataset_id": "ds-a", "filename": "crimes_2024_2025_robbery_x.csv", "resolution": 8},
                bars | {"filename": "crimes_2024_2025_robbery_xy"},
            ]
        )
        assert len({t["file"] for t in same}) == 2  # a layer named as its table's first 24 letters keeps its own file
        assert client.post("/api/ads/jobs", json=ask | {"tables": [bars, TABLES[0]]}).status_code == 422  # a layer is never the base
        assert client.post("/api/ads/jobs", json=ask | {"tables": [TABLES[0], bars, bars]}).status_code == 422


def test_a_question_needs_no_zone():
    """A question with no zone asks the graph about the whole map, sends no zones file and frames the task and the
    final report's methodology as the whole map's. So does one of a conversation with zones: the file an earlier
    question left in the session is deleted, for the one table that lists cells is not the map."""
    seen: dict[str, Any] = {}
    ask = {"question": "Де найбільше подій?", "tables": TABLES[:1]}
    with serve(upstream(seen)) as url, TestClient(create_app(settings(url))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["stage"] == "done" and seen["chat"] == [{"question": ask["question"], "use_llm": False, "period_only": True}]  # no geometry
        assert [b["zones"] for _, b in seen["zone_counts"]] == [[]] and seen["uploads"] == [
            "events_dsa_r8.csv",
            "zones.md",
            "metadata.md",
        ]  # no zones file
        assert "zone_cells" not in seen  # nor their cells
        query = job["ads_query"]
        assert query.count("Analyse the whole map") == 1 and "df_zone_cells" not in query and "O1 · in zone" not in query and "\nO1\n" in query
        assert "`map-wide`" not in query and "Zone lists" not in query and "how it lies in each zone" not in query  # no zone sentence
        assert "- No zone is asked: the question is about the whole map." in seen["files"]["zones.md"] and "# Уся мапа: методика" in seen["prompt"]
        two = ask | {"session_id": SID, "question": "А по місяцях?"}  # a follow-up of a conversation without zones
        assert finished(client, client.post("/api/ads/jobs", json=two).json()["job_id"])["stage"] == "done" and seen["sessions"] == 1
        bare = ask | {"session_id": SID, "question": "Період: 2031-03-01 — 2031-03-31 (включно)\n\n2031-03"}  # no words for the graph
        job = finished(client, client.post("/api/ads/jobs", json=bare).json()["job_id"])
        assert job["stage"] == "done" and len(seen["chat"]) == 2 and "SIG has no insights for this question." in job["ads_query"]
        kept = ask | {"session_id": SID, "zones": [{"id": "Z1", "polygon": POLYGON}]}
        zoned = finished(client, client.post("/api/ads/jobs", json=kept | {"focus": ["Z1"]}).json()["job_id"])  # a question about Z1
        assert zoned["stage"] == "done" and "zone_cells.csv" in seen["files"] and "deleted" not in seen
        # the next one ticks no zone: the whole map again, as if no zone were drawn, and the session holds no table that lists cells
        job = finished(client, client.post("/api/ads/jobs", json=kept).json()["job_id"])
        assert job["stage"] == "done" and seen["uploads"][-2:] == ["zones.md", "metadata.md"] and len(seen["zone_cells"]) == 1
        assert seen["deleted"] == [f"/workspace/{SID}/zone_cells.csv"] and "zone_cells.csv" not in seen["files"]
        assert job["ads_query"] == query and "Analyse the whole map (every row of the tables): " in query
        assert "No zone is asked" in seen["files"]["zones.md"] and "Zones of this conversation" not in seen["files"]["zones.md"]
        assert "# Уся мапа: методика" in seen["prompt"] and "- Зону не задано: питання стосується всіх рядків таблиць." in seen["prompt"]


def test_the_bridges_copies_of_the_kernel_agree_with_it():
    """The bridge imports nothing of the repository, so it keeps copies. Its ``wording.PERIOD`` finds what the graph would
    read as a period: a question of periods alone has no words for the graph, which is not asked about it (its words go
    as written, ``period_only``). Its zone fits are the kernel's: the zone's own the same pair, every ranked fit one."""
    samples = "2031-W13, 2031-q1, Q1 2031, 2031-03-24 10:00, 2031-03; not 2019-20, x2031-03, 2031-03-24-1, 12031-03"
    assert [m.group(0) for m in _PERIODS.finditer(samples)] == [m.group(0) for m in PERIOD.finditer(samples)]
    assert task.ZONE_OWN == ZONE_OWN == ZONE_FITS[:2] and set(task.FIT_RANK) - {None} <= set(ZONE_FITS)


def test_a_failed_stage_is_recorded_on_the_job_with_its_code():
    ask = {"question": "Q?", "tables": TABLES[:1], "zones": [{"id": "Z1", "polygon": POLYGON}], "focus": ["Z1"]}
    seen: dict[str, Any] = {}
    with serve(upstream(seen, polls=10**6)) as url, TestClient(create_app(settings(url, ads_job_timeout_s=0.05))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert (job["stage"], job["failed_stage"], job["error"]["code"]) == ("error", "ads", "ads_timeout")
        assert seen["cancelled"] == SID == job["session_id"]  # the job given up is cancelled: the session takes the next question
    seen = {}
    with serve(upstream(seen, flaky=10**6)) as url, TestClient(create_app(settings(url))) as client:  # the status never reads
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert (job["failed_stage"], job["error"]["code"]) == ("ads", "ads_failed") and seen["cancelled"] == SID  # given up: cancelled
    seen = {}
    with serve(upstream(seen, polls=10**6)) as url, TestClient(create_app(settings(url))) as client:  # the conversation ends meanwhile
        job_id = client.post("/api/ads/jobs", json=ask).json()["job_id"]
        while "query" not in seen:
            time.sleep(0.01)
        assert client.post(f"/api/ads/jobs/{job_id}/cancel").json()["cancelled"] and client.post("/api/ads/jobs/nope/cancel").status_code == 404
        job = finished(client, job_id)
        assert (job["failed_stage"], job["error"]["code"]) == ("ads", "cancelled") and seen["cancelled"] == seen["closed"] == SID
        assert sorted(seen["deleted"]) == [f"/workspace/{SID}/{f}" for f in ("events_dsa_r8.csv", "zone_cells.csv")]  # its tables gone
        assert not client.post(f"/api/ads/jobs/{job_id}/cancel").json()["cancelled"]  # an ended job has nothing to stop
    seen, hold = {}, threading.Event()
    with serve(upstream(seen, hold=hold)) as url, TestClient(create_app(settings(url))) as client:  # it ends while the report is written
        job_id = client.post("/api/ads/jobs", json=ask).json()["job_id"]
        while client.get(f"/api/ads/jobs/{job_id}").json()["stage"] != "joint":
            time.sleep(0.01)
        assert client.post(f"/api/ads/jobs/{job_id}/cancel").json()["cancelled"]
        hold.set()  # the report's one LLM call ends; the job then stops and closes its session
        job = finished(client, job_id)
        assert (job["failed_stage"], job["error"]["code"]) == ("joint", "cancelled") and seen["closed"] == SID
    seen = {}
    with serve(upstream(seen, polls=10**6)) as url:  # the bridge stops while ADS runs the job: cancelled in ADS, the session kept
        with TestClient(create_app(settings(url))) as client:
            client.post("/api/ads/jobs", json=ask)
            while "query" not in seen:
                time.sleep(0.01)
        assert seen["cancelled"] == SID and "closed" not in seen  # the conversation's next question goes on in it
    with serve(upstream({}, final=None)) as url, TestClient(create_app(settings(url))) as client:  # no final report: ADS's stands
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["stage"] == "done" and job["joint_report"] == "" and "503" in job["joint_error"] and job["ads_report"]
    with serve(upstream({}, report="")) as url, TestClient(create_app(settings(url))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["error"]["code"] == "ads_failed" and "no report" in job["error"]["detail"]
        gone = ask | {"tables": [{"dataset_id": "ds-gone", "filename": "gone.csv", "resolution": 8}]}
        job = finished(client, client.post("/api/ads/jobs", json=gone).json()["job_id"])
        assert (job["failed_stage"], job["error"]["code"]) == ("search", "sig_failed") and "unknown_dataset" in job["error"]["detail"]
    for fault in ({"docs": "down"}, {"cells": "broken"}):  # the graph fails to give a document, or sends another table for the zones' cells
        seen = dict(fault)
        with serve(upstream(seen)) as url, TestClient(create_app(settings(url))) as client:
            job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
            assert (job["failed_stage"], job["error"]["code"]) == ("search", "sig_failed") and "sessions" not in seen  # no ADS session opened
    about = build_metadata_md(name_tables(TABLES[:1]), {"ds-a": {"dataset": {"description": "planted events"}, "columns": {}}})
    assert about.splitlines()[2:] == ["## events_dsa_r8.csv (events.csv)", "planted events"]  # a description alone still goes
    with serve(upstream({}, final="no sections")) as url, TestClient(create_app(settings(url))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["stage"] == "done" and job["joint_report"] == "" and job["ads_report"]  # the ADS report stands alone
    with serve(upstream({})) as url, TestClient(create_app(settings(url, ads_url="http://127.0.0.1:9"))) as client:
        assert client.get("/api/ads/health").json() == {
            "ok": False,
            "ads": {"reachable": False, "url": "http://127.0.0.1:9"},
            "llm": {"reachable": True, "url": url},
        }
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert (job["failed_stage"], job["error"]["code"]) == ("ads", "ads_unavailable")
        assert client.post(f"/api/ads/sessions/{SID}/close").json()["closed"] is False  # best effort: the console has moved on
    # a service that does not answer is SIG's while the graph is asked, ADS's after; any other fault is the bridge's own
    assert (error_code(httpx.ConnectError("down"), "search"), error_code(ValueError("bug"), "ads")) == ("sig_failed", "internal_error")


@pytest.mark.browser
def test_the_console_asks_ads_about_a_drawn_zone(cfg, points_csv):
    """The header says whether ADS answers; a drawn zone's popup opens the form (the open dataset and that zone ticked, the
    task previewed), and the ask shows its stages and then the report, a table included, in the ADS tab; the tab's box
    asks the conversation's next question in the same session, New chat closes it. Without the bridge the pill says so
    and the button is off."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    upload_levels(engine, points_csv, geo="auto@8", time="when")  # timed: the period bar is shown
    seen: dict[str, Any] = {}
    table = "## Результати аналізу\n\n| група | рядків |\n|---|---|\n| у зоні | 3 |\n\n## Методологія\n\nполігон"
    # the first cell's hexagon at 1.5 times its size: the zone is that cell, the ring round it lies in the polygon off the zone's cells
    grow = "(k) => { const f = G.data.features[0], [cx, cy] = centre(f); return f.geometry.coordinates[0].map(([x, y]) => [cx + k * (x - cx), cy + k * (y - cy)]); }"
    draw = f"(() => {{ const grow = {grow}; G.map.jumpTo({{ center: centre(G.data.features[0]), zoom: 14 }}); GeoMap.clearZones(); GeoMap.addZone({{ type: 'Polygon', coordinates: [grow(1.5)] }}); GeoMap.openZone(GeoMap.zones()[0].id); }})()"
    off = f"(() => {{ const grow = {grow}; return G.map.project(grow(1.25)[0]); }})()"
    with serve(upstream(seen, polls=2, final=table)) as ads_url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        for with_ads in (True, False):
            port, app = stack(engine, ads_url if with_ads else None)
            with serve(app, port=port) as url:
                page = browser.new_page(viewport={"width": 1400, "height": 900})
                errors: list[str] = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                page.goto(url)
                page.wait_for_selector("#ads-pill")
                page.click("#view-switch button[data-view='map']")
                page.wait_for_function("G.data && G.map && G.map.loaded() && G.overlay", timeout=60000)
                page.evaluate(draw)
                if not with_ads:  # the person's one chat is the graph's: no ADS to ask, its form never opens
                    assert page.inner_text("#ads-pill") == "ADS вимкнено" and page.is_disabled(".sig-pop .pp-ads")
                    assert page.get_attribute(".sig-pop .pp-ads", "title").startswith("ADS вимкнено.")  # the pill's own state, not «недоступний»
                    page.set_viewport_size({"width": 1280, "height": 900})  # a laptop: the developer's chips fold away, this pill stays
                    assert page.is_visible("#ads-pill")
                    assert page.is_visible(".chat-tabs [data-tab='graph']") and not page.is_visible(".chat-tabs [data-tab='ads']")
                    page.evaluate("openAds()")
                    assert page.evaluate("S.tab === 'graph' && !$('ads-dialog').open")
                    page.close()
                    continue
                assert page.inner_text("#ads-pill") == "ADS активний"
                spot, box = page.evaluate(off), page.locator("#map").bounding_box()
                assert page.evaluate("GeoMap.zones()[0].cells.length") == 1 and page.evaluate("GeoMap.closePopup()")
                page.mouse.move(box["x"] + spot["x"], box["y"] + spot["y"])
                page.wait_for_function("G.map.getCanvas().style.cursor === 'pointer'")  # the polygon says it can be clicked
                page.evaluate("G.zones[0].cells = new Set(['8a2a1072b59ffff'])")  # a zone this map draws no cell of, as on another level
                page.mouse.click(box["x"] + spot["x"], box["y"] + spot["y"])  # the polygon is no popup there and keeps its zone
                assert page.evaluate("GeoMap.zones()[0].cells") == ["8a2a1072b59ffff"] and not page.evaluate("GeoMap.closePopup()")
                page.evaluate(draw)
                assert page.evaluate("GeoMap.closePopup()")
                page.mouse.click(box["x"] + spot["x"], box["y"] + spot["y"])  # anywhere in the polygon opens the zone's popup, not only on its cells
                assert not page.is_visible(".sig-pop .pp-ask") and not page.is_visible(
                    ".chat-tabs [data-tab='graph']"
                )  # the graph's chat: the developer's
                page.click(".sig-pop .pp-ads")
                whole = page.input_value("#ads-question")  # no period chosen: the bar's whole span, its last day included
                assert whole.startswith("Період: 2019-03-04 — 2019-0") and whole.endswith(" (включно)\n\n"), whole
                page.keyboard.press("Escape")  # the form closes alone: the map keeps its zones
                assert page.evaluate("!$('ads-dialog').open && GeoMap.zones().length === 1")
                page.evaluate("S.period = {start: '2026-08-09T06:00', end: '2026-08-10T18:00'}")
                page.mouse.click(box["x"] + spot["x"], box["y"] + spot["y"])
                page.click(".sig-pop .pp-ads")
                assert page.input_value("#ads-question") == "Період: 2026-08-09 06:00 — 2026-08-10 18:00 (кінець не включно)\n\n"
                page.evaluate("S.period = null")
                assert page.evaluate("[...document.querySelectorAll('#ads-tables input')].map((i) => i.checked)") == [True]
                zone = page.evaluate("GeoMap.zones()[0].id")  # the popup's zone, ticked
                assert page.evaluate("[...document.querySelectorAll('#ads-zones input:checked')].map((i) => i.value)") == [zone]
                many = "Object.assign(S.adsZones, Object.fromEntries([...Array(60)].map((_, i) => [`X${i}`, GeoMap.zones()[0].polygon])))"
                ids = page.evaluate(f"{many}; adsBody().zones.map((z) => z.id)")  # a long conversation: the bridge takes 50, the oldest go
                assert len(ids) == 50 and ids[-1] == zone and ids[0] == "X11" and page.evaluate("S.adsZones = {}; true")
                coarser = "G.datasets.push({...G.datasets[0], dataset_id: 'coarser', resolution: G.datasets[0].resolution - 1}); openAds()"
                page.evaluate(coarser)  # another level of the same upload: its rows go once
                assert page.evaluate("[...document.querySelectorAll('#ads-tables input')].map((i) => i.value)") == [page.evaluate("GeoMap.current()")]
                page.evaluate(
                    "S.datasets[GeoMap.current()] = {...S.datasets[GeoMap.current()], cross_layers: ['bars']}; openAds()"
                )  # a link run's layer
                assert page.evaluate("[...document.querySelectorAll('#ads-tables input[data-layer]')].map((i) => [i.dataset.layer, i.checked])") == [
                    ["bars", True]
                ]
                page.uncheck("#ads-tables input:not([data-layer])")
                assert page.evaluate("adsBody().tables") == []  # a layer goes with its own table alone
                assert page.is_disabled("#ads-tables input[data-layer]")  # and its box is off meanwhile
                page.evaluate("delete S.datasets[GeoMap.current()].cross_layers")
                page.evaluate("G.datasets.pop(); openAds()")
                page.wait_for_function("$('ads-preview').textContent.includes('Період:')")  # previewed: the period line alone asks nothing
                assert page.is_disabled("#ads-send")
                page.fill("#ads-question", "Скільки подій у зоні?")
                page.wait_for_function("$('ads-preview').textContent.includes('Скільки подій у зоні?') && !$('ads-send').disabled")
                assert "(after the ask)" in page.text_content("#ads-preview")  # folded: read, not shown
                page.uncheck("#ads-zones input")  # no zone ticked: the whole map's question
                page.wait_for_function("$('ads-preview').textContent.includes('Analyse the whole map') && !$('ads-send').disabled")
                page.check("#ads-zones input")
                page.wait_for_function("!$('ads-preview').textContent.includes('Analyse the whole map')")
                page.fill("#question", "Скільки подій у зоні?")  # the box's text, which the form took
                page.click("#ads-send")
                page.wait_for_function("S.tab === 'ads' && S.chat.at(-1).ads && S.chat.at(-1).ads.stage === 'done'", timeout=30000)
                assert page.input_value("#question") == ""
                card = page.locator(".msg-bot").last
                assert card.locator(".ads-stage.ok").count() == 3 and card.locator(".md-table td").all_inner_texts() == ["у зоні", "3"]
                assert f"зона {zone}" in page.locator(".msg-user").last.inner_text() and page.evaluate("S.adsSession") == SID
                assert page.evaluate("S.chat.at(-1).ads.ads_query") == seen["query"]
                page.evaluate("GeoMap.pick(Object.keys(G.patterns)[0])")  # the person's insight from the map: its focus and card alone
                page.wait_for_function("G.focusId && !$('map-card').hidden")
                page.wait_for_timeout(500)  # a drawer would open after its node's read
                assert page.evaluate("$('drawer').hidden && !S.threads.graph.length")
                table = next(f for f in seen["files"] if f.startswith("points_"))
                assert set(seen["files"]) == {table, "zone_cells.csv", "zones.md", "metadata.md"}  # no row stamps: the zone is its cells
                cell = next(c for c in seen["files"][table].splitlines()[0].split(",") if c.startswith("h3_r"))  # the table's finest cell column
                head, *listed = seen["files"]["zone_cells.csv"].splitlines()
                # the drawn zone's cells alone, on the table's own cell column
                assert head == f"zone_id,{cell}" and {line.split(",")[0] for line in listed} == {zone}
                assert seen["files"][table].splitlines()[0].startswith("row_id,")  # the graph service's rows, numbered
                page.fill("#question", "А по годинах?")  # the ADS tab's box: the conversation's next question, the same session
                page.press("#question", "Enter")
                page.wait_for_function("S.chat.length === 2 && S.chat.at(-1).ads.stage === 'done'", timeout=30000)
                assert seen["sessions"] == 1 and seen["uploads"].count(table) == 1 and seen["query"].count("А по годинах?") == 1
                asked = page.evaluate("S.adsLast.question")  # headed by the bar's period, as the form heads the first: ADS bounds by it
                assert asked.startswith("Період: 2019-03-04 — ") and asked.endswith("\n\nА по годинах?"), asked
                page.evaluate("S.period = {start: '2026-08-09T06:00', end: '2026-08-10T18:00'}")  # the period changed: the form again
                page.fill("#question", "А вночі?")
                page.press("#question", "Enter")
                assert page.evaluate("$('ads-dialog').open && S.chat.length === 2") and "Змінилося: період." in page.inner_text("#ads-status")
                assert page.input_value("#ads-question").endswith("\nА вночі?") and page.evaluate("S.period = null; $('ads-dialog').close()") is None
                page.click("#new-chat")  # the conversation ends: ADS deletes its tables, the session closes
                page.wait_for_function("!S.adsSession && S.chat.length === 0")
                assert page.is_visible(".chat-tabs [data-tab='ads']")
                page.route("**/api/ads/health", lambda route: route.fulfill(status=502, body="bad gateway"))
                page.evaluate("refreshHealth()")  # the bridge answered on this page before: no answer now is a failure, not a stack without ADS
                assert page.inner_text("#ads-pill") == "ADS недоступний" and seen["closed"] == SID
                assert page.evaluate("S.tab") == "ads"  # a stack with ADS keeps the person's chat while ADS is down
                assert page.evaluate("$('map-tab').hidden = true; syncTabs(); S.tab") == "graph"  # no map, no zone to ask ADS about
                page.evaluate("$('map-tab').hidden = false; syncTabs()")
                page.fill("#question", "А тепер?")
                page.press("#question", "Enter")  # ADS down: the turn says so at once, the text not left behind
                assert page.evaluate("S.chat.at(-1).ads.error.code") == "ads_unavailable" and page.input_value("#question") == ""
                assert not errors, errors
                page.close()
        browser.close()
