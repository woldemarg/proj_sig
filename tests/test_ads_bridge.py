"""The ADS bridge (docs/12_architecture.md): a conversation's questions about zones through the graph, ADS and the final
report, against one stub that plays the narrator, the graph service's exports, the ADS orchestrator and its LLM gateway."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from conftest import make_engine, serve, stack, upload_levels
from dev import chromium
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from ads_bridge_service.app import _PERIODS, create_app, error_code
from ads_bridge_service.settings import AdsSettings
from insight_contracts.wording import PERIOD

POLYGON = {"type": "Polygon", "coordinates": [[[30.0, 50.0], [30.2, 50.0], [30.2, 50.1], [30.0, 50.1], [30.0, 50.0]]]}
OTHER = {"type": "Polygon", "coordinates": [[[31.0, 50.0], [31.2, 50.0], [31.2, 50.1], [31.0, 50.1], [31.0, 50.0]]]}
SID = "11111111-2222-4333-8444-555555555555"
TABLES = [{"dataset_id": "ds-a", "filename": "events.csv", "resolution": 8}, {"dataset_id": "ds-b", "filename": "bars.csv", "resolution": 5}]
FINAL = "## Результати аналізу\nу зоні 3 події\n## Методологія\nполігон"


def item(key: str, credibility: int, dataset: str = "ds-a", resolution: int | None = 8, **more) -> dict:
    return {
        "provenance": {"dataset_id": dataset, "spatial": {"resolution": resolution} if resolution else None},
        **more,
        "key": key,
        "pattern_id": f"P-{key}",
        "scope": ["lisa_points=hot spot"],
        "scope_text": "lisa points is hot spot",
        "shift_text": ["points: strong increase, +2.1 sd (median 9 vs 3 overall)"],
        "validation": "adjusted p 0.001",
        "trust": "source F",
        "credibility": credibility,
    }


def upstream(
    seen: dict, polls: int = 1, report: str = "ADS: 3 events ![plot](/workspace/a.png)", final: str | None = FINAL, flaky: int = 0
) -> FastAPI:
    """``polls``: status reads until the job ends; ``flaky``: 502s before the first of them; ``final`` None: the gateway is down."""
    app = FastAPI()
    left = {"polls": polls, "flaky": flaky}

    @app.post("/api/chat/query")
    def chat(body: dict):
        seen.setdefault("chat", []).append(body)
        inside = {} if body["question"] else {"P1": 0.6}  # the text alone finds nothing in the zone
        files = {"P1": "events.csv", "P2": "other.csv", "P3": "events.csv", "P4": "events.csv"}
        provenance = [{"key": k, "filename": f} for k, f in files.items()]
        # P3: P1 again at another level, folded into it; P4: the base file's table level, a dataset of its own
        items = [item("P2", 1, "ds-x"), item("P3", 6, "ds-a5", 5, folded_into="P1"), item("P4", 6, "ds-t", None), item("P1", 6)]
        return {"view": {"evidence": {"items": items, "area": inside, "provenance": provenance}}}

    def table(dataset_id: str) -> dict | None:
        if dataset_id not in ("ds-a", "ds-b"):
            return None
        base = dataset_id == "ds-a"  # the base: analysed at 8, with a time column; the layer: at 5, a filtered reading
        meta = {"dataset_resolution": 8 if base else 5, "time": "seen_at" if base else None, "where": None if base else "kind=bar"}
        return meta | {"total": 5, "resolution": 8, "columns": ["row_id", "h3_r8", "h3_r7", "h3_r6"]}

    def unknown(dataset_id: str) -> JSONResponse:
        return JSONResponse({"detail": f"{dataset_id} is not a geo dataset", "code": "unknown_dataset"}, status_code=404)

    @app.post("/api/geo/{dataset_id}/zone_rows")
    def zone_rows(dataset_id: str, body: dict):
        seen.setdefault("zone_rows", []).append((dataset_id, body))
        if (meta := table(dataset_id)) is None:
            return unknown(dataset_id)
        zones = {z["id"]: {"inside": 3, "adjacent": 1, "far": 1, "zone_cells": 5} for z in body["zones"]}
        rows = "".join(f"{z},0,inside\n{z},3,adjacent\n" for z in zones)
        return Response(f"zone_id,row_id,aoi_zone\n{rows}", media_type="text/csv", headers={"X-Zones": json.dumps(meta | {"zones": zones})})

    @app.get("/api/geo/{dataset_id}/rows")
    def rows(dataset_id: str, resolution: int):
        seen.setdefault("rows", []).append((dataset_id, resolution))
        return Response(f"rows of {dataset_id}".encode(), media_type="text/csv") if table(dataset_id) else unknown(dataset_id)

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
    def delete(sid: str, body: dict):
        seen.setdefault("deleted", []).append(body["path"])
        return {"deleted": True}

    @app.post("/sessions/{sid}/jobs")
    def job(sid: str, body: dict):
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
        return {"choices": [{"message": {"content": final}}]}

    return app


def finished(client: TestClient, job_id: str) -> dict:
    for _ in range(200):
        job = client.get(f"/api/ads/jobs/{job_id}").json()
        if job["stage"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(job)


def settings(url: str, **over) -> AdsSettings:
    return AdsSettings(**{"ads_url": url, "ads_llm_url": url, "ads_graph_url": url, "ads_narrator_url": url, "ads_poll_s": 0.01} | over)


def test_a_conversation_asks_ads_in_one_session_and_sends_a_table_once():
    seen: dict = {}
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
        # the graph is asked with the polygon the rows are read against, and no model; the text found nothing inside, so the zone alone stands
        assert seen["chat"] == [
            {"question": ask["question"], "geometry": POLYGON, "use_llm": False},
            {"question": "", "geometry": POLYGON, "use_llm": False},
        ]
        assert [(i["key"], i["zones"]) for i in job["insights"]] == [("P1", {"Z1": True}), ("P2", {"Z1": False}), ("P4", {"Z1": False})]
        # every table's zones and rows at the base table's resolution: one cell column to join on
        assert [(d, b["resolution"]) for d, b in seen["zone_rows"]] == [("ds-a", 8), ("ds-b", 8)] and seen["rows"] == [("ds-a", 8), ("ds-b", 8)]
        assert seen["uploads"] == [*files, "zones.csv", "zones.md"] and seen["files"][files[0]] == "rows of ds-a"
        assert seen["files"]["zones.csv"].splitlines() == ["zone_id,table,row_id,aoi_zone"] + [
            f"Z1,{f},{r}" for f in files for r in ("0,inside", "3,adjacent")
        ]
        assert f"{files[0]} (events.csv)" in seen["files"]["zones.md"] and "events.csv: O1" in seen["files"]["zones.md"]
        assert "Zone Z1: a polygon" in seen["files"]["zones.md"] and "SIG analysed this table at H3 resolution 5" in seen["files"]["zones.md"]
        query = seen["query"]
        assert query == job["ads_query"] and ask["question"] in query and "Z1: inside=3, adjacent=1, far=1, cells=5" in query
        assert "O1 · in zone: Z1 yes" in query and "established: points: strong increase" in query and "O2 · in zone: Z1 no" in query
        assert "file: other.csv (another dataset of the graph" in query and query.count("another dataset of the graph") == 1
        assert "O3 · in zone: Z1 no" in query and "level: the table level of the base table's file" in query and query.count("  level: ") == 1
        assert "bars.csv, only its rows where kind=bar, 5 rows" in query and "Time: seen_at in df_events_dsa_r8" in query
        assert "h3_r8, h3_r7, h3_r6 are H3 cell ids" in query and "left-join them to the table on row_id" in query
        assert job["ads_report"] == "ADS: 3 events plot" and "closed" not in seen  # no link into ADS's own workspace; the session stays
        assert job["joint_report"] == FINAL and "ADS: 3 events plot" in seen["prompt"] and "Словник термінів" in seen["prompt"]
        assert "на рівні H3 8 базового шару: Z1 5" in seen["prompt"]  # the methodology counts the zone the rows were read against

        # the next question of the conversation: the same session, the tables it holds not sent again, the zones replaced
        two = ask | {
            "session_id": SID,
            "zones": [*ask["zones"], {"id": "Z2", "polygon": OTHER}],
            "focus": ["Z2", "Z1"],
            "question": "Чим Z2 відрізняється?",
        }
        job = finished(client, client.post("/api/ads/jobs", json=two).json()["job_id"])
        assert job["stage"] == "done" and job["session_id"] == SID and seen["sessions"] == 1 and "closed" not in seen
        assert seen["uploads"][4:] == ["zones.csv", "zones.md"] and len(seen["rows"]) == 2  # the tables went once
        assert {line.split(",")[0] for line in seen["files"]["zones.csv"].splitlines()[1:]} == {"Z1", "Z2"} and "Zone Z2" in seen["files"]["zones.md"]
        assert seen["query"].count("Compare the areas of interest Z2, Z1: ") == 1 and "O1 · in zone: Z2 yes, Z1 yes" in seen["query"]
        lone = ask | {"session_id": SID, "zones": two["zones"]}  # Z2 is the conversation's, the question is about Z1 alone
        assert (
            "The conversation's other zones (Z2) are in df_zones too"
            in finished(client, client.post("/api/ads/jobs", json=lone).json()["job_id"])["ads_query"]
        )
        seen["busy"] = True  # the conversation's last question still runs in ADS
        assert finished(client, client.post("/api/ads/jobs", json=two).json()["job_id"])["error"]["code"] == "ads_busy"

        # the graph reads the period bar alone: a period in the text is ADS's, the bar's goes beside it
        seen["chat"].clear()
        seen["busy"] = False
        dated = ask | {
            "question": "Період: 2026-08-09 — 2026-08-17 (включно)\nСкільки подій у 2026-W33 чи Q3 2026?",
            "period": ["2026-08-09T00:00", "2026-08-18T00:00"],
        }
        job = finished(client, client.post("/api/ads/jobs", json=dated).json()["job_id"])
        sent = seen["chat"][0]
        assert sent["period"] == dated["period"] and sent["question"].split() == ["Скільки", "подій", "у", "чи", "?"]
        assert dated["question"] in job["ads_query"] and "A period the question names bounds every figure" in job["ads_query"]

        # the conversation's end: its tables deleted from ADS, the session closed
        assert client.post(f"/api/ads/sessions/{SID}/close").json() == {"session_id": SID, "deleted": 3, "closed": True}
        assert seen["closed"] == SID and sorted(seen["deleted"]) == sorted(f"/workspace/{SID}/{f}" for f in (*files, "zones.csv"))
        assert client.post("/api/ads/sessions/..%2Freset/close").status_code in (404, 422) and "reset" not in seen

        assert client.post("/api/ads/jobs", json=ask | {"question": " "}).status_code == 400
        assert client.post("/api/ads/jobs", json=ask | {"question": "Період: 2026-08-09 — 2026-08-17 (включно)\n "}).status_code == 400
        assert client.post("/api/ads/jobs", json=ask | {"zones": []}).status_code == 422
        assert client.post("/api/ads/jobs", json=ask | {"focus": ["Z9"]}).status_code == 422  # a zone the conversation lacks
        assert client.post("/api/ads/jobs", json=ask | {"focus": [f"Z{n}" for n in range(6)]}).status_code == 422  # five at most
        assert client.post("/api/ads/jobs", json=ask | {"session_id": "../x"}).status_code == 422
        assert client.post("/api/ads/jobs", json=ask | {"tables": []}).status_code == 422
        escaping = ask | {"tables": [{"dataset_id": "../reset?x=", "filename": "x.csv", "resolution": 8}]}  # an id is one path segment, never a path
        assert finished(client, client.post("/api/ads/jobs", json=escaping).json()["job_id"])["error"]["code"] == "sig_failed" and "reset" not in seen
        assert client.get("/api/ads/jobs/nope").status_code == 404


def test_the_bridge_strips_the_periods_the_graph_reads():
    """The bridge's copy of ``wording.PERIOD`` finds what the graph would read as a period, so none overrides the bar's."""
    samples = "2031-W13, 2031-q1, Q1 2031, 2031-03-24 10:00, 2031-03; not 2019-20, x2031-03, 2031-03-24-1, 12031-03"
    assert [m.group(0) for m in _PERIODS.finditer(samples)] == [m.group(0) for m in PERIOD.finditer(samples)]


def test_a_failed_stage_is_recorded_on_the_job_with_its_code():
    ask = {"question": "Q?", "tables": TABLES[:1], "zones": [{"id": "Z1", "polygon": POLYGON}], "focus": ["Z1"]}
    seen: dict = {}
    with serve(upstream(seen, polls=10**6)) as url, TestClient(create_app(settings(url, ads_job_timeout_s=0.05))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert (job["stage"], job["failed_stage"], job["error"]["code"]) == ("error", "ads", "ads_timeout")
        assert seen["cancelled"] == SID == job["session_id"]  # the job given up is cancelled: the session takes the next question
    with serve(upstream({}, flaky=10**6)) as url, TestClient(create_app(settings(url))) as client:  # the status never reads
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert (job["failed_stage"], job["error"]["code"]) == ("ads", "ads_failed")
    with serve(upstream({}, final=None)) as url, TestClient(create_app(settings(url))) as client:  # no final report: ADS's stands
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["stage"] == "done" and job["joint_report"] == "" and "503" in job["joint_error"] and job["ads_report"]
    with serve(upstream({}, report="")) as url, TestClient(create_app(settings(url))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["error"]["code"] == "ads_failed" and "no report" in job["error"]["detail"]
        gone = ask | {"tables": [{"dataset_id": "ds-gone", "filename": "gone.csv", "resolution": 8}]}
        job = finished(client, client.post("/api/ads/jobs", json=gone).json()["job_id"])
        assert (job["failed_stage"], job["error"]["code"]) == ("search", "sig_failed") and "unknown_dataset" in job["error"]["detail"]
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
    seen: dict = {}
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
                if not with_ads:
                    assert page.inner_text("#ads-pill") == "ADS вимкнено" and page.is_disabled(".sig-pop .pp-ads")
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
                page.click(".sig-pop .pp-ads")
                whole = page.input_value("#ads-question")  # no period chosen: the bar's whole span, its last day included
                assert whole.startswith("Період: 2019-03-04 — 2019-0") and whole.endswith(" (включно)\n"), whole
                page.click("#ads-cancel")
                page.evaluate("S.period = {start: '2026-08-09T06:00', end: '2026-08-10T18:00'}")
                page.mouse.click(box["x"] + spot["x"], box["y"] + spot["y"])
                page.click(".sig-pop .pp-ads")
                assert page.input_value("#ads-question") == "Період: 2026-08-09 06:00 — 2026-08-10 18:00 (кінець не включно)\n"
                page.evaluate("S.period = null")
                assert page.evaluate("[...document.querySelectorAll('#ads-tables input')].map((i) => i.checked)") == [True]
                zone = page.evaluate("GeoMap.zones()[0].id")  # the popup's zone, ticked
                assert page.evaluate("[...document.querySelectorAll('#ads-zones input:checked')].map((i) => i.value)") == [zone]
                coarser = "G.datasets.push({...G.datasets[0], dataset_id: 'coarser', resolution: G.datasets[0].resolution - 1}); openAds()"
                page.evaluate(coarser)  # another level of the same upload: its rows go once
                assert page.evaluate("[...document.querySelectorAll('#ads-tables input')].map((i) => i.value)") == [page.evaluate("GeoMap.current()")]
                page.evaluate("G.datasets.pop(); openAds()")
                page.wait_for_function("$('ads-preview').textContent.includes('Період:')")  # previewed: the period line alone asks nothing
                assert page.is_disabled("#ads-send")
                page.fill("#ads-question", "Скільки подій у зоні?")
                page.wait_for_function("$('ads-preview').textContent.includes('Скільки подій у зоні?') && !$('ads-send').disabled")
                assert "(after the ask)" in page.text_content("#ads-preview")  # folded: read, not shown
                page.click("#ads-send")
                page.wait_for_function("S.tab === 'ads' && S.chat.at(-1).ads && S.chat.at(-1).ads.stage === 'done'", timeout=30000)
                card = page.locator(".msg-bot").last
                assert card.locator(".ads-stage.ok").count() == 3 and card.locator(".md-table td").all_inner_texts() == ["у зоні", "3"]
                assert f"зона {zone}" in page.locator(".msg-user").last.inner_text() and page.evaluate("S.adsSession") == SID
                assert page.evaluate("S.chat.at(-1).ads.ads_query") == seen["query"]
                table = next(f for f in seen["files"] if f.startswith("points_"))
                assert set(seen["files"]) == {table, "zones.csv", "zones.md"} and seen["files"]["zones.csv"].startswith(
                    "zone_id,table,row_id,aoi_zone\n"
                )
                assert seen["files"][table].splitlines()[0].startswith("row_id,")  # the graph service's rows, a row_id to join the zones on
                page.fill("#question", "А по годинах?")  # the ADS tab's box: the conversation's next question, the same session
                page.press("#question", "Enter")
                page.wait_for_function("S.chat.length === 2 && S.chat.at(-1).ads.stage === 'done'", timeout=30000)
                assert seen["sessions"] == 1 and seen["uploads"].count(table) == 1 and seen["query"].count("А по годинах?") == 1
                page.click("#new-chat")  # the conversation ends: ADS deletes its tables, the session closes
                page.wait_for_function("!S.adsSession && S.chat.length === 0")
                assert page.is_visible(".chat-tabs [data-tab='ads']")
                page.route("**/api/ads/health", lambda route: route.fulfill(status=502, body="bad gateway"))
                page.evaluate("refreshHealth()")  # the bridge answered on this page before: no answer now is a failure, not a stack without ADS
                assert page.inner_text("#ads-pill") == "ADS недоступний" and seen["closed"] == SID
                assert not errors, errors
                page.close()
        browser.close()
