"""The ADS bridge (docs/12_architecture.md): a zone's question through the graph, ADS and the final report, against one
stub that plays the narrator, the graph service's zone rows, the ADS orchestrator and its LLM gateway."""

from __future__ import annotations

import json
import time

import pytest
from conftest import make_engine, serve, stack, upload_levels
from dev import chromium
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from ads_bridge_service.app import create_app
from ads_bridge_service.settings import AdsSettings

POLYGON = {"type": "Polygon", "coordinates": [[[30.0, 50.0], [30.2, 50.0], [30.2, 50.1], [30.0, 50.1], [30.0, 50.0]]]}
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

    @app.post("/api/geo/{dataset_id}/aoi")
    def aoi(dataset_id: str, body: dict):
        seen.setdefault("aoi", []).append((dataset_id, body))
        if dataset_id != "ds-a" and dataset_id != "ds-b":
            return JSONResponse({"detail": f"{dataset_id} is not a geo dataset", "code": "unknown_dataset"}, status_code=404)
        base = dataset_id == "ds-a"  # the base: analysed at 8, with a time column; the layer: at 5, a filtered reading
        counts = {
            "dataset_resolution": 8 if base else 5,
            "time": "seen_at" if base else None,
            "where": None if base else "kind=bar",
            "total": 5,
            "inside": 3,
            "adjacent": 1,
            "far": 1,
            "resolution": 8,
            "zone_cells": 2,
            "columns": ["h3_r8", "h3_r7", "in_aoi", "aoi_zone"],
        }
        return Response(f"rows of {dataset_id}".encode(), media_type="text/csv", headers={"X-Aoi": json.dumps(counts)})

    @app.get("/health")
    def health():
        return {}

    @app.post("/sessions")
    def session():
        return {"session_id": "s1"}

    @app.post("/sessions/{sid}/datasets")
    async def dataset(sid: str, file: UploadFile):
        seen.setdefault("files", {})[file.filename] = (await file.read()).decode("utf-8")
        return {"session_id": sid}

    @app.post("/sessions/{sid}/jobs")
    def job(sid: str, body: dict):
        seen["query"] = body["user_query"]
        return {"job_id": "j1"}

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


def test_a_zones_question_runs_through_the_graph_ads_and_the_final_report():
    seen: dict = {}
    tables = [{"dataset_id": "ds-a", "filename": "events.csv"}, {"dataset_id": "ds-b", "filename": "bars.csv"}]
    ask = {"question": "Скільки подій у зоні?", "tables": tables, "polygon": POLYGON, "cells": ["c1", "c2"]}
    with serve(upstream(seen, polls=3, flaky=2)) as url, TestClient(create_app(settings(url))) as client:  # two bad status reads, read again
        assert client.get("/api/ads/health").json()["ok"]
        parts = client.post("/api/ads/preview", json={"question": "Q?", "tables": tables}).json()["parts"]
        assert [p["kind"] for p in parts] == ["primer", "frame", "user", "data", "insights", "guidance"] and parts[2]["text"] == "Q?"
        assert (
            "base_aoi.csv: events.csv" in parts[3]["text"]
            and "context_aoi.csv: bars.csv" in parts[3]["text"]
            and "(after the ask)" in parts[4]["text"]
        )

        started = client.post("/api/ads/jobs", json=ask)
        assert started.status_code == 202
        job = finished(client, started.json()["job_id"])
        assert job["stage"] == "done" and set(job["timings"]) == {"search", "ads", "joint"}, job
        # the graph is asked with the zone and no model; the text found nothing inside, so the zone alone stands
        assert seen["chat"] == [
            {"question": ask["question"], "cells": ["c1", "c2"], "use_llm": False},
            {"question": "", "cells": ["c1", "c2"], "use_llm": False},
        ]
        assert [(i["key"], i["inside"]) for i in job["insights"]] == [("P1", True), ("P2", False), ("P4", False)]  # the zone's own first; P3 folded
        # the layer's cells at the base table's resolution: one column to join on
        assert seen["aoi"] == [("ds-a", {"polygon": POLYGON, "resolution": None}), ("ds-b", {"polygon": POLYGON, "resolution": 8})]
        assert set(seen["files"]) == {"base_aoi.csv", "context_aoi.csv", "aoi.md"} and seen["files"]["base_aoi.csv"] == "rows of ds-a"
        assert "base_aoi.csv (events.csv)" in seen["files"]["aoi.md"] and "events.csv: O1" in seen["files"]["aoi.md"]
        query = seen["query"]
        assert query == job["ads_query"] and ask["question"] in query and "inside=3, adjacent=1, far=1; cells in the zone: 2" in query
        assert "O1 · in zone: yes" in query and "established: points: strong increase" in query and "O2 · in zone: no" in query
        assert "file: other.csv (another dataset of the graph" in query and query.count("another dataset of the graph") == 1
        assert "O3 · in zone: no" in query and "level: the table level of the base table's file" in query and query.count("  level: ") == 1
        assert "bars.csv, only its rows where kind=bar, 5 rows" in query and "Time: seen_at in df_base_aoi" in query
        assert "SIG analysed this table at H3 resolution 5" in seen["files"]["aoi.md"]
        assert job["ads_report"] == "ADS: 3 events plot" and seen["closed"] == "s1"  # no link into ADS's own workspace
        assert job["joint_report"] == FINAL and "ADS: 3 events plot" in seen["prompt"] and "Словник термінів" in seen["prompt"]

        assert client.post("/api/ads/jobs", json=ask | {"question": " "}).status_code == 400
        assert client.post("/api/ads/jobs", json=ask | {"cells": []}).json()["code"] == "empty_zone"
        assert client.post("/api/ads/jobs", json=ask | {"tables": []}).status_code == 422
        escaping = ask | {"tables": [{"dataset_id": "../reset?x=", "filename": "x.csv"}]}  # an id is one path segment, never a path
        assert finished(client, client.post("/api/ads/jobs", json=escaping).json()["job_id"])["error"]["code"] == "sig_failed" and "reset" not in seen
        assert client.get("/api/ads/jobs/nope").status_code == 404


def test_a_failed_stage_is_recorded_on_the_job_with_its_code():
    ask = {"question": "Q?", "tables": [{"dataset_id": "ds-a", "filename": "events.csv"}], "polygon": POLYGON, "cells": ["c1"]}
    with serve(upstream({}, polls=10**6)) as url, TestClient(create_app(settings(url, ads_job_timeout_s=0.05))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert (job["stage"], job["failed_stage"], job["error"]["code"]) == ("error", "ads", "ads_timeout")
    seen: dict = {}
    with serve(upstream(seen, flaky=10**6)) as url, TestClient(create_app(settings(url))) as client:  # the status never reads
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert (job["failed_stage"], job["error"]["code"]) == ("ads", "ads_failed") and seen["closed"] == "s1"  # the session closed all the same
    with serve(upstream({}, final=None)) as url, TestClient(create_app(settings(url))) as client:  # no final report: ADS's stands
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["stage"] == "done" and job["joint_report"] == "" and "503" in job["joint_error"] and job["ads_report"]
    with serve(upstream({}, report="")) as url, TestClient(create_app(settings(url))) as client:
        job = finished(client, client.post("/api/ads/jobs", json=ask).json()["job_id"])
        assert job["error"]["code"] == "ads_failed" and "no report" in job["error"]["detail"]
        gone = ask | {"tables": [{"dataset_id": "ds-gone", "filename": "gone.csv"}]}
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


@pytest.mark.browser
def test_the_console_asks_ads_about_a_drawn_zone(cfg, points_csv):
    """The header says whether ADS answers; a drawn zone's popup opens the form (the open dataset ticked, the task
    previewed), and the ask shows its stages and then the report, a table included, in the chat. Without the bridge
    the pill says so and the button is off."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    upload_levels(engine, points_csv, geo="auto@8")
    seen: dict = {}
    table = "## Результати аналізу\n\n| група | рядків |\n|---|---|\n| у зоні | 3 |\n\n## Методологія\n\nполігон"
    draw = "GeoMap.setPolygon(G.data.features[0].geometry)"
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
                page.click(".sig-pop .pp-ads")
                assert page.evaluate("[...document.querySelectorAll('#ads-tables input')].map((i) => i.checked)") == [True]
                assert page.is_disabled("#ads-send")  # no question yet
                page.fill("#ads-question", "Скільки подій у зоні?")
                page.wait_for_function("$('ads-preview').textContent.includes('Скільки подій у зоні?') && !$('ads-send').disabled")
                assert "(after the ask)" in page.text_content("#ads-preview")  # folded: read, not shown
                page.click("#ads-send")
                page.wait_for_function("S.chat.at(-1).ads && S.chat.at(-1).ads.stage === 'done'", timeout=30000)
                card = page.locator(".msg-bot").last
                assert card.locator(".ads-stage.ok").count() == 3 and card.locator(".md-table td").all_inner_texts() == ["у зоні", "3"]
                assert "зона на мапі" in page.locator(".msg-user").last.inner_text()
                assert page.evaluate("S.chat.at(-1).ads.ads_query") == seen["query"] and set(seen["files"]) == {"base_aoi.csv", "aoi.md"}
                assert seen["files"]["base_aoi.csv"].splitlines()[0].endswith("in_aoi,aoi_zone")  # the graph service's rows of the zone
                assert not errors, errors
                page.close()
        browser.close()
