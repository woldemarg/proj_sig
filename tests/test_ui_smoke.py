"""UI smoke test: dataset loads through the API and the graph page renders (docs/08_interface.md)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from conftest import FakeLLM, make_config
from fastapi.testclient import TestClient

from ltir.pipeline import Engine
from ltir.web.app import create_app


def _wait_ready(client, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        batches = client.get("/api/batches").json()
        if batches and all(b["status"] in {"READY", "FAILED", "SKIPPED"} for b in batches):
            return batches
        time.sleep(0.2)
    raise AssertionError("batch did not finish")


@pytest.fixture
def client(tmp_path):
    cfg = make_config(tmp_path / "ws")
    app = create_app(cfg, Engine(cfg, llm=FakeLLM()))
    with TestClient(app) as c:
        yield c


def test_upload_process_render_query(client, demo_csv):
    assert "cytoscape.min.js" in client.get("/").text and 'id="cy"' in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/vendor/cytoscape.min.js").status_code == 200
    with open(demo_csv, "rb") as f:
        r = client.post("/api/upload", files={"file": ("retail.csv", f, "text/csv")}, data={"bins": ""})
    assert r.status_code == 202 and r.json()["status"] == "UPLOADED"
    batches = _wait_ready(client)
    assert batches[0]["status"] == "READY"
    assert batches[0]["metrics"]["validated_insights"] > 0 and batches[0]["profile"]["rows"] == 5000

    g = client.get("/api/graph").json()
    kinds = {n["data"]["kind"] for n in g["nodes"]}
    types = {e["data"]["type"] for e in g["edges"]}
    assert {"Pattern", "Attractor", "Dimension", "Metric"} <= kinds
    assert {"ACTIVATES", "RELATED_TO", "SPECIALIZES", "GENERALIZES", "CONTRASTS", "SIBLING", "HAS_SCOPE", "TARGETS"} <= types
    node_ids = {n["data"]["id"] for n in g["nodes"]}
    assert all(e["data"]["source"] in node_ids and e["data"]["target"] in node_ids for e in g["edges"])

    attractor = next(n["data"]["id"] for n in g["nodes"] if n["data"]["kind"] == "Attractor")
    detail = client.get(f"/api/nodes/{attractor}").json()
    assert detail["node"]["kind"] == "Attractor" and "ACTIVATES (in)" in detail["neighbors"]

    qa = client.post("/api/query", json={"question": "Чому margin нижчий для phones у US?", "use_llm": True}).json()
    assert qa["highlight"]["edges"] and qa["highlight"]["seeds"] and qa["answer"]
    assert qa["evidence"]["parsed"]["targets"] == ["margin"] and qa["evidence"]["parsed"]["direction"] == -1  # literals parsed inside Ukrainian
    assert set(qa["highlight"]["edges"]) <= {e["data"]["id"] for e in g["edges"]}  # highlight ids resolve in the UI graph
    # delete through the API: the dataset, its graph and its record go; the mirror of the UI is the backend
    ds = batches[0]["dataset_id"]
    assert client.delete(f"/api/datasets/{ds}").json()["patterns_removed"] == batches[0]["metrics"]["validated_insights"]
    assert client.get("/api/batches").json() == [] and client.get("/api/graph").json()["nodes"] == []
    assert client.delete(f"/api/datasets/{ds}").status_code == 404
    assert client.post("/api/reset").json()["status"] == "reset"


def _chromium() -> str | None:
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    found = sorted(root.glob("chromium-*/chrome-win64/chrome.exe")) + sorted(root.glob("chromium-*/chrome-linux/chrome"))
    return str(found[-1]) if found else None


@pytest.mark.browser
def test_browser_renders_graph_and_highlights_path(tmp_path, demo_csv):
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = _chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    import uvicorn

    cfg = make_config(tmp_path / "ws", web_port=8799)
    engine = Engine(cfg, llm=FakeLLM())
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    server = uvicorn.Server(uvicorn.Config(create_app(cfg, engine), host="127.0.0.1", port=8799, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.05)
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(executable_path=exe)
            page = browser.new_page(viewport={"width": 1500, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto("http://127.0.0.1:8799/")
            page.wait_for_function("typeof S !== 'undefined' && S.cy && S.cy.nodes('[kind=\"Pattern\"]').length > 0", timeout=20000)
            page.wait_for_selector(".ds .chip.ok", timeout=10000)  # dataset card shows "Ready"
            assert page.inner_text("#health").strip()  # health pills rendered
            # insights table view lists every pattern node
            page.click("#view-switch button[data-view='table']")
            page.wait_for_selector("#ins-table tbody tr[data-id]")
            assert page.evaluate("document.querySelectorAll('#ins-table tbody tr[data-id]').length") == page.evaluate("S.nodes.length")
            page.click("#view-switch button[data-view='graph']")
            # chat: a Ukrainian suggestion -> grounded answer with citations and a highlighted retrieval path
            page.click("text=Чому margin нижчий для phones у US?")
            page.wait_for_selector(".answer-box", timeout=30000)
            assert page.evaluate("S.cy.edges('.hl-edge').length") > 0
            assert page.evaluate("S.cy.nodes('.hl-anchor').length") > 0
            assert page.evaluate("S.cy.nodes('.hl-seed').length") > 0
            assert page.evaluate("document.querySelectorAll('.ev').length") > 0
            # the evidence is the default chat tab; UI chrome stays English
            assert page.inner_text(".bot-tabs button.active").startswith("Evidence & how it was found")
            assert not page.is_hidden(".bot-pane[data-pane='0']") and page.is_hidden(".bot-pane[data-pane='1']")
            # clicking an evidence card opens the details drawer via its citation
            page.click(".answer-box a.cite >> nth=0")
            page.wait_for_selector("#drawer:not([hidden]) .shift-row", timeout=10000)
            assert "insight" in page.inner_text("#drawer-kind").lower()
            # one legend for every view; its link entries toggle the layers in the graph and in the sphere alike
            assert page.is_visible("#legend") and page.is_visible("#legend .toggle[data-layer='activates']")
            page.click("#view-switch button[data-view='sphere']")
            page.wait_for_function(
                "(() => { const w = document.querySelector('#sphere').contentWindow; return w && w.document.querySelector('.plotly-graph-div') && w.document.querySelector('.plotly-graph-div').data; })()",
                timeout=60000,
            )
            trace_visible = "(() => { const gd = document.querySelector('#sphere').contentWindow.document.querySelector('.plotly-graph-div'); return gd.data.filter(t => t.name === 'Memberships').map(t => t.visible !== false); })()"
            assert page.is_visible("#legend") and page.evaluate(trace_visible) == [True]
            assert page.evaluate("document.querySelector('.toggle[data-layer=\"schema\"]').disabled")  # columns have no 3D meaning
            page.click("#legend .toggle[data-layer='activates']")
            assert page.evaluate(trace_visible) == [False]
            page.click("#view-switch button[data-view='graph']")
            hidden, rest = (page.evaluate(f"S.cy.edges('[type=\"ACTIVATES\"]').not('.hl-edge'){f}.length") for f in (".filter('.hidden')", ""))
            assert hidden == rest > 0  # off in the graph too (the answer path stays visible by design)
            page.click("#theme-btn")
            assert page.evaluate("document.documentElement.dataset.theme") in {"dark", "light"}
            page.screenshot(path=str(tmp_path / "ui.png"))
            # deleting the dataset from its card empties the views, through the backend
            page.once("dialog", lambda d: d.accept())
            page.hover(".ds")
            page.click(".ds .ds-del")
            page.wait_for_selector(".empty-rail", timeout=20000)
            assert page.evaluate("S.cy.nodes().length") == 0 and not engine.graph().of_kind("Pattern")
            browser.close()
            assert not errors, errors
    finally:
        server.should_exit = True
        thread.join(timeout=10)
