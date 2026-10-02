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
    # an upload that failed before it had a dataset id is removed by its batch id
    client.post("/api/upload", files={"file": ("notes.json", b"{}", "application/json")})
    failed = _wait_ready(client)[0]
    assert failed["status"] == "FAILED" and failed["dataset_id"] is None
    assert client.delete(f"/api/datasets/{failed['batch_id']}").status_code == 200 and client.get("/api/batches").json() == []
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
            # the answer's panels start closed; a click opens the evidence, a second click closes it; UI chrome stays English
            assert page.evaluate("document.querySelectorAll('.bot-tabs button.active').length") == 0 and page.is_hidden(".bot-pane[data-pane='0']")
            tab = ".bot-tabs button[data-pane='0']"
            assert page.inner_text(tab).startswith("Evidence & how it was found")
            page.click(tab)
            assert page.is_visible(".bot-pane[data-pane='0'] .ev") and page.is_hidden(".bot-pane[data-pane='1']")
            page.click(tab)
            assert page.is_hidden(".bot-pane[data-pane='0']")
            # clicking an evidence card opens the details drawer via its citation
            page.click(".answer-box a.cite >> nth=0")
            page.wait_for_selector("#drawer:not([hidden]) .shift-row", timeout=10000)
            assert "insight" in page.inner_text("#drawer-kind").lower()
            # one legend for every view; by default only Hierarchy and Theme links are on, with live counts
            assert page.is_visible("#legend") and page.is_visible("#legend .toggle[data-key='activates']")
            assert page.evaluate("[...document.querySelectorAll('#legend .toggle.on')].map(b => b.dataset.key)") == ["lattice", "latent"]
            assert int(page.inner_text("[data-n='lattice']")) == page.evaluate("S.cy.edges('[type=\"SPECIALIZES\"]').length") > 0
            page.hover("#legend .key[data-key='cov']")  # spotlight: the rest of the graph dims
            assert page.evaluate("S.cy.elements('.lg-dim').length") > 0
            page.mouse.move(5, 5)
            assert page.evaluate("S.cy.elements('.lg-dim').length") == 0
            # the sphere: the same toggles switch its traces in place, a click on a point opens the drawer
            page.click("#view-switch button[data-view='sphere']")
            sphere = "document.querySelector('#sphere').contentWindow.document.querySelector('.plotly-graph-div')"
            page.wait_for_function(f"(() => {{ const gd = {sphere}; return !!(gd && gd.data && gd.on); }})()", timeout=60000)
            visible = f"(name) => {sphere}.data.filter(t => t.name === name).map(t => t.visible !== false)"
            assert page.evaluate(f"({visible})('Memberships')") == [False] and page.evaluate(f"({visible})('Hierarchy')") == [True]
            assert page.evaluate("document.querySelector('.toggle[data-key=\"schema\"]').disabled")  # columns have no 3D meaning
            page.click("#legend .toggle[data-key='activates']")
            page.click("#legend .toggle[data-key='lattice']")
            assert page.evaluate(f"({visible})('Memberships')") == [True] and page.evaluate(f"({visible})('Hierarchy')") == [False]
            page.evaluate("document.querySelector('#drawer-close').click()")
            pid = page.evaluate("S.nodes[0].id")
            page.evaluate(f"{sphere}.emit('plotly_click', {{ points: [{{ customdata: '{pid}' }}] }})")
            page.wait_for_selector("#drawer:not([hidden]) .shift-row", timeout=10000)
            page.click("#view-switch button[data-view='graph']")
            hidden, rest = (page.evaluate(f"S.cy.edges('[type=\"ACTIVATES\"]').not('.hl-edge'){f}.length") for f in (".filter('.hidden')", ""))
            assert hidden == 0 and rest > 0  # on in the graph too
            # an isolated evidence path stays isolated through a layer switch (it is the current highlight)
            page.click(tab)
            page.click(".bot-pane[data-pane='0'] .ev >> nth=0")
            assert page.evaluate("S.cy.nodes('.hl-evidence').length") == 1
            page.click("#legend .toggle[data-key='contrast']")
            assert page.evaluate("S.cy.nodes('.hl-evidence').length") == 1
            # the column borders can be dragged
            handle = page.locator(".rail .col-resize").bounding_box()
            before = page.evaluate("document.querySelector('.rail').getBoundingClientRect().width")
            page.mouse.move(handle["x"] + 4, handle["y"] + 300)
            page.mouse.down()
            page.mouse.move(handle["x"] + 84, handle["y"] + 300, steps=5)
            page.mouse.up()
            assert abs(page.evaluate("document.querySelector('.rail').getBoundingClientRect().width") - before - 80) < 4
            page.click("#theme-btn")
            assert page.evaluate("document.documentElement.dataset.theme") in {"dark", "light"}
            page.screenshot(path=str(tmp_path / "ui.png"))
            # deleting the dataset from its card empties the views, through the backend
            page.once("dialog", lambda d: d.accept())
            assert page.is_visible(".ds .ds-del")  # always visible, not only on hover
            page.click(".ds .ds-del")
            page.wait_for_selector(".empty-rail", timeout=20000)
            assert page.evaluate("S.cy.nodes().length") == 0 and not engine.graph().of_kind("Pattern")
            browser.close()
            assert not errors, errors
    finally:
        server.should_exit = True
        thread.join(timeout=10)
