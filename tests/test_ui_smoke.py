"""UI smoke test: dataset loads through the API and the graph page renders (docs/08_interface.md)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
import pytest
from conftest import FakeLLM, make_config, make_engine, serve
from dev import compose

from evidence_narrator_service.app import create_app as create_narrator
from evidence_narrator_service.settings import NarratorSettings
from insight_graph_service.core.settings import PROJECT_ROOT
from insight_graph_service.server.app import create_app


def _wait_ready(client):
    deadline = time.time() + 60
    while time.time() < deadline:
        batches = client.get("/api/batches").json()
        if batches and all(b["status"] in {"READY", "FAILED", "SKIPPED"} for b in batches):
            return batches
        time.sleep(0.2)
    raise AssertionError("batch did not finish")


def stack(engine):
    """The graph service and the narrator on one local port, as the console reaches them."""
    from conftest import free_port

    port = free_port()
    narrator = create_narrator(NarratorSettings(insight_graph_url=f"http://127.0.0.1:{port}"), llm=FakeLLM())
    return port, compose(create_app(engine), narrator)


@pytest.fixture
def client(tmp_path):
    cfg = make_config(tmp_path / "ws")
    port, app = stack(make_engine(cfg))
    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as c:
        yield c


def test_upload_process_render_query(client, demo_csv):
    page = client.get("/").text  # the console's static files, as nginx serves them
    assert "/vendor/cytoscape.min.js" in page and "/vendor/plotly-gl3d.min.js" in page and 'id="cy"' in page
    assert all(client.get(path).status_code == 200 for path in ("/app.js", "/style.css", "/vendor/cytoscape.min.js", "/vendor/plotly-gl3d.min.js"))
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
    assert {"ACTIVATES", "RELATED_TO", "SPECIALIZES", "CONTRASTS", "SIBLING", "HAS_SCOPE", "TARGETS"} <= types
    node_ids = {n["data"]["id"] for n in g["nodes"]}
    assert all(e["data"]["source"] in node_ids and e["data"]["target"] in node_ids for e in g["edges"])

    pattern = next(n["data"]["id"] for n in g["nodes"] if n["data"]["kind"] == "Pattern")
    checked = client.post(f"/api/patterns/{pattern}/verify").json()  # recomputed from the stored upload
    assert checked["status"] == "reproduced" and checked["diff"] == [] and "stability" in checked["not_checked"]
    assert client.post("/api/patterns/P-nope/verify").status_code == 404
    attractor = next(n["data"]["id"] for n in g["nodes"] if n["data"]["kind"] == "Attractor")
    detail = client.get(f"/api/nodes/{attractor}").json()
    assert detail["node"]["kind"] == "Attractor" and "ACTIVATES (in)" in detail["neighbors"]

    qa = client.post("/api/chat/query", json={"question": "Чому margin нижчий для phones у US?", "use_llm": True}).json()
    assert qa["answer_mode"] == "llm" and qa["citations"]["grounded"]  # the narrator asked the graph service over HTTP
    assert qa["view"]["highlight"]["edges"] and qa["view"]["highlight"]["seeds"] and qa["answer"]
    parsed = qa["view"]["evidence"]["parsed"]
    assert parsed["targets"] == ["margin"] and parsed["direction"] == -1  # literals parsed inside Ukrainian
    assert set(qa["view"]["highlight"]["edges"]) <= {e["data"]["id"] for e in g["edges"]}  # highlight ids resolve in the UI graph
    # the map demo: the shipped housing table as H3 cells, with its own resolution and band
    assert client.post("/api/demo/geo").status_code == 202
    geo_batch = next(b for b in _wait_ready(client) if b["filename"] == "california_housing.csv")
    assert geo_batch["status"] == "READY" and geo_batch["profile"]["geo"]["resolution"] == 5 and geo_batch["profile"]["geo"]["cells"] == 884
    assert geo_batch["metrics"]["validated_insights"] > 0 and "median_income_median_band" in geo_batch["profile"]["selected_dimensions"]
    geo_ds = geo_batch["dataset_id"]
    assert [d["dataset_id"] for d in client.get("/api/geo").json()] == [geo_ds]
    assert len(client.get(f"/api/geo/{geo_ds}/cells").json()["features"]) == 884
    assert client.get("/api/geo/ds-nope/cells").status_code == client.get("/api/geo/ds-../cells").status_code == 404
    overlay = client.get(f"/api/geo/{geo_ds}/overlay").json()  # the regions above its insights, their trust and scale
    assert overlay["features"] and all(f["properties"]["resolution"] < 5 and f["properties"]["patterns"] > 0 for f in overlay["features"])
    assert {r for f in overlay["features"] for r in f["properties"]["rows"]} <= set(range(884))
    assert set(overlay["patterns"]) == {p["id"] for p in client.get(f"/api/geo/{geo_ds}/cells").json()["patterns"]}
    assert {t["credibility"] for t in overlay["patterns"].values()} <= {1, 2, 5, 6}
    view = client.get(f"/api/graph?dataset={geo_ds}").json()
    regions = [n["data"] for n in view["nodes"] if n["data"]["kind"] == "Region"]
    assert regions and all(r["resolution"] < 5 for r in regions) and view["datasets"][geo_ds]["integrity"] == "ok"
    geo_pid = next(iter(overlay["patterns"]))
    detail = client.get(f"/api/nodes/{geo_pid}").json()
    assert detail["dataset"]["dataset_id"] == geo_ds and "regions" in detail["scale_chain"]
    assert client.get(f"/api/nodes/{regions[0]['id']}").json()["picture"]["patterns"] > 0
    assert client.delete(f"/api/datasets/{geo_ds}").json()["patterns_removed"] == geo_batch["metrics"]["validated_insights"]
    assert client.get("/api/geo").json() == [] and client.get(f"/api/geo/{geo_ds}/cells").status_code == 404  # the file went with the dataset
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
    """The newest Chromium in Playwright's browser cache, whichever revision this Playwright pins."""
    cache = {"win32": Path(os.environ.get("LOCALAPPDATA", "")), "darwin": Path.home() / "Library" / "Caches"}.get(
        sys.platform, Path.home() / ".cache"
    )
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or cache / "ms-playwright")
    patterns = ("chromium-*/chrome-win64/chrome.exe", "chromium-*/chrome-linux*/chrome", "chromium-*/chrome-mac*/*.app/Contents/MacOS/*")
    found = [exe for pattern in patterns for exe in root.glob(pattern)]
    return str(max(found, key=lambda exe: int(exe.relative_to(root).parts[0].removeprefix("chromium-")))) if found else None


@pytest.mark.browser
def test_browser_renders_graph_and_highlights_path(tmp_path, demo_csv):
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = _chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    cfg = make_config(tmp_path / "ws")
    engine = make_engine(cfg)
    assert engine.ingest_file(demo_csv)["status"] == "READY"
    housing = engine.process(
        engine.submit(PROJECT_ROOT / "data" / "demo" / "california_housing.csv", bins="median_income_median:4", geo="auto@5")["batch_id"]
    )
    assert housing["status"] == "READY"  # the map demo, as POST /api/demo/geo queues it
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("typeof S !== 'undefined' && S.cy && S.cy.nodes('[kind=\"Pattern\"]').length > 0", timeout=20000)
        page.wait_for_selector(".ds .chip.ok", timeout=10000)  # dataset card shows "Ready"
        assert page.inner_text("#health").strip()  # health pills rendered
        chips = "[...document.querySelectorAll('.chip-q')].map((c) => c.textContent)"
        assert len(page.evaluate(chips)) == 8 and {"Чому margin нижчий для phones у US?", "Де median_house_value вищий?"} <= set(
            page.evaluate(chips)
        )  # both demos' questions
        page.select_option("#dataset-filter", housing["dataset_id"])  # the filter narrows the welcome card to one dataset
        page.wait_for_function(f"({chips}).length === 4", timeout=10000)
        assert page.evaluate(chips)[0] == "Чому median house value нижчий у зонах INLAND?"
        page.select_option("#dataset-filter", "")
        page.wait_for_function(f"({chips}).length === 8", timeout=10000)
        options = page.evaluate("[...document.querySelectorAll('#dataset-filter option')].map((o) => o.textContent)")
        assert "california_housing.csv · H3 r5" in options  # a geo dataset names its level, as the levels of one upload must
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
        page.click("#verify-btn")  # the insight recomputed from the stored upload
        page.wait_for_selector("#verify-out .chip.ok", timeout=30000)
        assert "Reproduced" in page.inner_text("#verify-out")
        assert "Credibility" in page.inner_text("#inspector") and page.is_visible("#inspector .chip")  # the trust axes of the insight
        assert page.evaluate("document.querySelectorAll('.ev .ev-trust').length") > 0  # two uploads: every card shows its trust
        # one legend for every view; by default Hierarchy, Theme links and Memberships are on, with live counts
        assert page.is_visible("#legend") and page.is_visible("#legend .toggle[data-key='activates']")
        assert page.evaluate("[...document.querySelectorAll('#legend .toggle.on')].map(b => b.dataset.key)") == ["lattice", "latent", "activates"]
        assert page.evaluate("S.cy.edges('[type=\"ACTIVATES\"]').filter('.hidden').length") == 0  # memberships drawn from the start
        assert int(page.inner_text("[data-n='lattice']")) == page.evaluate("S.cy.edges('[type=\"SPECIALIZES\"]').length") > 0
        page.hover("#legend .key[data-key='cov']")  # spotlight: the rest of the graph dims
        assert page.evaluate("S.cy.elements('.lg-dim').length") > 0
        page.mouse.move(5, 5)
        assert page.evaluate("S.cy.elements('.lg-dim').length") == 0
        # the sphere: the same toggles switch its traces in place, a click on a point opens the drawer
        page.click("#view-switch button[data-view='sphere']")
        sphere = "document.querySelector('#sphere')"  # plotly.js draws into the div itself
        page.wait_for_function(f"(() => {{ const gd = {sphere}; return !!(gd && gd.data && gd.on); }})()", timeout=60000)
        page.screenshot(path=str(tmp_path / "sphere.png"))  # for a look by eye: the drawing is the console's now
        visible = f"(name) => {sphere}.data.filter(t => t.name === name).map(t => t.visible !== false)"
        assert page.evaluate(f"({visible})('Memberships')") == [True] and page.evaluate(f"({visible})('Hierarchy')") == [True]
        assert page.evaluate("['schema', 'region'].every((k) => document.querySelector(`.toggle[data-key=\"${k}\"]`).disabled)")  # no vectors
        page.click("#legend .toggle[data-key='activates']")
        page.click("#legend .toggle[data-key='lattice']")
        assert page.evaluate(f"({visible})('Memberships')") == [False] and page.evaluate(f"({visible})('Hierarchy')") == [False]
        page.evaluate("document.querySelector('#drawer-close').click()")
        pid = page.evaluate("S.nodes[0].id")
        page.evaluate(f"{sphere}.emit('plotly_click', {{ points: [{{ customdata: '{pid}' }}] }})")
        page.wait_for_selector("#drawer:not([hidden]) .shift-row", timeout=10000)
        page.click("#view-switch button[data-view='graph']")
        hidden, rest = (page.evaluate(f"S.cy.edges('[type=\"ACTIVATES\"]').not('.hl-edge'){f}.length") for f in (".filter('.hidden')", ""))
        assert hidden == rest > 0  # off in the graph too
        # an isolated evidence path stays isolated through a layer switch (it is the current highlight)
        page.click(tab)
        page.click(".bot-pane[data-pane='0'] .ev >> nth=0")
        assert page.evaluate("S.cy.nodes('.hl-evidence').length") == 1
        page.click("#legend .toggle[data-key='contrast']")
        assert page.evaluate("S.cy.nodes('.hl-evidence').length") == 1
        page.click("#legend .toggle[data-key='region']")  # the map regions, off by default
        assert page.evaluate("S.cy.nodes('[kind=\"Region\"]').not('.hidden').length") == int(page.inner_text("[data-n='region']")) > 0
        page.click("#legend .toggle[data-key='region']")
        # the column borders can be dragged
        handle = page.locator(".rail .col-resize").bounding_box()
        before = page.evaluate("document.querySelector('.rail').getBoundingClientRect().width")
        page.mouse.move(handle["x"] + 4, handle["y"] + 300)
        page.mouse.down()
        page.mouse.move(handle["x"] + 84, handle["y"] + 300, steps=5)
        page.mouse.up()
        assert abs(page.evaluate("document.querySelector('.rail').getBoundingClientRect().width") - before - 80) < 4
        # the map: cells selected as a zone ask a question without text
        page.evaluate("document.querySelector('#drawer-close').click()")
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function("G.data && G.map && G.map.getSource('cells') && G.map.loaded()", timeout=30000)
        nearest = """() => { const c = G.map.getCenter();  // the cells nearest the view's centre, as screen points
            return G.data.features.map((f) => { const r = f.geometry.coordinates[0].slice(1);
              const x = r.reduce((a, q) => a + q[0], 0) / r.length, y = r.reduce((a, q) => a + q[1], 0) / r.length;
              return { h3: f.properties.h3, d: (x - c.lng) ** 2 + (y - c.lat) ** 2, ...G.map.project([x, y]) }; })
              .sort((a, b) => a.d - b.d).slice(0, 3); }"""
        box, cells = page.locator("#map").bounding_box(), page.evaluate(nearest)
        for cell in cells:  # a real Shift-click on the map, as a user selects
            page.keyboard.down("Shift")
            page.mouse.click(box["x"] + cell["x"], box["y"] + cell["y"])
            page.keyboard.up("Shift")
        assert page.evaluate("G.regionMarks.length") > 0 and page.evaluate("G.overlay.features.some((f) => f.properties.resolution === G.level)")
        assert page.inner_text("#map-zone-n") == "ZONE · 3 cells" and sorted(page.evaluate("GeoMap.zone()")) == sorted(c["h3"] for c in cells)
        answers = page.evaluate("document.querySelectorAll('.answer-box').length")
        page.click("#map-zone-ask")
        page.wait_for_function(f"document.querySelectorAll('.answer-box').length > {answers}", timeout=30000)
        assert page.evaluate("S.chat.at(-1).zone") == 3 and "AREA:" in page.evaluate("S.chat.at(-1).qa.prompt")
        badges = "[...[...document.querySelectorAll('.msg-bot')].at(-1).querySelectorAll('.ev .role')].map((b) => b.textContent)"
        assert {"in the asked area", "outside the asked area"} & set(page.evaluate(badges))  # each item of the zone's levels says which
        page.fill("#question", "Де найбільше households?")  # the chat box asks about the selected zone too
        page.press("#question", "Enter")
        page.wait_for_function(f"document.querySelectorAll('.answer-box').length > {answers + 1}", timeout=30000)
        assert page.evaluate("S.chat.at(-1).zone") == 3 and "AREA:" in page.evaluate("S.chat.at(-1).qa.prompt")
        page.evaluate("GeoMap.toggleZone('8a2a1072b59ffff')")  # a cell this map does not draw (another level, elsewhere)
        assert page.inner_text("#map-zone-n") == "ZONE · 4 cells · 3 on this map"
        page.click("#map-zone-clear")
        assert page.is_hidden("#map-zone")
        page.click("#view-switch button[data-view='graph']")
        page.click("#theme-btn")
        assert page.evaluate("document.documentElement.dataset.theme") in {"dark", "light"}
        page.screenshot(path=str(tmp_path / "ui.png"))
        # deleting the datasets from their cards empties the views, through the backend
        assert page.is_visible(".ds .ds-del")  # always visible, not only on hover
        for left in (1, 0):
            page.once("dialog", lambda d: d.accept())
            page.click(".ds .ds-del >> nth=0")
            page.wait_for_function(f"document.querySelectorAll('.ds').length === {left}", timeout=20000)
        page.wait_for_selector(".empty-rail", timeout=20000)
        assert page.evaluate("S.cy.nodes().length") == 0 and not engine.graph().of_kind("Pattern")
        page.click("#new-chat")  # a fresh chat with no data: the welcome card offers no questions
        page.wait_for_selector(".welcome", timeout=10000)
        assert page.evaluate("document.querySelectorAll('.chip-q').length") == 0
        browser.close()
        assert not errors, errors
