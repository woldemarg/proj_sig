"""UI smoke test: dataset loads through the API and the graph page renders (docs/08_interface.md)."""

from __future__ import annotations

import re
import time

import httpx
import pytest
from conftest import FakeLLM, make_engine, serve
from dev import chromium, compose

from evidence_narrator_service.app import create_app as create_narrator
from evidence_narrator_service.settings import NarratorSettings
from graph_query_engine.vertical import ZONES_MAX
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
    from scratch import free_port

    port = free_port()
    narrator = create_narrator(NarratorSettings(insight_graph_url=f"http://127.0.0.1:{port}"), llm=FakeLLM())
    return port, compose(create_app(engine), narrator)


@pytest.fixture
def client(cfg):
    port, app = stack(make_engine(cfg))
    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as c:
        yield c


def test_upload_process_render_query(client, demo_csv):
    page = client.get("/").text  # the console's static files, as nginx serves them
    assert "/vendor/cytoscape.min.js" in page and "/vendor/plotly-gl3d.min.js" in page and 'id="cy"' in page
    assets = re.findall(r'(?:src|href)="(/[^"]+)"', page)  # every script and stylesheet the page loads
    assert "/chat.js" in assets and all(client.get(path).status_code == 200 for path in assets)
    with open(demo_csv, "rb") as f:
        r = client.post("/api/upload", files={"file": ("retail.csv", f, "text/csv")}, data={"bins": ""})
    assert r.status_code == 202 and r.json()["status"] == "UPLOADED"
    batches = _wait_ready(client)
    assert batches[0]["status"] == "READY"
    assert batches[0]["metrics"]["validated_insights"] > 0 and batches[0]["profile"]["rows"] == 5000

    graph = client.get("/api/graph", headers={"Accept-Encoding": "gzip"})  # compressed, as nginx compresses it
    assert graph.headers["content-encoding"] == "gzip"
    g = graph.json()
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
    cells = client.get(f"/api/geo/{geo_ds}/cells", headers={"Accept-Encoding": "gzip"})
    assert cells.headers["content-encoding"] == "gzip" and len(cells.json()["features"]) == 884
    assert client.get("/api/geo/ds-nope/cells").status_code == client.get("/api/geo/ds-../cells").status_code == 404
    overlay = client.get(f"/api/geo/{geo_ds}/overlay").json()  # its insights' places and the regions above, their trust and scale
    assert overlay["features"] and all(f["properties"]["resolution"] <= 5 and f["properties"]["patterns"] > 0 for f in overlay["features"])
    own = [f for f in overlay["features"] if f["properties"]["resolution"] == 5]
    assert len(own) <= ZONES_MAX * len(overlay["patterns"]) < 884  # at most five places each, not every cell of every insight
    assert {r for f in overlay["features"] for r in f["properties"]["rows"]} <= set(range(884))
    assert set(overlay["patterns"]) == {p["id"] for p in client.get(f"/api/geo/{geo_ds}/cells").json()["patterns"]}
    assert {t["credibility"] for t in overlay["patterns"].values()} <= {1, 2, 5, 6}
    view = client.get(f"/api/graph?dataset={geo_ds}").json()
    regions = [n["data"] for n in view["nodes"] if n["data"]["kind"] == "Region"]
    assert regions and all(r["resolution"] <= 5 and r["patterns"] > 0 for r in regions) and view["datasets"][geo_ds]["integrity"] == "ok"
    geo_pid = next(iter(overlay["patterns"]))
    detail = client.get(f"/api/nodes/{geo_pid}").json()
    assert detail["dataset"]["dataset_id"] == geo_ds and "regions" in detail["scale_chain"] and "labels" in detail
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


@pytest.mark.browser
def test_browser_renders_graph_and_highlights_path(cfg, tmp_path, demo_csv):
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
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
        assert page.evaluate("G.map.getFilter('regions-n')[2] === G.level && G.overlay.features.some((f) => f.properties.resolution === G.level)")
        assert page.evaluate("document.querySelectorAll('.maplibregl-marker').length") < 50  # region counts are a symbol layer, places capped
        ring = "[[179.87, 1.86], [179.18, 0.66], [179.84, -0.72], [-178.8, -0.94], [-178.08, 0.25], [-178.75, 1.67], [179.87, 1.86]]"
        lon, lat = page.evaluate(f"centre({{geometry: {{coordinates: [{ring}]}}}})")  # a cell across ±180°: where it is
        assert abs(lon + 179.47) < 0.5 and abs(lat - 0.46) < 0.5
        page.evaluate("loadOverlay('ds-unknown')")  # an overlay that cannot be read says so, not an empty map of regions
        assert page.is_visible("#map-overlay-error") and page.inner_text("#map-overlay-error").startswith("Regions and trust unavailable")
        page.evaluate("loadOverlay(G.ds)")
        assert page.is_hidden("#map-overlay-error") and page.evaluate("G.overlay.features.length") > 0
        page.evaluate("loadProjected('ds-unknown')")  # the other levels that cannot be read say so, not an upload without them
        assert page.is_visible("#map-projected-error") and page.inner_text("#map-projected-error").startswith("Insights of other levels unavailable")
        page.evaluate("loadProjected(G.ds)")
        assert page.is_hidden("#map-projected-error")
        assert page.inner_text("#map-zone-n") == "ZONE · 3 cells" and sorted(page.evaluate("GeoMap.zone()")) == sorted(c["h3"] for c in cells)
        answers = page.evaluate("document.querySelectorAll('.answer-box').length")
        page.click("#map-zone-ask")
        page.wait_for_function(f"document.querySelectorAll('.answer-box').length > {answers}", timeout=30000)
        assert page.evaluate("S.chat.at(-1).zone") == 3 and "AREA:" in page.evaluate("S.chat.at(-1).qa.prompt")
        badges = "[...[...document.querySelectorAll('.msg-bot')].at(-1).querySelectorAll('.ev .role')].map((b) => b.textContent)"
        assert {"in the asked area", "outside the asked area"} & set(page.evaluate(badges))  # each item of the zone's levels says which
        assert page.is_visible(".msg-bot[data-turn='0'] .bot-pane[data-pane='0'] .ev")  # the first answer's open panel survives the next
        page.fill("#question", "Де найбільше households?")  # the chat box asks about the selected zone too
        page.press("#question", "Enter")
        page.wait_for_function(f"document.querySelectorAll('.answer-box').length > {answers + 1}", timeout=30000)
        assert page.evaluate("S.chat.at(-1).zone") == 3 and "AREA:" in page.evaluate("S.chat.at(-1).qa.prompt")
        page.evaluate("GeoMap.toggleZone('8a2a1072b59ffff')")  # a cell this map does not draw (another level, elsewhere)
        assert page.inner_text("#map-zone-n") == "ZONE · 4 cells · 3 on this map"
        page.click("#map-zone-clear")
        assert page.is_hidden("#map-zone")
        drawn = """async () => { for (const p of G.data.patterns) { await inspect(p.id);  // a drawer row of an insight this map draws
            const row = [...document.querySelectorAll('#inspector .nb')].find((r) => r.dataset.id !== p.id && GeoMap.has(r.dataset.id));
            if (row) { row.closest('details').open = true; return row.dataset.id; } } return null; }"""  # its group opened, as a click on it does
        neighbour, title = page.evaluate(drawn), page.inner_text("#drawer-title")
        page.click(f"#inspector .nb[data-id='{neighbour}'] >> nth=0")  # with the map open: the map focuses it, the drawer follows
        page.wait_for_function(f"G.focusId === '{neighbour}' && $('drawer-title').textContent !== {title!r}", timeout=10000)
        retail = page.evaluate(
            f"[...document.querySelectorAll('#dataset-filter option')].map((o) => o.value).find((v) => v && v !== '{housing['dataset_id']}')"
        )
        page.select_option("#dataset-filter", retail)  # a filter the next answer's evidence does not live in
        page.wait_for_selector(f".ds.active[data-ds='{retail}']")
        answers = page.evaluate("document.querySelectorAll('.answer-box').length")
        page.evaluate("ask('Де median_house_value вищий?')")  # an answer paints nothing on the map until a citation is clicked
        page.wait_for_function(f"document.querySelectorAll('.answer-box').length > {answers}", timeout=30000)
        now = page.input_value("#dataset-filter")  # the filter moved to where the evidence lives, and the rail's active card with it
        assert now != retail and page.evaluate("[...document.querySelectorAll('.ds.active')].map((e) => e.dataset.ds)") == ([now] if now else [])
        painted = "G.map.getSource('focus').serialize().data.features.length"  # the focused insight's cells, its own source
        assert page.evaluate("G.evidence.length") > 0 and page.evaluate(painted) == 0 and page.is_hidden("#map-card")
        page.locator(".msg-bot").last.locator("a.cite").first.click()
        assert page.evaluate("G.focusId") and page.evaluate(painted) == page.evaluate("G.focus.length") > 0 and page.is_visible("#map-card")
        forged = """(pid) => { const t = G.overlay.patterns[pid]; G.overlay.patterns[pid] = { ...t, integrity: "mismatch" };
            renderCard(G.patterns[pid], null, null); const html = $("map-card-body").innerHTML;
            if (t) G.overlay.patterns[pid] = t; else delete G.overlay.patterns[pid];
            renderCard(G.patterns[pid], null, null); return html; }"""
        assert "integrity mismatch" not in page.inner_text("#map-card-body")  # a record that fails its digest is labelled on the card
        assert "integrity mismatch" in page.evaluate(forged, page.evaluate("G.focusId"))
        opacity = "G.map.getPaintProperty('focus-line', 'line-opacity')"
        page.wait_for_timeout(1500)  # the camera settles, then the outline pulses
        first = page.evaluate(opacity)
        page.wait_for_timeout(400)
        assert page.evaluate(opacity) != first
        page.keyboard.press("Escape")
        assert page.evaluate("G.focusId") is None and page.evaluate("G.focus.length") == 0 and page.evaluate(painted) == 0
        page.locator(".msg-bot").last.locator("a.cite").first.click()
        page.evaluate("loadDataset(G.ds)")  # a dataset loaded again, or another one: no cell of the old focus stays painted
        assert page.evaluate("G.focusId") is None and page.evaluate(painted) == 0
        # the map as a window on the graph: the view's insights under their themes; a click draws one and describes it in the chat
        page.wait_for_function("W.data && !W.data.error && W.data.themes.length", timeout=30000)
        if page.evaluate("W.folded"):  # a narrow map starts with the list folded
            page.click("#mw-fold")
        turns, row = page.evaluate("S.chat.length"), page.locator("#map-window .mw-i").first
        pid = row.get_attribute("data-pid")
        assert all(re.fullmatch(r"table|res \d+", i["level"]) for t in page.evaluate("W.data.themes") for i in t["insights"])  # never "res None"
        row.click()
        page.wait_for_function(f"G.focusId === '{pid}' && S.chat.length === {turns + 1}", timeout=30000)
        assert page.locator(".msg-bot").last.locator(".insight-card").count() == 1 and page.is_visible(f"#map-window .mw-i.on[data-pid='{pid}']")
        forged = """() => { const t = S.chat.at(-1), node = t.insight.node;  // the card's numbers are the journal record's: say when it fails its digest
            return insightCard({ ...t, insight: { ...t.insight, node: { ...node, props: { ...node.props, integrity: "mismatch" } } } }); }"""
        assert "integrity mismatch" not in page.locator(".msg-bot").last.inner_text() and "integrity mismatch" in page.evaluate(forged)
        page.keyboard.press("Escape")
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


@pytest.mark.browser
def test_the_drawer_lists_where_a_base_insight_lies(cfg):
    """A table level placed by its map: its insight's drawer has a Where section of its places, each a region whose
    drawer counts it among the insights it holds."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    with (PROJECT_ROOT / "data" / "demo" / "california_housing.csv").open("rb") as f:
        levels = engine.upload("california_housing.csv", f, geo="auto@5", table=True, bins="median_income_median:4,median_income:4")
    done = [engine.process(r["batch_id"]) for r in levels]
    table = next(r["dataset_id"] for r in done if r["level"] == "table")
    graph = engine.graph()
    pid = next(p for p, ins in graph.insights.items() if ins.dataset_id == table and graph.facts(p).regions)
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.cy && S.cy.nodes('[kind=\"Pattern\"]').length > 0", timeout=60000)
        page.evaluate(f"inspect('{pid}')")
        page.wait_for_function("[...document.querySelectorAll('#inspector h4')].some((h) => h.textContent === 'Where')", timeout=10000)
        region = page.evaluate("[...document.querySelectorAll('#inspector h4')].find((h) => h.textContent === 'Where').nextElementSibling.dataset.id")
        assert region.startswith("R:") and region in {r for r, _ in graph.facts(pid).regions.items()}
        page.evaluate(f"inspect('{region}')")
        page.wait_for_function(f"$('drawer-title').textContent.includes('{region.split(':')[2]}')", timeout=10000)
        held = engine.graph().nodes[region]["props"]["patterns"]
        assert f"{held} insight" in page.inner_text("#inspector .lead")  # its members of both bases
        browser.close()
        assert not errors, errors


# every console fetch goes through api(): this wraps it to hold the named paths back and to count what is asked
_SLOW_API = """() => { const real = api;
  window.__net = { slow: [], pending: 0, hits: [] };
  window.api = (path, opts) => {
    __net.hits.push(path);
    if (!__net.slow.some((s) => path.includes(s))) return real(path, opts);
    __net.pending++;
    return new Promise((r) => setTimeout(r, 1200)).then(() => real(path, opts)).finally(() => __net.pending--);
  }; }"""


@pytest.mark.browser
def test_a_late_answer_never_overwrites_a_newer_view(cfg, demo_csv):
    """A slow graph of the previous filter, slow points of the previously clicked insight and a second map refresh
    asked during the first: the view shows what was asked last, and the second refresh is a pass of its own."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    table = engine.ingest_file(demo_csv)["dataset_id"]
    geo = engine.process(engine.submit(PROJECT_ROOT / "data" / "demo" / "california_housing.csv", geo="auto@5")["batch_id"])["dataset_id"]
    graph = engine.graph()
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.cy && S.nodes.length > 0 && G.data", timeout=60000)
        page.evaluate(_SLOW_API)
        settled = "__net.pending === 0 && !G.refreshing"
        # the filter changed twice in a row: the first one's graph answers last
        page.evaluate(f"__net.slow = ['/api/graph?dataset={table}']")
        page.select_option("#dataset-filter", table)
        page.wait_for_function("__net.pending === 1")
        page.select_option("#dataset-filter", geo)
        page.wait_for_function(settled, timeout=10000)
        assert {n["id"] for n in page.evaluate("S.nodes")} == {pid for pid, ins in graph.insights.items() if ins.dataset_id == geo}
        # two insights clicked in a row on the map: the first one's points answer last
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function("G.ds && G.map.loaded()", timeout=30000)
        points = {
            pid: len(httpx.get(f"{url}/api/geo/{geo}/patterns/{pid}").json()["points"]) for pid in page.evaluate("G.data.patterns.map((p) => p.id)")
        }
        first, second = sorted(points, key=points.get)[-1], sorted(points, key=points.get)[0]
        assert points[first] != points[second]
        page.evaluate(f"__net.slow = ['/patterns/{first}']; GeoMap.focusPattern('{first}'); GeoMap.focusPattern('{second}')")
        page.wait_for_function(f"{settled} && !$('map-card-body').textContent.includes('…')", timeout=10000)
        assert page.evaluate("G.map.getSource('pts').serialize().data.features.length") == points[second]
        assert page.evaluate("G.focusId") == second
        # a refresh asked while one runs: the cells are read once, the overlay again by a pass that starts after it
        hits = "__net.hits.filter((h) => h.endsWith(`/${G.ds}/%s`)).length"
        page.evaluate("__net.slow = []; __net.hits = []; G.data = null")
        page.evaluate("Promise.all([GeoMap.refresh(), GeoMap.refresh()])")
        assert page.evaluate(hits % "cells") == 1 and page.evaluate(hits % "overlay") == 2
        browser.close()
        assert not errors, errors


@pytest.mark.browser
def test_the_form_adds_the_base_level_to_a_geo_file_while_the_map_is_open(cfg, demo_csv):
    """The form's base-level box goes with a geo option alone; the same file submitted again with it skips the geo level
    and commits the table level, whose insights reach the open map's list and overlay without a reload; the levels' cards
    stand together; a deletion is offered only once no batch runs."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine, housing, bins = make_engine(cfg), PROJECT_ROOT / "data" / "demo" / "california_housing.csv", "median_income_median:4"
    with housing.open("rb") as f:
        (level,) = engine.upload(housing.name, f, geo="auto@auto", bins=bins)
    geo = engine.process(level["batch_id"])
    queued = engine.submit(demo_csv)  # not run until the end: a batch runs all along
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.cy && S.nodes.length > 0 && G.data", timeout=60000)
        deletes = "[...document.querySelectorAll('.ds-del')].map((b) => b.disabled)"
        assert page.evaluate(deletes) == [True]  # the geo level's; the queued batch's card has none yet
        page.click(".adv summary")
        assert page.is_disabled("#table")
        page.fill("#geo", "auto@auto")
        assert page.is_enabled("#table") and page.is_checked("#table")
        page.fill("#bins", f"{bins},median_income:4")  # the table's own band beside the cells'
        page.evaluate("$('geo').value = ''; window.dispatchEvent(new PageTransitionEvent('pageshow'))")  # a form the browser restored
        assert page.is_disabled("#table")
        page.evaluate("$('geo').value = 'auto@auto'; window.dispatchEvent(new PageTransitionEvent('pageshow'))")
        assert page.is_enabled("#table")
        grouped = "uploadGroups([{batch_id: 'g', upload_id: 'u', level: '5'}, {batch_id: 't', upload_id: 'u', level: 'table', locator: 'auto'}, {batch_id: 'p', upload_id: 'u', level: 'table'}]).map((g) => g.map((b) => b.batch_id))"
        assert page.evaluate(grouped) == [["g", "t"], ["p"]]  # a plain upload of the same bytes is no level of it
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function("G.ds && G.map.loaded() && W.data", timeout=30000)
        held = "G.overlay.features.reduce((a, f) => a + f.properties.patterns, 0)"
        before = page.evaluate(held)
        assert page.evaluate("G.projected.length") == 0
        page.evaluate("window.__samePage = true")
        page.set_input_files("#file", str(housing))
        page.click("#upload-btn")
        page.wait_for_function("G.projected.some((p) => p.level === 'table')", timeout=60000)
        page.wait_for_function("W.data.themes.some((t) => t.insights.some((i) => G.patterns[i.id].level === 'table'))", timeout=10000)
        assert page.evaluate(held) > before and page.evaluate("window.__samePage")
        levels = httpx.get(f"{url}/api/batches", params={"upload_id": geo["upload_id"]}).json()
        assert sorted((b["level"], b["status"]) for b in levels) == [("5", "READY"), ("5", "SKIPPED"), ("table", "READY")]
        assert page.inner_text(f".ds-group[data-upload='{geo['upload_id']}'] .ds-group-head") == "california_housing.csv · 2 of 2 levels ready"
        assert page.evaluate(
            f"[...document.querySelectorAll(\".ds-group[data-upload='{geo['upload_id']}'] .ds-name\")].map((n) => n.textContent)"
        ) == [
            "california_housing.csv · H3 r5",
            "california_housing.csv · table",
        ]
        engine.process(queued["batch_id"])
        page.wait_for_function(f"{deletes}.length === 3 && {deletes}.every((d) => !d)", timeout=10000)
        browser.close()
        assert not errors, errors
