"""UI smoke test: dataset loads through the API and the graph page renders (docs/08_interface.md)."""

from __future__ import annotations

import json
import re
import time

import httpx
import pytest
from conftest import FakeLLM, make_engine, serve, upload_levels
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
    assert cells.headers["content-encoding"] == "gzip" and len(cells.json()["rings"]) == 884
    assert client.get("/api/geo/ds-nope/cells").status_code == client.get("/api/geo/ds-../cells").status_code == 404
    overlay = client.get(f"/api/geo/{geo_ds}/overlay").json()  # its insights' places and the regions above, their trust and scale
    assert overlay["features"] and all(f["properties"]["resolution"] <= 5 and f["properties"]["patterns"] > 0 for f in overlay["features"])
    own = [f for f in overlay["features"] if f["properties"]["resolution"] == 5]
    assert len(own) <= ZONES_MAX * len(overlay["patterns"]) < 884  # at most five places each, not every cell of every insight
    assert {r for f in overlay["features"] for r in f["properties"]["rows"]} <= set(range(884))
    assert set(overlay["patterns"]) == {p["id"] for p in client.get(f"/api/geo/{geo_ds}/cells").json()["patterns"]}
    assert {t["credibility"] for t in overlay["patterns"].values()} <= {1, 2, 5, 6}
    plain, view = (client.get(f"/api/graph?dataset={geo_ds}{q}").json() for q in ("", "&regions=1"))  # the regions with their switch alone
    regions = [n["data"] for n in view["nodes"] if n["data"]["kind"] == "Region"]
    assert regions and all(r["resolution"] <= 5 and r["patterns"] > 0 for r in regions) and view["datasets"][geo_ds]["integrity"] == "ok"
    assert plain["regions"] == view["regions"] == len(regions) and not [n for n in plain["nodes"] if n["data"]["kind"] == "Region"]
    assert not [e for e in plain["edges"] if e["data"]["type"] in ("IN_REGION", "PARENT_OF")] and len(plain["nodes"]) == len(view["nodes"]) - len(
        regions
    )
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
        counted = int(page.inner_text("[data-n='region']"))  # the map regions, off by default: counted, not loaded
        assert counted > 0 and page.evaluate("S.cy.nodes('[kind=\"Region\"]').length") == 0
        page.select_option("#layout", "force")
        placed = "S.cy.nodes('[kind=\"Pattern\"]').map((n) => [n.id(), Math.round(n.position('x')), Math.round(n.position('y'))])"
        before = page.evaluate(placed)
        page.click("#legend .toggle[data-key='region']")  # fetched with the switch, placed under what they hold: the force layout stays
        page.wait_for_function(f"S.cy.nodes('[kind=\"Region\"]').not('.hidden').length === {counted}", timeout=10000)
        assert page.evaluate(placed) == before and int(page.inner_text("[data-n='region']")) == counted
        spots = page.evaluate("S.cy.nodes('[kind=\"Region\"]').map((n) => [Math.round(n.position('x')), Math.round(n.position('y'))])")
        assert len(set(map(tuple, spots))) == len(spots) == counted  # in rows, none on another
        page.click("#legend .toggle[data-key='region']")
        page.select_option("#layout", "dual")
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
        assert not page.evaluate("G.data.patterns.some((p) => GeoMap.has(p.id))")  # no trust read: none of this level's insights is drawn
        assert page.evaluate("GeoMap.offMap(G.data.patterns[0].id)").startswith("Not drawn on this map: its trust could not be read")
        page.evaluate("loadOverlay(G.ds)")
        assert page.is_hidden("#map-overlay-error") and page.evaluate("G.overlay.features.length") > 0
        assert page.evaluate("G.data.patterns.some((p) => GeoMap.has(p.id))")
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
        # an insight the map does not draw, from a citation or a drawer row alike: the focus goes, the drawer says why
        off = page.evaluate("api('/api/graph').then((g) => g.nodes.map((n) => n.data).find((d) => d.kind === 'Pattern' && !GeoMap.has(d.id)).id)")
        page.evaluate(f"reveal('{off}')")  # what a click on its citation does
        page.wait_for_function("!G.focusId && document.querySelector('#inspector .drawer-note')", timeout=10000)
        assert page.inner_text("#inspector .drawer-note").startswith("Not on this map")
        before = page.evaluate("S.chat.length")
        page.evaluate(f"GeoMap.focusPattern('{neighbour}'); GeoMap.pick('{off}')")  # picked, as an IN VIEW row, but not drawn here
        page.wait_for_function(f"S.chat.length === {before + 1}")
        assert page.evaluate("G.focusId") is None  # no other insight stays painted beside its card
        page.evaluate(f"GeoMap.focusPattern('{neighbour}')")
        row = page.evaluate("""(() => { const r = [...document.querySelectorAll('#inspector .nb')].find((x) => x.dataset.id.startsWith('P-'));
            const g = r.closest('details'); if (g) g.open = true; return r.dataset.id; })()""")
        title = page.inner_text("#drawer-title")
        page.click(f"#inspector .nb[data-id='{row}'] >> nth=0")
        page.wait_for_function(
            f"!G.focusId && $('drawer-title').textContent !== {title!r} && document.querySelector('#inspector .drawer-note')", timeout=10000
        )
        page.evaluate("GeoMap.toggleZone(G.data.features[0].properties.h3)")  # a zone selected for other questions
        turns = page.evaluate("S.chat.length")
        page.evaluate(f"pickInsight('{off}', null)")  # the card of an insight this map does not draw: it has no place here
        page.wait_for_function(f"S.chat.length === {turns + 1}")
        page.locator(".msg-bot").last.locator(".ic-ask").click()
        page.wait_for_function(f"S.chat.length === {turns + 2} && !S.busy", timeout=30000)
        assert page.evaluate("S.chat.at(-1).zone") == 0  # asked without a zone, never with the one selected
        page.evaluate("GeoMap.clearZone()")
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
        forged = """(pid) => { const was = G.focusId, ev = G.evidence; GeoMap.focusPattern(pid, false);  // focused when a commit marks it
            const t = G.overlay.patterns[pid]; G.overlay.patterns[pid] = { ...t, integrity: "mismatch" }; mergePatterns();
            const drawn = [GeoMap.has(pid), G.byRow.some((r) => r.includes(pid)), G.focusId];
            G.overlay.patterns[pid] = t; mergePatterns(); G.evidence = ev; GeoMap.focusPattern(was); return [...drawn, GeoMap.has(pid)]; }"""
        own = page.evaluate("G.data.patterns.find((p) => GeoMap.has(p.id)).id")  # a record that fails its digest is never drawn
        assert page.evaluate(forged, own) == [False, False, None, True]  # and its focus goes
        opacity = "G.map.getPaintProperty('focus-line', 'line-opacity')"
        page.wait_for_timeout(1500)  # the camera settles, then the outline pulses
        first = page.evaluate(opacity)
        page.wait_for_timeout(400)
        assert page.evaluate(opacity) != first
        page.keyboard.press("Escape")
        assert page.evaluate("G.focusId") is None and page.evaluate("G.focus.length") == 0 and page.evaluate(painted) == 0
        assert page.is_visible("#drawer")  # Esc closes one layer a press: the focus, then the drawer
        page.keyboard.press("Escape")
        assert page.is_hidden("#drawer")
        page.locator(".msg-bot").last.locator("a.cite").first.click()
        page.evaluate("loadDataset(G.ds)")  # a dataset loaded again, or another one: no cell of the old focus stays painted
        assert page.evaluate("G.focusId") is None and page.evaluate(painted) == 0
        # two loads at once: the later call wins, and while it loads nothing of the old dataset is drawable on its cells
        real = page.evaluate("G.ds")

        def pack(*xs):  # the cell package of square cells 0.4° wide at these longitudes, as /cells serves it
            rings = [[x, 0, 40000, 0, 40000, 40000, 0, 40000] for x in xs]
            columns = {"h3": ["85754e67fffffff"] * len(xs), "points": [3] * len(xs)}
            return {"columns": columns, "rings": rings, "scale": 5, "bbox": [min(xs), 0, max(xs) + 0.4, 0.4], "patterns": []}

        stub = {
            "cells": pack(0),
            "projected": [],
            "overlay": {"type": "FeatureCollection", "features": [], "patterns": {}},
            "window": {"cells": 1, "points": 3, "first": "map", "insights": 0, "themes": []},
        }
        page.route("**/api/geo/ds-stub/**", lambda r: r.fulfill(json=stub[r.request.url.split("/")[6].split("?")[0]]))
        race = """async ([real, stub, own]) => { const orig = api, wait = (ms) => new Promise((r) => setTimeout(r, ms)), slow = {
              [`/${real}/cells`]: 800, [`/${stub}/cells`]: 300, [`/${stub}/projected`]: 400, [`/${real}/overlay`]: 600, [`/${real}/projected`]: 600 };
            api = async (path, o) => { const ms = Object.entries(slow).find(([k]) => path.endsWith(k)); if (ms) await wait(ms[1]); return orig(path, o); };
            try { const a = loadDataset(real), b = loadDataset(stub);
              await wait(100); GeoMap.focusPattern(own, false); const note = GeoMap.offMap(own);  // a focus made during the load
              loadOverlay(real); loadProjected(real);  // a refresh pass's re-reads of the dataset still on the map, during the load
              while (G.ds !== stub) await wait(5);
              const mid = Object.keys(G.patterns).length;
              await Promise.all([a, b]); await wait(700);  // the old dataset's re-reads land after the switch
              return [mid, G.ds, G.data.features.length, Object.keys(G.patterns).length, G.loading, G.overlay.features.length, G.focusId, note]; } finally { api = orig; } }"""
        assert page.evaluate(race, [real, "ds-stub", own]) == [0, "ds-stub", 1, 0, False, 0, None, "Not drawn yet: the map is still loading."]
        page.unroute("**/api/geo/ds-stub/**")
        across = {**stub, "cells": pack(179.0, -179.6)}  # its bbox the points' min and max: a band around the globe
        page.route("**/api/geo/ds-dateline/**", lambda r: r.fulfill(json=across[r.request.url.split("/")[6].split("?")[0]]))
        width = "loadDataset('ds-dateline').then(() => { const b = G.map.getBounds(); return b.getEast() - b.getWest(); })"
        assert page.evaluate(width) < 20  # cells either side of ±180°: the map opens on them, not on a band around the globe
        page.unroute("**/api/geo/ds-dateline/**")
        page.evaluate(f"loadDataset('{real}')")
        page.route("**/api/geo/ds-stub/**", lambda r: r.fulfill(json=stub[r.request.url.split("/")[6].split("?")[0]]))
        pick = """async ([real, stub]) => { const orig = api, wait = (ms) => new Promise((r) => setTimeout(r, ms));
            api = async (path, o) => { if (path.endsWith(`/${stub}/cells`)) await wait(300); return orig(path, o); };
            try { const a = loadDataset(stub); await wait(50);  // the map still shows real
              $("dataset-filter").value = real; await GeoMap.refresh(); await a;  // the filter picks it while the other loads
              return [G.ds, G.loading]; } finally { api = orig; $("dataset-filter").value = ""; } }"""
        assert page.evaluate(pick, [real, "ds-stub"]) == [real, False]  # the filter's choice wins over the load in flight
        page.unroute("**/api/geo/ds-stub/**")
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
        walk = """() => { const t = S.chat.at(-1), n = t.insight.neighbors, chain = t.insight.scale_chain || { up: [], down: [] };
            const steps = new Set([...chain.up, ...chain.down].map((s) => s.pattern_id)), free = (it) => !steps.has(it.id);
            const k = Object.keys(WALK).find((k) => (n[k] || []).some(free)), id = n[k].find(free).id, has = (d) => insightCard(d).includes(`data-id="${id}"`);
            return [has(t), has({ ...t, insight: { ...t.insight, neighbors: { ...n, [k]: n[k].map((it) => (it.id === id ? { ...it, integrity: "mismatch" } : it)) } } })]; }"""
        assert page.evaluate(walk) == [True, False]  # Walk on offers intact insights only
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
    if (!__net.slow.some((s) => path.includes(s)) || (__net.fast || []).some((f) => path.includes(f))) return real(path, opts);
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
        page.wait_for_function("typeof S !== 'undefined' && S.cy && S.nodes.length > 0", timeout=60000)
        page.evaluate(_SLOW_API)
        settled = "__net.pending === 0 && !G.refreshing"
        # the filter changed twice in a row: the first one's graph answers last
        page.evaluate(f"__net.slow = ['/api/graph?dataset={table}']")
        page.select_option("#dataset-filter", table)
        page.wait_for_function("__net.pending === 1")
        page.select_option("#dataset-filter", geo)
        page.wait_for_function(settled, timeout=10000)
        assert {n["id"] for n in page.evaluate("S.nodes")} == {pid for pid, ins in graph.insights.items() if ins.dataset_id == geo}
        # the regions switched on while the plain graph loads, their own request answering first: the graph keeps them
        page.evaluate(
            f"__net.slow = ['/api/graph?dataset={geo}']; __net.fast = ['regions=1']; loadGraph(); true"
        )  # not awaited: the switch comes during the load
        page.click("#legend .toggle[data-key='region']")
        page.wait_for_function(
            f"{settled} && S.regionsIn && S.regionCount > 0 && S.cy.nodes('[kind=\"Region\"]').not('.hidden').length === S.regionCount", timeout=10000
        )
        page.click("#legend .toggle[data-key='region']")
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
        page.wait_for_function("typeof S !== 'undefined' && S.cy && S.nodes.length > 0", timeout=60000)
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
        # zoomed onto the busiest cell, the list reads the view anew and leads with the map's own level
        listing = "[W.data.first, W.data.themes.flatMap((t) => t.insights.map((i) => i.id)).sort()]"
        wide = page.evaluate(listing)
        assert wide[0] == "coarse" and wide[1]
        page.evaluate(
            "G.map.jumpTo({ center: centre(G.data.features.reduce((a, f) => (f.properties.points > a.properties.points ? f : a))), zoom: 13 })"
        )
        page.wait_for_function("W.data && W.data.first === 'map'", timeout=10000)
        assert page.evaluate(listing)[1] != wide[1]
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


@pytest.mark.browser
def test_esc_closes_one_layer_and_an_empty_level_is_no_failure(cfg, points_csv):
    """A level with nothing to mine is a neutral card its group counts apart; the map holds one cell popup, closed by a
    focus from it (which shows the insight's card) and by another tab; Esc closes the popup, then the focus, then the
    drawer, one a press, and leaves the question field alone."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    fine, coarse, _ = upload_levels(engine, points_csv, geo="auto@8,5", table=True)
    assert coarse["status"] == "FAILED" and coarse["error"]["code"] == "invalid_schema"  # too few cells at 5
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        requested: list[str] = []
        page.on("request", lambda r: requested.append(r.url))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.nodes.length > 0 && G.want && !G.refreshing", timeout=60000)
        assert not [u for u in requested if "/cells" in u] and page.evaluate("G.stale && !G.map")  # no cell read, no map made, before its tab
        group = f".ds-group[data-upload='{fine['upload_id']}']"
        assert page.inner_text(f"{group} .ds-group-head") == "points.csv · 2 of 3 levels ready · 1 without findings"
        card = page.locator(f"{group} .ds", has_text="H3 r5")
        assert card.locator(".chip").inner_text() == "No findings" and not card.locator(".chip.bad, .ds-error").count()
        # without a READY level the upload failed, as the backend reads it; a table level that finds nothing failed too
        level = "(lv, st, code) => ({ batch_id: lv, upload_id: 'u', filename: 'f.csv', level: lv, status: st, locator: lv === 'table' ? 'auto' : null, error: code && { code, message: code } })"
        bad = f"(cards) => (uploadGroup(cards.map((c) => ({level})(...c)), false).match(/chip bad/g) || []).length"
        assert page.evaluate(bad, [["8", "FAILED", "no_viable_insights"], ["5", "FAILED", "no_candidates"]]) == 2
        assert page.evaluate(bad, [["7", "FAILED", "no_viable_insights"]]) == 1
        assert page.evaluate(bad, [["8", "READY", None], ["table", "FAILED", "no_candidates"]]) == 1
        turn = "{question: 'q', qa: {view: {evidence: {attractors: [], items: [{pattern_id: 'P-none', scope: [], statistics: {}, provenance: {dataset_id: GeoMap.current()}}]}}}}"
        assert page.evaluate("GeoMap.suggestions().length") and page.evaluate(  # an answer on the map's dataset: its questions, no cell read
            f"JSON.stringify(followUps({turn})) === JSON.stringify(GeoMap.suggestions().slice(0, 3))"
        )
        page.evaluate(
            "() => { window.__api = api; window.api = (path, opts) => (path.endsWith('/cells') ? (window.__down = true, Promise.reject(new Error('down'))) : __api(path, opts)); }"
        )
        page.click("#view-switch button[data-view='map']")  # the cells cannot be read: the map stays stale and the next open tries again
        page.wait_for_function("window.__down && !G.refreshing && !G.loading", timeout=30000)
        assert page.evaluate("G.stale && !G.data")
        page.evaluate("() => { window.api = __api; }")
        page.click("#view-switch button[data-view='graph']")
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function("G.data && G.map.loaded()", timeout=30000)
        assert [u for u in requested if u.endswith(f"/api/geo/{fine['dataset_id']}/cells")]  # read on the Map tab
        stored = json.loads(engine.ws.geo_cells_path(fine["dataset_id"]).read_text(encoding="utf-8"))["features"]
        rebuilt = page.evaluate("G.data.features")  # the package rebuilds every cell: its ring and the fields its popup shows
        assert [f["properties"] for f in rebuilt] == [{k: v for k, v in f["properties"].items() if not k.endswith("_band")} for f in stored]
        assert all(
            abs(a - b) <= 1e-5
            for f, g in zip(rebuilt, stored)
            for p, q in zip(f["geometry"]["coordinates"][0], g["geometry"]["coordinates"][0])
            for a, b in zip(p, q)
        )
        busiest = "G.data.features.reduce((a, f) => (G.byRow[f.properties.row].length > G.byRow[a.properties.row].length ? f : a))"
        popup = f"cellPopup({busiest}, centre({busiest}))"
        pid = page.evaluate(f"G.byRow[{busiest}.properties.row][0]")
        page.evaluate(f"{popup}; {popup}")
        assert page.locator(".sig-pop").count() == 1  # one popup at a time
        page.click(".sig-pop .pp-ins >> nth=0")  # a focus from the popup closes it: the card shows
        assert page.locator(".sig-pop").count() == 0 and page.is_visible("#map-card")
        page.evaluate(f"inspect('{pid}')")
        page.wait_for_selector("#drawer:not([hidden])")
        page.evaluate(popup)
        layers = "[document.querySelectorAll('.sig-pop').length, GeoMap.focused(), !$('drawer').hidden]"
        assert page.evaluate(layers) == [1, True, True]
        page.focus("#question")
        page.keyboard.press("Escape")  # a text field keeps its Esc
        assert page.evaluate(layers) == [1, True, True]
        page.focus("#lyr-cells")  # a checkbox does not
        for left in ([0, True, True], [0, False, True], [0, False, False]):
            page.keyboard.press("Escape")
            assert page.evaluate(layers) == left
        page.evaluate(popup)
        page.click("#view-switch button[data-view='graph']")  # another tab closes it
        assert page.locator(".sig-pop").count() == 0
        browser.close()
        assert not errors, errors


def test_the_console_knows_the_empty_level_codes():
    """The rail's neutral card reads ``uploads.EMPTY_LEVEL``, copied into ``datasets.js``: the two stay one set."""
    from insight_graph_service.core.uploads import EMPTY_LEVEL

    js = (PROJECT_ROOT / "sig_web_console" / "static" / "datasets.js").read_text(encoding="utf-8")
    copied = re.search(r"const EMPTY_LEVEL = new Set\(\[([^\]]*)\]\)", js).group(1)
    assert set(re.findall(r'"(\w+)"', copied)) == EMPTY_LEVEL


@pytest.mark.browser
def test_an_insight_goes_from_the_map_to_the_chat_and_back(cfg, tmp_path):
    """From the IN VIEW list to the insight's card: its theme is the one the list shows it under, its Walk on holds its
    scale chain's levels; asked from the card, the question carries the H3 cells it is drawn by, the answer has the
    AREA line and the insight, whose citation focuses it on the same map; New chat clears the focus and the list's mark;
    Show in graph on a base insight under a geo filter shows its node."""
    import pandas as pd

    from graph_query_engine.graph import primary_anchor
    from graph_query_engine.vertical import scale_chain

    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    with (PROJECT_ROOT / "data" / "demo" / "california_housing.csv").open("rb") as f:
        levels = engine.upload("california_housing.csv", f, geo="auto@6,5", table=True, bins="median_income_median:4,median_income:4")
    fine, _, table = (engine.process(r["batch_id"]) for r in levels)
    pd.read_csv(PROJECT_ROOT / "data" / "demo" / "california_housing.csv").iloc[::2].to_csv(half := tmp_path / "half.csv", index=False)
    other = engine.process(engine.submit(half, geo="auto@6", bins="median_income_median:4")["batch_id"])  # another upload: off this map
    graph = engine.graph()
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.nodes.length > 0", timeout=60000)
        page.select_option("#dataset-filter", fine["dataset_id"])
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function(f"G.ds === '{fine['dataset_id']}' && G.map.loaded() && W.data && W.data.themes.length", timeout=60000)
        page.evaluate("if (W.folded) $('mw-fold').click()")
        # a geo insight with a coarser step: its card names its theme and walks its scale chain
        chained = next(
            i["id"]
            for t in page.evaluate("W.data.themes")
            for i in t["insights"]
            if graph.facts(i["id"]).resolution == 6 and (chain := scale_chain(graph, [i["id"]])) and chain[0].up
        )
        page.evaluate(f"GeoMap.pick('{chained}')")
        page.wait_for_selector(".insight-card .nb", timeout=10000)
        card = page.locator(".msg-bot").last
        assert card.locator(".ic-theme").get_attribute("data-id") == primary_anchor(graph, chained)["target"]
        assert any(t.startswith("↑ res 5") for t in card.locator(".insight-card .nb").all_inner_texts())
        theme = card.locator(".ic-theme").get_attribute("data-id")  # the card's theme opens in the list
        page.evaluate(f"W.fold[{theme!r}] = false; renderWindow()")
        card.locator(".ic-theme").click()
        assert page.evaluate(f"W.fold[{theme!r}]") and page.locator(f".mw-theme.open .mw-th[data-id='{theme}']").count() == 1
        # a neighbour of another upload, which this map does not draw: walked on, the focus goes and its drawer says so
        offmap = "[...document.querySelectorAll('.msg-bot:last-child .insight-card .nb')].map((e) => e.dataset.id).filter((id) => !GeoMap.has(id))"
        for row in [i["id"] for t in page.evaluate("W.data.themes") for i in t["insights"] if graph.facts(i["id"]).resolution == 6]:
            page.evaluate(f"GeoMap.pick('{row}')")
            page.wait_for_function(f"S.chat.at(-1).insight && S.chat.at(-1).insight.node.id === '{row}'", timeout=10000)
            if ids := page.evaluate(offmap):
                break
        assert graph.insight(ids[0]).dataset_id == other["dataset_id"] and page.evaluate("G.focusId") == row
        page.locator(".msg-bot").last.locator(f".insight-card .nb[data-id='{ids[0]}']").click()
        page.wait_for_function("!$('drawer').hidden && $('inspector').textContent.includes('Not on this map')", timeout=10000)
        assert page.evaluate("G.focusId") is None
        # a base insight asked with its place: H3 cells as the zone, AREA in the prompt, the insight in the answer
        base = next(i["id"] for t in page.evaluate("W.data.themes") for i in t["insights"] if i["level"] == "table")
        page.evaluate(
            "() => { window.__asked = []; const real = api; window.api = (path, opts) => { if (path === '/api/chat/query') __asked.push(JSON.parse(opts.body)); return real(path, opts); }; }"
        )
        page.evaluate(f"GeoMap.pick('{base}')")
        page.wait_for_function("S.chat.at(-1).insight && S.chat.at(-1).insight.node.id === " + repr(base), timeout=10000)
        page.locator(".msg-bot").last.locator(".ic-ask").click()
        page.wait_for_function("S.chat.at(-1).qa", timeout=30000)
        (asked,) = page.evaluate("__asked")
        assert asked["cells"] and all(isinstance(c, str) and re.fullmatch(r"[0-9a-f]{15}", c) for c in asked["cells"])
        assert asked["cells"] == page.evaluate(f"GeoMap.placeOf('{base}')")
        assert "AREA:" in page.evaluate("S.chat.at(-1).qa.prompt")
        assert base in page.evaluate("S.chat.at(-1).qa.view.evidence.items.map((i) => i.pattern_id)")
        answer = page.locator(".msg-bot").last
        cite = answer.locator(f"a.cite[data-pid='{base}']")
        if cite.count():  # its citation in the text, else its evidence card
            cite.first.click()
        else:
            answer.locator(".bot-tabs button[data-pane='0']").click()
            answer.locator(
                f".ev[data-i='{page.evaluate(f'S.chat.at(-1).qa.view.evidence.items.findIndex((i) => i.pattern_id === {base!r})')}']"
            ).click()
        page.wait_for_function(f"G.focusId === '{base}'", timeout=10000)
        assert page.evaluate("G.ds") == fine["dataset_id"]  # the map stays on its dataset
        page.click("#new-chat")
        assert page.evaluate("G.focusId") is None and page.locator("#map-window .mw-i.on").count() == 0
        # Show in graph on a base insight under the geo filter: the filter gives way, the node is in the graph
        page.evaluate(f"GeoMap.pick('{base}')")
        page.wait_for_selector(".msg-bot >> nth=-1 >> .ic-graph", timeout=10000)
        page.locator(".msg-bot").last.locator(".ic-graph").click()
        page.wait_for_function(f"$('dataset-filter').value === '' && S.cy.getElementById('{base}').nonempty()", timeout=20000)
        assert table["dataset_id"] == graph.insight(base).dataset_id
        browser.close()
        assert not errors, errors


# /api/batches with an upload's two levels first, r7 READY and r5 running: its stage and stage_times are what a test sets on window.__run
_RUNNING = """() => { const real = api, ago = (s) => new Date(Date.now() - s * 1000).toISOString();
  const level = (lv, status) => ({ batch_id: `B-r${lv}`, dataset_id: `D-r${lv}`, upload_id: 'U', level: lv, filename: 'next.csv', status, created_at: ago(80), stage_times: { UPLOADED: ago(80) } });
  window.__run = level('5', 'EMBEDDING'); __run.stage_times.EMBEDDING = ago(72);
  const done = level('7', 'READY');
  window.api = async (path, opts) => (path === '/api/batches' ? [__run, done, ...(await real(path, opts))] : real(path, opts)); }"""


@pytest.mark.browser
def test_the_poll_keeps_the_rail_and_the_filter_while_a_batch_runs(cfg, demo_csv):
    """A poll during a batch replaces only the card whose markup changed: a READY card, alone or a level beside the running
    one, and the filter's options keep their nodes while the running card shows its stage and how long it has run; a
    new READY dataset adds its option."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    ds = engine.ingest_file(demo_csv)["dataset_id"]
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.nodes.length > 0 && !G.refreshing", timeout=60000)
        page.evaluate(_RUNNING)
        poll = "async () => { await refreshBatches(); clearTimeout(S.poll); }"
        step = "#batches .ds:not(.clickable) .ds-step"
        page.evaluate(poll)
        assert re.fullmatch(r"Embedding insights… step 6 of 10 · 1 min 1\d s", page.inner_text(step))
        kept = "[...document.querySelectorAll('#batches .ds.clickable')]"
        assert page.evaluate(f"{kept}.map((c) => c.dataset.ds).sort().join()") == ",".join(sorted(["D-r7", ds]))
        page.evaluate(f"() => {{ window.__cards = {kept}; window.__opt = $('dataset-filter').options[1]; }}")
        page.evaluate("() => { __run.status = 'UPDATING_ONTOLOGY'; __run.stage_times.UPDATING_ONTOLOGY = new Date().toISOString(); }")
        page.evaluate(poll)
        assert page.evaluate(f"{kept}.every((c, i) => c === __cards[i]) && $('dataset-filter').options[1] === __opt")
        shown = page.inner_text(step)
        assert re.fullmatch(r"Learning themes… step 7 of 10 · [0-2] s", shown)
        page.wait_for_function(f"document.querySelector('{step}').innerText !== {json.dumps(shown)}", timeout=5000)  # ticks between polls
        page.evaluate("() => { __run.status = 'READY'; }")
        page.evaluate(poll)
        assert page.evaluate("[...$('dataset-filter').options].some((o) => o.value === 'D-r5')")
        browser.close()
        assert not errors, errors
