"""The console's map in a browser (docs/08_interface.md §8.7): the base level's places, the form's geo levels, popups and Esc,
an insight from the map to the chat and back."""

from __future__ import annotations

import json
import re

import httpx
import pytest
from conftest import make_engine, serve, stack, upload_levels
from dev import chromium

from insight_graph_service.core.settings import PROJECT_ROOT


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
        page.wait_for_function("[...document.querySelectorAll('#inspector h4')].some((h) => h.textContent === 'Де')", timeout=10000)
        region = page.evaluate("[...document.querySelectorAll('#inspector h4')].find((h) => h.textContent === 'Де').nextElementSibling.dataset.id")
        assert region.startswith("R:") and region in {r for r, _ in graph.facts(pid).regions.items()}
        page.evaluate(f"inspect('{region}')")
        page.wait_for_function(f"$('drawer-title').textContent.includes('{region.split(':')[2]}')", timeout=10000)
        held = engine.graph().nodes[region]["props"]["patterns"]
        assert page.inner_text("#inspector .lead").startswith(f"{held} інсайт")  # its members of both bases, in the region's reading
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
        page.fill("#glossary", '{"ocean_proximity": "близькість до океану"}')  # the declared language goes with the file
        page.set_input_files(
            "#metadata", files=[{"name": "housing.md", "mimeType": "text/markdown", "buffer": b"Columns:\nmedian_income\n  Title (uk): dohid\n"}]
        )
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
        assert page.inner_text(f".ds-group[data-upload='{geo['upload_id']}'] .ds-group-head") == "california_housing.csv · готово рівнів: 2 з 2"
        assert page.evaluate(
            f"[...document.querySelectorAll(\".ds-group[data-upload='{geo['upload_id']}'] .ds-name\")].map((n) => n.textContent)"
        ) == [
            "california_housing.csv · H3 r5",
            "california_housing.csv · таблиця",
        ]
        table = next(b for b in levels if b["level"] == "table")
        assert table["user_glossary"] == {"ocean_proximity": "близькість до океану"} and set(table["metadata_doc"]["columns"]) == {"median_income"}
        chips = f".ds[data-ds='{table['dataset_id']}'] .ds-src .chip"
        assert page.locator(chips).all_inner_texts()[-2:] == ["глосарій · 2 колонки", "документ · 1 колонка"]
        # a PATCH re-signs the record: the next refresh reads the graph and the map again, the map card in the new words
        patched = httpx.patch(f"{url}/api/datasets/{geo['dataset_id']}", json={"glossary": '{"median_house_value": "вартість житла"}'}, timeout=60)
        assert patched.status_code == 200
        page.evaluate("refreshBatches()")
        page.wait_for_function("Object.values(G.patterns).some((p) => p.reading.what.includes('вартість житла'))", timeout=30000)
        renamed = page.evaluate("Object.keys(G.patterns).find((pid) => G.patterns[pid].reading.what.includes('вартість житла'))")
        page.evaluate(f"GeoMap.pick('{renamed}')")
        page.wait_for_function("$('map-card').querySelector('.mc-shift').textContent.includes('вартість житла')", timeout=10000)
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
        assert page.inner_text(f"{group} .ds-group-head") == "points.csv · готово рівнів: 2 з 3 · без знахідок: 1"
        card = page.locator(f"{group} .ds", has_text="H3 r5")
        assert card.locator(".chip").inner_text() == "Без знахідок" and not card.locator(".chip.bad, .ds-error").count()
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


@pytest.mark.browser
def test_an_insight_goes_from_the_map_to_the_chat_and_back(cfg, tmp_path):
    """From the IN VIEW list to the insight's card: its theme is the one the list shows it under, its Walk on holds its
    scale chain's levels; asked from the card, the question carries the H3 cells it is drawn by, the answer has the
    AREA line and the insight, whose citation focuses it on the same map; New chat clears the focus and the list's mark;
    Show in graph on a base insight under a geo filter shows its node. The map card (picked before any question), the
    chat's cards, the drawer and the answer without the model read an insight in the same words."""
    import pandas as pd

    from graph_query_engine.graph import primary_anchor
    from graph_query_engine.vertical import scale_chain
    from insight_contracts.wording import name, sentence
    from insight_graph_service.server.views import node_details

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
        assert any(t.startswith("↑ рівень 5") for t in card.locator(".insight-card .nb").all_inner_texts())
        walk = ["CONTRASTS", "CONTRASTS (in)", "SPECIALIZES", "SPECIALIZES (in)", "SIBLING", "SIBLING (in)", "CO_LOCATED", "CO_LOCATED (in)"]
        assert page.evaluate("Object.keys(WALK)") == walk and page.evaluate("Object.values(WALK)") == [
            name(k) for k in walk
        ]  # keys frozen, words TERMS'
        assert all(i.startswith("P-") for i in card.locator(".insight-card .nb").evaluate_all("(ns) => ns.map((n) => n.dataset.id)"))  # insights only
        words = node_details(graph, chained, engine.settings.thresholds)["reading"]  # one reading, from the overlay: no question yet
        assert page.inner_text("#map-card .mc-scope") == words["where"] and page.inner_text("#map-card .mc-shift") == words["what"]
        assert words["what"] in card.inner_text()
        labels = "[...document.querySelectorAll('#map-card .mc-grid > span')]"  # every number named by term(key), n_eff read out
        assert page.evaluate(f"{labels}.length === 6 && {labels}.every((s) => s.querySelector('.term[title]'))")
        assert "непорожн" in page.inner_text("#map-card .mc-note")  # n_eff read out: the cells it counts
        assert "(згладжено)" in page.get_attribute(f"#map-window .mw-i[data-pid='{chained}'] b", "title")
        page.evaluate(f"inspect('{chained}')")
        page.wait_for_function(f"$('drawer-title').textContent === {words['where']!r}", timeout=10000)
        assert page.inner_text("#inspector .lead") == words["what"]
        theme = card.locator(".ic-theme").get_attribute("data-id")  # the card's theme opens in the list, or the group it is in
        group = page.evaluate(f"W.data.themes.find((t) => t.id === {theme!r} || t.members.includes({theme!r})).id")
        assert group != theme and len(page.evaluate("W.data.themes")) > 8  # the whole map's themes are grouped: the folded path runs
        page.evaluate(f"W.fold[{group!r}] = false; renderWindow()")
        card.locator(".ic-theme").click()
        assert page.evaluate(f"W.fold[{group!r}]") and page.locator(f".mw-theme.open .mw-th[data-id='{group}']").count() == 1
        # a neighbour of another upload, which this map does not draw: walked on, the focus goes and its drawer says so
        offmap = "[...document.querySelectorAll('.msg-bot:last-child .insight-card .nb')].map((e) => e.dataset.id).filter((id) => !GeoMap.has(id))"
        for row in [i["id"] for t in page.evaluate("W.data.themes") for i in t["insights"] if graph.facts(i["id"]).resolution == 6]:
            page.evaluate(f"GeoMap.pick('{row}')")
            page.wait_for_function(f"S.chat.at(-1).insight && S.chat.at(-1).insight.node.id === '{row}'", timeout=10000)
            if ids := page.evaluate(offmap):
                break
        assert graph.insight(ids[0]).dataset_id == other["dataset_id"] and page.evaluate("G.focusId") == row
        page.locator(".msg-bot").last.locator(f".insight-card .nb[data-id='{ids[0]}']").click()
        page.wait_for_function("!$('drawer').hidden && $('inspector').textContent.includes('Не на цій мапі')", timeout=10000)
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
        items = page.evaluate("S.chat.at(-1).qa.view.evidence.items")
        held = next(n for n, i in enumerate(items) if i["pattern_id"] == base)
        assert items[held]["reading"] == node_details(graph, base, engine.settings.thresholds)["reading"]  # a base insight, projected
        assert page.locator(".msg-bot").last.locator(f".ev[data-i='{held}'] .ev-what").text_content() == items[held]["reading"]["what"]
        fallback = httpx.post(f"{url}/api/chat/query", json={**asked, "use_llm": False}, timeout=120).json()["answer"]
        assert sentence(items[held]["reading"]) in fallback  # the evidence-only answer says it in the same words
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
