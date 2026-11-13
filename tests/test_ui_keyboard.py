"""The console from the keyboard, on narrow screens and with reduced motion (docs/08_interface.md §8.2)."""

from __future__ import annotations

import json

import pytest
from conftest import make_engine, serve, stack, upload_levels
from dev import chromium


@pytest.mark.browser
def test_the_console_works_from_the_keyboard_and_on_narrow_screens(cfg, demo_csv, points_csv):
    """Every clickable row and link is reached by Tab and answers Enter or Space; a legend key spotlights on focus; the
    tabs say which is selected; a shipped dataset chosen fills the form and its upload waits for its request; reduced
    motion moves no camera; at 1,000 and 390 px the map's panels overlap neither each other nor the next section, and the
    rail's controls fit it."""
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    engine.ingest_file(demo_csv)
    upload_levels(engine, points_csv, geo="auto@8")
    port, app = stack(engine)
    keyed = "[...document.querySelectorAll(KEYED)]"
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # the developer mode: the graph and the technical fields
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.nodes.length > 0 && !G.refreshing", timeout=60000)
        rail = page.evaluate("[$('sample').getBoundingClientRect().right, document.querySelector('.rail').getBoundingClientRect().right]")
        assert rail[0] <= rail[1]  # the list of shipped datasets fits the rail rather than clips
        page.click("#view-switch button[data-view='table']")
        assert page.evaluate("[...document.querySelectorAll('#view-switch button')].map((b) => b.getAttribute('aria-selected'))")[:3] == [
            "false",
            "false",
            "true",
        ]
        page.focus("#ins-table tbody tr[data-id] >> nth=0")
        row = page.evaluate("document.activeElement.dataset.id")
        page.keyboard.press("Enter")
        page.wait_for_selector("#drawer:not([hidden])")
        page.focus("#inspector .nb >> nth=0")
        title = page.inner_text("#drawer-title")
        page.keyboard.press(" ")
        page.wait_for_function(f"$('drawer-title').textContent !== {json.dumps(title)}", timeout=10000)  # the drawer follows the row
        assert page.evaluate("document.activeElement.id") == "drawer-title"  # and the focus stays in it
        page.focus("#ins-table th[data-k='support']")
        page.keyboard.press("Enter")
        assert page.evaluate("S.sort.k") == "support"  # a header sorts from the keyboard too
        assert row and page.evaluate(f"{keyed}.every((el) => el.tabIndex === 0)")
        page.click("#view-switch button[data-view='graph']")
        page.focus("#batches .ds-pick >> nth=0")
        picked = page.evaluate("document.activeElement.closest('.ds').dataset.ds")
        page.keyboard.press("Enter")
        page.wait_for_function(
            f"$('dataset-filter').value === {json.dumps(picked)} && document.activeElement.matches('.ds-pick[aria-pressed=\"true\"]')", timeout=10000
        )
        assert page.evaluate("document.activeElement.closest('.ds').dataset.ds") == picked  # the replaced row keeps the focus
        page.evaluate("() => { $('dataset-filter').value = ''; return refreshBatches(); }")
        page.focus("#legend .key[data-key='up']")
        assert page.evaluate("S.cy.elements('.lg-dim').length > 0")
        page.focus("#question")
        assert page.evaluate("S.cy.elements('.lg-dim').length === 0")
        page.evaluate("() => { $('use-llm').checked = false; return ask('Чому margin нижчий для phones у US?'); }")
        page.click(".bot-tabs button >> nth=0")
        assert page.get_attribute(".bot-tabs button >> nth=0", "aria-selected") == "true"
        page.focus("#thread .ev >> nth=0")
        page.keyboard.press("Enter")
        assert page.evaluate("S.hl && S.hl.evidence.length === 1")  # the card's path isolated, as a click does
        page.evaluate(  # a shipped dataset's upload held, and never sent: the button waits for it
            "() => { window.__real = api; window.__sent = []; window.api = (path, opts) => path === '/api/upload'"
            " ? (__sent.push(opts.body.get('demo')), new Promise((r) => { window.__go = () => r({}); })) : __real(path, opts); }"
        )
        page.wait_for_function("$('sample').options.length > 1", timeout=10000)
        page.click("#add-btn")
        page.select_option("#sample", "retail_synthetic")  # the form takes its options and proposes its role
        assert page.evaluate("$('file-name').textContent === 'retail_synthetic.csv' && $('role-table').checked")
        page.click("#upload-btn")
        assert page.is_disabled("#upload-btn")
        page.evaluate("() => { window.api = __real; __go(); }")
        page.wait_for_function("$('sample').value === '' && !$('upload-dialog').open", timeout=10000)  # sent: the dialog closes
        assert page.evaluate("__sent") == ["retail_synthetic"]  # one request, by the dataset's key
        assert not page.evaluate("document.querySelector('#upload-form details.adv').open")  # sent: the options fold away
        page.wait_for_function("!S.cy.animated()", timeout=5000)  # the flight to the evidence card ended
        page.emulate_media(reduced_motion="reduce")
        page.evaluate("() => focusNodes([S.nodes[0].id])")
        assert not page.evaluate("S.cy.animated()")  # no camera flight
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function("G.data && G.map.loaded() && document.querySelector('#mw-body .mw-i')", timeout=60000)
        # the zone's edges: a Shift-click selects nothing, a polygon over no cell leaves no zone, a drawing ends with its view,
        # and a zone none of whose cells the map draws (another level's) is outlined by its polygon and dims nothing
        at = page.evaluate("""() => { closeDrawer(); const c = G.map.getCenter(), d = (f) => (centre(f)[0] - c.lng) ** 2 + (centre(f)[1] - c.lat) ** 2;
            const p = G.map.project(centre(G.data.features.reduce((a, f) => (d(f) < d(a) ? f : a)))), r = $('map').getBoundingClientRect();
            return [r.x + p.x, r.y + p.y]; }""")
        page.keyboard.down("Shift")
        page.mouse.click(*at)
        page.keyboard.up("Shift")
        page.wait_for_selector(".sig-pop .pp-head")
        assert not page.query_selector(".sig-pop .pp-ask") and page.evaluate("GeoMap.zone().length") == 0  # the cell's popup, no zone
        empty = "() => { setDrawing(true); G.draw.push([0, 0], [0.001, 0], [0, 0.001]); finishPolygon(); return [$('map-draw').textContent, GeoMap.zone().length, GeoMap.polygon()]; }"
        text, cells, polygon = page.evaluate(empty)
        assert text.startswith("У полігоні немає комірок мапи") and cells == 0 and polygon is None
        page.evaluate("setDrawing(true)")
        page.click("#view-switch button[data-view='table']")
        assert page.evaluate("G.draw") is None  # Enter and Backspace are the other view's again
        page.click("#view-switch button[data-view='map']")
        lost = "() => { GeoMap.setPolygon(G.data.features[0].geometry); G.zone = new Set(['8a2a1072b59ffff']); paintZone(); return [G.map.getPaintProperty('cells-fill', 'fill-opacity'), G.map.getSource('zone').serialize().data.features.map((f) => f.geometry.type)]; }"
        assert page.evaluate(lost) == [0.55, ["Polygon"]]
        page.evaluate("GeoMap.clearZone()")
        page.evaluate("() => document.querySelector('#mw-body .mw-i').click()")
        page.wait_for_selector("#map-card:not([hidden])")
        page.evaluate("() => { GeoMap.setPolygon(G.data.features[0].geometry); GeoMap.closePopup(); }")  # with a zone
        boxes = """() => { const r = (q) => { const e = document.querySelector(q); return e && !e.hidden && e.getClientRects().length ? e.getBoundingClientRect() : null; };
            const panels = ['.map-hud', '#map-window', '#map-card'].map(r).filter(Boolean), cut = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
            return { overlaps: panels.flatMap((a, i) => panels.slice(i + 1).filter((b) => cut(a, b))).length, map: r('#map').height, canvas: r('.canvas').bottom, chat: r('.chat').top }; }"""
        for width, height in ((1500, 844), (1200, 844), (1200, 600), (1001, 600), (1000, 844), (750, 844), (390, 844)):  # a short window too
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(300)
            got = page.evaluate(boxes)
            assert got["overlaps"] == 0 and got["map"] >= 300 and (width > 1000 or got["canvas"] <= got["chat"] + 1), (width, height, got)
        browser.close()
        assert not errors, errors
