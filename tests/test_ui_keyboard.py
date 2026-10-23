"""The console from the keyboard, on narrow screens and with reduced motion (docs/08_interface.md §8.2)."""

from __future__ import annotations

import json

import pytest
from conftest import make_engine, serve, stack, upload_levels
from dev import chromium


@pytest.mark.browser
def test_the_console_works_from_the_keyboard_and_on_narrow_screens(cfg, demo_csv, points_csv):
    """Every clickable row and link is reached by Tab and answers Enter or Space; a legend key spotlights on focus; the
    tabs say which is selected; the demo buttons wait for their request; reduced motion moves no camera; at 1,000 and
    390 px the map's panels overlap neither each other nor the next section, and the rail's buttons wrap."""
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
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.nodes.length > 0 && !G.refreshing", timeout=60000)
        rail = page.evaluate("[$('demo-geo-btn').getBoundingClientRect().right, document.querySelector('.rail').getBoundingClientRect().right]")
        assert rail[0] <= rail[1]  # the rail's buttons wrap rather than clip
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
        assert page.evaluate("document.activeElement.closest('.ds').dataset.ds") == picked  # the replaced card keeps the focus
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
        page.evaluate(  # the demo request held, and never sent: both buttons wait for it
            "() => { window.__real = api; window.api = (path, opts) => path === '/api/demo' ? new Promise((r) => { window.__go = () => r({}); }) : __real(path, opts); }"
        )
        page.click("#demo-btn")
        assert page.is_disabled("#demo-btn") and page.is_disabled("#demo-geo-btn")
        page.evaluate("() => { window.api = __real; __go(); }")
        page.wait_for_function("!$('demo-btn').disabled && !$('demo-geo-btn').disabled", timeout=10000)
        page.wait_for_function("!S.cy.animated()", timeout=5000)  # the flight to the evidence card ended
        page.emulate_media(reduced_motion="reduce")
        page.evaluate("() => focusNodes([S.nodes[0].id])")
        assert not page.evaluate("S.cy.animated()")  # no camera flight
        page.click("#view-switch button[data-view='map']")
        page.wait_for_function("G.data && G.map.loaded() && document.querySelector('#mw-body .mw-i')", timeout=60000)
        page.evaluate("() => document.querySelector('#mw-body .mw-i').click()")
        page.wait_for_selector("#map-card:not([hidden])")
        page.evaluate("() => G.data.features.slice(0, 3).forEach((f) => GeoMap.toggleZone(f.properties.h3))")  # the zone bar too
        page.wait_for_selector("#map-zone:not([hidden])")
        boxes = """() => { const r = (q) => { const e = document.querySelector(q); return e && !e.hidden && e.getClientRects().length ? e.getBoundingClientRect() : null; };
            const panels = ['.map-hud', '#map-window', '#map-card', '#map-zone'].map(r).filter(Boolean), cut = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
            return { overlaps: panels.flatMap((a, i) => panels.slice(i + 1).filter((b) => cut(a, b))).length, map: r('#map').height, canvas: r('.canvas').bottom, chat: r('.chat').top }; }"""
        for width, height in ((1500, 844), (1200, 844), (1200, 600), (1001, 600), (1000, 844), (750, 844), (390, 844)):  # a short window too
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(300)
            got = page.evaluate(boxes)
            assert got["overlaps"] == 0 and got["map"] >= 300 and (width > 1000 or got["canvas"] <= got["chat"] + 1), (width, height, got)
        browser.close()
        assert not errors, errors
