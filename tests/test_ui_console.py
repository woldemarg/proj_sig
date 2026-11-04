"""The console's screens for a person and the developer mode that adds the developer's tools to them, in a local
Chromium on the ``scripts/dev.py`` composite (docs/08_interface.md §8.2)."""

from __future__ import annotations

import pytest


@pytest.mark.browser
def test_the_developer_switch_hides_the_graph_the_sphere_and_the_technical_fields_by_default(cfg):
    """The console opens without the developer mode: no graph or 3D-sphere tab, no layout, no link toggles, the insights
    first; the switch turns them on, the choice survives a reload, and turning it off from the graph returns to the
    insights. A browser without storage opens with the switch off and no error (docs/08_interface.md §8.2)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    from conftest import make_engine, serve, stack
    from dev import chromium

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    port, app = stack(make_engine(cfg))
    tools = "#view-switch button[data-view='graph'], #view-switch button[data-view='sphere'], #layout, #legend section[data-dev]"
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("typeof S !== 'undefined' && S.view === 'table'", timeout=20000)
        assert not any(page.locator(sel).is_visible() for sel in tools.split(", ")) and page.get_attribute("#dev-btn", "aria-pressed") == "false"
        page.click("#dev-btn")
        assert all(page.locator(sel).is_visible() for sel in tools.split(", ")[:2]) and page.evaluate("localStorage.getItem('sig-dev')") == "on"
        page.reload()
        page.wait_for_function("typeof S !== 'undefined' && document.body.classList.contains('dev')", timeout=20000)
        page.click("#view-switch button[data-view='graph']")
        assert page.locator("#layout").is_visible()  # the graph's own control
        page.click("#dev-btn")
        assert page.evaluate("S.view") == "table" and not page.locator("#view-switch button[data-view='graph']").is_visible()
        private = browser.new_page(viewport={"width": 1500, "height": 900})
        private.add_init_script("Object.defineProperty(window, 'localStorage', { get() { throw new Error('denied'); } })")
        private.on("pageerror", lambda e: errors.append(str(e)))
        private.goto(f"{url}/")
        private.wait_for_function("typeof S !== 'undefined' && S.view === 'table'", timeout=20000)
        assert not private.locator("#view-switch button[data-view='graph']").is_visible()
    assert not errors, errors
