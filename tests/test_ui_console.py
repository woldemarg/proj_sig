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


@pytest.mark.browser
def test_the_form_proposes_a_role_and_the_user_can_change_it(cfg, tmp_path):
    """A file's header proposes its role — a latitude and a longitude make events, none a table —; the role confirmed or
    changed is the one sent: events read on the map with every field of ``/api/upload`` reachable under *Додатково*, a
    table without geo, *Лише шар* stored as a layer and never mined (docs/08_interface.md §8.2)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import re

    import numpy as np
    import pandas as pd
    from conftest import make_engine, serve, stack
    from dev import chromium

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    rng = np.random.default_rng(7)
    points = pd.DataFrame({"lat": 41.8 + rng.random(300) * 0.1, "lon": -87.7 + rng.random(300) * 0.1, "kind": rng.choice(["a", "b"], 300)})
    points.to_csv(tmp_path / "points.csv", index=False)
    points[["kind"]].assign(value=rng.random(300)).to_csv(tmp_path / "rows.csv", index=False)
    engine = make_engine(cfg)
    port, app = stack(engine)
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("typeof S !== 'undefined'", timeout=20000)
        assert page.is_disabled("#role-events")  # nothing picked: no role to confirm
        page.set_input_files("#file", str(tmp_path / "rows.csv"))
        page.wait_for_function("document.getElementById('role-table').checked", timeout=5000)
        assert page.is_disabled("#geo") and page.text_content("#upload-btn") == "Аналізувати"
        page.set_input_files("#file", str(tmp_path / "points.csv"))
        page.wait_for_function("document.getElementById('role-events').checked", timeout=5000)
        assert page.input_value("#geo") == "auto@auto" and page.is_enabled("#role-table")
        page.evaluate("document.querySelector('#upload-form details.adv').open = true")  # every field of the upload, held
        for field, value in {"#time": "when", "#where": "kind=a", "#count-values": "kind=a"}.items():
            page.fill(field, value)
        page.select_option("#baseline", "zone")
        page.set_input_files("#context", str(tmp_path / "points.csv"))
        page.set_input_files("#metadata", str(tmp_path / "rows.csv"))
        sent: list[bytes] = []
        page.route("**/api/upload", lambda route: (sent.append(route.request.post_data_buffer), route.abort()))
        page.click("#upload-btn")
        page.wait_for_function("document.getElementById('upload-msg').classList.contains('err')", timeout=5000)
        page.unroute("**/api/upload")
        page.fill("#time", "")  # the layer below: its points alone
        page.set_input_files("#metadata", [])
        assert {m.decode() for m in re.findall(rb'; name="([^"]+)"', sent[0])} == {
            *("file", "bins", "categories", "geo", "table", "time", "tz", "grid", "where", "count_values", "baseline"),
            *("context", "context_geo", "context_ring", "source", "reliability", "indicators", "glossary", "metadata"),
        }
        page.check("#role-layer")  # changed by hand: the points kept as a layer
        assert page.text_content("#upload-btn") == "Зберегти шар"
        page.click("#upload-btn")
        page.wait_for_function("document.getElementById('batches').textContent.includes('points.csv · шар')", timeout=20000)
        assert page.is_disabled("#role-events") and not page.is_checked("#role-layer")  # the form reset
    (layer,) = engine.ws.list_batches()
    assert (layer["status"], layer["geo"]) == ("LAYER", "auto")
    assert not errors, errors
