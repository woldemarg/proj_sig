"""The console's screens for a person and the developer mode that adds the developer's tools to them, in a local
Chromium on the ``scripts/dev.py`` composite (docs/08_interface.md §8.2)."""

from __future__ import annotations

import contextlib

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


@pytest.mark.browser
def test_the_add_context_layer_action_stores_a_layer_and_runs_it_one_to_one_on_one_level_and_one_ring(cfg, tmp_path, monkeypatch):
    """*+ контекстний шар* on a base layer's card (docs/08_interface.md §8.2): a shipped set picked in its dialog is kept
    as a layer with its time column and run with the base at its finest level, one layer, the default disk, the base's
    types confirmed; the same layer again is no new run and says so; another layer with no types is a second run of its
    own, without counters. Only the base's finest mined level offers the action, never a run or a layer."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import time

    import httpx
    import pandas as pd
    from conftest import make_engine, serve, stack
    from dev import chromium
    from test_crosstype import RES, planted, upload_file

    import insight_graph_service.server.app as app_module
    from insight_graph_service.core import uploads
    from insight_graph_service.core.workspace import TERMINAL

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    target, context = planted(tmp_path)
    pd.read_csv(context).assign(when="2025-06-15").to_csv(tmp_path / "patrols.csv", index=False)
    entries = [
        {"key": "schools", "file": "schools.csv", "title": "schools", "context_only": True, "options": {"geo": "lat,lon"}},
        {"key": "patrols", "file": "patrols.csv", "title": "patrols", "context_only": True, "options": {"geo": "lat,lon", "time": "when"}},
    ]
    monkeypatch.setattr(app_module, "DEMO_DIR", tmp_path)
    monkeypatch.setattr(app_module, "demo_datasets", lambda: entries)
    monkeypatch.setattr(uploads, "LINK_MIN_LEVEL", RES)  # the planted target is read at r7
    engine = make_engine(cfg)
    port, app = stack(engine)

    def settled() -> list[dict]:
        deadline = time.time() + 300
        while any(r["status"] not in TERMINAL for r in engine.ws.list_batches()):
            assert time.time() < deadline, "a batch still running"
            time.sleep(0.3)
        return engine.ws.list_batches()

    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as client, sync_api.sync_playwright() as p:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", count_values="kind=a")["upload_id"]
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_selector(".ds-link", timeout=20000)
        page.click(".ds-link")
        assert (page.input_value("#link-counts"), page.input_value("#link-ring")) == ("kind=a", "1")  # the base's types, the default disk
        page.select_option("#link-sample", "schools")
        page.click("#link-go")
        page.wait_for_function("document.getElementById('batches').textContent.includes('+ schools')", timeout=20000)
        settled()
        page.evaluate("refreshBatches()")
        assert page.locator(".ds-link").count() == 1  # the base alone: neither the run nor the layer
        (layer,) = [b["upload_id"] for b in engine.ws.list_batches() if b["status"] == "LAYER"]
        page.click(".ds-link")
        page.select_option("#link-layer", layer)  # the same layer again: the run exists
        page.click("#link-go")
        page.wait_for_function("document.getElementById('upload-msg').textContent.includes('уже є')", timeout=20000)
        page.click(".ds-link")
        page.select_option("#link-sample", "patrols")
        page.fill("#link-counts", "")  # no types: the composition alone
        page.click("#link-go")
        page.wait_for_function("document.getElementById('batches').textContent.includes('+ patrols')", timeout=20000)
        batches = settled()
    runs = sorted((b for b in batches if b.get("cross") and b["status"] != "SKIPPED"), key=lambda b: b["created_at"])
    patrols = next(b for b in batches if b["status"] == "LAYER" and b["filename"] == "patrols.csv")
    assert [(r["upload_id"], r["level"], r["context_ring"], len(r["cross"])) for r in runs] == [(a, str(RES), 1, 1)] * 2
    assert [c["value"] for c in runs[0]["count_values"]] == ["a"] and "count_values" not in runs[1] and patrols.get("time") == "when"
    assert sum(b["status"] == "SKIPPED" for b in batches) == 1
    assert not errors, errors


CARD = """(args) => {  // a layer link's card from the drawer's own function, laid out: its rows on lines of their own
  const p = { target: args.target, effect_size: args.effect ?? 1, shifts: [args.shift || { metric: args.target, local_median: 3, global_median: 1 }] };
  const d = { dataset: { cross: ["up-l"], cross_layers: ["schools"], filename: "target.csv", level: "9", ...args.ds },
              link: { disk: "у комірці та шести сусідніх", exposure: "на комірку ≈ 0,1 км²", pair: args.pair || null }, neighbors: args.neighbors || {} };
  const box = document.createElement("div"); box.innerHTML = linkCard(p, d);
  document.body.append(box); const text = box.innerText; box.remove(); return text;
}"""
COUNTED = {"count_values": [{"column": "kind", "value": "a", "status": "ok"}, {"column": "kind", "value": "b", "status": "sparse"}]}
SHARE = {"metric": "share_kind_a", "local_median": 0.12, "global_median": 0.07}
PAIR = {"id": "s", "target": "share_kind_a", "effect": 0.5}


@contextlib.contextmanager
def card_page(cfg):
    """The console on an empty workspace, ready to call its card function."""
    sync_api = pytest.importorskip("playwright.sync_api")
    from conftest import make_engine, serve, stack
    from dev import chromium

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    port, app = stack(make_engine(cfg))
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        page = p.chromium.launch(executable_path=exe).new_page()
        page.goto(f"{url}/")
        page.wait_for_function("typeof linkCard === 'function'", timeout=20000)
        yield page


@pytest.mark.browser
def test_the_card_answers_more_events_and_other_mix_in_two_rows_from_one_run(cfg):
    """A layer link's insight reads its run's two questions in words (docs/08_interface.md §8.2): the run's level and disk,
    *більше подій* from a count insight — its verdict and what a count is per —, *інший склад* from a share insight — the
    typical shares, never «у N раза» —, each answered by the insight or its pair; a run without counters says the count
    was not asked, a sparse value too few events, a zone baseline its mode instead of the shares; ``points`` answers the
    first question, a metric that is neither a count nor a share neither."""
    with card_page(cfg) as page:
        both = page.evaluate(CARD, {"target": "count_kind_a", "ds": COUNTED, "pair": PAIR})
        mix = page.evaluate(CARD, {"target": "share_kind_a", "shift": SHARE, "ds": COUNTED})
        unasked = page.evaluate(CARD, {"target": "share_kind_a", "shift": SHARE, "ds": {}})
        sparse = page.evaluate(CARD, {"target": "share_kind_b", "shift": SHARE | {"metric": "share_kind_b"}, "ds": COUNTED})
        zone = page.evaluate(CARD, {"target": "share_kind_a", "shift": SHARE, "ds": COUNTED | {"baseline": "zone"}})
        points = page.evaluate(CARD, {"target": "points", "ds": {}})
        median = page.evaluate(CARD, {"target": "pm25_median", "ds": COUNTED})
    assert "Біля schools — target.csv" in both and "рівень H3 r9 · у комірці та шести сусідніх" in both
    assert "так · не пояснюється фоном зони · на комірку ≈ 0,1 км²" in both and "Більше подій і інший склад" in both
    assert "типова частка 12 % проти 7 %" in mix and "раза" not in mix and "більше подій\nні" in mix.lower()
    assert "не перевірялось" in unasked and "замало даних для частоти на цьому рівні" in sparse
    assert "відносно своєї зони" in zone and "проти" not in zone
    assert "більше подій\nтак" in points.lower() and "більше подій" not in median.lower() and "Біля schools" in median


@pytest.mark.browser
def test_the_quadrant_needs_a_pair_with_the_same_expression_and_value(cfg):
    """*Більше подій і інший склад* together only with the server's pair (the run's insight with the same conditions on
    the same value's other question, ``views.link_pair``) and the count raised; without a pair, two rows on their own."""
    with card_page(cfg) as page:
        shown = {
            name: "Більше подій і інший склад" in page.evaluate(CARD, {"target": "count_kind_a", "effect": effect, "ds": COUNTED, "pair": pair})
            for name, effect, pair in (("pair", 1, PAIR), ("fewer", -1, PAIR), ("alone", 1, None))
        }
    assert shown == {"pair": True, "fewer": False, "alone": False}


@pytest.mark.browser
def test_same_places_reads_co_located_without_any_computation(cfg):
    """The same base layer's other links whose insights the graph places with this one (``CO_LOCATED`` rows the server
    names with their run's layers): the layer, the overlap's share for ``overlaps`` alone, the relation's words for the
    others; a row of no other link (no layers) is not one; no word of a joint effect or of nearness."""
    neighbors = {
        "CO_LOCATED": [{"id": "P2", "label": "x", "layers": ["stations"], "props": {"relation": "overlaps"}, "weight": 0.62}],
        "CO_LOCATED (in)": [
            {"id": "P3", "label": "y", "layers": ["shops"], "props": {"relation": "adjacent"}, "weight": 0.4},
            {"id": "P4", "label": "own run", "props": {"relation": "overlaps"}, "weight": 0.5},
        ],
    }
    args = {"target": "count_kind_a", "ds": COUNTED}
    with card_page(cfg) as page:
        text, alone = page.evaluate(CARD, args | {"neighbors": neighbors}), page.evaluate(CARD, args)
    places = text[text.index("також у тих самих місцях") :]
    assert "stations" in places and "ділянки перетинаються: 62 %" in places and "shops" in places and "ділянки торкаються" in places
    assert "own run" not in places and "також у тих самих місцях" not in alone
    assert not any(w in places.lower() for w in ("спільний ефект", "взаємод", "оцінка", "поруч"))
