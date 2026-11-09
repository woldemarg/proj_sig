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
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # *Додатково* is the developer's
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("typeof S !== 'undefined'", timeout=20000)
        page.click("#add-btn")
        assert page.is_disabled("#role-events")  # nothing picked: no role to confirm
        page.evaluate(
            "() => { S.samples = { ...S.samples, x: { key: 'x', title: 'X', file: 'x.csv', rows: 1, description: '', license: '', options: { geo: 'lat,lon@8', time: 'when', categories: 'kind' } } };"
            " $('sample').insertAdjacentHTML('beforeend', sampleItem(S.samples.x)); }"
        )
        page.select_option("#sample", "x")
        assert (page.input_value("#geo"), page.input_value("#time")) == ("lat,lon@8", "when")  # the set's own options
        page.set_input_files("#file", str(tmp_path / "rows.csv"))
        page.wait_for_function("document.getElementById('role-table').checked", timeout=5000)
        assert [page.input_value(f) for f in ("#sample", "#geo", "#time", "#categories")] == [""] * 4  # one's own file: none of the set's
        assert page.is_disabled("#geo") and page.text_content("#upload-btn") == "Аналізувати"
        page.set_input_files("#file", str(tmp_path / "points.csv"))
        page.wait_for_function("document.getElementById('role-events').checked", timeout=5000)
        assert page.input_value("#geo") == "auto@auto" and page.is_enabled("#role-table")
        page.evaluate("document.querySelector('#upload-form details.adv').open = true")  # every field of the upload, held
        for field, value in {"#time": "when", "#where": "kind=a"}.items():
            page.fill(field, value)
        page.select_option("#baseline", "zone")
        page.set_input_files("#metadata", str(tmp_path / "rows.csv"))
        sent: list[bytes] = []
        page.route("**/api/upload", lambda route: (sent.append(route.request.post_data_buffer), route.abort()))
        page.click("#upload-btn")
        page.wait_for_function(
            "document.getElementById('upload-note').classList.contains('err')", timeout=5000
        )  # said in the dialog, which stays open
        page.unroute("**/api/upload")
        page.fill("#time", "")  # the layer below: its points alone
        page.set_input_files("#metadata", [])
        assert {m.decode() for m in re.findall(rb'; name="([^"]+)"', sent[0])} == {
            *("file", "bins", "categories", "geo", "table", "time", "tz", "grid", "where", "baseline"),
            *("source", "reliability", "indicators", "glossary", "metadata"),
        }
        page.check("#role-layer")  # changed by hand: the points kept as a layer
        assert page.text_content("#upload-btn") == "Зберегти шар"
        page.click("#upload-btn")
        page.wait_for_function(
            "document.querySelector('#batches .ds-loose')?.textContent.includes('points')", timeout=20000
        )  # a layer no run reads, apart
        assert page.is_disabled("#role-events") and not page.is_checked("#role-layer")  # the form reset
        assert [page.input_value(f) for f in ("#geo", "#where", "#baseline")] == [""] * 3  # the next file starts from the defaults
    (layer,) = engine.ws.list_batches()
    assert (layer["status"], layer["geo"]) == ("LAYER", "auto")
    assert not errors, errors


@pytest.mark.browser
def test_the_add_context_layer_action_stores_a_layer_and_runs_it_one_to_one_on_one_level_and_one_ring(cfg, tmp_path, monkeypatch):
    """*+ Додати шар* under a base layer (docs/08_interface.md §8.2): a shipped set picked in its dialog is kept
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
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # the developer mode: the dialog's types and disk
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_selector(".ds-link", timeout=20000)
        page.click(".ds-link")
        assert (page.input_value("#link-counts"), page.input_value("#link-ring")) == (
            "kind=a",
            "0",
        )  # the base's types, the default disk: the cell alone
        assert page.is_checked(".tp[data-v='kind=a']") and page.inner_text("#link-sum") == "kind: a"  # its box, from the base's shares
        page.click("#link-dd summary")
        page.click(".tp-opt:has(.tp[data-v='kind=b'])")  # picked from the list: the field follows
        assert page.input_value("#link-counts") == "kind=a, kind=b" and page.inner_text("#link-sum") == "kind: a, b"
        page.click("#link-title")  # a click elsewhere closes the list
        assert not page.evaluate("$('link-dd').open")
        page.fill("#link-counts", "kind=a")
        page.select_option("#link-ring", "1")  # the planted dependence fills a school's disk: asked for, as a user would
        page.click(".link-opt[data-sample='schools']")  # one click runs it
        kids = "document.querySelector('#batches .ds-kids')?.textContent.includes('%s')"
        page.wait_for_function(kids % "schools", timeout=20000)  # the run under its base, named by its layer
        settled()
        page.evaluate("refreshBatches()")
        assert page.locator(".ds-link").count() == 1  # the base alone: neither the run nor the layer
        (layer,) = [b["upload_id"] for b in engine.ws.list_batches() if b["status"] == "LAYER"]
        page.click(".ds-link")
        assert "уже додано" in page.inner_text(f".link-opt[data-layer='{layer}']")
        page.select_option("#link-ring", "1")  # the same disk as the run
        page.click(f".link-opt[data-layer='{layer}']")  # the same layer again: the run exists
        page.wait_for_function("document.getElementById('upload-msg').textContent.includes('уже є')", timeout=20000)
        page.click(".ds-link")
        page.fill("#link-counts", "")  # no types: the composition alone
        page.select_option("#link-ring", "1")
        page.click(".link-opt[data-sample='patrols']")
        page.wait_for_function(kids % "patrols", timeout=20000)
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
PAIR = {"target": "share_kind_a", "effect": 0.5, "local": 0.12, "baseline": 0.07}


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
    typical shares, never «у N раза» —, one the insight's tested primary and the other the same cohort's other metric,
    untested on its own; a run without counters says the count was not asked, a sparse value too few events, a
    question the insight holds no metric for that it was not measured, a zone baseline its mode instead of the shares;
    ``points`` answers the first question, a metric that is neither a count nor a share neither."""
    with card_page(cfg) as page:
        both = page.evaluate(CARD, {"target": "count_kind_a", "ds": COUNTED, "pair": PAIR})
        mix = page.evaluate(CARD, {"target": "share_kind_a", "shift": SHARE, "ds": COUNTED})
        unasked = page.evaluate(CARD, {"target": "share_kind_a", "shift": SHARE, "ds": {}})
        sparse = page.evaluate(CARD, {"target": "share_kind_b", "shift": SHARE | {"metric": "share_kind_b"}, "ds": COUNTED})
        zone = page.evaluate(CARD, {"target": "share_kind_a", "shift": SHARE, "ds": COUNTED | {"baseline": "zone"}})
        points = page.evaluate(CARD, {"target": "points", "ds": {}})
        median = page.evaluate(CARD, {"target": "pm25_median", "ds": COUNTED})
    assert "Біля schools — target.csv" in both and "рівень H3 r9 · у комірці та шести сусідніх" in both
    assert "так · не пояснюється фоном зони · на комірку ≈ 0,1 км²" in both and "Більше подій і інший склад" not in both
    assert "так · та сама когорта, окремо не перевірялось · типова частка 12 % проти 7 %" in both  # the other question, described
    assert "типова частка 12 % проти 7 %" in mix and "раза" not in mix and "більше подій\nу цій знахідці не вимірювалось" in mix.lower()
    assert "не перевірялось" in unasked and "замало даних для частоти на цьому рівні" in sparse
    assert "відносно своєї зони" in zone and "проти" not in zone
    assert "більше подій\nтак" in points.lower() and "більше подій" not in median.lower() and "Біля schools" in median


@pytest.mark.browser
def test_the_other_question_reads_the_same_cohort_untested_and_never_a_quadrant(cfg):
    """The card never joins two tests into a quadrant: the other question is the same cohort's other metric when the
    insight holds it (``views.link_pair``), said to be untested on its own, with its sign; without it the question was
    not measured in this insight. A count's pair reads its counts per cell."""
    with card_page(cfg) as page:
        cards = {
            name: page.evaluate(CARD, {"target": target, "shift": shift, "ds": COUNTED, "pair": pair}).lower()
            for name, target, shift, pair in (
                ("count", "count_kind_a", None, PAIR),
                ("fewer", "count_kind_a", None, PAIR | {"effect": -0.5}),
                ("alone", "count_kind_a", None, None),
                ("share", "share_kind_a", SHARE, {"target": "count_kind_a", "effect": 1.2, "local": 3.0, "baseline": 1.0}),
            )
        }
    assert all("більше подій і інший склад" not in c for c in cards.values())
    assert "інший склад\nтак · та сама когорта, окремо не перевірялось · типова частка 12 % проти 7 %" in cards["count"]
    assert "інший склад\nменше · та сама когорта, окремо не перевірялось" in cards["fewer"]
    assert "інший склад\nу цій знахідці не вимірювалось" in cards["alone"]
    assert "більше подій\nтак · та сама когорта, окремо не перевірялось · 3 проти 1" in cards["share"]


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
    assert (
        "stations" in places
        and "ділянки з сусідніми комірками перетинаються: 62 %" in places
        and "shops" in places
        and "ділянки торкаються" in places
    )
    assert "own run" not in places and "також у тих самих місцях" not in alone
    assert not any(w in places.lower() for w in ("спільний ефект", "взаємод", "оцінка", "поруч"))


@pytest.mark.browser
def test_a_runs_insights_are_highlighted_in_every_block_and_the_base_layers_are_not(cfg, tmp_path):
    """A layer link's insights carry their layer in the blocks that are there (docs/08_interface.md §8.2): a chip in the
    insights table and the drawer, a ring of the layer's colour in the graph, the layer named in the legend, the chip an
    answer's card reads; the base layer's own insights carry none of it."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import httpx
    from conftest import make_engine, serve, stack
    from dev import chromium
    from test_crosstype import RES, cross_run, planted, upload_file

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    target, context = planted(tmp_path)
    engine = make_engine(cfg)
    port, app = stack(engine)
    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as client, sync_api.sync_playwright() as p:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        run = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
        own = {x["id"] for x in engine.ws.patterns() if x["dataset_id"] == run["dataset_id"]}
        base_ds = next(x["dataset_id"] for x in engine.ws.patterns() if x["dataset_id"] != run["dataset_id"])
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("S.nodes && S.nodes.length > 0 && !!S.cy", timeout=30000)
        page.click("#view-switch button[data-view='table']")
        table = dict(
            page.evaluate(
                "[...document.querySelectorAll('#ins-table tbody tr')].map((tr) => [tr.dataset.id, tr.querySelector('.tag.layer')?.textContent || ''])"
            )
        )
        ringed = dict(page.evaluate("""S.cy.nodes('[kind="Pattern"]').map((n) => [n.id(), n.style('border-width') === '3px'])"""))
        legend = page.evaluate("[document.getElementById('lg-layers').hidden, document.getElementById('lg-layers').textContent]")
        chips = page.evaluate(f"[layerChip('{run['dataset_id']}'), layerChip('{base_ds}')]")  # an answer's card reads its item's dataset
        page.evaluate(f"inspect('{sorted(own)[0]}')")
        page.wait_for_function("document.querySelector('#inspector .link-card .tag.layer')", timeout=10000)
        page.click("#dev-btn")  # the developer mode off: an insight revealed from the table stays on the table
        page.click("#view-switch button[data-view='table']")
        page.evaluate(f"reveal('{sorted(own)[0]}')")
        stayed = page.evaluate("S.view")
        page.click("#dev-btn")
    assert own and {pid for pid, chip in table.items() if chip} == own and all("schools" in table[pid] for pid in own)
    assert {pid for pid, has in ringed.items() if has} == own
    assert legend[0] is False and "schools" in legend[1] and "schools" in chips[0] and chips[1] == "" and stayed == "table"
    assert not errors, errors


# the rail's functions on records of their shape: an upload u read twice at r9, one layer run on it, the layer L
TREE = """() => {
  const row = (id, extra) => ({ batch_id: `B-${id}`, dataset_id: id, upload_id: "u", filename: "f.csv", level: "9", status: "READY", metrics: {}, ...extra });
  window.__twin = [row("dA", { linkable: true }), row("dB"), row("dR", { cross: ["L"], context: [{ name: "l" }] }),
                   { batch_id: "B-L", upload_id: "L", filename: "l.csv", level: "layer", status: "LAYER", layer_of: 1 }];
  window.__real = api; window.__deleted = [];
  window.api = (path, opts = {}) => {
    if (opts.method === "DELETE") return (window.__deleted.push(path), window.__fail && __deleted.length === 2 ? Promise.reject(new Error("busy")) : Promise.resolve({}));
    if (path === "/api/crosstype") return Promise.reject(new Error("refused"));
    return path === "/api/batches" ? Promise.resolve(S.batches) : __real(path, opts);
  };
  window.confirm = () => true;
}"""


@pytest.mark.browser
def test_the_rail_tree_says_why_no_layer_shows_a_levels_runs_once_and_deletes_with_care(cfg):
    """The tree's guards (docs/08_interface.md §8.2): an upload whose finest map is coarser than r8 says once, under its
    name, why no layer can be added; an upload's own bin takes every level with their runs and layer; two readings of one level show its runs once and keep them when one of the readings is deleted, one
    reading alone takes them and their layer along, and a failure midway says what is left; a failed run keeps the
    types and the disk picked for it; a share's column with a space or an *is* is split where its glossary says."""
    sync_api = pytest.importorskip("playwright.sync_api")
    from conftest import make_engine, serve, stack
    from dev import chromium

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    port, app = stack(make_engine(cfg))
    with serve(app, port=port) as url, sync_api.sync_playwright() as p:
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")  # the dialog's disk
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("typeof S !== 'undefined' && S.batches", timeout=20000)
        page.evaluate(TREE)
        coarse = page.evaluate("uploadGroup([{ batch_id: 'g', upload_id: 'c', filename: 'c.csv', level: '6', status: 'READY', metrics: {} }], false)")
        assert "рівня 8" in coarse and "ds-link" not in coarse
        group = """() => { const lv = (l) => ({ batch_id: `g${l}`, dataset_id: `d${l}`, upload_id: 'c', filename: 'c.csv', level: l, status: 'READY', metrics: {} });
          const box = document.createElement('div'); box.innerHTML = uploadGroup([lv('6'), lv('5'), lv('table')], false);
          return [box.querySelectorAll('.ds-why').length, box.querySelector('.ds-group-head + .ds-why') !== null, box.querySelector('.ds-group-head .ds-del').dataset.del]; }"""
        assert page.evaluate(group) == [1, True, "d6 d5 dtable"]  # said once under the file's name; its bin takes every level
        assert page.evaluate("(uploadGroup(__twin.slice(0, 3), false).match(/ds-glyph/g) || []).length") == 1  # under one reading
        delete = "async (rows) => { S.batches = rows; __deleted.length = 0; await deleteDataset('dB', 'x'); return __deleted; }"
        assert page.evaluate(f"({delete})(__twin)") == ["/api/datasets/dB"]  # the other reading keeps the runs
        alone = page.evaluate(f"({delete})([__twin[1], __twin[2], __twin[3]])")
        assert alone == ["/api/datasets/dR", "/api/datasets/dB", "/api/datasets/B-L"]  # the runs, the reading, the layer
        whole = page.evaluate(
            "() => { const box = document.createElement('div'); box.innerHTML = uploadGroup(__twin.slice(0, 3), false); return box.querySelector('.ds-group-head .ds-del').dataset.del; }"
        )
        assert page.evaluate(
            f"(async (rows) => {{ S.batches = rows; __deleted.length = 0; await deleteDataset('{whole}', 'x'); return __deleted; }})(__twin)"
        ) == [
            "/api/datasets/dR",
            "/api/datasets/dA",
            "/api/datasets/dB",
            "/api/datasets/B-L",
        ]  # a file's bin: both readings with their runs and the layer no run reads
        page.evaluate("window.__fail = true")
        page.evaluate(f"({delete})([__twin[1], __twin[2], __twin[3]])")
        assert page.inner_text("#upload-msg").startswith("Видалено не все: busy")
        page.evaluate(
            "() => { window.__fail = false; S.batches = __twin; openLink('B-dA'); $('link-counts').value = 'kind=a'; $('link-ring').value = '1'; }"
        )
        page.evaluate("runLink({ layer: 'L' })")
        page.wait_for_function("$('link-msg').textContent === 'refused'", timeout=10000)
        assert page.evaluate("[$('link-counts').value, $('link-ring').value]") == ["kind=a", "1"]  # as picked
        shares = """profileTypes({ profile: { numerics: ["share_Primary Type_THEFT", "share_this_is_x_yes"], geo: { glossary: {
          "share_Primary Type_THEFT": "share of the cell's events whose Primary Type is THEFT",
          "share_this_is_x_yes": "share of the cell's events whose this is x is yes (every such event also has: a b)" } } } })"""
        assert page.evaluate(shares) == {"Primary Type": ["THEFT"], "this_is_x": ["yes"]}
    assert not errors, errors
