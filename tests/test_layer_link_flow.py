"""The simplified path of a layer link end to end (docs/extra_funcs/spatial_analytics_multi_type/cross_layer_analysis_concept.md
§10): from an empty workspace, without the developer mode, on the shipped Chicago tables — the crimes read through the
three steps, a context layer added from their card (the bars: the one with a finding at the default disk, the cell alone), the result's card, the marks in the blocks that are there, the layer
hidden on the server and shown again, a second layer as a run of its own."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from insight_graph_service.core.workspace import TERMINAL

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.browser
def test_the_simplified_flow_runs_from_an_empty_workspace_without_the_developer_mode(cfg):
    """The crimes picked, their role proposed (events) and read at r9; *+ Додати шар* under them with the shipped
    'L' stations — kept as a layer, one run at the crimes' finest level and the default disk —; the run's insight in
    the table with its layer's chip and in the graph with its ring; its drawer's card naming the layer, the level and the
    disk and answering *інший склад*; the stations hidden from their card — the insight leaves the table and
    ``POST /api/search`` cites it no more — and shown again — it is cited again —; the bars added by the same action, a
    second run of its own; the crimes deleted with both layers. No cross-type panel, no joint run, every run of one layer."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import httpx
    from conftest import make_engine, serve, stack
    from dev import chromium

    exe = chromium()
    if exe is None:
        pytest.skip("no local Chromium for Playwright")
    engine = make_engine(cfg)
    port, app = stack(engine)

    def finished(**match) -> dict:
        deadline = time.time() + 300
        while not (done := [b for b in engine.ws.list_batches() if all(b.get(k) == v for k, v in match.items()) and b["status"] in TERMINAL]):
            assert time.time() < deadline, f"no finished batch with {match}"
            time.sleep(0.5)
        return done[-1]

    def link(page, key: str) -> None:  # *+ Додати шар* under the crimes with a shipped set: one click runs it
        page.click(".ds-link")
        page.click(f".link-opt[data-sample='{key}']")

    with serve(app, port=port) as url, httpx.Client(base_url=url, timeout=120) as client, sync_api.sync_playwright() as p:
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("document.querySelector('#sample option[value=chicago_crimes]')", timeout=20000)
        assert not page.locator("#view-switch button[data-view='graph']").is_visible()  # no developer mode
        page.click("#add-btn")
        page.select_option("#sample", "chicago_crimes")
        assert page.is_checked("#role-events")  # step 2: the role proposed
        page.click("#upload-btn")  # step 3
        crimes = finished(filename="chicago_crimes.csv")
        assert (crimes["status"], crimes["level"]) == ("READY", "9")
        page.wait_for_selector(".ds-link", timeout=30000)
        link(page, "chicago_bars")
        bars = finished(filename="chicago_bars.csv", status="LAYER")
        run = finished(cross=[bars["upload_id"]])
        assert (run["status"], run["level"], run["context_ring"]) == ("READY", "9", 0)  # the default disk: the cell alone
        (found,) = [x["id"] for x in engine.ws.patterns() if x["dataset_id"] == run["dataset_id"]]
        page.evaluate("refreshBatches()")
        page.wait_for_function(f"S.nodes.some((n) => n.id === '{found}')", timeout=30000)
        page.click("#view-switch button[data-view='table']")
        assert "контекстний шар: Бари й заклади з алкоголем у Чикаго" in page.inner_text(
            f"#ins-table tr[data-id='{found}']"
        )  # the layer by its title
        assert page.evaluate(f"S.cy.getElementById('{found}').style('border-width')") == "3px"
        page.evaluate(f"inspect('{found}')")
        page.wait_for_selector("#inspector .link-card", timeout=10000)
        card = page.text_content("#inspector .link-card")  # its words, not their case on screen
        assert "Біля chicago_bars — chicago_crimes.csv" in card and "рівень H3 r9 · у самій комірці" in card
        assert "інший склад" in card.lower() and "дослідницька знахідка" in card
        question = engine.graph().nodes[found]["label"]
        cited = lambda: {c["pattern_id"] for c in client.post("/api/search", json={"question": question}).json()["citations"].values()}  # noqa: E731
        assert found in cited()
        page.click(".ds.layer:has-text('Бари') .ds-hide")
        page.wait_for_function(f"!S.nodes.some((n) => n.id === '{found}')", timeout=30000)
        assert found not in cited() and "приховано" in page.inner_text("#ins-table")
        page.click(".ds.layer:has-text('Бари') .ds-hide")
        page.wait_for_function(f"S.nodes.some((n) => n.id === '{found}')", timeout=30000)
        assert found in cited()
        link(page, "chicago_l_stations")  # the second layer: the same action, a run of its own
        stations = finished(filename="chicago_l_stations.csv", status="LAYER")
        second = finished(cross=[stations["upload_id"]])
        assert second["level"] == "9" and second["status"] in TERMINAL
        assert page.locator("#cross-panel, #cross-btn").count() == 0
        runs = [b for b in engine.ws.list_batches() if b.get("cross")]
        assert len(runs) == 2 and all(len(b["cross"]) == 1 for b in runs)
        page.evaluate("refreshBatches()")
        page.once("dialog", lambda d: d.accept())  # the crimes deleted: their two layers go with them, nothing stays apart
        page.hover(".ds:not(.layer) >> nth=0")
        page.click(".ds:not(.layer) .ds-del >> nth=0")
        page.wait_for_function("S.batches.length === 0", timeout=60000)
        assert not engine.ws.list_batches() and page.locator(".ds-loose").count() == 0
    console = [
        *(ROOT / "sig_web_console" / "static").glob("*.js"),
        ROOT / "sig_web_console" / "static" / "index.html",
        *(ROOT / "insight_contracts").glob("*.py"),
    ]
    assert not [f.name for f in console if any(w in f.read_text(encoding="utf-8") for w in ("cross-panel", "Перевірити разом"))]
    assert not errors, errors
