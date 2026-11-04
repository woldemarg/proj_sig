"""A hidden context layer (docs/06_graph_and_storage.md §6.8, docs/07_question_answering.md §7.4): one unsigned flag on
its upload's records, its runs hidden through it, the search's pre-filter (model A) and the console's switch."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient
from test_crosstype import RES, cross_run, planted, upload_file

from insight_graph_service.core import uploads
from insight_graph_service.core.snapshot import metadata_digest
from insight_graph_service.server.app import create_app


def test_hidden_datasets_is_a_pure_function_of_the_registry():
    """A run reading a hidden layer is named by it (``hidden_by``) and its dataset is what a search leaves out; a run of
    another layer and the base stay; reading writes nothing."""
    batches = [
        {"batch_id": "base", "upload_id": "up-t", "dataset_id": "ds-t", "status": "READY"},
        {"batch_id": "layer", "upload_id": "up-l", "dataset_id": None, "status": "LAYER", "hidden": True},
        {"batch_id": "shown", "upload_id": "up-m", "dataset_id": None, "status": "LAYER"},
        {"batch_id": "r1", "upload_id": "up-t", "dataset_id": "ds-r1", "status": "READY", "cross": ["up-l"]},
        {"batch_id": "r2", "upload_id": "up-t", "dataset_id": "ds-r2", "status": "READY", "cross": ["up-m"]},
    ]
    before = copy.deepcopy(batches)
    state = uploads.cross_state(batches)
    assert state["r1"]["hidden_by"] == ["up-l"] and "hidden_by" not in state.get("r2", {}) and "hidden_by" not in state.get("base", {})
    assert uploads.hidden_datasets(batches) == {"ds-r1"} and batches == before


def test_a_layer_is_hidden_and_shown_through_its_upload_without_resigning_anything(engine, tmp_path):
    """``PATCH /api/uploads/{id}``: the layer's records say ``hidden``, its run ``hidden_by`` on ``GET /api/batches``,
    twice is once, ``false`` shows it again; no record's digest moves. Only a layer can be hidden: an upload no run reads
    as a layer is refused, an unknown one is 404."""
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        run = cross_run(client, a, b).json()["batch_id"]
        signed = {r["batch_id"]: r.get("metadata_digest") for r in engine.ws.list_batches()}
        assert client.patch(f"/api/uploads/{a}", json={"hidden": True}).status_code == 400  # the base: no run reads it as a layer
        assert client.patch("/api/uploads/up-000000000000", json={"hidden": True}).status_code == 404
        for _ in range(2):
            assert client.patch(f"/api/uploads/{b}", json={"hidden": True}).status_code == 200
        listed = {r["batch_id"]: r for r in client.get("/api/batches").json()}
        layer = next(r for r in listed.values() if r["upload_id"] == b)
        assert layer["hidden"] is True and listed[run]["hidden_by"] == [b]
        assert all(r.get("metadata_digest") == signed[r["batch_id"]] for r in engine.ws.list_batches())
        assert all(r.get("metadata_digest") in (None, metadata_digest(r)) for r in engine.ws.list_batches())  # still their own
        assert client.patch(f"/api/uploads/{b}", json={"hidden": False}).status_code == 200
        assert not any(r.get("hidden") or "hidden_by" in r for r in client.get("/api/batches").json())


def test_a_hidden_layer_is_prefiltered_from_seeds_expansion_and_evidence_unless_asked(engine, tmp_path):
    """Model A: with the schools hidden, a question their run answers best finds none of its insights — no seed, no step
    of the walk, no item, no vertical one, no slot of the budget taken —, the base layer's insights stay; asked with
    ``include_hidden`` the run's insights come back."""
    target, context = planted(tmp_path)
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        run = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
        assert client.patch(f"/api/uploads/{b}", json={"hidden": True}).status_code == 200
        graph = engine.graph()
        mine = {pid for pid, ins in graph.insights.items() if ins.dataset_id == run["dataset_id"]}
        question = graph.nodes[sorted(mine)[0]]["label"]  # a question the run's insight answers in its own words
        hidden = engine.search(question)
        shown = engine.search(question, include_hidden=True)
        payload = client.post("/api/search", json={"question": question}).json()
        asked = client.post("/api/search", json={"question": question, "include_hidden": True}).json()
    reached = lambda found: (  # noqa: E731
        {s.pattern_id for s in found.seeds}
        | {r.node_id for r in found.traversal.patterns}
        | set(found.traversal.traversed_nodes)
        | {r.node_id for r in found.traversal.excluded}
        | {it.pattern_id for it in found.evidence.items}
    )
    assert mine and not reached(hidden) & mine and reached(hidden)  # the base answers instead
    assert reached(shown) & mine
    cited = lambda body: {c["pattern_id"] for c in body["citations"].values()}  # noqa: E731  # what the narrator may cite
    assert not cited(payload) & mine and cited(asked) & mine


def test_a_hidden_run_is_no_vertical_step_of_a_visible_level():
    """A hidden coarser level of the same upload (two levels, one anchor, one region shared) is no step of a visible
    seed's scale chain: no vertical item, no step in the chains, and the prompt builds — before, the step stayed in the
    chain without its item and the payload raised (the closed review's F-01). Shown, it is the step."""
    from test_graph_query_engine_standalone import _levels, searched, state_of

    snap, vectors = _levels(-1.2, ("7", "5")), {"P1": [1.0, 0.0, 0.0], "P7": [0.0, 1.0, 0.0]}  # P7: the coarser level's dataset ds5
    steps = lambda found: {s["pattern_id"] for c in found.evidence.scale_chains for s in [*c["up"], *c["down"]]}  # noqa: E731
    hidden = searched(state_of(snap, vectors), hidden_datasets=frozenset({"ds5"}))
    assert "P7" not in steps(hidden) and all(it.pattern_id != "P7" for it in hidden.evidence.items)
    assert hidden.payload().evidence_prompt  # the prompt builds
    assert "P7" in steps(searched(state_of(snap, vectors)))


@pytest.mark.browser
def test_the_console_hides_a_layer_from_its_card_and_counts_what_it_hid(cfg, tmp_path):
    """The layer's card switches it (``PATCH``) and reads the state back from ``GET /api/batches``: hidden, its run's
    insights leave the table and the graph, the table's foot counts them, the card says *приховано*; a reload keeps it;
    shown again, they return."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import httpx
    from conftest import make_engine, serve, stack
    from dev import chromium

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
        mine = {x["id"] for x in engine.ws.patterns() if x["dataset_id"] == run["dataset_id"]}
        page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
        page.add_init_script("localStorage.setItem('sig-dev', 'on')")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{url}/")
        page.wait_for_function("S.nodes && S.nodes.length > 0 && !!S.cy", timeout=30000)
        page.click("#view-switch button[data-view='table']")
        every = page.evaluate("S.nodes.length")
        page.click(".ds-hide")
        page.wait_for_function(f"S.nodes.length === {every - len(mine)}", timeout=20000)
        assert not set(page.evaluate("S.nodes.map((n) => n.id)")) & mine
        assert not set(page.evaluate("""S.cy.nodes('[kind="Pattern"]').map((n) => n.id())""")) & mine
        assert f"ще {len(mine)}" in page.text_content("#ins-table") and "приховано" in page.text_content("#batches")
        page.reload()
        page.wait_for_function("S.nodes && S.nodes.length > 0", timeout=30000)
        assert not set(page.evaluate("S.nodes.map((n) => n.id)")) & mine  # the server's state, not the page's
        page.click(".ds-hide")
        page.wait_for_function(f"S.nodes.length === {every}", timeout=20000)
    assert not errors, errors
