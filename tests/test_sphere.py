"""3D latent sphere: lac's prosphera projection, drawn in the graph's language (docs/08_interface.md §8.4)."""

from __future__ import annotations

import threading

from conftest import FakeLLM, make_config
from fastapi.testclient import TestClient

from ltir.sphere import export_sphere, plotly_js_path, sphere_figure
from ltir.web.app import create_app


def test_sphere_uses_the_graphs_visual_language(hashed_engine):
    """The sphere draws what the graph draws: insight classes by colour, themes, the legend's link layers."""
    from ltir.sphere import LAYER_DEFAULTS

    fig = sphere_figure(hashed_engine)
    by_name = {t.name: t for t in fig.data if t.name}
    g = hashed_engine.graph()
    assert len(by_name["Themes"].x) == len(g.of_kind("Attractor")) and by_name["Themes"].marker.symbol == "diamond"
    classes = [by_name[n] for n in ("Metric higher", "Metric lower", "Correlation change") if n in by_name]
    assert sum(len(t.x) for t in classes) == len(g.of_kind("Pattern"))
    assert {i for t in classes for i in t.customdata} == {n["id"] for n in g.of_kind("Pattern")}  # click -> drawer
    assert list(by_name["Themes"].customdata) == sorted((n["id"] for n in g.of_kind("Attractor")), key=lambda a: int(a[2:]))
    assert max(abs(float(v)) for t in classes for v in list(t.x) + list(t.y) + list(t.z)) <= 1.0 + 1e-9  # on/inside the unit sphere
    for key, visible in LAYER_DEFAULTS.items():  # every layer the legend toggles exists and starts in the toggle's state
        traces = [t for t in fig.data if t.meta == key]
        assert traces and all(bool(t.visible) is visible for t in traces), key
    assert {"anchor", "up", "down"} <= {t.meta for t in fig.data}  # the legend's keys, not its display names, tag the traces
    assert fig.layout.showlegend is False and fig.layout.title.text is None  # the shared legend strip explains it
    flipped = sphere_figure(hashed_engine, layers={"sibling": True, "lattice": False}, palette={"bg": "#ffffff"})
    vis = {t.name: bool(t.visible) for t in flipped.data if t.name}
    assert vis["Siblings"] and not vis["Hierarchy"] and flipped.layout.template.layout.paper_bgcolor != fig.layout.template.layout.paper_bgcolor


def test_sphere_highlight_and_export(hashed_engine, tmp_path):
    qa = hashed_engine.ask("Why is margin lower for phones in the US?")
    fig = sphere_figure(hashed_engine, highlight=qa.highlight)
    names = {t.name for t in fig.data}
    assert {"Answer path", "Seed", "Evidence", "Themes visited"} <= names  # the graph's answer markers
    out = export_sphere(hashed_engine, tmp_path / "sphere.html")
    assert "plotly" in out.read_text(encoding="utf-8").lower()


def test_sphere_export_runs_off_the_batch_thread(tmp_path, demo_csv, monkeypatch):
    """A READY batch returns while its sphere export still runs."""
    import ltir.sphere as sphere
    from ltir.pipeline import Engine

    started, release = threading.Event(), threading.Event()

    def slow_export(engine, output=None, dataset=None):
        started.set()
        release.wait(30)
        return tmp_path / "sphere.html"

    monkeypatch.setattr(sphere, "export_sphere", slow_export)
    engine = Engine(make_config(tmp_path / "ws", sphere_export=True), llm=FakeLLM())
    record = engine.ingest_file(demo_csv)  # would block here if the export ran on the batch thread
    assert record["status"] == "READY" and started.wait(30)
    release.set()
    engine._sphere_pool.shutdown(wait=True)


def test_reset_during_sphere_export_leaves_nothing_behind(tmp_path, demo_csv, monkeypatch):
    """An export that finishes after a reset neither recreates the batch record nor keeps its page."""
    import ltir.sphere as sphere
    from ltir.pipeline import Engine

    started, release = threading.Event(), threading.Event()
    page = tmp_path / "sphere.html"

    def slow_export(engine, output=None, dataset=None):
        started.set()
        release.wait(30)
        page.write_text("stale", encoding="utf-8")
        return page

    monkeypatch.setattr(sphere, "export_sphere", slow_export)
    engine = Engine(make_config(tmp_path / "ws", sphere_export=True), llm=FakeLLM())
    record = engine.ingest_file(demo_csv)
    assert started.wait(30)
    engine.reset()
    release.set()
    engine._sphere_pool.shutdown(wait=True)
    assert engine.ws.load_batch(record["batch_id"]) is None and not page.exists()


def test_sphere_api(hashed_engine):
    with TestClient(create_app(hashed_engine.config, hashed_engine)) as client:
        page = client.get("/api/sphere")
        assert page.status_code == 200 and "/vendor/plotly.min.js" in page.text and "plotly-graph-div" in page.text
        post = client.post(
            "/api/sphere", json={"dataset": None, "highlight": {"seeds": [], "edges": []}, "palette": {"bg": "#f4f5f9"}, "layers": {"sibling": True}}
        )
        assert post.status_code == 200
        assert client.get("/vendor/plotly.min.js").status_code == 200 and plotly_js_path().is_file()
