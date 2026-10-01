"""3D latent sphere built on lac's prosphera projector (SDD 13 §Sphere)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ltir.sphere import export_sphere, plotly_js_path, sphere_figure
from ltir.web.app import create_app


def test_sphere_figure_layers(hashed_engine):
    fig = sphere_figure(hashed_engine)
    names = [t.name for t in fig.data]
    g = hashed_engine.graph()
    anchors = [n for n in g.of_kind("Attractor")]
    assert "Latent anchors (attractors)" in names and "RELATED_TO (mutual kNN)" in names
    groups = [n for n in names if n and n.startswith("A-")]
    assert len(groups) == len(anchors)  # one legend group per latent anchor
    n_points = sum(len(t.x) for t in fig.data if t.name and t.name.startswith("A-"))
    assert n_points == len(g.of_kind("Pattern"))
    pts = [t for t in fig.data if t.name and t.name.startswith("A-")][0]
    assert max(abs(float(v)) for v in list(pts.x) + list(pts.y) + list(pts.z)) <= 1.0 + 1e-9  # on/inside the unit sphere


def test_sphere_highlight_and_export(hashed_engine, tmp_path):
    qa = hashed_engine.ask("Why is margin lower for phones in the US?")
    fig = sphere_figure(hashed_engine, highlight=qa.highlight)
    names = {t.name for t in fig.data}
    assert {"Retrieval path", "Seed", "Evidence", "Anchors visited"} <= names
    out = export_sphere(hashed_engine, tmp_path / "sphere.html")
    assert "plotly" in out.read_text(encoding="utf-8").lower()


def test_sphere_api(hashed_engine):
    with TestClient(create_app(hashed_engine.config, hashed_engine)) as client:
        page = client.get("/api/sphere")
        assert page.status_code == 200 and "/vendor/plotly.min.js" in page.text and "Latent Insight Sphere" in page.text
        post = client.post("/api/sphere", json={"dataset": None, "highlight": {"seeds": [], "edges": []}})
        assert post.status_code == 200
        assert client.get("/vendor/plotly.min.js").status_code == 200 and plotly_js_path().is_file()
