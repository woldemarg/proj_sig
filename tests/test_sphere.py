"""The latent sphere as data (docs/08_interface.md §8.4): the graph service projects; the console draws."""

from __future__ import annotations

import numpy as np
from conftest import ask, make_config, make_engine
from fastapi.testclient import TestClient

from insight_graph_service.server.app import create_app
from insight_graph_service.server.views import project_to_sphere, sphere_points


def sphere(engine):
    return sphere_points(engine.committed(), engine.settings.topology.random_seed)


def test_the_sphere_holds_every_insight_and_its_themes(hashed_engine):
    d = sphere(hashed_engine)
    g = hashed_engine.graph()
    patterns = [p for p in d["points"] if p["kind"] == "Pattern"]
    themes = [p for p in d["points"] if p["kind"] == "Attractor"]
    assert {p["id"] for p in patterns} == {n["id"] for n in g.of_kind("Pattern")} and d["message"] is None
    assert [a["id"] for a in themes] == sorted((n["id"] for n in g.of_kind("Attractor")), key=lambda a: int(a[2:]))
    assert all(np.linalg.norm(p["xyz"]) <= 1.0 + 1e-6 for p in d["points"])  # on or inside the unit ball
    assert {p["cls"] for p in patterns} <= {"up", "down", "cov"} and all(p["hover"] for p in d["points"])
    placed = {p["id"] for p in d["points"]}
    assert all(e["source"] in placed and e["target"] in placed for e in d["edges"])
    assert {"ACTIVATES", "RELATED_TO", "SPECIALIZES"} <= {e["type"] for e in d["edges"]}


def test_an_answer_path_resolves_on_the_sphere(hashed_engine):
    """Every highlighted edge between placed nodes is in the sphere's edges: the console can draw the answer's path."""
    qa = ask(hashed_engine, "Why is margin lower for phones in the US?")
    d = sphere(hashed_engine)
    placed, edges = {p["id"] for p in d["points"]}, {e["id"]: e for e in d["edges"]}
    by_id = {e["id"]: e for e in hashed_engine.graph().edges}
    drawable = [i for i in qa.view["highlight"]["edges"] if by_id[i]["source"] in placed and by_id[i]["target"] in placed]
    assert drawable and all(i in edges for i in drawable) and set(qa.view["highlight"]["seeds"]) <= placed


def test_the_projection_is_deterministic_and_the_api_serves_it(hashed_engine, tmp_path):
    vectors = np.random.RandomState(0).normal(size=(12, 8))
    assert np.array_equal(project_to_sphere(vectors, 42), project_to_sphere(vectors, 42))
    with TestClient(create_app(hashed_engine)) as client:
        assert client.get("/api/sphere").json()["points"]
    cfg = make_config(tmp_path / "empty")
    assert "upload a dataset" in sphere(make_engine(cfg))["message"]
