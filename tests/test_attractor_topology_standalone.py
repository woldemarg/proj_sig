"""attractor_topology on its own: insights in, vectors and living attractors out (attractor_topology/README.md)."""

from __future__ import annotations

import numpy as np
from conftest import toy_insight
from doubles import HashingEmbedder

from insight_contracts import Shift


def test_insights_become_vectors_and_attractors(tmp_path):
    from attractor_topology import InsightEncoder, LatentOntology, TopologyConfig, canonicalize

    config = TopologyConfig()
    scopes = [("region", r, "category", c) for r in ("EU", "US", "APAC") for c in ("phones", "laptops")]
    insights = [
        toy_insight([(a, x), (b, y)], [Shift("margin", -2.0 if y == "phones" else 1.5, 10, 20, 3)], id=f"P-{x}-{y}", weight=0.6)
        for a, x, b, y in scopes
    ]
    encoded = InsightEncoder(HashingEmbedder(), config).encode([canonicalize(i, config) for i in insights])
    assert encoded["vector"].shape == (6, 1152)
    ontology = LatentOntology(config, tmp_path)
    update = ontology.ingest(encoded["vector"], np.array([i.weight for i in insights]), [i.id for i in insights], batch_seq=0, batch_id="b1")
    attractors = ontology.attractors()
    assert attractors and {a["pattern_id"] for a in update.activations} == {i.id for i in insights}
    assert all(abs(float(np.linalg.norm(a.centroid)) - 1.0) < 1e-4 and a.mass >= 1 for a in attractors)
    ontology.save()
    assert [a.id for a in LatentOntology(config, tmp_path).attractors()] == [a.id for a in attractors]  # the anchors persist
