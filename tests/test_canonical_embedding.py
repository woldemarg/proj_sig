"""Canonical insight -> embedding contract (SDD 05, SDD 06)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ltir.canonical import canonicalize
from ltir.config import load_config
from ltir.encoder import HashingEmbedder, InsightEncoder, SentenceTransformerEmbedder
from ltir.models import CANONICAL_VERSION, REPRESENTATION_VERSION, Condition, Insight, Shift


def insight(scope, shifts, emm=0.05, ptype="shift", cov=None):
    conds = tuple(sorted(Condition(k, v) for k, v in scope.items()))
    sh = tuple(Shift(m, z, 1.0 + z, 1.0, 1.0) for m, z in shifts)
    return Insight(
        id="P-x",
        dataset_id="d",
        batch_id="b",
        conditions=conds,
        expression="e",
        target=sh[0].metric,
        shifts=sh,
        support=200,
        support_fraction=0.04,
        baseline=1.0,
        local=1.0 + sh[0].robust_z,
        effect_size=sh[0].robust_z,
        sd_score=1.0,
        sd_raw_score=1.1,
        emm_score=emm,
        volume_utility=0.2,
        stability=0.9,
        integrated_index=1.0,
        p_value=1e-9,
        p_adjusted=1e-7,
        drivers=("[payment] heavily skewed to 'cash' (JS: 0.20)",),
        row_hash="h",
        phenomenon_type=ptype,
        covariance=cov or {"pair": ["margin", "discount"], "local_corr": 0.1, "global_corr": -0.6, "delta": 0.7},
    )


def test_canonical_sections_are_separate():
    cfg = load_config()
    c = canonicalize(insight({"region": "EU", "category": "laptops"}, [("margin", -2.3), ("discount", 1.2), ("delivery_days", 0.1)]), cfg, 5000)
    assert c.version == CANONICAL_VERSION
    assert c.target == "margin"
    assert c.scope == "category = laptops; region = EU"
    assert "margin strong decrease" in c.phenomenon and "discount moderate increase" in c.phenomenon
    assert "delivery" not in c.phenomenon  # |z| < MIN_COMPONENT_Z
    assert "region" not in c.phenomenon and "margin" not in c.scope  # structure vs statistics never mixed
    assert c.components == (("margin", -2.3), ("discount", 1.2))
    assert "cash" in c.confounders and "5,000" in c.support
    doc = c.document().splitlines()
    assert [line.split(":")[0] for line in doc] == ["TARGET", "SCOPE", "PHENOMENON", "COVARIANCE", "CONFOUNDERS", "SUPPORT"]


def test_covariance_canonical_component():
    cfg = load_config()
    c = canonicalize(insight({"region": "EU"}, [("discount", 0.2)], emm=0.3, ptype="covariance"), cfg)
    assert c.phenomenon.startswith("correlation between discount and margin weakens")
    label, coef = c.components[-1]
    assert label == "correlation between discount and margin" and coef < 0


@pytest.fixture(params=["hashing", pytest.param("model", marks=pytest.mark.model)])
def encoder(request):
    cfg = load_config()
    emb = HashingEmbedder() if request.param == "hashing" else SentenceTransformerEmbedder(cfg)
    return InsightEncoder(emb, cfg)


def test_embedding_contract(encoder):
    cfg = load_config()
    cans = [
        canonicalize(insight({"region": "EU", "category": "laptops"}, [("margin", 2.0)]), cfg),
        canonicalize(insight({"region": "EU", "category": "laptops"}, [("margin", -2.0)]), cfg),
        canonicalize(insight({"region": "US", "category": "phones"}, [("discount", 2.2), ("margin", -1.1)]), cfg),
        canonicalize(insight({"region": "APAC", "category": "tablets"}, [("discount", 2.1), ("margin", -1.0)]), cfg),
    ]
    out = encoder.encode(cans)
    spec = encoder.spec
    d = spec.block_dim
    assert out["vector"].shape == (4, 3 * d) and spec.dim == 3 * d
    assert out["vector"].dtype == np.float32 and spec.dtype == "float32"
    for key in ("vector", "scope", "target", "phenomenon"):
        assert np.allclose(np.linalg.norm(out[key], axis=1), 1.0, atol=1e-5)
    assert spec.representation_version == REPRESENTATION_VERSION and spec.canonical_version == CANONICAL_VERSION
    assert spec.model_id and len(spec.fingerprint) == 10
    # stable representation
    assert np.array_equal(encoder.encode(cans)["vector"], out["vector"])
    v = out["vector"]
    # direction lives in the phenomenon block: same scope, opposite shift -> dissimilar
    assert float(v[0] @ v[1]) < 0.0
    # same phenomenon in disjoint scopes -> similar (the transversal premise)
    assert float(v[2] @ v[3]) > float(v[2] @ v[0]) + 0.3
    assert float(v[2] @ v[3]) > 0.7


def test_fingerprint_changes_with_representation_choices():
    cfg = load_config()
    a = InsightEncoder(HashingEmbedder(), cfg).spec
    b = InsightEncoder(HashingEmbedder(), replace(cfg, block_weights=(1.0, 1.0, 1.0))).spec
    c = InsightEncoder(HashingEmbedder(dim=128), cfg).spec
    assert len({a.fingerprint, b.fingerprint, c.fingerprint}) == 3
