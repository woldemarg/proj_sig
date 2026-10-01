"""Canonical insight -> embedding contract (SDD 05, SDD 06)."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from ltir.canonical import canonicalize, describe_scope, format_p, format_value
from ltir.config import Config, load_config
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
    assert c.scope == "category = laptops; region = EU"  # embedding input: machine form, unchanged by canon-3
    assert c.scope_sentence == "category is laptops and region is EU"
    assert "margin: strong decrease, -2.30 sd" in c.phenomenon and "discount: moderate increase, +1.20 sd" in c.phenomenon
    assert "delivery" not in c.phenomenon  # |z| < MIN_COMPONENT_Z
    assert "region" not in c.phenomenon and "margin" not in c.scope  # structure vs statistics never mixed
    assert c.components == (("margin", -2.3), ("discount", 1.2))
    assert "cash" in c.confounders and "5,000" in c.support and "adjusted p < 0.001" in c.support
    doc = c.document()
    assert doc.isascii()
    lines = doc.splitlines()
    assert lines[0] == "### Subgroup finding P-x"
    keys = ["* Scope", "* Target metric", "* Observed shift", "* Metric relationships", "* Confounders", "* Validation"]
    assert [line.split(":")[0] for line in lines[1:]] == keys


def test_embedding_labels_carry_no_numbers():
    cfg = load_config()
    c = canonicalize(insight({"region": "EU"}, [("discount", 0.2)], emm=0.3, ptype="covariance"), cfg)
    assert all(not any(ch.isdigit() for ch in label) for label, _ in c.components)  # magnitudes live in the coefficients


def test_text_number_rules():
    assert format_value(0.0521) == "0.0521" and format_value(19.187) == "19.19" and format_value(206500.4) == "206,500"
    assert format_p(0.0) == "< 0.001" and format_p(0.004) == "< 0.01" and format_p(0.03) == "< 0.05" and format_p(1.0) == "1.00"
    conds = [Condition("a", "x"), Condition("b", "y"), Condition("c", "z")]
    assert describe_scope(conds[:1]) == "a is x" and describe_scope(conds) == "a is x, b is y, and c is z"


def test_covariance_canonical_component():
    cfg = load_config()
    c = canonicalize(insight({"region": "EU"}, [("discount", 0.2)], emm=0.3, ptype="covariance"), cfg)
    assert c.phenomenon.startswith("correlation between discount and margin weakens from -0.60 overall to +0.10 in the subgroup")
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
    d = replace(a, truncate_dim=384)  # Matryoshka width
    e = replace(a, query_instruction="Instruct: x\nQuery: ")
    assert len({a.fingerprint, b.fingerprint, c.fingerprint, d.fingerprint, e.fingerprint}) == 5


class RecordingEmbedder(HashingEmbedder):
    """Hashing vectors; records which texts went through ``embed`` vs ``embed_queries``."""

    query_instruction = "Instruct: test\nQuery: "

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, texts):
        self.calls.append(("embed", list(texts)))
        return super().embed(texts)

    def embed_queries(self, texts):
        self.calls.append(("query", list(texts)))
        return super().embed(texts)


def test_env_file_switches_to_the_minilm_block(tmp_path, monkeypatch):
    """The documented MiniLM block applies verbatim: an empty value clears the instruction."""
    monkeypatch.setattr(os, "environ", dict(os.environ))  # the dotenv loader writes into os.environ
    env = tmp_path / ".env"
    env.write_text(
        "EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2\nEMBEDDING_TRUNCATE_DIM=0\nEMBEDDING_QUERY_INSTRUCTION=\nMIN_ASSIGN_THRESHOLD=0.55\nVALIDATION_BUDGET=\n",
        encoding="utf-8",
    )
    cfg = load_config(env)
    assert (cfg.embedding_model, cfg.embedding_truncate_dim, cfg.embedding_query_instruction) == ("paraphrase-multilingual-MiniLM-L12-v2", 0, "")
    assert cfg.min_assign_threshold == 0.55
    assert cfg.validation_budget == Config().validation_budget  # an empty non-string value keeps the default


def test_query_instruction_only_reaches_free_question_text():
    cfg = load_config()
    emb = RecordingEmbedder()
    enc = InsightEncoder(emb, cfg)
    question = "Why is margin lower for phones in the US?"
    enc.encode_query("category = phones; region = US", "margin", (("margin", -2.0),), question)
    assert all(kind == "embed" for kind, _ in emb.calls)  # structured blocks and labels: document vocabulary, no prefix
    emb.calls.clear()
    enc.encode_query("", "", (), question)
    assert [kind for kind, _ in emb.calls] == ["query", "query", "query"] and all(texts == [question] for _, texts in emb.calls)
