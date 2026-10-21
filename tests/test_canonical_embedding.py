"""Canonical insight -> embedding contract (docs/04_representation.md)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from conftest import toy_insight
from doubles import HashingEmbedder

from attractor_topology.canonical import canonicalize
from attractor_topology.encoder import InsightEncoder, SentenceTransformerEmbedder, l2_normalize
from attractor_topology.models import CANONICAL_VERSION, REPRESENTATION_VERSION
from insight_contracts import Condition, Shift
from insight_contracts.text import describe_scope, format_p, format_value
from insight_graph_service.core.settings import load_env_file, load_settings
from subgroup_miner.config import MinerConfig


def insight(scope, shifts, emm=0.05, ptype="shift", cov=None):
    return toy_insight(
        sorted(scope.items()),
        [Shift(m, z, 1.0 + z, 1.0, 1.0) for m, z in shifts],
        support=200,
        support_fraction=0.04,
        sd_raw_score=1.1,
        emm_score=emm,
        p_adjusted=1e-7,
        drivers=("[payment] heavily skewed to 'cash' (JS: 0.20)",),
        row_hash="h",
        phenomenon_type=ptype,
        covariance=cov or {"pair": ["margin", "discount"], "local_corr": 0.1, "global_corr": -0.6, "delta": 0.7},
    )


def test_canonical_sections_are_separate():
    cfg = load_settings().topology
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
    cfg = load_settings().topology
    c = canonicalize(insight({"region": "EU"}, [("discount", 0.2)], emm=0.3, ptype="covariance"), cfg)
    assert all(not any(ch.isdigit() for ch in label) for label, _ in c.components)  # magnitudes live in the coefficients


def test_text_number_rules():
    assert format_value(0.0521) == "0.0521" and format_value(19.187) == "19.19" and format_value(206500.4) == "206,500"
    assert format_p(0.0) == "< 0.001" and format_p(0.004) == "< 0.01" and format_p(0.03) == "< 0.05" and format_p(1.0) == "1.00"
    conds = [Condition("a", "x"), Condition("b", "y"), Condition("c", "z")]
    assert describe_scope(conds[:1]) == "a is x" and describe_scope(conds) == "a is x, b is y, and c is z"


def test_covariance_canonical_component():
    cfg = load_settings().topology
    c = canonicalize(insight({"region": "EU"}, [("discount", 0.2)], emm=0.3, ptype="covariance"), cfg)
    assert c.phenomenon.startswith("correlation between discount and margin weakens from -0.60 overall to +0.10 in the subgroup")
    label, coef = c.components[-1]
    assert label == "correlation between discount and margin" and coef < 0
    # a covariance insight cites only its correlation change: an unvalidated shift is no component, no phrase
    retyped = replace(insight({"region": "EU"}, [("margin", -1.4)], emm=0.3, ptype="covariance"), stability=None, p_value=None, p_adjusted=None)
    r = canonicalize(retyped, cfg, 5000)
    assert [label for label, _ in r.components] == ["correlation between discount and margin"]
    assert "-1.40 sd" not in r.phenomenon and r.phenomenon.endswith("no validated median shift")
    assert "no median test" in r.support and "stability" not in r.support


@pytest.fixture(params=["hashing", pytest.param("model", marks=pytest.mark.model)])
def encoder(request):
    cfg = load_settings().topology
    emb = HashingEmbedder() if request.param == "hashing" else SentenceTransformerEmbedder(cfg)
    return InsightEncoder(emb, cfg)


@pytest.mark.model
def test_the_model_computes_in_float32_on_the_cpu():
    """bf16 has no fast CPU kernels: the CPU runs the model in float32, and the spec records it."""
    emb = SentenceTransformerEmbedder(replace(load_settings().topology, embedding_device="cpu"))
    assert emb.compute_dtype == "float32"
    assert InsightEncoder(emb, load_settings().topology).spec.compute_dtype == "float32"


def test_embedding_contract(encoder):
    cfg = load_settings().topology
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
    cfg = load_settings().topology
    a = InsightEncoder(HashingEmbedder(), cfg).spec
    b = InsightEncoder(HashingEmbedder(), replace(cfg, block_weights=(1.0, 1.0, 1.0))).spec
    c = InsightEncoder(HashingEmbedder(dim=128), cfg).spec
    d = replace(a, truncate_dim=384)  # Matryoshka width
    e = replace(a, query_instruction="Instruct: x\nQuery: ")
    variants = [a, b, c, d, e]
    # settings that shape the components, the compute dtype and the checkpoint revision
    for field, value in (("min_component_z", 1.0), ("min_emm_score", 0.2), ("weight_emm_ref", 0.2)):
        variants.append(InsightEncoder(HashingEmbedder(), replace(cfg, thresholds=replace(cfg.thresholds, **{field: value}))).spec)
    variants += [replace(a, compute_dtype="bfloat16"), replace(a, model_revision="97b0c614")]
    assert len({v.fingerprint for v in variants}) == len(variants)


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


def test_env_file_values_apply(tmp_path, monkeypatch):
    """A ``.env`` value overrides its default; an empty non-string value keeps the default."""
    monkeypatch.setattr(os, "environ", dict(os.environ))  # the dotenv loader writes into os.environ
    env = tmp_path / ".env"
    env.write_text("MIN_ASSIGN_THRESHOLD=0.55\nVALIDATION_BUDGET=\n", encoding="utf-8")
    load_env_file(env)
    cfg = load_settings()
    assert cfg.topology.min_assign_threshold == 0.55
    assert cfg.miner.validation_budget == MinerConfig().validation_budget  # an empty non-string value keeps the default


def test_query_instruction_only_reaches_free_question_text():
    cfg = load_settings().topology
    emb = RecordingEmbedder()
    enc = InsightEncoder(emb, cfg)
    question = "Why is margin lower for phones in the US?"
    enc.encode_query("category = phones; region = US", "margin", (("margin", -2.0),), question)
    assert all(kind == "embed" for kind, _ in emb.calls)  # structured blocks and labels: document vocabulary, no prefix
    emb.calls.clear()
    enc.encode_query("", "", (), question)
    assert [kind for kind, _ in emb.calls] == ["query", "query", "query"] and all(texts == [question] for _, texts in emb.calls)


class _SlowModel:
    """A stand-in for the loaded SentenceTransformer: 0.2 s per ``encode`` call, one row of text length per text."""

    device = "cpu"

    def __init__(self):
        self.calls: list[int] = []

    def parameters(self):
        return iter([np.zeros(1, dtype=np.float32)])

    def encode(self, texts, **_):
        self.calls.append(len(texts))
        time.sleep(0.2)
        return np.array([[len(t), 1.0, 0.0] for t in texts], dtype=np.float32)


def test_a_question_waits_for_one_chunk_of_a_batch_not_all_of_it():
    """The forward passes hold the model lock a chunk at a time: a question asked while a batch embeds 160 texts waits
    for one chunk, the loaded model's dtype for none, and every text gets the vector it gets alone."""
    emb = SentenceTransformerEmbedder(load_settings().topology)
    emb._model, emb.dim = _SlowModel(), 3
    texts = [f"insight {i:03d}" for i in range(160)]
    batch = threading.Thread(target=emb.embed, args=(texts,))
    batch.start()
    time.sleep(0.1)  # inside the batch's first chunk
    start = time.perf_counter()
    assert emb.compute_dtype == "float32" and time.perf_counter() - start < 0.05
    query = emb.embed_queries(["Чому margin нижчий?"])
    waited = time.perf_counter() - start
    batch.join()
    assert waited < 1.0, waited  # the whole batch is 10 chunks, 2 s
    assert max(emb._model.calls) == 16 and sum(emb._model.calls) == 161
    assert np.allclose(emb.embed(texts[:3]), l2_normalize(np.array([[len(t), 1.0, 0.0] for t in texts[:3]])))
    assert np.allclose(query, l2_normalize(np.array([[len("Чому margin нижчий?"), 1.0, 0.0]])))
