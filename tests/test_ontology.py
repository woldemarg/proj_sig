"""Insight vectors -> latent ontology (SDD 07): assignment, weights, orphans, new concepts, soft merge."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from ltir.config import load_config
from ltir.engines.lac.ontology_engine import repair_extraction
from ltir.ontology import LatentOntology


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def clusters(rng, centers, per, noise=0.05):
    rows, labels = [], []
    for k, c in enumerate(centers):
        for _ in range(per):
            rows.append(unit(c + rng.normal(0, noise, c.shape)))
            labels.append(k)
    return np.stack(rows), labels


@pytest.fixture
def toy():
    rng = np.random.RandomState(0)
    dim = 48
    centers = [unit(rng.normal(size=dim)) for _ in range(3)]
    return rng, dim, centers


def ingest(ont, x, w=None, seq=0, prefix="p"):
    ids = [f"{prefix}{seq}-{i}" for i in range(len(x))]
    with redirect_stdout(io.StringIO()):
        return ont.ingest(x, np.ones(len(x)) if w is None else w, ids, batch_seq=seq, batch_id=f"B{seq}"), ids


def test_cold_start_creates_attractors_with_full_activation_coverage(tmp_path, toy):
    rng, dim, centers = toy
    cfg = load_config()
    ont = LatentOntology(cfg, tmp_path)
    x, labels = clusters(rng, centers, 6)
    up, ids = ingest(ont, x)
    assert up.metrics["new_extracted"] >= 3 and len(ont.attractor_ids) == len(up.new_attractors)
    assert {a["pattern_id"] for a in up.activations} == set(ids)  # every insight ACTIVATES >= 1 attractor
    assert all(a["alignment"] > 0.5 for a in up.activations)
    assert np.allclose(np.linalg.norm(ont.store.embeddings, axis=1), 1.0, atol=1e-4)
    assert int(ont.store.chunk_counts.min()) >= 1
    # members of one planted cluster share their attractor
    by_pattern = {a["pattern_id"]: a["attractor_id"] for a in up.activations}
    for k in range(3):
        members = {by_pattern[i] for i, lab in zip(ids, labels) if lab == k}
        assert len(members) == 1


def test_assignment_orphans_and_new_concepts(tmp_path, toy):
    rng, dim, centers = toy
    cfg = load_config()
    ont = LatentOntology(cfg, tmp_path)
    ingest(ont, clusters(rng, centers[:2], 6)[0])
    n0 = len(ont.attractor_ids)
    # batch 1: near an existing attractor -> assigned (no orphan)
    up, _ = ingest(ont, clusters(rng, centers[:1], 2)[0], seq=1)
    assert up.metrics["orphaned"] == 0 and {a["source"] for a in up.activations} == {"assign"}
    assert len(ont.attractor_ids) == n0
    # batch 2: an unseen phenomenon -> orphans -> OMP -> new attractor
    up, ids = ingest(ont, clusters(rng, centers[2:], 3)[0], seq=2)
    assert up.metrics["orphaned"] == 3 and up.new_attractors
    assert {a["source"] for a in up.activations} <= {"omp", "reroute", "absorbed"}
    assert {a["pattern_id"] for a in up.activations} == set(ids)
    # batch 3: a single orphan uses lac's nearest-attractor fallback
    far = unit(rng.normal(size=dim))
    up, _ = ingest(ont, far[None, :], seq=3)
    assert {a["source"] for a in up.activations} == {"nearest"}


def test_soft_merge_absorbs_redundant_new_atoms(tmp_path, toy):
    rng, dim, centers = toy
    # strict assignment, permissive merge: near-miss orphans are absorbed, not minted
    cfg = load_config(min_assign_threshold=0.97, max_assign_threshold=0.99, soft_merge_low=0.5)
    ont = LatentOntology(cfg, tmp_path)
    ingest(ont, clusters(rng, centers[:1], 4, noise=0.01)[0])
    n0 = len(ont.attractor_ids)
    up, _ = ingest(ont, clusters(rng, centers[:1], 3, noise=0.12)[0], seq=1)
    assert up.metrics["orphaned"] == 3
    assert up.metrics["soft_merged"] >= 1 and len(ont.attractor_ids) == n0
    assert {a["source"] for a in up.activations} == {"absorbed"}


def test_insight_weight_scales_centroid_pull(tmp_path, toy):
    rng, dim, centers = toy
    cfg = load_config()
    probe = unit(centers[0] + 0.5 * rng.normal(size=dim) / np.sqrt(dim))
    moves = []
    for w, sub in ((1.0, "strong"), (0.1, "weak")):
        ont = LatentOntology(cfg, tmp_path / sub)
        ingest(ont, clusters(np.random.RandomState(1), centers, 5)[0])
        before = ont.store.embeddings.copy()
        up, _ = ingest(ont, probe[None, :], w=np.array([w]), seq=1)
        idx = ont.store.concept_ids.index(up.activations[0]["attractor_id"])
        moves.append(float(np.linalg.norm(ont.store.embeddings[idx] - before[idx])))
        assert up.activations[0]["strength"] == pytest.approx(up.activations[0]["alignment"] * w)
    assert moves[0] > 3 * moves[1] > 0  # evidence-magnitude encoding: w scales the EMA step


def test_sign_repair_flips_anti_aligned_atoms(toy):
    rng, dim, centers = toy
    rows = clusters(rng, centers[:1], 3, noise=0.01)[0]
    cents = -centers[0][None, :]  # OMP returned the atom with the "wrong" sign
    acts = [{"chunk_id": i, "concept_id": 0, "weight": 0.9} for i in range(3)]
    fixed, counts, fixed_acts = repair_extraction(cents, acts, rows, load_config())
    assert float(fixed[0] @ centers[0]) > 0.99 and counts.tolist() == [3] and len(fixed_acts) == 3


def test_state_roundtrip(tmp_path, toy):
    rng, dim, centers = toy
    cfg = load_config()
    ont = LatentOntology(cfg, tmp_path)
    ingest(ont, clusters(rng, centers, 4)[0])
    ont.save()
    again = LatentOntology(cfg, tmp_path)
    assert again.attractor_ids == ont.attractor_ids
    assert np.array_equal(again.store.embeddings, ont.store.embeddings)
    assert again.store.next_chunk_id == ont.store.next_chunk_id == 12
