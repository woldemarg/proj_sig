"""Model provisioning (insight_graph_service.core.model_store): only a verified checkpoint reaches the service."""

from __future__ import annotations

import hashlib

import pytest

from insight_graph_service.core import model_store


def _fake_manifest(monkeypatch, files: dict[str, bytes]) -> None:
    monkeypatch.setattr(model_store, "MANIFEST", {n: (len(b), hashlib.sha256(b).hexdigest()) for n, b in files.items()})


def test_a_verified_seed_is_copied_and_a_corrupt_copy_is_refused(tmp_path, monkeypatch):
    files = {"modules.json": b"[]", "1_Pooling/config.json": b"{}", "model.safetensors": b"weights"}
    _fake_manifest(monkeypatch, files)
    seed, target = tmp_path / "seed", tmp_path / "target"
    for name, data in files.items():
        path = model_store.model_folder(seed) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert model_store.provision(target, seed).startswith("copied from")
    assert model_store.provision(target, seed) == "present" and model_store.problems(model_store.model_folder(target)) == []
    weights = model_store.model_folder(target) / "model.safetensors"
    weights.write_bytes(b"weightz")  # same size, other content
    assert model_store.problems(model_store.model_folder(target)) == ["model.safetensors: sha256 mismatch"]
    assert model_store.problems(model_store.model_folder(target), hashes=False) == []  # the service's cheap pre-flight
    assert model_store.provision(target, seed).startswith("copied from") and weights.read_bytes() == b"weights"  # repaired from the seed


@pytest.mark.model
def test_the_bundled_checkpoint_matches_the_pinned_manifest():
    from insight_graph_service.core.settings import load_settings

    assert model_store.problems(model_store.model_folder(load_settings().topology.model_dir)) == []
