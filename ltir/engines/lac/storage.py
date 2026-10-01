"""ConceptStore — evolving attractors (EMA update with concept inertia) and the orphan buffer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ltir.config import Config


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ConceptStore:
    """In-memory concept state; chunk history lives on disk via chunk_journal."""

    def __init__(self) -> None:
        self.concept_ids: list[int] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.chunk_counts: np.ndarray = np.empty(0, dtype=np.int64)
        self.last_updated_batch: np.ndarray = np.empty(0, dtype=np.int64)
        self.created_at: list[str] = []

        self.orphan_embeddings: list[np.ndarray] = []
        self.orphan_chunk_ids: list[int] = []

        self.next_chunk_id: int = 0
        self.next_concept_id: int = 0
        self.dirty_concept_indices: set[int] = set()

    @property
    def is_empty(self) -> bool:
        return len(self.concept_ids) == 0

    def update_concept_centroid(self, concept_idx: int, chunk_vec: np.ndarray, alpha: float, batch_id: int, damping: float = 1.0) -> None:
        # Concept inertia: mature attractors resist drift (alpha decays with sqrt(chunk_count));
        # ``damping`` (<= 1) further slows over-represented attractors (SIG hub guard)
        decayed_alpha = max(0.01, alpha / np.sqrt(self.chunk_counts[concept_idx] + 1)) * damping

        c_old = self.embeddings[concept_idx].astype(np.float64)
        chunk = chunk_vec.astype(np.float64)
        blended = (1.0 - decayed_alpha) * c_old + decayed_alpha * chunk
        norm = np.linalg.norm(blended)
        if norm > 1e-12:
            self.embeddings[concept_idx] = (blended / norm).astype(np.float32)

        self.chunk_counts[concept_idx] += 1
        self.last_updated_batch[concept_idx] = batch_id
        self.dirty_concept_indices.add(concept_idx)

    def append_concepts(
        self,
        centroids: np.ndarray,
        *,
        chunk_counts: list[int] | np.ndarray,
        batch_id: int,
    ) -> list[int]:
        """Add new attractors; returns their ids."""
        if len(centroids) == 0:
            return []

        n_new = len(centroids)
        centroids = np.asarray(centroids, dtype=np.float32)
        if self.embeddings.size == 0:
            self.embeddings = centroids
        else:
            if centroids.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Centroid dimension mismatch")
            self.embeddings = np.vstack([self.embeddings, centroids])

        new_ids = list(range(self.next_concept_id, self.next_concept_id + n_new))
        self.next_concept_id += n_new
        self.concept_ids.extend(new_ids)

        self.chunk_counts = np.concatenate([self.chunk_counts, np.asarray(chunk_counts, dtype=np.int64)])
        self.last_updated_batch = np.concatenate([self.last_updated_batch, np.full(n_new, batch_id, dtype=np.int64)])
        self.created_at.extend([_utc_now_iso()] * n_new)

        start_idx = len(self.concept_ids) - n_new
        self.dirty_concept_indices.update(range(start_idx, len(self.concept_ids)))
        return new_ids

    def push_orphans(self, orphan_embeddings: np.ndarray, orphan_chunk_ids: list[int] | np.ndarray) -> None:
        ids = list(orphan_chunk_ids)
        for i, emb in enumerate(orphan_embeddings):
            self.orphan_embeddings.append(np.asarray(emb, dtype=np.float32))
            self.orphan_chunk_ids.append(int(ids[i]))

    def get_orphan_buffer(self) -> tuple[np.ndarray, list[int]]:
        if not self.orphan_embeddings:
            return np.empty((0, 0), dtype=np.float32), []
        return np.stack(self.orphan_embeddings), list(self.orphan_chunk_ids)

    def should_extract_orphans(self, config: Config, batch_orphan_count: int) -> bool:
        """Full buffer at threshold, or partial flush so orphans get ACTIVATES this batch."""
        size = len(self.orphan_embeddings)
        if size == 0:
            return False
        if size >= config.dictionary_k_min * config.orphan_buffer_min_factor:
            return True
        return batch_orphan_count > 0 and size >= 2

    def clear_orphan_buffer(self) -> None:
        self.orphan_embeddings.clear()
        self.orphan_chunk_ids.clear()

    def clear_dirty(self) -> None:
        self.dirty_concept_indices.clear()

    def save(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            state_dir / "concepts.npz",
            embeddings=self.embeddings,
            chunk_counts=self.chunk_counts,
            last_updated_batch=self.last_updated_batch,
        )

        orphan_path = state_dir / "orphan_buffer.npz"
        if self.orphan_embeddings:
            np.savez_compressed(
                orphan_path,
                embeddings=np.stack(self.orphan_embeddings),
                chunk_ids=np.array(self.orphan_chunk_ids, dtype=np.int64),
            )
        else:
            orphan_path.unlink(missing_ok=True)

        state = {
            "concept_ids": self.concept_ids,
            "created_at": self.created_at,
            "next_chunk_id": self.next_chunk_id,
            "next_concept_id": self.next_concept_id,
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings.size else 0,
        }
        (state_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, state_dir: Path) -> ConceptStore:
        store = cls()
        state_path = state_dir / "state.json"
        if not state_path.exists():
            return store

        state = json.loads(state_path.read_text(encoding="utf-8"))
        store.concept_ids = [int(x) for x in state["concept_ids"]]
        store.created_at = list(state["created_at"])
        store.next_chunk_id = int(state["next_chunk_id"])
        store.next_concept_id = int(state["next_concept_id"])

        concepts_path = state_dir / "concepts.npz"
        if concepts_path.exists():
            with np.load(concepts_path) as data:
                store.embeddings = data["embeddings"].astype(np.float32)
                store.chunk_counts = data["chunk_counts"].astype(np.int64)
                store.last_updated_batch = data["last_updated_batch"].astype(np.int64)

        orphan_path = state_dir / "orphan_buffer.npz"
        if orphan_path.exists():
            with np.load(orphan_path) as data:
                store.orphan_embeddings = list(data["embeddings"])
                store.orphan_chunk_ids = [int(i) for i in data["chunk_ids"]]

        return store
