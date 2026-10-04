"""Test doubles: deterministic stand-ins for the embedding model."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from attractor_topology.encoder import l2_normalize


class HashingEmbedder:
    """Deterministic offline embedder (word + char-trigram hashing): fast tests without the model."""

    truncate_dim = 0
    query_instruction = ""
    revision = ""
    compute_dtype = "float64"

    def __init__(self, dim: int = 384) -> None:  # the production block width (TRUNCATE_DIM)
        self.dim = dim
        self.model_id = f"hashing-ngram-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        words = re.findall(r"[a-z0-9]+", text.lower())
        grams = words + [f"#{w[i : i + 3]}" for w in words for i in range(max(1, len(w) - 2))]
        for gram in grams:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)
