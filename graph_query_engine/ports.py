"""What the query engine needs from an embedding model, as structural types: any object with these methods works
(``attractor_topology.InsightEncoder`` and its embedder, a fake in tests). The engine never imports
the representation package."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit rows for document-side text (literals, labels)."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Unit rows for free question text."""


class QueryEncoder(Protocol):
    embedder: Embedder

    def encode_query(self, scope_text: str, target_text: str, components: tuple[tuple[str, float], ...], text: str) -> np.ndarray:
        """A parsed question in the insight vector frame (the same composition as the stored insight vectors)."""
