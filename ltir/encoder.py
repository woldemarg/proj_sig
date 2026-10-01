"""Embedding layer: text embedders + tripartite ``InsightEncoder`` (SDD 06).

Composition (all blocks L2-normalised first):

    s = E(scope text)                                  scope block
    t = E(target text)                                 target block
    p = normalize( sum_k c_k * E(label_k) )            phenomenon block
        c_k = signed robust z of metric k (or signed covariance component)
    v = normalize( [w_s * s ; w_t * t ; w_p * p] )     insight vector, dim = 3 * d

Direction lives in the sign of c_k, so "margin up" and "margin down" point in
opposite phenomenon directions even though their sentences embed alike.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from ltir.config import Config
from ltir.models import CANONICAL_VERSION, REPRESENTATION_VERSION, CanonicalInsight, EmbeddingSpec


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


QUERY_PROMPT = "Instruct: {task}\nQuery: "  # Qwen3-Embedding / e5-instruct convention; documents stay un-prefixed
# rows per forward pass: bounds the transient VRAM peak on long documents (Qwen3: 1.7 GB at 16 vs 3.4 GB at 64)
ENCODE_BATCH_SIZE = 16


def model_folder(config: Config) -> Path:
    """Bundled copy of EMBEDDING_MODEL: MODEL_DIR/<last path segment> ('Qwen/Qwen3-Embedding-0.6B' -> models/Qwen3-Embedding-0.6B)."""
    return Path(config.model_dir) / config.embedding_model.split("/")[-1]


class TextEmbedder(Protocol):
    model_id: str
    dim: int
    truncate_dim: int  # 0 = native width
    query_instruction: str  # full query prefix, "" when the model takes none

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 rows with unit L2 norm."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Like ``embed`` for free question text (with the query instruction, if any)."""


class SentenceTransformerEmbedder:
    """Local sentence-transformers model from ``model_folder(config)`` (offline).

    ``EMBEDDING_TRUNCATE_DIM`` keeps the first d Matryoshka dimensions; every output row is
    re-normalised to unit length afterwards (a truncated slice of a unit vector is shorter
    than 1, and the tripartite composition assumes unit blocks).
    """

    def __init__(self, config: Config) -> None:
        self.model_name = config.embedding_model
        self.model_dir = Path(config.model_dir)
        self.local = model_folder(config)
        self.device_pref = config.embedding_device
        self.truncate_dim = int(config.embedding_truncate_dim)
        task = config.embedding_query_instruction.strip()
        self.query_instruction = QUERY_PROMPT.format(task=task) if task else ""
        self._model = None
        self._lock = threading.Lock()
        self._memo: dict[tuple[str, str], np.ndarray] = {}
        self.model_id = self.model_name if "/" in self.model_name else f"sentence-transformers/{self.model_name}"
        self.source = ""
        self.dim = 0

    @property
    def device(self) -> str:
        return str(self._model.device) if self._model is not None else f"{self.device_pref} (not loaded)"

    def _load(self):
        if self._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            from sentence_transformers import SentenceTransformer

            device = None if self.device_pref == "auto" else self.device_pref
            on_cuda = device.startswith("cuda") if device else torch.cuda.is_available()
            kwargs = {"truncate_dim": self.truncate_dim or None}
            if on_cuda:  # checkpoint dtype (bf16 for Qwen3, fp32 for MiniLM); CPU stays fp32
                kwargs["model_kwargs"] = {"dtype": "auto"}
            if (self.local / "modules.json").is_file():  # bundled copy in sig/models (offline)
                self.source = str(self.local)
                self._model = SentenceTransformer(str(self.local), device=device, **kwargs)
            else:  # fallback: resolve by name, caching the download under MODEL_DIR
                self.source = f"hub:{self.model_name}"
                self._model = SentenceTransformer(self.model_name, cache_folder=str(self.model_dir), device=device, **kwargs)
            self.dim = int(self._model.get_embedding_dimension())
        return self._model

    def _encode(self, texts: list[str], prompt: str) -> np.ndarray:
        """Memoised, re-normalised embeddings of ``texts`` with an optional prompt prefix."""
        with self._lock:
            model = self._load()
            missing = sorted({t for t in texts if (prompt, t) not in self._memo})
            if missing:
                vecs = model.encode(missing, prompt=prompt or None, batch_size=ENCODE_BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
                for text, vec in zip(missing, l2_normalize(np.asarray(vecs, dtype=np.float32))):
                    self._memo[(prompt, text)] = vec
            if not texts:
                return np.empty((0, self.dim), dtype=np.float32)
            return np.stack([self._memo[(prompt, t)] for t in texts]).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_instruction)


class HashingEmbedder:
    """Deterministic offline embedder (word + char-trigram hashing). Tests / no-model fallback."""

    truncate_dim = 0
    query_instruction = ""

    def __init__(self, dim: int = 256) -> None:
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


def make_text_embedder(config: Config) -> TextEmbedder:
    if config.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(config)


class InsightEncoder:
    """Model-independent tripartite encoder; the contract is ``spec``."""

    def __init__(self, embedder: TextEmbedder, config: Config) -> None:
        self.embedder = embedder
        self.block_weights = tuple(float(w) for w in config.block_weights)
        self.emm_component_weight = config.emm_component_weight

    @property
    def spec(self) -> EmbeddingSpec:
        if not self.embedder.dim:
            self.embedder.embed(["warmup"])
        d = int(self.embedder.dim)
        return EmbeddingSpec(
            model_id=self.embedder.model_id,
            truncate_dim=int(self.embedder.truncate_dim),
            query_instruction=self.embedder.query_instruction,
            block_dim=d,
            dim=3 * d,
            dtype="float32",
            normalization="l2(block) -> weighted concat -> l2",
            block_weights=self.block_weights,  # type: ignore[arg-type]
            emm_component_weight=self.emm_component_weight,
            canonical_version=CANONICAL_VERSION,
            representation_version=REPRESENTATION_VERSION,
        )

    def _phenomenon(self, components: list[tuple[tuple[str, float], ...]], fallback: list[str], *, query: bool = False) -> np.ndarray:
        """normalize(sum coef * E(label)); labels are embedded exactly as for documents (the signed
        composition needs identical label vectors on both sides). An empty sum falls back to the
        text — the question text for queries, embedded with the query instruction."""
        labels = sorted({label for comp in components for label, _ in comp})
        table = dict(zip(labels, self.embedder.embed(labels))) if labels else {}
        embed_fallback = self.embedder.embed_queries if query else self.embedder.embed
        rows = []
        for comp, text in zip(components, fallback):
            vec = np.zeros(self.embedder.dim or 1, dtype=np.float64)
            for label, coef in comp:
                vec = vec + coef * table[label].astype(np.float64)
            if np.linalg.norm(vec) < 1e-9:
                vec = embed_fallback([text])[0].astype(np.float64)
            rows.append(vec)
        return l2_normalize(np.stack(rows))

    def compose(self, scope: np.ndarray, target: np.ndarray, phenomenon: np.ndarray) -> np.ndarray:
        ws, wt, wp = self.block_weights
        return l2_normalize(np.hstack([ws * scope, wt * target, wp * phenomenon]))

    def encode(self, canonicals: list[CanonicalInsight]) -> dict[str, np.ndarray]:
        """-> {'vector': (n, 3d), 'scope': (n, d), 'target': (n, d), 'phenomenon': (n, d)} float32, n >= 1."""
        scope = self.embedder.embed([c.scope for c in canonicals])
        target = self.embedder.embed([c.target for c in canonicals])
        phen = self._phenomenon([c.components for c in canonicals], [c.phenomenon for c in canonicals])
        return {"vector": self.compose(scope, target, phen), "scope": scope, "target": target, "phenomenon": phen}

    def encode_query(self, scope_text: str, target_text: str, components: tuple[tuple[str, float], ...], text: str) -> np.ndarray:
        """Project a parsed question into the insight space (same composition).

        Recognised scope/target strings and component labels are the documents' own
        vocabulary and are embedded without the query instruction; only the free question
        text (standing in for an unrecognised block) gets it. This keeps the tripartite
        geometry exact and forgoes the instruction gain on structured seeds by design.
        """
        scope = self.embedder.embed([scope_text]) if scope_text else self.embedder.embed_queries([text])
        target = self.embedder.embed([target_text]) if target_text else self.embedder.embed_queries([text])
        phen = self._phenomenon([components], [text], query=True)
        return self.compose(scope, target, phen)[0]
