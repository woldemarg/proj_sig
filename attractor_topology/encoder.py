"""Embedding layer: text embedders + tripartite ``InsightEncoder`` (docs/04_representation.md).

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

import os
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from attractor_topology.config import TopologyConfig
from attractor_topology.models import CANONICAL_VERSION, REPRESENTATION_VERSION, CanonicalInsight, EmbeddingSpec


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


# The one embedder: Qwen3-Embedding-0.6B at a pinned checkpoint; the caller provisions the folder (``model_folder``).
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
TRUNCATE_DIM = 384  # Matryoshka: the first 384 dims, re-normalised -> 3 x 384 = 1152-d insight vectors
QUERY_TASK = "Given a quantitative analysis question, retrieve relevant statistical subgroup patterns"
QUERY_PROMPT = "Instruct: {task}\nQuery: "  # Qwen3-Embedding convention; documents stay un-prefixed
# rows per forward pass: bounds the transient VRAM peak on long documents (Qwen3: 1.7 GB at 16 vs 3.4 GB at 64)
ENCODE_BATCH_SIZE = 16


def model_folder(model_dir: Path) -> Path:
    """The bundled checkpoint: ``model_dir``/Qwen3-Embedding-0.6B."""
    return Path(model_dir) / QWEN_MODEL.split("/")[-1]


class TextEmbedder(Protocol):
    model_id: str
    revision: str  # checkpoint revision, "" when unknown
    dim: int
    truncate_dim: int  # 0 = native width
    query_instruction: str  # full query prefix, "" when the model takes none

    @property
    def compute_dtype(self) -> str:
        """Dtype the model computes in (part of the representation fingerprint)."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 rows with unit L2 norm."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Like ``embed`` for free question text (with the query instruction, if any)."""


class SentenceTransformerEmbedder:
    """Qwen3-Embedding-0.6B from ``model_folder(config.model_dir)``, offline.

    ``TRUNCATE_DIM`` keeps the first Matryoshka dimensions; every output row is re-normalised to unit
    length afterwards (a truncated slice of a unit vector is shorter than 1, and the tripartite
    composition assumes unit blocks). One re-entrant lock serialises the model load and the forward passes
    of every thread (a batch thread and request threads alike).
    """

    model_id = QWEN_MODEL
    revision = QWEN_REVISION
    truncate_dim = TRUNCATE_DIM
    query_instruction = QUERY_PROMPT.format(task=QUERY_TASK)

    def __init__(self, config: TopologyConfig) -> None:
        self.local = model_folder(config.model_dir)
        self.device_pref = config.embedding_device
        self._model = None
        self._lock = threading.RLock()  # re-entrant: ``_encode`` loads the model while holding it
        self._memo: dict[tuple[str, str], np.ndarray] = {}
        self.dim = 0

    @property
    def device(self) -> str:
        return str(self._model.device) if self._model is not None else f"{self.device_pref} (not loaded)"

    @property
    def compute_dtype(self) -> str:
        return str(next(self._load().parameters()).dtype).removeprefix("torch.")

    def _load(self):
        with self._lock:  # one load, however many threads ask first
            if self._model is None:
                if not (self.local / "modules.json").is_file():
                    raise FileNotFoundError(f"embedding model not found in {self.local}: download {QWEN_MODEL} at {QWEN_REVISION} there")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                import torch
                from sentence_transformers import SentenceTransformer

                device = None if self.device_pref == "auto" else self.device_pref
                # the checkpoint dtype on every device (bf16), recorded as spec.compute_dtype
                self._model = SentenceTransformer(str(self.local), device=device, truncate_dim=self.truncate_dim, model_kwargs={"dtype": "auto"})
                if self._model.device.type == "cpu":  # leave half the cores to pandas/numpy work running beside the encoder
                    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 2
                    torch.set_num_threads(max(1, cores // 2))
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


class InsightEncoder:
    """Model-independent tripartite encoder; the contract is ``spec``."""

    def __init__(self, embedder: TextEmbedder, config: TopologyConfig) -> None:
        self.embedder = embedder
        self.block_weights = tuple(float(w) for w in config.block_weights)
        self.emm_component_weight = config.emm_component_weight
        # canonicalisation settings that shape the components (recorded in the spec)
        self.min_component_z = config.thresholds.min_component_z
        self.min_emm_score = config.thresholds.min_emm_score
        self.weight_emm_ref = config.thresholds.weight_emm_ref

    @property
    def spec(self) -> EmbeddingSpec:
        if not self.embedder.dim:
            self.embedder.embed(["warmup"])
        d = int(self.embedder.dim)
        return EmbeddingSpec(
            model_id=self.embedder.model_id,
            model_revision=self.embedder.revision,
            truncate_dim=int(self.embedder.truncate_dim),
            query_instruction=self.embedder.query_instruction,
            block_dim=d,
            dim=3 * d,
            dtype="float32",
            compute_dtype=self.embedder.compute_dtype,
            normalization="l2(block) -> weighted concat -> l2",
            block_weights=self.block_weights,  # type: ignore[arg-type]
            emm_component_weight=self.emm_component_weight,
            min_component_z=self.min_component_z,
            min_emm_score=self.min_emm_score,
            weight_emm_ref=self.weight_emm_ref,
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
