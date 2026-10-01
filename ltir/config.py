"""Centralised runtime configuration (docs/09_operations.md §9.3; every field: docs/11_reference.md §11.3).

All tunables live here. Values come from environment variables (optionally
loaded from ``sig/.env``); names already used by the vendored lac engine keep their
lac names (``CENTROID_ALPHA``, ``DICTIONARY_K_MIN``, ...) but SIG ships defaults
sized for tens-to-hundreds of insights instead of thousands of text chunks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # sig/ — everything LTIR needs lives below it


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _coerce(raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if isinstance(default, tuple):
        return tuple(float(x) for x in raw.split(","))
    return raw


@dataclass(frozen=True)
class Config:
    # paths
    workspace_dir: Path = PROJECT_ROOT / "workspace"

    # ingestion (docs/02_discovery.md §2.1)
    min_rows: int = 50
    max_upload_mb: int = 200
    # "col:q,col2:q" — derive quantile-band categorical dimensions from numerics
    bin_columns: str = ""
    # "Store,Holiday_Flag" — integer/float-coded columns to treat as categorical dimensions
    categorical_columns: str = ""

    # discovery adapter (docs/02_discovery.md) — EDA defaults preserved
    compute_budget: int = 5000  # step3 compute_budget
    validation_budget: int = 50  # workflow: top-50 candidates to step4b
    min_search_dimensions: int = 3  # step2 min_categories backfill
    eda_random_seed: int = 42  # seeds step4b bootstrap

    # quality / redundancy / weight (docs/03_insights.md)
    min_support_rows: int = 30
    min_effect_z: float = 0.5
    min_emm_score: float = 0.08  # RMS correlation change per metric pair after reliability shrinkage
    min_stability: float = 0.5
    max_p_adjusted: float = 0.05
    min_insight_weight: float = 0.2
    redundancy_jaccard: float = 0.88  # pre-validation near-duplicate threshold (same primary metric and sign)
    max_insights_per_batch: int = 200
    min_component_z: float = 0.5  # secondary shifts kept in the phenomenon
    weight_effect_ref: float = 1.5
    weight_confidence_ref: float = 6.0  # -log10(p_adj) that saturates confidence
    weight_emm_ref: float = 0.08  # same scale as min_emm_score
    weight_exponents: tuple = (0.35, 0.25, 0.20, 0.10, 0.10)  # effect, stability, confidence, support, emm
    weight_floor: float = 0.05

    # embedding (docs/04_representation.md)
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # loaded from MODEL_DIR/<last path segment>
    embedding_truncate_dim: int = 384  # Matryoshka: first 384 dims (re-normalised) keep dim = 3 x 384 = 1152; 0 = native
    embedding_query_instruction: str = "Given a quantitative analysis question, retrieve relevant statistical subgroup patterns"
    model_dir: Path = PROJECT_ROOT / "models"  # holds <embedding_model>/ (sentence-transformers folder)
    embedding_device: str = "auto"  # auto | cpu | cuda
    embedding_backend: str = "sentence-transformers"  # or "hashing" (offline/test)
    block_weights: tuple = (0.45, 0.55, 1.0)  # scope, target, phenomenon
    emm_component_weight: float = 0.5

    # latent ontology (docs/05_latent_anchors.md) — lac names, SIG-sized defaults
    concepts_per_chunk: int = 1  # one dominant phenomenon per insight at extraction
    related_to_peer_count: int = 3
    related_to_min_weight: float = 0.30
    max_concept_count: int = 40
    dictionary_k_min: int = 4
    dictionary_k_step: int = 2
    reconstruction_error_tolerance: float = 0.015
    dead_concept_penalty: float = 0.05
    max_dead_concept_ratio: float = 0.25
    dictionary_batch_size: int = 256
    random_seed: int = 42
    centroid_alpha: float = 0.05
    top_k_assign: int = 2
    mixture_ratio: float = 0.90
    adaptive_percentile: float = 85.0
    # cosine thresholds are embedder-specific (docs/05_latent_anchors.md §5.10): Qwen3 -> 0.75; MiniLM -> 0.55
    min_assign_threshold: float = 0.75
    max_assign_threshold: float = 0.80
    soft_merge_low: float = 0.85
    orphan_buffer_min_factor: int = 3
    min_activation_alignment: float = 0.20  # ACTIVATES edges below this are dropped/rerouted
    # stability guards (docs/05_latent_anchors.md §5.6): adaptive hub threshold, per-attractor damping, trust region
    density_floor: float = 0.25  # hub threshold never below this share
    density_multiple: float = 3.0  # hub = more than this multiple of the uniform share 1/N_attractors
    max_centroid_step: float = 0.10  # max L2 move of one centroid per batch (measured healthy max 0.023)
    warn_orphan_rate: float = 0.50
    warn_min_extraction_yield: float = 0.10
    warn_avg_degree: tuple = (1.0, 8.0)  # mutual k-NN degree envelope
    # OMP input scale s: lac fits codes with lasso alpha=1; scaling inputs by s is alpha/s
    # (OMP directions and relative error are scale-invariant). EMA updates never see s.
    dictionary_input_scale: float = 10.0

    # structural graph (docs/06_graph_and_storage.md §6.1)
    contrast_min_overlap: float = 0.5
    contrast_min_shift: float = 0.5

    # traversal / evidence (docs/07_question_answering.md)
    seed_top_k: int = 3
    seed_min_score: float = 0.25
    seed_relative_min: float = 0.75  # seeds must score >= this fraction of the best seed
    activation_threshold: float = 0.40
    relation_threshold: float = 0.40
    max_latent_hops: int = 1
    structural_hops: int = 1
    traversal_max_depth: int = 5
    max_retrieved: int = 12
    structural_edge_decay: float = 0.85
    traversal_structural_edges: str = "SPECIALIZES,GENERALIZES,CONTRASTS"  # SIBLING stays in the graph
    evidence_max_patterns: int = 10

    # LLM (docs/07_question_answering.md §7.5)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gemma4"
    llm_api_key: str = ""
    llm_timeout_s: float = 120.0
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1200
    llm_reasoning_effort: str = ""
    # OpenRouter provider pinning (as spectr GEMMA_PROVIDER): "dekallm/bf16,parasail/bf16"
    llm_provider_order: str = ""
    llm_app_title: str = "SIG LTIR"  # OpenRouter X-Title attribution header

    # Neo4j (docs/06_graph_and_storage.md §6.6)
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "sigv1"
    neo4j_load_batch_size: int = 5000

    # web
    web_host: str = "127.0.0.1"
    web_port: int = 8765
    sphere_export: bool = True  # write workspace/graph/sphere.html after each READY batch

    @property
    def lattice_edges(self) -> frozenset[str]:
        """Structural edge types the traversal follows (``TRAVERSAL_STRUCTURAL_EDGES``)."""
        return frozenset(t.strip() for t in self.traversal_structural_edges.split(",") if t.strip())

    def public_dict(self) -> dict[str, Any]:
        """Config as JSON-safe dict with secrets masked (for logs/UI)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if "password" in f.name or "api_key" in f.name:
                value = "***" if value else ""
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out


def load_config(env_file: Path | None = None, **overrides: Any) -> Config:
    """Build Config from defaults ← environment (``NAME`` upper-case) ← overrides.

    An empty value clears a string field (``EMBEDDING_QUERY_INSTRUCTION=`` for MiniLM) and
    leaves any other field at its default. ``LTIR_NO_DOTENV=1`` skips ``sig/.env`` (the test
    suite sets it so developer credentials — OpenRouter key, Neo4j — never leak into tests).
    """
    if env_file is not None or not os.environ.get("LTIR_NO_DOTENV"):
        _load_dotenv(env_file or PROJECT_ROOT / ".env")
    values: dict[str, Any] = {}
    for f in fields(Config):
        raw = os.environ.get(f.name.upper())
        if raw is None or (raw == "" and not isinstance(f.default, str)):
            continue
        values[f.name] = _coerce(raw, f.default)
    values.update(overrides)
    return Config(**values)
