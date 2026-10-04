"""Attractor topology settings: the representation (docs/04_representation.md) and the living attractor space
(docs/05_latent_anchors.md; lac names, SIG-sized defaults). A host may fill the fields from the environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from insight_contracts import PhenomenonThresholds


@dataclass(frozen=True)
class TopologyConfig:
    thresholds: PhenomenonThresholds = PhenomenonThresholds()

    # representation (docs/04_representation.md)
    model_dir: Path = Path("models")  # holds Qwen3-Embedding-0.6B/ (sentence-transformers folder)
    embedding_device: str = "auto"  # auto | cpu | cuda
    block_weights: tuple = (0.45, 0.55, 1.0)  # scope, target, phenomenon
    emm_component_weight: float = 0.5

    # attractor extraction and assignment (lac)
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
    # cosine thresholds are embedder-specific (docs/05_latent_anchors.md §5.10), calibrated for Qwen3
    min_assign_threshold: float = 0.75
    max_assign_threshold: float = 0.80
    soft_merge_low: float = 0.85
    orphan_buffer_min_factor: int = 3
    min_activation_alignment: float = 0.20  # ACTIVATES edges below this are dropped/rerouted
    # stability guards (docs/05_latent_anchors.md §5.6): adaptive hub threshold, per-attractor damping, trust region
    density_floor: float = 0.25  # hub threshold never below this share
    density_multiple: float = 3.0  # hub = more than this multiple of the uniform share 1/N_attractors
    max_centroid_step: float = 0.10  # max L2 move of one centroid per batch (largest healthy move measured: 0.010; docs/05_latent_anchors.md §5.6)
    warn_orphan_rate: float = 0.50
    warn_min_extraction_yield: float = 0.10
    warn_avg_degree: tuple = (1.0, 8.0)  # mutual k-NN degree envelope
    # OMP input scale s: lac fits codes with lasso alpha=1; scaling inputs by s is alpha/s
    # (OMP directions and relative error are scale-invariant). EMA updates never see s.
    dictionary_input_scale: float = 10.0
