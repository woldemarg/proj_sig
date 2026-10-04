"""Latent Semantic Attractor Graph — assignment, OMP extraction, mutual k-NN topology.

Pipeline stages implemented here: Assign, Isolate (orphans), Extract (OMP + signed repair),
RELATED_TO topology. lac vocabulary: a *chunk* is a pattern row, a *concept* an attractor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.metrics.pairwise import cosine_similarity

from attractor_topology.config import TopologyConfig as Config  # lac's Config: the topology settings

from .storage import ConceptStore


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _step_scale(damping: np.ndarray | None, concept_idx: int) -> float:
    """EMA step multiplier of one attractor (1.0 when undamped or minted in this batch)."""
    return 1.0 if damping is None or concept_idx >= len(damping) else float(damping[concept_idx])


def compute_adaptive_threshold(store: ConceptStore, config: Config) -> float:
    k = len(store.embeddings)
    if k < 10:
        return config.min_assign_threshold

    sim_dist = cosine_similarity(store.embeddings)
    np.fill_diagonal(sim_dist, np.nan)
    # Concept-concept sim runs higher than chunk-concept; percentile is a loose upper bound
    adaptive_thresh = float(np.nanpercentile(sim_dist, config.adaptive_percentile))
    return min(
        config.max_assign_threshold,
        max(config.min_assign_threshold, adaptive_thresh),
    )


def assign_and_update(
    x: np.ndarray,
    global_chunk_ids: list[int],
    store: ConceptStore,
    config: Config,
    adaptive_thresh: float,
    batch_id: int,
    damping: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[int]]:
    activations: list[dict[str, Any]] = []
    orphan_embs: list[np.ndarray] = []
    orphan_ids: list[int] = []

    if len(store.embeddings) == 0:
        for i in range(len(x)):
            orphan_embs.append(x[i])
            orphan_ids.append(global_chunk_ids[i])
        return activations, orphan_embs, orphan_ids

    sims = cosine_similarity(x, store.embeddings)

    for i in range(len(x)):
        best_idx = int(np.argmax(sims[i]))
        best_score = float(sims[i, best_idx])

        if best_score >= adaptive_thresh:
            top_k_indices = np.argsort(sims[i])[-config.top_k_assign :]
            for tgt_idx in top_k_indices:
                score = float(sims[i, tgt_idx])
                if score >= (best_score * config.mixture_ratio) and score >= adaptive_thresh:
                    activations.append(
                        {
                            "chunk_id": global_chunk_ids[i],
                            "concept_id": store.concept_ids[tgt_idx],
                            "weight": score,
                        }
                    )
                    store.update_concept_centroid(tgt_idx, x[i], config.centroid_alpha, batch_id, _step_scale(damping, tgt_idx))
        else:
            orphan_embs.append(x[i])
            orphan_ids.append(global_chunk_ids[i])

    return activations, orphan_embs, orphan_ids


def assign_orphans_nearest(
    orphan_embeddings: np.ndarray,
    orphan_chunk_ids: list[int] | np.ndarray,
    store: ConceptStore,
    config: Config,
    batch_id: int,
    damping: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Wire lone orphans to their nearest attractor when OMP buffer is too small."""
    if len(orphan_embeddings) == 0 or len(store.embeddings) == 0:
        return []

    sims = cosine_similarity(orphan_embeddings, store.embeddings)
    activations: list[dict[str, Any]] = []
    ids = list(orphan_chunk_ids)
    for i in range(len(orphan_embeddings)):
        best_idx = int(np.argmax(sims[i]))
        score = float(sims[i, best_idx])
        activations.append(
            {
                "chunk_id": int(ids[i]),
                "concept_id": store.concept_ids[best_idx],
                "weight": score,
            }
        )
        store.update_concept_centroid(best_idx, orphan_embeddings[i], config.centroid_alpha, batch_id, _step_scale(damping, best_idx))
    return activations


def soft_merge_orphans(
    new_centroids: np.ndarray,
    store: ConceptStore,
    config: Config,
) -> tuple[np.ndarray, dict[int, int]]:
    if len(store.embeddings) == 0:
        return new_centroids, {}

    kept_centroids: list[np.ndarray] = []
    absorptions: dict[int, int] = {}
    sims = cosine_similarity(new_centroids, store.embeddings)

    for i in range(len(new_centroids)):
        best_idx = int(np.argmax(sims[i]))
        best_sim = float(sims[i, best_idx])
        if best_sim > config.soft_merge_low:
            absorptions[i] = store.concept_ids[best_idx]
        else:
            kept_centroids.append(new_centroids[i])

    kept = np.array(kept_centroids, dtype=np.float32) if kept_centroids else np.empty((0, new_centroids.shape[1]), dtype=np.float32)
    return kept, absorptions


def route_absorbed_activations(
    local_acts: list[dict[str, Any]],
    absorptions: dict[int, int],
    global_chunk_ids: list[int] | np.ndarray,
    buffer_embeddings: np.ndarray,
    store: ConceptStore,
    config: Config,
    batch_id: int,
    damping: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Re-map absorbed OMP concepts to existing attractors and update centroids."""
    routed: list[dict[str, Any]] = []
    for edge in local_acts:
        local_concept = edge["concept_id"]
        if local_concept not in absorptions:
            continue
        global_chunk_id = int(global_chunk_ids[edge["chunk_id"]])
        global_concept_id = absorptions[local_concept]
        routed.append(
            {
                "chunk_id": global_chunk_id,
                "concept_id": global_concept_id,
                "weight": edge["weight"],
            }
        )
        store_idx = store.concept_ids.index(global_concept_id)
        store.update_concept_centroid(
            store_idx,
            buffer_embeddings[edge["chunk_id"]],
            config.centroid_alpha,
            batch_id,
            _step_scale(damping, store_idx),
        )
    return routed


def build_kept_local_to_global(
    n_local_concepts: int,
    absorptions: dict[int, int],
    new_global_ids: list[int],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    kept_j = 0
    for local_i in range(n_local_concepts):
        if local_i in absorptions:
            continue
        mapping[local_i] = new_global_ids[kept_j]
        kept_j += 1
    return mapping


def remap_activation_edges(
    edges: list[dict[str, Any]],
    *,
    chunk_id_map: list[int] | np.ndarray,
    concept_id_map: list[int] | np.ndarray | dict[int, int],
    skip_absorbed: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for edge in edges:
        if skip_absorbed and edge["concept_id"] in skip_absorbed:
            continue
        if isinstance(concept_id_map, dict):
            cid = concept_id_map[edge["concept_id"]]
        else:
            cid = int(concept_id_map[edge["concept_id"]])
        out.append(
            {
                **edge,
                "chunk_id": int(chunk_id_map[edge["chunk_id"]]),
                "concept_id": int(cid),
            }
        )
    return out


def calculate_knn_topology(store: ConceptStore, config: Config) -> list[dict[str, Any]]:
    """Mutual k-NN RELATED_TO edges — both concepts must be in each other's top-k."""
    embeddings = store.embeddings
    n = len(embeddings)
    if n < 2:
        return []

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -1)
    peer_count = min(config.related_to_peer_count, n - 1)
    top_k_mask = np.argsort(sim_matrix, axis=1)[:, -peer_count:]

    seen: set[tuple[int, int]] = set()
    edges: list[dict[str, Any]] = []

    for src_idx in range(n):
        for tgt_idx in top_k_mask[src_idx]:
            if src_idx not in top_k_mask[tgt_idx]:
                continue
            weight = float(sim_matrix[src_idx, tgt_idx])
            if weight <= config.related_to_min_weight:
                continue
            src_id = int(store.concept_ids[src_idx])
            tgt_id = int(store.concept_ids[tgt_idx])
            key = (min(src_id, tgt_id), max(src_id, tgt_id))
            if key not in seen:
                seen.add(key)
                edges.append({"source": key[0], "target": key[1], "weight": weight})
    return edges


def _build_local_activations(
    coefficients_abs: np.ndarray,
    concepts_per_chunk: int,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for chunk_index in range(len(coefficients_abs)):
        top_indices = np.argsort(coefficients_abs[chunk_index])[-concepts_per_chunk:]
        for concept_index in top_indices:
            weight = float(coefficients_abs[chunk_index, concept_index])
            if weight > 1e-5:
                edges.append(
                    {
                        "chunk_id": chunk_index,
                        "concept_id": int(concept_index),
                        "weight": weight,
                    }
                )
    return edges


def _omp_fallback_unit_norm_rows(embeddings: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """When K-sweep is ill-conditioned (tiny orphan buffer), mint one concept per row."""
    centroids = _l2_normalize_rows(embeddings.astype(np.float64)).astype(np.float32)
    return centroids, [{"chunk_id": i, "concept_id": i, "weight": 1.0} for i in range(len(centroids))]


def _omp_extract(embeddings: np.ndarray, config: Config) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """K-sweep OMP; returns (centroid_embeddings, local_acts). The counts come from ``repair_extraction``."""
    if len(embeddings) == 0:
        return np.empty((0, 0)), []

    n_samples = len(embeddings)
    if n_samples < config.dictionary_k_min:
        return _omp_fallback_unit_norm_rows(embeddings)

    actual_batch_size = min(config.dictionary_batch_size, n_samples)
    concepts_per_chunk = config.concepts_per_chunk
    effective_k_min = max(2, min(config.dictionary_k_min, n_samples))
    k_upper = min(config.max_concept_count, n_samples)
    k_step = max(1, config.dictionary_k_step)

    best_coefficients = None
    best_dictionary = None
    last_coefficients = None
    last_dictionary = None
    previous_error = float("inf")
    matrix_norm_sq = np.linalg.norm(embeddings, "fro") ** 2

    for k in range(effective_k_min, k_upper + 1, k_step):
        dictionary_learner = MiniBatchDictionaryLearning(
            n_components=k,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=concepts_per_chunk,
            batch_size=actual_batch_size,
            random_state=config.random_seed,
        )
        coefficients = dictionary_learner.fit_transform(embeddings)
        dictionary = dictionary_learner.components_
        last_coefficients = coefficients
        last_dictionary = dictionary

        reconstruction = coefficients @ dictionary
        reconstruction_error = np.linalg.norm(embeddings - reconstruction, "fro") ** 2 / matrix_norm_sq

        concept_usage = np.sum(np.abs(coefficients) > 1e-5, axis=0)
        dead_ratio = int(np.sum(concept_usage == 0)) / k
        improvement = previous_error - reconstruction_error

        if dead_ratio > config.max_dead_concept_ratio:
            if best_coefficients is not None:
                break  # dead-node limit: keep the last K that passed (best_* hold it)
            continue

        dynamic_tolerance = config.reconstruction_error_tolerance + (dead_ratio * config.dead_concept_penalty)

        if previous_error != float("inf") and improvement < dynamic_tolerance:
            # elbow: the extra atoms did not pay for themselves -> keep the previous K
            # (best_* already hold it), the parsimonious model
            break

        best_coefficients = coefficients
        best_dictionary = dictionary
        previous_error = reconstruction_error

    if best_coefficients is None or best_dictionary is None:
        if last_coefficients is not None and last_dictionary is not None:
            # every swept K hit the dead-node limit: use the last attempt
            best_coefficients = last_coefficients
            best_dictionary = last_dictionary
        else:
            return _omp_fallback_unit_norm_rows(embeddings)

    coefficients_abs = np.abs(best_coefficients)
    concept_usage = np.sum(coefficients_abs > 1e-5, axis=0)
    active_indices = np.where(concept_usage > 0)[0]
    coefficients_abs = coefficients_abs[:, active_indices]
    dictionary = best_dictionary[active_indices, :]

    centroid_embeddings = _l2_normalize_rows(dictionary.astype(np.float64)).astype(np.float32)
    local_acts = _build_local_activations(coefficients_abs, concepts_per_chunk)
    return centroid_embeddings, local_acts


def repair_extraction(
    centroids: np.ndarray,
    local_acts: list[dict[str, Any]],
    unit_rows: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Turn OMP atoms into signed attractors before they are stored.

    OMP reports |coefficient|, so an anti-aligned row can "activate" an atom, and
    the K-sweep can split one phenomenon into near-duplicate atoms. This pass:

    1. flips an atom when its users are anti-aligned on balance (atom sign is arbitrary);
    2. merges atoms with cosine > ``soft_merge_low`` into the most-used one
       (usage-weighted mean) — the rule applied against existing centroids, also
       applied inside this extraction;
    3. drops activations below ``min_activation_alignment`` and reroutes uncovered
       rows to their best-aligned atom (``rerouted``);
    4. drops unused atoms and remaps concept ids to ``0 .. n-1``.
    """
    if len(centroids) == 0:
        return centroids, np.empty(0, dtype=np.int64), local_acts
    cents = _l2_normalize_rows(centroids.astype(np.float64))
    cos = unit_rows.astype(np.float64) @ cents.T  # (rows, atoms)
    for j in range(len(cents)):
        users = [a["chunk_id"] for a in local_acts if a["concept_id"] == j]
        if users and cos[users, j].sum() < 0:
            cents[j] *= -1.0
            cos[:, j] *= -1.0
    usage = np.array(
        [sum(1 for a in local_acts if a["concept_id"] == j) for j in range(len(cents))],
        dtype=float,
    )
    target = list(range(len(cents)))
    order = sorted(range(len(cents)), key=lambda j: -usage[j])
    kept_atoms: list[int] = []
    for j in order:
        host = next((k for k in kept_atoms if float(cents[j] @ cents[k]) > config.soft_merge_low), None)
        if host is None:
            kept_atoms.append(j)
        else:
            target[j] = host
    for k in kept_atoms:
        group = [j for j in range(len(cents)) if target[j] == k]
        if len(group) > 1:
            merged = sum(max(usage[j], 1.0) * cents[j] for j in group)
            cents[k] = merged / np.linalg.norm(merged)
    local_acts = [{**a, "concept_id": target[a["concept_id"]]} for a in local_acts]
    cos = unit_rows.astype(np.float64) @ cents.T
    for j in range(len(cents)):
        if target[j] != j:
            cos[:, j] = -np.inf  # absorbed atoms are no longer reroute candidates
    floor = config.min_activation_alignment
    kept = [a for a in local_acts if cos[a["chunk_id"], a["concept_id"]] >= floor]
    covered = {a["chunk_id"] for a in kept}
    for row in range(len(unit_rows)):
        if row not in covered:
            # every row keeps a home atom (coverage invariant); a reroute below the
            # alignment floor is recorded as weak so consumers can tell it apart
            j = int(np.argmax(cos[row]))
            kept.append(
                {"chunk_id": row, "concept_id": j, "weight": float(max(cos[row, j], 0.0)), "rerouted": True, "weak": bool(cos[row, j] < floor)}
            )
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for act in kept:  # merged hosts may receive the same row twice
        key = (act["chunk_id"], act["concept_id"])
        if key not in unique or act["weight"] > unique[key]["weight"]:
            unique[key] = act
    used = sorted({concept for _, concept in unique})
    remap = {old: new for new, old in enumerate(used)}
    acts = [{**a, "concept_id": remap[a["concept_id"]]} for a in unique.values()]
    counts = np.bincount([a["concept_id"] for a in acts], minlength=len(used)).astype(np.int64)
    return cents[used].astype(np.float32), counts, acts


def extract_attractors(rows: np.ndarray, unit_rows: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Mint attractors (cold start and orphan buffer): OMP on ``dictionary_input_scale * rows``,
    then the signed repair; returns (centroids, chunk_counts, local activations)."""
    scaled = np.asarray(rows, dtype=np.float32) * float(config.dictionary_input_scale)
    centroids, acts = _omp_extract(scaled, config)
    return repair_extraction(centroids, acts, unit_rows, config)
