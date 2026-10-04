"""Pure pieces of the batch lifecycle (docs/09_operations.md §9.1): admission into the graph, the journal records,
the batch metrics. The engine runs the stages and owns the state."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np

from attractor_topology.models import CanonicalInsight, EmbeddingSpec
from attractor_topology.ontology import OntologyUpdate
from insight_contracts import Insight, Rejection
from subgroup_miner.discovery import DiscoveryResult


def pattern_records(kept: list[Insight], canon: list[CanonicalInsight], row_ids: list[int], spec: EmbeddingSpec) -> list[dict[str, Any]]:
    """Journal records: the Insight plus its journal row, canonical form and vector contract."""
    embedding = {"fingerprint": spec.fingerprint, "model_id": spec.model_id, "dim": spec.dim, "representation_version": spec.representation_version}
    records = []
    for ins, can, row_id in zip(kept, canon, row_ids):
        rec = ins.to_record()
        rec.update(
            row_id=row_id,
            canonical={**asdict(can), "document": can.document(), "components": [list(c) for c in can.components]},
            embedding=embedding,
        )
        records.append(rec)
    return records


def admit_insights(ranked: list[Insight], min_weight: float, budget: int) -> tuple[list[Insight], list[Rejection]]:
    """The graph's admission policy over valid insights, heaviest first as ``select_insights`` returns them
    (docs/03_insights.md §3.2): R4 weight floor (``MIN_INSIGHT_WEIGHT``), then R7 batch budget (``MAX_INSIGHTS_PER_BATCH``)."""
    heavy = [i for i in ranked if i.weight >= min_weight]
    refused = [Rejection(i.expression, "low_weight", f"weight={i.weight:.2f}") for i in ranked if i.weight < min_weight]
    refused += [Rejection(i.expression, "budget", f"max_insights_per_batch={budget}") for i in heavy[budget:]]
    return heavy[:budget], refused


def batch_metrics(
    result: DiscoveryResult,
    kept: list[Insight],
    rejections: list[Rejection],
    update: OntologyUpdate,
    spec: EmbeddingSpec,
    snapshot: dict[str, Any],
    timings: dict[str, float],
    t0: float,
) -> dict[str, Any]:
    """Batch-record metrics (docs/09_operations.md §9.5)."""
    om, graph = update.metrics, snapshot["stats"]
    pruned = Counter(r.reason for r in rejections)
    return {
        "input_rows": result.profile.rows,
        "columns": result.profile.columns,
        "numeric_targets": len(result.profile.numerics),
        "dimensions": len(result.profile.selected_dimensions),
        "search_space": result.profile.search_space_size,
        "pass1_subgroups": result.pass1_subgroups,
        "candidate_patterns": len(result.candidates),
        "validated_candidates": len(result.validated),
        "validated_insights": len(kept),
        "pruned": dict(pruned),
        "pruned_total": sum(pruned.values()),
        "avg_insight_support": float(np.mean([i.support for i in kept])),
        "avg_insight_weight": float(np.mean([i.weight for i in kept])),
        "embedding_count": len(kept),
        "embedding_dim": spec.dim,
        "attractors_total": om["total_concepts"],
        "attractors_new": len(update.new_attractors),
        "orphan_rate": om["orphan_rate"],
        "activation_count": om["activation_count"],
        "soft_merged": om["soft_merged"],
        "centroid_drift": om["centroid_drift"],
        "adaptive_thresh": om["adaptive_thresh"],
        "max_concept_density_pct": om["max_concept_density_pct"],
        "density_threshold": om["density_threshold"],
        "damped_attractors": om["damped_attractors"],
        "max_centroid_step": om["max_centroid_step"],
        "clamped_attractors": om["clamped_attractors"],
        "avg_attractor_degree": graph["avg_attractor_degree"],
        "graph_edges": graph["edges"],
        "edge_counts": graph["edge_counts"],
        "graph_patterns": graph["patterns"],
        "ontology_warnings": om["warnings"],
        "timings": timings,
        "processing_duration_s": time.perf_counter() - t0,
    }
