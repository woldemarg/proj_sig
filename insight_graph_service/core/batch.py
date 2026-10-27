"""Pure pieces of the batch lifecycle (docs/09_operations.md §9.1): admission into the graph, the journal records,
the batch metrics, the time column's check and the series of the admitted insights. The engine runs the stages and owns
the state."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from attractor_topology.models import CanonicalInsight, EmbeddingSpec
from attractor_topology.ontology import OntologyUpdate
from insight_contracts import Insight, Rejection, record_digest
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import DiscoveryResult
from subgroup_miner.ingestion import IngestionError, LoadedDataset
from subgroup_miner.temporal import annotate, calendar_bins, extent_points, reference_scale


def pattern_records(kept: list[Insight], canon: list[CanonicalInsight], row_ids: list[int], spec: EmbeddingSpec) -> list[dict[str, Any]]:
    """Journal records: the Insight plus its journal row, canonical form, vector contract and record digest."""
    embedding = {"fingerprint": spec.fingerprint, "model_id": spec.model_id, "dim": spec.dim, "representation_version": spec.representation_version}
    records = []
    for ins, can, row_id in zip(kept, canon, row_ids):
        rec = ins.to_record()
        rec.update(
            row_id=row_id,
            canonical={**asdict(can), "document": can.document(), "components": [list(c) for c in can.components]},
            embedding=embedding,
        )
        rec["record_digest"] = record_digest(rec)
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


def series_times(loaded: LoadedDataset) -> pd.Series:
    """The times of the rows a batch's series are of (docs/02_discovery.md §2.10): a table's every row, a geo level's
    located points only (``point_cell >= 0``) — what VALIDATING checks and the commit bins."""
    if loaded.spatial is None:
        return loaded.time
    return loaded.time[loaded.spatial.point_cell >= 0].reset_index(drop=True)


def check_time(times: pd.Series, column: str, grid: str, min_bins: int, max_bins: int, rows: str = "", tz: str | None = None) -> None:
    """A time column spans at most ``max_bins`` bins of ``grid`` (``TEMPORAL_MAX_BINS``) and gives at least ``min_bins``
    measurable ones (``TEMPORAL_MIN_BINS``, never fewer than one) — bins holding rows and covered at least half by the
    span —, else ``invalid_options``: at upload over the file's rows, before a batch exists, and at VALIDATING over the
    rows its series will be of (``series_times``; ``rows`` names them)."""
    bins = calendar_bins(times, grid, max_bins, tz)[1]
    if (filled := sum(b["rows"] > 0 and b["covered"] >= 0.5 for b in bins)) < max(min_bins, 1):
        raise IngestionError(
            "invalid_options",
            f"time column '{column}'{rows}: {filled} {grid} bins measurable (rows, and half their days in the span; TEMPORAL_MIN_BINS={min_bins})",
        )


def annotate_batch(
    loaded: LoadedDataset, kept: list[Insight], rows: dict[str, np.ndarray], column: str, config: MinerConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every admitted insight's series over the time column's grid, and what the batch record signs of it but the series
    file's digest — the column, the grid, the span (the first and the last time of the rows), the bins, the rows without
    a time, each metric's reference scale
    (docs/02_discovery.md §2.10). A geo level's series are of its located points: a pattern's are those of its cells, a
    ``<c>_median`` metric's values those of point column ``c``."""
    times, frame, covers = series_times(loaded), loaded.frame, rows
    if loaded.spatial is not None:
        located = loaded.spatial.point_cell >= 0
        frame = loaded.point_values[located].reset_index(drop=True).rename(columns=lambda c: f"{c}_median")
        covers = extent_points(loaded.spatial.point_cell[located], rows)
    codes, bins = calendar_bins(times, loaded.grid, tz=loaded.tz)
    scale = reference_scale(frame)
    metrics = {i.id: [s.metric for s in i.shifts] for i in kept}
    series = annotate(frame, codes, bins, covers, metrics, scale, config.temporal_min_rows_bin, config.temporal_tie_share)
    signed = {
        "column": column,
        "grid": loaded.grid,
        "span": [t.tz_convert(loaded.tz or "UTC").isoformat() for t in (times.min(), times.max())],  # in the grid's zone, as its bins
        "bins": bins,
        "rows_without_time": int((codes < 0).sum()),
        "reference_scale": scale,
    }
    return signed, series
