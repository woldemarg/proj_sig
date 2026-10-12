"""Statistical discovery adapter over the vendored EDA engine ``subgroup_miner/vendor/eda/main_upd.py`` (docs/02_discovery.md).

Calls the EDA steps unchanged, in the order of the EDA workflow:

    step1_profile_data -> step2_evaluate_macro_groupings -> step3_generate_search_space
    -> step4_evaluate_micro_slices (pass 1) -> rank by temp_index -> step4b_deep_validation (pass 2)

and converts the resulting DataFrames into typed ``Insight`` objects. The adapter
adds only what the EDA does not produce: shift *direction*, a significance
estimate, the EDA workflow's ``temp_index`` ranking into validation (the upstream
standalone runner and its print-only step 5 are not vendored), and provenance.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, zscore

import subgroup_miner.vendor.eda.main_upd as eda
from insight_contracts import Condition, Insight, Rejection, Shift, pattern_id
from subgroup_miner.config import MinerConfig
from subgroup_miner.spatial import SpatialContext, effective_n_factor, subgroup_moran

EDA_SOURCE = "subgroup_miner/vendor/eda/main_upd.py"
NULL_LEVELS = frozenset({"nan", "<NA>", "None"})  # never a condition (same literals as EDA step 3)
# asymptotic s.e. of the median: 1.2533 * sigma / sqrt(n); sigma ~= 1.4826 * MAD
_MEDIAN_SE_FACTOR = 1.2533 * 1.4826


class DiscoveryError(RuntimeError):
    """Raised with a stable ``code`` for the UI (docs/02_discovery.md §2.7)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numerics: list[str]
    categoricals: list[str]
    dropped_columns: list[str]
    selected_dimensions: list[str]
    search_space_size: int
    global_medians: dict[str, float]
    global_mads: dict[str, float]
    dimension_cardinality: dict[str, int]
    dimension_entropy: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Candidate:
    """One distinct pass-1 cohort (closed intent), optionally enriched with pass-2 validation."""

    expression: str  # EDA selector kept as provenance (the member with most conditions)
    conditions: tuple[Condition, ...]  # closed intent: every condition that holds on the whole extent
    row_indices: np.ndarray
    row_count: int
    volume_utility: float
    top_shifts: list[tuple[str, float]]  # EDA (metric, |robust z|)
    sd_aggregate_score: float
    emm_stabilized_score: float
    temp_index: float = 0.0
    final_sd_score: float = 0.0
    drivers: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # selectors merged into this cohort
    restated: frozenset[str] = frozenset()  # the metrics a closed condition is computed from: never its shifts (§2.9)


@dataclass
class DiscoveryResult:
    profile: DatasetProfile
    candidates: list[Candidate]  # distinct cohorts (one per extent)
    validated: list[Candidate]
    data: pd.DataFrame  # EDA data_safe (row positions == source file rows)
    n_tests: int  # multiple-testing family size (distinct cohorts x metrics)
    pass1_subgroups: int  # pass-1 selectors before merging identical extents
    rejections: list[Rejection] = field(default_factory=list)  # cover_equivalent / near_duplicate / tautological
    spatial: SpatialContext | None = None  # rows are H3 cells (docs/02_discovery.md §2.9)
    ranks: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # the metrics' global ranks (EDA profile)
    global_corr: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # their rank correlation over all rows


def emm_pair_scale(n_metrics: int) -> float:
    """sqrt(m(m-1)): number of off-diagonal entries of an m x m correlation matrix."""
    return math.sqrt(n_metrics * (n_metrics - 1)) if n_metrics > 1 else 1.0


def _z_positive(values: pd.Series) -> pd.Series:
    """EDA ranking primitive: zscore -> fillna(0) -> clip(lower=0)."""
    return pd.Series(zscore(values.astype(float)), index=values.index).fillna(0).clip(lower=0)


def _entropy(series: pd.Series) -> float:
    p = series.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def _conditions_of(selector: Any) -> tuple[Condition, ...]:
    return tuple(sorted(Condition(str(s.attribute_name), str(s.attribute_value)) for s in selector.selectors))


def closed_intent(data: pd.DataFrame, rows: np.ndarray, categoricals: list[str]) -> tuple[Condition, ...]:
    """Galois closure int(ext(S)): every (attribute, value) that holds on all covered rows (docs/02_discovery.md §2.3)."""
    covered = data.iloc[rows]
    closed = []
    for col in categoricals:
        values = covered[col].unique()
        if len(values) == 1 and str(values[0]) not in NULL_LEVELS:
            closed.append(Condition(col, str(values[0])))
    return tuple(sorted(closed))


def merge_identical_extents(
    cands: list[Candidate], data: pd.DataFrame, categoricals: list[str], derived_from: dict[str, list[str]] | None = None
) -> tuple[list[Candidate], list[Rejection]]:
    """One cohort per extent, described by its closed intent; the other selectors become aliases (R5). The metrics a
    closed condition is computed from (``derived_from``: a geo dataset's neighbourhood conditions, §2.9) are the
    cohort's ``restated`` set and their shifts leave ``top_shifts`` here: pass 1 scored none for the selector's own
    conditions (``eda.step4_evaluate_micro_slices(forbidden=)``), the closure can add conditions."""
    groups: dict[bytes, list[Candidate]] = {}
    for cand in cands:
        groups.setdefault(np.sort(cand.row_indices).tobytes(), []).append(cand)
    merged: list[Candidate] = []
    rejections: list[Rejection] = []
    for members in groups.values():
        members.sort(key=lambda c: (-len(c.conditions), c.expression))
        head, rest = members[0], members[1:]
        closed = closed_intent(data, head.row_indices, categoricals)
        enumerated = {cond for member in members for cond in member.conditions}
        if not enumerated <= set(closed):  # a selector value that does not match the data string
            raise DiscoveryError("internal_error", f"closure of {head.expression} misses {sorted(enumerated - set(closed))}")
        head.conditions = closed
        head.restated = frozenset(m for c in closed for m in (derived_from or {}).get(c.attribute, ()))
        if head.restated:
            head.top_shifts = [s for s in head.top_shifts if s[0] not in head.restated]
            head.sd_aggregate_score = sum(v for _, v in head.top_shifts)  # as pass 1 sums the shifts it keeps
        head.aliases = [m.expression for m in rest]
        rejections += [Rejection(m.expression, "cover_equivalent", f"same rows as {head.expression}") for m in rest]
        merged.append(head)
    return merged, rejections


def _primary_key(cand: Candidate, values: np.ndarray, column: dict[str, int], global_medians: dict[str, float]) -> tuple[str, int] | None:
    """(primary metric, shift sign) of a pass-1 cohort; None without shifts. ``values``: the metric columns as one array."""
    if not cand.top_shifts:
        return None
    metric = cand.top_shifts[0][0]
    local = float(np.nanmedian(values[cand.row_indices, column[metric]]))
    return metric, 1 if local >= global_medians[metric] else -1


def _jaccard(rows: np.ndarray, other: np.ndarray) -> float:
    inter = np.intersect1d(rows, other, assume_unique=True).size
    return inter / (rows.size + other.size - inter)


def prune_near_duplicates(
    ranked: list[Candidate], values: np.ndarray, numerics: list[str], global_medians: dict[str, float], threshold: float, limit: int | None = None
) -> tuple[list[Candidate], list[Rejection]]:
    """Greedy in rank order (R6): a cohort with the primary metric and sign of a higher-ranked kept
    cohort and Jaccard(rows) >= ``threshold`` is absorbed as an alias of it. At most ``limit`` cohorts are
    kept (the validation budget: nothing below it is validated); the rest are scanned only as aliases of those."""
    kept: list[tuple[Candidate, tuple[str, int] | None, np.ndarray]] = []
    rejections: list[Rejection] = []
    column = {name: j for j, name in enumerate(numerics)}
    for cand in ranked:
        key = _primary_key(cand, values, column, global_medians)
        rows = np.sort(cand.row_indices)
        host = next(
            (other for other, other_key, other_rows in kept if key is not None and key == other_key and _jaccard(rows, other_rows) >= threshold), None
        )
        if host is None:
            if limit is None or len(kept) < limit:
                kept.append((cand, key, rows))
            continue
        host.aliases += [cand.expression, *cand.aliases]
        rejections.append(Rejection(cand.expression, "near_duplicate", f"Jaccard>={threshold} with {host.expression}"))
    return [cand for cand, _, _ in kept], rejections


def _validation_frame(cands: list[Candidate]) -> pd.DataFrame:
    """The columns ``step4b_deep_validation`` reads; scope attributes include implied ones."""
    return pd.DataFrame(
        {
            "dimensions": [c.expression for c in cands],
            "dimension_attrs": [{cond.attribute for cond in c.conditions} for c in cands],
            "row_indices": [c.row_indices for c in cands],
            "row_count": [c.row_count for c in cands],
            "volume_utility": [c.volume_utility for c in cands],
            "top_shifts": [c.top_shifts for c in cands],
            "restated": [c.restated for c in cands],
            "sd_aggregate_score": [c.sd_aggregate_score for c in cands],
            "emm_stabilized_score": [c.emm_stabilized_score for c in cands],
        }
    )


def run_discovery(
    df: pd.DataFrame, config: MinerConfig, on_stage: Callable[[str], None] | None = None, spatial: SpatialContext | None = None
) -> DiscoveryResult:
    """EDA pass 1 -> distinct closed cohorts -> ranking -> near-duplicate pruning -> pass 2.

    Deduplication runs *before* the validation budget is spent, so the bootstrap only
    sees distinct cohorts (docs/02_discovery.md §2.3)."""
    profile = eda.step1_profile_data(df)
    numerics: list[str] = list(profile["numerics"])
    categoricals: list[str] = list(profile["categoricals"])
    data: pd.DataFrame = profile["data_safe"]
    if not numerics:
        raise DiscoveryError("no_numeric_targets", "No usable numeric target columns after profiling.")
    if not categoricals:
        raise DiscoveryError("invalid_schema", "No categorical dimensions to slice by (consider BIN_COLUMNS).")

    dims = eda.step2_evaluate_macro_groupings(profile, min_categories=config.min_search_dimensions)
    if spatial:  # the neighbourhood conditions are the geo mode's question: always searched (§2.9)
        forced = [c for c in categoricals if c in spatial.conditions]
        dims = forced + [d for d in dims if d not in forced]
    space = eda.step3_generate_search_space(profile, dims, compute_budget=config.compute_budget)
    global_medians = {n: float(np.median(data[n].dropna())) for n in numerics}
    global_mads = {n: float(eda.calculate_mad(data[n].dropna())) for n in numerics}
    prof = DatasetProfile(
        rows=int(profile["total_rows"]),
        columns=int(df.shape[1]),
        numerics=numerics,
        categoricals=categoricals,
        dropped_columns=[c for c in df.columns if c not in numerics and c not in categoricals],
        selected_dimensions=list(dims),
        search_space_size=len(space),
        global_medians=global_medians,
        global_mads=global_mads,
        dimension_cardinality={c: int(data[c].nunique()) for c in categoricals},
        dimension_entropy={c: _entropy(data[c]) for c in categoricals},
    )
    if not space:
        raise DiscoveryError("no_candidates", f"Search space is empty (dimensions selected: {dims}).")

    if on_stage:
        on_stage("DISCOVERING")
    raw = eda.step4_evaluate_micro_slices(profile, space, spatial.derived_from if spatial else None)
    if raw.empty:
        raise DiscoveryError("no_candidates", "No subgroup passed the pass-1 size screen.")
    # The EDA's EMM score is a Frobenius norm over all m(m-1) off-diagonal entries, so its
    # scale grows with the number of metrics; dividing by sqrt(m(m-1)) turns it into the
    # RMS correlation change per metric pair (in [0, 2]), which the thresholds refer to.
    raw["emm_stabilized_score"] = raw["emm_stabilized_score"] / emm_pair_scale(len(numerics))

    by_expr = {str(sel): sel for sel in space}
    pass1 = [
        Candidate(
            expression=row["dimensions"],
            conditions=_conditions_of(by_expr[row["dimensions"]]),
            row_indices=np.asarray(row["row_indices"], dtype=np.int64),
            row_count=int(row["row_count"]),
            volume_utility=float(row["volume_utility"]),
            top_shifts=[(str(m), float(v)) for m, v in row["top_shifts"]],
            sd_aggregate_score=float(row["sd_aggregate_score"]),
            emm_stabilized_score=float(row["emm_stabilized_score"]),
        )
        for _, row in raw.iterrows()
    ]
    merged, rejections = merge_identical_extents(pass1, data, categoricals, spatial.derived_from if spatial else None)
    candidates = []
    for c in merged:  # a cohort without a shift stands on its covariance pair alone; without one it can become no insight
        if c.top_shifts or _covariance_pair(c.row_indices, profile["ranks"], profile["global_corr"], numerics, c.restated):
            candidates.append(c)
        elif c.restated:
            rejections.append(Rejection(c.expression, "tautological", f"only shifts of {sorted(c.restated)}, no correlation change"))
        else:
            rejections.append(Rejection(c.expression, "no_shift", "no metric has a median inside the cohort, no correlation change"))

    # EDA workflow ranking (upstream runner's temp_index), over the distinct cohorts
    scores = pd.DataFrame([(c.sd_aggregate_score, c.emm_stabilized_score, c.volume_utility) for c in candidates], columns=["sd", "emm", "vol"])
    for cand, index in zip(candidates, _z_positive(scores["sd"]) + _z_positive(scores["emm"]) + _z_positive(scores["vol"])):
        cand.temp_index = float(index)
    ranked = sorted(candidates, key=lambda c: -c.temp_index)  # stable: pass-1 order breaks ties
    eligible, near = prune_near_duplicates(ranked, profile["values"], numerics, global_medians, config.redundancy_jaccard, config.validation_budget)
    rejections += near

    np.random.seed(config.eda_random_seed)  # step4b bootstrap uses the global RNG
    top = _validation_frame(eligible)
    validated_df = eda.step4b_deep_validation(
        data, top, numerics, categoricals, global_medians, global_mads, spatial.blocks if spatial else np.arange(len(data))
    )
    by_candidate = {c.expression: c for c in candidates}
    validated: list[Candidate] = []
    for _, row in validated_df.iterrows():
        cand = by_candidate[row["dimensions"]]
        cand.final_sd_score = float(row["final_sd_score"])
        cand.drivers = list(row["root_cause_drivers"])
        validated.append(cand)

    return DiscoveryResult(
        prof,
        candidates,
        validated,
        data,
        n_tests=len(candidates) * len(numerics),
        pass1_subgroups=len(pass1),
        rejections=rejections,
        spatial=spatial,
        ranks=profile["ranks"],
        global_corr=profile["global_corr"],
    )


def _median_test(values: np.ndarray, global_median: float, global_mad: float, n_eff_factor: float = 1.0) -> float:
    """Two-sided asymptotic test of H0: subgroup median == global median; ``n_eff_factor`` < 1 shrinks the
    sample to its effective size under spatial autocorrelation (docs/02_discovery.md §2.9)."""
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return 1.0
    root_n = math.sqrt(max(n * n_eff_factor, 1.0))
    scale = eda.calculate_mad(values) or global_mad  # same robust scale (incl. the zero-MAD fallback) as the EDA
    se = _MEDIAN_SE_FACTOR * scale / root_n if scale else float(np.std(values, ddof=1)) / root_n
    if not se or not np.isfinite(se):
        return 1.0
    z = (float(np.median(values)) - global_median) / se
    return float(2.0 * norm.sf(abs(z)))


def _covariance_pair(
    row_indices: np.ndarray, ranks: np.ndarray, glob: np.ndarray, numerics: list[str], restated: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Metric pair whose rank correlation inside the cohort diverges most from the global one ``glob`` (explains the
    EMM score; the same measure, ``eda.rank_correlation``, on the same global ranks). A ``restated`` metric is in no
    pair: a neighbourhood condition never explains the metric it is computed from."""
    if len(numerics) < 2 or len(row_indices) < 3 * len(numerics):
        return {}
    # a metric constant inside the subgroup has an undefined (NaN) correlation: excluded,
    # never read as "correlation dropped to 0"
    local = eda.rank_correlation(ranks[row_indices])
    allowed = np.array([m not in restated for m in numerics])
    delta = np.where(np.triu(np.ones_like(local, dtype=bool), k=1) & allowed[:, None] & allowed, local - glob, np.nan)
    if not np.isfinite(delta).any() or np.nanmax(np.abs(delta)) <= 1e-12:
        return {}
    i, j = np.unravel_index(int(np.nanargmax(np.abs(delta))), delta.shape)
    return {
        "pair": [numerics[i], numerics[j]],
        "local_corr": float(local[i, j]),
        "global_corr": float(glob[i, j]),
        "delta": float(local[i, j] - glob[i, j]),
    }


def build_insights(result: DiscoveryResult, config: MinerConfig, *, dataset_id: str, batch_id: str, filename: str) -> list[Insight]:
    """Validated candidates -> canonical ``Insight`` objects (unfiltered, weight unset)."""
    validated = result.validated
    if not validated:
        return []
    prof = result.profile
    spatial = result.spatial
    insights: list[Insight] = []
    for cand in validated:
        rows = result.data.iloc[cand.row_indices]
        covariance = _covariance_pair(cand.row_indices, result.ranks, result.global_corr, prof.numerics, cand.restated)
        shifts: list[Shift] = []  # never empty: a cohort without a shift entered the rank on its covariance pair
        for metric, magnitude in cand.top_shifts:
            local_median = float(np.median(rows[metric].dropna()))
            sign = np.sign(local_median - prof.global_medians[metric]) or 1.0
            shifts.append(Shift(metric, float(sign * magnitude), local_median, prof.global_medians[metric], prof.global_mads[metric]))
        for metric in covariance.get("pair", []):
            if all(s.metric != metric for s in shifts):
                local_median = float(np.median(rows[metric].dropna()))
                gm, gd = prof.global_medians[metric], prof.global_mads[metric]
                sign = np.sign(local_median - gm) or 1.0
                shifts.append(Shift(metric, float(sign * eda.robust_z_score(local_median, gm, gd)), local_median, gm, gd, "covariance_pair"))
        shifts.sort(key=lambda s: s.magnitude, reverse=True)
        primary = shifts[0]
        values = rows[primary.metric].to_numpy(dtype=float)
        rho = subgroup_moran(values, cand.row_indices, spatial) if spatial else 0.0  # the subgroup's own dependence, not the map's
        n_eff_factor = effective_n_factor(rho)
        p_value = _median_test(values, primary.global_median, primary.global_mad, n_eff_factor)
        pid = pattern_id(dataset_id, cand.conditions)
        provenance_extra = {}
        if spatial:
            p_iid = _median_test(values, primary.global_median, primary.global_mad)
            provenance_extra["spatial"] = {
                "resolution": spatial.resolution,
                "cells": cand.row_count,
                "moran_i": round(rho, 4),
                "n_eff": round(int(np.isfinite(values).sum()) * n_eff_factor, 1),
                "p_adjusted_iid": min(1.0, p_iid * result.n_tests),
            }
        insights.append(
            Insight(
                id=pid,
                dataset_id=dataset_id,
                batch_id=batch_id,
                conditions=cand.conditions,
                expression=cand.expression,
                target=primary.metric,
                shifts=tuple(shifts),
                support=cand.row_count,
                support_fraction=cand.row_count / max(prof.rows, 1),
                baseline=primary.global_median,
                local=primary.local_median,
                effect_size=primary.robust_z,
                sd_score=cand.final_sd_score,
                sd_raw_score=cand.sd_aggregate_score,
                emm_score=cand.emm_stabilized_score,
                volume_utility=cand.volume_utility,
                stability=cand.final_sd_score / cand.sd_aggregate_score if cand.sd_aggregate_score > 0 else 0.0,
                p_value=p_value,
                p_adjusted=min(1.0, p_value * result.n_tests),
                drivers=tuple(cand.drivers),
                row_hash=hashlib.sha1(np.sort(cand.row_indices).tobytes()).hexdigest()[:16],
                aliases=tuple(cand.aliases),
                covariance=covariance,
                provenance={
                    "dataset_id": dataset_id,
                    "filename": filename,
                    "batch_id": batch_id,
                    "engine": EDA_SOURCE,
                    "steps": [
                        "step1_profile_data",
                        "step2_evaluate_macro_groupings",
                        "step3_generate_search_space",
                        "step4_evaluate_micro_slices",
                        "step4b_deep_validation",
                    ],
                    "expression": cand.expression,
                    "multiple_testing_family": result.n_tests,
                    **provenance_extra,
                },
            )
        )
    return insights


def covers_of(result: DiscoveryResult) -> dict[str, np.ndarray]:
    """Expression -> covered row positions, for redundancy checks and provenance."""
    return {c.expression: c.row_indices for c in result.candidates}
