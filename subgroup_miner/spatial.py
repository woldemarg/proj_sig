"""Point data -> a table of H3 cells with spatial conditions (docs/02_discovery.md §2.9).

One row per occupied H3 cell. Column roles follow the source dtypes only, never column names:
counters (``points``, ``<time>_active_days``) for every cell; intensive metrics (medians, shares,
``<time>_late_share``) only for cells with at least ``min_cell_points`` points (NaN otherwise);
categorical modes and neighbourhood conditions as text dimensions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import h3
import numpy as np
import pandas as pd

from insight_contracts import Insight

POINT_PARSE_SHARE = 0.95  # share of non-null values that must parse as points for a column to be the location
LAT_NAMES = frozenset({"lat", "latitude"})
LON_NAMES = frozenset({"lon", "lng", "long", "longitude"})
_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_GEOJSON = re.compile(r'"coordinates"\s*:\s*\[\s*' + _NUM + r"\s*,\s*" + _NUM)
_WKT = re.compile(r"^\s*POINT\s*\(\s*" + _NUM + r"\s+" + _NUM + r"\s*\)\s*$", re.IGNORECASE)
LISA_LABELS = {(True, True): "hot spot", (False, False): "cold spot", (True, False): "high outlier", (False, True): "low outlier"}
LISA_COLUMN = "lisa_points"
LISA_WORDS = {
    "hot spot": "a hot spot: a busy cell surrounded by busy cells",
    "cold spot": "a cold spot: a quiet cell surrounded by quiet cells",
    "high outlier": "a busy cell surrounded by quiet cells",
    "low outlier": "a quiet cell surrounded by busy cells",
    "no cluster": "not part of a busy or quiet cluster",
}
QUARTERS = {"q1": "among the quietest quarter", "q2": "below the middle", "q3": "above the middle", "q4": "among the busiest quarter"}
NO_CLUSTER = "no cluster"  # not "none": a bare English word grounds onto unrelated question words
COUNTER = "points"
PLACE_PURITY = 0.9  # mean top-level share per dense cell above which a categorical describes the place
FUNCTIONAL_SHARE = 0.98  # nesting above which a mode column is dropped (it restates a share column)
NESTED_SHARE = 0.9  # nesting above which a mode condition never explains the other column's shares
COLOCATED_TOP = 3  # CO_LOCATED partners kept per pattern (the largest overlaps)
MIN_MIXED_SHARE = 0.1  # a share metric needs this share of dense cells strictly between 0 and 1


class SpatialError(ValueError):
    """A geo option that cannot be applied (mapped to ``invalid_options`` by ingestion)."""


@dataclass
class SpatialContext:
    """What the cell table cannot hold: the cell ids, point -> cell provenance and the neighbourhood graph."""

    resolution: int
    location: str  # the column(s) that held the points
    cells: np.ndarray  # H3 id per cell-table row
    point_cell: np.ndarray  # source row -> cell row (-1: no valid point)
    lonlat: np.ndarray  # float32 (source rows, 2)
    nbr_ptr: np.ndarray  # CSR over cell rows: ring-1 neighbours present in the table
    nbr_idx: np.ndarray
    min_cell_points: int
    sparse_cells: int = 0
    lag_columns: list[str] = field(default_factory=list)  # numeric neighbourhood columns, banded by ingestion
    conditions: list[str] = field(default_factory=list)  # the neighbourhood dimensions, always searched
    derived_from: dict[str, list[str]] = field(default_factory=dict)  # condition column -> the metrics it restates
    moran: dict[str, float] = field(default_factory=dict)  # global Moran's I per metric (row-standardised ring-1)
    glossary: dict[str, str] = field(default_factory=dict)  # derived column (or "column=value") -> plain words

    def neighbours(self, row: int) -> np.ndarray:
        return self.nbr_idx[self.nbr_ptr[row] : self.nbr_ptr[row + 1]]


def parse_geo(spec: str, default_resolution: int) -> tuple[str, int]:
    """``"auto"``, ``"<col>"``, ``"<lat>,<lon>"``, each optionally ``@<res>`` -> (locator, resolution)."""
    locator, _, res = spec.strip().partition("@")
    try:
        resolution = int(res) if res.strip() else default_resolution
    except ValueError as exc:
        raise SpatialError(f"Bad geo resolution '{res}' (expected an integer 0-15)") from exc
    if not 0 <= resolution <= 15 or not locator.strip():
        raise SpatialError(f"Bad geo spec '{spec}' (expected auto|<column>|<lat>,<lon> with optional @<0-15>)")
    return locator.strip(), resolution


def parse_points(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """GeoJSON Point or WKT ``POINT (x y)`` strings -> (lat, lon), NaN where a value is not a point."""
    text = values.astype("string")
    geo = text.str.extract(_GEOJSON).where(text.str.contains('"Point"', regex=False).fillna(False), axis=0)
    wkt = text.str.extract(_WKT)
    lon = pd.to_numeric(geo[0].fillna(wkt[0]), errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(geo[1].fillna(wkt[1]), errors="coerce").to_numpy(dtype=float)
    return _in_range(lat, lon)


def _in_range(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bad = ~((np.abs(lat) <= 90) & (np.abs(lon) <= 180))
    lat, lon = lat.copy(), lon.copy()
    lat[bad] = lon[bad] = np.nan
    return lat, lon


def _tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", name.lower()))


def locate_points(frame: pd.DataFrame, locator: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(lat, lon, location columns). ``auto`` takes the first text column whose values parse as points,
    else the numeric pair named like lat/latitude and lon/lng/longitude."""
    if "," in locator:
        names = [c.strip() for c in locator.split(",")]
        missing = [c for c in names if c not in frame.columns]
        if len(names) != 2 or missing:
            raise SpatialError(f"Geo columns not in dataset: {missing or names}")
        lat, lon = (pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float) for c in names)
        return (*_in_range(lat, lon), names)
    if locator != "auto":
        if locator not in frame.columns:
            raise SpatialError(f"Geo column not in dataset: {locator}")
        return (*parse_points(frame[locator]), [locator])
    for col in frame.columns:
        if frame[col].dtype.kind in "OSU" or isinstance(frame[col].dtype, pd.StringDtype):
            sample = frame[col].dropna().head(200)
            if len(sample) and np.isfinite(parse_points(sample)[0]).mean() >= POINT_PARSE_SHARE:
                return (*parse_points(frame[col]), [col])
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    lats = [c for c in numeric if _tokens(str(c)) & LAT_NAMES]
    lons = [c for c in numeric if _tokens(str(c)) & LON_NAMES]
    if lats and lons:
        return locate_points(frame, f"{lats[0]},{lons[0]}")
    raise SpatialError("No point column found (a GeoJSON/WKT point column, or numeric lat/lon columns); name it in the geo option")


def _as_datetime(values: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    if values.dtype.kind in "OSU" or isinstance(values.dtype, pd.StringDtype):
        sample = values.dropna().astype(str).head(200)
        if (
            len(sample)
            and sample.str.len().min() >= 8
            and pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() >= POINT_PARSE_SHARE
        ):
            return pd.to_datetime(values, errors="coerce", format="mixed")
    return None


def _neighbour_graph(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """CSR of ring-1 neighbours present among ``cells`` and, per cell, its ring size (6; 5 at pentagons)."""
    pos = {c: i for i, c in enumerate(cells)}
    ptr, idx, ring = [0], [], []
    for c in cells:
        around = [n for n in h3.grid_ring(c, 1)]
        ring.append(len(around))
        idx += [pos[n] for n in around if n in pos]
        ptr.append(len(idx))
    return np.asarray(ptr, dtype=np.int64), np.asarray(idx, dtype=np.int64), ring


def spatial_lag(values: np.ndarray, ptr: np.ndarray, idx: np.ndarray, ring: np.ndarray | None) -> np.ndarray:
    """Mean of the neighbours' values, the cell itself excluded. ``ring`` given (a counter): absent neighbours
    count as 0 and the mean runs over the whole ring; otherwise over the present, non-NaN neighbours."""
    owner = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    vals = values[idx]
    ok = np.isfinite(vals)
    total = np.bincount(owner[ok], weights=vals[ok], minlength=len(ptr) - 1)
    count = ring.astype(float) if ring is not None else np.bincount(owner[ok], minlength=len(ptr) - 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def moran_i(values: np.ndarray, ptr: np.ndarray, idx: np.ndarray) -> float:
    """Global Moran's I with row-standardised weights over present, non-NaN neighbours; 0 when undefined."""
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return 0.0
    z = np.where(ok, values - values[ok].mean(), np.nan)
    lag = spatial_lag(z, ptr, idx, None)
    use = ok & np.isfinite(lag)
    denom = float((z[ok] ** 2).sum()) / ok.sum()
    if not use.any() or denom <= 0:
        return 0.0
    return float((z[use] * lag[use]).sum() / use.sum() / denom)


def local_moran(values: np.ndarray, ptr: np.ndarray, idx: np.ndarray, permutations: int, seed: int, alpha: float = 0.05) -> np.ndarray:
    """LISA class per cell over the occupied cells (row-standardised ring-1 weights); conditional permutation
    test, folded p <= ``alpha``, else ``NO_CLUSTER``."""
    z = (values - values.mean()) / (values.std() or 1.0)
    n = len(z)
    lag = spatial_lag(z, ptr, idx, None)
    local = z * np.nan_to_num(lag)
    k = np.diff(ptr)
    rng = np.random.default_rng(seed)
    others = rng.integers(0, max(n - 1, 1), size=(n, permutations, int(k.max(initial=1))))
    others += others >= np.arange(n)[:, None, None]  # never draw the cell itself
    mask = np.arange(others.shape[2]) < k[:, None, None]
    perm_lag = (z[np.minimum(others, n - 1)] * mask).sum(axis=2) / np.maximum(k, 1)[:, None]
    extreme = (np.abs(z[:, None] * perm_lag) >= np.abs(local)[:, None]).sum(axis=1)
    p = (extreme + 1) / (permutations + 1)
    labels = np.array([LISA_LABELS[(bool(a > 0), bool(b > 0))] for a, b in zip(z, np.nan_to_num(lag))], dtype=object)
    return np.where((p <= alpha) & (k > 0), labels, NO_CLUSTER)


def build_cells(
    frame: pd.DataFrame,
    locator: str,
    resolution: int,
    *,
    min_cell_points: int,
    share_max_levels: int,
    mode_max_levels: int,
    lisa_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, SpatialContext, list[str]]:
    """Source rows -> (cell table, spatial context, warnings)."""
    lat, lon, location = locate_points(frame, locator)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        raise SpatialError(f"No valid point in {location}")
    ids = np.array([h3.latlng_to_cell(a, b, resolution) for a, b in zip(lat[ok], lon[ok])], dtype=object)
    codes, cells = pd.factorize(ids, sort=True)
    cells = np.asarray(cells, dtype=object)
    point_cell = np.full(len(frame), -1, dtype=np.int64)
    point_cell[ok] = codes
    counts = np.bincount(codes, minlength=len(cells)).astype(float)
    dense = counts >= min_cell_points
    warnings = [f"geo: {int((~ok).sum())} rows without a valid point ignored"] if (~ok).any() else []

    out: dict[str, np.ndarray] = {COUNTER: counts}
    glossary = {COUNTER: "number of events in the map cell"}
    counters = [COUNTER]
    dropped: list[str] = []
    categorical: dict[str, pd.Series] = {}
    rows = frame.loc[ok].drop(columns=location)
    for col in rows.columns:
        values = rows[col]
        times = _as_datetime(values)
        if times is not None:
            t = times.to_numpy(dtype="datetime64[ns]")
            good = ~np.isnat(t)
            mid = t[good].min() + (t[good].max() - t[good].min()) / 2
            out[f"{col}_active_days"] = (
                pd.Series(t.astype("datetime64[D]")).groupby(codes).nunique().reindex(range(len(cells)), fill_value=0).to_numpy(float)
            )
            counters.append(f"{col}_active_days")
            glossary[f"{col}_active_days"] = f"number of different days with events in the cell (by {_words(col)})"
            glossary[f"{col}_late_share"] = f"share of the cell's events that happened in the second half of the period (by {_words(col)})"
            late = pd.Series(np.where(good, t > mid, np.nan)).groupby(codes).mean().reindex(range(len(cells))).to_numpy(float)
            out[f"{col}_late_share"] = np.where(dense, late, np.nan)
        elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            med = values.groupby(codes).median().reindex(range(len(cells))).to_numpy(float)
            out[f"{col}_median"] = np.where(dense, med, np.nan)
            glossary[f"{col}_median"] = f"typical (median) {_words(col)} of the cell's events"
        else:
            text = values.astype("string")
            levels = text.dropna().value_counts()
            if 2 <= len(levels) <= mode_max_levels:
                categorical[str(col)] = text
            else:
                dropped.append(str(col))
    mode_cols, share_cols = _categorical_roles(categorical, codes, len(cells), dense, share_max_levels)
    nested: dict[str, list[str]] = {}  # mode column -> share columns it is (largely) nested with
    for col in mode_cols:
        if any(_nesting(categorical[col], categorical[o]) >= FUNCTIONAL_SHARE for o in share_cols):
            dropped.append(col)  # its mode would explain those shares by construction
            continue
        nested[col] = [o for o in share_cols if _nesting(categorical[col], categorical[o]) >= NESTED_SHARE]
        table = _level_counts(categorical[col], codes, len(cells))
        total = table.sum(axis=1).to_numpy(float)
        out[f"{col}_mode"] = np.where(total > 0, table.columns.to_numpy()[table.to_numpy().argmax(axis=1)], "missing")
        glossary[f"{col}_mode"] = f"the most frequent {_words(col)} among the cell's events"
    for col in share_cols:
        table = _level_counts(categorical[col], codes, len(cells))
        total = table.sum(axis=1).to_numpy(float)
        kept = table.columns[-1:] if table.shape[1] == 2 else table.columns  # a binary column: its rarer level
        for level in kept:
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(dense & (total > 0), table[level].to_numpy(float) / total, np.nan)
            if np.nanmean((share > 0) & (share < 1)) >= MIN_MIXED_SHARE:
                out[f"share_{col}_{level}"] = share
                glossary[f"share_{col}_{level}"] = f"share of the cell's events whose {_words(col)} is {level}" + _kind_of(col, level, categorical)
    if dropped:
        warnings.append(f"geo: columns not aggregated (too many or too few levels, or determining a share column): {dropped}")

    ptr, idx, ring = _neighbour_graph(cells)
    ring_arr = np.asarray(ring)
    metrics = [c for c, v in out.items() if v.dtype.kind == "f"]
    lag_columns, derived = [], {}
    for metric in counters:  # ponytail: counters only; intensive-metric lags when a dataset needs them
        name = f"neighbours_{metric}"
        lag = spatial_lag(out[metric], ptr, idx, ring_arr)
        if len(np.unique(lag[np.isfinite(lag)])) < 2:
            continue
        out[name] = lag
        lag_columns.append(name)
        band, by = f"{name}_band", glossary[metric]
        glossary[band] = f"how busy the six surrounding cells are ({by}), in quarters of all cells"
        for q, words in QUARTERS.items():
            glossary[f"{band}={q}"] = f"the surrounding cells are {words} ({by})"
        glossary[f"{band}=missing"] = "no surrounding cell has events"
        derived[f"{name}_band"] = counters  # every counter measures the same activity
    neighbourhood = [*derived, LISA_COLUMN]
    for col, others in nested.items():
        if others:
            derived[f"{col}_mode"] = [m for o in others for m in out if m.startswith(f"share_{o}_")]
    out[LISA_COLUMN] = local_moran(np.log1p(counts), ptr, idx, lisa_permutations, seed)
    derived[LISA_COLUMN] = counters
    glossary[LISA_COLUMN] = "whether the cell and its neighbours form a cluster of many or few events"
    glossary.update({f"{LISA_COLUMN}={k}": v for k, v in LISA_WORDS.items()})
    table = pd.DataFrame(out)
    ctx = SpatialContext(
        resolution=resolution,
        location=",".join(location),
        cells=cells,
        point_cell=point_cell,
        lonlat=np.column_stack([lon, lat]).astype(np.float32),
        nbr_ptr=ptr,
        nbr_idx=idx,
        min_cell_points=min_cell_points,
        sparse_cells=int((~dense).sum()),
        lag_columns=lag_columns,
        conditions=neighbourhood,
        derived_from=derived,
        moran={m: moran_i(out[m], ptr, idx) for m in metrics},
        glossary={k: v for k, v in glossary.items() if k.split("=", 1)[0] in table.columns or k.split("=", 1)[0] in derived},
    )
    return table, ctx, warnings


def _words(name: str) -> str:
    return str(name).replace("_", " ")


def _kind_of(col: str, level: str, categorical: dict[str, pd.Series]) -> str:
    """' (all of them: <column> <value>, ...)' for coarser columns whose value is fixed among ``level``'s rows."""
    rows = (categorical[col] == level).fillna(False).to_numpy(bool)
    hints = []
    for other, text in categorical.items():
        if other == col or text.nunique() >= categorical[col].nunique():
            continue
        top = text[rows].value_counts(normalize=True)
        if len(top) and top.iloc[0] >= FUNCTIONAL_SHARE:
            hints.append(f"{_words(other)} {top.index[0]}")
    return f" (every such event also has: {', '.join(hints[:3])})" if hints else ""


def _level_counts(text: pd.Series, codes: np.ndarray, n_cells: int) -> pd.DataFrame:
    """Points per (cell, level); columns ordered by overall frequency."""
    order = text.value_counts().index
    return pd.crosstab(codes, text.to_numpy()).reindex(index=range(n_cells), columns=order, fill_value=0)


def _categorical_roles(
    columns: dict[str, pd.Series], codes: np.ndarray, n_cells: int, dense: np.ndarray, share_max_levels: int
) -> tuple[list[str], list[str]]:
    """(mode columns, share columns). A column whose dense cells are almost pure (mean top-level share >=
    ``PLACE_PURITY``) describes the place: its mode is a condition. A mixed column with few levels becomes
    per-level share metrics; a mixed column with more levels a mode condition."""
    modes, shares = [], []
    for col, text in columns.items():
        table = _level_counts(text, codes, n_cells).to_numpy(float)[dense]
        total = table.sum(axis=1)
        purity = float(np.mean(table.max(axis=1)[total > 0] / total[total > 0])) if (total > 0).any() else 1.0
        if purity < PLACE_PURITY and table.shape[1] <= share_max_levels:
            shares.append(col)
        else:
            modes.append(col)
    return modes, shares


def _nesting(a: pd.Series, b: pd.Series) -> float:
    """Share of rows on which one column's level fixes the other's (the larger of both directions)."""
    both = a.notna() & b.notna()
    if not both.any():
        return 0.0
    table = pd.crosstab(a[both].to_numpy(), b[both].to_numpy()).to_numpy()
    return float(max(table.max(axis=1).sum(), table.max(axis=0).sum()) / table.sum())


def compactness(rows: np.ndarray, ctx: SpatialContext) -> float:
    """Share of the extent's neighbour links that stay inside the extent (1 = one solid block, 0 = isolated cells)."""
    inside = np.zeros(len(ctx.cells), dtype=bool)
    inside[rows] = True
    links = np.concatenate([ctx.neighbours(r) for r in rows]) if len(rows) else np.empty(0, dtype=np.int64)
    return float(inside[links].mean()) if len(links) else 0.0


def colocated(extents: dict[str, np.ndarray], ctx: SpatialContext, min_overlap: float) -> list[tuple[str, str, float]]:
    """Pairs of extents whose one-ring dilations overlap by ``|A ∩ B| / min(|A|, |B|) >= min_overlap``."""
    ids = sorted(extents)
    if len(ids) < 2:
        return []
    grown = np.zeros((len(ids), len(ctx.cells)), dtype=np.float32)
    for i, pid in enumerate(ids):
        rows = extents[pid]
        grown[i, rows] = 1
        if len(rows):
            grown[i, np.concatenate([ctx.neighbours(r) for r in rows])] = 1
    inter = grown @ grown.T
    size = grown.sum(axis=1)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            overlap = float(inter[i, j] / max(min(size[i], size[j]), 1))
            if overlap >= min_overlap:
                pairs.append((ids[i], ids[j], round(overlap, 4)))
    return pairs


def cell_boundary(cell: str) -> list[list[float]]:
    """Closed GeoJSON ring ([lon, lat] pairs) of one cell."""
    ring = [[round(lng, 6), round(lat, 6)] for lat, lng in h3.cell_to_boundary(cell)]
    return ring + ring[:1]


def effective_n_factor(rho: float) -> float:
    """n_eff / n for a metric with spatial autocorrelation ``rho`` (Clifford, Richardson & Hémon 1989, one-parameter
    approximation n_eff = n (1 - rho) / (1 + rho)); no correction for rho <= 0."""
    rho = min(max(rho, 0.0), 0.99)
    return (1.0 - rho) / (1.0 + rho)


def annotate_extents(insights: list[Insight], covers: dict[str, np.ndarray], ctx: SpatialContext, min_overlap: float) -> None:
    """Add ``compactness`` and the ``colocated`` partners to each insight's ``provenance["spatial"]`` (in place)."""
    extents = {i.id: covers[i.expression] for i in insights}
    partners: dict[str, list[dict[str, float | str]]] = {i.id: [] for i in insights}
    for a, b, overlap in colocated(extents, ctx, min_overlap):
        partners[a].append({"pattern_id": b, "overlap": overlap})
        partners[b].append({"pattern_id": a, "overlap": overlap})
    for found in partners.values():
        found.sort(key=lambda x: (-x["overlap"], x["pattern_id"]))
        del found[COLOCATED_TOP:]
    for ins in insights:
        spatial = ins.provenance.setdefault("spatial", {"resolution": ctx.resolution, "cells": len(extents[ins.id])})
        spatial["compactness"] = round(compactness(extents[ins.id], ctx), 4)
        spatial["colocated"] = partners[ins.id]


def sac_summary(insights: list[Insight], alpha: float) -> dict[str, int]:
    """How many validated insights pass the adjusted-p rule with and without the n_eff correction."""
    spatial = [i for i in insights if "spatial" in i.provenance and i.p_adjusted is not None]
    return {
        "validated": len(spatial),
        "significant_iid": sum(i.provenance["spatial"]["p_adjusted_iid"] <= alpha for i in spatial),
        "significant_corrected": sum(i.p_adjusted <= alpha for i in spatial),
    }
