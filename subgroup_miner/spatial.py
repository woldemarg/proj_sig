"""Point data -> a table of H3 cells with spatial conditions (docs/02_discovery.md §2.9).

One row per occupied H3 cell. Column roles follow the source dtypes only, never column names:
counters (``points``, ``<time>_active_days``) for every cell; intensive metrics (medians, shares,
``<time>_late_share``) only for cells with at least ``min_cell_points`` points (NaN otherwise);
categorical modes and neighbourhood conditions as text dimensions.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import h3
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import false_discovery_control, norm

from insight_contracts import Insight, region_node_id

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
GEO_SHARE_MAX_LEVELS = 8  # categoricals with at most this many levels also become per-level share metrics
GEO_MODE_MAX_LEVELS = 40  # categoricals with more levels are not aggregated
GEO_COLOCATION_MIN = 0.3  # CO_LOCATED: overlap of the one-ring-dilated extents
COLOCATED_TOP = 3  # CO_LOCATED partners per pattern, of its own dataset and of the others (the strongest relations)
GEO_REGION_MIN_SHARE = 0.2  # IN_REGION: |P ∩ R| over the smaller of the pattern and the region's occupied cells
MIN_MIXED_SHARE = 0.1  # a share metric needs this share of dense cells strictly between 0 and 1
GEO_AUTO_RESOLUTIONS = range(10, 2, -1)  # the resolutions ``@auto`` weighs, finest first
GEO_DENSE_SHARE_MIN = 0.3  # ``@auto``: the finest level needs this share of cells with at least min_cell_points points
GEO_LEVELS_MAX = 3  # ``@auto``: levels chosen, the finest included


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
    blocks: np.ndarray  # per cell row, its parent cell (one resolution coarser) as an int: the bootstrap's resampling unit
    min_cell_points: int
    sparse_cells: int = 0
    lag_columns: list[str] = field(default_factory=list)  # numeric neighbourhood columns, banded by ingestion
    conditions: list[str] = field(default_factory=list)  # the neighbourhood dimensions, always searched
    derived_from: dict[str, list[str]] = field(default_factory=dict)  # condition column -> the metrics it restates
    moran: dict[str, float] = field(default_factory=dict)  # global Moran's I per metric (row-standardised ring-1): a description of the map
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


def parse_geo_levels(spec: str, default_resolution: int) -> list[tuple[str, int]]:
    """``<locator>@<r1>,<r2>,...`` -> one (locator, resolution) per level, finest first; without ``@`` one level."""
    locator, _, res = spec.strip().partition("@")
    if res.strip() and not all(r.strip() for r in res.split(",")):
        raise SpatialError(f"Bad geo spec '{spec}' (an empty resolution in the list)")
    levels = [parse_geo(f"{locator}@{r}" if r.strip() else locator, default_resolution) for r in res.split(",")]
    if len({r for _, r in levels}) < len(levels):
        raise SpatialError(f"Bad geo spec '{spec}' (a resolution is listed twice)")
    return sorted(levels, key=lambda level: -level[1])


def select_levels(lat: np.ndarray, lon: np.ndarray, region_levels: tuple[int, ...], *, min_rows: int, min_cell_points: int) -> dict[str, Any]:
    """The resolutions ``@auto`` picks for these points (docs/02_discovery.md §2.9): the finest of
    ``GEO_AUTO_RESOLUTIONS`` with at least ``min_rows`` cells of which ``GEO_DENSE_SHARE_MIN`` hold ``min_cell_points``
    points, then the coarser ``region_levels`` that still have ``min_rows`` cells, at most ``GEO_LEVELS_MAX`` in all.
    Returns ``{chosen, candidates: {res: {cells, dense_share, median_points}}, reasons: {res: why}}``, keyed by the
    resolution as text (as JSON stores it)."""
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        raise SpatialError("No valid point for @auto")
    cells = pd.Series([h3.latlng_to_cell(a, b, GEO_AUTO_RESOLUTIONS[0]) for a, b in zip(lat[ok], lon[ok])]).value_counts()
    candidates = {}
    for res in GEO_AUTO_RESOLUTIONS:  # each level from the one finer: a parent's parent is the coarser parent
        cells = cells.groupby([h3.cell_to_parent(c, res) for c in cells.index]).sum()
        candidates[str(res)] = {
            "cells": len(cells),
            "dense_share": round(float((cells >= min_cell_points).mean()), 4),
            "median_points": float(cells.median()),
        }
    reasons, chosen = {}, []
    for res, c in candidates.items():
        if c["cells"] < min_rows:
            reasons[res] = f"{c['cells']} cells < MIN_ROWS {min_rows}"
        elif not chosen and c["dense_share"] < GEO_DENSE_SHARE_MIN:
            reasons[res] = f"dense share {c['dense_share']} < {GEO_DENSE_SHARE_MIN}"
        elif chosen and int(res) not in region_levels:
            reasons[res] = "not a GEO_REGION_LEVELS level"
        elif len(chosen) == GEO_LEVELS_MAX:
            reasons[res] = f"GEO_LEVELS_MAX {GEO_LEVELS_MAX} reached"
        else:
            reasons[res] = "finest level" if not chosen else "region level"
            chosen.append(int(res))
    if not chosen:
        raise SpatialError("no resolution fits @auto: " + "; ".join(f"{res}: {why}" for res, why in reasons.items()))
    return {"chosen": chosen, "candidates": candidates, "reasons": reasons}


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


def local_moran(values: np.ndarray, ptr: np.ndarray, idx: np.ndarray, q: float = 0.05) -> np.ndarray:
    """LISA class per cell over the occupied cells (row-standardised ring-1 weights). Each cell's local I is scored
    against its conditional-randomisation null — the cell fixed, its k neighbours' values drawn without replacement
    from the other n − 1 cells (Anselin 1995) — with that null's exact mean and variance (Sokal, Oden & Thomson 1998)
    and a normal approximation, so the p-values have no floor whatever the map's size; they are cut at false-discovery
    rate ``q`` over the cells that have neighbours (one test per cell is a family), the survivors labelled by their
    quadrant, every other cell ``NO_CLUSTER``."""
    n = len(values)
    z = (values - values.mean()) / (values.std() or 1.0)  # population scaling: sum(z^2) = n
    k = np.diff(ptr)
    lag = np.nan_to_num(spatial_lag(z, ptr, idx, None))
    tested = np.flatnonzero(k > 0)
    # under the null the lag is the mean of k draws without replacement from the other n - 1 standardised values
    mean = -z / max(n - 1, 1)
    var = ((n - z**2) / max(n - 1, 1) - mean**2) / np.maximum(k, 1) * (n - 1 - k) / max(n - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.sign(z) * (lag - mean) / np.sqrt(var)  # (I - E[I]) / sd[I] for I = z * lag: |z| cancels
    p = 2.0 * norm.sf(np.abs(np.nan_to_num(score[tested])))
    labels = np.array([LISA_LABELS[(bool(a > 0), bool(b > 0))] for a, b in zip(z, lag)], dtype=object)
    keep = np.zeros(n, dtype=bool)
    if len(tested):
        keep[tested[false_discovery_control(p) <= q]] = True  # Benjamini-Hochberg over the tested cells
    return np.where(keep, labels, NO_CLUSTER)


def induced_links(rows: np.ndarray, ctx: SpatialContext) -> sparse.csr_array:
    """The ring-1 links among ``rows`` as a CSR over the subgroup's own rows: what stays inside the subgroup."""
    n = len(ctx.cells)
    return sparse.csr_array((np.ones(len(ctx.nbr_idx)), ctx.nbr_idx, ctx.nbr_ptr), shape=(n, n))[rows][:, rows]


def subgroup_moran(values: np.ndarray, rows: np.ndarray, ctx: SpatialContext) -> float:
    """Moran's I of ``values`` (one per row of ``rows``) over the subgroup's own cells and the ring-1 links that stay
    inside it: the dependence the median test must discount. Scattered cells have no inside links and give 0."""
    w = induced_links(rows, ctx)
    return moran_i(np.asarray(values, dtype=float), w.indptr, w.indices)


def build_cells(
    frame: pd.DataFrame,
    locator: str,
    resolution: int,
    *,
    min_cell_points: int,
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
            if 2 <= len(levels) <= GEO_MODE_MAX_LEVELS:
                categorical[str(col)] = text
            else:
                dropped.append(str(col))
    mode_cols, share_cols = _categorical_roles(categorical, codes, len(cells), dense)
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
    blocks = pd.factorize(np.array([h3.cell_to_parent(c, resolution - 1) if resolution else c for c in cells], dtype=object))[0]
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
    out[LISA_COLUMN] = local_moran(np.log1p(counts), ptr, idx)
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
        blocks=blocks,
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


def _categorical_roles(columns: dict[str, pd.Series], codes: np.ndarray, n_cells: int, dense: np.ndarray) -> tuple[list[str], list[str]]:
    """(mode columns, share columns). A column whose dense cells are almost pure (mean top-level share >=
    ``PLACE_PURITY``) describes the place: its mode is a condition. A mixed column with few levels becomes
    per-level share metrics; a mixed column with more levels a mode condition."""
    modes, shares = [], []
    for col, text in columns.items():
        table = _level_counts(text, codes, n_cells).to_numpy(float)[dense]
        total = table.sum(axis=1)
        purity = float(np.mean(table.max(axis=1)[total > 0] / total[total > 0])) if (total > 0).any() else 1.0
        if purity < PLACE_PURITY and table.shape[1] <= GEO_SHARE_MAX_LEVELS:
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


def compactness(rows: np.ndarray, nbr_ptr: np.ndarray, nbr_idx: np.ndarray) -> float:
    """Share of the extent's neighbour links that stay inside the extent (1 = one solid block, 0 = isolated cells)."""
    inside = np.zeros(len(nbr_ptr) - 1, dtype=bool)
    inside[rows] = True
    owner = np.repeat(np.arange(len(nbr_ptr) - 1), np.diff(nbr_ptr))
    total = int(np.diff(nbr_ptr)[rows].sum())
    return int((inside[owner] & inside[nbr_idx]).sum()) / total if total else 0.0


@dataclass
class GeoDataset:
    """What ``derive_spatial`` reads of one geo dataset: its ``geo.npz`` arrays and its patterns' cell rows (``covers``)."""

    dataset_id: str
    cells: np.ndarray  # H3 id per cell row
    nbr_ptr: np.ndarray  # CSR over cell rows: ring-1 neighbours present in the dataset
    nbr_idx: np.ndarray
    covers: dict[str, np.ndarray]  # pattern id -> its extent's cell rows


@dataclass
class SpatialStructure:
    """The derived spatial plane of the committed geo patterns (docs/06_graph_and_storage.md §6.1)."""

    regions: dict[str, dict[str, int | str]] = field(default_factory=dict)  # R:<res>:<h3> -> resolution, h3, cells_covered, patterns
    in_region: list[tuple[str, str, float]] = field(default_factory=list)  # (pattern, region, share of the region's cells it covers)
    parent_of: list[tuple[str, str]] = field(default_factory=list)  # (region, its parent at the next coarser listed level)
    co_located: list[tuple[str, str, str, float]] = field(default_factory=list)  # (a, b, relation, weight), a < b
    compactness: dict[str, float] = field(default_factory=dict)
    occupied: dict[str, set[str]] = field(default_factory=dict)  # dataset -> the regions above that hold its cells


RELATIONS = ("overlaps", "adjacent", "same_region")  # strongest first


def derive_spatial(datasets: list[GeoDataset], levels: tuple[int, ...]) -> SpatialStructure:
    """Regions, IN_REGION, PARENT_OF, CO_LOCATED with its relation and compactness, from the stored cells and extents.

    A pattern is IN_REGION of a region at every listed level not finer than its dataset (its own level included) when
    their overlap reaches ``GEO_REGION_MIN_SHARE`` of the smaller of its cells and the region's occupied cells (a focus inside
    a region scores 1, as does a pattern covering the region); ``occupied`` keeps, per dataset, those of these regions
    that hold its cells. Two patterns of one resolution, of any
    datasets, are CO_LOCATED: ``overlaps`` when their one-ring-dilated extents overlap by at least ``GEO_COLOCATION_MIN``
    of the smaller, else ``adjacent`` when the extents touch or share cells, else ``same_region`` when both are
    IN_REGION of one region at the nearest listed level at least two coarser. Each pattern keeps its ``COLOCATED_TOP``
    strongest partners (relation first, then weight) of its own dataset and as many of the other datasets; a pair that
    either side keeps is an edge."""
    out = SpatialStructure()
    levels = tuple(sorted(set(levels), reverse=True))
    raw, grown, ring, res_of, dataset_of, shares = {}, {}, {}, {}, {}, {}
    covered: Counter[str] = Counter()
    for d in datasets:
        if not len(d.cells):
            continue
        res = h3.get_resolution(str(d.cells[0]))
        own = [lv for lv in levels if lv <= res]
        parents = {lv: np.array([c if lv == res else h3.cell_to_parent(c, lv) for c in map(str, d.cells)]) for lv in own}
        occupied = {lv: dict(zip(*np.unique(parents[lv], return_counts=True), strict=True)) for lv in own}
        out.occupied[d.dataset_id] = {region_node_id(lv, str(c)) for lv in own for c in occupied[lv]}
        extents = [np.asarray(r, dtype=np.int64) for r in d.covers.values()]
        union = np.unique(np.concatenate(extents)) if extents else np.empty(0, dtype=np.int64)
        for lv in own:  # each dataset's cells inside a region that some pattern covers
            for cell, n in zip(*np.unique(parents[lv][union], return_counts=True), strict=True):
                covered[region_node_id(lv, str(cell))] += int(n)
        for pid, rows in d.covers.items():
            rows = np.unique(np.asarray(rows, dtype=np.int64))
            res_of[pid], dataset_of[pid] = res, d.dataset_id
            raw[pid] = {str(c) for c in d.cells[rows]}
            near = np.concatenate([rows, *(d.nbr_idx[d.nbr_ptr[r] : d.nbr_ptr[r + 1]] for r in rows)]) if len(rows) else rows
            grown[pid] = {str(c) for c in d.cells[near]}
            ring[pid] = {n for c in raw[pid] for n in h3.grid_ring(c, 1)}
            out.compactness[pid] = round(compactness(rows, d.nbr_ptr, d.nbr_idx), 4)
            for lv in own:
                regions, counts = np.unique(parents[lv][rows], return_counts=True)
                for cell, n in zip(regions, counts, strict=True):
                    rid = region_node_id(lv, str(cell))
                    share = float(n / min(len(rows), occupied[lv][cell]))
                    if share >= GEO_REGION_MIN_SHARE:  # the threshold reads the exact share, the edge stores it rounded
                        share = round(share, 4)
                        out.in_region.append((pid, rid, share))
                        shares[(pid, rid)] = share
                        for up in [x for x in levels if x <= lv]:  # the region and its ancestors at the coarser listed levels
                            up_cell = str(cell) if up == lv else h3.cell_to_parent(str(cell), up)
                            out.regions.setdefault(region_node_id(up, up_cell), {"resolution": up, "h3": up_cell})
    members = Counter(rid for _, rid, _ in out.in_region)
    out.occupied = {ds: rids & members.keys() for ds, rids in out.occupied.items()}
    for rid, props in out.regions.items():
        props.update(cells_covered=covered[rid], patterns=members[rid])
        coarser = [x for x in levels if x < props["resolution"]]
        if coarser:
            out.parent_of.append((rid, region_node_id(coarser[0], h3.cell_to_parent(props["h3"], coarser[0]))))
    regions_of: defaultdict[str, set[str]] = defaultdict(set)
    for pid, rid in shares:
        regions_of[pid].add(rid)
    by_res: defaultdict[int, list[str]] = defaultdict(list)
    for pid in sorted(raw):
        by_res[res_of[pid]].append(pid)
    links: dict[tuple[str, str], tuple[str, float]] = {}
    for res, pids in by_res.items():
        level = next((lv for lv in levels if res - lv >= 2), None)
        for i, a in enumerate(pids):
            for b in pids[i + 1 :]:
                overlap = len(grown[a] & grown[b]) / max(min(len(grown[a]), len(grown[b])), 1)
                if overlap >= GEO_COLOCATION_MIN:
                    links[(a, b)] = ("overlaps", round(overlap, 4))
                elif raw[a] & raw[b] or ring[a] & raw[b]:  # share of both extents' cells on the common edge
                    edge = len((raw[a] | ring[a]) & raw[b]) + len((raw[b] | ring[b]) & raw[a])
                    links[(a, b)] = ("adjacent", round(edge / (len(raw[a]) + len(raw[b])), 4))
                else:
                    common = [r for r in regions_of[a] & regions_of[b] if level is not None and r.startswith(f"R:{level}:")]
                    if common:
                        links[(a, b)] = ("same_region", max(min(shares[(a, r)], shares[(b, r)]) for r in common))
    pool: defaultdict[tuple[str, bool], list[tuple[int, float, str]]] = defaultdict(list)  # (pattern, across datasets)
    for (a, b), (relation, weight) in links.items():
        cross = dataset_of[a] != dataset_of[b]
        pool[(a, cross)].append((RELATIONS.index(relation), -weight, b))
        pool[(b, cross)].append((RELATIONS.index(relation), -weight, a))
    top = {key: {p for _, _, p in sorted(found)[:COLOCATED_TOP]} for key, found in pool.items()}
    out.co_located = [
        (a, b, relation, weight)
        for (a, b), (relation, weight) in sorted(links.items())
        if b in top[(a, dataset_of[a] != dataset_of[b])] or a in top[(b, dataset_of[a] != dataset_of[b])]
    ]
    out.in_region.sort()
    out.parent_of.sort()
    return out


def parse_zone(cells: list[str]) -> frozenset[str]:
    """A map zone given as H3 cells, of any resolutions; ``SpatialError`` names the first that is not a cell."""
    if not cells:
        raise SpatialError("the zone holds no cell")
    if bad := next((c for c in cells if not (isinstance(c, str) and h3.is_valid_cell(c))), None):
        raise SpatialError(f"'{bad}' is not an H3 cell")
    return frozenset(cells)


# ponytail: no size cap; a continent at resolution 10 takes minutes. Cap by the shape's area before exposing it beyond localhost
def geometry_to_cells(geometry: dict[str, Any], resolution: int) -> frozenset[str]:
    """The cells at ``resolution`` whose centres lie in ``geometry``: a GeoJSON Polygon, MultiPolygon, Feature or
    FeatureCollection, ``{"center": [lon, lat], "radius_m": r}`` or ``{"bbox": [west, south, east, north]}``. A shape that
    holds no cell centre gives the cells under its vertices (a circle: the cell under its centre)."""
    try:
        if "bbox" in geometry:
            w, s, e, n = map(float, geometry["bbox"])
            geometry = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        if "center" in geometry:
            lon, lat = map(float, geometry["center"])
            radius, centre = float(geometry["radius_m"]), h3.latlng_to_cell(lat, lon, resolution)
            k = int(radius / (1.5 * h3.average_hexagon_edge_length(resolution, "m"))) + 2  # ring k lies at least 1.5·k edges away
            inside = {c for c in h3.grid_disk(centre, k) if h3.great_circle_distance(h3.cell_to_latlng(c), (lat, lon), "m") <= radius}
            return frozenset(inside or {centre})
        kind = geometry.get("type")
        if kind == "FeatureCollection":
            return frozenset().union(*(geometry_to_cells(f, resolution) for f in geometry["features"]))
        if kind == "Feature":
            return geometry_to_cells(geometry["geometry"], resolution)
        if kind not in ("Polygon", "MultiPolygon"):
            raise SpatialError(f"geometry type {kind!r} is not a Polygon, MultiPolygon, Feature, FeatureCollection, center or bbox")
        cells = set(h3.geo_to_cells(geometry, resolution))
        if not cells:
            rings = geometry["coordinates"] if kind == "Polygon" else [r for poly in geometry["coordinates"] for r in poly]
            cells = {h3.latlng_to_cell(float(lat), float(lon), resolution) for ring in rings for lon, lat in ring}
        return frozenset(cells)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:  # h3's errors are ValueErrors, as is SpatialError
        raise SpatialError(str(exc) if isinstance(exc, SpatialError) else f"unreadable geometry: {type(exc).__name__}: {exc}") from exc


def zone_overlap(cells: np.ndarray, covers: dict[str, np.ndarray], zone: frozenset[str]) -> dict[str, float]:
    """How much each pattern and ``zone`` are one place: ``|P ∩ Z| / sqrt(|P| · |Z|)`` over one dataset's cells (``geo.npz``)
    and extents (``covers``), ``Z`` the dataset's cells inside the zone (a zone cell finer than the dataset counts its parent,
    a coarser one every dataset cell below it). A focus that is the zone scores 1, as does a pattern that is; a pattern far
    larger or smaller than the zone scores little; one outside it scores 0."""
    if not len(cells):
        return {}
    res, names = h3.get_resolution(str(cells[0])), cells.astype(str)
    by_res: defaultdict[int, set[str]] = defaultdict(set)
    for c in zone:
        by_res[h3.get_resolution(c)].add(c)
    inside = np.zeros(len(names), dtype=bool)
    for z, group in by_res.items():
        if z >= res:
            inside |= np.isin(names, [h3.cell_to_parent(c, res) for c in group])
        else:
            inside |= np.isin([h3.cell_to_parent(c, z) for c in names], list(group))
    out, size = {}, int(inside.sum())
    for pid, rows in covers.items():
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        hit = int(inside[rows].sum())
        out[pid] = round(hit / float(np.sqrt(len(rows) * size)), 4) if hit else 0.0
    return out


def cell_boundary(cell: str) -> list[list[float]]:
    """Closed GeoJSON ring ([lon, lat] pairs) of one cell."""
    ring = [[round(lng, 6), round(lat, 6)] for lat, lng in h3.cell_to_boundary(cell)]
    return ring + ring[:1]


def effective_n_factor(rho: float) -> float:
    """n_eff / n for a sample with spatial autocorrelation ``rho`` (Clifford, Richardson & Hémon 1989, one-parameter
    approximation n_eff = n (1 - rho) / (1 + rho)); no correction for rho <= 0. ``rho`` is the subgroup's own
    (``subgroup_moran``), so a solid block is discounted and scattered cells are not."""
    rho = min(max(rho, 0.0), 0.99)
    return (1.0 - rho) / (1.0 + rho)


def sac_summary(insights: list[Insight], alpha: float) -> dict[str, int]:
    """How many validated insights pass the adjusted-p rule with and without the n_eff correction."""
    spatial = [i for i in insights if "spatial" in i.provenance and i.p_adjusted is not None]
    return {
        "validated": len(spatial),
        "significant_iid": sum(i.provenance["spatial"]["p_adjusted_iid"] <= alpha for i in spatial),
        "significant_corrected": sum(i.p_adjusted <= alpha for i in spatial),
    }
