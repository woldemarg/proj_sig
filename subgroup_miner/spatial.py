"""Point data -> a table of H3 cells with spatial conditions (docs/02_discovery.md §2.9).

One row per occupied H3 cell. Column roles follow the source dtypes only, never column names:
counters (``points``, ``<time>_active_days``) for every cell; intensive metrics (medians, shares,
``<time>_late_share``) only for cells with at least ``min_cell_points`` points (NaN otherwise);
categorical modes and neighbourhood conditions as text dimensions.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import h3
import numpy as np
import pandas as pd
from pandas.tseries.api import guess_datetime_format
from scipy import sparse
from scipy.stats import false_discovery_control, norm

from insight_contracts import Insight
from subgroup_miner.context import ContextLayer, ContextSource, context_bands

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
GEO_DENSE_POINTS_MIN = 0.5  # ``@auto``: the finest level needs this share of the points in cells with at least min_cell_points
GEO_AUTO_MAX_CELLS = 5_000  # ``@auto``: the finest level's cells at most, counted at its region level (docs/02 §2.9)
GEO_AUTO_MIN_CELLS = 500  # ``@auto``: a coarser level's cells at least (the miner's least subgroup, 5 cells per metric, fills a smaller one)
GEO_LEVELS_MAX = 3  # ``@auto``: levels chosen, the finest included
GEO_FOOTPRINT_SHARE = 0.8  # geo_coverage's footprint of a base insight: the busiest cells holding this share of its points
GEO_LOCATED_MIN = 0.8  # a base insight with fewer of its events located is partial, without a place (docs/06 §6.1)
GEO_LIFT_MIN = 1.5  # a base insight's region by over-representation: its share there over the map's
GEO_LIFT_MIN_REGIONS = 12  # a level of fewer occupied regions places a base insight by share, one of more by lift
GEO_LINK_MIN_EVENTS = 30  # a base insight's events in its region at least (a small insight: 5 % of them, 5 at least)
GEO_LINK_FDR = 0.05  # Benjamini-Hochberg over the level's occupied regions, for the lift rule


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
    context: dict[str, ContextLayer] = field(default_factory=dict)  # layer name -> its counts and disks (context.py), the null's input; never stored


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
    ``GEO_AUTO_RESOLUTIONS`` with at least ``min_rows`` dense cells (of at least ``min_cell_points`` points), which hold
    ``GEO_DENSE_POINTS_MIN`` of the points, and at most ``GEO_AUTO_MAX_CELLS`` cells (``cap_cells``): those of the finest of ``region_levels`` not finer
    than it, where its patterns' IN_REGION edges lie, or its own when it is finer than every region level or below them
    (the coarsest such level when every one has more), then the coarser ``region_levels`` that keep
    ``GEO_AUTO_MIN_CELLS`` cells, at most ``GEO_LEVELS_MAX`` in all.
    Returns ``{chosen, capped, candidates: {res: {cells, dense_cells, dense_points, median_points, cap_cells}}, reasons: {res: why}}``,
    ``capped`` the finer resolutions left out by the cap alone, finest first; keyed by the resolution as text (as JSON
    stores it)."""
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        raise SpatialError("No valid point for @auto")
    lat, lon, counts = lat[ok], lon[ok], {}
    finest_region = max(region_levels, default=-1)
    region_of = {res: max((lv for lv in region_levels if lv <= res), default=None) for res in GEO_AUTO_RESOLUTIONS}
    cap_of = {res: res if lv is None or res > finest_region else lv for res, lv in region_of.items()}  # where cap_cells count
    pairs, back = distinct_points(lat, lon)
    for res in sorted({*GEO_AUTO_RESOLUTIONS, *cap_of.values()}, reverse=True):
        # each point indexed at each level as build_cells does: H3 children do not tile their parent
        counts[res] = pd.Series(np.array([h3.latlng_to_cell(a, b, res) for a, b in pairs], dtype=object)[back]).value_counts()
    candidates = {
        res: {
            "cells": len(counts[res]),
            "dense_cells": int((counts[res] >= min_cell_points).sum()),
            "dense_points": round(float(counts[res][counts[res] >= min_cell_points].sum() / counts[res].sum()), 4),
            "median_points": float(counts[res].median()),
            "cap_cells": len(counts[cap_of[res]]),
        }
        for res in GEO_AUTO_RESOLUTIONS
    }
    fits = [res for res, c in candidates.items() if c["dense_cells"] >= min_rows and c["dense_points"] >= GEO_DENSE_POINTS_MIN]
    finest = next((res for res in fits if candidates[res]["cap_cells"] <= GEO_AUTO_MAX_CELLS), fits[-1] if fits else None)
    reasons, chosen, capped = {}, [], []
    for res, c in candidates.items():
        if c["cells"] < min_rows:
            reasons[res] = f"{c['cells']} cells < MIN_ROWS {min_rows}"
        elif finest is None or res > finest:
            if c["dense_cells"] < min_rows:  # medians in too few rows for the miner, however many points they hold
                reasons[res] = f"{c['dense_cells']} dense cells < MIN_ROWS {min_rows}"
            elif c["dense_points"] < GEO_DENSE_POINTS_MIN:
                reasons[res] = f"points in dense cells {c['dense_points']} < {GEO_DENSE_POINTS_MIN}"
            else:
                reasons[res] = f"{c['cap_cells']} cells at {cap_of[res]} > GEO_AUTO_MAX_CELLS"
                capped.append(res)
        elif res == finest:
            reasons[res] = "finest level"
            chosen.append(res)
        elif res not in region_levels:
            reasons[res] = "not a GEO_REGION_LEVELS level"
        elif c["cells"] < GEO_AUTO_MIN_CELLS:
            reasons[res] = f"{c['cells']} cells < GEO_AUTO_MIN_CELLS"
        elif len(chosen) == GEO_LEVELS_MAX:
            reasons[res] = f"GEO_LEVELS_MAX {GEO_LEVELS_MAX} reached"
        else:
            reasons[res] = "region level"
            chosen.append(res)
    if not chosen:
        raise SpatialError("no resolution fits @auto: " + "; ".join(f"{res}: {why}" for res, why in reasons.items()))
    return {  # as JSON stores it: the resolutions of the two maps as text
        "chosen": chosen,
        "capped": capped,
        "candidates": {str(res): c for res, c in candidates.items()},
        "reasons": {str(res): why for res, why in reasons.items()},
    }


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
    """(lat, lon, location columns). ``<lat>,<lon>`` by position, unless the names say longitude then latitude;
    ``auto`` takes the first text column whose values parse as points, else the numeric pair named like lat/latitude
    and lon/lng/longitude."""
    if "," in locator:
        names = [c.strip() for c in locator.split(",")]
        missing = [c for c in names if c not in frame.columns]
        if len(names) != 2 or missing:
            raise SpatialError(f"Geo columns not in dataset: {missing or names}")
        first, second = _tokens(names[0]), _tokens(names[1])
        if first & LON_NAMES and not first & LAT_NAMES and second & LAT_NAMES and not second & LON_NAMES:
            names.reverse()  # `longitude,latitude`: in range either way, so position alone would swap the axes silently
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


def parse_times(values: pd.Series) -> pd.Series:
    """``values`` as UTC times, read in the one format that reads most of them — ISO 8601, or the first value's with the
    day before the month, or with the month first (that order on a tie) —, so a value in another format is NaT, never
    another date; a typed timestamp column as it is (docs/02_discovery.md §2.10). The one reading of a column, for its
    series (``ingestion.read_times``) and for the days its cells count (``point_profile``) alike."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True)
    times = pd.to_datetime(values, errors="coerce", utc=True, format="ISO8601")
    if times.isna().sum() > values.isna().sum():  # a value that is no ISO 8601 time: the first value's format may read more
        with warnings.catch_warnings():  # a guess with the day first warns when that value is ISO 8601
            warnings.simplefilter("ignore", UserWarning)
            formats = dict.fromkeys(f for d in (True, False) if (f := guess_datetime_format(str(values.dropna().iloc[0]), dayfirst=d)))
        readings = [times, *(pd.to_datetime(values, errors="coerce", utc=True, format=f) for f in formats)]
        times = max(readings, key=lambda r: r.notna().sum())
    return times


def _as_datetime(values: pd.Series) -> pd.Series | None:
    """A column that reads as dates (``parse_times`` reads ``POINT_PARSE_SHARE`` of a sample of it) as naive UTC times;
    None for any other."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return parse_times(values).dt.tz_localize(None)
    if values.dtype.kind in "OSU" or isinstance(values.dtype, pd.StringDtype):
        sample = values.dropna().astype(str).head(200)
        if len(sample) and sample.str.len().min() >= 8 and parse_times(sample).notna().mean() >= POINT_PARSE_SHARE:
            return parse_times(values).dt.tz_localize(None)
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


Coded = tuple[np.ndarray, pd.Index]  # a categorical column's levels by frequency and each row's (-1: missing)


@dataclass
class PointProfile:
    """What ``build_cells`` reads of the points, the same at every resolution of one upload: where they are
    (``locate_points``), each column's kind, a date column's times and a categorical column's codes."""

    lat: np.ndarray
    lon: np.ndarray
    location: list[str]
    times: dict[str, np.ndarray] = field(default_factory=dict)
    numeric: list[str] = field(default_factory=list)
    codes: dict[str, Coded] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)


_profile: tuple[object, PointProfile] | None = None  # ponytail: one upload's, as its levels run one after another


def point_profile(frame: pd.DataFrame, locator: str, key: object = None) -> PointProfile:
    """The profile of the points of ``frame`` (``locator`` names their columns); with ``key`` (the upload's parse and
    options) the levels of one upload share it until ``forget_profile``."""
    global _profile
    if key is not None and (kept := _profile) and kept[0] == key:  # read once: an upload's refusal may forget it meanwhile
        return kept[1]
    lat, lon, location = locate_points(frame, locator)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        raise SpatialError(f"No valid point in {location}")
    profile = PointProfile(lat, lon, location)
    for col, values in frame.loc[ok].drop(columns=location).items():
        if (times := _as_datetime(values)) is not None:
            profile.times[str(col)] = times.to_numpy(dtype="datetime64[us]")  # nanoseconds end in 2262
        elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            profile.numeric.append(str(col))
        else:
            text = values.astype("string")
            order = text.dropna().value_counts().index  # by frequency, as every table of the column is ordered
            if 2 <= len(order) <= GEO_MODE_MAX_LEVELS:
                profile.codes[str(col)] = (order.get_indexer(text), order)
            else:
                profile.dropped.append(str(col))
    _profile = (key, profile) if key is not None else _profile
    return profile


def forget_profile() -> None:
    """Drop the profile kept for the levels of an upload."""
    global _profile
    _profile = None


def build_cells(
    frame: pd.DataFrame,
    locator: str,
    resolution: int,
    *,
    min_cell_points: int,
    profile_key: object = None,
    context: Sequence[ContextSource] = (),
    context_ring: int = 1,
) -> tuple[pd.DataFrame, SpatialContext, list[str]]:
    """Source rows -> (cell table, spatial context, warnings); the levels of one upload (``profile_key``) share a ``point_profile``.
    Each ``context`` layer (points of another type) becomes one forced band, the layer's points in the cell and its
    ``context_ring`` rings (``context.py``); a layer with one band only at this resolution is left out with a warning."""
    profile = point_profile(frame, locator, profile_key)
    lat, lon, location = profile.lat, profile.lon, profile.location
    ok = np.isfinite(lat) & np.isfinite(lon)
    pairs, back = distinct_points(lat[ok], lon[ok])
    ids = np.array([h3.latlng_to_cell(a, b, resolution) for a, b in pairs], dtype=object)[back]
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
    dropped = list(profile.dropped)
    for col, values in frame.loc[ok].drop(columns=location).items():
        if (t := profile.times.get(str(col))) is not None:
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
        elif str(col) in profile.numeric:
            med = values.groupby(codes).median().reindex(range(len(cells))).to_numpy(float)
            out[f"{col}_median"] = np.where(dense, med, np.nan)
            glossary[f"{col}_median"] = f"typical (median) {_words(col)} of the cell's events"
    mode_cols, share_cols = _categorical_roles(profile.codes, codes, len(cells), dense)
    nested: dict[str, list[str]] = {}  # mode column -> share columns it is (largely) nested with
    for col in mode_cols:
        if any(_nesting(profile, col, o) >= FUNCTIONAL_SHARE for o in share_cols):
            dropped.append(col)  # its mode would explain those shares by construction
            continue
        nested[col] = [o for o in share_cols if _nesting(profile, col, o) >= NESTED_SHARE]
        table = _level_counts(profile.codes[col], codes, len(cells))
        total = table.sum(axis=1).to_numpy(float)
        out[f"{col}_mode"] = np.where(total > 0, table.columns.to_numpy()[table.to_numpy().argmax(axis=1)], "missing")
        glossary[f"{col}_mode"] = f"the most frequent {_words(col)} among the cell's events"
    for col in share_cols:
        table = _level_counts(profile.codes[col], codes, len(cells))
        total = table.sum(axis=1).to_numpy(float)
        kept = table.columns[-1:] if table.shape[1] == 2 else table.columns  # a binary column: its rarer level
        for level in kept:
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(dense & (total > 0), table[level].to_numpy(float) / total, np.nan)
            if np.nanmean((share > 0) & (share < 1)) >= MIN_MIXED_SHARE:
                out[f"share_{col}_{level}"] = share
                glossary[f"share_{col}_{level}"] = f"share of the cell's events whose {_words(col)} is {level}" + _kind_of(profile, col, level)
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
    bands, layers, words, notes = context_bands(context, cells, resolution, context_ring)
    out |= bands
    derived |= {column: [] for column in bands}  # forced, and restating no metric of the target
    glossary |= words
    warnings += notes
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
        context=layers,
    )
    return table, ctx, warnings


def _words(name: str) -> str:
    return str(name).replace("_", " ")


def _kind_of(profile: PointProfile, col: str, level: str) -> str:
    """' (every such event also has: <column> <value>, ...)' for coarser columns whose value is fixed among ``level``'s rows."""
    codes, order = profile.codes[col]
    rows = codes == order.get_loc(level)
    hints = []
    for other, (other_codes, other_order) in profile.codes.items():
        if other == col or len(other_order) >= len(order):
            continue
        present = other_codes[rows]
        present = present[present >= 0]
        if len(present) and (counts := np.bincount(present)).max() >= FUNCTIONAL_SHARE * len(present):
            hints.append(f"{_words(other)} {other_order[counts.argmax()]}")
    return f" (every such event also has: {', '.join(hints[:3])})" if hints else ""


def distinct_points(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the distinct (lat, lon) pairs, each point's row among them): a place many events share is indexed once."""
    pairs, back = np.unique(np.column_stack([lat, lon]), axis=0, return_inverse=True)
    return pairs, back.ravel()


def _level_counts(column: Coded, codes: np.ndarray, n_cells: int) -> pd.DataFrame:
    """Points per (cell, level) of a profiled column; columns ordered by overall frequency (a missing value counts in none)."""
    level, order = column
    kept = level >= 0
    counts = np.bincount(codes[kept] * len(order) + level[kept], minlength=n_cells * len(order))
    return pd.DataFrame(counts.reshape(n_cells, len(order)), index=pd.RangeIndex(n_cells), columns=order)


def _categorical_roles(columns: dict[str, Coded], codes: np.ndarray, n_cells: int, dense: np.ndarray) -> tuple[list[str], list[str]]:
    """(mode columns, share columns). A column whose dense cells are almost pure (mean top-level share >=
    ``PLACE_PURITY``) describes the place: its mode is a condition. A mixed column with few levels becomes
    per-level share metrics; a mixed column with more levels a mode condition."""
    modes, shares = [], []
    for col, column in columns.items():
        table = _level_counts(column, codes, n_cells).to_numpy(float)[dense]
        total = table.sum(axis=1)
        purity = float(np.mean(table.max(axis=1)[total > 0] / total[total > 0])) if (total > 0).any() else 1.0
        if purity < PLACE_PURITY and table.shape[1] <= GEO_SHARE_MAX_LEVELS:
            shares.append(col)
        else:
            modes.append(col)
    return modes, shares


def _nesting(profile: PointProfile, a: str, b: str) -> float:
    """Share of rows on which one column's level fixes the other's (the larger of both directions)."""
    (ca, oa), (cb, ob) = profile.codes[a], profile.codes[b]
    both = (ca >= 0) & (cb >= 0)
    if not both.any():
        return 0.0
    table = np.bincount(ca[both] * len(ob) + cb[both], minlength=len(oa) * len(ob)).reshape(len(oa), len(ob))
    return float(max(table.max(axis=1).sum(), table.max(axis=0).sum()) / table.sum())


def compactness(rows: np.ndarray, nbr_ptr: np.ndarray, nbr_idx: np.ndarray) -> float:
    """Share of the extent's neighbour links that stay inside the extent (1 = one solid block, 0 = isolated cells)."""
    inside = np.zeros(len(nbr_ptr) - 1, dtype=bool)
    inside[rows] = True
    owner = np.repeat(np.arange(len(nbr_ptr) - 1), np.diff(nbr_ptr))
    total = int(np.diff(nbr_ptr)[rows].sum())
    return int((inside[owner] & inside[nbr_idx]).sum()) / total if total else 0.0


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
