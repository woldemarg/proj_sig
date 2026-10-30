"""Dataset ingestion: file validation, loading, derived band dimensions (docs/02_discovery.md §2.1)."""

from __future__ import annotations

import csv
import functools
import hashlib
import zoneinfo
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insight_contracts import TIME_GRIDS
from subgroup_miner.config import MinerConfig
from subgroup_miner.context import ContextSource
from subgroup_miner.spatial import SpatialContext, SpatialError, build_cells, forget_profile, locate_points, parse_geo, parse_times, select_levels

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet"}


class IngestionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    dataset_id: str
    filename: str
    bins: dict[str, int] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    derived_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    geo: str | None = None  # the applied geo option, "<locator>@<resolution>"
    spatial: SpatialContext | None = None  # rows are H3 cells when set
    points: str | None = None  # a table level's left-out point columns, as a geo level's ``location`` names them
    time: pd.Series | None = None  # each row's time (UTC; NaT where its value reads as none), the time column left out; a geo level's: each point's
    point_values: pd.DataFrame | None = None  # a geo level with a time column: the points' numeric columns its cell medians summarise
    grid: str | None = None  # the calendar unit its times are binned by (``TIME_GRIDS``)
    tz: str | None = None  # the zone whose calendar the grid is, and in which a time without one is read (UTC when None)


def parse_bins(spec: str | None) -> dict[str, int]:
    """``"median_income:4, housing_median_age:4"`` -> {column: quantile count}."""
    bins: dict[str, int] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, q = part.partition(":")
        try:
            bins[name.strip()] = int(q or 4)
        except ValueError as exc:
            raise IngestionError("invalid_options", f"Bad bin spec '{part}' (expected column:quantiles)") from exc
    return bins


INDICATOR_FIELD_MAX = 60


def parse_indicators(spec: str | None) -> dict[str, dict[str, str]]:
    """``"price:retail_price:UAH, qty:units_sold"`` -> {metric: {"indicator", "unit" when given}}: what each metric
    measures, so that two datasets' metrics can be compared by meaning rather than by name (whitespace collapsed)."""
    registry: dict[str, dict[str, str]] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        fields = [" ".join(f.split()) for f in part.split(":")]
        if len(fields) not in (2, 3) or not all(fields) or max(map(len, fields)) > INDICATOR_FIELD_MAX or fields[0] in registry:
            raise IngestionError("invalid_options", f"Bad indicator spec '{part.strip()}' (expected metric:indicator[:unit], each metric once)")
        registry[fields[0]] = {"indicator": fields[1], **({"unit": fields[2]} if len(fields) == 3 else {})}
    return registry


def parse_columns(spec: str | None) -> list[str]:
    return [c.strip() for c in (spec or "").split(",") if c.strip()]


READING_FIX = b"reading=2"  # a file read otherwise than before: its own dataset id (docs/02_discovery.md §2.1)


def dataset_fingerprint(
    digest: Any,
    bins: dict[str, int],
    categories: list[str] | None = None,
    geo: str | None = None,
    left_out: list[str] | None = None,
    time: str | None = None,
    context: list[str] | None = None,
) -> str:
    """Content-addressed dataset id from the sha256 of the file's bytes (``read_file``): same bytes + same derivation
    options -> same id. ``time``: ``<column>:<grid>`` (``auto`` as asked: its unit is fitted after admission) — the same
    file with its time apart is another analysis. ``context``: ``<layer>:<sha12>:<locator>`` per context layer, in any
    order — the same points with another layer beside them are another analysis too."""
    h = digest.copy()
    h.update(repr(sorted(bins.items())).encode("utf-8"))
    if categories:  # mixed in only when overrides apply: a plain file keeps its plain id
        h.update(repr(sorted(categories)).encode("utf-8"))
    if geo:
        h.update(f"geo={geo}".encode())
    if left_out:  # the table level of a geo upload: its point columns left out
        h.update(f"left_out={sorted(left_out)}".encode())
    if time:
        h.update(f"time={time}".encode())
    if context:
        h.update(f"context={sorted(context)}".encode())
    return "ds-" + h.hexdigest()[:12]


def read_file(path: Path, *, shared: bool = False) -> tuple[pd.DataFrame, Any]:
    """(the file as a frame, the sha256 of its bytes, with ``READING_FIX`` for a file whose table ``_parse`` changed). ``shared``: the levels of one upload read one parse, kept until
    ``forget_tables`` (each gets a shallow copy: no level writes into the columns it reads); any other read, a re-check
    of the source among them, parses the file as it is now."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError("unsupported_file", f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    try:
        st = path.stat()
    except OSError as exc:  # removed meanwhile
        raise IngestionError("unsupported_file", f"File not found: {path}") from exc
    if not shared:
        return _parse(str(path), ext, st.st_mtime_ns, st.st_size)
    frame, digest = _kept(str(path), ext, st.st_mtime_ns, st.st_size)
    return frame.copy(deep=False), digest.copy()


def _parse(path: str, ext: str, *_stamp: int) -> tuple[pd.DataFrame, Any]:
    """One parse. A CSV's delimiter is sniffed from its first line and the C parser reads the whole file at once, as
    pandas' Python parser reads it: the same columns, types and values, a leading BOM dropped. The digest of a file
    whose table that reading changed (a BOM; a TSV whose chunks typed a column apart) carries ``READING_FIX``."""
    try:
        with open(path, "rb") as f:
            bom = f.read(3) == b"\xef\xbb\xbf"
            f.seek(0)
            digest = hashlib.file_digest(f, "sha256")
        if ext == ".parquet":
            return pd.read_parquet(path), digest
        if ext == ".tsv":
            frame = pd.read_csv(path, sep="\t", engine="c")
            # a chunk typed a column apart: codes then text leave values of two types, an integer chunk beside a float or
            # an unsigned one makes floats of integers past 2**53; then the whole file decides, else it is read once
            if any(
                pd.api.types.infer_dtype(s, skipna=True).startswith("mixed") or (s.dtype.kind == "f" and s.abs().max() >= 2**53)
                for _, s in frame.items()
            ):
                whole = pd.read_csv(path, sep="\t", engine="c", low_memory=False)
                if not whole.equals(frame):
                    frame = whole
                    digest.update(READING_FIX)
            return frame, digest
        with open(path, encoding="utf-8-sig", newline="") as f:  # a BOM before a quote would be sniffed as the delimiter
            sep = csv.Sniffer().sniff(f.readline()).delimiter
        if bom:
            digest.update(READING_FIX)
        return pd.read_csv(path, sep=sep, engine="c", low_memory=False), digest  # one dtype per column; drops a BOM
    except Exception as exc:  # parser errors, encoding errors, empty files
        raise IngestionError("unreadable_file", f"Could not parse {Path(path).name}: {exc}") from exc


_kept = functools.lru_cache(maxsize=1)(_parse)  # ponytail: one file, as the levels of one upload run one after another


def forget_tables() -> None:
    """Drop the parse kept for the levels of an upload (``read_file(shared=True)``), its points' profile and its
    context layers' points."""
    _kept.cache_clear()
    _context_points.cache_clear()
    forget_profile()


def select_present(
    columns: list[str], frame: pd.DataFrame, *, explicit: bool, option: str, tolerated: set[str] | frozenset[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """Split an option's columns into (present, missing). An explicit option must name existing
    columns, or ``tolerated`` ones (``invalid_options``); a configured default skips the ones this dataset lacks."""
    missing = [c for c in columns if c not in frame.columns]
    if explicit and (refused := [c for c in missing if c not in tolerated]):
        raise IngestionError("invalid_options", f"{option} column(s) not in dataset: {refused}")
    return [c for c in columns if c in frame.columns], missing


def apply_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast code-like columns (e.g. ``Store`` 1..45, ``Holiday_Flag`` 0/1) to text dimensions."""
    out = df.copy()
    for col in columns:
        values = out[col]
        if pd.api.types.is_float_dtype(values) and values.dropna().eq(values.dropna().round()).all():
            values = values.astype("Int64")  # 1.0 -> "1", not "1.0"
        out[col] = values.astype(str).where(values.notna(), "missing")
    return out


def apply_bins(df: pd.DataFrame, bins: dict[str, int]) -> tuple[pd.DataFrame, list[str]]:
    """Replace numeric columns by quantile-band categoricals ``<col>_band`` (q1..qk).

    The source column is dropped so its band is a *dimension*, not a tautological target.
    """
    out = df.copy()
    derived: list[str] = []
    for col, q in bins.items():
        if not pd.api.types.is_numeric_dtype(out[col]):
            raise IngestionError("invalid_options", f"Bin column '{col}' is not numeric")
        labels = pd.qcut(out[col], q=q, duplicates="drop")
        codes = labels.cat.codes.to_numpy()
        names = np.array([f"q{i + 1}" for i in range(len(labels.cat.categories))], dtype=object)
        band = np.where(codes >= 0, names[np.clip(codes, 0, None)], "missing")
        name = f"{col}_band"
        out[name] = band.astype(str)
        out = out.drop(columns=[col])
        derived.append(name)
    return out, derived


GRID_FREQ = {"day": "D", "week": "W-SUN", "month": "M", "quarter": "Q"}  # a week runs Monday to Sunday, as ISO 8601's
GRID_MIN_BINS, GRID_MIN_ROWS = 14, 300  # what a grid should give a span: bins enough for a series, rows enough per bin for an insight's share


def parse_grid(spec: str | None) -> str:
    """The calendar unit of a time column's grid (``TIME_GRIDS``), or ``auto`` (the default: ``resolve_grid`` picks it
    from the span). A duration such as ``7D`` is refused: a bin is a calendar period, the same for every upload."""
    unit = (spec or "auto").strip().lower()
    if unit != "auto" and unit not in TIME_GRIDS:
        raise IngestionError("invalid_options", f"grid {spec!r}: one of {', '.join(TIME_GRIDS)} or auto — a calendar unit, not a duration such as 7D")
    return unit


def resolve_grid(times: pd.Series, grid: str) -> str:
    """``auto`` resolved from ``times``: of quarter, month, week and day the unit that best balances the bins its span
    makes against ``GRID_MIN_BINS`` and the rows a bin holds against ``GRID_MIN_ROWS`` — the smaller of the two ratios
    the highest, a tie to the coarser: a month of dense data gets days, a year weeks, a thin year months. The span is
    the 1st to the 99th percentile of the times, so one stray date cannot coarsen the grid (``check_time`` then refuses
    it on the unit chosen). A named grid is itself."""
    if grid != "auto":
        return grid
    naive = times.dropna().dt.tz_convert("UTC").dt.tz_localize(None)
    if naive.empty:  # no time reads (a geo level none of whose points has one): any unit, the check refuses it
        return "week"
    first, last = naive.quantile(0.01), naive.quantile(0.99)

    def balance(unit: str) -> float:
        bins = pd.Period(last, GRID_FREQ[unit]).ordinal - pd.Period(first, GRID_FREQ[unit]).ordinal + 1
        return min(bins / GRID_MIN_BINS, len(naive) / bins / GRID_MIN_ROWS)

    return max(("quarter", "month", "week", "day"), key=balance)


# a zone after a clock time, never a date's -2031
ZONED = r"[0-9]:[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]+)?)?\s*(?:Z|UTC|GMT|[+-][0-9]{2}(?::?[0-9]{2})?)\s*$"


def parse_tz(tz: str | None) -> str | None:
    """A time zone by its IANA name (``Europe/Kyiv``, as written); None for none, empty or UTC (one calendar, one dataset id).
    Another name is ``invalid_options``."""
    name = (tz or "").strip()
    if name.upper() in ("", "UTC", "ETC/UTC"):
        return None
    if name not in zoneinfo.available_timezones():  # not ZoneInfo: it reads a folder's name as a file, case-blind on Windows
        raise IngestionError("invalid_options", f"time zone {tz!r}: an IANA name such as Europe/Kyiv")
    return name


def read_times(frame: pd.DataFrame, column: str, tz: str | None = None) -> pd.Series:
    """A time column as UTC times, read in the one format that reads most of it (``spatial.parse_times``: the reading
    a geo level's cells count their days by too); a time without a zone is taken in ``tz``, UTC by default. A column the
    file lacks, a numeric one (no unit to read it in) or one with no value that reads as a time is ``invalid_options``."""
    if column not in frame.columns:
        raise IngestionError("invalid_options", f"time column '{column}' not in dataset")
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        raise IngestionError("invalid_options", f"time column '{column}' holds numbers: give its times as dates")
    times = parse_times(values)
    if times.isna().all():
        raise IngestionError("invalid_options", f"no value of time column '{column}' reads as a time")
    if tz:  # a time written without a zone is a wall time of tz: shift it from UTC to there; one with a zone keeps it
        zoned = values.astype("string").str.contains(ZONED, case=False, regex=True, na=False)  # a typed timestamp's text too
        # the autumn's repeated hour read as its first: either lies in the same day; the spring's missing one as the next
        local = times.dt.tz_localize(None).dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
        times = times.where(zoned, local)
    return times


def load_dataset(
    path: Path,
    config: MinerConfig,
    *,
    filename: str | None = None,
    bins: str | None = None,
    categories: str | None = None,
    geo: str | None = None,
    strict: bool = True,
    locator: str | None = None,
    shared: bool = False,
    time: str | None = None,
    grid: str | None = None,
    tz: str | None = None,
    context: Sequence[tuple[str, Path, str]] = (),
) -> LoadedDataset:
    """``bins`` / ``categories`` = None -> the configured defaults, applied leniently (columns a
    dataset does not have are skipped); an explicit value is applied strictly. With ``strict`` False (the levels of one
    upload share their options) a geo level skips the file's own columns and the table level any column: a name in
    neither the file nor the cells still fails the geo levels. ``locator`` (the table level of a geo upload) leaves
    the columns it locates the points in out of the table. ``geo`` turns the
    point rows into a table of H3 cells first (§2.9); bands and categories then apply to that table. ``shared``: a level
    of an upload, reading the parse its levels share (``read_file``). ``time`` names a time column: read apart
    (``read_times``) and left out of the table, binned later by ``grid`` (``parse_grid``; ``auto``, the default, is
    ``resolve_grid`` of the span of the rows the series are of — a geo level's located points, a table's every row —,
    a unit the commit may coarsen for the admitted insights; §2.10) in the calendar of ``tz`` (``parse_tz``; UTC by
    default, and the zone of a time written without one) — discovery never sees it. ``context``: ``(layer, path,
    locator)`` per context layer — another type's points, read apart (``read_context``) and counted into the cells as one
    forced band each (§2.9); it needs ``geo``."""
    path = Path(path)
    if not path.is_file():
        raise IngestionError("unsupported_file", f"File not found: {path}")
    bin_spec = parse_bins(bins if bins is not None else config.bin_columns)
    cat_spec = parse_columns(categories if categories is not None else config.categorical_columns)
    try:
        geo_spec = parse_geo(geo, config.geo_resolution) if geo else None
    except SpatialError as exc:
        raise IngestionError("invalid_options", str(exc)) from exc
    if context and not geo_spec:
        raise IngestionError("invalid_options", "context layers need the geo option: they are counted into its cells")
    layers = [read_context(*layer) for layer in context]
    frame, digest = read_file(path, shared=shared)
    if frame.shape[1] < 2:
        raise IngestionError("invalid_schema", "Need at least one categorical and one numeric column")
    if len(frame) < config.min_rows:
        raise IngestionError("invalid_schema", f"Only {len(frame)} rows (MIN_ROWS={config.min_rows})")
    warnings = []
    spatial = None
    left_out = []
    times = None
    option = None
    if time:
        tz = parse_tz(tz)
        times = read_times(frame, time, tz)
        option = parse_grid(grid)  # auto names the dataset: its unit is fitted after admission (temporal.fit_grid)
        if not geo_spec:  # a geo level's cells count its days as they would without a time option: time adds, never removes
            frame = frame.drop(columns=[time])
        if unread := int(times.isna().sum()):
            warnings.append(f"{unread} rows without a readable time: mined, outside every time bin")
    if locator and not geo_spec:
        try:
            left_out = locate_points(frame, locator)[2]
        except SpatialError as exc:
            raise IngestionError("invalid_options", str(exc)) from exc
        frame = frame.drop(columns=left_out)
        warnings.append(f"point columns {left_out} left out of the table level")
    tolerated = set() if strict else set(frame.columns) if geo_spec else set(cat_spec) | set(bin_spec)
    point_values = None
    if geo_spec:
        points = frame
        key = (digest.hexdigest(), geo_spec[0]) if shared else None  # the levels of one upload: one point profile
        frame, spatial, geo_warnings = _to_cells(frame, geo_spec, config, key, [source for source, _ in layers])
        warnings += geo_warnings
        if times is not None:  # the series of a geo pattern read its points (docs/02_discovery.md §2.10)
            point_values = points.select_dtypes(include=[np.number]).drop(columns=spatial.location.split(","), errors="ignore")
    if time:  # auto's unit from the rows the series are of: a geo level's located points, a table's every row (§2.10)
        grid = resolve_grid(times if spatial is None else times[spatial.point_cell >= 0], option)
    cats, missing_cats = select_present(cat_spec, frame, explicit=categories is not None, option="Categorical", tolerated=tolerated)
    frame = apply_categories(frame, cats)
    bin_cols, missing_bins = select_present(list(bin_spec), frame, explicit=bins is not None, option="Bin", tolerated=tolerated)
    present_bins = {c: bin_spec[c] for c in bin_cols}
    frame, derived = apply_bins(frame, present_bins)
    if missing_cats or missing_bins:
        warnings.append(f"column options not in this dataset (skipped): {missing_cats + missing_bins}")
    if not any(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns):
        raise IngestionError("no_numeric_targets", "Dataset has no numeric columns to analyse")
    empty = [c for c in frame.columns if frame[c].isna().all()]
    if empty:
        warnings.append(f"all-null columns ignored by profiling: {empty}")
    return LoadedDataset(
        frame=frame,
        dataset_id=dataset_fingerprint(
            digest,
            present_bins,
            cats,
            geo=_geo_text(geo_spec),
            left_out=left_out,
            time=f"{time}:{option}" + (f":{tz}" if tz else "") if time else None,
            context=[fingerprint for _, fingerprint in layers],
        ),
        filename=filename or path.name,
        bins=present_bins,
        categories=cats,
        derived_columns=derived,
        warnings=warnings,
        geo=_geo_text(geo_spec),
        spatial=spatial,
        points=",".join(left_out) or None,
        time=times,
        grid=grid if time else None,
        tz=tz if time else None,
        point_values=point_values,
    )


def auto_levels(path: Path, locator: str, config: MinerConfig, region_levels: tuple[int, ...]) -> dict[str, Any]:
    """``select_levels`` for the points of a stored file (the ``@auto`` resolution of an upload), its parse kept for
    the levels (``read_file(shared=True)``)."""
    try:
        lat, lon, _ = locate_points(read_file(Path(path), shared=True)[0], locator)
        return select_levels(lat, lon, region_levels, min_rows=config.min_rows, min_cell_points=config.min_cell_points)
    except SpatialError as exc:
        raise IngestionError("invalid_options", str(exc)) from exc


def _geo_text(spec: tuple[str, int] | None) -> str | None:
    return f"{spec[0]}@{spec[1]}" if spec else None


def read_context(name: str, path: Path, locator: str) -> tuple[ContextSource, str]:
    """A context layer's points -> (them, its fingerprint part ``<layer>:<sha12>:<locator>``); a file without a point
    is refused. Read once for the levels of an upload (kept by the file's stamp, as ``read_file`` keeps the target's;
    a file changed since reads anew), never through the shared parse: that holds the upload's own file."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError as exc:  # removed meanwhile: as a missing source
        raise IngestionError("unsupported_file", f"context layer {name!r}: file not found: {path}") from exc
    try:
        lat, lon, sha = _context_points(str(path), locator, st.st_mtime_ns, st.st_size)
    except SpatialError as exc:
        raise IngestionError("invalid_options", f"context layer {name!r}: {exc}") from exc
    if not (np.isfinite(lat) & np.isfinite(lon)).any():
        raise IngestionError("invalid_options", f"context layer {name!r}: No valid point in {path.name}")
    return ContextSource(name, lat, lon), f"{name}:{sha[:12]}:{locator}"


@functools.lru_cache(maxsize=4)  # ponytail: the layers of the upload running now (at most two); keyed by the file's stamp
def _context_points(path: str, locator: str, *_stamp: int) -> tuple[np.ndarray, np.ndarray, str]:
    frame, digest = read_file(Path(path))
    lat, lon, _ = locate_points(frame, locator)
    return lat, lon, digest.hexdigest()


def _to_cells(
    frame: pd.DataFrame, spec: tuple[str, int], config: MinerConfig, profile_key: object = None, context: Sequence[ContextSource] = ()
) -> tuple[pd.DataFrame, SpatialContext, list[str]]:
    """Point rows -> H3 cell rows; the neighbourhood lags become quartile-band dimensions. ``profile_key``: the levels of
    one upload profile its point columns once (``spatial.point_profile``); ``context``: the layers counted into the cells."""
    try:
        cells, ctx, warnings = build_cells(
            frame,
            *spec,
            min_cell_points=config.min_cell_points,
            profile_key=profile_key,
            context=context,
            context_ring=config.context_ring,
        )
    except SpatialError as exc:
        raise IngestionError("invalid_options", str(exc)) from exc
    if len(cells) < config.min_rows:
        raise IngestionError("invalid_schema", f"Only {len(cells)} H3 cells at resolution {spec[1]} (MIN_ROWS={config.min_rows})")
    cells, _ = apply_bins(cells, {c: 4 for c in ctx.lag_columns})
    return cells, ctx, warnings
