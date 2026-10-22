"""Dataset ingestion: file validation, loading, derived band dimensions (docs/02_discovery.md §2.1)."""

from __future__ import annotations

import csv
import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from subgroup_miner.config import MinerConfig
from subgroup_miner.spatial import SpatialContext, SpatialError, build_cells, locate_points, parse_geo, select_levels

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
    digest: Any, bins: dict[str, int], categories: list[str] | None = None, geo: str | None = None, left_out: list[str] | None = None
) -> str:
    """Content-addressed dataset id from the sha256 of the file's bytes (``read_file``): same bytes + same derivation
    options -> same id."""
    h = digest.copy()
    h.update(repr(sorted(bins.items())).encode("utf-8"))
    if categories:  # mixed in only when overrides apply: a plain file keeps its plain id
        h.update(repr(sorted(categories)).encode("utf-8"))
    if geo:
        h.update(f"geo={geo}".encode())
    if left_out:  # the table level of a geo upload: its point columns left out
        h.update(f"left_out={sorted(left_out)}".encode())
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
            if len(frame.select_dtypes("object").columns):  # chunks may have typed a column apart: the whole file decides
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
    """Drop the parse kept for the levels of an upload (``read_file(shared=True)``)."""
    _kept.cache_clear()


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
) -> LoadedDataset:
    """``bins`` / ``categories`` = None -> the configured defaults, applied leniently (columns a
    dataset does not have are skipped); an explicit value is applied strictly. With ``strict`` False (the levels of one
    upload share their options) a geo level skips the file's own columns and the table level any column: a name in
    neither the file nor the cells still fails the geo levels. ``locator`` (the table level of a geo upload) leaves
    the columns it locates the points in out of the table. ``geo`` turns the
    point rows into a table of H3 cells first (§2.9); bands and categories then apply to that table. ``shared``: a level
    of an upload, reading the parse its levels share (``read_file``)."""
    path = Path(path)
    if not path.is_file():
        raise IngestionError("unsupported_file", f"File not found: {path}")
    bin_spec = parse_bins(bins if bins is not None else config.bin_columns)
    cat_spec = parse_columns(categories if categories is not None else config.categorical_columns)
    try:
        geo_spec = parse_geo(geo, config.geo_resolution) if geo else None
    except SpatialError as exc:
        raise IngestionError("invalid_options", str(exc)) from exc
    frame, digest = read_file(path, shared=shared)
    if frame.shape[1] < 2:
        raise IngestionError("invalid_schema", "Need at least one categorical and one numeric column")
    if len(frame) < config.min_rows:
        raise IngestionError("invalid_schema", f"Only {len(frame)} rows (MIN_ROWS={config.min_rows})")
    warnings = []
    spatial = None
    left_out = []
    if locator and not geo_spec:
        try:
            left_out = locate_points(frame, locator)[2]
        except SpatialError as exc:
            raise IngestionError("invalid_options", str(exc)) from exc
        frame = frame.drop(columns=left_out)
        warnings.append(f"point columns {left_out} left out of the table level")
    tolerated = set() if strict else set(frame.columns) if geo_spec else set(cat_spec) | set(bin_spec)
    if geo_spec:
        frame, spatial, geo_warnings = _to_cells(frame, geo_spec, config)
        warnings += geo_warnings
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
        dataset_id=dataset_fingerprint(digest, present_bins, cats, geo=_geo_text(geo_spec), left_out=left_out),
        filename=filename or path.name,
        bins=present_bins,
        categories=cats,
        derived_columns=derived,
        warnings=warnings,
        geo=_geo_text(geo_spec),
        spatial=spatial,
        points=",".join(left_out) or None,
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


def _to_cells(frame: pd.DataFrame, spec: tuple[str, int], config: MinerConfig) -> tuple[pd.DataFrame, SpatialContext, list[str]]:
    """Point rows -> H3 cell rows; the neighbourhood lags become quartile-band dimensions."""
    try:
        cells, ctx, warnings = build_cells(
            frame,
            *spec,
            min_cell_points=config.min_cell_points,
        )
    except SpatialError as exc:
        raise IngestionError("invalid_options", str(exc)) from exc
    if len(cells) < config.min_rows:
        raise IngestionError("invalid_schema", f"Only {len(cells)} H3 cells at resolution {spec[1]} (MIN_ROWS={config.min_rows})")
    cells, _ = apply_bins(cells, {c: 4 for c in ctx.lag_columns})
    return cells, ctx, warnings
