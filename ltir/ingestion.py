"""Dataset ingestion: file validation, loading, derived band dimensions (docs/02_discovery.md §2.1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ltir.config import Config

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


def parse_columns(spec: str | None) -> list[str]:
    return [c.strip() for c in (spec or "").split(",") if c.strip()]


def dataset_fingerprint(payload: bytes, bins: dict[str, int], categories: list[str] | None = None) -> str:
    """Content-addressed dataset id: same bytes + same derivation options -> same id."""
    h = hashlib.sha256(payload)
    h.update(repr(sorted(bins.items())).encode("utf-8"))
    if categories:  # mixed in only when overrides apply: a plain file keeps its plain id
        h.update(repr(sorted(categories)).encode("utf-8"))
    return "ds-" + h.hexdigest()[:12]


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError("unsupported_file", f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    try:
        if ext == ".parquet":
            return pd.read_parquet(path)
        sep = "\t" if ext == ".tsv" else None
        return pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")
    except Exception as exc:  # parser errors, encoding errors, empty files
        raise IngestionError("unreadable_file", f"Could not parse {path.name}: {exc}") from exc


def select_present(columns: list[str], frame: pd.DataFrame, *, explicit: bool, option: str) -> tuple[list[str], list[str]]:
    """Split an option's columns into (present, missing). An explicit option must name existing
    columns (``invalid_options``); a workspace default skips the ones this dataset lacks."""
    missing = [c for c in columns if c not in frame.columns]
    if missing and explicit:
        raise IngestionError("invalid_options", f"{option} column(s) not in dataset: {missing}")
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
    config: Config,
    *,
    filename: str | None = None,
    bins: str | None = None,
    categories: str | None = None,
) -> LoadedDataset:
    """``bins`` / ``categories`` = None -> workspace defaults, applied leniently (columns a
    dataset does not have are skipped); an explicit value is applied strictly."""
    path = Path(path)
    if not path.is_file():
        raise IngestionError("unsupported_file", f"File not found: {path}")
    if path.stat().st_size > config.max_upload_mb * 1024 * 1024:
        raise IngestionError("unsupported_file", f"File exceeds {config.max_upload_mb} MB")
    bin_spec = parse_bins(bins if bins is not None else config.bin_columns)
    cat_spec = parse_columns(categories if categories is not None else config.categorical_columns)
    frame = read_table(path)
    if frame.shape[1] < 2:
        raise IngestionError("invalid_schema", "Need at least one categorical and one numeric column")
    if len(frame) < config.min_rows:
        raise IngestionError("invalid_schema", f"Only {len(frame)} rows (MIN_ROWS={config.min_rows})")
    warnings = []
    cats, missing_cats = select_present(cat_spec, frame, explicit=categories is not None, option="Categorical")
    frame = apply_categories(frame, cats)
    bin_cols, missing_bins = select_present(list(bin_spec), frame, explicit=bins is not None, option="Bin")
    present_bins = {c: bin_spec[c] for c in bin_cols}
    frame, derived = apply_bins(frame, present_bins)
    if missing_cats or missing_bins:
        warnings.append(f"workspace column options not in this dataset (skipped): {missing_cats + missing_bins}")
    if not any(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns):
        raise IngestionError("no_numeric_targets", "Dataset has no numeric columns to analyse")
    empty = [c for c in frame.columns if frame[c].isna().all()]
    if empty:
        warnings.append(f"all-null columns ignored by profiling: {empty}")
    return LoadedDataset(
        frame=frame,
        dataset_id=dataset_fingerprint(path.read_bytes(), present_bins, cats),
        filename=filename or path.name,
        bins=present_bins,
        categories=cats,
        derived_columns=derived,
        warnings=warnings,
    )
