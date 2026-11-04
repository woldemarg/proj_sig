"""One upload (docs/02_discovery.md §2.9, docs/06_graph_and_storage.md §6.8): the levels it makes, the version it
replaces, a layer kept for a cross-type run and the run of two stored uploads. Functions of the options, the batch
registry and one record; the engine applies what they decide."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Sequence
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from insight_contracts import Insight, Rejection
from insight_contracts.text import RELIABILITY
from insight_contracts.wording import source_column
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.settings import Settings
from insight_graph_service.core.workspace import TERMINAL, Workspace, new_batch_id, utc_now
from subgroup_miner.config import MinerConfig
from subgroup_miner.ingestion import (
    IngestionError,
    check_counts,
    parse_baseline,
    parse_indicators,
    parse_ring,
    parse_tz,
    parse_where,
    read_context,
    read_file,
    read_rows,
    read_times,
)
from subgroup_miner.spatial import SpatialError, locate_points, parse_geo_levels

EMPTY_LEVEL = frozenset({"invalid_schema", "no_numeric_targets", "no_candidates", "no_viable_insights"})  # a level with nothing to mine
GLOSSARY_VALUE_MAX = 120  # characters of one declared glossary value
METADATA_DOC_MAX_BYTES = 64 * 1024  # a metadata document as uploaded
SOURCE_MAX_CHARS = 120  # characters of a declared source
# context files of one upload, one forced band each (docs/02_discovery.md §2.9): a third multiplies the Bonferroni family,
# fills the validation budget with context cohorts and, on a wide table, empties the 3-conjunctions (spec §14.6)
MAX_CONTEXT_LAYERS = 2
# a layer link coarser than this H3 level is warned of, and the console offers none: a zone this wide already holds
# different places (concept §9); preliminary, the spike #281 may move it
LINK_MIN_LEVEL = 8
TRUST_FIELDS = ("source", "reliability", "indicators", "user_glossary", "metadata_doc")  # what a cross-type run keeps of its target
_FIELD = re.compile(r"([A-Za-z][\w ()/.-]{0,40}):\s*(.*)$")  # "Key: value" (a key of at most 41 characters)


def parse_metadata_doc(text: str) -> dict[str, Any]:
    """A metadata document (docs/02_discovery.md §2.1) -> ``{"dataset": {field: value}, "columns": {column: {field: value}}}``.
    The lines before ``Columns:`` describe the dataset as ``Key: value`` lines; after it come blocks, one per column: its
    name on an unindented line, its ``Key: value`` fields indented. A field sits at the indentation of its block's (or the
    header's) first field; any other line, a deeper one or one without a key, continues the previous value. A key is its
    label in snake case (``Title (uk)`` -> ``title_uk``); unknown keys are kept, a block without fields is just its name.
    A BOM and CRLF are read; ``ValueError`` without ``Columns:`` or a column."""
    lines = text.removeprefix(chr(0xFEFF)).splitlines()
    marker = next((i for i, ln in enumerate(lines) if ln.strip().lower() == "columns:"), None)
    if marker is None:
        raise ValueError("no 'Columns:' line")
    dataset: dict[str, str] = {}
    columns: dict[str, dict[str, str]] = {}
    fields: dict[str, str] | None = dataset
    key, indent = None, None
    for i, line in enumerate(lines):
        depth = len(line) - len(line.lstrip())
        if i == marker or not line.strip():
            fields = None if i == marker else fields
            continue
        if i > marker and depth == 0:  # a column's name opens its block
            fields, key, indent = columns.setdefault(line.strip(), {}), None, None
        elif fields is not None and (m := _FIELD.match(line.strip())) and indent in (None, depth):
            key, indent = re.sub(r"[^0-9a-z]+", "_", m[1].lower()).strip("_"), depth
            fields[key] = m[2].strip()
        elif fields is not None and key:
            fields[key] = f"{fields[key]} {line.strip()}".strip()
    if not columns:
        raise ValueError("no column after 'Columns:'")
    return {"dataset": {k: v for k, v in dataset.items() if v}, "columns": {c: {k: v for k, v in f.items() if v} for c, f in columns.items()}}


def resolve_dataset_metadata(source: str | None, reliability: str | None, indicators: str | None = None) -> dict[str, Any]:
    """The given dataset metadata, normalised: ``source`` on one line (whitespace collapsed), ``reliability`` an
    upper-case Admiralty code A-F, ``indicators`` the parsed registry (``parse_indicators``); an empty value is None
    (undeclared). A value not given (None) is left out."""
    options: dict[str, Any] = {}
    if source is not None:
        options["source"] = " ".join(source.split()) or None
        if len(options["source"] or "") > SOURCE_MAX_CHARS:
            raise PipelineError("invalid_options", f"source longer than {SOURCE_MAX_CHARS} characters")
    if reliability is not None:
        options["reliability"] = reliability.strip().upper() or None
        if options["reliability"] not in (None, *RELIABILITY):
            raise PipelineError("invalid_options", f"reliability '{reliability}' is not one of {', '.join(RELIABILITY)}")
    if indicators is not None:
        try:
            options["indicators"] = parse_indicators(indicators) or None
        except IngestionError as exc:
            raise PipelineError(exc.code, str(exc)) from exc
    return options


def _parse_glossary(text: str) -> dict[str, str] | None:
    """A declared glossary's JSON, whitespace collapsed; None when empty."""
    parsed = json.loads(text) if text.strip() else {}
    if not isinstance(parsed, dict) or not all(isinstance(v, str) for v in parsed.values()):
        raise ValueError('expected a JSON object of strings, {"column": "meaning", "column=value": "phrase"}')
    if long := [k for k, v in parsed.items() if len(v) > GLOSSARY_VALUE_MAX]:
        raise ValueError(f"values longer than {GLOSSARY_VALUE_MAX} characters: {long}")
    return {" ".join(k.split()): " ".join(v.split()) for k, v in parsed.items() if v.strip()} or None


def resolve_declared_metadata(
    glossary: str | None, metadata: str | None, columns: Collection[str] | None = None, strict: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """The declared glossary (JSON text: ``{"column": "meaning", "column=value": "phrase"}``) and metadata document
    (``parse_metadata_doc``), each None when not given and "" to clear it: (``{"user_glossary", "metadata_doc"}`` for
    those given, None when cleared; warnings), fitted to the dataset's ``columns`` when given (``fit_declared``). A
    malformed value is refused (``invalid_options``)."""
    declared: dict[str, Any] = {}
    try:
        if glossary is not None:
            declared["user_glossary"] = _parse_glossary(glossary)
    except ValueError as exc:  # json.JSONDecodeError is one
        raise PipelineError("invalid_options", f"glossary: {exc}") from exc
    try:
        if metadata is not None:
            if len(metadata.encode("utf-8")) > METADATA_DOC_MAX_BYTES:
                raise ValueError(f"over {METADATA_DOC_MAX_BYTES // 1024} KB")
            declared["metadata_doc"] = parse_metadata_doc(metadata) if metadata.strip() else None
    except ValueError as exc:
        raise PipelineError("invalid_options", f"metadata document: {exc}") from exc
    return fit_declared(declared, columns, strict) if columns is not None else (declared, [])


def fit_declared(declared: dict[str, Any], columns: Collection[str], strict: bool = False, frame: Any = None) -> tuple[dict[str, Any], list[str]]:
    """``declared`` (``user_glossary`` / ``metadata_doc``, parsed) kept to the dataset's ``columns``: a column it does not
    hold, as itself, its band or a geo derivation (``wording.source_column``), is dropped with a warning, or refused for
    the glossary with ``strict`` (a PATCH: ``invalid_options``); with the dataset's ``frame`` (in memory while it is
    mined) a ``column=value`` phrase whose value the column never takes is a warning too. Returns (kept, warnings)."""
    kept, warnings, held = dict(declared), [], list(columns)

    def unknown(names: Collection[str]) -> list[str]:
        return sorted(set(names) - {source_column(c, names) for c in held})

    if user := kept.get("user_glossary"):
        if missing := unknown({k.split("=", 1)[0] for k in user}):
            message = f"glossary: no column named {missing} in the dataset (its columns: {held})"
            if strict:
                raise PipelineError("invalid_options", message)
            warnings.append(message + "; dropped")
            kept["user_glossary"] = user = {k: v for k, v in user.items() if k.split("=", 1)[0] not in missing} or None
        pairs = [k.split("=", 1) for k in user or {} if "=" in k]
        named = [(c, v) for c, v in pairs if frame is not None and c in frame and frame[c].dtype.kind not in "biufc"]  # categories only
        if absent := [f"{c}={v}" for c, v in named if v not in set(frame[c].astype(str))]:
            warnings.append(f"glossary: no row holds {absent}; kept")
    if doc := kept.get("metadata_doc"):
        if missing := unknown(doc["columns"]):
            warnings.append(f"metadata: {len(missing)} documented columns not among this level's, left out: {missing}")
            kept["metadata_doc"] = {**doc, "columns": {k: v for k, v in doc["columns"].items() if k not in missing}}
    return kept, warnings


def declared_glossary(batch: dict[str, Any]) -> dict[str, str]:
    """A dataset's declared names in one glossary (docs/02_discovery.md §2.1), what every reading and the prompt use: a
    documented column's Ukrainian title, else its display name; over it, for a metric the document does not name, its
    indicator ``<indicator> (<unit>)``; over everything, the explicit glossary."""
    doc = (batch.get("metadata_doc") or {}).get("columns", {})
    names = {c: f.get("title_uk") or f.get("display_name") for c, f in doc.items()}
    out = {c: v if len(v) <= GLOSSARY_VALUE_MAX else v[: GLOSSARY_VALUE_MAX - 1] + "…" for c, v in names.items() if v}  # a name, not a text
    for metric, e in (batch.get("indicators") or {}).items():
        out.setdefault(metric, f"{e['indicator']} ({e['unit']})" if e.get("unit") else e["indicator"])
    return out | (batch.get("user_glossary") or {})


def point_columns(batch: dict[str, Any]) -> tuple[str | None, str | None]:
    """The point columns a level's events lie at: as its signed profile resolves them (a geo level's ``geo.location``, a
    table level's left-out ``points``), and as its options name them (a geo level's locator, a table level's
    ``locator``), for a record stored without the first."""
    profile = batch.get("profile") or {}
    geo = batch.get("geo")
    signed = (profile.get("geo") or {}).get("location") if geo else profile.get("points")
    return signed, (geo.partition("@")[0] if geo else batch.get("locator") or "").strip() or None


def same_points(batch: dict[str, Any], owner: dict[str, Any]) -> bool:
    """Whether ``batch``'s rows are the events of ``owner``'s points (docs/06_graph_and_storage.md §6.1): one non-empty
    ``upload_id`` and one set of point columns, another pair of coordinates of the same file being another map; the
    columns as both signed profiles resolve them, else as both records' options name them."""
    if not owner.get("upload_id") or batch.get("upload_id") != owner["upload_id"]:
        return False
    (signed, named), (owner_signed, owner_named) = point_columns(batch), point_columns(owner)
    return signed == owner_signed if signed and owner_signed else named == owner_named


def resolve_levels(geo: str | None, table: bool, region_levels: tuple[int, ...], default_resolution: int) -> list[tuple[str | None, str]]:
    """The levels one upload makes, as (``geo`` option of the level's dataset, level): each listed resolution, finest
    first, then ``table`` (no geo) when asked. A coarser resolution must be one of ``GEO_REGION_LEVELS``."""
    if not geo:
        if table:
            raise PipelineError("invalid_options", "table=true needs the geo option: a plain upload is the table level already")
        return [(None, "table")]
    try:
        levels = parse_geo_levels(geo, default_resolution)
    except SpatialError as exc:
        raise PipelineError("invalid_options", str(exc)) from exc
    if stray := [r for _, r in levels[1:] if r not in region_levels]:
        raise PipelineError("invalid_options", f"coarser geo levels {stray} are not GEO_REGION_LEVELS {list(region_levels)}")
    return [(f"{loc}@{r}", str(r)) for loc, r in levels] + ([(None, "table")] if table else [])


def auto_level_hint(record: dict[str, Any], code: str) -> str:
    """What a failed level adds to its message: when ``@auto``'s finest level found nothing to mine, the nearest finer
    level the cell cap left out ("" otherwise)."""
    levels = record["profile"].get("geo", {}).get("levels") or {}
    if code in EMPTY_LEVEL and levels.get("capped") and str(levels["chosen"][0]) == str(record.get("level")):
        return f"; @auto left {record['geo'].partition('@')[0]}@{min(levels['capped'])} out for GEO_AUTO_MAX_CELLS alone, try it"
    return ""


def check_target(batches: list[dict[str, Any]], target: str | None) -> str | None:
    """``target`` stripped (None when empty), refused unless it names a READY dataset or an upload with batches."""
    target = (target or "").strip() or None
    if target and not any(target in (b.get("dataset_id"), b.get("upload_id")) and b["status"] in ("READY", "SKIPPED") for b in batches):
        raise PipelineError("invalid_options", f"supersedes {target}: no READY dataset or upload has this id")
    if target:
        check_removal(batches, [b for b in batches if target in (b.get("dataset_id"), b.get("upload_id"))])
    return target


def replaced_by(
    record: dict[str, Any], batches: list[dict[str, Any]], outcome: tuple[str, str | None]
) -> tuple[set[str], list[dict[str, Any]]] | None:
    """What ``record``'s batch replaces now (``targets``), its own ``outcome`` (status, error code) given: its
    ``supersedes`` once every level of the same upload call (one stored file) is done, one holds insights and none
    failed but for having nothing to mine (``EMPTY_LEVEL``), so the last level applies it; otherwise the old version
    stays, with a warning."""
    if not record.get("supersedes"):
        return None
    levels = [b for b in batches if b["batch_id"] != record["batch_id"] and b.get("source_path") == record["source_path"]]
    if any(b["status"] not in TERMINAL for b in levels):
        return None
    done = [outcome, *((b["status"], (b.get("error") or {}).get("code")) for b in levels)]
    if any(status == "FAILED" and code not in EMPTY_LEVEL for status, code in done) or all(status == "FAILED" for status, _ in done):
        record["warnings"].append(
            f"supersedes {record['supersedes']} not applied: a level of this upload failed or none holds insights; the old version stays"
        )
        return None
    try:
        return targets(record, batches, {b.get("dataset_id") for b in levels})
    except PipelineError as exc:  # busy: the batch commits, the old version stays
        record["warnings"].append(f"supersedes {record['supersedes']} not applied: {exc}")
        return None


def targets(record: dict[str, Any], batches: list[dict[str, Any]], own: set[str | None]) -> tuple[set[str], list[dict[str, Any]]] | None:
    """The datasets named by ``record['supersedes']`` (a dataset id, or every dataset of an upload id) and all their
    batches, never ``record``'s own dataset nor those in ``own``; None, with a warning, when none is left; ``busy`` while
    one of those batches is still to run."""
    target = record["supersedes"]
    every = [b for b in batches if b["batch_id"] != record["batch_id"]]
    ids = {b["dataset_id"] for b in every if target in (b.get("dataset_id"), b.get("upload_id")) and b.get("dataset_id")}
    ids -= own | {record.get("dataset_id")}
    if not ids:
        record["warnings"].append(f"supersedes {target}: nothing left to replace")
        return None
    theirs = [b for b in every if b.get("dataset_id") in ids or b.get("upload_id") == target]
    if any(b["status"] not in TERMINAL for b in theirs):
        raise PipelineError("busy", f"supersedes {target}: a batch of it is still to run")
    return ids, theirs


def layer_name(filename: str, taken: Collection[str]) -> str:
    """A context file's layer name: its stem as an identifier (``[a-z][a-z0-9_]``, at most 40 characters, the column
    ``context_<name>_band``), unique within the upload (``schools``, ``schools_2``)."""
    base = re.sub(r"[^0-9a-z]+", "_", Path(filename).stem.lower()).strip("_")[:40] or "layer"
    base = base if base[0].isalpha() else f"layer_{base}"
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    return name


def save_context(ws: Workspace, files: Sequence[tuple], max_bytes: int) -> list[dict[str, Any]]:
    """Context files (docs/02_discovery.md §2.9) — ``(filename, stream, locator, where)`` each —, each stored
    (``MAX_UPLOAD_MB`` apiece) or, given as the stored ``Path`` of the target's own file, read in place (``in_place``:
    no copy, never removed by ``discard_context``), named (``layer_name``; ``<stem>_<value>`` under a filter) and its
    points located once (``read_context``, ``where``'s rows): ``{name, path, geo[, where, in_place]}`` per layer, as the
    geo levels' records keep them. More than ``MAX_CONTEXT_LAYERS`` files, or a file without a point, refuses them all
    and keeps none."""
    if len(files) > MAX_CONTEXT_LAYERS:
        raise PipelineError("invalid_options", f"at most {MAX_CONTEXT_LAYERS} context files, got {len(files)}")
    layers: list[dict[str, str]] = []
    try:
        for filename, source, locator, where in files:
            named = f"{Path(filename).stem}_{where.partition('=')[2]}{Path(filename).suffix}" if where else filename
            name, in_place = layer_name(named, [layer["name"] for layer in layers]), isinstance(source, Path)
            path = source if in_place else ws.save_upload(filename, source, max_bytes)
            layers.append(
                {"name": name, "path": str(path), "geo": locator} | ({"where": where} if where else {}) | ({"in_place": True} if in_place else {})
            )
            read_context(name, Path(path), locator, where=where)
    except IngestionError as exc:  # no point, an unreadable file
        discard_context(layers)
        raise PipelineError(exc.code, str(exc)) from exc
    except Exception:  # too large, or the workspace gone meanwhile
        discard_context(layers)
        raise
    return layers


def resolve_counts(rows: Any, spec: str | None, miner: MinerConfig, geo: str | None = None) -> list[dict[str, Any]]:
    """An upload's ``count_values`` (``column=value,...``) checked against its rows (``ingestion.check_counts``): each
    value with its rows and status — ``ok`` with at least ``MIN_ROWS × MIN_CELL_POINTS`` rows, else ``sparse`` (no
    counter: too few events for a frequency, docs/02_discovery.md §2.9); none asked, none. With ``geo`` the rows are
    the located ones — what the cells will count — so a value held mostly by unlocated rows reads ``sparse``."""
    if rows is not None and spec and geo:
        lat, lon, _ = locate_points(rows, geo.partition("@")[0].strip())
        rows = rows[np.isfinite(lat) & np.isfinite(lon)]
    return check_counts(rows, spec, miner.min_rows * miner.min_cell_points)


def reading(record: dict[str, Any]) -> dict[str, Any]:
    """The options ``load_dataset`` reads a batch's stored file with, from its record: its processing's, a re-check's
    (``verify``). A geo level counts its ``count_values`` of status ``ok`` (no table level holds any); a record made
    before an option reads its default."""
    return {
        "filename": record["filename"],
        "bins": record.get("bins"),
        "categories": record.get("categories"),
        "geo": record.get("geo"),
        "strict": record.get("strict", True),
        "locator": record.get("locator"),
        "time": record.get("time"),
        "grid": record.get("grid"),
        "tz": record.get("tz"),
        "context": context_files(record),
        "context_ring": record.get("context_ring", 1),
        "where": record.get("where"),
        "count_values": [f"{c['column']}={c['value']}" for c in record.get("count_values", []) if c["status"] == "ok"],  # a geo level's
        "baseline": record.get("baseline"),
    }


def check_geo_options(context: Sequence, geo: str | None, ring: int | str | None, count_values: str | None, baseline: str | None) -> None:
    """The options of an upload's cells need its geo option — context files and per-type counters are counted into them,
    a zone baseline (``parse_baseline``'s) reads them —; and a disk needs context files."""
    if (context or count_values or baseline) and not geo:
        raise PipelineError("invalid_options", "context files, count_values or a zone baseline without geo: they belong to the geo cells")
    if str(ring or "").strip() and not context:
        raise PipelineError("invalid_options", "context_ring without context files: it is the disk their points are counted over")


def discard_context(layers: Sequence[dict[str, str]]) -> None:
    """Remove the stored files of an upload's context layers (a refused upload keeps none); a layer read in place is
    the target's own file and stays."""
    for layer in layers:
        if not layer.get("in_place"):
            Path(layer["path"]).unlink(missing_ok=True)


def upload_id(path: Path, where: str | None = None) -> str:
    """An upload's id: ``up-`` and the first twelve hex digits of its stored bytes' SHA-256 (the same bytes, one upload);
    of the bytes and ``where`` for a filtered reading, an upload of its own over the same file (docs/02_discovery.md §2.9)."""
    with path.open("rb") as stored:
        digest = hashlib.file_digest(stored, "sha256")
    if where:
        digest.update(f"where={where}".encode())
    return "up-" + digest.hexdigest()[:12]


def stored_copy(path: Path, batches: Sequence[dict[str, Any]]) -> Path:
    """The stored file of ``path``'s bytes: an earlier upload's, ``path`` itself removed — a filtered reading reads the
    file stored once (docs/02_discovery.md §2.9) —; else ``path``. Called once the upload is accepted, so a refusal
    never removes another upload's file."""
    size, digest = path.stat().st_size, upload_id(path)
    for b in batches:
        other = Path(b.get("source_path") or "")
        if other != path and other.is_file() and other.stat().st_size == size and upload_id(other) == digest:
            path.unlink()
            return other
    return path


def resolution(record: dict[str, Any]) -> int | None:
    """The H3 resolution a batch reads its points at (``<locator>@<r>``); None for a table, a plain file or a layer."""
    res = (record.get("geo") or "").partition("@")[2]
    return int(res) if res.isdigit() else None


def store_layer(
    ws: Workspace,
    name: str,
    stream: BinaryIO,
    geo: str | None,
    max_bytes: int,
    others: Sequence[str] = (),
    time: str | None = None,
    tz: str | None = None,
) -> dict[str, Any]:
    """A file kept as another type's points only (the upload's ``layer`` flag): stored, its points located once, one
    ``LAYER`` record of its own upload — never mined, never in the graph; a cross-type run reads it as a context layer
    (``cross_runs``). ``others``: the mining options sent with it, refused. ``time`` (and ``tz``) name its time column
    and zone: the column is read once for the record's ``span``, what a run's time rule compares (``layer_time``). A file
    without a point is not kept."""
    locator = (geo or "").partition("@")[0].strip()
    if not locator:
        raise PipelineError("invalid_options", "a layer needs the geo option: the columns of its points")
    if others:
        raise PipelineError("invalid_options", f"a layer is not mined: {', '.join(others)} do not apply")
    if tz and not time:
        raise PipelineError("invalid_options", "tz without time: name the time column it zones")
    path = ws.save_upload(name, stream, max_bytes)
    try:
        points = read_context(layer_name(name, ()), path, locator)[0]
        tz = parse_tz(tz)
        times = read_times(read_file(path)[0], time, tz) if time else None
    except IngestionError as exc:
        path.unlink(missing_ok=True)
        raise PipelineError(exc.code, str(exc)) from exc
    created = utc_now()
    record = {
        "batch_id": new_batch_id(created),
        "dataset_id": None,
        "filename": name,
        "source_path": str(path),
        "upload_id": upload_id(path),
        "level": "layer",
        "geo": locator,
        "points": int((np.isfinite(points.lat) & np.isfinite(points.lon)).sum()),
        **({"time": time, "tz": tz, "span": [t.tz_convert(tz or "UTC").isoformat() for t in (times.min(), times.max())]} if time else {}),
        "status": "LAYER",
        "stage_times": {"LAYER": created},
        "created_at": created,
        "metrics": {},
        "profile": {},
        "warnings": [],
        "error": None,
    }
    ws.save_batch(record)
    return record


def layer_time(stored: dict[str, Any], theirs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A layer's time rule for a run's context entry (docs/02_discovery.md §2.9): ``static`` for an upload without a time
    column; ``events`` with its ``time``, ``tz`` and ``span`` — a ``LAYER`` record's own, else the latest reading of the
    upload that has a time record (None when none has: the spans are then not compared)."""
    if not stored.get("time"):
        return {"time_rule": "static"}
    timed = max((b for b in theirs if b.get("temporal")), key=lambda b: b["created_at"], default=None)
    return {
        "time_rule": "events",
        "time": stored["time"],
        "tz": stored.get("tz"),
        "span": stored.get("span") or (timed["temporal"]["span"] if timed else None),
    }


def overlap_share(target: Sequence[str], layer: Sequence[str]) -> float:
    """The share of ``target``'s span (two ISO times) that ``layer``'s covers; a target of one instant is covered or not."""
    (tlo, thi), (lo, hi) = ([datetime.fromisoformat(s) for s in span] for span in (target, layer))
    covered = max(timedelta(0), min(thi, hi) - max(tlo, lo))
    return covered / (thi - tlo) if thi > tlo else float(lo <= tlo <= hi)


def time_warning(entry: dict[str, Any], target_span: Sequence[str] | None) -> str:
    """A run's warning for an event layer: it is read as static over its whole span (no window is mined); both spans and
    the share of the target's the layer's covers, or that a span is unknown and the two were not compared."""
    if entry.get("span") and target_span:
        lo, hi = entry["span"]
        return (
            f"{entry['name']}: an event layer read as static over its whole span {lo}..{hi}; "
            f"the target spans {target_span[0]}..{target_span[1]}, {overlap_share(target_span, entry['span']):.0%} of it covered"
        )
    return f"{entry['name']}: an event layer read as static; its span or the target's is unknown (no reading with a time record), not compared"


def cross_runs(
    ws: Workspace,
    batches: Sequence[dict[str, Any]],
    target: str,
    layers: Sequence[str],
    settings: Settings,
    levels: Sequence[int | str] | None = None,
    ring: int | str | None = None,
    align: str | None = None,
    layer_where: dict[str, str] | None = None,
    count_values: str | None = None,
    baseline: str | None = None,
    allow_coarse: bool = False,
) -> list[dict[str, Any]]:
    """A cross-type run of two stored uploads (docs/02_discovery.md §2.9): ``target``'s levels read again from its stored
    file with one layer (another upload, mined or ``LAYER``) as its context — a copy of its stored file, so deleting
    the layer's own upload leaves the run whole; a second layer is a second run (1:1). ``target`` is an upload id, or
    the dataset id of the reading the run is asked from (the card's): that record's options and trust, not the
    level's latest. The levels: ``levels`` when given (each one of the target's), otherwise the target's finest — a
    layer is counted from its points on the target's grid, so the levels a mined layer was read at say nothing about
    the run —, each read from its latest processed record, never a repeat's (``SKIPPED``). A level coarser
    than ``LINK_MIN_LEVEL`` is refused unless ``allow_coarse`` (scripts and eval; the console never sets it), then named
    in the run's warnings. One ``Engine.submit`` keyword set per level, finest first: the level's own record's options
    and trust, ``cross`` naming the layer's upload, the context entry's time rule (``layer_time``) and, for a target
    with a time column, a warning for an event layer (``time_warning``): read as static over its whole span. ``align``
    ``span`` counts an event layer's events within the target's span only (its own reading's time record) — the
    ``window`` of its entry. ``layer_where`` (``{upload id: column=value}``) reads a layer stored whole under a filter,
    as a filtered reading reads its own rows (``where`` on its entry); a layer of the target's own file is read in
    place, not copied. ``count_values`` (``column=value,...``) asks for per-type counters, checked against the
    target's located rows (``resolve_counts``) — an empty one asks for none —, and ``baseline`` (``map`` | ``zone``)
    what its metrics are read against; without them the run keeps the target level's own."""
    clicked = next((b for b in batches if b.get("dataset_id") == target and "cross" not in b and resolution(b) is not None), None)
    if clicked:  # the reading asked from: its level reads that record
        target = clicked["upload_id"]
    if len(layers) != 1 or target in layers:
        raise PipelineError("invalid_options", "one layer, another upload than the target: a second layer is a second run")
    if (align := (align or "").strip() or None) not in (None, "span"):
        raise PipelineError("invalid_options", f"align {align!r}: span (the target's) or none")
    try:
        ring = parse_ring(ring)
        chosen = {int(x) for x in levels or ()}
        layer_where = {u: w for u, spec in (layer_where or {}).items() if (w := parse_where(spec))}
        asked = {"baseline": parse_baseline(baseline)} if baseline else {}
    except (IngestionError, ValueError) as exc:
        raise PipelineError("invalid_options", f"cross-type run: {exc}") from exc
    own: dict[int, dict[str, Any]] = {}
    for b in sorted(batches, key=lambda b: b["created_at"]):  # per level its latest reading, a run's own batches and repeats (SKIPPED) aside
        if b.get("upload_id") == target and resolution(b) is not None and "cross" not in b and b.get("status") != "SKIPPED":
            own[resolution(b)] = b
    if clicked:
        own[resolution(clicked)] = clicked
    if not own:
        raise PipelineError("invalid_options", f"{target}: no upload with a geo level to read the layers on")
    if stray := sorted(set(layer_where) - set(layers)):
        raise PipelineError("invalid_options", f"layer_where names no layer of the run: {stray}")
    sources, rules = [], []
    for upload in layers:
        theirs = [b for b in batches if b.get("upload_id") == upload]
        stored = max(theirs, key=lambda b: b["created_at"], default=None)
        if stored is None or not Path(stored["source_path"]).is_file() or not (stored.get("geo") or stored.get("locator")):
            raise PipelineError("invalid_options", f"{upload}: no stored upload with points")
        if stored.get("where") and upload in layer_where:
            raise PipelineError("invalid_options", f"{upload}: a filtered reading already ({stored['where']}); a run filters a layer stored whole")
        locator = (stored.get("geo") or stored["locator"]).partition("@")[0]
        sources.append((stored["filename"], Path(stored["source_path"]), locator, stored.get("where") or layer_where.get(upload)))
        rules.append(layer_time(stored, theirs))
    if missing := sorted(chosen - set(own)):
        raise PipelineError("invalid_options", f"levels {missing} are not levels of {target} ({sorted(own, reverse=True)})")
    chosen = chosen or {max(own)}  # the target's finest
    if (coarse := sorted(r for r in chosen if r < LINK_MIN_LEVEL)) and not allow_coarse:
        raise PipelineError(
            "invalid_options",
            f"levels {coarse} are coarser than r{LINK_MIN_LEVEL}: a zone this wide already holds different places (allow_coarse runs them)",
        )
    picked = [own[r] for r in sorted(chosen, reverse=True)]  # the records the runs read, finest first
    try:  # asked for the run, checked as an upload checks them: the target's rows, its reading's filter
        rows = read_rows(Path(picked[0]["source_path"]), picked[0].get("where")) if (count_values or "").strip() else None
        counts = None if count_values is None else resolve_counts(rows, count_values, settings.miner, picked[0].get("geo"))
    except IngestionError as exc:
        raise PipelineError(exc.code, f"cross-type run: {exc}") from exc
    shared = {Path(b["source_path"]) for b in own.values()}  # the target's own file: a layer of it is read in place
    with ExitStack() as files:
        opened = [(n, p if p in shared else files.enter_context(p.open("rb")), loc, w) for n, p, loc, w in sources]
        context = save_context(ws, opened, settings.max_upload_bytes)
    for entry, rule in zip(context, rules):
        entry.update(rule)
    target_span = next((b["temporal"]["span"] for b in picked if b.get("temporal")), None)
    if align:
        if not target_span or context[0]["time_rule"] != "events":
            discard_context(context)
            raise PipelineError("invalid_options", "align=span needs a target with a time record and a layer with a time column")
        context[0]["window"], warnings = list(target_span), []
    else:
        warnings = [time_warning(e, target_span) for e in context if e["time_rule"] == "events"] if any(b.get("time") for b in own.values()) else []
    runs = []
    for b in picked:
        keys = ("filename", "bins", "categories", "geo", "time", "grid", "tz", "where", "count_values", "baseline")
        options = {k: b.get(k) for k in keys} | {"strict": b.get("strict", True)} | ({"count_values": counts} if counts is not None else {}) | asked
        inherited = {k: b[k] for k in TRUST_FIELDS if k in b} | {"cross": list(layers)}
        runs.append(
            options
            | {
                "path": Path(b["source_path"]),
                "upload": {"upload_id": target, "level": b["level"]},
                "context": context,
                "context_ring": ring,
                "inherited": inherited,
                "warnings": warnings
                + [
                    f"r{resolution(b)}: a layer link coarser than r{LINK_MIN_LEVEL}; a zone this wide already holds different places, a link on it is preliminary"
                ]
                * (resolution(b) < LINK_MIN_LEVEL),
            }
        )
    return runs


def context_files(record: dict[str, Any]) -> list[tuple]:
    """A record's context layers as ``load_dataset`` reads them: ``(layer, path, locator, window)``, the window an aligned
    run's ``(time, tz, first, last)`` (``cross_runs``), else None, and ``where``, a layer's row filter."""
    return [
        (c["name"], Path(c["path"]), c["geo"], (c["time"], c.get("tz"), *c["window"]) if "window" in c else None, c.get("where"))
        for c in record.get("context", ())
    ]


def cross_state(batches: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """What the registry says of cross-type runs and their layers, by batch id: a run naming an upload no record holds is
    ``stale`` — ``{layers, reason}``, ``superseded`` when a stored version names that upload, else ``deleted`` —, and
    every record of an upload a run reads as a layer counts those runs (``layer_of``); an upload's finest geo level, mined
    — READY, or none of its findings admitted alone — and not coarser than ``LINK_MIN_LEVEL``, can take a layer (``linkable``: the console's *+ контекстний шар*, a run of
    that level, ``cross_runs``'s default); a run reading a hidden layer names it (``hidden_by``). Read off the registry, never
    stored: a deletion rolled back leaves no mark, and the run, which holds copies of its layers, stays whole and
    reproducible (docs/06_graph_and_storage.md §6.8)."""
    stored, replaced = {b.get("upload_id") for b in batches}, {b.get("supersedes") for b in batches}
    hidden = {b.get("upload_id") for b in batches if b.get("hidden")}
    runs: dict[str, int] = {}
    finest: dict[str, int] = {}  # an upload's finest geo level, a run's own aside
    out: dict[str, dict[str, Any]] = {}
    for b in batches:
        crossed = b.get("cross", ()) if b.get("status") != "SKIPPED" else ()  # a repeat is no run: it reads nothing, hides nothing
        for upload in crossed:
            runs[upload] = runs.get(upload, 0) + 1
        if (r := resolution(b)) is not None and "cross" not in b:
            finest[b.get("upload_id")] = max(r, finest.get(b.get("upload_id"), r))
        if gone := [u for u in crossed if u not in stored]:
            out[b["batch_id"]] = {"stale": {"layers": gone, "reason": "superseded" if all(u in replaced for u in gone) else "deleted"}}
        if off := [u for u in crossed if u in hidden]:  # a run of a hidden layer: out of the answers and the views
            out.setdefault(b["batch_id"], {})["hidden_by"] = off
    for b in batches:
        if n := runs.get(b.get("upload_id")):
            out.setdefault(b["batch_id"], {})["layer_of"] = n
        mined = b.get("status") == "READY" or (b.get("error") or {}).get("code") == "no_viable_insights"  # read, mined: a layer may add
        if mined and "upload_id" in b and "cross" not in b and LINK_MIN_LEVEL <= (resolution(b) or -1) == finest.get(b["upload_id"]):
            out.setdefault(b["batch_id"], {})["linkable"] = True
    return out


def set_hidden(ws: Workspace, batches: Sequence[dict[str, Any]], upload_id: str, hidden: bool) -> list[str]:
    """A layer hidden from the answers and the views, or shown again (docs/06_graph_and_storage.md §6.8): ``hidden`` on
    every record of its upload, unsigned as ``stale`` is — no digest moves, nothing is rebuilt —, and its runs hidden
    through it (``cross_state``'s ``hidden_by``). Only a layer-only upload or one a run reads as a layer; twice is once."""
    mine = [b for b in batches if b.get("upload_id") == upload_id]
    if not mine:
        raise PipelineError("unknown_upload", upload_id)
    if all(b["status"] != "LAYER" for b in mine) and not any(upload_id in b.get("cross", ()) for b in batches):
        raise PipelineError("invalid_options", f"{upload_id}: no run reads it as a layer")
    for b in mine:  # a copy, saved whole: a failed write leaves the cached record as it was on disk
        ws.save_batch({**b, "hidden": hidden})
    return [b["batch_id"] for b in mine]


def check_removal(batches: Sequence[dict[str, Any]], picked: Sequence[dict[str, Any]]) -> None:
    """A hidden layer that runs still read is not removed — deleted or superseded —: its runs would come back into the
    answers with no hide left to hold them (docs/06_graph_and_storage.md §6.8); show it first."""
    hidden, ids = {b["upload_id"] for b in picked if b.get("hidden")}, {b["batch_id"] for b in picked}
    if hidden and any(b["batch_id"] not in ids and b.get("status") != "SKIPPED" and set(b.get("cross", ())) & hidden for b in batches):
        raise PipelineError("invalid_options", f"{sorted(hidden)}: a hidden layer read by runs; show it (hidden: false) before removing it")


def hidden_datasets(batches: Sequence[dict[str, Any]]) -> frozenset[str]:
    """The datasets of the runs that read a hidden layer: what a search leaves out (model A, docs/07_question_answering.md
    §7.4), read off the registry as ``cross_state`` is."""
    state = cross_state(batches)
    return frozenset(b["dataset_id"] for b in batches if b.get("dataset_id") and "hidden_by" in state.get(b["batch_id"], {}))


def cross_type_only(insights: Sequence[Insight]) -> tuple[list[Insight], list[Rejection]]:
    """A cross-type run's insights: those on a layer's band; the rest are the target's own (its levels hold them), refused
    as ``outside_cross_type``."""
    on = [i for i in insights if any(c.attribute.startswith("context_") for c in i.conditions)]
    kept = {i.id for i in on}
    return on, [
        Rejection(i.expression, "outside_cross_type", "no layer in the scope: the target's own levels hold it") for i in insights if i.id not in kept
    ]
