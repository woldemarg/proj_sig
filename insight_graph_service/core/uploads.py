"""One upload (docs/02_discovery.md §2.9, docs/06_graph_and_storage.md §6.8): the levels it makes, the version it
replaces, a layer kept for a cross-type run and the run of two stored uploads. Functions of the options, the batch
registry and one record; the engine applies what they decide."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from insight_contracts import Insight, Rejection
from insight_contracts.wording import source_column
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.workspace import TERMINAL, Workspace, new_batch_id, utc_now
from subgroup_miner.ingestion import IngestionError, parse_ring, read_context
from subgroup_miner.spatial import SpatialError, parse_geo_levels

EMPTY_LEVEL = frozenset({"invalid_schema", "no_numeric_targets", "no_candidates", "no_viable_insights"})  # a level with nothing to mine
GLOSSARY_VALUE_MAX = 120  # characters of one declared glossary value
METADATA_DOC_MAX_BYTES = 64 * 1024  # a metadata document as uploaded
# context files of one upload, one forced band each (docs/02_discovery.md §2.9): a third multiplies the Bonferroni family,
# fills the validation budget with context cohorts and, on a wide table, empties the 3-conjunctions (spec §14.6)
MAX_CONTEXT_LAYERS = 2
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


def save_context(ws: Workspace, files: Sequence[tuple[str, BinaryIO, str]], max_bytes: int) -> list[dict[str, str]]:
    """Context files (docs/02_discovery.md §2.9) — ``(filename, stream, locator)`` each —, each stored (``MAX_UPLOAD_MB``
    apiece), named (``layer_name``) and its points located once: ``{name, path, geo}`` per layer, as the geo levels'
    records keep them. More than ``MAX_CONTEXT_LAYERS`` files, or a file without a point, refuses them all and keeps none."""
    if len(files) > MAX_CONTEXT_LAYERS:
        raise PipelineError("invalid_options", f"at most {MAX_CONTEXT_LAYERS} context files, got {len(files)}")
    layers: list[dict[str, str]] = []
    try:
        for filename, stream, locator in files:
            name = layer_name(filename, [layer["name"] for layer in layers])
            layers.append({"name": name, "path": str(ws.save_upload(filename, stream, max_bytes)), "geo": locator})
            read_context(name, Path(layers[-1]["path"]), locator)
    except IngestionError as exc:  # no point, an unreadable file
        discard_context(layers)
        raise PipelineError(exc.code, str(exc)) from exc
    except Exception:  # too large, or the workspace gone meanwhile
        discard_context(layers)
        raise
    return layers


def discard_context(layers: Sequence[dict[str, str]]) -> None:
    """Remove the stored files of an upload's context layers (a refused upload keeps none)."""
    for layer in layers:
        Path(layer["path"]).unlink(missing_ok=True)


def upload_id(path: Path) -> str:
    """An upload's id: ``up-`` and the first twelve hex digits of its stored bytes' SHA-256 (the same bytes, one upload)."""
    with path.open("rb") as stored:
        return "up-" + hashlib.file_digest(stored, "sha256").hexdigest()[:12]


def resolution(record: dict[str, Any]) -> int | None:
    """The H3 resolution a batch reads its points at (``<locator>@<r>``); None for a table, a plain file or a layer."""
    res = (record.get("geo") or "").partition("@")[2]
    return int(res) if res.isdigit() else None


def store_layer(ws: Workspace, name: str, stream: BinaryIO, geo: str | None, max_bytes: int, others: Sequence[str] = ()) -> dict[str, Any]:
    """A file kept as another type's points only (the upload's ``layer`` flag): stored, its points located once, one
    ``LAYER`` record of its own upload — never mined, never in the graph; a cross-type run reads it as a context layer
    (``cross_runs``). ``others``: the mining options sent with it, refused. A file without a point is not kept."""
    locator = (geo or "").partition("@")[0].strip()
    if not locator:
        raise PipelineError("invalid_options", "a layer needs the geo option: the columns of its points")
    if others:
        raise PipelineError("invalid_options", f"a layer is not mined: {', '.join(others)} do not apply")
    path = ws.save_upload(name, stream, max_bytes)
    try:
        points = read_context(layer_name(name, ()), path, locator)[0]
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


def cross_runs(
    ws: Workspace,
    batches: Sequence[dict[str, Any]],
    target: str,
    layers: Sequence[str],
    levels: Sequence[int | str] | None,
    ring: int | str | None,
    max_bytes: int,
) -> list[dict[str, Any]]:
    """A cross-type run of two stored uploads (docs/02_discovery.md §2.9): ``target``'s levels read again from its stored
    file with ``layers`` (at most ``MAX_CONTEXT_LAYERS`` other uploads, mined or ``LAYER``) as its context — copies of
    their stored files, so deleting a layer's own upload leaves the run whole. The levels: ``levels`` when given (each one
    of the target's), otherwise the target's finest and every level each mined layer was read at too (a layer is counted
    from its points on the target's grid, so its levels only say where it was looked at). One ``Engine.submit`` keyword
    set per level, finest first: the level's own record's options and trust, ``cross`` naming the layers' uploads."""
    if not 0 < len(layers) <= MAX_CONTEXT_LAYERS or len(set(layers)) < len(layers) or target in layers:
        raise PipelineError("invalid_options", f"one or {MAX_CONTEXT_LAYERS} layers, each another upload than the target and named once")
    try:
        ring = parse_ring(ring)
        chosen = {int(x) for x in levels or ()}
    except (IngestionError, ValueError) as exc:
        raise PipelineError("invalid_options", f"cross-type run: {exc}") from exc
    own: dict[int, dict[str, Any]] = {}
    for b in sorted(batches, key=lambda b: b["created_at"]):  # per level its latest reading, a run's own batches aside
        if b.get("upload_id") == target and resolution(b) is not None and "cross" not in b:
            own[resolution(b)] = b
    if not own:
        raise PipelineError("invalid_options", f"{target}: no upload with a geo level to read the layers on")
    sources, read_at = [], []
    for upload in layers:
        theirs = [b for b in batches if b.get("upload_id") == upload]
        stored = max(theirs, key=lambda b: b["created_at"], default=None)
        if stored is None or not Path(stored["source_path"]).is_file() or not (stored.get("geo") or stored.get("locator")):
            raise PipelineError("invalid_options", f"{upload}: no stored upload with points")
        sources.append((stored["filename"], Path(stored["source_path"]), (stored.get("geo") or stored["locator"]).partition("@")[0]))
        read_at += [{resolution(b) for b in theirs} - {None}] if stored["status"] != "LAYER" else []
    if missing := sorted(chosen - set(own)):
        raise PipelineError("invalid_options", f"levels {missing} are not levels of {target} ({sorted(own, reverse=True)})")
    if not chosen:  # the target's finest, and the levels every mined layer was read at too
        chosen = {max(own)}.union(set(own).intersection(*read_at) if read_at else ())
    with ExitStack() as files:
        context = save_context(ws, [(name, files.enter_context(path.open("rb")), locator) for name, path, locator in sources], max_bytes)
    runs = []
    for b in (own[r] for r in sorted(chosen, reverse=True)):
        options = {k: b.get(k) for k in ("filename", "bins", "categories", "geo", "time", "grid", "tz")} | {"strict": b.get("strict", True)}
        inherited = {k: b[k] for k in TRUST_FIELDS if k in b} | {"cross": list(layers)}
        runs.append(
            options
            | {
                "path": Path(b["source_path"]),
                "upload": {"upload_id": target, "level": b["level"]},
                "context": context,
                "context_ring": ring,
                "inherited": inherited,
            }
        )
    return runs


def cross_type_only(insights: Sequence[Insight]) -> tuple[list[Insight], list[Rejection]]:
    """A cross-type run's insights: those on a layer's band; the rest are the target's own (its levels hold them), refused
    as ``outside_cross_type``."""
    on = [i for i in insights if any(c.attribute.startswith("context_") for c in i.conditions)]
    kept = {i.id for i in on}
    return on, [
        Rejection(i.expression, "outside_cross_type", "no layer in the scope: the target's own levels hold it") for i in insights if i.id not in kept
    ]
