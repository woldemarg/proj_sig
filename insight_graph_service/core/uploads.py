"""One upload (docs/02_discovery.md §2.9, docs/06_graph_and_storage.md §6.8): the levels it makes and the version it
replaces. Functions of the options, the batch registry and one record; the engine applies what they decide."""

from __future__ import annotations

from typing import Any

from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.workspace import TERMINAL
from subgroup_miner.spatial import SpatialError, parse_geo_levels

EMPTY_LEVEL = frozenset({"invalid_schema", "no_numeric_targets", "no_candidates", "no_viable_insights"})  # a level with nothing to mine


def point_columns(batch: dict[str, Any]) -> tuple[str | None, str | None]:
    """The point columns a level's events lie at: as its signed profile resolves them (a geo level's ``geo.location``, a
    table level's left-out ``points``), and as its options name them (a geo level's locator, a table level's
    ``locator``), for a record stored without the first."""
    profile = batch.get("profile") or {}
    geo = batch.get("geo")
    signed = (profile.get("geo") or {}).get("location") if geo else profile.get("points")
    return signed, (geo.partition("@")[0] if geo else batch.get("locator") or "").strip() or None


def same_points(batch: dict[str, Any], owner: dict[str, Any]) -> bool:
    """Whether ``batch``'s rows are the events of ``owner``'s points (epic #114, contract 6): one non-empty
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
