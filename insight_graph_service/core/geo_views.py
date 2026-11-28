"""The map's reads (docs/08_interface.md §8.7): what the console's map draws of a READY geo dataset, from its stored files
(``geo.npz``, ``covers.npz``, ``cells.geojson``) and the graph of one commit.

Functions of the workspace and a ``DualGraph``; nothing here writes. The engine guards each one (a degraded workspace serves
no map) and passes the graph of one commit, so one answer never mixes two commits. What one commit derives once — an
upload's insights projected on a dataset's cells, the overlay — lives in the graph's memo.
"""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np

from graph_query_engine.graph import DualGraph, declared_words, insight_reading, primary_anchor, theme_glossary
from graph_query_engine.vertical import affinity, restating, scale_chain, scale_label
from insight_contracts import Insight, PhenomenonThresholds, ZonePlace, dataset_node_id, region_node_id, wording
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.geosparql import build_turtle
from insight_graph_service.core.uploads import context_files, hidden_datasets, point_columns, same_points, upload_id
from insight_graph_service.core.workspace import Workspace
from subgroup_miner.config import MinerConfig
from subgroup_miner.context import points_near
from subgroup_miner.ingestion import IngestionError, read_context, table_rows, zone_counts
from subgroup_miner.spatial import SpatialError, compactness
from subgroup_miner.spatial_plane import (
    cell_boundary,
    cell_centres,
    cell_column_names,
    drawn_zone_lists,
    event_regions,
    footprint,
    geometry_to_cells,
    link_min_events,
    parse_zone,
    points_in_zone,
    region_features,
    smoothed_lift,
    zone_measures,
)


def geo_datasets(ws: Workspace) -> list[dict[str, Any]]:
    """READY datasets analysed as H3 cells, with what the map needs to offer them; a hidden layer's runs left out."""
    out, hidden = [], hidden_datasets(ws.list_batches())
    for b in ws.list_batches():
        geo = (b.get("profile") or {}).get("geo")
        if b.get("status") == "READY" and geo and b["dataset_id"] not in hidden:
            out.append(
                {
                    "dataset_id": b["dataset_id"],
                    "upload_id": b.get("upload_id"),
                    "filename": b["filename"],
                    **geo,
                    "metrics": b["profile"].get("numerics", []),
                }
            )
    return out


def _geo_ready(ws: Workspace, dataset_id: str) -> bool:
    """Only a dataset with a READY geo batch is served: its files survive a failed batch, its listing does not."""
    return any(b["dataset_id"] == dataset_id and b["status"] == "READY" and (b.get("profile") or {}).get("geo") for b in ws.list_batches())


def _geo(ws: Workspace, dataset_id: str, keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    geo = ws.load_geo(dataset_id, keys) if _geo_ready(ws, dataset_id) else None
    if geo is None:
        raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
    return geo


def _owner(ws: Workspace, dataset_id: str) -> dict[str, Any]:
    """The dataset's READY batch, or ``unknown_dataset``: a deletion may take it after ``_geo`` read its files."""
    owner = ws.ready_batch_for(dataset_id)
    if owner is None:
        raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
    return owner


def geo_profile(loaded: Any, miner: MinerConfig) -> dict[str, Any]:
    """A geo level's ``profile.geo`` (docs/02_discovery.md §2.9), written to its batch record: what ``geo_datasets`` offers;
    ``types`` keeps the values a per-type counter would count (as many events as ``check_counts`` asks) and its
    ``column=value,…`` spec reads back as they are (no ``,``, no edge spaces, no ``=`` in the column)."""
    ctx, least = loaded.spatial, miner.min_rows * miner.min_cell_points
    types = {
        col: {v: n for v, n in values.items() if n >= least and v == v.strip() and v and "," not in v}
        for col, values in ctx.types.items()
        if "=" not in col and col == col.strip()
    }
    return {
        "option": loaded.geo,
        "resolution": ctx.resolution,
        "location": ctx.location,
        "cells": len(ctx.cells),
        "points": int((ctx.point_cell >= 0).sum()),
        "sparse_cells": ctx.sparse_cells,
        "min_cell_points": ctx.min_cell_points,
        "base": "H3 cells holding at least one point",
        "moran_i": {k: round(v, 4) for k, v in ctx.moran.items()},
        "glossary": ctx.glossary,
        **({"exposure": ctx.exposure} if ctx.exposure else {}),  # what a per-type count is per (docs/02_discovery.md §2.9)
        "types": {col: values for col, values in types.items() if values},
    }


def place_scores(
    ws: Workspace, graph: DualGraph, cells: list[str] | None, geometry: dict[str, Any] | None, resolution: int | None
) -> dict[str, ZonePlace] | None:
    """How every geo pattern lies in a map zone (``zone_measures``, docs/07_question_answering.md §7.2), over the geo
    datasets of at most ``resolution`` (None: all), and every placed base pattern's by the cells holding its events on the geo level
    that places it, of at most ``resolution`` too; ``geometry`` becomes the cells of the finest of them. None: no zone."""
    if cells is None and geometry is None:
        return None
    levels = sorted(
        {
            (graph.insight(pid).dataset_id, r)
            for pid in graph.insights
            if (r := graph.facts(pid).resolution) is not None and (resolution is None or r <= resolution)
        }
    )
    placed = graph.memoised(("placed_by", ""), lambda: _placed_by(graph))
    res_of = {by: ((ws.ready_batch_for(by) or {}).get("profile") or {}).get("geo", {}).get("resolution") for _, by in placed}
    tables = sorted(t for t in placed if res_of[t[1]] is not None and (resolution is None or res_of[t[1]] <= resolution))
    try:
        finest = max([r for _, r in levels] + [res_of[by] for _, by in tables], default=0)
        zone = parse_zone(cells) if cells is not None else geometry_to_cells(geometry or {}, finest)
    except SpatialError as exc:
        raise PipelineError("invalid_options", str(exc)) from exc
    if not levels and not tables:
        return {}
    try:
        scores = _zone_scores_cached(ws, zone, tuple(levels), tuple(tables))
    except FileNotFoundError:  # a dataset deleted while this was read: this answer only, never cached
        scores = _zone_scores(ws, zone, tuple(levels), tuple(tables), strict=False)
    kept = set().union(*(placed[t] for t in tables))
    return {pid: v for pid, v in scores.items() if pid in kept or graph.facts(pid).resolution is not None}


def _placed_by(graph: DualGraph) -> dict[tuple[str, str], set[str]]:
    """(table level, the geo level placing it) -> its placed patterns: the base patterns' IN_REGION by events (#119)."""
    placed: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for e in graph.edges:
        if e["type"] == "IN_REGION" and (by := e["props"].get("placed_by")):
            placed[(graph.insight(e["source"]).dataset_id, by)].add(e["source"])
    return placed


def _zone_scores(
    ws: Workspace, zone: frozenset[str], levels: tuple[tuple[str, int], ...], tables: tuple[tuple[str, str], ...], strict: bool = True
) -> dict[str, ZonePlace]:
    """``zone_measures`` over the stored cells, points and extents of each geo dataset in ``levels``, and over each
    (table level, geo level placing it) of ``tables`` by the table's patterns' events on that level; ``strict``: a
    missing file raises ``FileNotFoundError`` instead of leaving its patterns out."""
    out: dict[str, ZonePlace] = {}
    for dataset_id, by in [(d, None) for d, _ in levels] + list(tables):
        geo, covers = ws.load_geo(by or dataset_id, ("cells", "point_cell", "lonlat")), ws.load_covers(dataset_id)
        if geo is None or not covers:
            if strict:
                raise FileNotFoundError(dataset_id)
            continue
        in_zone = points_in_zone(geo["lonlat"], zone)
        out.update(zone_measures(geo["cells"], geo["point_cell"], in_zone, covers, zone, by_points=by is not None))
    return out


# cached on its arguments: a dataset id fixes its files (a READY id is never mined again, its duplicate is SKIPPED; one
# uploaded again after its deletion is mined to the same files), and a commit that changes the datasets changes them
_zone_scores_cached = functools.lru_cache(maxsize=64)(_zone_scores)


def _export_reading(
    ws: Workspace, dataset_id: str, resolution: int | None, layer: str | None = None
) -> tuple[dict[str, Any], tuple[Path, str, str | None, tuple[str, str | None, str, str] | None], int]:
    """What the reader of a geo dataset's export must know of its table, the reading of its rows and the cell resolution
    of the export: the dataset's own by default; only a resolution some geo dataset is analysed at is taken (a finer one
    over a wide zone is millions of cells). A hidden layer's dataset is not served, as the map does not offer it.
    ``layer``: a layer link run's context layer by its name, its points read as the run read them (its filter, an aligned
    run's window), instead of the dataset's own rows; ``ring`` the disk its band counts over. A stored file that no longer
    holds its upload's bytes is ``source_changed``: its insights describe another table (docs/13_trust.md, P5)."""
    record = _owner(ws, dataset_id)
    geo = (record.get("profile") or {}).get("geo")
    if not geo or dataset_id in hidden_datasets(ws.list_batches()):
        raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
    if resolution is None:
        resolution = geo["resolution"]
    elif resolution not in (known := sorted({d["resolution"] for d in geo_datasets(ws)})):
        raise PipelineError("invalid_options", f"resolution {resolution} is not a geo dataset's: {known}")
    table = {"dataset_id": dataset_id, "upload_id": record.get("upload_id"), "filename": record["filename"], "dataset_resolution": geo["resolution"]}
    if layer is not None:
        entry = next((c for c in context_files(record) if c[0] == layer), None)
        if entry is None:
            raise PipelineError("unknown_dataset", f"{dataset_id} holds no layer {layer!r}")
        _, path, locator, window, where = entry
        table |= {
            "layer": layer,
            "filename": f"{layer}{path.suffix}",
            "upload_id": None,
            "ring": record.get("context_ring", 1),
        }  # a run before #307 counted a ring
        span = {"window": list(window[2:])} if window else {}
        return table | {"time": window and window[0], "where": where} | span, (path, locator, where, window), resolution
    reading = (Path(record["source_path"]), point_columns(record)[1] or "auto", record.get("where"), None)
    # ponytail: the dataset's own file alone; a layer's copy has no digest of its own on the record (the run's id mixes it in)
    if record.get("upload_id") and reading[0].is_file() and upload_id(reading[0], record.get("where")) != record["upload_id"]:
        raise PipelineError("source_changed", f"{dataset_id}: its stored file no longer holds the bytes its insights were mined from")
    return table | {"time": record.get("time"), "where": record.get("where")}, reading, resolution


def geo_table_rows(ws: Workspace, dataset_id: str, resolution: int | None = None, layer: str | None = None) -> tuple[bytes, dict[str, Any]]:
    """A geo dataset's rows as CSV with a ``row_id`` and their cell columns of ``resolution`` (``ingestion.table_rows``),
    and what the reader must know of the table: what an analysis outside SIG reads; tables exported at one resolution share
    their cell columns. It does not depend on a zone: a zone is a cell list of its own (``geo_zone_cells``)."""
    table, reading, resolution = _export_reading(ws, dataset_id, resolution, layer)
    try:
        data, counts = table_rows(*reading, resolution)
    except (SpatialError, IngestionError) as exc:
        raise PipelineError("invalid_options", str(exc)) from exc
    return data, counts | table


def geo_zone_counts(
    ws: Workspace, dataset_id: str, zones: list[tuple[str, dict[str, Any]]], resolution: int | None = None, layer: str | None = None
) -> dict[str, Any]:
    """How many of a geo dataset's rows each drawn zone (id, GeoJSON Polygon) holds at ``resolution``, by cell
    (``ingestion.zone_counts``), and what ``geo_table_rows`` tells of the table: the counts a reader states beside the
    zones' cell lists (``geo_zone_cells``)."""
    table, reading, resolution = _export_reading(ws, dataset_id, resolution, layer)
    try:
        return zone_counts(*reading, resolution, zones) | table
    except (SpatialError, IngestionError) as exc:
        raise PipelineError("invalid_options", str(exc)) from exc


def geo_zone_cells(ws: Workspace, dataset_id: str, zones: list[tuple[str, dict[str, Any]]], resolution: int | None = None) -> bytes:
    """The drawn zones as cell lists at ``resolution`` (``drawn_zone_lists``), with rows or without: CSV
    ``zone_id,h3_r<resolution>``, each zone's cells under its id; every ``zone_id`` is a drawn zone. A row of a table
    exported at that resolution is in a zone when its cell is listed for it: no row is stamped and no polygon goes out."""
    _, _, resolution = _export_reading(ws, dataset_id, resolution)
    try:
        lists = drawn_zone_lists(zones, resolution)
    except SpatialError as exc:
        raise PipelineError("invalid_options", str(exc)) from exc
    lines = [f"zone_id,{cell_column_names(resolution)[0]}", *(f"{zone_id},{c}" for zone_id, cells in lists.items() for c in sorted(cells))]
    return ("\n".join(lines) + "\n").encode("utf-8")


def point_times(times: Any) -> np.ndarray | None:
    """Each point's time (``LoadedDataset.time``, UTC) as epoch milliseconds, the int64 minimum (NaT) where it does not read; None
    without a time column: what ``geo.npz`` keeps for ``geo_period_counts``."""
    return None if times is None else times.to_numpy(dtype="datetime64[ms]").view(np.int64)


def geo_period_counts(ws: Workspace, dataset_id: str, period: tuple[datetime, datetime]) -> dict[str, Any]:
    """Each cell's points in ``period`` ([start, end), wall clock: ``temporal.parse_period``), read on the dataset's
    calendar as an asked period is (docs/07_question_answering.md §7.1): ``{counts: [per cell row], points, timed}``;
    ``counts`` is None for a dataset without a time column (``timed`` false) or whose points were stored without their times."""
    counts = period_cell_counts(ws, dataset_id, period)
    if counts is None:
        return {"counts": None, "points": 0, "timed": bool(_owner(ws, dataset_id).get("temporal"))}
    return {"counts": counts.tolist(), "points": int(counts.sum()), "timed": True}


def idle_insights(
    ws: Workspace,
    graph: DualGraph,
    dataset_id: str,
    period: tuple[datetime, datetime],
    thresholds: PhenomenonThresholds,
    datasets: tuple[str, ...] = (),
) -> list[str]:
    """The insights the map lists (``geo_overlay``'s region ``ids``) that do not act in ``period``: a trajectory whose core
    does not meet it and no event inside it (``DualGraph.members``, docs/07_question_answering.md §7.2); one with
    neither a core nor an event always acts."""
    listed = {
        pid for f in geo_overlay(ws, graph, dataset_id, thresholds, datasets)["features"] for ids in f["properties"]["ids"].values() for pid in ids
    }
    members = graph.members(period)
    acting = {*members["active"], *(pid for pid, _ in members["peaked"]), *(pid for pid, _ in members["dropped"])}
    timed = {pid for pid, t in graph.trajectories.items() if t.get("core") or t["events"]}
    return sorted(listed & timed - acting)


def points_in_period(ws: Workspace, dataset_id: str, period: tuple[datetime, datetime]) -> np.ndarray | None:
    """Which of the dataset's points lie in ``period`` ([start, end), wall clock, read on the dataset's calendar as an
    asked period is: docs/07_question_answering.md §7.1); None without their times (no time column, or stored before
    ``geo.npz`` kept them)."""
    temporal = _owner(ws, dataset_id).get("temporal") or {}
    geo = _geo(ws, dataset_id, ("point_time",))
    if not temporal or "point_time" not in geo:
        return None
    zone = ZoneInfo(temporal["tz"]) if temporal.get("tz") else UTC
    lo_ms, hi_ms = (int(t.replace(tzinfo=zone).timestamp() * 1000) for t in period)
    return np.asarray((geo["point_time"] >= lo_ms) & (geo["point_time"] < hi_ms))  # NaT, the int64 minimum, lies before every period


def period_cell_counts(ws: Workspace, dataset_id: str, period: tuple[datetime, datetime]) -> np.ndarray | None:
    """Each cell row's points in ``period`` (``points_in_period``); None without the points' times."""
    inside = points_in_period(ws, dataset_id, period)
    if inside is None:
        return None
    geo = _geo(ws, dataset_id, ("cells", "point_cell"))
    cell = geo["point_cell"]
    return np.bincount(cell[inside & (cell >= 0)], minlength=len(geo["cells"]))


def geo_cells_pack(ws: Workspace, dataset_id: str) -> bytes:
    """The map's cell package (docs/08_interface.md §8.1): the dataset's ``cells.geojson`` as columns — each cell's ring
    as its first vertex and integer offsets in ``10**-scale`` degrees, each column of the analysed table once, the
    ``*_band`` texts left out — with its ``bbox`` and ``patterns``; built once per file, which its batch writes and
    nothing changes afterwards."""
    path = ws.geo_cells_path(dataset_id)
    if not (_geo_ready(ws, dataset_id) and path.is_file()):
        raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
    return _pack(str(path), path.stat().st_mtime_ns)


@functools.lru_cache(maxsize=8)
def _pack(path: str, _mtime: int) -> bytes:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload["features"]
    scale = 5 if payload["meta"]["resolution"] <= 12 else 7  # a step far below a cell's edge: 1.1 m, 1.1 cm past r12
    names = [k for k in (features[0]["properties"] if features else {}) if k != "row" and not k.endswith("_band")]
    rings = []
    for f in features:
        (x0, y0), *rest = f["geometry"]["coordinates"][0][:-1]
        x0, y0 = round(x0, scale), round(y0, scale)  # the offsets from the vertex as sent: each coordinate within half a step
        rings.append([x0, y0, *(round((c - o) * 10**scale) for x, y in rest for c, o in ((x, x0), (y, y0)))])
    columns = {k: [f["properties"][k] for f in features] for k in names}
    pack = {"columns": columns, "rings": rings, "scale": scale, "bbox": payload["bbox"], "patterns": payload["patterns"]}
    return json.dumps(pack, separators=(",", ":"), ensure_ascii=False).encode()


def geo_cells(ws: Workspace, dataset_id: str, loaded: Any) -> dict[str, Any]:
    """The dataset's cells as a GeoJSON FeatureCollection with the journaled patterns' extents; properties are the columns
    of the analysed cell table (``loaded.frame``) but the ``*_band`` texts no view reads, read once, when the batch writes
    the file."""
    ctx, frame = loaded.spatial, loaded.frame
    cols = {c: frame[c].to_numpy(dtype=float if frame[c].dtype.kind in "fiu" else str) for c in frame.columns if not c.endswith("_band")}
    features = []
    for i, cell in enumerate(ctx.cells):
        props: dict[str, Any] = {"h3": str(cell), "row": i}
        for name, values in cols.items():
            v = values[i]
            props[name] = (None if not np.isfinite(v) else round(float(v), 4)) if values.dtype.kind == "f" else str(v)
        props["points"] = int(ctx.events[i])  # the hover's events: as counted, whatever the baseline
        features.append({"type": "Feature", "id": i, "geometry": {"type": "Polygon", "coordinates": [cell_boundary(str(cell))]}, "properties": props})
    lonlat = ctx.lonlat[ctx.point_cell >= 0]  # never empty: build_cells refuses a level without a valid point
    west, east = lon_span(lonlat[:, 0])
    bbox = [west, float(lonlat[:, 1].min()), east, float(lonlat[:, 1].max())]
    covers = ws.load_covers(dataset_id)
    patterns = [
        _pattern_json(
            Insight.from_record(r),
            covers[r["id"]],
            spatial={**r["provenance"].get("spatial", {}), "compactness": round(compactness(covers[r["id"]], ctx.nbr_ptr, ctx.nbr_idx), 4)},
        )
        for r in ws.patterns()
        if r["dataset_id"] == dataset_id and r["id"] in covers
    ]
    return {
        "type": "FeatureCollection",
        "features": features,
        "bbox": bbox,
        "meta": {"resolution": ctx.resolution, "location": ctx.location, "columns": list(frame.columns)},
        "patterns": patterns,
    }


def lon_span(lons: np.ndarray) -> tuple[float, float]:
    """West and east of the narrowest longitude interval holding every value: west > east across ±180°, as a GeoJSON
    ``bbox`` writes it (RFC 7946 §5.2)."""
    x = np.unique(lons)
    gaps = np.diff(x)
    if not len(gaps) or 360 - (x[-1] - x[0]) >= gaps.max():  # the gap round the back is the widest: no crossing
        return float(x[0]), float(x[-1])
    i = int(gaps.argmax())
    return float(x[i + 1]), float(x[i])


def _props(graph: DualGraph, pid: str) -> dict[str, Any]:
    """The props of a pattern's Dataset node (empty for a dataset the graph lost)."""
    props: dict[str, Any] = graph.nodes.get(dataset_node_id(graph.insight(pid).dataset_id), {"props": {}})["props"]
    return props


def _pattern_json(ins: Insight, rows: np.ndarray, **extra: Any) -> dict[str, Any]:
    """An insight as the map lists it (``geo_cells``, ``geo_projected``): its cell rows, scope, shift, support and weight."""
    return {
        "id": ins.id,
        "dataset_id": ins.dataset_id,
        "rows": rows.tolist(),
        "scope": [f"{c.attribute} = {c.value}" for c in ins.conditions],
        "target": ins.target,
        "effect_size": ins.effect_size,
        "phenomenon_type": ins.phenomenon_type,
        "support": ins.support,
        "weight": ins.weight,
        **extra,
    }


LAYER_POINTS_MAX = 2000  # a focused insight's layer points drawn beside it, at most


def layer_points(record: dict[str, Any], cells: np.ndarray) -> list[dict[str, Any]]:
    """A run's layers' points in and around ``cells`` (their disk one ring wider than the run's), as the run counted them
    (its window and filter): ``[{name, n_points, points: [[lon, lat], …]}]``, ``n_points`` all of them and ``points`` at
    most ``LAYER_POINTS_MAX``, every k-th in file order as an insight's own points are thinned, so the layer's density
    holds over the whole disk and span; the map draws the two layers side by side. A dataset of no layer has none."""
    if not record.get("cross") or not len(cells):
        return []
    out = []
    for name, path, locator, window, where in context_files(record):
        source = read_context(name, Path(path), locator, window, where)[0]
        near = points_near(source, cells, record.get("context_ring", 1) + 1)
        drawn = near[:: max(1, int(np.ceil(len(near) / LAYER_POINTS_MAX)))]
        out.append(
            {"name": name, "n_points": int(len(near)), "points": [[round(float(source.lon[i]), 6), round(float(source.lat[i]), 6)] for i in drawn]}
        )
    return out


def geo_pattern(
    ws: Workspace,
    graph: DualGraph,
    dataset_id: str,
    pattern_id: str,
    max_points: int,
    rings: bool = False,
    period: tuple[datetime, datetime] | None = None,
) -> dict[str, Any]:
    """A pattern's cells (rows of the cell table) and a deterministic sample of the points inside them; an insight of
    another level of the upload (``geo_projected``): the cells it is drawn by and its own points inside them. An insight
    whose record fails its digest is drawn by neither (``unknown_dataset``), as the projection and the window skip it.
    With the ``level`` it was found at and its numbers, and on request its cells' ``rings``, so a map that holds no cell
    of this dataset draws and describes it. With a ``period``, the points are those in it (``points_in_period``) when the
    dataset has their times (``in_period``)."""
    try:
        geo = _geo(ws, dataset_id)
        rows = ws.load_covers(dataset_id).get(pattern_id)
        if rows is not None and (pattern_id not in graph.nodes or graph.tampered(pattern_id)):
            raise PipelineError("unknown_dataset", f"no intact pattern {pattern_id} in {dataset_id}")
        if rows is not None:
            inside, level = np.flatnonzero(np.isin(geo["point_cell"], rows)), f"res {graph.facts(pattern_id).resolution}"
        else:
            found = _projection(ws, graph, dataset_id)["patterns"].get(pattern_id)
            batch = ws.ready_batch_for(graph.insight(pattern_id).dataset_id) if found else None
            members = member_rows(ws, batch, _owner(ws, dataset_id), len(geo["point_cell"])) if batch else None
            own = ws.load_covers(batch["dataset_id"]).get(pattern_id) if batch and members else None
            if own is None or members is None:  # unknown here, or its dataset deleted since this commit (its record, or its covers, gone)
                raise PipelineError("unknown_dataset", f"no pattern {pattern_id} in {dataset_id}")
            rows, level = found.events, found.level
            src = members(own)
            inside = src[np.isin(geo["point_cell"][src], rows)]
        timed = points_in_period(ws, dataset_id, period) if period is not None else None
        if timed is not None:
            inside = inside[timed[inside]]
        step = max(1, int(np.ceil(len(inside) / max_points)))
        points = geo["lonlat"][inside[::step]]
        ins = graph.insight(pattern_id)
        return {
            "pattern_id": pattern_id,
            "cells": [str(c) for c in geo["cells"][rows]],
            "rows": rows.tolist(),
            "n_points": int(len(inside)),
            "in_period": timed is not None,  # the points are those of the asked period
            "points": np.round(points.astype(float), 5).tolist(),
            "level": level,
            "effect_size": ins.effect_size,
            "phenomenon_type": ins.phenomenon_type,
            "support": ins.support,
            "weight": ins.weight,
            # the insight's run's layer beside it, whichever map draws it: a base map holds a run's insights, not its layer
            "layer_points": layer_points(ws.ready_batch_for(ins.dataset_id) or {}, geo["cells"][rows]),
        } | ({"rings": [cell_boundary(str(c)) for c in geo["cells"][rows]]} if rings else {})
    except FileNotFoundError as exc:  # a deletion moved the dataset's files while this read them: as a read after it
        raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset") from exc


def geo_overlay(ws: Workspace, graph: DualGraph, dataset_id: str, thresholds: PhenomenonThresholds, datasets: tuple[str, ...] = ()) -> dict[str, Any]:
    """What the map draws over a geo dataset: every region holding a cell of the datasets ``map_set`` keeps, at each
    level of ``_region_sets``, as GeoJSON with the rows of ``dataset_id``'s cells inside it, ``ids``, the IN VIEW rows a
    view of that region alone lists per dataset (``_rows``; the datasets' insights are of different uploads, so they
    add), and ``n``, their number; and each pattern of the dataset and of the table levels of its upload, and each one
    another dataset of the set lists, with its credibility, integrity, scale and reading — the latter with ``map_dataset``,
    the dataset whose ``/patterns`` draws it. Derived once per commit and hidden set (model A: a hidden layer's runs
    are not listed, not counted, not drawn)."""
    hidden, kept = hidden_datasets(ws.list_batches()), map_set(ws, dataset_id, datasets)  # map_set offers no hidden run

    def build() -> dict[str, Any]:
        geo = _geo(ws, dataset_id, ("cells",))  # inside: a memo hit reads no file
        owner = _owner(ws, dataset_id)
        tables = {b["dataset_id"] for b in ws.list_batches() if b.get("status") == "READY" and b.get("level") == "table" and same_points(b, owner)}
        pids = [n["id"] for n in graph.of_kind("Pattern") if n["props"]["dataset_id"] == dataset_id or n["props"]["dataset_id"] in tables]
        held: defaultdict[str, dict[str, list[str]]] = defaultdict(dict)
        levels = _region_sets(ws, graph, dataset_id, hidden).keys()  # the map's: another dataset's finer ones would lie inside its cells
        for ds in kept:
            for lv, sets in _region_sets(ws, graph, ds, hidden).items():
                for cell, listed in zip(sets.ids, sets.listed, strict=True) if lv in levels else ():
                    held[region_node_id(lv, str(cell))][ds] = list(_rows(graph, ds, listed))
        features = region_features(geo["cells"], sorted(held))
        for f in features:
            by = held[f["properties"]["id"]]
            f["properties"].update(n=sum(map(len, by.values())), ids=by)
        elsewhere = {pid: ds for ds in kept[1:] for pid in _projection(ws, graph, ds, hidden)["patterns"]}
        trust = {
            pid: {k: graph.nodes[pid]["props"].get(k) for k in ("credibility", "integrity")}
            | {
                "scale": scale_label(graph, pid),
                "reading": insight_reading(graph, pid, thresholds),
                "numbers": wording.value_words(graph.insight(pid), bool(_props(graph, pid).get("weight_by_events"))),
            }
            | ({"map_dataset": elsewhere[pid]} if pid in elsewhere else {})
            for pid in [*pids, *elsewhere]
        }
        return {"type": "FeatureCollection", "features": features, "patterns": trust}

    return graph.memoised(("overlay", *kept, hidden), build)


def map_set(ws: Workspace, dataset_id: str, datasets: tuple[str, ...]) -> tuple[str, ...]:
    """The geo datasets a map counts: ``dataset_id``, then those of ``datasets`` (each a geo dataset, else
    ``invalid_options``), one per upload — the map's own, else the finest —, since a level counts its upload's others."""
    if not datasets:
        return (dataset_id,)
    geo = {d["dataset_id"]: d for d in geo_datasets(ws)}
    if unknown := sorted(set(datasets) - set(geo)):
        raise PipelineError("invalid_options", f"not geo datasets: {', '.join(unknown)}")

    def upload(ds: str) -> str:
        return (ws.ready_batch_for(ds) or {}).get("upload_id") or ds

    kept = {upload(dataset_id): dataset_id}
    for ds in sorted(set(datasets), key=lambda d: (-geo[d]["resolution"], d)):
        kept.setdefault(upload(ds), ds)
    return (dataset_id, *sorted(ds for ds in kept.values() if ds != dataset_id))


class RegionSets(NamedTuple):
    """One region level over a geo dataset's cells: the regions holding a cell (``ids``, H3), each cell's region
    (``idx``), and per region the insights it lists with their lift (``listed``) and those characteristic of its own
    cells (``own``); the others it takes from the coarser regions its cells lie in (two on a coarser boundary: H3 nests only
    approximately)."""

    ids: np.ndarray
    idx: np.ndarray
    listed: list[dict[str, float]]
    own: list[set[str]]


def _region_sets(ws: Workspace, graph: DualGraph, dataset_id: str, hidden: frozenset[str] = frozenset()) -> dict[int, RegionSets]:
    """``RegionSets`` at every region level of the graph coarser than the dataset's cells (its own cells without one)
    where a region lists an insight — a level of none gives the map no hint, so it is left out unless every level is —,
    derived once per commit. An insight of the projection is characteristic of a region by the rule of
    docs/08_interface.md §8.7: with ``x`` of its ``|P_loc|`` located events in the region's cells and ``E`` the number
    its map-wide share puts there, ``x ≥ link_min_events(|P_loc|)`` and ``(x + a) / (E + a) ≥ 1``, which is ``x ≥ E`` whatever ``a``; a base insight the
    graph leaves unplaced (``geo: partial``) is never one. A region also lists, at their lift there, the insights of the
    coarser regions holding its cells that have an event in it: a zoom never loses what the coarser region listed."""

    def build() -> dict[int, RegionSets]:
        proj = _projection(ws, graph, dataset_id, hidden)
        cells = _geo(ws, dataset_id, ("cells",))["cells"]
        per_cell, res = proj["per_cell"], proj["resolution"]
        total = int(per_cell.sum())
        levels = tuple(sorted({n["props"]["resolution"] for n in graph.of_kind("Region")} & set(range(res)))) or (res,)
        regions = event_regions(cells, levels)
        out: dict[int, RegionSets] = {}
        coarser: RegionSets | None = None
        for lv in sorted(regions):  # coarsest first: a region takes what the region above it lists
            ids, idx = np.unique(regions[lv], return_inverse=True)
            here = np.bincount(idx, weights=per_cell, minlength=len(ids))
            listed: list[dict[str, float]] = [{} for _ in ids]
            own: list[set[str]] = [set() for _ in ids]
            events: dict[str, np.ndarray] = {}
            patterns = proj["patterns"].items() if total else ()  # no located point: the expected count would divide by it
            for pid, p in patterns:
                mine = int(p.counts.sum())
                if graph.nodes[pid]["props"].get("geo") == "partial" or not mine:
                    continue
                x = events[pid] = np.bincount(idx[p.events], weights=p.counts, minlength=len(ids))
                lift = smoothed_lift(x, mine * here / total)
                for r in np.flatnonzero((x >= link_min_events(mine)) & (lift >= 1)):
                    listed[r][pid] = float(lift[r])
                    own[r].add(pid)
            if coarser is not None:
                for r, up in set(zip(idx.tolist(), coarser.idx.tolist(), strict=True)):  # a region and each coarser one its cells lie in
                    for pid, lift in coarser.listed[up].items():
                        if pid not in own[r] and events[pid][r] > 0:
                            listed[r][pid] = max(listed[r].get(pid, 0.0), lift)
            out[lv] = coarser = RegionSets(ids, idx, listed, own)
        return {lv: s for lv, s in out.items() if any(s.listed)} or {min(out): out[min(out)]}

    return graph.memoised(("region_sets", dataset_id, hidden), build)


def _rows(graph: DualGraph, dataset_id: str, listed: dict[str, Any]) -> dict[str, list[str]]:
    """The rows of a set of characteristic insights, each result once: every base insight, and every geo host with the
    levels it folds (``_fold_levels``)."""
    hosts = _fold_levels(graph, dataset_id, listed)
    return {pid: hosts.get(pid, []) for pid in listed if pid in hosts or graph.facts(pid).resolution is None}


# ponytail: the map draws other levels' insights on read, once per commit
def member_rows(ws: Workspace, batch: dict[str, Any], owner: dict[str, Any], n_points: int) -> Callable[[np.ndarray], np.ndarray] | None:
    """The point rows of ``owner`` (the map's batch) that a pattern's covered rows name, for a READY batch of the same
    upload and point columns (``same_points``): a geo level's rows are cells, whose points its ``point_cell`` names; a
    table level's rows are the file's rows. None otherwise, or when the rows do not index the points: a matching row
    count alone links nothing."""
    if not same_points(batch, owner):
        return None
    geo = ws.load_geo(batch["dataset_id"], ("point_cell",))
    if geo is not None:
        cells_of = geo["point_cell"]
        return (lambda rows: np.flatnonzero(np.isin(cells_of, rows))) if len(cells_of) == n_points else None
    return (lambda rows: rows) if (batch.get("profile") or {}).get("rows") == n_points else None


class Projected(NamedTuple):
    """One insight of a map's upload on the map's cells: the ``level`` it was found at, and every cell row holding its
    located events with their counts (``events``, the rows it is drawn by, and ``counts``)."""

    level: str
    events: np.ndarray
    counts: np.ndarray


def _projection(ws: Workspace, graph: DualGraph, dataset_id: str, hidden: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Every intact insight of the dataset's upload on the dataset's cells (``Projected``), a ``hidden`` dataset's (a
    hidden layer's run) left out, derived once per commit and hidden set: an
    insight of the dataset is drawn by its cells, one of another level by every cell holding its located events, so it
    is drawn where it is counted and a region listing it shows it. ``per_cell``: points;
    ``centres``; ``resolution``: the dataset's."""

    def build() -> dict[str, Any]:
        geo = _geo(ws, dataset_id, ("cells", "point_cell"))  # inside: a memo hit reads no file
        cells, point_cell = geo["cells"], geo["point_cell"]
        per_cell = np.bincount(point_cell[point_cell >= 0], minlength=len(cells))
        owner = _owner(ws, dataset_id)
        patterns: dict[str, Projected] = {}
        for b in ws.list_batches():
            if b.get("status") != "READY" or b["dataset_id"] in hidden:
                continue
            own = b["dataset_id"] == dataset_id
            members = None if own else member_rows(ws, b, owner, len(point_cell))
            if not own and members is None:
                continue
            res = ((b.get("profile") or {}).get("geo") or {}).get("resolution", b.get("level"))  # also without upload levels
            level = "table" if b.get("level") == "table" else f"res {res}"
            for pid, rows in ws.load_covers(b["dataset_id"]).items():
                if pid not in graph.nodes or graph.tampered(pid):
                    continue
                if members is None:  # the dataset's own
                    patterns[pid] = Projected(level, rows, per_cell[rows])
                    continue
                events, counts = footprint(point_cell, members(rows), 1.0)  # every cell holding its located events
                patterns[pid] = Projected(level, events, counts)
        return {
            "patterns": patterns,
            "per_cell": per_cell,
            "centres": cell_centres(cells),
            "resolution": int(owner["profile"]["geo"]["resolution"]),
        }

    return graph.memoised(("projection", dataset_id, hidden), build)


def geo_projected(ws: Workspace, graph: DualGraph, dataset_id: str, thresholds: PhenomenonThresholds) -> list[dict[str, Any]]:
    """The insights of the dataset's upload found at another level (the table, another resolution) drawn on its cells as
    ``geo_cells`` draws its own: the rows holding their located events, scope, shift and reading; a hidden layer's runs
    left out."""
    out, hidden = [], hidden_datasets(ws.list_batches())
    for pid, p in _projection(ws, graph, dataset_id, hidden)["patterns"].items():
        ins = graph.insight(pid)
        if ins.dataset_id == dataset_id:
            continue
        out.append(_pattern_json(ins, p.events, level=p.level, reading=insight_reading(graph, pid, thresholds)))
    return out


def warm(ws: Workspace, graph: DualGraph, thresholds: PhenomenonThresholds) -> None:
    """The READY geo datasets' map reads built ahead of the console — each one's cell package, overlay, projected insights
    and the whole map's window, which reads every insight's facts — so the first Map open after a commit answers from the
    package cache and the memo; newest first, as many as the package cache holds, so it never evicts its own. Stops while
    a batch waits, whose commit would throw them away."""
    for d in geo_datasets(ws)[::-1][: _pack.cache_info().maxsize]:
        if ws.unfinished_batches():
            return
        ds = d["dataset_id"]
        with suppress(PipelineError, FileNotFoundError):  # deleted meanwhile: nothing to read
            geo_cells_pack(ws, ds)
            geo_overlay(ws, graph, ds, thresholds)
            geo_projected(ws, graph, ds, thresholds)
            geo_window(ws, graph, ds, "-180,-90,180,90", thresholds)


def window_view(centres: np.ndarray, bbox: str) -> np.ndarray:
    """The cells whose centres (lon, lat) lie in ``bbox`` ("west,south,east,north" in degrees); west > east is a view
    across ±180°."""
    try:
        w, s, e, n = (float(x) for x in bbox.split(","))
    except ValueError:
        raise PipelineError("invalid_options", f"bbox {bbox!r} is not west,south,east,north") from None
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= n <= 90):
        raise PipelineError("invalid_options", f"bbox {bbox!r} is not west,south,east,north in degrees")
    lon, lat = centres.T
    across = ((lon >= w) | (lon <= e)) if w > e else ((lon >= w) & (lon <= e))
    return np.asarray(across & (lat >= s) & (lat <= n))


def _restates(graph: DualGraph, pid: str) -> set[str]:
    """The levels of its upload that restate a geo insight (``restating``), derived once per commit."""
    return graph.memoised(("restates", pid), lambda: next((restating(graph, c) for c in scale_chain(graph, [pid])), set()))


def geo_window(
    ws: Workspace,
    graph: DualGraph,
    dataset_id: str,
    bbox: str,
    thresholds: PhenomenonThresholds,
    level: int | None = None,
    datasets: tuple[str, ...] = (),
    period: tuple[datetime, datetime] | None = None,
) -> dict[str, Any]:
    """The IN VIEW list (docs/08_interface.md §8.7), a hidden layer's runs left out: the regions of ``level`` (the finest of ``_region_sets`` without one)
    holding a cell whose centre lies in ``bbox`` (``window_view``), and the rows of the insights characteristic of any
    of them, over the datasets ``map_set`` keeps, each result once (``_rows``); ranked (``rank``) by how many of those
    regions it is characteristic of in its own right — a row a region lists only as a coarser one's comes after —,
    then its best lift there, then weight, and listed under its theme
    (``primary_anchor``). A view of one region lists the ``n`` the overlay draws on it; ``cells`` and ``points`` add over
    the set's datasets, each of its own grid and file, over the whole span; with a ``period``, ``period_points`` are the
    points in it of the datasets that have their times (``period_cell_counts``; None when none has)."""
    rows: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    cells = points = 0
    period_points: int | None = None
    regions: set[str] = set()
    hidden = hidden_datasets(ws.list_batches())
    own_levels = _region_sets(ws, graph, dataset_id, hidden)
    lv = max(own_levels) if level is None else level  # the map's level reads every dataset of the set
    if lv not in own_levels:
        raise PipelineError("invalid_options", f"level {level} is not a region level of {dataset_id}")
    for ds in map_set(ws, dataset_id, datasets):
        proj, sets = _projection(ws, graph, ds, hidden), _region_sets(ws, graph, ds, hidden)
        if lv not in sets:
            continue
        view, at = window_view(proj["centres"], bbox), sets[lv]
        cells, points = cells + int(view.sum()), points + int(proj["per_cell"][view].sum())
        if period is not None and (counted := period_cell_counts(ws, ds, period)) is not None:
            period_points = (period_points or 0) + int(counted[view].sum())
        own: defaultdict[str, list[float]] = defaultdict(list)  # the regions in focus each insight is characteristic of
        wide: defaultdict[str, list[float]] = defaultdict(list)  # those listing it only as a coarser region's
        for r in np.unique(at.idx[view]):
            regions.add(region_node_id(lv, str(at.ids[r])))
            for pid, lift in at.listed[r].items():
                (own if pid in at.own[r] else wide)[pid].append(lift)
        for pid, folded in _rows(graph, ds, {**wide, **own}).items():
            ins, p = graph.insight(pid), proj["patterns"][pid]
            x, mine, lift = int(p.counts[view[p.events]].sum()), int(p.counts.sum()), max(own[pid] or wide[pid])
            rows.append(
                (
                    (-len(own[pid]), -lift, -ins.weight, pid),  # a wider row after every row of the place's own
                    {
                        "id": pid,
                        "dataset_id": ds,
                        "label": graph.nodes[pid]["label"],
                        "level": p.level,
                        "values": wording.short_conditions(ins, *declared_words(graph, pid)),  # a value, or the phrase a code needs
                        "target": ins.target,
                        "effect": round(ins.effect_size, 3),
                        "ptype": ins.phenomenon_type,
                        "weight": round(ins.weight, 3),
                        "support": ins.support,
                        "points": x,
                        "share": round(x / mine, 3),
                        "cells": int(view[p.events].sum()),
                        "regions": len(own[pid]),
                        "lift": round(lift, 1),  # to one decimal, as lift_words reads it: the badge and its words agree
                        "lift_words": wording.lift_words("window", lift),
                        "wider": not own[pid],  # listed only as a coarser region's, at its lift there
                        "also": [{"id": q, "level": proj["patterns"][q].level} for q in folded],
                    },
                )
            )
    themes: dict[str | None, dict[str, Any]] = {}
    for rank, (_, row) in enumerate(sorted(rows, key=lambda r: r[0])):
        anchor = primary_anchor(graph, row["id"])
        aid = anchor["target"] if anchor else None
        node = graph.nodes.get(anchor["target"]) if anchor else None
        theme = themes.setdefault(
            aid,
            {
                "id": aid,
                "label": node["label"] if node else "без теми",
                "reading": wording.theme_reading(node["props"], theme_glossary(graph, node["id"])) if node else "",
                "insights": [],
            },
        )
        theme["insights"].append(row | {"rank": rank})
    order = sorted(themes.values(), key=lambda t: t["id"] is None)  # by their leading row; the insights without a theme last
    return {
        "cells": cells,
        "points": points,
        "period_points": period_points,
        "level": lv,
        "regions": len(regions),
        "insights": len(rows),
        "themes": order,
    }


def _fold_levels(graph: DualGraph, dataset_id: str, listed: dict[str, Any]) -> dict[str, list[str]]:
    """The hosts among the listed geo insights, each with the listed levels it folds (finest first): hosts go the open
    dataset's first, then the finest; a level restated by several (``restating``) goes to the one whose subgroup is
    closest to its own (``affinity``), the earlier host on a tie."""
    res = {q: r for q in listed if (r := graph.facts(q).resolution) is not None}
    geo = sorted(res, key=lambda q: (graph.insight(q).dataset_id != dataset_id, -res[q], q))
    rank = {q: i for i, q in enumerate(geo)}
    claims: defaultdict[str, list[str]] = defaultdict(list)
    for h in geo:
        for q in _restates(graph, h) & listed.keys():
            claims[q].append(h)
    hosts: dict[str, list[str]] = {}
    folded: set[str] = set()
    for pid in geo:
        if pid in folded:
            continue
        hosts[pid] = []
        for q in sorted(_restates(graph, pid) & listed.keys(), key=lambda q: (-res[q], q)):
            live = [h for h in claims[q] if h not in folded]
            if q not in folded and q not in hosts and max(live, key=lambda h: (affinity(graph, h, q), -rank[h])) == pid:
                hosts[pid].append(q)
                folded.add(q)
    return hosts


def export_geosparql(ws: Workspace, graph: DualGraph) -> str:
    """The committed graph as GeoSPARQL Turtle (``geosparql.build_turtle``), each geo insight with the H3 cells of its extent."""
    extents = {}
    for d in geo_datasets(ws):
        if (geo := ws.load_geo(d["dataset_id"], ("cells",))) is None:  # a deletion moving its files aside
            continue
        for pid, rows in ws.load_covers(d["dataset_id"]).items():
            if pid in graph.nodes:
                extents[pid] = [str(c) for c in geo["cells"][rows]]
    return build_turtle(graph, extents)


class MapReads(NamedTuple):
    """The map's reads (docs/08_interface.md §8.1) over one workspace and the graph of one commit, with the settings they
    take: what ``Engine.maps`` hands out while the workspace is usable."""

    ws: Workspace
    graph: DualGraph
    thresholds: PhenomenonThresholds
    max_points: int  # a pattern's points sent to the map

    def datasets(self) -> list[dict[str, Any]]:
        return geo_datasets(self.ws)

    def cells_pack(self, dataset_id: str) -> bytes:
        return geo_cells_pack(self.ws, dataset_id)

    def pattern(self, dataset_id: str, pattern_id: str, rings: bool = False, period: tuple[datetime, datetime] | None = None) -> dict[str, Any]:
        return geo_pattern(self.ws, self.graph, dataset_id, pattern_id, self.max_points, rings, period)

    def table_rows(self, dataset_id: str, resolution: int | None = None, layer: str | None = None) -> tuple[bytes, dict[str, Any]]:
        return geo_table_rows(self.ws, dataset_id, resolution, layer)

    def zone_counts(
        self, dataset_id: str, zones: list[tuple[str, dict[str, Any]]], resolution: int | None = None, layer: str | None = None
    ) -> dict[str, Any]:
        return geo_zone_counts(self.ws, dataset_id, zones, resolution, layer)

    def zone_cells(self, dataset_id: str, zones: list[tuple[str, dict[str, Any]]], resolution: int | None = None) -> bytes:
        return geo_zone_cells(self.ws, dataset_id, zones, resolution)

    def overlay(self, dataset_id: str, datasets: tuple[str, ...] = ()) -> dict[str, Any]:
        return geo_overlay(self.ws, self.graph, dataset_id, self.thresholds, datasets)

    def period_counts(self, dataset_id: str, period: tuple[datetime, datetime], datasets: tuple[str, ...] = ()) -> dict[str, Any]:
        """The cells' points in ``period`` and the listed insights that do not act in it (``idle_insights``)."""
        idle = idle_insights(self.ws, self.graph, dataset_id, period, self.thresholds, datasets)
        return geo_period_counts(self.ws, dataset_id, period) | {"idle": idle}

    def projected(self, dataset_id: str) -> list[dict[str, Any]]:
        return geo_projected(self.ws, self.graph, dataset_id, self.thresholds)

    def window(
        self, dataset_id: str, bbox: str, level: int | None = None, datasets: tuple[str, ...] = (), period: tuple[datetime, datetime] | None = None
    ) -> dict[str, Any]:
        return geo_window(self.ws, self.graph, dataset_id, bbox, self.thresholds, level, datasets, period)
