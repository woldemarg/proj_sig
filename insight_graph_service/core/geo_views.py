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
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from graph_query_engine.graph import DualGraph, declared_words, insight_reading, primary_anchor, theme_glossary
from graph_query_engine.vertical import affinity, restating, scale_chain, scale_label
from insight_contracts import Insight, PhenomenonThresholds, region_node_id, wording
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.geosparql import build_turtle
from insight_graph_service.core.uploads import same_points
from insight_graph_service.core.workspace import Workspace
from subgroup_miner.spatial import (
    SpatialError,
    cell_boundary,
    cell_centres,
    compactness,
    event_regions,
    footprint,
    geometry_to_cells,
    link_min_events,
    parse_zone,
    region_features,
    zone_overlap,
)

WINDOW_LIFT_PRIOR = 30  # pseudo-events added to an insight's events in a region and to their expected count (docs/08 §8.7)


def geo_datasets(ws: Workspace) -> list[dict[str, Any]]:
    """READY datasets analysed as H3 cells, with what the map needs to offer them."""
    out = []
    for b in ws.list_batches():
        geo = (b.get("profile") or {}).get("geo")
        if b.get("status") == "READY" and geo:
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


def geo_profile(loaded: Any) -> dict[str, Any]:
    """A geo level's ``profile.geo`` (docs/02_discovery.md §2.9), written to its batch record: what ``geo_datasets`` offers."""
    ctx = loaded.spatial
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
    }


def place_scores(
    ws: Workspace, graph: DualGraph, cells: list[str] | None, geometry: dict[str, Any] | None, resolution: int | None
) -> dict[str, float] | None:
    """Every geo pattern's overlap with a map zone (``zone_overlap``, docs/07_question_answering.md §7.2), over the geo
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
) -> dict[str, float]:
    """``zone_overlap`` over the stored cells and extents of each geo dataset in ``levels``, and over each (table level,
    geo level placing it) of ``tables`` by the cells holding the table's patterns' events on that level; ``strict``: a
    missing file raises ``FileNotFoundError`` instead of leaving its patterns out."""
    out: dict[str, float] = {}
    for dataset_id, _ in levels:
        geo, covers = ws.load_geo(dataset_id, ("cells",)), ws.load_covers(dataset_id)
        if geo is None or not covers:
            if strict:
                raise FileNotFoundError(dataset_id)
            continue
        out.update(zone_overlap(geo["cells"], covers, zone))
    for table, placed_by in tables:
        geo, covers = ws.load_geo(placed_by, ("cells", "point_cell")), ws.load_covers(table)
        if geo is None or not covers:
            if strict:
                raise FileNotFoundError(table)
            continue
        prints = {pid: footprint(geo["point_cell"], rows, 1.0)[0] for pid, rows in covers.items()}  # a table's rows are the points
        out.update(zone_overlap(geo["cells"], prints, zone))
    return out


# cached on its arguments: a dataset id fixes its files (a READY id is never mined again, its duplicate is SKIPPED; one
# uploaded again after its deletion is mined to the same files), and a commit that changes the datasets changes them
_zone_scores_cached = functools.lru_cache(maxsize=64)(_zone_scores)


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


def _pattern_json(ins: Insight, rows: np.ndarray, **extra: Any) -> dict[str, Any]:
    """An insight as the map lists it (``geo_cells``, ``geo_projected``): its cell rows, scope, shift, support and weight."""
    return {
        "id": ins.id,
        "rows": rows.tolist(),
        "scope": [f"{c.attribute} = {c.value}" for c in ins.conditions],
        "target": ins.target,
        "effect_size": ins.effect_size,
        "phenomenon_type": ins.phenomenon_type,
        "support": ins.support,
        "weight": ins.weight,
        **extra,
    }


def geo_pattern(ws: Workspace, graph: DualGraph, dataset_id: str, pattern_id: str, max_points: int) -> dict[str, Any]:
    """A pattern's cells (rows of the cell table) and a deterministic sample of the points inside them; an insight of
    another level of the upload (``geo_projected``): the cells it is drawn by and its own points inside them. An insight
    whose record fails its digest is drawn by neither (``unknown_dataset``), as the projection and the window skip it.
    With its cells' ``rings``, the ``level`` it was found at and its numbers, so a map that holds no cell of this dataset
    draws and describes it."""
    geo = _geo(ws, dataset_id)
    rows = ws.load_covers(dataset_id).get(pattern_id)
    if rows is not None and (pattern_id not in graph.nodes or graph.tampered(pattern_id)):
        raise PipelineError("unknown_dataset", f"no intact pattern {pattern_id} in {dataset_id}")
    if rows is not None:
        inside, level = np.flatnonzero(np.isin(geo["point_cell"], rows)), f"res {graph.facts(pattern_id).resolution}"
    else:
        found = _projection(ws, graph, dataset_id)["patterns"].get(pattern_id)
        batch = ws.ready_batch_for(graph.insight(pattern_id).dataset_id) if found else None
        members = member_rows(ws, batch, ws.ready_batch_for(dataset_id), len(geo["point_cell"])) if batch else None
        if members is None:  # unknown here, or its dataset deleted since this commit
            raise PipelineError("unknown_dataset", f"no pattern {pattern_id} in {dataset_id}")
        rows, level = found.events, found.level
        src = members(ws.load_covers(batch["dataset_id"])[pattern_id])
        inside = src[np.isin(geo["point_cell"][src], rows)]
    step = max(1, int(np.ceil(len(inside) / max_points)))
    points = geo["lonlat"][inside[::step]]
    ins = graph.insight(pattern_id)
    return {
        "pattern_id": pattern_id,
        "cells": [str(c) for c in geo["cells"][rows]],
        "rings": [cell_boundary(str(c)) for c in geo["cells"][rows]],  # a map that holds no cell of this dataset draws them
        "rows": rows.tolist(),
        "n_points": int(len(inside)),
        "points": np.round(points.astype(float), 5).tolist(),
        "level": level,
        "effect_size": ins.effect_size,
        "phenomenon_type": ins.phenomenon_type,
        "support": ins.support,
        "weight": ins.weight,
    }


def geo_overlay(ws: Workspace, graph: DualGraph, dataset_id: str, thresholds: PhenomenonThresholds, datasets: tuple[str, ...] = ()) -> dict[str, Any]:
    """What the map draws over a geo dataset: every region holding a cell of the datasets ``map_set`` keeps, at each
    level of ``_region_sets``, as GeoJSON with the rows of ``dataset_id``'s cells inside it and ``n``, the IN VIEW rows a
    view of that region alone lists (``_rows``; the datasets' insights are of different uploads, so their counts add), and
    ``by``, that count per dataset; and each pattern of the dataset and of the table levels of its upload, and each one
    another dataset of the set lists, with its credibility, integrity, scale and reading — the latter with ``map_dataset``,
    the dataset whose ``/patterns`` draws it. Derived once per commit."""
    kept = map_set(ws, dataset_id, datasets)

    def build() -> dict[str, Any]:
        geo = _geo(ws, dataset_id, ("cells",))  # inside: a memo hit reads no file
        owner = _owner(ws, dataset_id)
        tables = {b["dataset_id"] for b in ws.list_batches() if b.get("status") == "READY" and b.get("level") == "table" and same_points(b, owner)}
        pids = [n["id"] for n in graph.of_kind("Pattern") if n["props"]["dataset_id"] == dataset_id or n["props"]["dataset_id"] in tables]
        held: defaultdict[str, dict[str, int]] = defaultdict(dict)
        levels = _region_sets(ws, graph, dataset_id).keys()  # the map's: another dataset's finer ones would lie inside its cells
        for ds in kept:
            for lv, sets in _region_sets(ws, graph, ds).items():
                for cell, listed in zip(sets.ids, sets.listed, strict=True) if lv in levels else ():
                    held[region_node_id(lv, str(cell))][ds] = len(_rows(graph, ds, listed))
        features = region_features(geo["cells"], sorted(held))
        for f in features:
            by = held[f["properties"]["id"]]
            f["properties"].update(n=sum(by.values()), by=by)
        elsewhere = {pid: ds for ds in kept[1:] for pid in _projection(ws, graph, ds)["patterns"]}
        trust = {
            pid: {k: graph.nodes[pid]["props"].get(k) for k in ("credibility", "integrity")}
            | {
                "scale": scale_label(graph, pid),
                "reading": insight_reading(graph, pid, thresholds),
                "numbers": wording.value_words(graph.insight(pid)),
            }
            | ({"map_dataset": elsewhere[pid]} if pid in elsewhere else {})
            for pid in [*pids, *elsewhere]
        }
        return {"type": "FeatureCollection", "features": features, "patterns": trust}

    return graph.memoised(("overlay", *kept), build)


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
    cells (``own``); the others it takes from the coarser region it lies in."""

    ids: np.ndarray
    idx: np.ndarray
    listed: list[dict[str, float]]
    own: list[set[str]]


def _region_sets(ws: Workspace, graph: DualGraph, dataset_id: str) -> dict[int, RegionSets]:
    """``RegionSets`` at every region level of the graph coarser than the dataset's cells (its own cells without one)
    where a region lists an insight — a level of none gives the map no hint, so it is left out unless every level is —,
    derived once per commit. An insight of the projection is characteristic of a region by the rule of
    docs/08_interface.md §8.7: with ``x`` of its ``|P_loc|`` located events in the region's cells and ``E`` the number
    its map-wide share puts there, ``x ≥ link_min_events(|P_loc|)`` and ``(x + a) / (E + a) ≥ 1``; a base insight the
    graph leaves unplaced (``geo: partial``) is never one. A region also lists, at their lift there, the insights of the
    coarser regions holding its cells that have an event in it: a zoom never loses what the coarser region listed."""

    def build() -> dict[int, RegionSets]:
        proj = _projection(ws, graph, dataset_id)
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
                lift = (x + WINDOW_LIFT_PRIOR) / (mine * here / total + WINDOW_LIFT_PRIOR)
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

    return graph.memoised(("region_sets", dataset_id), build)


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


def _projection(ws: Workspace, graph: DualGraph, dataset_id: str) -> dict[str, Any]:
    """Every intact insight of the dataset's upload on the dataset's cells (``Projected``), derived once per commit: an
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
            if b.get("status") != "READY":
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
                if own:
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

    return graph.memoised(("projection", dataset_id), build)


def geo_projected(ws: Workspace, graph: DualGraph, dataset_id: str, thresholds: PhenomenonThresholds) -> list[dict[str, Any]]:
    """The insights of the dataset's upload found at another level (the table, another resolution) drawn on its cells as
    ``geo_cells`` draws its own: the rows holding their located events, scope, shift and reading."""
    out = []
    for pid, p in _projection(ws, graph, dataset_id)["patterns"].items():
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
    return across & (lat >= s) & (lat <= n)


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
) -> dict[str, Any]:
    """The IN VIEW list (docs/08_interface.md §8.7): the regions of ``level`` (the finest of ``_region_sets`` without one)
    holding a cell whose centre lies in ``bbox`` (``window_view``), and the rows of the insights characteristic of any
    of them, over the datasets ``map_set`` keeps, each result once (``_rows``); ranked (``rank``) by how many of those
    regions it is characteristic of in its own right — a row a region lists only as a coarser one's comes after —,
    then its best lift there, then weight, and listed under its theme
    (``primary_anchor``). A view of one region lists the ``n`` the overlay draws on it."""
    rows: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    cells = points = 0
    regions: set[str] = set()
    own_levels = _region_sets(ws, graph, dataset_id)
    lv = max(own_levels) if level is None else level  # the map's level reads every dataset of the set
    if lv not in own_levels:
        raise PipelineError("invalid_options", f"level {level} is not a region level of {dataset_id}")
    for ds in map_set(ws, dataset_id, datasets):
        proj, sets = _projection(ws, graph, ds), _region_sets(ws, graph, ds)
        if lv not in sets:
            continue
        view, at = window_view(proj["centres"], bbox), sets[lv]
        cells, points = cells + int(view.sum()), points + int(proj["per_cell"][view].sum())
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
        node = graph.nodes.get(aid)
        theme = themes.setdefault(
            aid,
            {
                "id": aid,
                "label": node["label"] if node else "без теми",
                "reading": wording.theme_reading(node["props"], theme_glossary(graph, aid)) if node else "",
                "insights": [],
            },
        )
        theme["insights"].append(row | {"rank": rank})
    order = sorted(themes.values(), key=lambda t: t["id"] is None)  # by their leading row; the insights without a theme last
    return {"cells": cells, "points": points, "level": lv, "regions": len(regions), "insights": len(rows), "themes": order}


def _fold_levels(graph: DualGraph, dataset_id: str, listed: dict[str, Any]) -> dict[str, list[str]]:
    """The hosts among the listed geo insights, each with the listed levels it folds (finest first): hosts go the open
    dataset's first, then the finest; a level restated by several (``restating``) goes to the one whose subgroup is
    closest to its own (``affinity``), the earlier host on a tie."""
    geo = sorted(
        (q for q in listed if graph.facts(q).resolution is not None),
        key=lambda q: (graph.insight(q).dataset_id != dataset_id, -graph.facts(q).resolution, q),
    )
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
        for q in sorted(_restates(graph, pid) & listed.keys(), key=lambda q: (-graph.facts(q).resolution, q)):
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
