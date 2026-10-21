"""The map's reads (docs/08_interface.md §8.7): what the console's map draws of a READY geo dataset, from its stored files
(``geo.npz``, ``covers.npz``, ``cells.geojson``) and the graph of one commit.

Functions of the workspace and a ``DualGraph``; nothing here writes. The engine guards each one (a degraded workspace serves
no map) and passes the graph of one commit, so one answer never mixes two commits. What one commit derives once — an
upload's insights projected on a dataset's cells, the overlay — lives in the graph's memo.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from graph_query_engine.graph import DualGraph, primary_anchor
from graph_query_engine.vertical import picture, scale_label, shown_regions
from insight_contracts import Insight
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.geosparql import build_turtle
from insight_graph_service.core.uploads import same_points
from insight_graph_service.core.workspace import Workspace
from subgroup_miner.spatial import (
    GEO_FOOTPRINT_SHARE,
    SpatialError,
    cell_boundary,
    cell_centres,
    compactness,
    footprint,
    geometry_to_cells,
    parse_zone,
    region_features,
    zone_overlap,
)

WINDOW_MIN_POINTS = 10  # an insight lists in a map view holding at least this many of its points


def geo_datasets(ws: Workspace) -> list[dict[str, Any]]:
    """READY datasets analysed as H3 cells, with what the map needs to offer them."""
    out = []
    for b in ws.list_batches():
        geo = (b.get("profile") or {}).get("geo")
        if b.get("status") == "READY" and geo:
            out.append({"dataset_id": b["dataset_id"], "filename": b["filename"], **geo, "metrics": b["profile"].get("numerics", [])})
    return out


def _geo_ready(ws: Workspace, dataset_id: str) -> bool:
    """Only a dataset with a READY geo batch is served: its files survive a failed batch, its listing does not."""
    return any(b["dataset_id"] == dataset_id and b["status"] == "READY" and (b.get("profile") or {}).get("geo") for b in ws.list_batches())


def _geo(ws: Workspace, dataset_id: str, keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    geo = ws.load_geo(dataset_id, keys) if _geo_ready(ws, dataset_id) else None
    if geo is None:
        raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
    return geo


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
    datasets of at most ``resolution`` (None: all), and every placed base pattern's by its footprint on the geo level
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
    placed: defaultdict[tuple[str, str], set[str]] = defaultdict(set)  # (table level, the geo level placing it) -> its placed patterns
    for e in graph.edges:
        if e["type"] == "IN_REGION" and (by := e["props"].get("placed_by")):
            placed[(graph.insight(e["source"]).dataset_id, by)].add(e["source"])
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


def _zone_scores(
    ws: Workspace, zone: frozenset[str], levels: tuple[tuple[str, int], ...], tables: tuple[tuple[str, str], ...], strict: bool = True
) -> dict[str, float]:
    """``zone_overlap`` over the stored cells and extents of each geo dataset in ``levels``, and over each (table level,
    geo level placing it) of ``tables`` by the footprints of the table's patterns on that level's cells; ``strict``: a
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
        prints = {pid: footprint(geo["point_cell"], rows)[0] for pid, rows in covers.items()}  # a table's rows are the points
        out.update(zone_overlap(geo["cells"], prints, zone))
    return out


# cached on its arguments: a dataset id fixes its files (a READY id is never mined again, its duplicate is SKIPPED; one
# uploaded again after its deletion is mined to the same files), and a commit that changes the datasets changes them
_zone_scores_cached = functools.lru_cache(maxsize=64)(_zone_scores)


def geo_cells_file(ws: Workspace, dataset_id: str) -> Path:
    """``geo_cells`` as written with the dataset's batch (the cells, their rings and the patterns' extents never change
    afterwards): the map reads the file on every open."""
    if not (_geo_ready(ws, dataset_id) and ws.geo_cells_path(dataset_id).is_file()):
        raise PipelineError("unknown_dataset", f"{dataset_id} is not a geo dataset")
    return ws.geo_cells_path(dataset_id)


def geo_cells(ws: Workspace, dataset_id: str, loaded: Any) -> dict[str, Any]:
    """The dataset's cells as a GeoJSON FeatureCollection with the journaled patterns' extents; properties are the columns
    of the analysed cell table (``loaded.frame``), read once, when the batch writes the file."""
    ctx, frame = loaded.spatial, loaded.frame
    cols = {c: frame[c].to_numpy(dtype=float if frame[c].dtype.kind in "fiu" else str) for c in frame.columns}
    features = []
    for i, cell in enumerate(ctx.cells):
        props: dict[str, Any] = {"h3": str(cell), "row": i}
        for name, values in cols.items():
            v = values[i]
            props[name] = (None if not np.isfinite(v) else round(float(v), 4)) if values.dtype.kind == "f" else str(v)
        features.append({"type": "Feature", "id": i, "geometry": {"type": "Polygon", "coordinates": [cell_boundary(str(cell))]}, "properties": props})
    lonlat = ctx.lonlat[ctx.point_cell >= 0]
    bbox = [float(lonlat[:, 0].min()), float(lonlat[:, 1].min()), float(lonlat[:, 0].max()), float(lonlat[:, 1].max())]
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
    another level of the upload (``geo_projected``): its footprint's cells and its own points inside them."""
    geo = _geo(ws, dataset_id)
    rows = ws.load_covers(dataset_id).get(pattern_id)
    if rows is not None:
        inside = np.flatnonzero(np.isin(geo["point_cell"], rows))
    else:
        found = _projection(ws, graph, dataset_id)["patterns"].get(pattern_id)
        batch = ws.ready_batch_for(graph.insight(pattern_id).dataset_id) if found else None
        members = member_rows(ws, batch, ws.ready_batch_for(dataset_id), len(geo["point_cell"])) if batch else None
        if members is None:  # unknown here, or its dataset deleted since this commit
            raise PipelineError("unknown_dataset", f"no pattern {pattern_id} in {dataset_id}")
        rows = found[0]
        src = members(ws.load_covers(batch["dataset_id"])[pattern_id])
        inside = src[np.isin(geo["point_cell"][src], rows)]
    step = max(1, int(np.ceil(len(inside) / max_points)))
    points = geo["lonlat"][inside[::step]]
    return {
        "pattern_id": pattern_id,
        "cells": [str(c) for c in geo["cells"][rows]],
        "rows": rows.tolist(),
        "n_points": int(len(inside)),
        "points": np.round(points.astype(float), 5).tolist(),
    }


def geo_overlay(ws: Workspace, graph: DualGraph, dataset_id: str) -> dict[str, Any]:
    """What the graph says about a geo dataset's map: the regions drawn for its patterns and for the base patterns of its
    upload and points (``shown_regions``, none finer than its cells) as GeoJSON, each with its ``picture`` and the
    dataset's cell rows inside it, and each pattern's credibility, integrity and scale; derived once per commit."""

    def build() -> dict[str, Any]:
        geo = _geo(ws, dataset_id, ("cells",))  # inside: a memo hit reads no file
        owner = ws.ready_batch_for(dataset_id)
        tables = {b["dataset_id"] for b in ws.list_batches() if b.get("status") == "READY" and b.get("level") == "table" and same_points(b, owner)}
        pids = [n["id"] for n in graph.of_kind("Pattern") if n["props"]["dataset_id"] == dataset_id or n["props"]["dataset_id"] in tables]
        res = owner["profile"]["geo"]["resolution"]
        drawn = [r for r in shown_regions(graph, pids) if graph.nodes[r]["props"]["resolution"] <= res]  # a finer one is inside a cell
        features = region_features(geo["cells"], sorted(drawn))
        for f in features:
            f["properties"].update(picture(graph, f["properties"]["id"]))
        trust = {
            pid: {k: graph.nodes[pid]["props"].get(k) for k in ("credibility", "integrity")} | {"scale": scale_label(graph, pid)} for pid in pids
        }
        return {"type": "FeatureCollection", "features": features, "patterns": trust}

    return graph.memoised(("overlay", dataset_id), build)


# ponytail: the map draws other levels' insights by their footprint, derived on read per commit; their regions come
# from IN_REGION of the commit (shown_regions)
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


def _projection(ws: Workspace, graph: DualGraph, dataset_id: str) -> dict[str, Any]:
    """Every intact insight of the dataset's upload on the dataset's cells, derived once per commit: an insight of the
    dataset keeps its cells, any other (the table's, another level's) the ``footprint`` of its member points.
    ``patterns``: id -> (cell rows, their points, the level it was found at); ``per_cell``: points; ``centres``."""

    def build() -> dict[str, Any]:
        geo = _geo(ws, dataset_id, ("cells", "point_cell"))  # inside: a memo hit reads no file
        point_cell = geo["point_cell"]
        per_cell = np.bincount(point_cell[point_cell >= 0], minlength=len(geo["cells"]))
        owner = ws.ready_batch_for(dataset_id)  # READY: _geo checked it
        patterns: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
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
                patterns[pid] = (rows, per_cell[rows], level) if own else (*footprint(point_cell, members(rows)), level)
        return {"patterns": patterns, "per_cell": per_cell, "centres": cell_centres(geo["cells"])}

    return graph.memoised(("projection", dataset_id), build)


def geo_projected(ws: Workspace, graph: DualGraph, dataset_id: str) -> list[dict[str, Any]]:
    """The insights of the dataset's upload found at another level (the table, another resolution) drawn on its cells as
    ``geo_cells`` draws its own: the rows of their ``footprint``, their scope and shift."""
    out = []
    for pid, (rows, counts, level) in _projection(ws, graph, dataset_id)["patterns"].items():
        ins = graph.insight(pid)
        if ins.dataset_id == dataset_id:
            continue
        out.append(_pattern_json(ins, rows, level=level, spatial={"footprint_share": GEO_FOOTPRINT_SHARE, "footprint_points": int(counts.sum())}))
    return out


def geo_window(ws: Workspace, graph: DualGraph, dataset_id: str, bbox: str) -> dict[str, Any]:
    """The map as a window on the graph: the insights of the dataset's upload with at least ``WINDOW_MIN_POINTS`` of their
    points in the cells whose centres lie in ``bbox`` ("west,south,east,north"), under their theme (``primary_anchor``).
    ``lift``: an insight's share of the view's points over its share of the map's, so the insights characteristic of the
    view come first; themes by their best insight, then by their points in view."""
    try:
        w, s, e, n = (float(x) for x in bbox.split(","))
    except ValueError:
        raise PipelineError("invalid_options", f"bbox {bbox!r} is not west,south,east,north") from None
    proj = _projection(ws, graph, dataset_id)
    lon, lat = proj["centres"].T
    view = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    total, here = int(proj["per_cell"].sum()), int(proj["per_cell"][view].sum())
    themes: dict[str | None, dict[str, Any]] = {}
    for pid, (rows, counts, level) in proj["patterns"].items():
        inside, mine = view[rows], int(counts.sum())
        points = int(counts[inside].sum())
        if points < WINDOW_MIN_POINTS:
            continue
        ins = graph.insight(pid)
        anchor = primary_anchor(graph, pid)
        aid = anchor["target"] if anchor else None
        theme = themes.setdefault(aid, {"id": aid, "label": graph.nodes[aid]["label"] if aid else "no theme", "insights": []})
        theme["insights"].append(
            {
                "id": pid,
                "label": graph.nodes[pid]["label"],
                "level": level,
                "values": [c.value for c in ins.conditions],
                "target": ins.target,
                "effect": round(ins.effect_size, 3),
                "ptype": ins.phenomenon_type,
                "weight": round(ins.weight, 3),
                "points": points,
                "share": round(points / mine, 3),
                "cells": int(inside.sum()),
                "lift": round(points / here / (mine / total), 2),
            }
        )
    for t in themes.values():
        t["insights"].sort(key=lambda i: (-i["lift"], -i["weight"]))
    order = sorted(themes.values(), key=lambda t: (-t["insights"][0]["lift"], -sum(i["points"] for i in t["insights"])))
    return {"cells": int(view.sum()), "points": here, "insights": sum(len(t["insights"]) for t in order), "themes": order}


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
