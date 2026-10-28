"""The derived spatial plane of the committed geo patterns (docs/06_graph_and_storage.md §6.1): from each dataset's
cells and its patterns' extents the regions, IN_REGION, PARENT_OF, CO_LOCATED and compactness (``derive_spatial``), a
table level placed on the map by a geo level of its upload (``place_table``), and the map's geometry — a question's
zone (``parse_zone``, ``geometry_to_cells``, ``zone_overlap``), a pattern's footprint, cell centres, boundaries and WKT.
The ingest of points into cells, the neighbourhood statistics and the n_eff correction stay in ``spatial``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import h3
import numpy as np
from scipy.stats import binom, false_discovery_control

from insight_contracts import region_node_id
from subgroup_miner.spatial import (
    COLOCATED_TOP,
    GEO_COLOCATION_MIN,
    GEO_FOOTPRINT_SHARE,
    GEO_LIFT_MIN,
    GEO_LIFT_MIN_REGIONS,
    GEO_LINK_FDR,
    GEO_LINK_MIN_EVENTS,
    GEO_LOCATED_MIN,
    GEO_REGION_MIN_SHARE,
    SpatialError,
    compactness,
)


@dataclass
class GeoDataset:
    """What ``derive_spatial`` reads of one geo dataset: its ``geo.npz`` arrays and its patterns' cell rows (``covers``)."""

    dataset_id: str
    cells: np.ndarray  # H3 id per cell row
    nbr_ptr: np.ndarray  # CSR over cell rows: ring-1 neighbours present in the dataset
    nbr_idx: np.ndarray
    covers: dict[str, np.ndarray]  # pattern id -> its extent's cell rows


@dataclass
class TableDataset:
    """What ``derive_spatial`` reads of a table level: its patterns' file rows (``covers``), and the cells and point -> cell
    row of the finest READY geo level of its upload that holds one row per point (``placed_by``; docs/06_graph_and_storage.md §6.1)."""

    dataset_id: str
    placed_by: str
    cells: np.ndarray
    point_cell: np.ndarray
    covers: dict[str, np.ndarray]


@dataclass
class SpatialStructure:
    """The derived spatial plane of the committed geo patterns (docs/06_graph_and_storage.md §6.1)."""

    regions: dict[str, dict[str, int | str]] = field(default_factory=dict)  # R:<res>:<h3> -> resolution, h3, cells_covered, patterns
    in_region: list[tuple[str, str, float]] = field(default_factory=list)  # (pattern, region, share of the region's cells it covers)
    parent_of: list[tuple[str, str]] = field(default_factory=list)  # (region, its parent at the next coarser listed level)
    co_located: list[tuple[str, str, str, float]] = field(default_factory=list)  # (a, b, relation, weight), a < b
    compactness: dict[str, float] = field(default_factory=dict)
    places: dict[str, dict[int, frozenset[str]]] = field(default_factory=dict)  # pattern -> resolution -> its cells or their parents
    occupied: dict[str, dict[int, frozenset[str]]] = field(default_factory=dict)  # dataset -> resolution -> the same, of all its cells
    edge_props: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)  # (pattern, region) -> its IN_REGION props
    partial: set[str] = field(default_factory=set)  # base patterns with fewer than GEO_LOCATED_MIN of their events located
    own_cells: dict[str, frozenset[str]] = field(default_factory=dict)  # a geo pattern at a region level -> its cells: its regions there


RELATIONS = ("overlaps", "adjacent", "same_region")  # strongest first


def link_min_events(n: int) -> int:
    """The events an insight of ``n`` located events needs in a place: ``GEO_LINK_MIN_EVENTS``, 5 % of them for a small
    one, 5 at least (docs/06_graph_and_storage.md §6.1)."""
    return min(GEO_LINK_MIN_EVENTS, max(5, math.ceil(0.05 * n)))


CENTRE_TIE_DEG = 1e-7  # ≈ 1 cm: far above where math libraries disagree (≈ 1e-12°), far below any cell


def event_regions(cells: np.ndarray, levels: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Each cell's region at every listed level not finer than its own: the cell holding its centre (docs/06_graph_and_storage.md
    §6.1), as the points were indexed; a centre on a boundary by one rule on every platform (``centre_region``)."""
    names = [str(c) for c in cells]
    res = h3.get_resolution(names[0])
    return {lv: np.array(names if lv == res else [centre_region(c, lv) for c in names]) for lv in levels if lv <= res}


def centre_region(cell: str, level: int) -> str:
    """The cell at ``level`` holding ``cell``'s centre, read at four points ``CENTRE_TIE_DEG`` around it: when they
    disagree the centre lies on a boundary, and it goes to the H3 parent if that is among them, else to the least of
    them. Many centres lie exactly on a coarser boundary, where platforms' floating point would otherwise decide."""
    lat, lng = h3.cell_to_latlng(cell)
    d = CENTRE_TIE_DEG
    around = {h3.latlng_to_cell(lat + a, lng + b, level) for a, b in ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d))}
    if len(around) == 1:
        return around.pop()
    parent = h3.cell_to_parent(cell, level)
    return parent if parent in around else min(around)


def place_table(table: TableDataset, levels: tuple[int, ...]) -> tuple[list[tuple[str, str, float, dict[str, Any]]], set[str]]:
    """A table level's patterns placed by their events (docs/06_graph_and_storage.md §6.1): ``(pattern, region, share,
    props)`` per IN_REGION, and the patterns too little located to be placed. Shares and lifts are over the pattern's
    located events; a level of at least ``GEO_LIFT_MIN_REGIONS`` occupied regions (holding a located event) places by
    lift under BH over them, a coarser one by share where the pattern is no rarer than the map (``rule``)."""
    point_cell = table.point_cell
    located = point_cell[point_cell >= 0]
    per_level = {}
    for lv, names in event_regions(table.cells, levels).items():
        regions, idx = np.unique(names, return_inverse=True)
        per_level[lv] = (regions, idx, np.bincount(idx[located], minlength=len(regions)))
    edges, partial = [], set()
    for pid, rows in table.covers.items():
        cells = point_cell[np.unique(np.asarray(rows, dtype=np.int64))]
        loc = cells[cells >= 0]
        if not len(cells) or len(loc) / len(cells) < GEO_LOCATED_MIN:
            partial.add(pid)
            continue
        n = len(loc)
        x_min = link_min_events(n)
        for lv, (regions, idx, n_r) in per_level.items():
            occupied = n_r > 0
            x = np.bincount(idx[loc], minlength=len(regions))
            share = x / n
            lift = np.divide(share, n_r / len(located), out=np.zeros(len(regions)), where=occupied)
            q = np.ones(len(regions))
            # the binomial tail of R's events falling in P: where small and x >= 5 (a link's least), never below the exact hypergeometric one, and fast
            q[occupied] = false_discovery_control(np.where(x > 0, binom.sf(x - 1, n_r, n / len(located)), 1.0)[occupied])
            rule = "lift" if occupied.sum() >= GEO_LIFT_MIN_REGIONS else "share"
            hit = (q <= GEO_LINK_FDR) & (lift >= GEO_LIFT_MIN) if rule == "lift" else (share >= GEO_REGION_MIN_SHARE) & (lift >= 1)
            for i in np.flatnonzero(hit & (x >= x_min)):
                props = {"basis": "events", "rule": rule, "share": round(float(share[i]), 4), "lift": round(float(lift[i]), 4), "n": int(x[i])}
                edges.append(
                    (pid, region_node_id(lv, str(regions[i])), props["share"], props | {"q": float(f"{q[i]:.4g}"), "placed_by": table.placed_by})
                )
    return edges, partial


@dataclass
class _DatasetPart:
    """What ``derive_spatial`` derives of one geo dataset alone (its files never change once READY): ``places`` holds its
    patterns' places, regions, IN_REGION, props and compactness and the dataset's occupied cells (``plane``)."""

    res: int
    plane: SpatialStructure
    covered: Counter[str]
    raw: dict[str, set[str]]
    grown: dict[str, set[str]]
    ring: dict[str, set[str]]
    shares: dict[tuple[str, str], float]


def _dataset_part(d: GeoDataset, levels: tuple[int, ...]) -> _DatasetPart:
    out = SpatialStructure()
    raw, grown, ring, shares = {}, {}, {}, {}
    covered: Counter[str] = Counter()
    res = h3.get_resolution(str(d.cells[0]))
    own = [lv for lv in levels if lv <= res]
    # each cell in the region holding its centre, as the points were indexed (docs/06_graph_and_storage.md §6.1)
    grid = event_regions(d.cells, (res, *own))  # its own resolution too, listed or not
    occupied = {lv: dict(zip(*np.unique(grid[lv], return_counts=True), strict=True)) for lv in own}
    out.occupied[d.dataset_id] = {lv: frozenset(cells.tolist()) for lv, cells in grid.items()}
    extents = [np.asarray(r, dtype=np.int64) for r in d.covers.values()]
    union = np.unique(np.concatenate(extents)) if extents else np.empty(0, dtype=np.int64)
    for lv in own:  # each dataset's cells inside a region that some pattern covers
        for cell, n in zip(*np.unique(grid[lv][union], return_counts=True), strict=True):
            covered[region_node_id(lv, str(cell))] += int(n)
    for pid, rows in d.covers.items():
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        out.places[pid] = {lv: frozenset(cells[rows].tolist()) for lv, cells in grid.items()}
        if res in levels:  # at its own level a region is one of its cells: those, not an IN_REGION per cell
            out.own_cells[pid] = out.places[pid][res]
        raw[pid] = {str(c) for c in d.cells[rows]}
        near = np.concatenate([rows, *(d.nbr_idx[d.nbr_ptr[r] : d.nbr_ptr[r + 1]] for r in rows)]) if len(rows) else rows
        grown[pid] = {str(c) for c in d.cells[near]}
        ring[pid] = {n for c in raw[pid] for n in h3.grid_ring(c, 1)}
        out.compactness[pid] = round(compactness(rows, d.nbr_ptr, d.nbr_idx), 4)
        for lv in (x for x in own if x != res):
            regions, counts = np.unique(grid[lv][rows], return_counts=True)
            for cell, n in zip(regions, counts, strict=True):
                rid = region_node_id(lv, str(cell))
                share = float(n / min(len(rows), occupied[lv][cell]))
                if share >= GEO_REGION_MIN_SHARE:  # the threshold reads the exact share, the edge stores it rounded
                    share = round(share, 4)
                    out.in_region.append((pid, rid, share))
                    out.edge_props[(pid, rid)] = {"basis": "cells", "share": share, "n": int(n)}  # n: its cells there
                    shares[(pid, rid)] = share
                    _add_region(out, levels, lv, str(cell))
    return _DatasetPart(res, out, covered, raw, grown, ring, shares)


def derive_spatial(
    datasets: list[GeoDataset], levels: tuple[int, ...], tables: Iterable[TableDataset] = (), memo: dict[tuple, Any] | None = None
) -> SpatialStructure:
    """Regions, IN_REGION, PARENT_OF, CO_LOCATED with its relation and compactness, from the stored cells and extents.

    A pattern is IN_REGION of a region at every listed level coarser than its dataset when
    their overlap reaches ``GEO_REGION_MIN_SHARE`` of the smaller of its cells and the region's occupied cells (a focus inside
    a region scores 1, as does a pattern covering the region); a region holds the cells whose centres it holds
    (``event_regions``), as the events, the overlay's rows, zones and credibility place them, and PARENT_OF follows the
    H3 hierarchy of the regions themselves. ``places`` and ``occupied`` give each pattern's and each dataset's cells at
    its own resolution and, at every listed level coarser, the cells holding their centres. Two patterns of one resolution, of any
    datasets, are CO_LOCATED: ``overlaps`` when their one-ring-dilated extents overlap by at least ``GEO_COLOCATION_MIN``
    of the smaller, else ``adjacent`` when the extents touch or share cells, else ``same_region`` when both are
    IN_REGION of one region at the nearest listed level at least two coarser. Each pattern keeps its ``COLOCATED_TOP``
    strongest partners (relation first, then weight) of its own dataset and as many of the other datasets; a pair that
    either side keeps is an edge. A table level's patterns are IN_REGION by their events (``place_table``): never
    CO_LOCATED, in ``places`` or in ``occupied``. ``edge_props`` holds every IN_REGION's props, of either basis.
    At its own resolution, when that is a listed level, a geo pattern has no IN_REGION: a region there is one of its
    cells, and ``own_cells`` gives them; a region's ``patterns`` counts those patterns too.

    ``memo`` (the caller's, kept between calls): each geo dataset's own part (``_dataset_part``) by its id and the levels,
    each table's placement by its id, its ``placed_by`` and the levels; read when present, filled when not, and left
    holding only what this call used. The result holds copies, so the memo stays as built."""
    out = SpatialStructure()
    levels = tuple(sorted(set(levels), reverse=True))
    raw, grown, ring, res_of, dataset_of, shares = {}, {}, {}, {}, {}, {}
    covered: Counter[str] = Counter()
    memo = {} if memo is None else memo
    used = set()
    for d in datasets:
        if not len(d.cells):
            continue
        used.add(key := ("geo", d.dataset_id, levels))
        part = memo[key] if key in memo else memo.setdefault(key, _dataset_part(d, levels))
        for pid in part.plane.places:
            res_of[pid], dataset_of[pid] = part.res, d.dataset_id
        out.occupied.update({k: dict(v) for k, v in part.plane.occupied.items()})  # copies: the memo stays as built
        out.places.update({k: dict(v) for k, v in part.plane.places.items()})
        out.compactness.update(part.plane.compactness)
        out.own_cells.update(part.plane.own_cells)
        out.in_region.extend(part.plane.in_region)
        out.edge_props.update({k: dict(v) for k, v in part.plane.edge_props.items()})
        for rid, props in part.plane.regions.items():
            out.regions.setdefault(rid, dict(props))
        covered.update(part.covered)
        raw.update(part.raw)
        grown.update(part.grown)
        ring.update(part.ring)
        shares.update(part.shares)
    for table in tables:
        used.add(key := ("table", table.dataset_id, table.placed_by, levels))
        placed, partial = memo[key] if key in memo else memo.setdefault(key, place_table(table, levels))
        out.partial |= partial
        for pid, rid, share, props in placed:
            out.in_region.append((pid, rid, share))
            out.edge_props[(pid, rid)] = dict(props)
            _, lv, cell = rid.split(":")  # R:<level>:<h3>
            _add_region(out, levels, int(lv), cell)
    for stale in set(memo) - used:  # a deleted dataset, or a table placed by another level now
        del memo[stale]
    members = Counter(rid for _, rid, _ in out.in_region)  # and the patterns whose own cell a region is
    members.update(rid for pid, cells in out.own_cells.items() for c in cells if (rid := region_node_id(res_of[pid], c)) in out.regions)
    for rid, props in out.regions.items():
        props.update(cells_covered=covered[rid], patterns=members[rid])
        coarser = [x for x in levels if x < props["resolution"]]
        if coarser:
            out.parent_of.append((rid, region_node_id(coarser[0], h3.cell_to_parent(props["h3"], coarser[0]))))
    regions_of: defaultdict[str, set[str]] = defaultdict(set)
    for pid, rid in shares:
        regions_of[pid].add(rid)
    by_res: defaultdict[int, list[str]] = defaultdict(list)
    for pid in sorted(raw):
        by_res[res_of[pid]].append(pid)
    links: dict[tuple[str, str], tuple[str, float]] = {}
    for res, pids in by_res.items():
        level = next((lv for lv in levels if res - lv >= 2), None)
        for i, a in enumerate(pids):
            for b in pids[i + 1 :]:
                overlap = len(grown[a] & grown[b]) / max(min(len(grown[a]), len(grown[b])), 1)
                if overlap >= GEO_COLOCATION_MIN:
                    links[(a, b)] = ("overlaps", round(overlap, 4))
                elif raw[a] & raw[b] or ring[a] & raw[b]:  # share of both extents' cells on the common edge
                    edge = len((raw[a] | ring[a]) & raw[b]) + len((raw[b] | ring[b]) & raw[a])
                    links[(a, b)] = ("adjacent", round(edge / (len(raw[a]) + len(raw[b])), 4))
                else:
                    common = [r for r in regions_of[a] & regions_of[b] if level is not None and r.startswith(f"R:{level}:")]
                    if common:
                        links[(a, b)] = ("same_region", max(min(shares[(a, r)], shares[(b, r)]) for r in common))
    pool: defaultdict[tuple[str, bool], list[tuple[int, float, str]]] = defaultdict(list)  # (pattern, across datasets)
    for (a, b), (relation, weight) in links.items():
        cross = dataset_of[a] != dataset_of[b]
        pool[(a, cross)].append((RELATIONS.index(relation), -weight, b))
        pool[(b, cross)].append((RELATIONS.index(relation), -weight, a))
    top = {key: {p for _, _, p in sorted(found)[:COLOCATED_TOP]} for key, found in pool.items()}
    out.co_located = [
        (a, b, relation, weight)
        for (a, b), (relation, weight) in sorted(links.items())
        if b in top[(a, dataset_of[a] != dataset_of[b])] or a in top[(b, dataset_of[a] != dataset_of[b])]
    ]
    out.in_region.sort()
    out.parent_of.sort()
    return out


def _add_region(out: SpatialStructure, levels: tuple[int, ...], lv: int, cell: str) -> None:
    """The region and its ancestors at the coarser listed levels."""
    for up in [x for x in levels if x <= lv]:
        up_cell = cell if up == lv else h3.cell_to_parent(cell, up)
        out.regions.setdefault(region_node_id(up, up_cell), {"resolution": up, "h3": up_cell})


def parse_zone(cells: list[str]) -> frozenset[str]:
    """A map zone given as H3 cells, of any resolutions; ``SpatialError`` names the first that is not a cell."""
    if not cells:
        raise SpatialError("the zone holds no cell")
    if bad := next((c for c in cells if not (isinstance(c, str) and h3.is_valid_cell(c))), None):
        raise SpatialError(f"'{bad}' is not an H3 cell")
    return frozenset(cells)


# ponytail: no size cap; a continent at resolution 10 takes minutes. Cap by the shape's area before exposing it beyond localhost
def geometry_to_cells(geometry: dict[str, Any], resolution: int) -> frozenset[str]:
    """The cells at ``resolution`` whose centres lie in ``geometry``: a GeoJSON Polygon, MultiPolygon, Feature or
    FeatureCollection (its own ``bbox`` member is metadata), ``{"center": [lon, lat], "radius_m": r}`` or
    ``{"bbox": [west, south, east, north]}`` in degrees, west <= east (a box across 180° is a MultiPolygon of two). A shape that holds no
    cell centre gives the cells under its vertices (a circle: the cell under its centre)."""
    try:
        if "bbox" in geometry and "type" not in geometry:
            w, s, e, n = map(float, geometry["bbox"])
            if not (-180 <= w <= e <= 180 and -90 <= s <= n <= 90):
                raise SpatialError(f"bbox {[w, s, e, n]} is not [west, south, east, north] in degrees with west <= east, south <= north")
            geometry = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        if "center" in geometry:
            lon, lat = map(float, geometry["center"])
            if not (abs(lon) <= 180 and abs(lat) <= 90):
                raise SpatialError(f"center {[lon, lat]} is not [lon, lat] in degrees")
            radius, centre = float(geometry["radius_m"]), h3.latlng_to_cell(lat, lon, resolution)
            k = int(radius / (1.5 * h3.average_hexagon_edge_length(resolution, "m"))) + 2  # ring k lies at least 1.5·k edges away
            inside = {c for c in h3.grid_disk(centre, k) if h3.great_circle_distance(h3.cell_to_latlng(c), (lat, lon), "m") <= radius}
            return frozenset(inside or {centre})
        kind = geometry.get("type")
        if kind == "FeatureCollection":
            return frozenset().union(*(geometry_to_cells(f, resolution) for f in geometry["features"]))
        if kind == "Feature":
            return geometry_to_cells(geometry["geometry"], resolution)
        if kind not in ("Polygon", "MultiPolygon"):
            raise SpatialError(f"geometry type {kind!r} is not a Polygon, MultiPolygon, Feature, FeatureCollection, center or bbox")
        # a longitude past ±180 (a map ring, cell_boundary) back in range: H3 reads a crossing by the jump of its edge
        polygons = [geometry["coordinates"]] if kind == "Polygon" else geometry["coordinates"]
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [[[[x if -180 <= x <= 180 else (x + 180) % 360 - 180, y] for x, y, *_ in ring] for ring in poly] for poly in polygons],
        }
        cells = set(h3.geo_to_cells(geometry, resolution))
        if not cells:
            rings = [r for poly in geometry["coordinates"] for r in poly]
            cells = {h3.latlng_to_cell(float(lat), float(lon), resolution) for ring in rings for lon, lat in ring}
        return frozenset(cells)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:  # h3's errors are ValueErrors, as is SpatialError
        raise SpatialError(str(exc) if isinstance(exc, SpatialError) else f"unreadable geometry: {type(exc).__name__}: {exc}") from exc


def cells_inside(cells: np.ndarray, area: Iterable[str]) -> np.ndarray:
    """Which of one level's ``cells`` lie in ``area``, cells of any resolutions, through the cell centre (docs/06_graph_and_storage.md
    §6.1): an area cell finer than them counts the cell holding its centre, a coarser one every cell whose centre it holds
    (H3 children do not tile their parent)."""
    inside = np.zeros(len(cells), dtype=bool)
    by_res: defaultdict[int, list[str]] = defaultdict(list)
    for c in area:
        by_res[h3.get_resolution(c)].append(c)
    for group in by_res.values():
        if len(cells):
            inside[build_projection(cells, np.array(group))(np.arange(len(group)))] = True
    return inside


def build_projection(cells: np.ndarray, other: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """The rows of one level's ``cells`` inside a set of rows of another level's ``other`` through the cell centre
    (``cells_inside``), the two levels' centres matched once for every set."""
    res, z = h3.get_resolution(str(cells[0])), h3.get_resolution(str(other[0]))
    if z >= res:  # finer: the cell holding each of their centres
        index = {str(c): i for i, c in enumerate(cells)}
        holder = np.array([index.get(c, -1) for c in event_regions(other, (res,))[res]], dtype=np.int64)
        return lambda rows: np.unique(holder[rows][holder[rows] >= 0])
    index = {str(c): i for i, c in enumerate(other)}  # coarser: every cell whose centre one of them holds
    parent = np.array([index.get(c, -1) for c in event_regions(cells, (z,))[z]], dtype=np.int64)
    return lambda rows: np.flatnonzero(np.isin(parent, rows))


def zone_overlap(cells: np.ndarray, covers: dict[str, np.ndarray], zone: frozenset[str]) -> dict[str, float]:
    """How much each pattern and ``zone`` are one place: ``|P ∩ Z| / sqrt(|P| · |Z|)`` over one dataset's cells (``geo.npz``)
    and extents (``covers``), ``Z`` the dataset's cells inside the zone (``cells_inside``). A focus that is the zone
    scores 1, as does a pattern that is; a pattern far larger or smaller than the zone scores little; one outside it 0."""
    if not len(cells):
        return {}
    inside = cells_inside(cells, zone)
    out, size = {}, int(inside.sum())
    for pid, rows in covers.items():
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        hit = int(inside[rows].sum())
        out[pid] = round(hit / float(np.sqrt(len(rows) * size)), 4) if hit else 0.0
    return out


def footprint(point_cell: np.ndarray, members: np.ndarray, share: float = GEO_FOOTPRINT_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Where an insight's points lie on one level's cells: the busiest cell rows that together hold at least ``share`` of
    its located points (``members`` are source rows, as ``point_cell`` indexes them), busiest first, and their counts."""
    cells = point_cell[members]
    rows, counts = np.unique(cells[cells >= 0], return_counts=True)
    order = np.lexsort((rows, -counts))
    rows, counts = rows[order], counts[order]
    k = int(np.searchsorted(np.cumsum(counts), share * counts.sum())) + 1 if len(rows) else 0
    return rows[:k], counts[:k]


def cell_centres(cells: np.ndarray) -> np.ndarray:
    """(lon, lat) of each cell's centre."""
    return np.array([h3.cell_to_latlng(str(c))[::-1] for c in cells], dtype=float).reshape(-1, 2)


def cell_boundary(cell: str) -> list[list[float]]:
    """Closed GeoJSON ring ([lon, lat] pairs) of one cell, its longitudes taken around its first vertex: a cell across
    ±180° stays one small polygon (a longitude past 180, which the map draws on the next copy of the world), not a band
    around the globe."""
    vertices = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
    x0 = vertices[0][0]
    ring = [[round(x0 + (x - x0 + 540) % 360 - 180, 6), round(y, 6)] for x, y in vertices]
    return ring + ring[:1]


def region_features(cells: np.ndarray, regions: list[str]) -> list[dict[str, Any]]:
    """GeoJSON features of regions (``R:<res>:<h3>``, none finer than ``cells``, one dataset's), each with the rows of the
    cells whose centre it holds (``event_regions``), as a base pattern's events and a zone are placed, and its H3
    ``centre`` (lon, lat), where the map draws its count; the window tests the cells' own centres (``window_view``)."""
    held = event_regions(cells, tuple({int(rid.split(":")[1]) for rid in regions})) if len(cells) else {}
    out = []
    for rid in regions:
        _, res, cell = rid.split(":")
        lv = int(res)
        rows = np.flatnonzero(held[lv] == cell).tolist()
        lat, lng = h3.cell_to_latlng(cell)
        out.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [cell_boundary(cell)]},
                "properties": {"id": rid, "resolution": lv, "rows": rows, "centre": [round(lng, 6), round(lat, 6)]},
            }
        )
    return out


def cells_to_wkt(cells: list[str]) -> str:
    """WKT ``MULTIPOLYGON`` (longitude latitude, as GeoSPARQL's default CRS84 reads it) of the area the cells cover."""
    if not cells:
        return "MULTIPOLYGON EMPTY"
    # ponytail: no antimeridian split; an area across ±180° reads as a band around the globe, split it when data goes there
    shape = h3.cells_to_h3shape(cells).__geo_interface__
    polygons = [shape["coordinates"]] if shape["type"] == "Polygon" else shape["coordinates"]
    rings = ["(" + ", ".join("(" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in r) + ")" for r in p) + ")" for p in polygons]
    return "MULTIPOLYGON (" + ", ".join(rings) + ")"
