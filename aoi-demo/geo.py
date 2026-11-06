"""The demo's H3 helpers (docs/geo_harness_plan.md §5, §8): the layers' cells at SIG's own resolution and its parents, a
polygon's zone by the rule SIG applies to a geometry, and the rows' place relative to the zone. Pure functions over pandas
frames, no network; the resolution always comes from SIG (GET /api/geo), never from here."""

from __future__ import annotations

from typing import Any

import h3
import numpy as np
import pandas as pd

H3_LEVELS = (0, 1, 2)  # the columns: SIG's resolution R and the parents R-1, R-2 (h3_r9, h3_r8, h3_r7 for Chicago)
AOI_RING = 1  # "adjacent": the zone's cells and their first ring, as SIG's own neighbourhood (six neighbours)


def h3_columns(resolution: int, levels: tuple[int, ...] = H3_LEVELS) -> list[str]:
    """The column names, finest first: h3_r<res> with the absolute resolution in the name."""
    return [f"h3_r{resolution - k}" for k in levels]


def enrich_h3(df: pd.DataFrame, resolution: int, lat: str = "lat", lon: str = "lon", levels: tuple[int, ...] = H3_LEVELS) -> pd.DataFrame:
    """A copy of ``df`` with the H3 columns: the cell of every row with valid coordinates (``h3.latlng_to_cell``, the call
    SIG indexes points with), then its parents one and two levels up; "" where the coordinates are missing or out of range."""
    out = df.copy()
    la, lo = pd.to_numeric(out[lat], errors="coerce"), pd.to_numeric(out[lon], errors="coerce")
    ok = la.between(-90, 90) & lo.between(-180, 180)
    points = list(zip(la[ok], lo[ok], strict=True))
    cell_of = {p: h3.latlng_to_cell(p[0], p[1], resolution) for p in set(points)}  # distinct points only, as SIG does
    cols = h3_columns(resolution, levels)
    out[cols[0]] = pd.Series([cell_of[p] for p in points], index=out.index[ok]).reindex(out.index, fill_value="")
    for col, k in zip(cols[1:], levels[1:], strict=True):
        parent_of = {c: h3.cell_to_parent(c, resolution - k) for c in set(cell_of.values())}
        out[col] = out[cols[0]].map(lambda c, p=parent_of: p.get(c, ""))
    return out


def zone_h3_cells(polygon: dict[str, Any], resolution: int, occupied: list[str]) -> list[str]:
    """The zone as SIG reads a geometry (subgroup_miner.spatial_plane.geometry_to_cells): a longitude past ±180 folded back
    in range, the cells whose centre lies in the GeoJSON polygon, or the cells under its vertices when it holds no centre;
    of those, the map's occupied cells. A ring H3 refuses (degenerate, self-crossing) gives no cells, as SIG's 400."""
    rings = [[[x if -180 <= x <= 180 else (x + 180) % 360 - 180, y] for x, y, *_ in ring] for ring in polygon["coordinates"]]
    try:
        cells = set(h3.geo_to_cells({"type": "Polygon", "coordinates": rings}, resolution))
        if not cells:
            cells = {h3.latlng_to_cell(float(lat), float(lon), resolution) for ring in rings for lon, lat in ring}
    except (ValueError, h3.H3BaseException):
        return []
    return sorted(cells & set(occupied))


def aoi_zone(df: pd.DataFrame, zone: list[str], cell_col: str, ring: int = AOI_RING) -> pd.Series:
    """inside: in the polygon (``in_aoi`` 1); adjacent: outside it but in a zone cell or within ``ring`` rings of one — a
    point in the same cell as the zone is not inside the zone; far: the rest, rows without a cell included."""
    near = set().union(*(h3.grid_disk(c, ring) for c in zone)) if zone else set()
    place = np.where(df["in_aoi"] == 1, "inside", np.where(df[cell_col].isin(near), "adjacent", "far"))
    return pd.Series(place, index=df.index, name="aoi_zone")
