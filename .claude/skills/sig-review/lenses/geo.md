# Lens: place

Report: `02-geospatial.md`. Apply the rules of `../SKILL.md`.

Review place: does the system interpret coordinates, cells, geometries and geographic relations as the same place throughout?

`docs/current_docs/02_discovery.md` §2.9 and `docs/current_docs/06_graph_and_storage.md` §6.1 (the spatial plane) are the contract. H3 is the index the code uses; do not assume another. Spatial statistics (Moran's I, `n_eff`, a lift, a false-discovery cut, a band's quarters) belong to the `math` lens. Whether a place fact survives a file or a service call belongs to `data` and `system`. What the map looks like belongs to `ux`.

## What to check

### 1. Coordinates

- Are latitude and longitude in the same order in the input, the cell, GeoJSON and the API (GeoJSON is `[lon, lat]`)?
- Are degrees mixed with linear units (metres, kilometres) without the conversion the chapter states?
- Does a missing or invalid coordinate locate a point nowhere, rather than at a default?
- Is a zone's axis order the one the parser documents (`cells`, `bbox`, a polygon ring, `{center, radius_m}`)?

### 2. Cells and regions

Check the operations the change actually uses:

- a point's cell at the stated resolution;
- a coarser region by the cell's centre, including the centre-tie rule;
- a disk of `ring` 0, 1 or 2 as a set of cells;
- a footprint, a cell boundary, a ring that crosses ±180°;
- a zone overlap: which cells a geometry contains.

A parent that does not contain the centre, or a neighbour counted from a different ring than the one stored, is a wrong place. Euclidean distance on raw latitude and longitude is not a ground distance.

### 3. Edge cases

- a cell or a bbox across the antimeridian (`west > east`);
- a centre on a region boundary;
- an empty or invalid geometry;
- a resolution the chapter refuses.

Do not demand a case the chapters exclude. Check that the exclusion still holds.

### 4. The same place

Two operations in this lens's code that decide "where" must agree: the cell a point was indexed to, the region that holds that centre, the cells a zone returns. A grid, a resolution or a ring must not change silently between them.

A disagreement that is a dropped field or a stale response is not this lens.

## Expected result

For each finding:

- the place operation;
- the coordinate or cell rule, if established;
- where it goes wrong;
- a concrete location (a point, a cell, a bbox) that shows the wrong place;
- the fix;
- control coordinates that tell the right cell from the wrong one.

An unknown coordinate contract is an uncertainty, not a confirmed defect.
