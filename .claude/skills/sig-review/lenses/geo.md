# Lens: geospatial correctness

Report: `02-geospatial.md`. Apply the rules of `../SKILL.md`.

Review the geographic semantics and spatial-data handling of the target: does the system interpret places, coordinates, geometries, spatial distances and geographic relations correctly?

Do not assume a GIS library, coordinate system or spatial database until the code or docs confirm it.

## What to check

### 1. Coordinates and reference systems

- Are the coordinate system and datum consistent between components?
- Are latitude and longitude swapped anywhere?
- Is axis order consistent between input data, libraries and APIs?
- Are geographic coordinates in degrees mixed with projected coordinates in linear units?
- Are coordinate-system transformations correct?
- Is coordinate-system metadata kept where it is needed?

### 2. Spatial operations

Check the operations actually implemented:

- distances between points;
- containment in an area;
- nearest-object search;
- intersection, inclusion and spatial adjacency;
- bounding boxes;
- buffering;
- geometric transformations;
- spatial filters and indexes.

Establish whether each operation matches the real geographic meaning of the task. For example, Euclidean distance between latitude/longitude values is not automatically a correct geographic distance.

### 3. Geographic edge cases

Depending on the task:

- points near the antimeridian;
- high latitudes;
- zero or invalid coordinates;
- empty and invalid geometries;
- region borders and bounding boxes;
- ambiguity or loss of spatial precision;
- geographic units of measurement;
- spatial-index behaviour for special locations.

Do not demand handling of scenarios the requirements explicitly exclude, but check that those exclusions are documented.

### 4. Preserving geographic meaning

Follow the geographic information through its transformations. Check that the meaning of coordinates, geometries, coordinate systems or units does not change without an explicit transformation or matching metadata.

### 5. Consistent results

Check that related components treat geographic data the same way — no case where one part of the system considers objects adjacent and another uses incompatible criteria.

## Expected result

For each finding:

- the spatial process concerned;
- the actual coordinate system or geographic semantics, if established;
- where the error arises;
- a geographic situation in which the result is wrong;
- the correct fix;
- a test or control coordinates that verify correctness.

Behaviour that depends on an unknown coordinate system or an undescribed contract is an uncertainty, not a confirmed defect.
