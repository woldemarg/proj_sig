# Provenance of the vendored browser libraries

| File | Original | sha256 | Use |
|---|---|---|---|
| `static/vendor/cytoscape.min.js` | Cytoscape.js 3.30.2 (cdnjs) | `83e8c54a6bec655bfd81df07df605649c268af69aeca67a5ea2da54ea42dac81` | the 2D dual graph |
| `static/vendor/plotly-gl3d.min.js` | plotly.js 3.6.0, the `gl3d` partial bundle (npm `plotly.js-gl3d-dist-min@3.6.0`) | `3de590d2c369cbf6511068daedbfa2a7d2db53c927ac04850d247b94cfcd20a1` | the 3D sphere |

| `static/vendor/maplibre-gl.js`, `static/vendor/maplibre-gl.css` | MapLibre GL JS 4.7.1 (npm `maplibre-gl@4.7.1`, `dist/`) | `.js` `be9633c4d870e26fb37f1cfe5c5a77181667114003ea16207ac7850d8da8add1`, `.css` `576b085fdd9487a65a19215328c1e086c07ce5bf6da09b666b3806d3d008dae9` | the map (BSD-3-Clause) |
| `static/vendor/geo/basemap.json` | Natural Earth 10m v5.1.2 (`nvkelso/natural-earth-vector`, `geojson/`): `admin_0_countries_ukr` (the Ukrainian point of view; `NAME_UK`, `ukr` for `ADM0_A3` UKR), `admin_1_states_provinces_lines` with `SCALERANK` ≤ 5 (boundaries only), `rivers_lake_centerlines` and `lakes` (scalerank ≤ 9), each clipped to its box (shapely `clip_by_rect`), simplified (`preserve_topology`) and rounded to 3 decimals: Ukraine and its neighbours from the Baltic to the Caucasus, lon 10–52, lat 38–61, countries 0.01°, boundaries and rivers 0.005°, lakes 0.005°; then the US West around California (the map demo), lon −132 – −104, lat 26–50, countries and lakes 0.005°, boundaries and rivers 0.01°, appended | `060f48224b463d30916e1556bdb9aa018924c862c317ef4fcfd2dc39e0c39db7` | the offline basemap |
| `static/vendor/geo/places.json` | Natural Earth 10m v5.1.2 `populated_places` in the same boxes: `NAME_UK` (else `NAME`), `POP_MAX`, `SCALERANK` (Moscow's set to 10, the least label), `ADM0CAP`, coordinates rounded to 4 decimals | `1acc8954491befc8174b968777c1ac13110c019073c75bb31503959615bb3fe6` | place labels |

The libraries are unmodified copies (Cytoscape.js and plotly.js MIT, MapLibre GL JS BSD-3-Clause); Natural Earth is public domain. The console loads nothing from a CDN or a tile server, so it works offline.
