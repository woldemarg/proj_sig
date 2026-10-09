# Provenance of the vendored browser libraries

| File | Original | sha256 | Use |
|---|---|---|---|
| `static/vendor/cytoscape.min.js` | Cytoscape.js 3.30.2 (cdnjs) | `83e8c54a6bec655bfd81df07df605649c268af69aeca67a5ea2da54ea42dac81` | the 2D dual graph |
| `static/vendor/plotly-gl3d.min.js` | plotly.js 3.6.0, the `gl3d` partial bundle (npm `plotly.js-gl3d-dist-min@3.6.0`) | `3de590d2c369cbf6511068daedbfa2a7d2db53c927ac04850d247b94cfcd20a1` | the 3D sphere |

| `static/vendor/maplibre-gl.js`, `static/vendor/maplibre-gl.css` | MapLibre GL JS 4.7.1 (npm `maplibre-gl@4.7.1`, `dist/`) | `.js` `be9633c4d870e26fb37f1cfe5c5a77181667114003ea16207ac7850d8da8add1`, `.css` `576b085fdd9487a65a19215328c1e086c07ce5bf6da09b666b3806d3d008dae9` | the map (BSD-3-Clause) |
| `static/vendor/geo/basemap.json` | Natural Earth 10m v5.1.2 (`nvkelso/natural-earth-vector`): `admin_0_countries_ukr` (the Ukrainian point of view), `admin_1_states_provinces` (boundaries only), `rivers_lake_centerlines` and `lakes` (scalerank ≤ 9), clipped to lon 18–44, lat 41.5–57.5, simplified (0.01° / 0.005°), rounded to 3 decimals | `e8248f921e3f0be974afe292b40d3154ff69d4e7d8b244b8b3b8fe808efc8f11` | the offline basemap |
| `static/vendor/geo/places.json` | Natural Earth 10m v5.1.2 `populated_places` in the same box: `NAME_UK` (else `NAME`), `POP_MAX`, `SCALERANK`, `ADM0CAP` | `00f9fc94bb0549e5c47c7e1456dc3ce06f6a3b865c24b7ce78a46a8b5525e75b` | place labels |

The libraries are unmodified copies (Cytoscape.js and plotly.js MIT, MapLibre GL JS BSD-3-Clause); Natural Earth is public domain. The console loads nothing from a CDN or a tile server, so it works offline.
