# Provenance of the vendored browser libraries

| File | Original | sha256 | Use |
|---|---|---|---|
| `static/vendor/cytoscape.min.js` | Cytoscape.js 3.30.2 (cdnjs) | `83e8c54a6bec655bfd81df07df605649c268af69aeca67a5ea2da54ea42dac81` | the 2D dual graph |
| `static/vendor/plotly-gl3d.min.js` | plotly.js 3.6.0, the `gl3d` partial bundle (npm `plotly.js-gl3d-dist-min@3.6.0`) | `3de590d2c369cbf6511068daedbfa2a7d2db53c927ac04850d247b94cfcd20a1` | the 3D sphere |

Both are unmodified copies under the MIT license. The console loads nothing from a CDN, so it works offline.
