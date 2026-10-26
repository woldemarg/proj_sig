/* SIG map view: H3 cells of geo datasets over an offline basemap, the answer's evidence on the map (docs/08_interface.md §8.7). */
"use strict";

const G = { map: null, ready: null, datasets: [], ds: null, data: null, patterns: {}, projected: [], byRow: [], bounds: [], box: null, metric: "points", pending: null, evidence: [], focus: [], placeData: [], places: [], zone: new Set(), overlay: null, levels: [], level: null, refreshing: null, dirty: false, loading: false, loadSeq: 0, target: null, popup: null, want: null, stale: false };
const RAMP = ["#241046", "#3f1670", "#6a1d96", "#9b2a9f", "#cf4475", "#f2733f", "#ffc23f"];
const C = { up: "#ff8a3d", down: "#36d6ff", cov: "#3ccb92", hot: "#ff3b4f", cold: "#3d8bff", amber: "#ffb000", grid: "#121a26", region: "#4cc9c0" };
const PLACES_MAX = 40;  // place names drawn at once: each DOM marker is repositioned on every frame of a move

const GeoMap = {
  has: (pid) => Boolean(G.patterns[pid]),
  offMap: (pid) => (G.loading ? "Ще не намальовано: мапа завантажується."  // why the drawer of an insight the map does not draw opens
    : !G.overlay && G.data?.patterns.some((p) => p.id === pid) ? "Не на цій мапі: довіру до нього не вдалося прочитати."
    : G.overlay?.patterns[pid]?.integrity === "mismatch" ? "Не на цій мапі: його запис не пройшов перевірку цілісності."
    : "Не на цій мапі: його рівень тут нічого не малює."),
  current: () => G.want,  // the geo dataset the map holds or opens on (its questions are suggestions())
  zone: () => [...G.zone],  // the selected cells' H3 ids: the zone a question asks about

  toggleZone(h3) {
    if (G.zone.has(h3)) G.zone.delete(h3); else G.zone.add(h3);
    paintZone();
  },

  clearZone() { G.zone.clear(); paintZone(); },

  refreshing: () => G.refreshing,  // the pass in flight, if any
  refresh() {  // one pass at a time; a call during a pass runs one more pass after it, which reads the filter then
    if (G.refreshing) { G.dirty = true; return G.refreshing; }
    G.refreshing = (async () => {
      let failed;
      do { G.dirty = false; failed = await refreshPass().then(() => null, (e) => e); } while (G.dirty);
      G.refreshing = null;
      if (failed) throw failed;
    })();
    return G.refreshing;
  },

  async show() {
    await ensureMap();
    G.map.resize();
    if (G.stale) {  // what changed while another tab was open; a failed load is tried again on the next open
      G.stale = false;  // a pass asked meanwhile sets it again
      try { await GeoMap.refresh(); } catch (e) { G.stale = true; console.error("map load failed", e); }
    }
    if (G.pending) { const qa = G.pending; G.pending = null; GeoMap.onAnswer(qa); }
    else if (G.data && !G.evidence.length) G.map.fitBounds(G.box, { padding: viewPadding(), duration: 0 });
    scheduleWindow();
  },

  pick(pid, fit = false) {  // an insight chosen in the view's list (the view stays) or walked to: drawn, described in the chat
    if (!GeoMap.has(pid)) GeoMap.unfocus();  // one the map cannot draw (no trust read): no other insight stays painted
    GeoMap.focusPattern(pid, fit);
    pickInsight(pid, windowItem(pid));
  },

  suggestions() {
    /* questions in everyday words: no column names, only a category value the data holds; from the graph's insights of
       the map's dataset, so they need no cell of it loaded */
    const ps = S.nodes.filter((n) => n.dataset === G.want), scopes = ps.flatMap((p) => p.scope || []);
    const glossary = (G.datasets.find((d) => d.dataset_id === G.want) || {}).glossary || {};
    const out = [];
    if (scopes.some((s) => /=\s*hot spot$/.test(s))) out.push("Що особливого в найгарячіших зонах, де подій найбільше?");
    const share = [...ps].sort((a, b) => b.weight - a.weight)
      .find((p) => p.ptype !== "covariance" && p.effect > 0 && / is (.+?)(?: \(|$)/.test(glossary[p.target] || ""));
    if (share) out.push(`Де частіше трапляється «${glossary[share.target].match(/ is (.+?)(?: \(|$)/)[1]}»?`);
    if (scopes.some((s) => /=\s*q1$/.test(s))) out.push("Що відбувається там, де подій найменше?");
    if (ps.length > 3) out.push("Які зони схожі між собою, хоча розташовані в різних місцях?");
    return out;
  },

  onAnswer(qa) {
    const items = ((qa.view && qa.view.evidence && qa.view.evidence.items) || []);
    const geoDs = items.some((i) => G.patterns[i.pattern_id]) ? G.ds  // the open map draws one of them (another level of its upload too): it stays
      : items.map((i) => i.provenance.dataset_id).find((ds) => G.datasets.some((d) => d.dataset_id === ds));
    if (!geoDs) { GeoMap.clear(); return; }
    if (!G.map || S.view !== "map") { G.pending = qa; G.want = geoDs; return; }  // the map opens on the answer's dataset
    // the answer's insights wait for a click on their citation or card in the chat: nothing is painted before
    const apply = () => { GeoMap.unfocus(); G.evidence = items.filter((i) => G.patterns[i.pattern_id]); };
    // a load in flight is superseded, the latest answer's dataset winning; a later load drops this one
    if (geoDs !== G.ds || G.loading) loadDataset(geoDs).then((done) => done && apply()); else apply();
  },

  clear() { GeoMap.unfocus(); G.evidence = []; },

  closePopup() {  // whether a cell popup was open
    const open = Boolean(G.popup); if (open) G.popup.remove();
    G.popup = null; return open;
  },

  focused: () => Boolean(G.focusId),

  placeOf: (pid) => (G.patterns[pid] ? G.patterns[pid].rows.map((r) => G.data.features[r].properties.h3) : []),  // its cells' H3 ids: a zone a question can carry

  openTheme(aid) { W.fold[aid ?? ""] = true; renderWindow(); },  // the IN VIEW list opens the theme a card names

  unfocus() {
    GeoMap.closePopup();
    if (!G.map || !G.map.getSource("pts")) return;
    G.map.getSource("focus").setData(empty()); G.focus = []; G.focusId = null;
    G.map.setPaintProperty("cells-fill", "fill-opacity", 0.55);
    G.map.getSource("pts").setData(empty());
    $("map-card").hidden = true;
    document.querySelectorAll("#map-window .mw-i").forEach((b) => b.classList.remove("on"));
  },

  async focusPattern(pid, fit = true) {
    const p = G.patterns[pid]; if (!p || !G.map) return;
    GeoMap.unfocus();
    G.focusId = pid;
    const item = G.evidence.find((i) => i.pattern_id === pid);
    const color = (item || p).phenomenon_type === "covariance" ? C.cov : (item ? item.statistics.effect_size : p.effect_size) >= 0 ? C.up : C.down;
    for (const [id, key] of [["hl-fill", "fill-color"], ["hl-glow", "line-color"], ["hl-line", "line-color"]]) G.map.setPaintProperty(id, key, color);
    G.focus = p.rows; G.map.getSource("focus").setData(cellsOf(p.rows));  // only its cells: the cost follows the insight, not the map
    G.map.setPaintProperty("cells-fill", "fill-opacity", 0.14);
    if (fit) fitRows(p.rows, 90);
    renderCard(p, item, null);
    document.querySelectorAll("#map-window .mw-i").forEach((b) => b.classList.toggle("on", b.dataset.pid === pid));
    try {
      const pts = await api(`/api/geo/${encodeURIComponent(G.ds)}/patterns/${encodeURIComponent(pid)}`);
      if (G.focusId !== pid) return;  // another insight was focused meanwhile
      G.map.getSource("pts").setData({ type: "FeatureCollection", features: pts.points.map((c) => ({ type: "Feature", geometry: { type: "Point", coordinates: c }, properties: {} })) });
      renderCard(p, item, pts);
    } catch (e) { /* the card stays without the point count */ }
  },
};

async function refreshPass() {
  try { G.datasets = await api("/api/geo"); } catch (e) { G.datasets = []; }
  $("map-tab").hidden = !G.datasets.length;
  if (!G.datasets.length) { G.loadSeq++; G.loading = false; G.ds = G.want = null; G.data = null; G.patterns = {}; W.data = null; renderWindow(); if (S.view === "map") setView("graph"); return; }
  const filter = $("dataset-filter").value;
  const want = G.datasets.find((d) => d.dataset_id === filter) || G.datasets.find((d) => d.dataset_id === G.want) || G.datasets[0];
  G.want = want.dataset_id;
  if (S.view !== "map") G.stale = true;  // the map reads its cells on its tab alone (show)
  else if (want.dataset_id !== (G.loading ? G.target : G.ds) || !G.data) await loadDataset(want.dataset_id);  // a load in flight counts
  else { await Promise.all([loadOverlay(G.ds), loadProjected(G.ds)]); scheduleWindow(); }  // a commit changes the graph over the same cells
  if (!S.chat.length) renderThread();  // the welcome card offers this dataset's questions
}

const empty = () => ({ type: "FeatureCollection", features: [] });
const cellsOf = (rows) => ({ type: "FeatureCollection", features: rows.map((r) => G.data.features[r]).filter(Boolean) });
const cellSource = () => ({ type: "geojson", data: empty(), maxzoom: 10 });  // one parameter set for every source of cells: their outlines coincide

const unwrap = (x, x0) => x0 + ((x - x0 + 540) % 360) - 180;  // a longitude taken around x0: a span across ±180° stays whole
const MW_WIDTH = 342;  // the IN VIEW list's width and right offset (.map-window)
const viewPadding = () => {  // every cell clear of the open list, where the map is wide enough to hold both
  const w = $("map").clientWidth, open = !(W.folded ?? w < 800);
  return { top: 40, bottom: 40, left: 40, right: 40 + (open && w > 2 * MW_WIDTH ? MW_WIDTH : 0) };
};

function boundsOf(rows) {  // the cells' boxes, each moved next to the first one: cells across ±180° stay one box
  const x0 = G.bounds[rows[0]][0];
  const b = rows.reduce((a, r) => {
    const x = G.bounds[r]; if (!x) return a;
    const dx = unwrap(x[0], x0) - x[0];
    return [Math.min(a[0], x[0] + dx), Math.min(a[1], x[1]), Math.max(a[2], x[2] + dx), Math.max(a[3], x[3])];
  }, [Infinity, 90, -Infinity, -90]);
  return [[b[0], b[1]], [b[2], b[3]]];
}

function fitRows(rows, padding) {
  if (rows.length && G.bounds[rows[0]]) G.map.fitBounds(boundsOf(rows), { padding, maxZoom: 10, duration: calm() ? 0 : 900 });
}

function graticule() {
  const f = [];
  for (let x = -180; x <= 180; x += 1) f.push({ type: "Feature", properties: { major: x % 5 === 0 }, geometry: { type: "LineString", coordinates: [[x, -80], [x, 80]] } });
  for (let y = -80; y <= 80; y += 1) f.push({ type: "Feature", properties: { major: y % 5 === 0 }, geometry: { type: "LineString", coordinates: [[-180, y], [180, y]] } });
  return { type: "FeatureCollection", features: f };
}

function ensureMap() {
  if (G.ready) return G.ready;
  G.map = new maplibregl.Map({  // no box zoom: Shift-click selects zone cells
    container: "map", attributionControl: false, dragRotate: false, pitchWithRotate: false, boxZoom: false, center: [36.5, 48.3], zoom: 6, maxZoom: 13,
    fadeDuration: 0,  // with the zero transition below a pulse step is one frame, and the region counts appear without fading
    style: { version: 8, transition: { duration: 0, delay: 0 }, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#05080d" } }] },
  });
  G.map.touchZoomRotate.disableRotation();
  G.map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  G.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  G.ready = new Promise((resolve) => G.map.on("load", resolve)).then(async () => {
    const base = await fetch("/vendor/geo/basemap.json").then((r) => r.json());
    const m = G.map;
    m.addSource("base", { type: "geojson", data: base });
    m.addSource("grid", { type: "geojson", data: graticule() });
    m.addSource("cells", cellSource());
    m.addSource("focus", cellSource());  // the focused insight's cells
    m.addSource("zone", cellSource());  // the selected zone's cells
    m.addSource("pts", { type: "geojson", data: empty() });
    m.addSource("regions", { type: "geojson", data: empty() });
    m.addSource("region-labels", { type: "geojson", data: empty() });  // a point per region, its count as an icon
    const layer = (id, type, source, paint, filter, extra = {}) => m.addLayer({ id, type, source, paint, ...(filter ? { filter } : {}), ...extra });
    layer("land", "fill", "base", { "fill-color": ["case", ["get", "ukr"], "#0c121b", "#080c12"] }, ["==", ["get", "layer"], "country"]);
    layer("grid", "line", "grid", { "line-color": C.grid, "line-width": ["case", ["get", "major"], 1, 0.5] });
    layer("lake", "fill", "base", { "fill-color": "#0a1726" }, ["==", ["get", "layer"], "lake"]);
    layer("river", "line", "base", { "line-color": "#11314c", "line-width": 1 }, ["==", ["get", "layer"], "river"]);
    layer("admin1", "line", "base", { "line-color": "#283246", "line-width": 0.8, "line-dasharray": [3, 2] }, ["==", ["get", "layer"], "admin1"]);
    layer("border-glow", "line", "base", { "line-color": "#6b7da0", "line-width": 5, "line-blur": 4, "line-opacity": 0.18 }, ["==", ["get", "layer"], "country"]);
    layer("border", "line", "base", { "line-color": "#7d8daa", "line-width": 1.1 }, ["==", ["get", "layer"], "country"]);
    layer("cells-fill", "fill", "cells", { "fill-color": "#3f1670", "fill-opacity": 0.55 });
    layer("cells-line", "line", "cells", { "line-color": "#a265ff", "line-width": 0.5, "line-opacity": 0.32 });
    layer("lisa-hot", "line", "cells", { "line-color": C.hot, "line-width": 2.4, "line-blur": 2.5, "line-opacity": 0.6 }, ["==", ["get", "lisa_points"], "hot spot"]);
    layer("lisa-cold", "line", "cells", { "line-color": C.cold, "line-width": 2.4, "line-blur": 2.5, "line-opacity": 0.6 }, ["==", ["get", "lisa_points"], "cold spot"]);
    layer("hl-fill", "fill", "focus", { "fill-color": C.up, "fill-opacity": 0.6 });  // its colour set by focusPattern
    layer("hl-glow", "line", "focus", { "line-color": C.up, "line-width": 8, "line-blur": 6, "line-opacity": 0.6 });
    layer("hl-line", "line", "focus", { "line-color": C.up, "line-width": 1.6 });
    layer("focus-line", "line", "focus", { "line-color": C.amber, "line-width": 2.6, "line-opacity": 1 });
    layer("regions-line", "line", "regions", { "line-color": C.region, "line-width": 1.4, "line-opacity": 0.75, "line-dasharray": [4, 2] }, ["==", ["get", "resolution"], -1]);
    layer("regions-n", "symbol", "region-labels", {}, ["==", ["get", "resolution"], -1], { layout: { "icon-image": ["get", "icon"], "icon-allow-overlap": true, "icon-ignore-placement": true } });
    layer("zone-line", "line", "zone", { "line-color": "#eef3ff", "line-width": 2, "line-dasharray": [2, 1] });
    layer("pts-halo", "circle", "pts", { "circle-color": "#ff4d6d", "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 3, 11, 8], "circle-blur": 1, "circle-opacity": 0.45 });
    layer("pts-core", "circle", "pts", { "circle-color": "#ffe0e6", "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 0.8, 11, 2.4] });
    // each pulse step redraws the whole map: ~15 steps a second, none while it moves (until it is idle again) or is out of sight
    let rest = true, still = true, last = 0;
    m.on("movestart", () => { rest = false; });
    m.on("idle", () => { rest = true; });
    const pulse = (now) => {
      requestAnimationFrame(pulse);
      if (!(G.focus.length && rest && S.view === "map" && !document.hidden) || calm()) {
        if (!still) { m.setPaintProperty("focus-line", "line-opacity", 1); still = true; }
        return;
      }
      if (now - last < 66) return;
      last = now; still = false;
      m.setPaintProperty("focus-line", "line-opacity", 0.55 + 0.45 * Math.sin(now / 280));  // a period of ≈ 1.8 s
    };
    requestAnimationFrame(pulse);
    addPlaces(await fetch("/vendor/geo/places.json").then((r) => r.json()));
    let pointer = null;  // the latest pointer event, read once per frame
    m.on("mousemove", (e) => {
      if (!pointer) requestAnimationFrame(() => {
        const ev = pointer, f = m.queryRenderedFeatures(ev.point, { layers: ["cells-fill"] })[0];
        pointer = null;
        $("map-cursor").textContent = `${ev.lngLat.lat.toFixed(4)}°N ${ev.lngLat.lng.toFixed(4)}°E · z${m.getZoom().toFixed(1)}` + (f ? ` · ${f.properties.h3} · ${plural(f.properties.points, "подія", "події", "подій")}` : "");
        m.getCanvas().style.cursor = f ? "pointer" : "";
      });
      pointer = e;
    });
    m.on("click", "cells-fill", (e) => (e.originalEvent.shiftKey ? GeoMap.toggleZone(e.features[0].properties.h3) : cellPopup(e.features[0], e.lngLat)));
    m.on("moveend", () => { placeDensity(); paintRegions(); scheduleWindow(); });  // once a move ends, not on each of its frames
    wireHud();
  });
  return G.ready;
}

function addPlaces(places) { G.placeData = places.features; placeDensity(); }

function placeDensity() {
  /* the most populous places in view above the zoom's population floor, at most PLACES_MAX, as DOM markers */
  G.places.forEach((m) => m.remove());
  const z = G.map.getZoom(), view = G.map.getBounds(), minPop = z < 6.5 ? 600000 : z < 7.5 ? 100000 : z < 8.5 ? 30000 : 0;
  G.places = !$("lyr-labels").checked ? [] : G.placeData
    .filter((f) => f.properties.pop >= minPop && view.contains(f.geometry.coordinates))
    .sort((a, b) => b.properties.pop - a.properties.pop).slice(0, PLACES_MAX).map((f) => {
      const p = f.properties, el = document.createElement("div");
      el.className = `place r${Math.min(p.rank, 10)}${p.pop > 500000 ? " big" : ""}`;
      el.innerHTML = `<i></i><span>${esc(p.name)}</span>`;
      return new maplibregl.Marker({ element: el, anchor: "left" }).setLngLat(f.geometry.coordinates).addTo(G.map);
    });
}

function cellFeatures(pack) {
  /* the cell package (GET /api/geo/{ds}/cells) as a GeoJSON FeatureCollection: each ring from its first vertex and
     integer offsets in 10^-scale degrees, each cell's columns as its properties */
  const names = Object.keys(pack.columns), unit = 10 ** pack.scale;
  const features = pack.rings.map((r, i) => {
    const ring = [[r[0], r[1]]];
    for (let k = 2; k < r.length; k += 2) ring.push([r[0] + r[k] / unit, r[1] + r[k + 1] / unit]);
    ring.push(ring[0]);
    const properties = { row: i };
    for (const n of names) properties[n] = pack.columns[n][i];
    return { type: "Feature", id: i, geometry: { type: "Polygon", coordinates: [ring] }, properties };
  });
  return { type: "FeatureCollection", features, bbox: pack.bbox, patterns: pack.patterns };
}

async function loadDataset(ds) {
  /* the latest call wins (whether the map ends on ``ds``): an earlier one stops at its next step and writes nothing */
  await ensureMap();
  GeoMap.unfocus();  // row ids are per dataset: nothing of the old one stays painted
  let seq = ++G.loadSeq;
  G.target = G.want = ds;  // the map opens on it: a refresh pass meanwhile keeps to it, a filter's other choice wins
  G.loading = true; W.seq++; W.data = null; renderWindow();  // the list waits for the new dataset; a view asked before is dropped
  try {
    const data = cellFeatures(await api(`/api/geo/${encodeURIComponent(ds)}/cells`));
    if (seq !== G.loadSeq) return false;
    seq = ++G.loadSeq;  // a read of the old dataset started meanwhile (a refresh pass) is dropped too
    GeoMap.unfocus();  // a focus or popup made during the load was the old dataset's
    G.ds = G.want = ds; G.data = data; G.evidence = [];
    G.projected = []; G.overlay = null; mergePatterns();  // nothing of the old dataset indexes the new cells
    for (const id of ["regions", "region-labels"]) G.map.getSource(id).setData(empty());
    G.bounds = data.features.map((f) => { const r = f.geometry.coordinates[0], x0 = r[0][0];  // a cell across ±180° stays one box
      return r.reduce((a, c) => { const x = unwrap(c[0], x0); return [Math.min(a[0], x), Math.min(a[1], c[1]), Math.max(a[2], x), Math.max(a[3], c[1])]; }, [Infinity, 90, -Infinity, -90]); });
    G.box = boundsOf(data.features.map((_, i) => i));  // what the map opens on: its cells, as they are drawn
    G.map.getSource("cells").setData(data);
    G.map.getSource("pts").setData(empty());
    await Promise.all([loadProjected(ds), loadOverlay(ds)]);
    if (seq !== G.loadSeq) return false;
    const info = G.datasets.find((d) => d.dataset_id === ds) || {};
    const numeric = Object.keys(data.features[0] ? data.features[0].properties : {}).filter((k) => k !== "row" && data.features.some((f) => typeof f.properties[k] === "number"));
    if (!numeric.includes(G.metric)) G.metric = numeric[0] || "points";
    $("map-metric").innerHTML = numeric.map((k) => `<option value="${esc(k)}" ${k === G.metric ? "selected" : ""}>${esc(human(k))}</option>`).join("");
    colorBy(G.metric);
    const sac = info.sac || {};
    $("map-title").textContent = info.filename || ds;
    $("map-meta").innerHTML = `H3 рівень ${info.resolution} · ${plural(info.cells, "комірка", "комірки", "комірок")} · ${plural(info.points, "подія", "події", "подій")}` +
      (sac.validated ? `<br>${term("sac")}: ${sac.significant_iid} → <b>${sac.significant_corrected}</b> значущих із ${sac.validated}` : "");
    G.map.fitBounds(G.box, { padding: viewPadding(), duration: 0 });
    paintZone();
    return true;
  } finally { if (seq === G.loadSeq) { G.loading = false; scheduleWindow(); } }
}

async function loadProjected(ds, seq = G.loadSeq) {
  /* the upload's insights found at another level (the table, another resolution) on this map's cells, as /projected draws them */
  let failed = "";
  const projected = await api(`/api/geo/${encodeURIComponent(ds)}/projected`).catch((e) => { failed = e.message; return []; });
  if (seq !== G.loadSeq) return;  // a dataset loaded since: these rows index other cells
  G.projected = projected;
  $("map-projected-error").hidden = !failed;  // not an upload without other levels: they could not be read
  $("map-projected-error").textContent = failed && `Інсайти інших рівнів недоступні: ${failed}`;
  mergePatterns();
}

function mergePatterns() {
  /* what the map draws, each row's insights for the popup: its own level's insights that the commit's overlay holds intact
     (a record that fails its digest is never drawn, and without an overlay none can be trusted) and the other levels'
     (/projected, intact only) */
  const trust = (G.overlay && G.overlay.patterns) || {};
  const own = G.data.patterns.filter((p) => trust[p.id] && trust[p.id].integrity !== "mismatch").map((p) => ({ ...p, reading: trust[p.id].reading, numbers: trust[p.id].numbers }));
  G.patterns = Object.fromEntries([...own, ...G.projected].map((p) => [p.id, p]));
  G.byRow = G.data.features.map(() => []);
  Object.values(G.patterns).forEach((p) => p.rows.forEach((r) => G.byRow[r] && G.byRow[r].push(p.id)));
  if (G.focusId && !G.patterns[G.focusId]) GeoMap.unfocus();  // a commit's overlay marked the focused one: it goes
}

async function loadOverlay(ds, seq = G.loadSeq) {
  /* the graph's regions, trust and scale over the map: they change with every commit, the cells never */
  let failed = "";
  const overlay = await api(`/api/geo/${encodeURIComponent(ds)}/overlay`).catch((e) => { failed = e.message; return null; });
  if (seq !== G.loadSeq) return;  // a dataset loaded since
  G.overlay = overlay;
  $("map-overlay-error").hidden = !failed;  // not an empty graph: the regions, trust and scale could not be read
  $("map-overlay-error").textContent = failed && `Регіони й довіра недоступні, тож інсайти цього рівня не намальовано: ${failed}`;
  mergePatterns();
  const regions = (G.overlay || empty()).features;
  G.levels = [...new Set(regions.map((f) => f.properties.resolution))].sort((a, b) => b - a);  // finest first
  G.map.getSource("regions").setData(G.overlay || empty());
  G.map.getSource("region-labels").setData({ type: "FeatureCollection", features: regions.filter((f) => f.properties.n > 0).map((f) => ({  // a 0 is not written
    type: "Feature", geometry: { type: "Point", coordinates: centre(f) }, properties: { resolution: f.properties.resolution, icon: countIcon(f.properties.n) } })) });
  G.level = undefined;
  paintRegions();
}

const centre = (f) => {  // the region's H3 centre, where its count is drawn; a ring's mean for a feature without one
  if ((f.properties || {}).centre) return f.properties.centre;
  const r = f.geometry.coordinates[0].slice(1), x0 = r[0][0], lon = r.reduce((a, c) => a + unwrap(c[0], x0), 0) / r.length;
  return [unwrap(lon, 0), r.reduce((a, c) => a + c[1], 0) / r.length];
};

function countIcon(n) {
  /* a region's insight count as an image for the symbol layer: the offline basemap has no glyphs to draw text with */
  const id = `rn-${n}`;
  if (G.map.hasImage(id)) return id;
  const text = String(n), ratio = 2, w = 8 + 7 * text.length, h = 16, canvas = document.createElement("canvas");
  canvas.width = w * ratio; canvas.height = h * ratio;
  const g = canvas.getContext("2d");
  g.scale(ratio, ratio);
  g.font = "700 10.5px ui-monospace, Consolas, Menlo, monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.lineWidth = 3; g.strokeStyle = "rgba(0, 0, 0, .85)"; g.strokeText(text, w / 2, h / 2);
  g.fillStyle = C.region; g.fillText(text, w / 2, h / 2);
  G.map.addImage(id, g.getImageData(0, 0, canvas.width, canvas.height), { pixelRatio: ratio });
  return id;
}

function regionLevel() {
  /* the region level that fits the zoom (zoom − 3 ≈ H3 resolution); the drawn one stays until another fits by half a
     zoom step more, so a zoom near a threshold does not flip it back and forth */
  const want = G.map.getZoom() - 3, levels = G.levels;
  if (!levels.length) return null;
  const best = levels.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
  return levels.includes(G.level) && Math.abs(want - G.level) - Math.abs(want - best) < 0.5 ? G.level : best;
}

function paintRegions() {
  /* one level outlined with its counts, and the next coarser one pale beneath it: crossing a threshold onto a sparse
     finer level leaves the coarser outlines on the map */
  if (!G.map || !G.map.getLayer("regions-line")) return;
  const level = $("lyr-regions").checked ? regionLevel() : null;
  if (level === G.level) return;
  G.level = level;
  const coarser = G.levels.find((lv) => level !== null && lv < level);
  G.map.setFilter("regions-line", ["in", ["get", "resolution"], ["literal", level === null ? [] : [level, coarser ?? level]]]);
  G.map.setPaintProperty("regions-line", "line-opacity", ["*", ["case", ["==", ["get", "resolution"], level ?? -1], 0.75, 0.22],
    ["case", [">", ["get", "n"], 0], 1, 0.3]]);  // a region without a characteristic insight stays pale
  G.map.setFilter("regions-n", ["==", ["get", "resolution"], level ?? -1]);  // -1: no region
}

function paintZone() {
  /* the selected cells outlined (H3 ids are global: the zone survives a switch of dataset, and a map of another level,
     which does not draw them, says how many it has), and the zone bar */
  const inZone = G.data ? G.data.features.filter((f) => G.zone.has(f.properties.h3)) : [], shown = inZone.length;
  if (G.map && G.map.getSource("zone")) G.map.getSource("zone").setData({ type: "FeatureCollection", features: inZone });
  $("map-zone").hidden = !G.zone.size;
  $("map-zone-n").textContent = `ЗОНА · ${plural(G.zone.size, "комірка", "комірки", "комірок")}` + (G.data && shown < G.zone.size ? ` · на цій мапі: ${shown}` : "");
}

function colorBy(metric) {
  G.metric = metric;
  const vals = G.data.features.map((f) => f.properties[metric]).filter((v) => typeof v === "number").sort((a, b) => a - b);
  const stops = [];
  for (let i = 1; i < RAMP.length; i++) { const q = vals[Math.floor((i / RAMP.length) * (vals.length - 1))]; if (q !== undefined && (!stops.length || q > stops[stops.length - 1])) stops.push(q); }
  const step = ["step", ["get", metric], RAMP[0]];
  stops.forEach((q, i) => step.push(q, RAMP[Math.min(i + 1, RAMP.length - 1)]));
  G.map.setPaintProperty("cells-fill", "fill-color", ["case", ["==", ["typeof", ["get", metric]], "number"], stops.length ? step : RAMP[3], "#141a26"]);
  $("map-ramp").innerHTML = `<div class="ramp">${RAMP.map((c) => `<i style="background:${c}"></i>`).join("")}</div>` +
    `<div class="ramp-lbl"><span>${num(vals[0])}</span><span>${esc(human(metric))} · квантилі</span><span>${num(vals[vals.length - 1])}</span></div>`;
}

function wireHud() {
  $("map-metric").addEventListener("change", (e) => colorBy(e.target.value));
  const vis = (ids, on) => ids.forEach((id) => G.map.setLayoutProperty(id, "visibility", on ? "visible" : "none"));
  $("lyr-cells").addEventListener("change", (e) => vis(["cells-fill", "cells-line"], e.target.checked));
  $("lyr-lisa").addEventListener("change", (e) => vis(["lisa-hot", "lisa-cold"], e.target.checked));
  $("lyr-pts").addEventListener("change", (e) => vis(["pts-halo", "pts-core"], e.target.checked));
  $("lyr-borders").addEventListener("change", (e) => vis(["border", "border-glow", "admin1", "river", "lake"], e.target.checked));
  $("lyr-labels").addEventListener("change", placeDensity);
  $("lyr-regions").addEventListener("change", () => { G.level = undefined; paintRegions(); });
  $("map-card-close").addEventListener("click", () => GeoMap.unfocus());
  $("map-zone-ask").addEventListener("click", () => ask($("question").value, { cells: GeoMap.zone() }));
  $("map-zone-clear").addEventListener("click", () => GeoMap.clearZone());
  $("mw-fold").addEventListener("click", () => { W.folded = !W.folded; renderWindow(); });
}


function renderCard(p, item, pts) {
  const sp = p.spatial || {};
  $("map-card-key").textContent = item ? item.key : "ІНСАЙТ";
  $("map-card-body").innerHTML =
    `<div class="mc-scope">${esc(p.reading.where)}</div>` +
    `<div class="mc-shift"><span class="${p.phenomenon_type === "covariance" ? "cov" : p.effect_size >= 0 ? "up" : "down"}">${esc(p.reading.what)}</span></div>` +
    `<div class="mc-grid">${(p.level  // found at another level: drawn by the cells holding most of its points
      ? [["де знайдено", esc(levelName(p.level))], [term(p.level === "table" ? "cells_footprint" : "cells_inside"), `${num(p.rows.length, 0)} (${num(sp.footprint_share * 100, 0)}% ${word("points")})`], [term("points"), pts ? num(pts.n_points, 0) : "…"], [term("support"), num(p.support, 0)], [term("weight"), num(p.weight)]]
      : [[term("cells_extent"), num(p.rows.length, 0)], [term("compactness"), num(sp.compactness)], [term("points"), pts ? num(pts.n_points, 0) : "…"], [term("n_eff"), num(sp.n_eff, 1)], [term("moran_i"), num(sp.moran_i)], [term("weight"), num(p.weight)]])
      .map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("")}</div>` +
    (p.numbers && p.numbers.n_eff ? `<div class="mc-note">${esc(p.numbers.n_eff)}</div>` : "") +
    (item ? `<div class="mc-path">${esc(word(reached(item)))}</div>` : "");
  $("map-card").hidden = false;
}

function trustTags(pid) {
  /* an insight's credibility and scale label, as the graph holds them */
  const t = G.overlay && G.overlay.patterns[pid];
  if (!t) return "";
  const [cls, words] = cred(t.credibility);
  return `<i class="pp-tag ${cls}">${TERMS[`credibility_${t.credibility}`] ? term(`credibility_${t.credibility}`) : esc(words)}</i>` + (t.scale ? `<i class="pp-tag">${term(`scale_${t.scale}`)}</i>` : "");
}

function cellPopup(f, lngLat) {
  const p = f.properties, row = p.row;
  const skip = new Set(["row", "h3"]);
  const rows = Object.entries(p).filter(([k, v]) => !skip.has(k) && v !== null && v !== "missing" && !k.endsWith("_band"))
    .map(([k, v]) => `<tr><th>${esc(human(k))}</th><td>${typeof v === "number" ? num(v, 3) : esc(v)}</td></tr>`).join("");
  const pids = (G.byRow[row] || []).sort((a, b) => G.patterns[b].weight - G.patterns[a].weight);
  const ins = pids.slice(0, 6).map((pid) => { const q = G.patterns[pid]; return `<div class="pp-ins" data-pid="${esc(pid)}" tabindex="0" role="button"><b class="${q.effect_size >= 0 ? "up" : "down"}">${esc(human(q.target))} ${signed(q.effect_size)} sd</b><span>${esc(q.scope.map(human).join(" · "))}</span>${q.level ? `<i class="pp-tag">${esc(levelName(q.level))}</i>` : ""}${trustTags(pid)}</div>`; }).join("");
  const region = G.level != null && G.overlay ? G.overlay.features.find((f) => f.properties.resolution === G.level && f.properties.rows.includes(row)) : null;
  const reg = region ? `<div class="pp-sub">${cap(word("region"))}</div><div class="pp-reg">${esc(region.properties.id)}<span>${esc(plural(region.properties.n, "характерний інсайт", "характерні інсайти", "характерних інсайтів"))}</span></div>` : "";
  const html = `<div class="pp"><div class="pp-head"><span>${esc(p.h3)}</span><span class="lisa ${esc(String(p.lisa_points).replace(" ", "-"))}">${esc(p.lisa_points)}</span></div>` +
    `<div class="pp-scroll">${ins ? `<div class="pp-sub">Інсайти цієї комірки (${pids.length})</div>${ins}` : ""}${reg}${ins || reg ? `<div class="pp-sub">${cap(word("cell"))}</div>` : ""}<table>${rows}</table></div>` +
    `<button type="button" class="pp-zone">${G.zone.has(p.h3) ? "− Прибрати із зони" : "+ Додати до зони"}</button>` +
    `<button type="button" class="pp-ask">Запитати про цю зону</button></div>`;
  GeoMap.closePopup();
  const popup = G.popup = new maplibregl.Popup({ maxWidth: "360px", className: "sig-pop" }).setLngLat(lngLat).setHTML(html).addTo(G.map);
  popup.on("close", () => { if (G.popup === popup) G.popup = null; });
  const el = popup.getElement();
  el.querySelectorAll(".pp-ins").forEach((x) => x.addEventListener("click", () => GeoMap.focusPattern(x.dataset.pid)));
  el.querySelector(".pp-zone").addEventListener("click", () => { GeoMap.closePopup(); GeoMap.toggleZone(p.h3); });
  el.querySelector(".pp-ask").addEventListener("click", () => {  // the selected zone if it holds this cell, else the cell; the typed text, if any
    GeoMap.closePopup(); ask($("question").value, { cells: G.zone.has(p.h3) ? GeoMap.zone() : [p.h3] });
  });
}

// ponytail: PoC — the map as a window on the hidden graph (GET /api/geo/{ds}/window): the themes and insights of the view,
// re-read after every pan or zoom; a click draws the insight and describes it in the chat, whose links walk the graph on
const W = { seq: 0, timer: null, data: null, fold: {}, folded: null, all: false };  // folded: decided by the map's width at the first render
const ROWS_SHOWN = 8;  // the rows listed before «показати всі»: those characteristic of the most regions in focus
const FOCUS_INSET = 0.15;  // the share of the map left out at each edge, where the eye rarely rests (docs/08 §8.7)

function scheduleWindow() { clearTimeout(W.timer); W.timer = setTimeout(loadWindow, 250); }

function focusBox() {
  /* what the list reads, as [west, south, east, north]: the map less FOCUS_INSET at each edge, the panels over it not apart */
  const w = $("map").clientWidth, h = $("map").clientHeight, dx = w * FOCUS_INSET, dy = h * FOCUS_INSET;
  const nw = G.map.unproject([dx, dy]), se = G.map.unproject([w - dx, h - dy]);
  return [nw.lng, se.lat, se.lng, nw.lat];
}

async function loadWindow() {
  if (!G.map || !G.ds || G.loading || S.view !== "map") return;
  const [x0, s, x1, n] = focusBox(), seq = ++W.seq, lat = (y) => Math.max(-90, Math.min(90, y)).toFixed(4), lon = (x) => unwrap(x, 0).toFixed(4);
  const [w, e] = x1 - x0 >= 360 ? ["-180", "180"] : [lon(x0), lon(x1)];  // west > east: across ±180°
  const level = G.level ?? regionLevel();  // the drawn level, or the one the zoom would draw with the regions switched off
  const query = `bbox=${[w, lat(s), e, lat(n)].join(",")}${level == null ? "" : `&level=${level}`}`;
  const data = await api(`/api/geo/${encodeURIComponent(G.ds)}/window?${query}`).catch((e) => ({ error: e.message }));
  if (seq !== W.seq) return;  // a later view answered first
  W.data = data; renderWindow();
}

const windowItem = (pid) => ((W.data && W.data.themes) || []).flatMap((t) => t.insights).find((i) => i.id === pid) || null;
const insightTail = (i) => i.label.slice(i.label.lastIndexOf(": ") + 2);  // the headline after its scope: "score +1.20 sd"

const reading = (pid) => { const r = (G.patterns[pid] || {}).reading; return r ? `${r.where} — ${r.what}` : ""; };  // a row's insight as the map holds it

function renderWindow() {
  const el = $("map-window"), d = W.data;
  el.hidden = !d || !G.ds;
  if (el.hidden) return;
  if (W.folded === null) W.folded = $("map").clientWidth < 800;  // a narrow map starts with the header alone: the list would cover its cells
  el.classList.toggle("folded", W.folded); $("mw-fold").textContent = W.folded ? "+" : "–";
  if (d.error) { $("mw-meta").textContent = ""; $("mw-body").innerHTML = `<div class="mw-err">${esc(d.error)}</div>`; return; }
  $("mw-meta").textContent = `${plural(d.insights, "інсайт", "інсайти", "інсайтів")} · ${plural(d.themes.length, "тема", "теми", "тем")} · ${plural(d.points, "подія", "події", "подій")}`;
  const row = (i) => `<button type="button" class="mw-i${i.id === G.focusId ? " on" : ""}" data-pid="${esc(i.id)}" title="${esc(reading(i.id) || i.label)} · ${word("points")} у вікні: ${num(i.points, 0)} (${num(i.share * 100, 0)}% усіх) · ${word("cells_in_view")}: ${num(i.cells, 0)} · характерний для регіонів у фокусі: ${num(i.regions, 0)}${i.wider ? " · характерний для ширшої ділянки: тут замало подій, щоб це перевірити" : ""}">` +
    `<i class="lvl${i.level === "table" ? " tbl" : ""}"${i.also.length ? ` title="також знайдено: ${esc(i.also.map((a) => levelName(a.level)).join(", "))}"` : ""}>${esc(levelName(i.level))}${i.also.length ? " +" + i.also.length : ""}</i><span class="mw-nm">${esc(i.values.map(human).join(" · "))}` +
    `<em class="${i.ptype === "covariance" ? "cov" : i.effect >= 0 ? "up" : "down"}">${esc(((G.patterns[i.id] || {}).reading || {}).headline || human(insightTail(i)))}</em></span>` +  // the reading's headline in the declared words; the label's tail without one
    `<b title="${esc(`${word("window_lift")}: ${i.lift_words}\n${TERMS.window_lift[1]}`)}">×${num(i.lift, 1)}</b></button>`;
  const more = d.insights <= ROWS_SHOWN ? "" : `<div class="mw-more">${W.all ? `Показано всі ${d.insights}` : `Показано ${ROWS_SHOWN} з ${d.insights} · наблизьтесь, щоб побачити особливості ділянок`}` +
    `<button type="button" class="mw-all">${W.all ? `лише перші ${ROWS_SHOWN}` : "показати всі"}</button></div>`;
  const themes = d.themes.map((t) => ({ ...t, rows: t.insights.filter((i) => W.all || i.rank < ROWS_SHOWN) })).filter((t) => t.rows.length);
  $("mw-body").innerHTML = !d.themes.length ? `<div class="mw-err">Тут нічого характерного — віддаліться, щоб побачити ширшу ділянку.</div>` : more + themes.map((t, k) => {
    const open = W.fold[t.id ?? ""] ?? k < 3;  // the three leading themes open until folded
    return `<div class="mw-theme${open ? " open" : ""}"><button type="button" class="mw-th" data-id="${esc(t.id)}" title="${esc(t.reading || t.label)}"><span>${open ? "▾" : "▸"}</span><span class="mw-tl">${esc(t.label)}</span><b>${t.rows.length}</b></button>` +
      (open ? t.rows.map(row).join("") : "") + `</div>`;
  }).join("");
}
