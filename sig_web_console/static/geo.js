/* SIG map view: H3 cells of geo datasets over an offline basemap, the answer's evidence on the map and its tour (docs/08_interface.md §8.7). */
"use strict";

const G = { map: null, ready: null, datasets: [], ds: null, data: null, patterns: {}, byRow: [], bounds: [], metric: "points", pending: null, evidence: [], hl: [], focus: [], tour: null, places: [] };
const RAMP = ["#241046", "#3f1670", "#6a1d96", "#9b2a9f", "#cf4475", "#f2733f", "#ffc23f"];
const C = { up: "#ff8a3d", down: "#36d6ff", hot: "#ff3b4f", cold: "#3d8bff", amber: "#ffb000", grid: "#121a26" };

const GeoMap = {
  has: (pid) => Boolean(G.patterns[pid]),
  datasetOf: (pid) => (G.patterns[pid] ? G.ds : null),

  async refresh() {
    try { G.datasets = await api("/api/geo"); } catch (e) { G.datasets = []; }
    $("map-tab").hidden = !G.datasets.length;
    if (!G.datasets.length) { G.ds = null; G.data = null; G.patterns = {}; if (S.view === "map") setView("graph"); return; }
    const filter = $("dataset-filter").value;
    const want = G.datasets.find((d) => d.dataset_id === filter) || G.datasets.find((d) => d.dataset_id === G.ds) || G.datasets[0];
    if (want.dataset_id !== G.ds || !G.data) await loadDataset(want.dataset_id);
    if (!S.chat.length) renderThread();  // the welcome card offers this dataset's questions
  },

  async show() {
    await ensureMap();
    G.map.resize();
    if (G.pending) { const qa = G.pending; G.pending = null; GeoMap.onAnswer(qa); }
    else if (G.data && !G.evidence.length) G.map.fitBounds([[G.data.bbox[0], G.data.bbox[1]], [G.data.bbox[2], G.data.bbox[3]]], { padding: 40, duration: 0 });
  },

  suggestions() {
    if (!G.data) return [];
    const seen = new Set(), out = [];
    for (const p of [...G.data.patterns].sort((a, b) => b.weight - a.weight)) {
      if (p.phenomenon_type === "covariance" || seen.has(p.target)) continue;
      seen.add(p.target);
      out.push(`Де ${human(p.target)} ${p.effect_size > 0 ? "вища" : "нижча"}, ніж зазвичай?`);
      if (out.length === 2) break;
    }
    const hot = G.data.patterns.find((p) => p.scope.some((s) => s.includes("hot spot")));
    if (hot) out.unshift("Що відрізняє зони hot spot від решти?");
    const named = (p) => p.scope.filter((s) => !/= (no cluster|missing)$/.test(s));
    const far = G.data.patterns.find((p) => (p.spatial.colocated || []).length && named(p).length >= 2 && !named(p).some((s) => s.includes("neighbours")));
    if (far) out.push(`Що ще відбувається там, де ${named(far).map((s) => s.replace(" = ", " ")).join(" і ")}?`);
    return out.slice(0, 4);
  },

  onAnswer(qa) {
    const items = ((qa.view && qa.view.evidence && qa.view.evidence.items) || []);
    const geoDs = items.map((i) => i.provenance.dataset_id).find((ds) => G.datasets.some((d) => d.dataset_id === ds));
    if (!geoDs) { GeoMap.clear(); return; }
    if (!G.map || S.view !== "map") { G.pending = qa; return; }
    const apply = () => {
      G.evidence = items.filter((i) => G.patterns[i.pattern_id]);
      setStates(G.hl, "hl", 0); G.hl = [];
      for (const it of [...G.evidence].reverse()) {  // P1 is painted last: it wins a shared cell
        const sign = it.phenomenon_type === "covariance" ? 0.5 : (it.statistics.effect_size >= 0 ? 1 : -1);
        const rows = G.patterns[it.pattern_id].rows;
        rows.forEach((r) => G.map.setFeatureState({ source: "cells", id: r }, { hl: sign }));
        G.hl.push(...rows);
      }
      G.map.setPaintProperty("cells-fill", "fill-opacity", G.evidence.length ? 0.14 : 0.55);
      renderTourBar();
      if (G.evidence.length) GeoMap.focusPattern(G.evidence[0].pattern_id);
    };
    if (geoDs !== G.ds) loadDataset(geoDs).then(apply); else apply();
  },

  clear() {
    stopTour();
    if (!G.map || !G.map.getSource("cells")) return;
    setStates(G.hl, "hl", 0); setStates(G.focus, "focus", false); G.hl = []; G.focus = []; G.evidence = [];
    G.map.setPaintProperty("cells-fill", "fill-opacity", 0.55);
    G.map.getSource("pts").setData(empty());
    $("map-card").hidden = true; renderTourBar();
  },

  async focusPattern(pid) {
    const p = G.patterns[pid]; if (!p || !G.map) return;
    setStates(G.focus, "focus", false);
    G.focus = p.rows; setStates(G.focus, "focus", true);
    fitRows(p.rows, 90);
    const item = G.evidence.find((i) => i.pattern_id === pid);
    renderCard(p, item, null);
    document.querySelectorAll("#map-tour .tk").forEach((b) => b.classList.toggle("on", b.dataset.pid === pid));
    try {
      const pts = await api(`/api/geo/${encodeURIComponent(G.ds)}/patterns/${encodeURIComponent(pid)}`);
      G.map.getSource("pts").setData({ type: "FeatureCollection", features: pts.points.map((c) => ({ type: "Feature", geometry: { type: "Point", coordinates: c }, properties: {} })) });
      renderCard(p, item, pts);
    } catch (e) { /* the card stays without the point count */ }
  },
};

const empty = () => ({ type: "FeatureCollection", features: [] });
function setStates(rows, key, value) { if (G.map && G.map.getSource("cells")) rows.forEach((r) => G.map.setFeatureState({ source: "cells", id: r }, { [key]: value })); }

function fitRows(rows, padding) {
  if (!rows.length) return;
  const b = rows.reduce((a, r) => { const x = G.bounds[r]; return x ? [Math.min(a[0], x[0]), Math.min(a[1], x[1]), Math.max(a[2], x[2]), Math.max(a[3], x[3])] : a; }, [180, 90, -180, -90]);
  G.map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding, maxZoom: 10, duration: 900 });
}

function graticule() {
  const f = [];
  for (let x = 18; x <= 44; x += 1) f.push({ type: "Feature", properties: { major: x % 5 === 0 }, geometry: { type: "LineString", coordinates: [[x, 41.5], [x, 57.5]] } });
  for (let y = 42; y <= 57; y += 1) f.push({ type: "Feature", properties: { major: y % 5 === 0 }, geometry: { type: "LineString", coordinates: [[18, y], [44, y]] } });
  return { type: "FeatureCollection", features: f };
}

function ensureMap() {
  if (G.ready) return G.ready;
  G.map = new maplibregl.Map({
    container: "map", attributionControl: false, dragRotate: false, pitchWithRotate: false, center: [36.5, 48.3], zoom: 6, maxZoom: 13,
    style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#05080d" } }] },
  });
  G.map.touchZoomRotate.disableRotation();
  G.map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  G.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  G.ready = new Promise((resolve) => G.map.on("load", resolve)).then(async () => {
    const base = await fetch("/vendor/geo/basemap.json").then((r) => r.json());
    const m = G.map;
    m.addSource("base", { type: "geojson", data: base });
    m.addSource("grid", { type: "geojson", data: graticule() });
    m.addSource("cells", { type: "geojson", data: empty() });
    m.addSource("pts", { type: "geojson", data: empty() });
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
    const hl = ["number", ["feature-state", "hl"], 0];
    const hlColor = ["case", [">", hl, 0.75], C.up, ["<", hl, 0], C.down, "#3ccb92"];
    layer("hl-fill", "fill", "cells", { "fill-color": hlColor, "fill-opacity": ["case", ["boolean", ["feature-state", "focus"], false], 0.6, ["!=", hl, 0], 0.22, 0] });
    layer("hl-glow", "line", "cells", { "line-color": hlColor, "line-width": ["case", ["boolean", ["feature-state", "focus"], false], 8, 0], "line-blur": 6, "line-opacity": 0.6 });
    layer("hl-line", "line", "cells", { "line-color": hlColor, "line-width": ["case", ["!=", hl, 0], 1.6, 0] });
    layer("focus-line", "line", "cells", { "line-color": C.amber, "line-width": ["case", ["boolean", ["feature-state", "focus"], false], 2.6, 0], "line-opacity": 1 });
    layer("pts-halo", "circle", "pts", { "circle-color": "#ff4d6d", "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 3, 11, 8], "circle-blur": 1, "circle-opacity": 0.45 });
    layer("pts-core", "circle", "pts", { "circle-color": "#ffe0e6", "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 0.8, 11, 2.4] });
    let t = 0;
    const pulse = () => { t += 0.06; if (G.focus.length) m.setPaintProperty("focus-line", "line-opacity", 0.55 + 0.45 * Math.sin(t)); requestAnimationFrame(pulse); };
    requestAnimationFrame(pulse);
    addPlaces(await fetch("/vendor/geo/places.json").then((r) => r.json()));
    m.on("mousemove", (e) => {
      const f = m.queryRenderedFeatures(e.point, { layers: ["cells-fill"] })[0];
      $("map-cursor").textContent = `${e.lngLat.lat.toFixed(4)}°N ${e.lngLat.lng.toFixed(4)}°E · z${m.getZoom().toFixed(1)}` + (f ? ` · ${f.properties.h3} · ${num(f.properties.points, 0)} pts` : "");
      m.getCanvas().style.cursor = f ? "pointer" : "";
    });
    m.on("click", "cells-fill", (e) => cellPopup(e.features[0], e.lngLat));
    m.on("zoom", placeDensity);
    wireHud();
  });
  return G.ready;
}

function addPlaces(places) {
  G.places = places.features.map((f) => {
    const p = f.properties, el = document.createElement("div");
    el.className = `place r${Math.min(p.rank, 10)}${p.pop > 500000 ? " big" : ""}`;
    el.innerHTML = `<i></i><span>${esc(p.name)}</span>`;
    return { el, rank: p.rank, pop: p.pop, marker: new maplibregl.Marker({ element: el, anchor: "left" }).setLngLat(f.geometry.coordinates).addTo(G.map) };
  });
  placeDensity();
}

function placeDensity() {
  const z = G.map.getZoom(), on = $("lyr-labels").checked;
  const minPop = z < 6.5 ? 600000 : z < 7.5 ? 100000 : z < 8.5 ? 30000 : 0;
  G.places.forEach((p) => { p.el.hidden = !on || p.pop < minPop; });
}

async function loadDataset(ds) {
  await ensureMap();
  const data = await api(`/api/geo/${encodeURIComponent(ds)}/cells`);
  G.ds = ds; G.data = data; G.hl = []; G.focus = []; G.evidence = [];
  G.patterns = Object.fromEntries(data.patterns.map((p) => [p.id, p]));
  G.byRow = data.features.map(() => []);
  data.patterns.forEach((p) => p.rows.forEach((r) => G.byRow[r] && G.byRow[r].push(p.id)));
  G.bounds = data.features.map((f) => f.geometry.coordinates[0].reduce((a, c) => [Math.min(a[0], c[0]), Math.min(a[1], c[1]), Math.max(a[2], c[0]), Math.max(a[3], c[1])], [180, 90, -180, -90]));
  G.map.getSource("cells").setData(data);
  G.map.getSource("pts").setData(empty());
  const info = G.datasets.find((d) => d.dataset_id === ds) || {};
  const numeric = Object.keys(data.features[0] ? data.features[0].properties : {}).filter((k) => k !== "row" && data.features.some((f) => typeof f.properties[k] === "number"));
  if (!numeric.includes(G.metric)) G.metric = numeric[0] || "points";
  $("map-metric").innerHTML = numeric.map((k) => `<option value="${esc(k)}" ${k === G.metric ? "selected" : ""}>${esc(human(k))}</option>`).join("");
  colorBy(G.metric);
  const sac = info.sac || {};
  $("map-title").textContent = info.filename || ds;
  $("map-meta").innerHTML = `H3 res ${info.resolution} · ${num(info.cells, 0)} cells · ${num(info.points, 0)} points` +
    (sac.validated ? `<br><span title="Validated candidates significant without / with the effective-sample-size correction for spatial autocorrelation">SAC: ${sac.significant_iid} → <b>${sac.significant_corrected}</b> significant of ${sac.validated} (n<sub>eff</sub>)</span>` : "");
  G.map.fitBounds([[data.bbox[0], data.bbox[1]], [data.bbox[2], data.bbox[3]]], { padding: 40, duration: 0 });
  $("map-card").hidden = true; renderTourBar();
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
    `<div class="ramp-lbl"><span>${num(vals[0])}</span><span>${esc(human(metric))} · quantiles</span><span>${num(vals[vals.length - 1])}</span></div>`;
}

function wireHud() {
  $("map-metric").addEventListener("change", (e) => colorBy(e.target.value));
  const vis = (ids, on) => ids.forEach((id) => G.map.setLayoutProperty(id, "visibility", on ? "visible" : "none"));
  $("lyr-cells").addEventListener("change", (e) => vis(["cells-fill", "cells-line"], e.target.checked));
  $("lyr-lisa").addEventListener("change", (e) => vis(["lisa-hot", "lisa-cold"], e.target.checked));
  $("lyr-pts").addEventListener("change", (e) => vis(["pts-halo", "pts-core"], e.target.checked));
  $("lyr-borders").addEventListener("change", (e) => vis(["border", "border-glow", "admin1", "river", "lake"], e.target.checked));
  $("lyr-labels").addEventListener("change", placeDensity);
  $("map-card-close").addEventListener("click", () => { $("map-card").hidden = true; setStates(G.focus, "focus", false); G.focus = []; G.map.getSource("pts").setData(empty()); });
}

function shiftText(p, item) {
  if (item && item.phenomenon_type === "covariance") return esc(item.relationship || "correlation change");
  const z = item ? item.statistics.effect_size : p.effect_size;
  return `<b class="${z >= 0 ? "up" : "down"}">${esc(human(p.target))} ${signed(z)} sd</b>`;
}

function renderCard(p, item, pts) {
  const sp = p.spatial || {};
  $("map-card-key").textContent = item ? item.key : "INSIGHT";
  $("map-card-body").innerHTML =
    `<div class="mc-scope">${p.scope.map((s) => esc(human(s))).join(" · ")}</div>` +
    `<div class="mc-shift">${shiftText(p, item)}</div>` +
    `<div class="mc-grid"><span>cells</span><b>${num(p.rows.length, 0)}</b><span>compactness</span><b>${num(sp.compactness)}</b>` +
    `<span>points</span><b>${pts ? num(pts.n_points, 0) : "…"}</b><span>n<sub>eff</sub></span><b>${num(sp.n_eff, 0)}</b>` +
    `<span>Moran's I</span><b>${num(sp.moran_i)}</b><span>weight</span><b>${num(p.weight)}</b></div>` +
    (item && item.path_text ? `<div class="mc-path">${esc(item.path_text)}</div>` : "");
  $("map-card").hidden = false;
}

function renderTourBar() {
  const bar = $("map-tour");
  if (!G.evidence.length) { bar.hidden = true; return; }
  bar.hidden = false;
  bar.innerHTML = `<button type="button" class="tour-play" title="Fly through the evidence">${G.tour ? "■" : "▶"} TOUR</button>` +
    G.evidence.map((i) => `<button type="button" class="tk ${i.statistics.effect_size >= 0 ? "up" : "down"}" data-pid="${esc(i.pattern_id)}" title="${esc(i.scope.join(" · "))}">${esc(i.key)}</button>`).join("");
  bar.querySelector(".tour-play").addEventListener("click", () => (G.tour ? stopTour() : startTour()));
  bar.querySelectorAll(".tk").forEach((b) => b.addEventListener("click", () => { stopTour(); GeoMap.focusPattern(b.dataset.pid); }));
}

function startTour() {
  let i = 0;
  const step = () => { if (i >= G.evidence.length) { stopTour(); return; } GeoMap.focusPattern(G.evidence[i++].pattern_id); };
  G.tour = setInterval(step, 4200); step(); renderTourBar();
}
function stopTour() { if (G.tour) { clearInterval(G.tour); G.tour = null; renderTourBar(); } }

function cellPopup(f, lngLat) {
  const p = f.properties, row = p.row;
  const skip = new Set(["row", "h3"]);
  const rows = Object.entries(p).filter(([k, v]) => !skip.has(k) && v !== null && v !== "missing" && !k.endsWith("_band"))
    .map(([k, v]) => `<tr><th>${esc(human(k))}</th><td>${typeof v === "number" ? num(v, 3) : esc(v)}</td></tr>`).join("");
  const pids = (G.byRow[row] || []).sort((a, b) => G.patterns[b].weight - G.patterns[a].weight);
  const ins = pids.slice(0, 6).map((pid) => { const q = G.patterns[pid]; return `<div class="pp-ins" data-pid="${esc(pid)}">${shiftText(q, null)}<span>${esc(q.scope.map(human).join(" · "))}</span></div>`; }).join("");
  const html = `<div class="pp"><div class="pp-head"><span>${esc(p.h3)}</span><span class="lisa ${esc(String(p.lisa_points).replace(" ", "-"))}">${esc(p.lisa_points)}</span></div>` +
    `<div class="pp-scroll">${ins ? `<div class="pp-sub">Insights covering this cell (${pids.length})</div>${ins}<div class="pp-sub">Cell</div>` : ""}<table>${rows}</table></div>` +
    `<button type="button" class="pp-ask">Запитати про цю зону</button></div>`;
  const popup = new maplibregl.Popup({ maxWidth: "360px", className: "sig-pop" }).setLngLat(lngLat).setHTML(html).addTo(G.map);
  const el = popup.getElement();
  el.querySelectorAll(".pp-ins").forEach((x) => x.addEventListener("click", () => GeoMap.focusPattern(x.dataset.pid)));
  el.querySelector(".pp-ask").addEventListener("click", () => {
    const top = pids[0] && G.patterns[pids[0]];
    const q = top ? `Що відбувається там, де ${top.scope.map((s) => s.replace(" = ", " ")).join(" і ")}?` : `Що характерно для зон ${p.lisa_points}?`;
    popup.remove(); ask(q);
  });
}
