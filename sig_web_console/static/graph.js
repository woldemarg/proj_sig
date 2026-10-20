/* SIG UI — the graph, the latent sphere and the insights table, the answer's highlight, the details drawer (docs/08_interface.md §8.2–8.4). */
"use strict";

// graph
function graphStyle() {
  const ink = cssVar("--graph-ink") || "#3a4050", edge = cssVar("--graph-edge") || "#c8cdd8";
  const up = cssVar("--up"), down = cssVar("--down"), cov = cssVar("--cov"), anchor = cssVar("--anchor"), path = cssVar("--path"), bad = cssVar("--bad");
  const surface = cssVar("--surface") || "#fff", muted = cssVar("--muted"), region = cssVar("--region");
  return [
    { selector: "node", style: { label: "data(label)", "font-size": 11, "font-family": "Inter, Segoe UI, system-ui, sans-serif", "text-wrap": "wrap", "text-max-width": 120, "text-valign": "bottom", "text-margin-y": 4, color: ink, "border-width": 0, "min-zoomed-font-size": 8, "text-outline-color": surface, "text-outline-width": 2 } },
    { selector: 'node[kind="Pattern"]', style: { shape: "ellipse", width: "mapData(weight, 0, 1, 12, 30)", height: "mapData(weight, 0, 1, 12, 30)", "background-color": up } },
    { selector: 'node[kind="Pattern"][direction < 0]', style: { "background-color": down } },
    { selector: 'node[kind="Pattern"][ptype="covariance"]', style: { "background-color": cov, shape: "round-rectangle" } },
    { selector: 'node[kind="Attractor"]', style: { shape: "hexagon", width: "mapData(n_patterns, 1, 12, 36, 66)", height: "mapData(n_patterns, 1, 12, 32, 58)", "background-color": anchor, "font-size": 11.5, "font-weight": 700, color: ink, "text-max-width": 170 } },
    { selector: 'node[kind="Dimension"]', style: { shape: "round-rectangle", width: 44, height: 18, "background-color": muted, "font-size": 9, "text-valign": "center", "text-margin-y": 0, color: "#fff", "text-outline-width": 0 } },
    { selector: 'node[kind="Metric"]', style: { shape: "tag", width: 48, height: 18, "background-color": cov, "font-size": 9, "text-valign": "center", "text-margin-y": 0, color: "#fff", "text-outline-width": 0 } },
    { selector: 'node[kind="Region"]', style: { shape: "diamond", width: "mapData(patterns, 1, 12, 18, 34)", height: "mapData(patterns, 1, 12, 18, 34)", "background-color": region, "font-size": 9 } },
    { selector: 'node[kind="Dataset"], node[kind="Batch"]', style: { display: "none" } },
    { selector: 'node[kind="PlaneLabel"]', style: { label: "data(label)", "background-opacity": 0, "font-size": 13, "font-weight": 700, color: muted, "text-valign": "center", "text-halign": "right", events: "no", "text-outline-width": 0 } },
    { selector: "edge", style: { width: 1, "line-color": edge, "curve-style": "bezier", opacity: 0.75 } },
    { selector: 'edge[type="SPECIALIZES"]', style: { "line-color": muted, "target-arrow-shape": "triangle", "target-arrow-color": muted, "arrow-scale": 0.7, opacity: 0.6 } },
    { selector: 'edge[type="SIBLING"]', style: { "line-style": "dotted", "line-color": edge } },
    { selector: 'edge[type="CONTRASTS"]', style: { "line-style": "dashed", "line-color": bad, width: 1.5, opacity: 0.7 } },
    { selector: 'edge[type="CO_LOCATED"]', style: { "line-style": "dashed", "line-color": cssVar("--path"), width: "mapData(weight, 0, 1, 0.6, 2.4)", opacity: 0.6 } },
    { selector: 'edge[type="IN_REGION"]', style: { "line-style": "dashed", "line-color": region, width: "mapData(weight, 0, 1, 0.5, 2)", opacity: 0.55 } },
    { selector: 'edge[type="PARENT_OF"]', style: { "line-color": region, "target-arrow-shape": "triangle", "target-arrow-color": region, "arrow-scale": 0.6, opacity: 0.6 } },
    { selector: 'edge[type="ACTIVATES"]', style: { "line-color": anchor, width: "mapData(weight, 0, 1, 0.4, 2)", opacity: 0.22 } },
    { selector: 'edge[type="ACTIVATES"][?weak]', style: { "line-style": "dashed" } },
    { selector: 'edge[type="RELATED_TO"]', style: { "curve-style": "unbundled-bezier", "control-point-distances": "data(cpd)", "control-point-weights": 0.5, "line-color": anchor, width: "mapData(weight, 0, 1, 1, 6)", opacity: 0.75, label: "data(weight)", "font-size": 8.5, color: anchor, "text-background-color": surface, "text-background-opacity": 0.9, "text-background-padding": 2 } },
    { selector: 'edge[type="CO_OCCURS"]', style: { "line-style": "dashed", "line-color": anchor, width: "mapData(weight, 0, 1, 1, 4)", opacity: 0.6 } },
    { selector: 'edge[type="HAS_SCOPE"], edge[type="TARGETS"], edge[type="DISCOVERED_IN"], edge[type="OF_DATASET"]', style: { "line-color": edge, width: 0.6, opacity: 0.5 } },
    { selector: ".hidden", style: { display: "none" } },
    { selector: ".faded", style: { opacity: 0.09 } },
    { selector: "edge.faded", style: { opacity: 0.03 } },
    { selector: ".hl-node", style: { opacity: 1 } },
    { selector: ".hl-evidence", style: { "border-width": 3, "border-color": ink } },
    { selector: ".hl-cross", style: { "border-width": 4, "border-color": bad, "border-style": "double" } },
    { selector: ".hl-seed", style: { "border-width": 5, "border-color": path } },
    { selector: ".hl-anchor", style: { "border-width": 5, "border-color": path } },
    { selector: "edge.hl-edge", style: { "line-color": path, "target-arrow-color": path, width: 5, opacity: 1, "z-index": 999, display: "element" } },
    { selector: "node:selected", style: { "overlay-color": anchor, "overlay-opacity": 0.14, "overlay-padding": 7 } },
    { selector: ".lg-dim", style: { opacity: 0.06 } },
  ];
}

function attractorOrder(cy) {
  /* greedy chain: strongly linked themes (RELATED_TO or CO_OCCURS) end up adjacent on the latent row */
  const atts = cy.nodes('[kind="Attractor"]').toArray();
  if (!atts.length) return [];
  const w = (a, b) => { let m = 0; a.edgesWith(b).filter(MARKS.latent).forEach((e) => { m = Math.max(m, e.data("weight")); }); return m; };
  atts.sort((a, b) => b.data("n_patterns") - a.data("n_patterns"));
  const order = [atts.shift()];
  while (atts.length) {
    const end = order[order.length - 1];
    let bi = 0, bw = -1;
    atts.forEach((a, i) => { const x = w(end, a); if (x > bw) { bw = x; bi = i; } });
    order.push(atts.splice(bi, 1)[0]);
  }
  return order;
}

function dualPlanePositions(cy) {
  const pos = {};
  const attractors = attractorOrder(cy);
  const groups = new Map(attractors.map((a) => [a.id(), []]));
  const loose = [];
  cy.nodes('[kind="Pattern"]').forEach((p) => { const a = p.data("anchor") /* its strongest ACTIVATES edge */; (a && groups.has(a) ? groups.get(a) : loose).push(p); });
  if (loose.length) groups.set("__loose", loose);
  const colW = 116, rowH = 104, top = 250;
  let x = 0, maxY = top;
  for (const [aid, pats] of groups) {
    pats.sort((a, b) => b.data("weight") - a.data("weight"));
    const cols = Math.max(1, Math.ceil(Math.sqrt(pats.length) / 1.6));
    const width = Math.max(cols * colW, 170);
    const cx = x + width / 2;
    if (aid !== "__loose") pos[aid] = { x: cx, y: 0 };
    pats.forEach((p, i) => {
      const r = Math.floor(i / cols), c = i % cols;
      pos[p.id()] = { x: cx + (c - (Math.min(cols, pats.length) - 1) / 2) * colW, y: top + r * rowH };
      maxY = Math.max(maxY, top + r * rowH);
    });
    x += width + 50;
  }
  let y = maxY + 150;
  const regions = layerState().region ? cy.nodes('[kind="Region"]') : cy.collection();  // rows only while drawn
  for (const res of [...new Set(regions.map((r) => r.data("resolution")))].sort((a, b) => b - a)) {
    const row = regions.filter((r) => r.data("resolution") === res).map((r) => {
      const below = r.incomers('edge[type="IN_REGION"], edge[type="PARENT_OF"]').sources().filter((n) => pos[n.id()]);
      return [r, below.length ? below.reduce((a, n) => a + pos[n.id()].x, 0) / below.length : 0];
    }).sort((a, b) => a[1] - b[1]);
    let last = -Infinity;
    for (const [r, at] of row) { last = Math.max(at, last + 70); pos[r.id()] = { x: last, y }; }
    y += 90;
  }
  const bottom = cy.nodes('[kind="Dimension"], [kind="Metric"]');
  const span = Math.max(x - 50, 400);
  bottom.forEach((n, i) => { pos[n.id()] = { x: (i + 0.5) * span / Math.max(bottom.length, 1), y: regions.length ? y + 60 : maxY + 170 }; });
  pos.__lbl_latent = { x: -150, y: 0 };
  pos.__lbl_struct = { x: -150, y: top };
  return pos;
}

function arcLatentEdges(cy, dual) {
  cy.edges('[type="RELATED_TO"]').forEach((e) => {
    const dx = Math.abs(e.source().position("x") - e.target().position("x"));
    e.data("cpd", dual ? -Math.min(0.22 * dx, 160) : 0);
  });
}

function runLayout() {
  const cy = S.cy; if (!cy || !cy.nodes().length) return;
  if ($("layout").value === "dual") {
    const pos = dualPlanePositions(cy);
    cy.nodes().forEach((n) => { if (pos[n.id()]) n.position(pos[n.id()]); });
    arcLatentEdges(cy, true);
    cy.fit(cy.elements(":visible"), 40);
  } else {
    cy.layout({ name: "cose", animate: false, idealEdgeLength: 90, nodeRepulsion: 9000, randomize: true, fit: true, padding: 40,
      eles: cy.elements(":visible").not('[kind="PlaneLabel"]') }).run();
    arcLatentEdges(cy, false);
  }
}

function layerState() { return Object.fromEntries([...document.querySelectorAll(".toggle[data-key]")].map((b) => [b.dataset.key, b.classList.contains("on")])); }

function applyVisibility() {
  const cy = S.cy;
  if (cy) {
    for (const [key, on] of Object.entries(layerState())) cy.$(LEGEND[key]).toggleClass("hidden", !on);
    cy.$('node[kind="PlaneLabel"]').toggleClass("hidden", $("layout").value !== "dual");
  }
  applySphereLayers();
}

// the latent sphere (docs/08_interface.md §8.4): /api/sphere gives the projected points and the edges between them;
// the traces are drawn here in the graph's visual language and restyled in place (layers, spotlight), never re-fetched
// for a highlight or a theme change. Every trace a legend entry explains carries the entry's key as `meta`.
const HOVER = "%{hovertext}<extra></extra>";
const SPHERE_LINKS = [  // legend key, edge type, name, palette colour, width, dash
  ["latent", "RELATED_TO", "Theme links", "anchor", 4, null],
  ["latent", "CO_OCCURS", "Co-occurring themes", "anchor", 2.5, "dash"],
  ["lattice", "SPECIALIZES", "Hierarchy", "muted", 2, null],
  ["contrast", "CONTRASTS", "Contrasts", "bad", 2, "dash"],
  ["sibling", "SIBLING", "Siblings", "muted", 1.5, "dot"],
  ["colocated", "CO_LOCATED", "Same place", "path", 1.5, "dash"],
];
const ANSWER_MARKS = [  // highlight field, legend key, name, palette colour, size, ring width
  ["evidence", "ev", "Evidence", "ink", 8, 2], ["transversal_only", "cross", "Other segment", "bad", 10, 3],
  ["seeds", "seed", "Seed", "path", 9, 4], ["anchors", "path", "Themes visited", "path", 16, 4],
];

function rgba(hex, alpha) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

function sphereTraces(d, pal, on, hl) {
  const at = Object.fromEntries(d.points.map((p) => [p.id, p]));
  const xyz = (pts) => ({ x: pts.map((p) => p.xyz[0]), y: pts.map((p) => p.xyz[1]), z: pts.map((p) => p.xyz[2]) });
  const focus = !!hl && ["traversed", "evidence", "seeds"].some((k) => (hl[k] || []).length);
  const traces = [];
  const lines = (name, key, edges, color, width, dash) => {
    if (!edges.length) return;
    const x = [], y = [], z = [];
    for (const e of edges) { const a = at[e.source].xyz, b = at[e.target].xyz; x.push(a[0], b[0], null); y.push(a[1], b[1], null); z.push(a[2], b[2], null); }
    traces.push({ type: "scatter3d", mode: "lines", name, meta: key, visible: key in on ? on[key] : true, x, y, z,
      line: { color, width, ...(dash ? { dash } : {}) }, hoverinfo: "skip" });
  };
  const themes = d.points.filter((p) => p.kind === "Attractor");
  traces.push({ type: "scatter3d", mode: "markers+text", name: "Themes", meta: "anchor", ...xyz(themes), text: themes.map((a) => a.id),
    customdata: themes.map((a) => a.id), textposition: "top center", textfont: { color: pal.ink, size: 11 },
    marker: { symbol: "diamond", size: themes.map((a) => 8 + 1.2 * a.n_patterns), color: pal.anchor, line: { color: pal.ink, width: 1 } },
    hovertext: themes.map((a) => a.hover), hovertemplate: HOVER });
  for (const v of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])  // the six half-axes
    traces.push({ type: "scatter3d", mode: "lines", x: [0, v[0]], y: [0, v[1]], z: [0, v[2]], line: { color: rgba(pal.muted, 0.35), width: 0.5 }, showlegend: false, hoverinfo: "none" });
  const acts = d.edges.filter((e) => e.type === "ACTIVATES");  // memberships under the nodes; weak = dashed
  for (const weak of [false, true])
    lines(weak ? "Memberships (weak)" : "Memberships", "activates", acts.filter((e) => e.weak === weak), rgba(pal.anchor, focus ? 0.08 : 0.3), 1.5, weak ? "dash" : null);
  for (const [key, name, symbol] of [["up", "Metric higher", "circle"], ["down", "Metric lower", "circle"], ["cov", "Correlation change", "square"]]) {
    const pts = d.points.filter((p) => p.kind === "Pattern" && p.cls === key);
    if (pts.length) traces.push({ type: "scatter3d", mode: "markers", name, meta: key, opacity: focus ? 0.35 : 0.9, ...xyz(pts),
      marker: { symbol, size: pts.map((p) => 3 + 6 * p.weight), color: pal[key], line: { width: 0 } },
      hovertext: pts.map((p) => p.hover), customdata: pts.map((p) => p.id), hovertemplate: HOVER });
  }
  const circle = (f) => [...Array(97).keys()].map((i) => f((2 * Math.PI * i) / 96));
  const gx = [], gy = [], gz = [];  // faint great circles for depth
  for (const [fx, fy, fz] of [[Math.cos, Math.sin, () => 0], [Math.cos, () => 0, Math.sin], [() => 0, Math.cos, Math.sin]]) {
    gx.push(...circle(fx), null); gy.push(...circle(fy), null); gz.push(...circle(fz), null);
  }
  traces.push({ type: "scatter3d", mode: "lines", x: gx, y: gy, z: gz, showlegend: false, hoverinfo: "skip", line: { color: rgba(pal.muted, 0.2), width: 1 } });
  for (const [key, type, name, color, width, dash] of SPHERE_LINKS) lines(name, key, d.edges.filter((e) => e.type === type), pal[color], width, dash);
  if (hl) {  // the answer, with the graph's markers: gold seed and path, ink evidence ring, red cross-segment ring
    const byId = Object.fromEntries(d.edges.map((e) => [e.id, e]));
    lines("Answer path", "path", (hl.edges || []).filter((id) => byId[id]).map((id) => byId[id]), pal.path, 7);
    for (const [field, key, name, color, size, width] of ANSWER_MARKS) {
      const pts = (hl[field] || []).filter((id) => at[id]).map((id) => at[id]);
      if (pts.length) traces.push({ type: "scatter3d", mode: "markers", name, meta: key, ...xyz(pts),
        marker: { symbol: "circle-open", size, color: pal[color], line: { color: pal[color], width } },
        hovertext: pts.map((p) => p.hover), customdata: pts.map((p) => p.id), hovertemplate: HOVER });
    }
  }
  return traces;
}

function sphereLayout(pal) {
  const hidden = { visible: false, showgrid: false, zeroline: false, range: [-1, 1] };
  return { showlegend: false, paper_bgcolor: pal.bg, plot_bgcolor: pal.bg, font: { color: pal.ink }, hoverlabel: { align: "left" }, margin: { l: 0, r: 0, b: 0, t: 0 },
    scene: { xaxis: hidden, yaxis: hidden, zaxis: hidden, aspectmode: "cube", bgcolor: pal.bg, camera: { eye: { x: 1.1, y: 1.1, z: 1.1 } } },
    uirevision: "sig-sphere" };  // the camera survives a re-render
}

function sphereDiv() { const el = $("sphere"); return el && el.data ? el : null; }

function applySphereLayers() {
  const gd = sphereDiv(); if (!gd) return;
  const on = layerState(), idx = gd.data.map((t, i) => (t.meta in on ? i : -1)).filter((i) => i >= 0);
  if (idx.length) Plotly.restyle(gd, { visible: idx.map((i) => on[gd.data[i].meta]) }, idx);
}

function spotlight(key) {
  /* hovering a legend entry: everything else dims, in the graph and on the sphere */
  if (S.cy) {
    S.cy.elements().removeClass("lg-dim");
    const sel = key ? S.cy.$(LEGEND[key]) : null;
    if (sel && sel.nonempty()) S.cy.elements().not(sel.union(sel.filter("edge").connectedNodes())).addClass("lg-dim");
  }
  const gd = sphereDiv(); if (!gd) return;
  gd._sigOpacity = gd._sigOpacity || gd.data.map((t) => (t.opacity === undefined ? 1 : t.opacity));
  const hit = key && gd.data.some((t) => t.meta === key);
  Plotly.restyle(gd, { opacity: gd.data.map((t, i) => (!hit || t.meta === key ? gd._sigOpacity[i] : 0.07)) });
}

function setCounts(counts) {
  for (const [k, v] of Object.entries(counts)) document.querySelectorAll(`[data-n="${k}"]`).forEach((el) => { el.textContent = v ? v : ""; });
}

function spherePalette() {
  const v = (n) => cssVar(n);
  return { bg: v("--sunken"), ink: v("--graph-ink"), muted: v("--muted"), anchor: v("--anchor"), up: v("--up"), down: v("--down"), cov: v("--cov"), path: v("--path"), bad: v("--bad") };
}

async function loadGraph() {
  const ds = $("dataset-filter").value;
  let g;
  S.sphere = null;  // the graph changed (or is unknown): project again
  try { g = await api("/api/graph" + (ds ? `?dataset=${encodeURIComponent(ds)}` : "")); } catch (e) { return; }
  S.datasets = g.datasets || {};
  S.nodes = g.nodes.map((n) => n.data).filter((d) => d.kind === "Pattern");
  $("ins-count").textContent = S.nodes.length ? S.nodes.length : "";
  const themes = g.nodes.filter((n) => n.data.kind === "Attractor").length;
  $("view-caption").textContent = S.nodes.length ? `${S.nodes.length} insights · ${themes} themes in view` : "No insights in view";
  $("empty-hint").hidden = g.nodes.length > 0 || S.view === "table";
  const els = [...g.nodes, ...g.edges];
  if (g.nodes.length) {
    els.push({ data: { id: "__lbl_latent", kind: "PlaneLabel", label: "THEMES" }, selectable: false, grabbable: false });
    els.push({ data: { id: "__lbl_struct", kind: "PlaneLabel", label: "INSIGHTS" }, selectable: false, grabbable: false });
  }
  if (!S.cy) {
    S.cy = cytoscape({ container: $("cy"), elements: els, style: graphStyle(), wheelSensitivity: 0.25, minZoom: 0.12, maxZoom: 3 });
    S.cy.on("tap", "node", (evt) => { if (evt.target.data("kind") !== "PlaneLabel") inspect(evt.target.id()); });
    S.cy.on("tap", (evt) => { if (evt.target === S.cy) closeDrawer(); });
  } else {
    S.cy.elements().remove(); S.cy.add(els);
  }
  // legend counts: what each entry marks (nodes for an entry with nodes, else its edges)
  setCounts(Object.fromEntries(Object.entries(MARKS).map(([k, sel]) => { const m = S.cy.$(sel); return [k, m.nodes().length || m.length]; })));
  applyVisibility(); runLayout();
  if (S.hl) highlight(S.hl);
  renderTable();
  renderSphere();
}

// views
function setView(view) {
  S.view = view;
  document.querySelectorAll("#view-switch button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("cy").hidden = view !== "graph";
  $("sphere").hidden = view !== "sphere";
  $("table-view").hidden = view !== "table";
  $("map-view").hidden = view !== "map";
  document.querySelector(".canvas-body").classList.toggle("map-mode", view === "map");
  document.querySelectorAll(".graph-only").forEach((el) => { el.hidden = view !== "graph"; });
  $("fit-btn").hidden = view === "table" || view === "map";
  for (const key of NO_VECTORS) {  // column and region nodes have no vectors: graph only
    const t = document.querySelector(`.toggle[data-key="${key}"]`);
    t.dataset.title = t.dataset.title || t.title;
    t.disabled = view === "sphere"; t.title = view === "sphere" ? "Not drawn in 3D (no vectors)" : t.dataset.title;
  }
  $("empty-hint").hidden = view === "table" || view === "map" || S.nodes.length > 0;
  if (view === "sphere") renderSphere();
  if (view === "map") GeoMap.show();
  if (view === "graph" && S.cy) { S.cy.resize(); S.cy.fit(S.cy.elements(":visible"), 40); }
}

function sphereMessage(text) { const el = $("sphere"); Plotly.purge(el); el.textContent = text; }

async function renderSphere() {
  if (S.view !== "sphere") return;
  const ds = $("dataset-filter").value, seq = ++S.sphereSeq, el = $("sphere");
  if (!S.sphere || S.sphere.ds !== ds) {  // one projection per graph and dataset, shared by every render that waits for it
    if (!sphereDiv()) sphereMessage("Projecting insight vectors on the sphere…");
    S.sphere = { ds, data: api("/api/sphere" + (ds ? `?dataset=${encodeURIComponent(ds)}` : "")) };
  }
  const pending = S.sphere.data;
  let d;
  try { d = await pending; } catch (e) {
    if (S.sphere && S.sphere.data === pending) S.sphere = null;  // the next render asks again
    if (seq === S.sphereSeq) sphereMessage(e.message);
    return;
  }
  if (seq !== S.sphereSeq) return;  // a newer render started
  const pal = spherePalette();
  if (d.message) { sphereMessage(d.message); return; }
  if (!sphereDiv()) el.textContent = "";
  await Plotly.react(el, sphereTraces(d, pal, layerState(), S.hl), sphereLayout(pal), { displaylogo: false, responsive: true });
  el._sigOpacity = el.data.map((t) => (t.opacity === undefined ? 1 : t.opacity));
  el.removeAllListeners("plotly_click");  // clicks open the drawer (customdata = node id)
  el.on("plotly_click", (ev) => { const id = ev.points && ev.points[0] && ev.points[0].customdata; if (id) inspect(id); });
}

function renderTable() {
  const q = $("table-search").value.trim().toLowerCase();
  const { k, asc } = S.sort;
  const rows = S.nodes.filter((d) => !q || [d.full_label, d.target, d.anchor_label, ...(d.scope || [])].join(" ").toLowerCase().includes(q));
  rows.sort((a, b) => {
    const va = k === "scope" ? (a.scope || []).join() : a[k], vb = k === "scope" ? (b.scope || []).join() : b[k];
    const c = typeof va === "number" && typeof vb === "number" ? va - vb : String(va ?? "").localeCompare(String(vb ?? ""));
    return asc ? c : -c;
  });
  document.querySelectorAll("#ins-table th").forEach((th) => { th.classList.toggle("sorted", th.dataset.k === k); th.classList.toggle("asc", th.dataset.k === k && asc); });
  const maxW = Math.max(...S.nodes.map((d) => d.weight || 0), 0.01);
  $("ins-table").querySelector("tbody").innerHTML = rows.map((d) => {
    const cls = d.ptype === "covariance" ? "cov" : d.direction > 0 ? "up" : "down";
    const effect = d.ptype === "covariance" ? "corr. change" : `${signed(d.effect)} sd`;
    return `<tr data-id="${esc(d.id)}"><td><div class="scope-tags">${(d.scope || []).map((c) => `<span class="tag">${esc(human(c).replace("=", " = "))}</span>`).join("")}</div></td>` +
      `<td>${esc(human(d.target))}</td><td class="num"><span class="effect ${cls}">${effect}</span></td><td class="num">${num(d.support, 0)}</td>` +
      `<td class="num"><span class="bar" style="width:${Math.round((d.weight / maxW) * 40)}px"></span>${num(d.weight)}</td>` +
      `<td>${d.anchor_label ? `<span class="tag anchor">${esc(d.anchor_label)}</span>` : ""}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted" style="text-align:center;padding:24px">${S.nodes.length ? "No insight matches the filter." : "No insights yet."}</td></tr>`;
  $("table-summary").textContent = S.nodes.length ? `${rows.length} of ${S.nodes.length}` : "";
  $("ins-table").querySelectorAll("tbody tr[data-id]").forEach((tr) => tr.addEventListener("click", () => inspect(tr.dataset.id)));
}

// highlighting
function clearHighlight() {
  if (S.cy) S.cy.elements().removeClass("faded hl-node hl-seed hl-anchor hl-evidence hl-cross hl-edge");
  document.querySelectorAll(".ev.on").forEach((el) => el.classList.remove("on"));
  setCounts({ seed: 0, ev: 0, cross: 0, path: 0 });
}

function highlight(h) {
  /* draw S.hl — the whole answer or one isolated evidence path — over the graph */
  const cy = S.cy; if (!cy || !h) return;
  clearHighlight();
  cy.elements().not('[kind="PlaneLabel"]').addClass("faded");
  const mark = (ids, cls) => (ids || []).forEach((id) => { const el = cy.getElementById(id); if (el.nonempty()) el.removeClass("faded").addClass("hl-node " + cls); });
  mark(h.traversed, "");
  mark(h.evidence, "hl-evidence");
  mark(h.transversal_only, "hl-cross");
  mark(h.anchors, "hl-anchor");
  mark(h.seeds, "hl-seed");
  (h.edges || []).forEach((id) => {
    const e = cy.getElementById(id);
    if (e.nonempty()) { e.removeClass("faded hidden").addClass("hl-edge"); e.connectedNodes().removeClass("faded").addClass("hl-node"); }
  });
  setCounts({ seed: (h.seeds || []).length, ev: (h.evidence || []).length, cross: (h.transversal_only || []).length, path: (h.edges || []).length });
  $("clear-btn").hidden = false;
}

function highlightPath(turn, item, el) {
  const h = turn.qa.view.highlight;
  const edges = item.path.map((st) => st.edge_id);
  S.hl = { seeds: h.seeds, anchors: item.path.filter((s) => s.target.startsWith("A-")).map((s) => s.target), evidence: [item.pattern_id],
    transversal_only: item.transversal_only ? [item.pattern_id] : [], edges, traversed: [] };
  if (S.view === "table") setView("graph");
  highlight(S.hl);
  if (S.view === "sphere") renderSphere(); else focusNodes([item.pattern_id, ...item.path.map((s) => s.source)]);
  document.querySelectorAll(".ev.on").forEach((x) => x.classList.remove("on"));
  if (el) el.classList.add("on");
}

function focusNodes(ids) {
  const cy = S.cy; if (!cy) return;
  const els = cy.collection(ids.map((id) => cy.getElementById(id)).filter((e) => e.nonempty()));
  if (els.nonempty()) cy.animate({ fit: { eles: els, padding: 90 } }, { duration: 350 });
}

// details drawer
function closeDrawer() { $("drawer").hidden = true; if (S.cy) S.cy.$(":selected").unselect(); }

function kv(rows) {
  return `<dl class="kv">${rows.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

function shiftRows(shifts) {
  const max = Math.max(...shifts.map((s) => Math.abs(s.robust_z)), 1);
  return shifts.map((s) => {
    const dir = s.robust_z >= 0 ? "up" : "down";
    return `<div class="shift-row"><span>${esc(human(s.metric))} <span class="muted">${num(s.local_median, 4)} vs ${num(s.global_median, 4)}</span></span>` +
      `<b class="effect ${dir}">${signed(s.robust_z)} sd</b>` +
      `<div class="meter"><i class="${dir}" style="width:${Math.min(100, Math.round((Math.abs(s.robust_z) / max) * 100))}%"></i></div></div>`;
  }).join("");
}

const VERIFIED = {  // POST /api/patterns/{id}/verify: status -> chip class, words
  reproduced: ["ok", "Reproduced", "every recomputed number matches the journal"],
  differs: ["bad", "Differs", "the journal does not match the stored data"],
  source_changed: ["bad", "Source changed", "the stored upload no longer gives this dataset"],
  source_missing: ["warn", "Source missing", "the upload is no longer stored"],
};

async function verifyPattern(id) {
  const btn = $("verify-btn"), out = $("verify-out");  // this drawer's: another may open while the answer is on its way
  btn.disabled = true; out.textContent = "Recomputing…";
  let html;
  try {
    const r = await api(`/api/patterns/${encodeURIComponent(id)}/verify`, { method: "POST" });
    const [cls, label, words] = VERIFIED[r.status] || ["warn", r.status, ""];
    const rows = (r.diff || []).map((d) => `<div class="shift-row"><span class="mono">${esc(d.field)}</span><span class="muted">${esc(String(d.journal))} → ${esc(String(d.recomputed))}</span></div>`).join("");
    html = `<span class="chip ${cls}">${esc(label)}</span> ${esc(words)}${rows}<div class="muted">Not recomputed: ${esc(r.not_checked.join(", "))}</div>`;
  } catch (e) { html = esc(`Re-check failed: ${e.message}`); }
  if (out.isConnected) { out.innerHTML = html; btn.disabled = false; }
}

const chip = (cls, text, title = "") => `<span class="chip ${cls}"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
const nodeLabel = (id, labels = {}) => labels[id] || (S.nodes.find((d) => d.id === id) || {}).full_label || id;
const nbRow = (id, text, note = "") => `<div class="nb" data-id="${esc(id)}"><span>${esc(text)}</span><span class="muted">${esc(note)}</span></div>`;

function trustRows(p, ds, labels) {
  /* an insight's three trust axes: its source's reliability, its credibility from other sources, its integrity */
  const [ccls, cwords] = cred(p.credibility);
  const basis = p.credibility_basis || {}, [icls, iwords] = INTEGRITY[p.integrity] || INTEGRITY.unverified;
  const source = ds.integrity === "mismatch" ? `${chip("bad", "unknown")} its dataset's metadata failed its integrity check`
    : `${chip("", ds.reliability || "F")} ${esc(ds.source || "source not declared")} — ${esc(RELIABILITY[ds.reliability || "F"])}${ds.integrity === "unverified" ? " (metadata unverified)" : ""}`;
  const by = basis.rule === "place" ? "compared by place" : basis.rule === "anchor" ? "compared by theme and metric" : "";
  const others = [...(basis.confirmed_by || []), ...(basis.patterns || []).filter((x) => !(basis.confirmed_by || []).includes(x))];
  return kv([
    ["Source", source],
    ["Credibility", `${chip(ccls, String(p.credibility ?? "–"))} ${esc(cwords)}${(basis.sources || []).length ? ` · ${esc(basis.sources.join(", "))}` : ""}${by ? ` <span class="muted">(${by})</span>` : ""}`],
    ["Integrity", `${chip(icls, p.integrity || "unverified")} ${esc(iwords)}`],
  ]) + (others.length ? `<details class="nb-group"><summary>Basis <span class="muted">(${others.length})</span></summary>${others.map((x) => nbRow(x, nodeLabel(x, labels), (basis.confirmed_by || []).includes(x) ? "confirms" : "")).join("")}</details>` : "");
}

function scaleRows(c, labels) {
  /* the scale chain the graph already holds: ↑ coarser and ↓ finer levels of one phenomenon, then the regions above it */
  const step = (st, arrow) => nbRow(st.pattern_id, `${arrow} ${nodeLabel(st.pattern_id, labels)}`, st.relation === "corresponds" ? "same shift" : "opposite shift");
  return `<h4>Scale</h4><p class="lead">${esc(SCALE[c.scale] || "a single level")}</p>` +
    c.up.map((st) => step(st, "↑")).join("") + c.down.map((st) => step(st, "↓")).join("") +
    c.regions.map((r) => nbRow(r.id, `▣ ${r.id}`, picture(r))).join("");
}

async function inspect(id) {
  let d;
  try { d = await api(`/api/nodes/${encodeURIComponent(id)}`); } catch (e) { return; }
  const n = d.node, p = n.props;
  const kinds = { Pattern: "Insight", Attractor: "Theme · latent anchor", Dimension: "Grouping column", Metric: "Metric", Dataset: "Dataset", Batch: "Batch", Region: "Map region" };
  $("drawer-kind").textContent = kinds[n.kind] || n.kind;
  let html = "";
  if (n.kind === "Pattern") {
    const scope = p.conditions.map((c) => `${human(c.attribute)} = ${c.value}`).join(" and ");
    $("drawer-title").textContent = scope;
    const cov = p.covariance && p.covariance.pair ? p.covariance : null;
    const lead = p.phenomenon_type === "covariance" && cov
      ? `The relationship between <b>${esc(human(cov.pair[0]))}</b> and <b>${esc(human(cov.pair[1]))}</b> changes here: correlation ${num(cov.global_corr)} overall → <b>${num(cov.local_corr)}</b> in this subgroup.`
      : `<b>${esc(human(p.target))}</b> is ${p.effect_size > 0 ? "higher" : "lower"} here — median ${num(p.local, 4)} vs ${num(p.baseline, 4)} overall (${signed(p.effect_size)} sd).`;
    html += `<p class="lead">${lead}</p>`;
    html += p.phenomenon_type === "covariance" ? `<h4>Median differences (not validated)</h4>${shiftRows(p.shifts)}` : `<h4>Shifts</h4>${shiftRows(p.shifts)}`;
    html += `<h4>Evidence</h4>` + kv([
      ["Rows", `${num(p.support, 0)} (${num(p.support_fraction * 100, 1)}% of data)`],
      ["Evidence weight", `<b>${num(p.weight)}</b>`],
      ["Adjusted p-value", p.p_adjusted === null || p.p_adjusted === undefined ? "not tested (correlation change)" : Number(p.p_adjusted) > 0 ? Number(p.p_adjusted).toExponential(1) : "< 1e-300 (underflow)"],
      ["Bootstrap stability", p.stability === null || p.stability === undefined ? "not measured (correlation change)" : num(p.stability)],
      ["Correlation change", cov ? `${esc(human(cov.pair.join(" ~ ")))}: ${num(cov.global_corr)} → ${num(cov.local_corr)}` : ""],
      ["Confounders", esc((p.drivers || []).join("; ") || "none detected")],
      ["Also known as", esc((p.aliases || []).join("; "))],
    ]);
    html += `<h4>Trust</h4>${trustRows(p, d.dataset || {}, d.labels)}`;
    if (d.scale_chain) html += scaleRows(d.scale_chain, d.labels);
    html += `<div class="verify"><button type="button" class="btn ghost sm" id="verify-btn" title="Recompute this insight's statistics from the stored upload and compare them with the journal">Re-check from the raw data</button><div class="msg" id="verify-out"></div></div>`;
    html += `<details class="nb-group"><summary>Score breakdown</summary>` + kv(Object.entries(p.weight_factors || {}).map(([k, v]) => [k, num(v)]).concat([["SD score", num(p.sd_score)], ["EMM score", num(p.emm_score, 3)], ["Volume utility", num(p.volume_utility, 3)]])) + `</details>`;
    html += `<details class="nb-group"><summary>Canonical representation</summary><pre class="canon">${esc(p.canonical.document)}</pre></details>`;
    html += `<details class="nb-group"><summary>Provenance</summary>` + kv([
      ["Dataset", esc(`${p.provenance.filename}`)], ["Dataset id", `<span class="mono">${esc(p.provenance.dataset_id)}</span>`], ["Batch", `<span class="mono">${esc(p.provenance.batch_id)}</span>`],
      ["Selector", `<span class="mono">${esc(p.expression)}</span>`], ["Engine", esc(p.provenance.engine)], ["Rows ref", `<span class="mono">${esc(p.provenance.rows_ref)}</span>`],
      ["Embedding", esc(`${p.embedding.model_id} · ${p.embedding.dim}d · ${p.embedding.representation_version}`)],
    ]) + `</details>`;
  } else if (n.kind === "Attractor") {
    $("drawer-title").textContent = n.label;
    html += `<p class="lead">A recurring pattern learned from <b>${num(p.n_patterns, 0)}</b> insights across <b>${num(p.distinct_scopes, 0)}</b> different subgroups.</p>`;
    html += `<h4>Signature</h4>` + p.signature.map((s) => `<div class="shift-row"><span>${esc(s.component)}</span><b class="effect ${s.value >= 0 ? "up" : "down"}">${signed(s.value)}</b></div>`).join("");
    html += `<h4>Details</h4>` + kv([
      ["Columns involved", esc(p.dimensions.map(human).join(", "))], ["Metrics", esc(p.targets.map(human).join(", "))],
      ["Mass", num(p.mass, 0)], ["Evidence mass", num(p.evidence_mass)], ["Dispersion", num(p.dispersion, 3)],
      ["Last updated", `<span class="mono">${esc(p.last_updated_batch)}</span>`],
    ]);
  } else if (n.kind === "Region") {
    const pic = d.picture || {};
    $("drawer-title").textContent = `res ${p.resolution} · ${p.h3}`;
    html += `<p class="lead">An H3 cell at resolution ${num(p.resolution, 0)} that holds ${esc(picture(pic))}.</p>`;
    html += kv([["Cells covered", num(p.cells_covered, 0)], ["Region id", `<span class="mono">${esc(n.id)}</span>`]]);
  } else {
    $("drawer-title").textContent = n.label;
    html += kv(Object.entries(p).map(([k, v]) => [human(k), esc(typeof v === "object" ? JSON.stringify(v) : v)]));
  }
  const groups = Object.entries(d.neighbors);
  if (groups.length) {
    const friendly = { "ACTIVATES": "Belongs to theme", "ACTIVATES (in)": "Member insights", "RELATED_TO": "Related themes", "RELATED_TO (in)": "Related themes", "CO_OCCURS": "Co-occurring themes", "CO_OCCURS (in)": "Co-occurring themes",
      "SPECIALIZES": "Narrower version of", "SPECIALIZES (in)": "Narrower insights", "IN_REGION": "Map regions", "IN_REGION (in)": "Insights in this region", "PARENT_OF": "Parent region", "PARENT_OF (in)": "Sub-regions",
      "CONTRASTS": "Contrasts with", "CONTRASTS (in)": "Contrasts with", "CO_LOCATED": "Same place", "CO_LOCATED (in)": "Same place", "SIBLING": "Siblings", "SIBLING (in)": "Siblings", "HAS_SCOPE": "Columns", "TARGETS": "Metrics" };
    html += `<h4>Connections</h4>` + groups.filter(([t]) => !["DISCOVERED_IN", "OF_DATASET", "HAS_SCOPE (in)", "TARGETS (in)"].includes(t)).map(([type, items]) =>
      `<details class="nb-group" ${/ACTIVATES|RELATED|CO_OCCURS|CONTRAST/.test(type) ? "open" : ""}><summary>${esc(friendly[type] || type)} <span class="muted">(${items.length})</span></summary>` +
      items.map((it) => nbRow(it.id, it.label, (it.props.relation ? `${RELATION[it.props.relation] || human(it.props.relation)} · ` : "") + num(Number(it.weight)))).join("") + `</details>`).join("");
  }
  $("inspector").innerHTML = html;
  $("inspector").querySelectorAll(".nb").forEach((el) => el.addEventListener("click", () => { focusNodes([el.dataset.id]); inspect(el.dataset.id); }));
  $("verify-btn")?.addEventListener("click", () => verifyPattern(n.id));
  $("drawer").hidden = false;
  if (S.cy) { S.cy.$(":selected").unselect(); S.cy.getElementById(id).select(); }
}
