/* SIG UI — datasets, graph / sphere / table views, details drawer, AI chat with grounded citations (docs/08_interface.md). */
"use strict";

const S = { cy: null, qa: null, poll: null, lastReady: null, view: "graph", sphereSeq: 0, nodes: [], sort: { k: "weight", asc: false }, chat: [], busy: false };
const STAGES = ["UPLOADED", "VALIDATING", "PROFILING", "DISCOVERING", "VALIDATING_INSIGHTS", "EMBEDDING", "UPDATING_ONTOLOGY", "BUILDING_GRAPH", "PERSISTING", "READY"];
const STAGE_TEXT = {
  UPLOADED: "Queued", VALIDATING: "Reading the file", PROFILING: "Profiling columns", DISCOVERING: "Searching subgroups",
  VALIDATING_INSIGHTS: "Validating insights", EMBEDDING: "Embedding insights", UPDATING_ONTOLOGY: "Learning themes",
  BUILDING_GRAPH: "Building the graph", PERSISTING: "Saving", READY: "Ready",
};
const TERMINAL = new Set(["READY", "FAILED", "SKIPPED"]);
const ERROR_HELP = {
  no_candidates: ["No subgroups to analyse", "There are no usable columns to group rows by (text columns with a handful of values).", "Mark number-coded columns such as IDs or flags under “Treat as categories”, or split a numeric column into bands."],
  invalid_schema: ["The table can’t be analysed", "It needs at least one grouping column, one numeric column and enough rows.", "Check the file, or mark code-like columns as categories."],
  no_numeric_targets: ["No numeric columns", "SIG measures shifts in numeric columns and found none it can use.", "Make sure numbers aren’t stored as text."],
  no_viable_insights: ["Nothing significant found", "Subgroups were tested, but none passed the significance, stability and effect-size checks.", "Try more rows or other grouping columns."],
  unsupported_file: ["Unsupported file", "Use CSV, TSV or Parquet (up to the upload limit).", ""],
  unreadable_file: ["Couldn’t read the file", "The parser failed — check the delimiter and encoding.", ""],
  invalid_options: ["Column option problem", "A column named in the options doesn’t exist or isn’t numeric.", "Check the spelling of the column names."],
  representation_mismatch: ["Workspace mismatch", "This workspace was built with a different embedding setup.", "Reset the workspace to change the embedding model or weights."],
  duplicate_patterns: ["Already ingested", "These insights are already in the graph.", ""],
  interrupted: ["Interrupted", "Processing stopped before finishing; nothing partial was kept.", "Upload the file again."],
};
// chat content is Ukrainian; data literals (column names, values) stay exactly as the data holds them
const SUGGESTIONS = [
  "Чому margin нижчий для phones у US?",
  "Що пов'язано з вищим return_rate?",
  "Розкажи про margin для laptops у EU",
  "Де руйнується кореляція між discount і margin?",
];
// sphere traces named after the legend's layer toggles (ltir/sphere.py LAYER_TRACES)
const SPHERE_LAYERS = { lattice: ["Hierarchy"], contrast: ["Contrasts"], sibling: ["Siblings"], latent: ["Theme links"], activates: ["Memberships", "Memberships (weak)"] };
// a legend entry under the mouse spotlights its elements in both views
const LEGEND_TRACES = { anchor: ["Themes"], up: ["Metric higher"], down: ["Metric lower"], cov: ["Correlation change"], ...SPHERE_LAYERS,
  seed: ["Seed"], ev: ["Evidence"], cross: ["Other segment"], path: ["Answer path", "Themes visited"] };
const LAYERS = {
  lattice: 'edge[type="SPECIALIZES"]', contrast: 'edge[type="CONTRASTS"]', sibling: 'edge[type="SIBLING"]',
  latent: 'edge[type="RELATED_TO"]', activates: 'edge[type="ACTIVATES"]',
  schema: 'node[kind="Dimension"], node[kind="Metric"], edge[type="HAS_SCOPE"], edge[type="TARGETS"]',
};
const LEGEND_SEL = {
  anchor: 'node[kind="Attractor"]', up: 'node[kind="Pattern"][ptype != "covariance"][direction > 0]',
  down: 'node[kind="Pattern"][ptype != "covariance"][direction < 0]', cov: 'node[kind="Pattern"][ptype = "covariance"]', ...LAYERS,
  seed: ".hl-seed", ev: ".hl-evidence", cross: ".hl-cross", path: ".hl-edge, .hl-anchor",
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(Number(x))) ? "–" : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
const signed = (x, d = 2) => (x > 0 ? "+" : "") + num(x, d);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const human = (s) => String(s || "").replace(/_/g, " ");

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}

// theme + health
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("sig-theme", t); } catch (e) { /* private mode */ }
  if (S.cy) S.cy.style(graphStyle());
  renderSphere();  // same palette in 3D
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const llm = h.llm || {};
    const llmCls = llm.reachable ? (llm.model_available ? "ok" : "warn") : "bad";
    const model = String(llm.model || "LLM").split("/").pop();
    const llmTxt = llm.reachable ? (llm.model_available ? model : `${model} not found`) : `${model} offline`;
    const emb = h.embedding ? `${String(h.embedding.model_id).split("/").pop().replace("paraphrase-multilingual-", "")} · ${h.embedding_device}` : `embeddings · ${h.embedding_device}`;
    $("health").innerHTML =
      `<span class="pill ${llmCls}" title="Language model · ${esc(llm.base_url || "")}"><i class="dot"></i>${esc(llmTxt)}</span>` +
      `<span class="pill ${h.neo4j.enabled ? "ok" : ""} opt" title="Graph database mirror"><i class="dot"></i>${h.neo4j.enabled ? "Neo4j " + esc(h.neo4j.database) : "Neo4j off"}</span>` +
      `<span class="pill opt" title="Embedding model and device"><i class="dot"></i>${esc(emb)}</span>`;
    $("llm-label").textContent = `${model} explanation`;
  } catch (e) { $("health").innerHTML = `<span class="pill bad"><i class="dot"></i>Server unreachable</span>`; }
}

// datasets
function datasetCard(b) {
  const running = !TERMINAL.has(b.status);
  const m = b.metrics || {}, p = b.profile || {};
  const active = b.status === "READY" && $("dataset-filter").value === b.dataset_id;
  let body = "";
  if (running) {
    const i = Math.max(0, STAGES.indexOf(b.status));
    body = `<div class="progress"><i style="width:${Math.round(((i + 1) / STAGES.length) * 100)}%"></i></div>` +
      `<div class="ds-step">${esc(STAGE_TEXT[b.status] || b.status)}… <span class="muted">step ${i + 1} of ${STAGES.length}</span></div>`;
  } else if (b.status === "READY") {
    body = `<div class="ds-stats">` +
      `<div class="stat"><b>${num(m.validated_insights, 0)}</b><span>insights</span></div>` +
      `<div class="stat" title="Themes first learned from this dataset"><b>${num(m.attractors_new ?? m.attractors_total, 0)}</b><span>themes</span></div>` +
      `<div class="stat"><b>${num(m.input_rows, 0)}</b><span>rows</span></div></div>` +
      `<div class="ds-note">${num(p.columns ?? (p.numerics || []).length + (p.categoricals || []).length, 0)} columns · ${num(m.candidate_patterns, 0)} subgroups tested · ${num(m.pruned_total, 0)} filtered out · ${num(m.processing_duration_s, 1)} s</div>` +
      (p.selected_dimensions && p.selected_dimensions.length ? `<div class="ds-dims">${p.selected_dimensions.map((d) => `<span class="tag">${esc(human(d))}</span>`).join("")}</div>` : "");
  } else if (b.status === "FAILED" && b.error) {
    const [title, why, tip] = ERROR_HELP[b.error.code] || ["Processing failed", b.error.message, ""];
    body = `<div class="ds-error"><b>${esc(title)}</b><div class="why">${esc(why)}</div>` +
      (tip ? `<div class="tip">💡 ${esc(tip)}</div>` : "") + `<div class="why mono" style="margin-top:6px">${esc(b.error.message)}</div></div>`;
  } else if (b.status === "SKIPPED") {
    body = `<div class="ds-note">Already analysed — nothing new to add.</div>`;
  }
  const chip = running ? `<span class="chip run">Processing</span>` : b.status === "READY" ? `<span class="chip ok">Ready</span>`
    : b.status === "FAILED" ? `<span class="chip bad">Failed</span>` : `<span class="chip">Skipped</span>`;
  const when = b.created_at ? new Date(b.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
  const del = running ? "" : `<div class="ds-foot"><button type="button" class="ds-del" data-del="${esc(b.dataset_id || b.batch_id)}" data-name="${esc(b.filename)}" title="Remove this dataset and its insights from the knowledge base">` +
    `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>Delete</button></div>`;
  return `<div class="ds ${b.status === "READY" ? "clickable" : ""} ${active ? "active" : ""}" data-ds="${esc(b.status === "READY" ? b.dataset_id : "")}">` +
    `<div class="ds-head"><span class="ds-name" title="${esc(b.filename)}">${esc(b.filename)}</span>${chip}</div>` +
    `<div class="ds-sub">${esc(when)}</div>${body}${del}</div>`;
}

async function deleteDataset(id, name) {
  if (!confirm(`Delete “${name}” from the knowledge base?\n\nIts insights, vectors and memberships are removed; themes left with no insight and no theme link are removed too.`)) return;
  try { await api(`/api/datasets/${encodeURIComponent(id)}`, { method: "DELETE" }); setUploadMsg(`Deleted ${name}.`); }
  catch (err) { setUploadMsg(err.message, true); }
  if ($("dataset-filter").value === id) $("dataset-filter").value = "";
  S.lastReady = null;  // force the views to reload
  refreshBatches();
}

async function refreshBatches() {
  let batches = [];
  try { batches = await api("/api/batches"); } catch (e) { return []; }
  // one card per dataset (newest first): its READY batch if it has one, otherwise its latest attempt
  const byDataset = new Map();
  for (const b of batches) {
    const key = b.dataset_id || b.batch_id, kept = byDataset.get(key);
    if (!kept || (b.status === "READY" && kept.status !== "READY")) byDataset.set(key, b);
  }
  const cards = [...byDataset.values()];
  $("batches").innerHTML = cards.length ? cards.map(datasetCard).join("") : `<div class="empty-rail">No datasets yet.<br>Upload a table or try the demo.</div>`;
  $("batches").querySelectorAll(".ds.clickable").forEach((el) => el.addEventListener("click", () => {
    const sel = $("dataset-filter"); sel.value = sel.value === el.dataset.ds ? "" : el.dataset.ds; loadGraph(); refreshBatches();
  }));
  $("batches").querySelectorAll(".ds-del").forEach((el) => el.addEventListener("click", (e) => { e.stopPropagation(); deleteDataset(el.dataset.del, el.dataset.name); }));
  const ready = [...new Map(batches.filter((b) => b.status === "READY").map((b) => [b.dataset_id, b.filename])).entries()];
  const sel = $("dataset-filter"), cur = sel.value;
  sel.innerHTML = `<option value="">All datasets</option>` + ready.map(([id, f]) => `<option value="${esc(id)}">${esc(f)}</option>`).join("");
  sel.value = ready.some(([id]) => id === cur) ? cur : "";
  const readyKey = batches.filter((b) => b.status === "READY").map((b) => b.batch_id).join(",");
  if (readyKey !== S.lastReady) { S.lastReady = readyKey; await loadGraph(); refreshHealth(); }
  clearTimeout(S.poll);
  if (batches.some((b) => !TERMINAL.has(b.status))) S.poll = setTimeout(refreshBatches, 900);
  return batches;
}

function setUploadMsg(text, err = false) { const el = $("upload-msg"); el.className = "msg" + (err ? " err" : ""); el.textContent = text; }

async function upload() {
  const f = $("file").files[0]; if (!f) return;
  const fd = new FormData();
  fd.append("file", f); fd.append("bins", $("bins").value); fd.append("categories", $("categories").value);
  setUploadMsg(`Uploading ${f.name}…`);
  $("upload-btn").disabled = true;
  try { await api("/api/upload", { method: "POST", body: fd }); setUploadMsg("Queued — progress is shown below."); $("file").value = ""; $("file-name").textContent = ""; refreshBatches(); }
  catch (err) { setUploadMsg(err.message, true); $("upload-btn").disabled = false; }
}

// graph
function graphStyle() {
  const ink = cssVar("--graph-ink") || "#3a4050", edge = cssVar("--graph-edge") || "#c8cdd8";
  const up = cssVar("--up"), down = cssVar("--down"), cov = cssVar("--cov"), anchor = cssVar("--anchor"), path = cssVar("--path"), bad = cssVar("--bad");
  const surface = cssVar("--surface") || "#fff", muted = cssVar("--muted");
  return [
    { selector: "node", style: { label: "data(label)", "font-size": 11, "font-family": "Inter, Segoe UI, system-ui, sans-serif", "text-wrap": "wrap", "text-max-width": 120, "text-valign": "bottom", "text-margin-y": 4, color: ink, "border-width": 0, "min-zoomed-font-size": 8, "text-outline-color": surface, "text-outline-width": 2 } },
    { selector: 'node[kind="Pattern"]', style: { shape: "ellipse", width: "mapData(weight, 0, 1, 12, 30)", height: "mapData(weight, 0, 1, 12, 30)", "background-color": up } },
    { selector: 'node[kind="Pattern"][direction < 0]', style: { "background-color": down } },
    { selector: 'node[kind="Pattern"][ptype="covariance"]', style: { "background-color": cov, shape: "round-rectangle" } },
    { selector: 'node[kind="Attractor"]', style: { shape: "hexagon", width: "mapData(n_patterns, 1, 12, 36, 66)", height: "mapData(n_patterns, 1, 12, 32, 58)", "background-color": anchor, "font-size": 11.5, "font-weight": 700, color: ink, "text-max-width": 170 } },
    { selector: 'node[kind="Dimension"]', style: { shape: "round-rectangle", width: 44, height: 18, "background-color": muted, "font-size": 9, "text-valign": "center", "text-margin-y": 0, color: "#fff", "text-outline-width": 0 } },
    { selector: 'node[kind="Metric"]', style: { shape: "tag", width: 48, height: 18, "background-color": cov, "font-size": 9, "text-valign": "center", "text-margin-y": 0, color: "#fff", "text-outline-width": 0 } },
    { selector: 'node[kind="Dataset"], node[kind="Batch"]', style: { display: "none" } },
    { selector: 'node[kind="PlaneLabel"]', style: { label: "data(label)", "background-opacity": 0, "font-size": 13, "font-weight": 700, color: muted, "text-valign": "center", "text-halign": "right", events: "no", "text-outline-width": 0 } },
    { selector: "edge", style: { width: 1, "line-color": edge, "curve-style": "bezier", opacity: 0.75 } },
    { selector: 'edge[type="SPECIALIZES"]', style: { "line-color": muted, "target-arrow-shape": "triangle", "target-arrow-color": muted, "arrow-scale": 0.7, opacity: 0.6 } },
    { selector: 'edge[type="GENERALIZES"]', style: { display: "none" } },
    { selector: 'edge[type="SIBLING"]', style: { "line-style": "dotted", "line-color": edge } },
    { selector: 'edge[type="CONTRASTS"]', style: { "line-style": "dashed", "line-color": bad, width: 1.5, opacity: 0.7 } },
    { selector: 'edge[type="ACTIVATES"]', style: { "line-color": anchor, width: "mapData(weight, 0, 1, 0.4, 2)", opacity: 0.22 } },
    { selector: 'edge[type="ACTIVATES"][?weak]', style: { "line-style": "dashed" } },
    { selector: 'edge[type="RELATED_TO"]', style: { "curve-style": "unbundled-bezier", "control-point-distances": "data(cpd)", "control-point-weights": 0.5, "line-color": anchor, width: "mapData(weight, 0, 1, 1, 6)", opacity: 0.75, label: "data(weight)", "font-size": 8.5, color: anchor, "text-background-color": surface, "text-background-opacity": 0.9, "text-background-padding": 2 } },
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

function primaryAttractor(node) {
  let best = null, w = -1;
  node.connectedEdges('[type="ACTIVATES"]').forEach((e) => { if (e.data("weight") > w) { w = e.data("weight"); best = e.target().id(); } });
  return best;
}

function attractorOrder(cy) {
  /* greedy chain: strongly RELATED_TO themes end up adjacent on the latent row */
  const atts = cy.nodes('[kind="Attractor"]').toArray();
  if (!atts.length) return [];
  const w = (a, b) => { let m = 0; a.edgesWith(b).filter('[type="RELATED_TO"]').forEach((e) => { m = Math.max(m, e.data("weight")); }); return m; };
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
  cy.nodes('[kind="Pattern"]').forEach((p) => { const a = primaryAttractor(p); (a && groups.has(a) ? groups.get(a) : loose).push(p); });
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
  const bottom = cy.nodes('[kind="Dimension"], [kind="Metric"]');
  const span = Math.max(x - 50, 400);
  bottom.forEach((n, i) => { pos[n.id()] = { x: (i + 0.5) * span / Math.max(bottom.length, 1), y: maxY + 170 }; });
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

function layerOn(name) { const b = document.querySelector(`.toggle[data-layer="${name}"]`); return b && b.classList.contains("on"); }

function layerState() { return Object.fromEntries(Object.keys(SPHERE_LAYERS).map((k) => [k, layerOn(k)])); }

function applyVisibility() {
  const cy = S.cy;
  if (cy) {
    for (const [name, sel] of Object.entries(LAYERS)) cy.$(sel).toggleClass("hidden", !layerOn(name));
    cy.$('node[kind="PlaneLabel"]').toggleClass("hidden", $("layout").value !== "dual");
  }
  applySphereLayers();
}

function sphereDiv() {
  /* the sphere page is a same-origin srcdoc frame: its traces are restyled in place, no re-render */
  const win = $("sphere").contentWindow, gd = win && win.document && win.document.querySelector(".plotly-graph-div");
  return gd && win.Plotly && gd.data ? gd : null;
}

function applySphereLayers() {
  const gd = sphereDiv(); if (!gd) return;
  for (const [name, traces] of Object.entries(SPHERE_LAYERS)) {
    const idx = gd.data.map((t, i) => (traces.includes(t.name) ? i : -1)).filter((i) => i >= 0);
    if (idx.length) $("sphere").contentWindow.Plotly.restyle(gd, { visible: layerOn(name) }, idx);
  }
}

function wireSphere() {
  /* a fresh sphere page: clicks open the drawer (customdata = node id), layers follow the legend */
  const gd = sphereDiv(); if (!gd) return;
  gd.on("plotly_click", (ev) => { const id = ev.points && ev.points[0] && ev.points[0].customdata; if (id) inspect(id); });
  applySphereLayers();
}

function spotlight(key) {
  /* hovering a legend entry: everything else dims, in the graph and on the sphere */
  if (S.cy) {
    S.cy.elements().removeClass("lg-dim");
    const sel = key ? S.cy.$(LEGEND_SEL[key]) : null;
    if (sel && sel.nonempty()) S.cy.elements().not(sel.union(sel.filter("edge").connectedNodes())).addClass("lg-dim");
  }
  const gd = sphereDiv(); if (!gd) return;
  gd._sigOpacity = gd._sigOpacity || gd.data.map((t) => (t.opacity === undefined ? 1 : t.opacity));
  const names = LEGEND_TRACES[key] || [];
  const hit = gd.data.some((t) => names.includes(t.name));
  $("sphere").contentWindow.Plotly.restyle(gd, { opacity: gd.data.map((t, i) => (!hit || names.includes(t.name) ? gd._sigOpacity[i] : 0.07)) });
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
  try { g = await api("/api/graph" + (ds ? `?dataset=${encodeURIComponent(ds)}` : "")); } catch (e) { return; }
  S.nodes = g.nodes.map((n) => n.data).filter((d) => d.kind === "Pattern");
  $("ins-count").textContent = S.nodes.length ? S.nodes.length : "";
  const nodes = (f) => g.nodes.filter((n) => f(n.data)).length, edges = (t) => g.edges.filter((e) => e.data.type === t).length;
  const pat = (f) => nodes((d) => d.kind === "Pattern" && f(d));
  setCounts({
    anchor: nodes((d) => d.kind === "Attractor"), up: pat((d) => d.ptype !== "covariance" && d.direction > 0),
    down: pat((d) => d.ptype !== "covariance" && d.direction < 0), cov: pat((d) => d.ptype === "covariance"),
    lattice: edges("SPECIALIZES"), latent: edges("RELATED_TO"), contrast: edges("CONTRASTS"), sibling: edges("SIBLING"), activates: edges("ACTIVATES"),
    schema: nodes((d) => d.kind === "Dimension" || d.kind === "Metric"),
  });
  $("view-caption").textContent = S.nodes.length ? `${S.nodes.length} insights · ${nodes((d) => d.kind === "Attractor")} themes in view` : "No insights in view";
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
  applyVisibility(); runLayout();
  if (S.qa) highlight(S.qa.highlight);
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
  $("sphere-open").hidden = view !== "sphere";
  document.querySelectorAll(".graph-only").forEach((el) => { el.hidden = view !== "graph"; });
  $("fit-btn").hidden = view === "table";
  const columns = document.querySelector('.toggle[data-layer="schema"]');  // column nodes have no vectors: graph only
  columns.disabled = view === "sphere"; columns.title = view === "sphere" ? "Columns are not drawn in 3D (they have no vectors)" : "Dimension and metric columns (graph only)";
  $("empty-hint").hidden = view === "table" || S.nodes.length > 0;
  if (view === "sphere") renderSphere();
  if (view === "graph" && S.cy) { S.cy.resize(); S.cy.fit(S.cy.elements(":visible"), 40); }
}

async function renderSphere(hl = null) {
  const ds = $("dataset-filter").value;
  $("sphere-open").href = "/api/sphere" + (ds ? `?dataset=${encodeURIComponent(ds)}` : "");
  if (S.view !== "sphere") return;
  const seq = ++S.sphereSeq;
  const frame = $("sphere");
  const pal = spherePalette();
  frame.srcdoc = `<body style="background:${pal.bg};color:${pal.muted};font:14px system-ui;padding:24px">Projecting insight vectors on the sphere…</body>`;
  try {
    const res = await fetch("/api/sphere", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: ds || null, highlight: hl || (S.qa && S.qa.highlight) || null, palette: pal, layers: layerState() }) });
    const html = await res.text();
    if (seq === S.sphereSeq) frame.srcdoc = html;  // ignore stale renders
  } catch (e) { if (seq === S.sphereSeq) frame.srcdoc = `<body style="background:${pal.bg};color:${pal.bad};font:14px system-ui;padding:24px">${esc(e.message)}</body>`; }
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

function highlight(h, onlyEdges = null) {
  const cy = S.cy; if (!cy || !h) return;
  clearHighlight();
  cy.elements().not('[kind="PlaneLabel"]').addClass("faded");
  const mark = (ids, cls) => (ids || []).forEach((id) => { const el = cy.getElementById(id); if (el.nonempty()) el.removeClass("faded").addClass("hl-node " + cls); });
  mark(onlyEdges ? [] : h.traversed, "");
  mark(h.evidence, "hl-evidence");
  mark(h.transversal_only, "hl-cross");
  mark(h.anchors, "hl-anchor");
  mark(h.seeds, "hl-seed");
  (onlyEdges || h.edges || []).forEach((id) => {
    const e = cy.getElementById(id);
    if (e.nonempty()) { e.removeClass("faded hidden").addClass("hl-edge"); e.connectedNodes().removeClass("faded").addClass("hl-node"); }
  });
  setCounts({ seed: (h.seeds || []).length, ev: (h.evidence || []).length, cross: (h.transversal_only || []).length, path: (onlyEdges || h.edges || []).length });
  $("clear-btn").hidden = false;
}

function highlightPath(turn, item, el) {
  const h = turn.qa.highlight;
  const edges = item.path.map((st) => st.edge_id);
  const one = { seeds: h.seeds, anchors: item.path.filter((s) => s.target.startsWith("A-")).map((s) => s.target), evidence: [item.pattern_id],
    transversal_only: item.transversal_only ? [item.pattern_id] : [], edges, traversed: [] };
  S.qa = turn.qa;
  if (S.view === "sphere") { renderSphere(one); } else {
    if (S.view === "table") setView("graph");
    highlight(one, edges);
    focusNodes([item.pattern_id, ...item.path.map((s) => s.source)]);
  }
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

async function inspect(id) {
  let d;
  try { d = await api(`/api/nodes/${encodeURIComponent(id)}`); } catch (e) { return; }
  const n = d.node, p = n.props;
  const kinds = { Pattern: "Insight", Attractor: "Theme · latent anchor", Dimension: "Grouping column", Metric: "Metric", Dataset: "Dataset", Batch: "Batch" };
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
      ["Correlation change", cov ? `${esc(human(cov.pair.join(" ~ ")))}: ${num(cov.global_corr)} → ${num(cov.local_corr)}` + (cov.p_adjusted !== undefined ? ` (p ${Number(cov.p_adjusted).toExponential(1)})` : "") : ""],
      ["Confounders", esc((p.drivers || []).join("; ") || "none detected")],
      ["Also known as", esc((p.aliases || []).join("; "))],
    ]);
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
  } else {
    $("drawer-title").textContent = n.label;
    html += kv(Object.entries(p).map(([k, v]) => [human(k), esc(typeof v === "object" ? JSON.stringify(v) : v)]));
  }
  const groups = Object.entries(d.neighbors);
  if (groups.length) {
    const friendly = { "ACTIVATES": "Belongs to theme", "ACTIVATES (in)": "Member insights", "RELATED_TO": "Related themes", "RELATED_TO (in)": "Related themes",
      "SPECIALIZES": "Narrower version of", "GENERALIZES": "Broader version of", "SPECIALIZES (in)": "Narrower insights", "GENERALIZES (in)": "Broader insights",
      "CONTRASTS": "Contrasts with", "CONTRASTS (in)": "Contrasts with", "SIBLING": "Siblings", "SIBLING (in)": "Siblings", "HAS_SCOPE": "Columns", "TARGETS": "Metrics" };
    html += `<h4>Connections</h4>` + groups.filter(([t]) => !["DISCOVERED_IN", "OF_DATASET", "HAS_SCOPE (in)", "TARGETS (in)"].includes(t)).map(([type, items]) =>
      `<details class="nb-group" ${/ACTIVATES|RELATED|CONTRAST/.test(type) ? "open" : ""}><summary>${esc(friendly[type] || type)} <span class="muted">(${items.length})</span></summary>` +
      items.map((it) => `<div class="nb" data-id="${esc(it.id)}"><span>${esc(it.label)}</span><span class="muted">${num(Number(it.weight))}</span></div>`).join("") + `</details>`).join("");
  }
  $("inspector").innerHTML = html;
  $("inspector").querySelectorAll(".nb").forEach((el) => el.addEventListener("click", () => { focusNodes([el.dataset.id]); inspect(el.dataset.id); }));
  $("drawer").hidden = false;
  if (S.cy) { S.cy.$(":selected").unselect(); S.cy.getElementById(id).select(); }
}

// chat
function md(text, keyTo) {
  /* small, safe markdown: escape first, then bold/italic/code, headings, lists, paragraphs, citations */
  const cite = (s) => s.replace(/\[(P\d+(?:\s*[,;]\s*P\d+)*)\]/g, (m, g) => g.split(/\s*[,;]\s*/).map((k) => keyTo[k] ? `<a class="cite" data-pid="${esc(keyTo[k])}" title="${esc(keyTo[k])}">${k}</a>` : k).join(""));
  const inline = (s) => cite(s.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>"));
  const lines = esc(text).split(/\r?\n/);
  let html = "", list = null, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const flushList = () => { if (list) { html += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trim();
    let m;
    if (!line) { flushPara(); flushList(); continue; }
    if ((m = line.match(/^#{1,4}\s+(.*)$/)) || (m = line.match(/^(?:<b>)?([A-ZА-ЯІЇЄҐ][^:]{2,60}):(?:<\/b>)?$/))) { flushPara(); flushList(); html += `<h4>${inline(m[1])}</h4>`; continue; }
    if ((m = line.match(/^[*\-•]\s+(.*)$/))) { flushPara(); if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; } list.items.push(m[1]); continue; }
    if ((m = line.match(/^\d+[.)]\s+(.*)$/))) { flushPara(); if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; } list.items.push(m[1]); continue; }
    flushList(); para.push(line);
  }
  flushPara(); flushList();
  return html;
}

function renderChain(item) {
  if (item.role === "seed" || !item.path.length) return `<div class="chain"><span class="n">matched your question</span></div>`;
  let html = `<span class="n">${esc(item.path[0].source)}</span>`;
  for (const st of item.path) {
    const label = { ACTIVATES: st.reverse ? "member" : "theme", RELATED_TO: "related theme", SPECIALIZES: "broader", GENERALIZES: "narrower", CONTRASTS: "contrast" }[st.edge_type] || st.edge_type;
    html += `<span class="e">→ ${esc(label)} ${num(st.weight)} →</span><span class="n ${st.target.startsWith("A-") ? "A" : ""}">${esc(st.target)}</span>`;
  }
  return `<div class="chain">${html}</div>`;
}

function botCard(turn) {
  const qa = turn.qa;
  const items = (qa.evidence && qa.evidence.items) || [];
  const keyTo = Object.fromEntries(items.map((i) => [i.key, i.pattern_id]));
  const llm = qa.llm || {};
  const answer = qa.answer || "";
  const notice = qa.answer_mode !== "fallback" ? ""
    : llm.error === "disabled" ? "Showing the evidence only (explanation turned off)."
    : "The language model is unavailable — showing the verified evidence only.";
  const cit = qa.citations || {};
  const b = (qa.traversal && qa.traversal.baselines) || {};
  const cross = items.filter((i) => i.transversal_only).length;
  const model = String(llm.model || "").split("/").pop();
  const meta = [
    qa.answer_mode === "llm" ? `<span class="chip">✨ ${esc(model)} · ${num(llm.latency_s, 1)} s</span>` : `<span class="chip warn">Evidence only</span>`,
    cit.grounded ? `<span class="chip ok">✓ ${(cit.cited || []).length} sources cited</span>` : `<span class="chip warn">No citations</span>`,
    cross ? `<span class="chip run" title="Reached only through a theme, in a different part of the data">⤳ ${cross} cross-segment</span>` : "",
  ].join("");
  // literal grounding (docs/07 §7.1.1): which words of the question were read as which data literals
  const grounded = ((qa.evidence && qa.evidence.parsed && qa.evidence.parsed.grounding) || []).filter((g) => g.symbol !== "up" && g.symbol !== "down");
  const understood = grounded.length ? `<div class="understood" title="Question words matched to data literals (exact, by characters, or by meaning)">Understood: ${grounded.map((g) => `<b>${esc(g.span)}</b> → ${esc(g.literal)}`).join(" · ")}</div>` : "";
  const evid = items.map((it, i) => {
    const s = it.statistics;
    const sh = it.phenomenon_type === "covariance"  // its median shifts are not validated: show the correlation change
      ? esc(it.relationship || "correlation change")
      : s.shifts.slice(0, 2).map((x) => `${esc(human(x.metric))} ${signed(x.robust_z)} sd`).join(" · ");
    return `<div class="ev" data-i="${i}"><div class="ev-head"><span class="ev-key">${esc(it.key)}</span><span class="role ${esc(it.role)}">${esc({ seed: "match", structural: "lattice", transversal: "via theme" }[it.role] || it.role)}</span>` +
      (it.transversal_only ? `<span class="role cross">other segment</span>` : "") + `</div>` +
      `<div class="ev-scope">${esc(it.scope.map((c) => human(c).replace("=", " = ")).join(" · "))}</div>` +
      `<div class="ev-stats">${sh} · ${num(s.support, 0)} rows · evidence ${num(s.weight)}</div>${renderChain(it)}</div>`;
  }).join("");
  const hypo = b.transversal_only ? `<div class="hypo"><b>Трансверсальна перевірка:</b> ${b.transversal_only.length} із цих інсайтів не мають жодної спільної умови з відправною точкою і знайдені лише через тему; ${(b.not_in_naive_topk || []).length} пропустив би звичайний текстовий пошук.</div>` : "";
  const panes = [
    items.length ? ["Evidence & how it was found", `<span class="count chip">${items.length}</span>`, `<p class="ev-prefix">Click an item to isolate its path in the graph or on the sphere.</p>${hypo}${evid}`] : null,
    ["Sources", "", `<div class="muted" style="font-size:12.5px">${esc(qa.provenance_footer || "")}</div>`],
    qa.evidence && qa.evidence.prompt ? ["Prompt", "", `<p class="ev-prefix">The exact evidence the model was shown.</p><pre class="canon">${esc(qa.evidence.prompt)}</pre>`] : null,
  ].filter(Boolean);
  return `<div class="bot-card">${notice ? `<div class="notice">${esc(notice)}</div>` : ""}<div class="answer-box">${md(answer, keyTo)}</div>` +
    `<div class="bot-meta">${meta}</div>${understood}` +
    `<div class="bot-tabs" role="tablist">${panes.map(([t, c], i) => `<button type="button" role="tab" data-pane="${i}" aria-expanded="false">${esc(t)}${c}</button>`).join("")}</div>` +
    panes.map(([, , body], i) => `<div class="bot-pane" data-pane="${i}" hidden>${body}</div>`).join("") +
    `</div>`;
}

function followUps(turn) {
  const items = (turn.qa.evidence && turn.qa.evidence.items) || [];
  const atts = (turn.qa.evidence && turn.qa.evidence.attractors) || [];
  const out = [];
  if (atts[0]) out.push(`Де ще трапляється «${atts[0].label}»?`);
  const cross = items.find((i) => i.transversal_only);
  if (cross) out.push(`Розкажи про ${cross.target} для ${cross.scope.map((c) => c.split("=")[1]).join(" ")}`);
  const seed = items[0];
  if (seed && seed.statistics.covariance && seed.statistics.covariance.pair) out.push(`Як пов'язані ${seed.statistics.covariance.pair.join(" і ")}?`);
  return out.slice(0, 3);
}

const chipButtons = (questions) => questions.map((q) => `<button type="button" class="chip-q">${esc(q)}</button>`).join("");

function botTurn(turn, isLast) {
  if (turn.pending) {
    return `<div class="bot-card"><div class="typing"><i></i><i></i><i></i><span>Searching the graph${$("use-llm").checked ? " and writing an answer" : ""}…</span></div></div>`;
  }
  if (turn.error) return `<div class="bot-card error"><div class="answer-box">${esc(turn.error)}</div></div>`;
  const next = isLast ? followUps(turn) : [];
  return botCard(turn) + (next.length ? `<div class="followups">${chipButtons(next)}</div>` : "");
}

function renderThread() {
  const t = $("thread");
  if (!S.chat.length) {
    t.innerHTML = `<div class="welcome"><div class="art"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 5h16v11H8l-4 4z"/></svg></div>` +
      `<h3>Ask about your data</h3><p>Answers are built only from validated insights in the graph, with every claim linked to its source.</p>` +
      `<div class="suggest">${chipButtons(SUGGESTIONS)}</div></div>`;
  } else {
    t.innerHTML = S.chat.map((turn, idx) =>
      `<div class="msg-user">${esc(turn.question)}</div><div class="msg-bot" data-turn="${idx}">${botTurn(turn, idx === S.chat.length - 1)}</div>`).join("");
  }
  t.querySelectorAll(".chip-q").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));
  t.querySelectorAll(".msg-bot[data-turn]").forEach((box) => {
    const turn = S.chat[Number(box.dataset.turn)];
    if (!turn || !turn.qa) return;
    const items = turn.qa.evidence.items || [];
    box.querySelectorAll("a.cite").forEach((a) => a.addEventListener("click", () => { if (S.view === "table") setView("graph"); focusNodes([a.dataset.pid]); inspect(a.dataset.pid); }));
    box.querySelectorAll(".ev").forEach((el) => el.addEventListener("click", () => highlightPath(turn, items[Number(el.dataset.i)], el)));
    box.querySelectorAll(".bot-tabs button").forEach((tab) => tab.addEventListener("click", () => {
      const open = !tab.classList.contains("active");  // a click on the open panel closes it
      box.querySelectorAll(".bot-tabs button").forEach((t) => { t.classList.toggle("active", open && t === tab); t.setAttribute("aria-expanded", String(open && t === tab)); });
      box.querySelectorAll(".bot-pane").forEach((pane) => { pane.hidden = !open || pane.dataset.pane !== tab.dataset.pane; });
    }));
  });
  t.scrollTop = t.scrollHeight;
}

async function ask(question) {
  question = (question || "").trim();
  if (!question || S.busy) return;
  S.busy = true; $("ask-btn").disabled = true;
  $("question").value = ""; autosize();
  const turn = { question, pending: true };
  S.chat.push(turn); renderThread();
  try {
    const qa = await api("/api/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, use_llm: $("use-llm").checked }) });
    Object.assign(turn, { pending: false, qa });
    S.qa = qa;
    const evDatasets = ((qa.evidence && qa.evidence.datasets) || []).map((d) => d.dataset_id);
    if ($("dataset-filter").value && evDatasets.length && !evDatasets.includes($("dataset-filter").value)) {
      $("dataset-filter").value = evDatasets.length === 1 ? evDatasets[0] : "";  // show where the evidence lives
      await loadGraph();
    }
    renderThread();
    highlight(qa.highlight);
    if (S.view === "graph") focusNodes([...(qa.highlight.traversed || [])]);
    renderSphere();
  } catch (e) { Object.assign(turn, { pending: false, error: e.message }); renderThread(); }
  S.busy = false; $("ask-btn").disabled = !$("question").value.trim();
}

function autosize() { const q = $("question"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 140) + "px"; $("ask-btn").disabled = S.busy || !q.value.trim(); }

function setLegend(show) {
  $("legend").hidden = !show;
  $("legend-btn").setAttribute("aria-pressed", String(show));
  try { localStorage.setItem("sig-legend", show ? "on" : "off"); } catch (e) { /* private mode */ }
  if (S.cy) S.cy.resize();
}

function initResize() {
  /* drag the border of the datasets rail or of the chat; widths are remembered, a double-click resets */
  const app = document.querySelector(".app");
  try { for (const col of ["rail", "chat"]) { const w = localStorage.getItem(`sig-${col}-w`); if (w) app.style.setProperty(`--${col}-w`, w); } } catch (e) { /* private mode */ }
  document.querySelectorAll(".col-resize").forEach((h) => {
    const col = h.dataset.col, other = document.querySelector(col === "rail" ? ".chat" : ".rail");
    const save = () => { try { localStorage.setItem(`sig-${col}-w`, app.style.getPropertyValue(`--${col}-w`)); } catch (e) { /* private mode */ } };
    h.addEventListener("pointerdown", (e) => {
      e.preventDefault(); h.setPointerCapture(e.pointerId); h.classList.add("on"); document.body.classList.add("resizing");
      const box = app.getBoundingClientRect(), max = box.width - other.getBoundingClientRect().width - 380;
      const move = (ev) => {
        const w = col === "rail" ? ev.clientX - box.left : box.right - ev.clientX;
        app.style.setProperty(`--${col}-w`, `${Math.round(Math.min(Math.max(w, col === "rail" ? 220 : 300), max))}px`);
        if (S.cy) S.cy.resize();
      };
      const up = () => {
        h.classList.remove("on"); document.body.classList.remove("resizing");
        h.removeEventListener("pointermove", move); h.removeEventListener("pointerup", up); save();
      };
      h.addEventListener("pointermove", move); h.addEventListener("pointerup", up);
    });
    h.addEventListener("dblclick", () => { app.style.removeProperty(`--${col}-w`); save(); if (S.cy) S.cy.resize(); });
  });
}

// wiring
function init() {
  $("theme-btn").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  // upload: click or drop
  const drop = $("drop");
  $("file").addEventListener("change", () => { const f = $("file").files[0]; $("file-name").textContent = f ? f.name : ""; $("upload-btn").disabled = !f; });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { $("file").files = e.dataTransfer.files; $("file").dispatchEvent(new Event("change")); } });
  $("upload-form").addEventListener("submit", (e) => { e.preventDefault(); upload(); });
  $("demo-btn").addEventListener("click", async () => {
    setUploadMsg("Preparing the demo dataset…");
    try { await api("/api/demo", { method: "POST" }); setUploadMsg("Demo queued — progress is shown below."); refreshBatches(); } catch (err) { setUploadMsg(err.message, true); }
  });
  // canvas
  document.querySelectorAll("#view-switch button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  document.querySelectorAll(".toggle[data-layer]").forEach((b) => b.addEventListener("click", () => { b.classList.toggle("on"); applyVisibility(); if (S.qa && S.cy) highlight(S.qa.highlight); }));
  $("layout").addEventListener("change", () => { applyVisibility(); runLayout(); });
  $("dataset-filter").addEventListener("change", () => { loadGraph(); refreshBatches(); });
  $("fit-btn").addEventListener("click", () => {
    const gd = S.view === "sphere" && sphereDiv();
    if (gd) $("sphere").contentWindow.Plotly.relayout(gd, { "scene.camera.eye": { x: 1.1, y: 1.1, z: 1.1 }, "scene.camera.center": { x: 0, y: 0, z: 0 } });
    else if (S.cy) S.cy.fit(S.cy.elements(":visible"), 40);
  });
  $("sphere").addEventListener("load", wireSphere);
  document.querySelectorAll("#legend .key[data-key]").forEach((k) => {
    k.addEventListener("mouseenter", () => spotlight(k.dataset.key));
    k.addEventListener("mouseleave", () => spotlight(null));
  });
  $("legend-btn").addEventListener("click", () => setLegend($("legend").hidden));
  $("clear-btn").addEventListener("click", () => { S.qa = null; clearHighlight(); $("clear-btn").hidden = true; renderSphere(); });
  $("drawer-close").addEventListener("click", closeDrawer);
  $("table-search").addEventListener("input", renderTable);
  document.querySelectorAll("#ins-table th").forEach((th) => th.addEventListener("click", () => {
    S.sort = { k: th.dataset.k, asc: S.sort.k === th.dataset.k ? !S.sort.asc : false }; renderTable();
  }));
  // chat
  $("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
  $("question").addEventListener("input", autosize);
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("question").value); } });
  $("new-chat").addEventListener("click", () => { S.chat = []; S.qa = null; clearHighlight(); $("clear-btn").hidden = true; renderThread(); renderSphere(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  initResize();
  try { if (localStorage.getItem("sig-legend") === "off") setLegend(false); } catch (e) { /* private mode */ }
  renderThread(); refreshHealth(); refreshBatches();
  setInterval(refreshHealth, 30000);
}

document.addEventListener("DOMContentLoaded", init);
