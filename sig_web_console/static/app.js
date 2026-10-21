/* SIG UI — the shell: state, vocabulary, the API helper, health, views' wiring (docs/08_interface.md); datasets.js, graph.js, chat.js and geo.js hold the surfaces. */
"use strict";

const S = { datasets: {}, cy: null, hl: null, poll: null, lastReady: null, options: null, view: "graph", sphere: null, sphereSeq: 0, graphSeq: 0, regionsIn: false, regionCount: 0, nodes: [], sort: { k: "weight", asc: false }, chat: [], busy: false };
const STAGE_TEXT = {
  UPLOADED: "Queued", VALIDATING: "Reading the file", PROFILING: "Profiling columns", DISCOVERING: "Searching subgroups",
  VALIDATING_INSIGHTS: "Validating insights", EMBEDDING: "Embedding insights", UPDATING_ONTOLOGY: "Learning themes",
  BUILDING_GRAPH: "Building the graph", PERSISTING: "Saving", READY: "Ready",
};
const STAGES = Object.keys(STAGE_TEXT);  // in processing order
const TERMINAL = new Set(["READY", "FAILED", "SKIPPED"]);
const ERROR_HELP = {
  no_candidates: ["No subgroups to analyse", "There are no usable columns to group rows by (text columns with a handful of values).", "Mark number-coded columns such as IDs or flags under “Treat as categories”, or split a numeric column into bands."],
  invalid_schema: ["The table can’t be analysed", "It needs at least one grouping column, one numeric column and enough rows.", "Check the file, or mark code-like columns as categories."],
  no_numeric_targets: ["No numeric columns", "SIG measures shifts in numeric columns and found none it can use.", "Make sure numbers aren’t stored as text."],
  no_viable_insights: ["Nothing significant found", "Subgroups were tested, but none passed the significance, stability and effect-size checks.", "Try more rows or other grouping columns."],
  unsupported_file: ["Unsupported file", "Use CSV, TSV or Parquet (up to the upload limit).", ""],
  unreadable_file: ["Couldn’t read the file", "The parser failed — check the delimiter and encoding.", ""],
  invalid_options: ["Option problem", "A column named in the options doesn’t exist or isn’t numeric, or a map zone could not be read.", "Check the spelling of the column names, or the zone’s cells or shape."],
  representation_mismatch: ["Workspace mismatch", "This workspace was built with a different embedding setup.", "Reset the workspace to change the embedding model or weights."],
  duplicate_patterns: ["Already ingested", "These insights are already in the graph.", ""],
  interrupted: ["Interrupted", "Processing stopped before finishing; nothing partial was kept.", "Upload the file again."],
  chat_timeout: ["No answer in time.", "The graph and the language model did not answer within the time limit.", "Ask about one dataset or a smaller zone, or turn the explanation off to get the evidence alone."],
};
const CHAT_TIMEOUT_S = 800;  // a little over nginx's 780 s for /api/chat/
// chat content is Ukrainian; data literals (column names, values) stay exactly as the data holds them.
// The welcome card's questions per demo dataset (by filename); any other geo dataset gets GeoMap.suggestions().
const SUGGESTIONS = {
  "retail_synthetic.csv": [
    "Чому margin нижчий для phones у US?",
    "Що пов'язано з вищим return_rate?",
    "Розкажи про margin для laptops у EU",
    "Де руйнується кореляція між discount і margin?",
  ],
  "california_housing.csv": [
    "Чому median house value нижчий у зонах INLAND?",
    "Де median_house_value вищий?",
    "Що особливого в зонах NEAR OCEAN?",
    "Де найбільше households?",
  ],
};
// The legend (index.html data-key) -> the graph elements of each entry; a sphere trace carries its entry's key as
// `meta` (sphereTraces), and the toggles (.toggle[data-key]) show or hide their entry in both views.
const MARKS = {
  anchor: 'node[kind="Attractor"]', up: 'node[kind="Pattern"][ptype != "covariance"][direction > 0]',
  down: 'node[kind="Pattern"][ptype != "covariance"][direction < 0]', cov: 'node[kind="Pattern"][ptype = "covariance"]',
  lattice: 'edge[type="SPECIALIZES"]', contrast: 'edge[type="CONTRASTS"]', sibling: 'edge[type="SIBLING"]', colocated: 'edge[type="CO_LOCATED"]',
  latent: 'edge[type="RELATED_TO"], edge[type="CO_OCCURS"]', activates: 'edge[type="ACTIVATES"]',
  schema: 'node[kind="Dimension"], node[kind="Metric"], edge[type="HAS_SCOPE"], edge[type="TARGETS"]',
  region: 'node[kind="Region"], edge[type="IN_REGION"], edge[type="PARENT_OF"]',
};
const NO_VECTORS = ["schema", "region"];  // legend entries the sphere cannot draw
// trust (docs/06_graph_and_storage.md §6.2) and scale (docs/07_question_answering.md §7.4) in the console's words
const RELIABILITY = { A: "completely reliable", B: "usually reliable", C: "fairly reliable", D: "not usually reliable", E: "unreliable", F: "reliability cannot be judged" };
const CREDIBILITY = {  // level -> chip class, words
  1: ["ok", "confirmed by another source"], 2: ["warn", "not confirmed by another source"],
  5: ["bad", "contradicted by another source"], 6: ["", "no independent source covers this"],
};
const INTEGRITY = { ok: ["ok", "its statistics match their digest"], mismatch: ["bad", "its statistics no longer match their digest"], unverified: ["warn", "recorded before digests"] };
const SCALE = {
  regional: "regional — holds at a coarser level", "scale-dependent": "scale-dependent — reverses at a coarser level",
  not_found_coarser: "not found at the coarser level", insufficient: "no coarser level analysed",
};
const cred = (level) => CREDIBILITY[level] || ["warn", "not rated"];
const RELATION = { overlaps: "zones overlap", adjacent: "zones touch", same_region: "same region, zones apart" };
const LEGEND = { ...MARKS, seed: ".hl-seed", ev: ".hl-evidence", cross: ".hl-cross", path: ".hl-edge, .hl-anchor" };  // + the answer's marks

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(Number(x))) ? "–" : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
const signed = (x, d = 2) => (x > 0 ? "+" : "") + num(x, d);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const human = (s) => String(s || "").replace(/_/g, " ");
const plural = (n, word) => `${num(n, 0)} ${word}${n === 1 ? "" : "s"}`;
const on = (root, selector, fn) =>  // one click listener on a container for every ``selector`` its renders produce
  root.addEventListener("click", (e) => { const el = e.target.closest(selector); if (el && root.contains(el)) fn(el, e); });
const scopeText = (p) => p.conditions.map((c) => `${human(c.attribute)} = ${c.value}`).join(" and ");
function insightLead(p, tail = "") {
  /* an insight in one sentence (docs/08_interface.md §8.5): its target's shift, or its correlation change */
  const cov = p.covariance && p.covariance.pair ? p.covariance : null;
  return p.phenomenon_type === "covariance" && cov
    ? `The relationship between <b>${esc(human(cov.pair[0]))}</b> and <b>${esc(human(cov.pair[1]))}</b> changes here: correlation ${num(cov.global_corr)} overall → <b>${num(cov.local_corr)}</b>${tail}.`
    : `<b>${esc(human(p.target))}</b> is ${p.effect_size > 0 ? "higher" : "lower"} here — median ${num(p.local, 4)} vs ${num(p.baseline, 4)} overall (${signed(p.effect_size)} sd).`;
}
const chip = (cls, text, title = "") => `<span class="chip ${cls}"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
const nbRow = (id, text, note = "") => `<div class="nb" data-id="${esc(id)}"><span>${esc(text)}</span><span class="muted">${esc(note)}</span></div>`;
const picture = (r) => `${plural(r.patterns, "insight")} · ${plural(r.datasets, "dataset")} · ${plural(r.themes, "theme")}`;  // what a region holds

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(typeof body.detail === "string" ? body.detail : body.detail ? JSON.stringify(body.detail) : res.statusText), { status: res.status });
  return body;
}

// theme + health
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("sig-theme", t); } catch (e) { /* private mode */ }
  if (S.cy) S.cy.style(graphStyle());
  renderSphere();  // same palette in 3D
}

const CPU_HINT = "Embeddings run on the CPU: slower ingests. On a Mac run the graph service on the host, where it uses the GPU (MPS; README, Mac: GPU); with an NVIDIA GPU start the stack with compose.gpu.yaml.";

async function refreshHealth() {
  try {
    const [h, chat] = await Promise.all([api("/api/health"), api("/api/chat/health").catch(() => ({}))]);
    const llm = chat.llm || {};
    const llmCls = llm.reachable ? (llm.model_available ? "ok" : "warn") : "bad";
    const model = String(llm.model || "LLM").split("/").pop();
    const llmTxt = llm.reachable ? (llm.model_available ? model : `${model} not found`) : `${model} offline`;
    const cpu = h.embedding_device === "cpu";  // amber: on a Mac, Docker sees no GPU (README, Mac: GPU)
    const emb = h.embedding ? `${String(h.embedding.model_id).split("/").pop()} · ${h.embedding_device}` : `embeddings · ${h.embedding_device}`;
    $("health").innerHTML =
      `<span class="pill ${llmCls}" title="Language model · ${esc(llm.base_url || "")}"><i class="dot"></i>${esc(llmTxt)}</span>` +
      `<span class="pill ${h.neo4j.enabled ? "ok" : ""} opt" title="Graph database mirror"><i class="dot"></i>${h.neo4j.enabled ? "Neo4j " + esc(h.neo4j.database) : "Neo4j off"}</span>` +
      `<span class="pill ${cpu ? "warn " : ""}opt" title="${cpu ? CPU_HINT : "Embedding model and device"}"><i class="dot"></i>${esc(emb)}</span>`;
    $("llm-label").textContent = `${model} explanation`;
    $("reset-btn").hidden = !h.problem;  // a degraded workspace: only a reset makes it usable again
    if (h.problem) setUploadMsg(`The workspace can’t be used: ${h.problem}`, true);
  } catch (e) { $("health").innerHTML = `<span class="pill bad"><i class="dot"></i>Server unreachable</span>`; }
}

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
function clearAnswer() {
  /* the one "clear the answer" action: its highlight in the graph and on the sphere, the map's focus and evidence */
  S.hl = null; clearHighlight(); $("clear-btn").hidden = true; renderSphere(); GeoMap.clear();
}

function init() {
  $("theme-btn").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  // upload: click or drop
  const drop = $("drop");
  $("file").addEventListener("change", () => { const f = $("file").files[0]; $("file-name").textContent = f ? f.name : ""; $("upload-btn").disabled = !f; });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { $("file").files = e.dataTransfer.files; $("file").dispatchEvent(new Event("change")); } });
  const syncTable = () => { $("table").disabled = !$("geo").value.trim(); };  // the base level is one of a geo upload's
  $("geo").addEventListener("input", syncTable);
  window.addEventListener("pageshow", syncTable);  // a value the browser restored fires no input
  $("upload-form").addEventListener("submit", (e) => { e.preventDefault(); upload(); });
  for (const [id, path] of [["demo-btn", "/api/demo"], ["demo-geo-btn", "/api/demo/geo"]]) {
    $(id).addEventListener("click", async () => {
      setUploadMsg("Preparing the demo dataset…");
      try { await api(path, { method: "POST" }); setUploadMsg("Demo queued — progress is shown below."); refreshBatches(); } catch (err) { setUploadMsg(err.message, true); }
    });
  }
  $("reset-btn").addEventListener("click", async () => {
    if (!confirm("Reset the workspace?\n\nEvery dataset, insight and theme is deleted; upload the data again afterwards.")) return;
    try { await api("/api/reset", { method: "POST" }); setUploadMsg("The workspace was reset."); } catch (err) { setUploadMsg(err.message, true); }
    S.lastReady = null; refreshBatches(); refreshHealth();
  });
  // canvas
  document.querySelectorAll("#view-switch button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  document.querySelectorAll(".toggle[data-key]").forEach((b) => b.addEventListener("click", () => {
    b.classList.toggle("on");
    if (b.dataset.key === "region") return toggleRegions();
    applyVisibility(); highlight(S.hl);
  }));
  $("layout").addEventListener("change", () => { applyVisibility(); runLayout(); });
  $("dataset-filter").addEventListener("change", () => refreshBatches());  // the filter is part of its key
  $("fit-btn").addEventListener("click", () => {
    const gd = S.view === "sphere" && sphereDiv();
    if (gd) Plotly.relayout(gd, { "scene.camera.eye": { x: 1.1, y: 1.1, z: 1.1 }, "scene.camera.center": { x: 0, y: 0, z: 0 } });
    else if (S.cy) S.cy.fit(S.cy.elements(":visible"), 40);
  });
  document.querySelectorAll("#legend .key[data-key]").forEach((k) => {
    k.addEventListener("mouseenter", () => spotlight(k.dataset.key));
    k.addEventListener("mouseleave", () => spotlight(null));
  });
  $("legend-btn").addEventListener("click", () => setLegend($("legend").hidden));
  $("clear-btn").addEventListener("click", clearAnswer);
  $("drawer-close").addEventListener("click", closeDrawer);
  $("table-search").addEventListener("input", renderTable);
  document.querySelectorAll("#ins-table th").forEach((th) => th.addEventListener("click", () => {
    S.sort = { k: th.dataset.k, asc: S.sort.k === th.dataset.k ? !S.sort.asc : false }; renderTable();
  }));
  // what the views render, clicked: bound once on their containers, so a render only writes HTML
  const thread = $("thread");
  on(thread, ".chip-q", (c) => ask(c.textContent));
  on(thread, ".ask-cancel", (el) => turnOf(el).ctl.abort("cancel"));
  on(thread, "a.cite", (a) => reveal(a.dataset.pid));
  on(thread, ".ev", (el) => { const turn = turnOf(el), item = turn.qa.view.evidence.items[Number(el.dataset.i)]; reveal(item.pattern_id, { isolate: { turn, item, el } }); });
  on(thread, ".bot-tabs button", toggleTab);
  on(thread, ".insight-card .nb", (el) => reveal(el.dataset.id, { walk: true }));
  on(thread, ".insight-card .ic-theme", (el) => { inspect(el.dataset.id); GeoMap.openTheme(el.dataset.id); });
  on(thread, ".insight-card .ic-ask", (el) => {  // asked with its place: the cells it is drawn by on the open map, as a zone
    const p = turnOf(el).insight.node.props, cells = GeoMap.placeOf(p.id);
    ask(askAbout(p), cells.length ? { cells } : false);  // never with a zone selected for other questions
  });
  on(thread, ".insight-card .ic-graph", async (el) => {  // a filter on another dataset would hide its node
    const p = turnOf(el).insight.node.props, sel = $("dataset-filter");
    if (sel.value && sel.value !== p.dataset_id) { sel.value = ""; await refreshBatches(); }
    setView("graph"); reveal(p.id);
  });
  on($("batches"), ".ds-del", (el, e) => { e.stopImmediatePropagation(); deleteDataset(el.dataset.del, el.dataset.name); });
  on($("batches"), ".ds.clickable", (el) => { const sel = $("dataset-filter"); sel.value = sel.value === el.dataset.ds ? "" : el.dataset.ds; refreshBatches(); });
  on($("ins-table"), "tbody tr[data-id]", (tr) => inspect(tr.dataset.id));
  on($("inspector"), ".nb", (el) => {  // the drawer follows the row; on the map an insight is focused there when drawn, as reveal does
    const id = el.dataset.id, onMap = S.view === "map" && id.startsWith("P-");
    if (onMap && GeoMap.has(id)) GeoMap.focusPattern(id); else if (onMap) GeoMap.unfocus(); else focusNodes([id]);
    inspect(id, onMap && !GeoMap.has(id) ? GeoMap.offMap(id) : undefined);
  });
  on($("mw-body"), ".mw-th", (b) => { W.fold[b.dataset.id] = !b.parentElement.classList.contains("open"); renderWindow(); });
  on($("mw-body"), ".mw-i", (b) => GeoMap.pick(b.dataset.pid));
  // chat
  $("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
  $("question").addEventListener("input", autosize);
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("question").value); } });
  $("new-chat").addEventListener("click", () => { S.chat.forEach((t) => t.ctl && t.ctl.abort("new chat")); S.chat = []; clearAnswer(); renderThread(); });
  document.addEventListener("keydown", (e) => {  // Esc closes one layer a press: the cell popup, the map's focus, the drawer; a text field keeps its own
    if (e.key !== "Escape" || e.target.matches("textarea, input:not([type]), input[type=text], input[type=search]")) return;
    if (GeoMap.closePopup()) return;
    if (S.view === "map" && GeoMap.focused()) { GeoMap.unfocus(); return; }
    closeDrawer();
  });
  initResize();
  try { if (localStorage.getItem("sig-legend") === "off") setLegend(false); } catch (e) { /* private mode */ }
  renderThread(); refreshHealth(); refreshBatches();
  setInterval(refreshHealth, 30000);
  setInterval(() => tick(document), 1000);  // a running stage's and a pending question's time, between renders
}

document.addEventListener("DOMContentLoaded", init);
