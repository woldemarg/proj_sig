/* SIG UI — the datasets rail: upload, the batch cards and their polling, deletion (docs/08_interface.md §8.2). */
"use strict";

function datasetName(b) {
  /* the file and its level: the levels of one upload are datasets of one file (docs/02_discovery.md §2.9) */
  const res = /^\d+$/.test(b.level || "") ? b.level : ((b.profile || {}).geo || {}).resolution;
  return res != null ? `${b.filename} · H3 r${res}` : b.locator ? `${b.filename} · table` : b.filename;
}

function datasetCard(b, busy) {
  const running = !TERMINAL.has(b.status);
  const m = b.metrics || {}, p = b.profile || {};
  const active = b.status === "READY" && $("dataset-filter").value === b.dataset_id, name = datasetName(b);
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
  } else if (emptyLevel(b)) {  // a normal result of a level, not an error: the other levels stand
    body = `<div class="ds-note">Nothing significant at this level.</div><div class="ds-note mono">${esc(b.error.message)}</div>`;
  } else if (b.status === "FAILED" && b.error) {
    const [title, why, tip] = ERROR_HELP[b.error.code] || ["Processing failed", b.error.message, ""];
    body = `<div class="ds-error"><b>${esc(title)}</b><div class="why">${esc(why)}</div>` +
      (tip ? `<div class="tip">💡 ${esc(tip)}</div>` : "") + `<div class="why mono" style="margin-top:6px">${esc(b.error.message)}</div></div>`;
  } else if (b.status === "SKIPPED") {
    body = `<div class="ds-note">Already analysed — nothing new to add.</div>`;
  }
  const status = running ? chip("run", "Processing") : b.status === "READY" ? chip("ok", "Ready") : emptyLevel(b) ? chip("", "No findings") : b.status === "FAILED" ? chip("bad", "Failed") : chip("", "Skipped");
  const when = b.created_at ? new Date(b.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
  const del = running ? "" : `<div class="ds-foot"><button type="button" class="ds-del" data-del="${esc(b.dataset_id || b.batch_id)}" data-name="${esc(name)}" ` +
    (busy ? `disabled title="A batch is running: a deletion waits for it, so it is offered once processing ends">` : `title="Remove this dataset and its insights from the knowledge base">`) +
    `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>Delete</button></div>`;
  return `<div class="ds ${b.status === "READY" ? "clickable" : ""} ${active ? "active" : ""}" data-ds="${esc(b.status === "READY" ? b.dataset_id : "")}">` +
    `<div class="ds-head"><span class="ds-name" title="${esc(name)}">${esc(name)}</span>${status}</div>` +
    `<div class="ds-sub">${esc(when)}</div>` +
    datasetTrust(b.status === "READY" ? S.datasets[b.dataset_id] : null) +
    `${body}${del}</div>`;
}

const geoLevel = (b) => /^\d+$/.test(b.level || "");
const EMPTY_LEVEL = new Set(["invalid_schema", "no_numeric_targets", "no_candidates", "no_viable_insights"]);  // uploads.EMPTY_LEVEL
const emptyLevel = (b) => b.status === "FAILED" && b.error && EMPTY_LEVEL.has(b.error.code) && (geoLevel(b) || Boolean(b.locator));  // a level with nothing to mine
const levelRank = (b) => (geoLevel(b) ? -Number(b.level) : 99);  // finest first, the table last: the queue's order

function uploadGroups(cards) {
  /* the cards of one file's levels together (its newest card's place), in the order the levels run; a plain upload of
     the same bytes is no level of it (docs/02_discovery.md §2.9) */
  const groups = Map.groupBy(cards, (b) => ((geoLevel(b) || b.locator) && b.upload_id) || b.batch_id);
  return [...groups.values()].map((g) => g.sort((a, b) => levelRank(a) - levelRank(b)));
}

function uploadGroup(cards, busy) {
  if (cards.length === 1) return datasetCard(cards[0], busy);
  const ready = cards.filter((b) => b.status === "READY").length, empty = cards.filter(emptyLevel).length;
  const failed = cards.filter((b) => b.status === "FAILED").length - empty;
  return `<div class="ds-group" data-upload="${esc(cards[0].upload_id)}"><div class="ds-group-head">${esc(cards[0].filename)} · ${ready} of ${cards.length} levels ready` +
    (empty ? ` · ${empty} without findings` : "") + (failed ? ` · <b>${failed} failed</b>, the others stand on their own` : "") + `</div>${cards.map((b) => datasetCard(b, busy)).join("")}</div>`;
}

function datasetTrust(t) {
  /* a dataset card's trust chips, from its Dataset node */
  if (!t) return "";
  if (t.integrity === "mismatch") return `<div class="ds-src">${chip("bad", "metadata failed its check", "The batch record no longer matches its metadata digest: its source, reliability and indicators are ignored")}</div>`;
  const chips = (t.source || t.reliability !== "F" ? chip("", `${t.source || "source"} · ${t.reliability}`, `Source and its reliability (Admiralty code): ${RELIABILITY[t.reliability] || ""}`) : "") +
    (t.integrity === "unverified" ? chip("warn", "metadata unverified", "Recorded before metadata digests: its source and rating are not checked") : "");
  return chips ? `<div class="ds-src">${chips}</div>` : "";
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
  try { batches = await api("/api/batches"); } catch (e) { clearTimeout(S.poll); S.poll = setTimeout(refreshBatches, 3000); return; }
  // one card per dataset (newest first): its READY batch if it has one, otherwise its latest attempt
  const byDataset = new Map();
  for (const b of batches) {
    const key = b.dataset_id || b.batch_id, kept = byDataset.get(key);
    if (!kept || (b.status === "READY" && kept.status !== "READY")) byDataset.set(key, b);
  }
  const cards = [...byDataset.values()];
  const ready = [...new Map(batches.filter((b) => b.status === "READY").map((b) => [b.dataset_id, b])).values()];
  const sel = $("dataset-filter"), cur = sel.value;
  sel.innerHTML = `<option value="">All datasets</option>` +
    ready.map((b) => `<option value="${esc(b.dataset_id)}" data-file="${esc(b.filename)}">${esc(datasetName(b))}</option>`).join("");
  sel.value = ready.some((b) => b.dataset_id === cur) ? cur : "";
  // one key for what every view follows: the READY batches and the dataset filter
  const viewKey = batches.filter((b) => b.status === "READY").map((b) => b.batch_id).join(",") + "|" + sel.value;
  let map;
  if (viewKey !== S.lastReady) {
    S.lastReady = viewKey;
    map = GeoMap.refresh().catch((e) => console.error("map refresh failed", e));  // the map never breaks the rail or an answer
    await loadGraph(); refreshHealth();
    if (!S.chat.length) renderThread();  // the welcome card's questions follow the datasets and the filter
  }
  const busy = batches.some((b) => !TERMINAL.has(b.status));
  $("batches").innerHTML = cards.length ? uploadGroups(cards).map((g) => uploadGroup(g, busy)).join("") : `<div class="empty-rail">No datasets yet.<br>Upload a table or try a demo.</div>`;  // after the graph: the cards read its trust
  clearTimeout(S.poll);
  if (busy) S.poll = setTimeout(refreshBatches, 900);
  return map;  // a caller that acts on the map afterwards (ask) awaits its load too; the poll does not
}

function setUploadMsg(text, err = false) { const el = $("upload-msg"); el.className = "msg" + (err ? " err" : ""); el.textContent = text; }

async function upload() {
  const f = $("file").files[0]; if (!f) return;
  const fd = new FormData();
  fd.append("file", f); fd.append("bins", $("bins").value); fd.append("categories", $("categories").value); fd.append("geo", $("geo").value);
  if ($("geo").value.trim() && $("table").checked) fd.append("table", "true");
  fd.append("source", $("source").value); fd.append("reliability", $("reliability").value); fd.append("indicators", $("indicators").value);
  setUploadMsg(`Uploading ${f.name}…`);
  $("upload-btn").disabled = true;
  try { await api("/api/upload", { method: "POST", body: fd }); setUploadMsg("Queued — progress is shown below."); $("file").value = ""; $("file-name").textContent = ""; refreshBatches(); }
  catch (err) { setUploadMsg(err.message, true); $("upload-btn").disabled = false; }
}
