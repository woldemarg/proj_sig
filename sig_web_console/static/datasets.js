/* SIG UI — the data rail: the datasets with their layers as a tree, upload, polling, deletion (docs/08_interface.md §8.2). */
"use strict";

const stem = (file) => String(file || "").replace(/\.[^.]+$/, "");
function titleOf(file) {
  /* a shipped set's title, else the file's name without its extension */
  const d = Object.values(S.samples || {}).find((x) => stem(x.file) === stem(file));
  return d ? d.title.replace(/\s*\(контекстний шар\)$/, "") : stem(file);
}
const geoLevel = (b) => /^\d+$/.test(b.level || "");
function levelOf(b) {  // the level a dataset reads its upload at: the levels of one upload are datasets of one file (docs/02_discovery.md §2.9)
  const res = geoLevel(b) ? b.level : ((b.profile || {}).geo || {}).resolution;
  return b.status === "LAYER" ? "шар" : res != null ? `H3 r${res}` : b.locator ? "таблиця" : "";
}
function datasetName(b) {  // its title, its filter, the layers beside it
  return titleOf(b.filename) + (b.where ? ` · ${b.where}` : "") + (b.context ? ` + ${b.context.map((c) => titleOf(c.name)).join(", ")}` : "");
}

const PRUNED_SHOWN = ["context_confounded", "context_family", "outside_cross_type"];  // the layer's verdict; the other reasons stay in the total
const prunedBy = (pruned = {}) => {
  const shown = PRUNED_SHOWN.filter((k) => pruned[k]).map((k) => `${term(k)}: ${num(pruned[k], 0)}`);
  return shown.length ? ` (${shown.join(", ")})` : "";
};

const timeNotes = (b) => (b.time_words || []).map((w) => `<div class="ds-note">⏱ ${esc(w)}</div>`).join("");  // an event layer's time, in words
const warningsOf = (b) =>  // the record's warnings but an event layer's own, which ``timeNotes`` words
  (b.warnings || []).filter((w) => !(b.time_words || []).length || !(b.context || []).some((c) => w.startsWith(`${c.name}: an event layer`)));
const refusals = (b) => (b.cross && b.dataset_id ? `<details class="ds-refused" data-rej="${esc(b.dataset_id)}"><summary>${esc(word("context_layer"))}: які когорти відсіяно</summary><div class="hint">…</div></details>` : "");

async function showRefusals(box) {
  /* a run's refused layer cohorts in words (``GET /api/datasets/{id}/rejections``): the metric, where, why */
  const rows = (await api(`/api/datasets/${encodeURIComponent(box.dataset.rej)}/rejections`)).filter((r) => ["context_confounded", "context_family"].includes(r.reason));
  box.querySelector(".hint").innerHTML = rows.length
    ? rows.map((r) => `<div>${esc(r.metric_words || r.metric || "")} — ${esc(r.where)}: <i>${esc(word(r.reason))}</i></div>`).join("")
    : "немає";
}

const staleChip = (b) =>  // a run whose layer's upload is gone (uploads.cross_state, read off the registry)
  b.stale ? chip("bad", b.stale.reason === "superseded" ? "шар замінено" : "шар видалено", "Завантаження шару цього прогону вже немає; прогін тримає копію шару й лишається відтворюваним. Новий шар — новий прогін.") : "";

const ICON = {
  eye: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
  eyeOff: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.2 4.1M6.6 6.6C3.8 8.4 2 12 2 12s3.5 7 10 7c1.9 0 3.6-.6 5-1.5M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>',
  bin: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
  layer: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3 2 8l10 5 10-5-10-5z"/><path d="m2 13 10 5 10-5"/></svg>',
};

const more = (b, html) =>  // a row's record for the developer, folded: the tree stays one line a dataset
  `<details class="ds-more" data-dev data-b="${esc(b.batch_id)}"><summary>подробиці</summary>${html}</details>`;

function rowBody(b, neutral) {
  /* under a row's head: progress, an error, a verdict for everyone; the record's details for the developer */
  const m = b.metrics || {}, p = b.profile || {};
  const refused = m.pruned_total ? `<div class="ds-note">відсіяно: ${num(m.pruned_total, 0)}${prunedBy(m.pruned)}</div>` : "";  // a level that admitted none
  const when = b.created_at ? new Date(b.created_at).toLocaleString("uk", { dateStyle: "medium", timeStyle: "short" }) : "";
  if (!TERMINAL.has(b.status)) {
    const i = Math.max(0, STAGES.indexOf(b.status));
    return `<div class="progress"><i style="width:${Math.round(((i + 1) / STAGES.length) * 100)}%"></i></div>` +
      `<div class="ds-step">${esc(STAGE_TEXT[b.status] || b.status)}… <span class="muted">крок ${i + 1} з ${STAGES.length}` +
      ((b.stage_times || {})[b.status] ? ` · <span class="ds-elapsed" data-since="${esc(b.stage_times[b.status])}"></span>` : "") + `</span></div>`;
  }
  if (b.status === "READY") {
    return more(b, `<div class="ds-sub">${esc(when)}</div>` + datasetTrust(S.datasets[b.dataset_id]) +
      `<div class="ds-note">${plural(m.attractors_new ?? m.attractors_total, "нова тема", "нові теми", "нових тем")} · ${plural(m.input_rows, "рядок", "рядки", "рядків")} · ` +
      `${plural(p.columns ?? (p.numerics || []).length + (p.categoricals || []).length, "колонка", "колонки", "колонок")} · перевірено сегментів: ${num(m.candidate_patterns, 0)} · ` +
      `<a href="/api/datasets/${esc(b.dataset_id)}/rejections" target="_blank" rel="noopener" title="Список відсіяних з причинами">відсіяно: ${num(m.pruned_total, 0)}</a>${prunedBy(m.pruned)} · ${num(m.processing_duration_s, 1)} с</div>` +
      timeNotes(b) + warningsOf(b).map((w) => `<div class="ds-note">⚠ ${esc(w)}</div>`).join("") + refusals(b) +
      (p.selected_dimensions && p.selected_dimensions.length ? `<div class="ds-dims">${p.selected_dimensions.map((d) => `<span class="tag">${esc((b.dimension_words || {})[d] || human(d))}</span>`).join("")}</div>` : ""));
  }
  if (neutral) return `<div class="ds-note">${b.cross ? "Знахідок з умовою на шар немає." : "На цьому рівні стійких знахідок немає."}</div>${more(b, `${refused}${refusals(b)}${timeNotes(b)}<div class="ds-note mono">${esc(b.error.message)}</div>`)}`;  // the codes' dict: the developer's
  if (b.status === "FAILED" && b.error) {
    const [title, why, tip] = ERROR_HELP[b.error.code] || ["Обробка не вдалася", b.error.message, ""];
    return `<div class="ds-error"><b>${esc(title)}</b><div class="why">${esc(why)}</div>` + (tip ? `<div class="tip">💡 ${esc(tip)}</div>` : "") +
      `<div class="why mono" data-dev style="margin-top:6px">${esc(b.error.message)}</div></div><div data-dev>${refused}</div>`;
  }
  return "";
}

function dsRow(b, label, busy, { neutral = false, layer = null } = {}) {
  /* one dataset of the tree: a click shows it alone; a layer's row hides or shows its layer (PATCH /api/uploads/{id}) */
  const ready = b.status === "READY", running = !TERMINAL.has(b.status), active = ready && $("dataset-filter").value === b.dataset_id;
  const off = !!layer && (b.status === "LAYER" ? !!b.hidden : (b.hidden_by || []).includes(layer));
  const m = b.metrics || {};
  const state = running ? chip("run", "Обробка") : ready ? `<span class="ds-count">${plural(m.validated_insights ?? 0, "інсайт", "інсайти", "інсайтів")}</span>`
    : neutral ? chip("", "Без знахідок")
    : b.status === "FAILED" ? chip("bad", "Помилка") : b.status === "LAYER" ? `<span class="ds-count">${plural(b.points, "точка", "точки", "точок")}</span>` : chip("", "Пропущено", "Уже проаналізовано — нового немає.");
  const name = ready ? `<button type="button" class="ds-name ds-pick" title="${esc(label)}" aria-pressed="${active}">${esc(label)}</button>` : `<span class="ds-name" title="${esc(label)}">${esc(label)}</span>`;
  const level = !layer && levelOf(b) ? `<span class="ds-lvl" data-dev>${esc(levelOf(b))}</span>` : "";
  const hide = layer ? `<button type="button" class="ds-hide ds-icon" data-hide="${esc(layer)}" data-on="${off ? "" : "1"}"${busy ? " disabled" : ""} ` +
    `aria-label="${off ? "Показати шар" : "Сховати шар"}" title="${busy ? "Іде обробка: зачекайте" : off ? "Показати шар" : "Сховати шар (з усіх датасетів, де він є)"}">${off ? ICON.eyeOff : ICON.eye}</button>` : "";
  const del = running ? "" : `<button type="button" class="ds-del ds-icon" data-del="${esc(b.dataset_id || b.batch_id)}" data-name="${esc(b.cross ? label : datasetName(b))}"${b.cross ? ` data-layer="${esc(layer)}"` : ""} aria-label="Видалити" ` +
    `title="${busy ? "Іде обробка: видалення відхиляється, доки вона не скінчиться" : layer ? "Прибрати шар із цього датасету" : "Прибрати з бази знань"}"${busy ? " disabled" : ""}>${ICON.bin}</button>`;
  return `<div class="ds${ready ? " clickable" : ""}${active ? " active" : ""}${layer ? " layer" : ""}${off ? " off" : ""}" data-ds="${esc(ready ? b.dataset_id : "")}">` +
    `<div class="ds-head">${layer ? `<span class="ds-glyph">${ICON.layer}</span>` : ""}${name}${level}${staleChip(b)}${off ? chip("", "приховано") : ""}${state}${hide}${del}</div>` +
    `${rowBody(b, neutral)}</div>`;
}

const EMPTY_LEVEL = new Set(["invalid_schema", "no_numeric_targets", "no_candidates", "no_viable_insights"]);  // uploads.EMPTY_LEVEL
const emptyLevel = (b) => b.status === "FAILED" && b.error && EMPTY_LEVEL.has(b.error.code) && geoLevel(b);  // a geo level with nothing to mine
const verdict = (b) => emptyLevel(b) && !!b.cross;  // a run between stored uploads that found nothing: the pair's result, not an error
const levelRank = (b) => (geoLevel(b) ? -Number(b.level) : 99);  // finest first, the table last: the queue's order

function uploadGroups(cards) {
  /* the datasets of one upload together (its newest card's place): its levels, then its layer runs; a plain upload of
     the same bytes is no level of it (docs/02_discovery.md §2.9) */
  const groups = Map.groupBy(cards, (b) => ((geoLevel(b) || b.locator) && b.upload_id) || b.batch_id);
  return [...groups.values()].map((g) => g.sort((a, b) => levelRank(a) - levelRank(b)));
}

function uploadGroup(cards, busy) {
  const levels = cards.filter((b) => !b.cross), runs = cards.filter((b) => b.cross);
  const ready = levels.filter((b) => b.status === "READY").length, neutral = (b) => ready > 0 && emptyLevel(b);
  const lead = levels[0] || runs[0];
  let head;
  if (levels.length === 1) head = dsRow(levels[0], datasetName(levels[0]), busy);
  else {
    head = `<div class="ds-group-head"><span class="ds-name">${esc(titleOf(lead.filename) + (lead.where ? ` · ${lead.where}` : ""))}</span></div>` +
      levels.map((b) => dsRow(b, geoLevel(b) ? `мапа · рівень ${b.level}` : "таблиця", busy, { neutral: neutral(b) })).join("");
  }
  const kids = runs.map((r) => dsRow(r, r.context.map((c) => titleOf(c.name)).join(", "), busy, { neutral: verdict(r), layer: r.cross[0] })).join("");
  const host = levels.find((b) => b.linkable);
  const add = host ? `<button type="button" class="ds-link" data-link="${esc(host.batch_id)}">+ Додати шар</button>` : "";  // uploads.cross_state
  return `<div class="ds-group" data-upload="${esc(lead.upload_id)}">${head}${kids || add ? `<div class="ds-kids">${kids}${add}</div>` : ""}</div>`;
}

function renderTree(cards, busy) {
  /* the uploads with their layer runs; a layer no run reads stands apart, a layer that runs read lives under them */
  const loose = cards.filter((b) => b.status === "LAYER" && !b.layer_of);
  const groups = uploadGroups(cards.filter((b) => b.status !== "LAYER")).map((g) => uploadGroup(g, busy)).join("");
  return groups + (loose.length ? `<div class="ds-group ds-loose"><div class="ds-group-head"><span class="ds-name">Шари без датасету</span></div>` +
    loose.map((b) => dsRow(b, titleOf(b.filename), busy, { layer: b.hidden ? b.upload_id : null })).join("") + `</div>` : "");
}

function datasetTrust(t) {
  /* a dataset's trust chips, from its Dataset node */
  if (!t) return "";
  if (t.integrity === "mismatch") return `<div class="ds-src">${chip("bad", "метадані не пройшли перевірку", `Запис датасету більше не збігається з відбитком своїх метаданих: його ${DECLARED} ігноруються`)}</div>`;
  const chips = (t.source || t.reliability !== "F" ? chip("", `${t.source || "джерело"} · ${t.reliability}`, `${word("reliability")} (шкала Адміралтейства): ${word(`reliability_${t.reliability}`)}`) : "") +
    (t.integrity === "unverified" ? chip("warn", "метадані не перевірено", "Записано до появи відбитків метаданих: джерело й оцінку не перевірено") : "") +
    (t.glossary_columns ? chip("", `глосарій · ${plural(t.glossary_columns, "колонка", "колонки", "колонок")}`, "Заявлені назви колонок і категорій: так їх читають інсайти й відповіді") : "") +
    (t.metadata_columns ? chip("", `документ · ${plural(t.metadata_columns, "колонка", "колонки", "колонок")}`, t.about || "Документ метаданих: довідка про колонки для відповідей") : "");
  return chips ? `<div class="ds-src">${chips}</div>` : "";
}

async function deleteDataset(id, name, layer = "") {
  /* a dataset out of the knowledge base; a layer's run (``layer``) takes its layer along when no other dataset reads it */
  const ask = layer ? `Прибрати шар «${name}»?\n\nЙого знахідки зникнуть; сам шар — теж, якщо його не використовує інший датасет.`
    : `Видалити «${name}» з бази знань?\n\nЙого інсайти та їхня належність до тем зникнуть; теми, що лишаться без інсайтів і без зв'язків, — теж.`;
  if (!confirm(ask)) return;
  const kept = layer && S.batches.find((b) => b.upload_id === layer && b.status === "LAYER");  // a layer-only upload, never another dataset
  const shared = S.batches.some((b) => (b.cross || []).includes(layer) && b.dataset_id !== id && b.status !== "SKIPPED");
  try {
    await api(`/api/datasets/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (kept && !shared) await api(`/api/datasets/${encodeURIComponent(kept.batch_id)}`, { method: "DELETE" });
    setUploadMsg("");
  } catch (err) { setUploadMsg(err.message, true); }
  if ($("dataset-filter").value === id) $("dataset-filter").value = "";
  S.lastReady = null;  // force the views to reload
  refreshBatches();
}

const elapsed = (iso) => { const s = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000)); return s < 60 ? `${s} с` : `${Math.floor(s / 60)} хв ${s % 60} с`; };
const tick = (root) => root.querySelectorAll("[data-since]").forEach((el) => { el.textContent = elapsed(el.dataset.since); });

function swapChanged(el, to) {  // a group swaps its changed rows one by one
  if (el.isEqualNode(to)) return;
  if ((el.matches(".ds-group") && el.dataset.upload === to.dataset.upload || el.matches(".ds-kids")) && el.children.length === to.children.length) {
    const fresh = [...to.children];
    [...el.children].forEach((c, i) => swapChanged(c, fresh[i]));
  } else el.replaceWith(to);
}

function renderRail(html) {
  /* a row whose markup is unchanged keeps its nodes: a click on it survives the poll */
  const box = $("batches"), next = document.createElement("div");
  next.innerHTML = html;
  tick(box); tick(next);  // the same clock on both sides: a running row's time is no change
  box.querySelectorAll(".ds-more[open]").forEach((d) => next.querySelector(`.ds-more[data-b="${CSS.escape(d.dataset.b)}"]`)?.setAttribute("open", ""));  // an opened record stays open
  const fresh = [...next.children], picked = box.contains(document.activeElement) && document.activeElement.closest(".ds")?.dataset.ds;
  if (box.children.length !== fresh.length) box.replaceChildren(...fresh);
  else [...box.children].forEach((el, i) => swapChanged(el, fresh[i]));
  if (picked && !box.contains(document.activeElement)) box.querySelector(`.ds[data-ds="${CSS.escape(picked)}"] .ds-pick`)?.focus();  // a replaced row keeps the focus
}

async function refreshBatches() {
  let batches = [];
  try { batches = await api("/api/batches"); } catch (e) { clearTimeout(S.poll); S.poll = setTimeout(refreshBatches, 3000); return; }
  S.batches = batches;  // the stored layers the link dialog offers
  S.hidden = new Set(batches.filter((b) => b.hidden_by).map((b) => b.dataset_id));  // a hidden layer's runs: left out of every view
  if (S.linkRun && batches.find((b) => b.batch_id === S.linkRun)?.status === "SKIPPED") { S.linkRun = null; setUploadMsg("Такий прогін уже є — нового немає."); }  // a repeat, known once read
  // one card per dataset (newest first): its READY batch if it has one, otherwise its latest attempt
  const byDataset = new Map();
  for (const b of batches) {
    const key = b.dataset_id || b.batch_id, kept = byDataset.get(key);
    if (!kept || (b.status === "READY" && kept.status !== "READY")) byDataset.set(key, b);
  }
  const cards = [...byDataset.values()];
  const ready = [...new Map(batches.filter((b) => b.status === "READY").map((b) => [b.dataset_id, b])).values()];
  const sel = $("dataset-filter"), cur = sel.value, options = `<option value="">Усі датасети</option>` +
    ready.map((b) => `<option value="${esc(b.dataset_id)}" data-file="${esc(b.filename)}">${esc(datasetName(b) + (levelOf(b) ? ` · ${levelOf(b)}` : ""))}</option>`).join("");
  if (options !== S.options) {  // rewritten only when the READY set changes: an open list stays open
    S.options = sel.innerHTML = options;
    sel.value = ready.some((b) => b.dataset_id === cur) ? cur : "";
  }
  // one key for what every view follows: the READY batches and the dataset filter
  const viewKey = batches.filter((b) => b.status === "READY").map((b) => `${b.batch_id}:${b.metadata_digest || ""}${b.hidden_by ? ":off" : ""}`).join(",") + "|" + sel.value;  // a PATCH re-signs its record
  let map;
  if (viewKey !== S.lastReady) {
    S.lastReady = viewKey;
    map = GeoMap.refresh().catch((e) => console.error("map refresh failed", e));  // the map never breaks the rail or an answer
    await loadGraph(); refreshHealth();
    if (!S.chat.length) renderThread();  // the welcome card's questions follow the datasets and the filter
  }
  const busy = batches.some((b) => !TERMINAL.has(b.status));
  renderRail(cards.length ? renderTree(cards, busy) : `<div class="empty-rail">Даних ще немає.<br>Натисніть «+ Додати».</div>`);  // after the graph: the rows read its trust
  clearTimeout(S.poll);
  if (busy) S.poll = setTimeout(refreshBatches, 2500);
  // a caller that acts on the map afterwards (ask) awaits its load too, or the pass a poll began for this key; the poll does not
  return map || GeoMap.refreshing()?.catch((e) => console.error("map refresh failed", e));
}

function setUploadMsg(text, err = false) { const el = $("upload-msg"); el.className = "msg" + (err ? " err" : ""); el.textContent = text; }
function setUploadNote(text, err = false) { const el = $("upload-note"); el.className = "msg" + (err ? " err" : ""); el.textContent = text; }

function syncTable() {  // a table's geo is held, not sent: its field disabled
  /* the base level is one of a geo upload's — none for a shipped set whose manifest says it has no number of its own
     beyond the point ("table": false): without cells it has nothing to measure, so the box cannot be ticked */
  const d = (S.samples || {})[$("sample").value], none = !!d && (d.options || {}).table === false;
  $("table").disabled = $("geo").disabled || !$("geo").value.trim() || none;
  if (none) $("table").checked = false;
  $("table").title = none ? "У цього набору немає власних чисел поза координатами: без комірок він нічого не виміряє" : "";
}

const sampleItem = (d) => `<option value="${esc(d.key)}">${esc(d.title)} · ${num(d.rows, 0)} рядків</option>`;

// the upload's role (docs/08_interface.md §8.2): proposed from a shipped set's manifest or a file's header, confirmed or changed
const POINT_COLUMNS = /^(lat|latitude|lon|lng|long|longitude|geometry|geom|point|location|wkt|coordinates)$/i;

async function fileRole(f) {  // events when the header names a coordinate column, else a table; a Parquet header is not read here
  if (/\.parquet$/i.test(f.name)) return "table";
  const head = (await f.slice(0, 65536).text()).split(/\r?\n/)[0];
  return head.split(/[\t;,]/).some((n) => POINT_COLUMNS.test(n.trim().replace(/^"|"$/g, ""))) ? "events" : "table";
}

function setRole(r, layerOnly = false) {
  /* the role's radio, the geo option it needs and the button's words; a set meant only as a layer mines nothing */
  $("roles").disabled = !r;
  for (const v of ["events", "table", "layer"]) { const el = $(`role-${v}`); el.checked = v === r; el.disabled = layerOnly && v !== "layer"; }
  $("geo").disabled = r === "table";  // held for a switch back: a shipped set keeps its level
  $("count-box").hidden = r !== "events";  // per-type counters: the map's cells only
  if (r && r !== "table" && !$("geo").value.trim()) $("geo").value = r === "layer" ? "auto" : "auto@auto";
  $("upload-btn").textContent = r === "layer" ? "Зберегти шар" : "Аналізувати";
  syncTable();
}

// *+ Додати шар* (docs/08_interface.md §8.2): one layer, one run of the base layer's finest level — the API's two calls
function openLink(batchId) {
  /* the layers this upload can take: the stored ones, then the shipped layer sets not stored yet (every geo set for the developer) */
  const b = S.batches.find((x) => x.batch_id === batchId);  // a row's button: rendered from the batches stored with it
  const linked = new Set(S.batches.filter((x) => x.upload_id === b.upload_id && x.cross && x.status !== "SKIPPED").flatMap((x) => x.cross));
  const stored = new Map(S.batches.filter((x) => x.upload_id !== b.upload_id && !x.cross && (x.status === "LAYER" || (x.status === "READY" && geoLevel(x)))).map((x) => [x.upload_id, x]));
  const kept = new Set([...stored.values()].map((x) => stem(x.filename)));
  const opt = (attr, value, label, note, done = false, devOnly = false) =>
    `<button type="button" class="link-opt" data-${attr}="${esc(value)}"${devOnly ? " data-dev" : ""}>${esc(label)}<span class="muted">${esc(done ? "уже додано" : note)}</span></button>`;  // a repeat ends SKIPPED
  $("link-dialog").dataset.batch = batchId;
  $("link-dialog").dataset.target = b.dataset_id || b.upload_id;  // the reading the row stands for
  $("link-title").textContent = `Додати шар · ${datasetName(b)}`;
  $("link-form").reset(); $("link-msg").textContent = "";
  $("link-opts").innerHTML =
    [...stored.values()].map((x) => opt("layer", x.upload_id, titleOf(x.filename) + (x.where ? ` · ${x.where}` : ""), x.status === "LAYER" ? plural(x.points, "точка", "точки", "точок") : "датасет", linked.has(x.upload_id))).join("") +
    Object.values(S.samples || {}).filter((d) => (d.options || {}).geo && !kept.has(stem(d.file)))
      .map((d) => opt("sample", d.key, titleOf(d.file), `${num(d.rows, 0)} точок`, false, !d.context_only)).join("");
  $("link-counts").value = (b.count_values || []).filter((c) => c.status === "ok").map((c) => `${c.column}=${c.value}`).join(", ");  // the base's, to confirm or change
  $("link-dialog").showModal();
}

async function runLink(src) {
  /* ``src``: a stored upload (layer), a shipped set (sample) or a file; one not stored is kept as *Лише шар* first */
  const o = src.sample ? S.samples[src.sample].options || {} : {}, batchId = $("link-dialog").dataset.batch;
  $("link-dialog").querySelectorAll(".link-opt").forEach((el) => { el.disabled = true; });  // one click, one run
  $("link-msg").className = "msg"; $("link-msg").textContent = "Запускаю…";
  try {
    let layer = src.layer;
    if (!layer) {
      const fd = new FormData();
      if (src.file) fd.append("file", src.file); else fd.append("demo", src.sample);
      fd.append("layer", "true"); fd.append("geo", o.geo || "auto"); fd.append("time", o.time || ""); fd.append("tz", o.tz || "");
      layer = (await api("/api/upload", { method: "POST", body: fd })).upload_id;
    }
    const body = { target: $("link-dialog").dataset.target, layers: [layer], ring: $("link-ring").value, count_values: $("link-counts").value };  // empty: no types
    const run = await api("/api/crosstype", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("link-dialog").close();
    S.linkRun = run.batch_id;
    refreshBatches();
  } catch (err) {
    await refreshBatches(); openLink(batchId);  // a layer kept before the failure is offered as stored
    $("link-msg").className = "msg err"; $("link-msg").textContent = err.message;
  }
}

async function hideLayer(upload, on) {  // PATCH /api/uploads/{id}: the server holds the state, every view reads it back
  try { await api(`/api/uploads/${encodeURIComponent(upload)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ hidden: on }) }); }
  catch (err) { setUploadMsg(err.message, true); }
  refreshBatches();
}

async function upload() {
  const f = $("file").files[0], sample = $("sample").value; if (!f && !sample) return;
  const fd = new FormData(), geo = $("geo").disabled ? "" : $("geo").value;
  if (f) fd.append("file", f); else fd.append("demo", sample);  // a shipped dataset: the server reads its own copy
  fd.append("bins", $("bins").value); fd.append("categories", $("categories").value); fd.append("geo", geo);
  if ($("time").value.trim()) { fd.append("time", $("time").value); fd.append("tz", $("tz").value); }  // a layer's too: its span, for a run's time rule
  if ($("role-layer").checked) fd.append("layer", "true");  // kept as another type's points: none of the mining options
  else {
    if (geo.trim() && $("table").checked) fd.append("table", "true");
    if ($("time").value.trim()) fd.append("grid", $("grid").value);
    if ($("where").value.trim()) fd.append("where", $("where").value);  // its own reading: the rows holding one value
    if (geo.trim()) { fd.append("count_values", $("count-values").value); fd.append("baseline", $("baseline").value); }  // the geo cells'
  }
  fd.append("source", $("source").value); fd.append("reliability", $("reliability").value); fd.append("indicators", $("indicators").value);
  fd.append("glossary", $("glossary").value);
  if ($("metadata").files[0]) fd.append("metadata", $("metadata").files[0]);
  setUploadNote(`Завантажую ${f ? f.name : $("file-name").textContent}…`);
  $("upload-btn").disabled = true;
  try {
    await api("/api/upload", { method: "POST", body: fd }); setUploadNote(""); $("upload-dialog").close();
    document.querySelector("#upload-form details.adv").open = false;  // sent: the options fold away
    $("file").value = ""; $("sample").value = ""; $("sample-hint").textContent = ""; $("metadata").value = ""; $("metadata-name").textContent = ""; $("where").value = ""; $("count-values").value = ""; $("baseline").value = ""; setRole(""); $("file-name").textContent = ""; refreshBatches();
  }
  catch (err) { setUploadNote(err.message, true); $("upload-btn").disabled = false; }
}
