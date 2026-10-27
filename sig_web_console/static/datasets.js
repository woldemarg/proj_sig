/* SIG UI — the datasets rail: upload, the batch cards and their polling, deletion (docs/08_interface.md §8.2). */
"use strict";

function datasetName(b) {
  /* the file and its level: the levels of one upload are datasets of one file (docs/02_discovery.md §2.9) */
  const res = /^\d+$/.test(b.level || "") ? b.level : ((b.profile || {}).geo || {}).resolution;
  return res != null ? `${b.filename} · H3 r${res}` : b.locator ? `${b.filename} · таблиця` : b.filename;
}

function datasetCard(b, busy, neutral = false) {  // neutral: an empty level beside a READY one
  const running = !TERMINAL.has(b.status);
  const m = b.metrics || {}, p = b.profile || {};
  const active = b.status === "READY" && $("dataset-filter").value === b.dataset_id, name = datasetName(b);
  let body = "";
  if (running) {
    const i = Math.max(0, STAGES.indexOf(b.status));
    body = `<div class="progress"><i style="width:${Math.round(((i + 1) / STAGES.length) * 100)}%"></i></div>` +
      `<div class="ds-step">${esc(STAGE_TEXT[b.status] || b.status)}… <span class="muted">крок ${i + 1} з ${STAGES.length}` +
      ((b.stage_times || {})[b.status] ? ` · <span class="ds-elapsed" data-since="${esc(b.stage_times[b.status])}"></span>` : "") + `</span></div>`;
  } else if (b.status === "READY") {
    body = `<div class="ds-stats">` +
      `<div class="stat"><b>${num(m.validated_insights, 0)}</b><span>${form(m.validated_insights, "інсайт", "інсайти", "інсайтів")}</span></div>` +
      `<div class="stat" title="Теми, вперше знайдені в цьому датасеті"><b>${num(m.attractors_new ?? m.attractors_total, 0)}</b><span>${form(m.attractors_new ?? m.attractors_total, "тема", "теми", "тем")}</span></div>` +
      `<div class="stat"><b>${num(m.input_rows, 0)}</b><span>${form(m.input_rows, "рядок", "рядки", "рядків")}</span></div></div>` +
      `<div class="ds-note">${plural(p.columns ?? (p.numerics || []).length + (p.categoricals || []).length, "колонка", "колонки", "колонок")} · перевірено сегментів: ${num(m.candidate_patterns, 0)} · відсіяно: ${num(m.pruned_total, 0)} · ${num(m.processing_duration_s, 1)} с</div>` +
      (p.selected_dimensions && p.selected_dimensions.length ? `<div class="ds-dims">${p.selected_dimensions.map((d) => `<span class="tag">${esc(human(d))}</span>`).join("")}</div>` : "");
  } else if (neutral) {  // a normal result of a level, not an error: the other levels stand
    body = `<div class="ds-note">На цьому рівні стійких знахідок немає.</div><div class="ds-note mono">${esc(b.error.message)}</div>`;
  } else if (b.status === "FAILED" && b.error) {
    const [title, why, tip] = ERROR_HELP[b.error.code] || ["Обробка не вдалася", b.error.message, ""];
    body = `<div class="ds-error"><b>${esc(title)}</b><div class="why">${esc(why)}</div>` +
      (tip ? `<div class="tip">💡 ${esc(tip)}</div>` : "") + `<div class="why mono" style="margin-top:6px">${esc(b.error.message)}</div></div>`;
  } else if (b.status === "SKIPPED") {
    body = `<div class="ds-note">Уже проаналізовано — нового немає.</div>`;
  }
  const status = running ? chip("run", "Обробка") : b.status === "READY" ? chip("ok", "Готово") : neutral ? chip("", "Без знахідок") : b.status === "FAILED" ? chip("bad", "Помилка") : chip("", "Пропущено");
  const when = b.created_at ? new Date(b.created_at).toLocaleString("uk", { dateStyle: "medium", timeStyle: "short" }) : "";
  const del = running ? "" : `<div class="ds-foot"><button type="button" class="ds-del" data-del="${esc(b.dataset_id || b.batch_id)}" data-name="${esc(name)}" ` +
    (busy ? `disabled title="Іде обробка: видалення відхиляється, доки вона не скінчиться">` : `title="Прибрати цей датасет і його інсайти з бази знань">`) +
    `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>Видалити</button></div>`;
  return `<div class="ds ${b.status === "READY" ? "clickable" : ""} ${active ? "active" : ""}" data-ds="${esc(b.status === "READY" ? b.dataset_id : "")}">` +
    `<div class="ds-head">${b.status === "READY" ? `<button type="button" class="ds-name ds-pick" title="${esc(name)}" aria-pressed="${active}">${esc(name)}</button>` : `<span class="ds-name" title="${esc(name)}">${esc(name)}</span>`}${status}</div>` +
    `<div class="ds-sub">${esc(when)}</div>` +
    datasetTrust(b.status === "READY" ? S.datasets[b.dataset_id] : null) +
    `${body}${del}</div>`;
}

const geoLevel = (b) => /^\d+$/.test(b.level || "");
const EMPTY_LEVEL = new Set(["invalid_schema", "no_numeric_targets", "no_candidates", "no_viable_insights"]);  // uploads.EMPTY_LEVEL
const emptyLevel = (b) => b.status === "FAILED" && b.error && EMPTY_LEVEL.has(b.error.code) && geoLevel(b);  // a geo level with nothing to mine
const levelRank = (b) => (geoLevel(b) ? -Number(b.level) : 99);  // finest first, the table last: the queue's order

function uploadGroups(cards) {
  /* the cards of one file's levels together (its newest card's place), in the order the levels run; a plain upload of
     the same bytes is no level of it (docs/02_discovery.md §2.9) */
  const groups = Map.groupBy(cards, (b) => ((geoLevel(b) || b.locator) && b.upload_id) || b.batch_id);
  return [...groups.values()].map((g) => g.sort((a, b) => levelRank(a) - levelRank(b)));
}

function uploadGroup(cards, busy) {
  if (cards.length === 1) return datasetCard(cards[0], busy);  // a level alone that finds nothing failed, as its upload did
  const ready = cards.filter((b) => b.status === "READY").length, empty = ready ? cards.filter(emptyLevel).length : 0;
  const failed = cards.filter((b) => b.status === "FAILED").length - empty;
  return `<div class="ds-group" data-upload="${esc(cards[0].upload_id)}"><div class="ds-group-head">${esc(cards[0].filename)} · готово рівнів: ${ready} з ${cards.length}` +
    (empty ? ` · без знахідок: ${empty}` : "") + (failed ? ` · <b>з помилкою: ${failed}</b>, інші рівні чинні` : "") + `</div>${cards.map((b) => datasetCard(b, busy, ready > 0 && emptyLevel(b))).join("")}</div>`;
}

function datasetTrust(t) {
  /* a dataset card's trust chips, from its Dataset node */
  if (!t) return "";
  if (t.integrity === "mismatch") return `<div class="ds-src">${chip("bad", "метадані не пройшли перевірку", `Запис датасету більше не збігається з відбитком своїх метаданих: його ${DECLARED} ігноруються`)}</div>`;
  const chips = (t.source || t.reliability !== "F" ? chip("", `${t.source || "джерело"} · ${t.reliability}`, `${word("reliability")} (шкала Адміралтейства): ${word(`reliability_${t.reliability}`)}`) : "") +
    (t.integrity === "unverified" ? chip("warn", "метадані не перевірено", "Записано до появи відбитків метаданих: джерело й оцінку не перевірено") : "") +
    (t.glossary_columns ? chip("", `глосарій · ${plural(t.glossary_columns, "колонка", "колонки", "колонок")}`, "Заявлені назви колонок і категорій: так їх читають інсайти й відповіді") : "") +
    (t.metadata_columns ? chip("", `документ · ${plural(t.metadata_columns, "колонка", "колонки", "колонок")}`, t.about || "Документ метаданих: довідка про колонки для відповідей") : "");
  return chips ? `<div class="ds-src">${chips}</div>` : "";
}

async function deleteDataset(id, name) {
  if (!confirm(`Видалити «${name}» з бази знань?\n\nЙого інсайти та їхня належність до тем зникнуть; теми, що лишаться без інсайтів і без зв'язків, — теж.`)) return;
  try { await api(`/api/datasets/${encodeURIComponent(id)}`, { method: "DELETE" }); setUploadMsg(`Видалено «${name}».`); }
  catch (err) { setUploadMsg(err.message, true); }
  if ($("dataset-filter").value === id) $("dataset-filter").value = "";
  S.lastReady = null;  // force the views to reload
  refreshBatches();
}

const elapsed = (iso) => { const s = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000)); return s < 60 ? `${s} с` : `${Math.floor(s / 60)} хв ${s % 60} с`; };
const tick = (root) => root.querySelectorAll("[data-since]").forEach((el) => { el.textContent = elapsed(el.dataset.since); });

function swapChanged(el, to) {  // a group of levels swaps its changed cards one by one
  if (el.isEqualNode(to)) return;
  if (el.matches(".ds-group") && el.dataset.upload === to.dataset.upload && el.children.length === to.children.length) {
    const fresh = [...to.children];
    [...el.children].forEach((c, i) => swapChanged(c, fresh[i]));
  } else el.replaceWith(to);
}

function renderRail(html) {
  /* a card whose markup is unchanged keeps its nodes: a click on it survives the poll */
  const box = $("batches"), next = document.createElement("div");
  next.innerHTML = html;
  tick(box); tick(next);  // the same clock on both sides: a running card's time is no change
  const fresh = [...next.children], picked = box.contains(document.activeElement) && document.activeElement.closest(".ds")?.dataset.ds;
  if (box.children.length !== fresh.length) box.replaceChildren(...fresh);
  else [...box.children].forEach((el, i) => swapChanged(el, fresh[i]));
  if (picked && !box.contains(document.activeElement)) box.querySelector(`.ds[data-ds="${CSS.escape(picked)}"] .ds-pick`)?.focus();  // a replaced card keeps the focus
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
  const sel = $("dataset-filter"), cur = sel.value, options = `<option value="">Усі датасети</option>` +
    ready.map((b) => `<option value="${esc(b.dataset_id)}" data-file="${esc(b.filename)}">${esc(datasetName(b))}</option>`).join("");
  if (options !== S.options) {  // rewritten only when the READY set changes: an open list stays open
    S.options = sel.innerHTML = options;
    sel.value = ready.some((b) => b.dataset_id === cur) ? cur : "";
  }
  // one key for what every view follows: the READY batches and the dataset filter
  const viewKey = batches.filter((b) => b.status === "READY").map((b) => `${b.batch_id}:${b.metadata_digest || ""}`).join(",") + "|" + sel.value;  // a PATCH re-signs its record
  let map;
  if (viewKey !== S.lastReady) {
    S.lastReady = viewKey;
    map = GeoMap.refresh().catch((e) => console.error("map refresh failed", e));  // the map never breaks the rail or an answer
    await loadGraph(); refreshHealth();
    if (!S.chat.length) renderThread();  // the welcome card's questions follow the datasets and the filter
  }
  const busy = batches.some((b) => !TERMINAL.has(b.status));
  renderRail(cards.length ? uploadGroups(cards).map((g) => uploadGroup(g, busy)).join("") : `<div class="empty-rail">Датасетів ще немає.<br>Завантажте таблицю або спробуйте демо.</div>`);  // after the graph: the cards read its trust
  clearTimeout(S.poll);
  if (busy) S.poll = setTimeout(refreshBatches, 2500);
  // a caller that acts on the map afterwards (ask) awaits its load too, or the pass a poll began for this key; the poll does not
  return map || GeoMap.refreshing()?.catch((e) => console.error("map refresh failed", e));
}

function setUploadMsg(text, err = false) { const el = $("upload-msg"); el.className = "msg" + (err ? " err" : ""); el.textContent = text; }

async function upload() {
  const f = $("file").files[0]; if (!f) return;
  const fd = new FormData();
  fd.append("file", f); fd.append("bins", $("bins").value); fd.append("categories", $("categories").value); fd.append("geo", $("geo").value);
  if ($("geo").value.trim() && $("table").checked) fd.append("table", "true");
  if ($("time").value.trim()) { fd.append("time", $("time").value); fd.append("grid", $("grid").value); }
  fd.append("source", $("source").value); fd.append("reliability", $("reliability").value); fd.append("indicators", $("indicators").value);
  fd.append("glossary", $("glossary").value);
  if ($("metadata").files[0]) fd.append("metadata", $("metadata").files[0]);
  setUploadMsg(`Завантажую ${f.name}…`);
  $("upload-btn").disabled = true;
  try { await api("/api/upload", { method: "POST", body: fd }); setUploadMsg("У черзі — поступ видно нижче."); $("file").value = ""; $("metadata").value = ""; $("metadata-name").textContent = ""; $("file-name").textContent = ""; refreshBatches(); }
  catch (err) { setUploadMsg(err.message, true); $("upload-btn").disabled = false; }
}
