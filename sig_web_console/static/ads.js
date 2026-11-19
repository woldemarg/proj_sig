/* SIG UI — the ADS conversation: questions about the map's zones to the data-science agent behind the bridge, the form, the
   job's stages and its report in the ADS tab; every question of a conversation goes to one ADS session (docs/08_interface.md §8.8). */
"use strict";

const ADS_STAGES = [["search", "SIG: знахідки"], ["ads", "ADS: аналіз таблиць"], ["joint", "Фінальний звіт"]];
const ADS_POLL_MS = 3000;
const ADS_POLL_RETRIES = 5;  // failed reads of a job in a row before the turn gives it up
const ADS_MAX_ZONES = 50;  // the zones one ask carries (the bridge's AskIn.zones): a long conversation's oldest go first
const adsPost = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
let adsPreviewTimer = null;
const PERIOD_LINE = /^Період:[^\n]*\n?/;  // the line openAds heads the question with: no question of its own
const adsRunning = () => S.threads.ads.some((t) => !["done", "error"].includes(t.ads.stage));  // ADS runs one question of a session at a time

function adsBody() {
  /* the ask as the form holds it: the first ticked table is the base, the rest its context layers (a link run's layer
     by its name, only with its own table ticked); the ticked zones are the question's, none the whole map's, sent with every zone the conversation asked
     before (a later question may name one); each zone is its polygon, which the bridge's graph question and the tables'
     rows both read (not the cells the map lights) */
  const own = new Set([...document.querySelectorAll("#ads-tables input:checked:not([data-layer])")].map((i) => i.value));
  const tables = [...document.querySelectorAll("#ads-tables input:checked")].filter((i) => !i.dataset.of || own.has(i.dataset.of)).map((i) =>
    ({ dataset_id: i.value, filename: i.dataset.file, resolution: +i.dataset.res, ...(i.dataset.layer ? { layer: i.dataset.layer } : {}) }));
  const focus = [...document.querySelectorAll("#ads-zones input:checked")].map((i) => i.value);
  const drawn = Object.fromEntries(GeoMap.zones().filter((z) => focus.includes(z.id)).map((z) => [z.id, z.polygon]));
  const before = Object.entries(S.adsZones).filter(([id]) => !(id in drawn));  // the conversation's earlier zones, the oldest first
  const zones = [...before.slice(Math.max(0, before.length + Object.keys(drawn).length - ADS_MAX_ZONES)), ...Object.entries(drawn)]
    .map(([id, polygon]) => ({ id, polygon }));
  const p = S.period;  // SIG reads the period bar alone; ADS reads the question's words
  return { question: $("ads-question").value.trim(), tables, zones, focus, ...(S.adsSession ? { session_id: S.adsSession } : {}), ...(p ? { period: [p.start, p.end] } : {}) };
}

function syncLayers() {
  /* a layer's box is off while its own table is not ticked: it goes with that table alone */
  const own = new Set([...document.querySelectorAll("#ads-tables input:checked:not([data-layer])")].map((i) => i.value));
  for (const i of document.querySelectorAll("#ads-tables input[data-layer]")) i.disabled = !own.has(i.dataset.of);
}

function adsState() {
  /* what a follow-up would not see change: the period bar's days, the map's zones, the datasets and their layers */
  const shown = Timeline.shown(), data = Object.keys(S.datasets).map((id) => [id, ...layersOf(id)].join("+")).sort();
  return { "період": shown ? `${shown.start}/${shown.end}` : "", "зони": GeoMap.zones().map((z) => z.id).join(), "дані": data.join() };
}

function periodHeader(p) {
  /* the period bar's [start, end) as the question's first line, ISO dates with the last day included */
  const lo = wall(p.start), hi = wall(p.end);
  if (lo % DAY === 0 && hi % DAY === 0) return `Період: ${isoDay(lo)} — ${isoDay(hi - DAY)} (включно)`;
  return `Період: ${p.start.replace("T", " ")} — ${p.end.replace("T", " ")} (кінець не включно)`;
}

function previewAds() {
  /* the task ADS will get, but for the zones' figures and insights, which the ask brings */
  const body = adsBody(), ready = body.tables.length;  // a layer goes with its table alone, so never as the base; no zone: the whole map
  $("ads-send").disabled = !body.question.replace(PERIOD_LINE, "").trim() || !ready || adsRunning();
  $("ads-send").title = adsRunning() ? "ADS ще відповідає на попереднє питання розмови" : "";
  clearTimeout(adsPreviewTimer);
  adsPreviewTimer = setTimeout(async () => {
    if (!ready) { $("ads-preview").textContent = "Позначте хоча б одну таблицю."; return; }
    const d = await api("/api/ads/preview", adsPost(body)).catch((e) => ({ parts: [{ text: e.message }] }));
    $("ads-preview").textContent = d.parts.map((p) => p.text).join("");
  }, 250);
}

function openAds(zoneId, changed = []) {
  if (!S.ads) return;  // as the popup's button: nothing to ask
  /* one row per upload, as its levels share the rows: the level open on the map, else the finest; under it the layers
     of the upload's layer link runs, ticked with the open table; the zone asked from ticked, the map's others offered
     beside it (without one, every zone ticked; none ticked asks about the whole map) */
  const open = GeoMap.current(), byUpload = new Map();
  for (const d of [...GeoMap.datasets()].sort((a, b) => (b.dataset_id === open) - (a.dataset_id === open) || b.resolution - a.resolution))
    if (!byUpload.has(uploadOf(d))) byUpload.set(uploadOf(d), d);
  const rows = [...byUpload.values()];
  const layers = (d) => {  // each layer once, read by the run that holds it (the open one first)
    const runs = new Map();
    for (const x of GeoMap.datasets().filter((x) => uploadOf(x) === uploadOf(d)).sort((a, b) => (b.dataset_id === open) - (a.dataset_id === open)))
      for (const name of layersOf(x.dataset_id)) if (!runs.has(name)) runs.set(name, x);
    return [...runs].map(([name, x]) => `<label class="small chk ads-layer"><input type="checkbox" value="${esc(x.dataset_id)}" data-file="${esc(name)}" data-res="${esc(x.resolution)}" ` +
      `data-layer="${esc(name)}" data-of="${esc(d.dataset_id)}"${d.dataset_id === open ? " checked" : ""}> шар ${esc(layerName(name))} · до ${esc(d.filename)}</label>`).join("");
  };
  $("ads-tables").innerHTML = rows.map((d) => `<label class="small chk"><input type="checkbox" value="${esc(d.dataset_id)}" data-file="${esc(d.filename)}" data-res="${esc(d.resolution)}"${d.dataset_id === open ? " checked" : ""}> ` +
    `${esc(d.filename)} · рівень ${esc(d.resolution)}${d.dataset_id === open ? " · відкрита на мапі" : ""}</label>${layers(d)}`).join("");
  const zones = GeoMap.zones();
  $("ads-zones").innerHTML = zones.map((z) => `<label class="small chk"><input type="checkbox" value="${esc(z.id)}"${!zoneId || z.id === zoneId ? " checked" : ""}> ` +
    `${esc(z.id)} · ${esc(plural(z.cells.length, "комірка", "комірки", "комірок"))}</label>`).join("") +
    `<p class="small muted">${zones.length ? "Якщо не позначити жодної, питання буде про всю мапу." : "Зон на мапі немає: питання про всю мапу."}</p>`;
  const asked = Object.keys(S.adsZones);
  $("ads-status").textContent = S.adsSession ? `Питання продовжить розмову з ADS (${plural(S.threads.ads.length, "питання", "питання", "питань")}${asked.length ? `, зони ${asked.join(", ")}` : ""}).` : "Нова розмова з ADS.";
  if (changed.length) $("ads-status").textContent += ` Змінилося: ${changed.join(", ")}. Перевірте таблиці, зони й період.`;
  const shown = Timeline.shown();  // the bar's days head the question even when they are the whole span: ADS reads them, the person may edit them
  $("ads-question").value = (shown ? `${periodHeader(shown)}\n\n` : "") + $("question").value;
  syncLayers();
  $("ads-dialog").showModal();
  previewAds();
}

function askAdsFurther(question) {
  /* the ADS tab's box: the conversation's next question, with the tables, zones and period of its last one; the form
     instead when it starts the conversation or the period, the zones or the data changed since */
  if (!question || adsRunning()) return;
  if (!S.ads) {  // neither the form nor a job would run: the turn says why
    S.threads.ads.push({ question, since: new Date().toISOString(), ads: { stage: "error", failed_stage: "search", timings: {}, error: { code: "ads_unavailable", detail: "health: ADS не відповідає" } } });
    $("question").value = ""; autosize(); renderThread(); return;
  }
  const now = adsState(), changed = S.adsLast ? Object.keys(now).filter((k) => now[k] !== S.adsState[k]) : [];
  if (!S.adsLast || changed.length) { openAds(null, changed); return; }
  $("question").value = ""; autosize();
  const shown = Timeline.shown();  // the bar's days head it, as the form heads the first: ADS bounds every figure by them
  askAds({ ...S.adsLast, question: (shown ? `${periodHeader(shown)}\n\n` : "") + question, ...(S.adsSession ? { session_id: S.adsSession } : {}) });
}

async function askAds(body) {
  for (const z of body.zones) S.adsZones[z.id] = z.polygon;  // the conversation's zones: a later question may name them
  S.adsLast = body; S.adsState = adsState();
  const turn = { question: body.question, zones: body.focus, since: new Date().toISOString(), ads: { stage: "search", timings: {} } };
  S.threads.ads.push(turn); syncTabs(); showTab("ads");
  const paint = () => { if (S.tab === "ads") { renderThread(); autosize(); } if ($("ads-dialog").open) previewAds(); };
  const fail = (e) => { turn.ads = { stage: "error", timings: {}, error: { code: e.code || "ads_unavailable", detail: e.message } }; paint(); };
  const started = await api("/api/ads/jobs", adsPost(body)).catch(fail);
  turn.jobId = started?.job_id;  // New chat cancels it
  for (let failures = 0; started && S.threads.ads.includes(turn);) {  // New chat drops the turn: its job is read no more
    const job = await api(`/api/ads/jobs/${started.job_id}`).catch((e) => (++failures > ADS_POLL_RETRIES ? fail(e) : null));
    if (failures > ADS_POLL_RETRIES) return;
    if (job) {  // a read that failed is read again: the job runs on
      const moved = job.stage !== turn.ads.stage;
      failures = 0; turn.ads = job;
      if (job.session_id && S.threads.ads.includes(turn)) S.adsSession = job.session_id;  // the conversation's next question goes to it
      if (moved) { turn.since = new Date().toISOString(); paint(); }
      if (job.stage === "done" || job.stage === "error") return;
    }
    await new Promise((r) => setTimeout(r, ADS_POLL_MS));
  }
  if (started && !S.threads.ads.includes(turn)) api(`/api/ads/jobs/${started.job_id}/cancel`, { method: "POST" }).catch(() => {});  // its conversation ended
}

function adsCard(turn) {
  const a = turn.ads, failed = a.stage === "error", at = ADS_STAGES.findIndex(([k]) => k === (failed ? a.failed_stage : a.stage));
  const stages = ADS_STAGES.map(([k, label], i) => {
    const state = a.stage === "done" || (at >= 0 && i < at) ? "ok" : i !== at ? "" : failed ? "bad" : "run";
    const took = a.timings[k] != null ? `${num(a.timings[k], 1)} с` : state === "run" ? `<span class="ds-elapsed" data-since="${turn.since}"></span>` : "";
    return `<div class="ads-stage ${state}"><i class="dot"></i><span>${esc(label)}</span><span class="muted">${took}</span></div>`;
  }).join("");
  const help = failed ? ERROR_HELP[a.error.code] : null, report = a.joint_report || a.ads_report || "";
  return `<div class="bot-card${failed ? " error" : ""}"><div class="answer-box"><div class="ads-stages">${stages}</div>` +
    (failed ? `<p><b>${esc(help ? help[0] : "Запит до ADS не вдався")}</b></p><p>${esc(help ? help[1] : a.error.detail)}</p>${help ? `<p class="muted" data-dev>${esc(a.error.detail)}</p>` : ""}` : "") +
    (a.stage === "done" && !a.joint_report ? `<p class="muted">Фінальний звіт не зібрано: нижче звіт ADS як є.</p>${a.joint_error ? `<p class="muted" data-dev>${esc(a.joint_error)}</p>` : ""}` : "") +
    (a.stage === "done" ? `<div class="ads-report">${md(report, {})}</div>` : "") +
    (a.joint_report ? `<details class="ads-more" data-dev><summary>Звіт ADS</summary><div class="ads-report">${md(a.ads_report, {})}</div></details>` : "") +
    (a.ads_query ? `<details class="ads-more" data-dev><summary>Запит до ADS</summary><pre>${esc(a.ads_query)}</pre></details>` : "") + `</div></div>`;
}

$("ads-question").addEventListener("input", previewAds);
$("ads-tables").addEventListener("change", () => { syncLayers(); previewAds(); });
$("ads-zones").addEventListener("change", (e) => {  // a question compares at most MAX_ZONES
  if ([...document.querySelectorAll("#ads-zones input:checked")].length > MAX_ZONES) e.target.checked = false;
  previewAds();
});
$("ads-cancel").addEventListener("click", () => $("ads-dialog").close());
$("ads-form").addEventListener("submit", (e) => { e.preventDefault(); $("ads-dialog").close(); $("question").value = ""; autosize(); askAds(adsBody()); });  // the box's text went with the form
