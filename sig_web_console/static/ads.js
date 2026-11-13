/* SIG UI — a zone's question to ADS, the data-science agent behind the bridge: the form, the job's stages and its report in the chat (docs/08_interface.md §8.8). */
"use strict";

const ADS_STAGES = [["search", "SIG: питання + зона"], ["ads", "ADS: таблиці з H3, in_aoi і aoi_zone"], ["joint", "Фінальний звіт"]];
const ADS_POLL_MS = 3000;
const ADS_POLL_RETRIES = 5;  // failed reads of a job in a row before the turn gives it up
const adsPost = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
let adsPreviewTimer = null;

function adsBody() {
  /* the ask as the form holds it: the first ticked table is the base, the rest its context layers */
  const tables = [...document.querySelectorAll("#ads-tables input:checked")].map((i) => ({ dataset_id: i.value, filename: i.dataset.file }));
  return { question: $("ads-question").value.trim(), tables, polygon: GeoMap.polygon(), cells: GeoMap.zone() };
}

function previewAds() {
  /* the task ADS will get, but for the zone's figures and insights, which the ask brings */
  clearTimeout(adsPreviewTimer);
  adsPreviewTimer = setTimeout(async () => {
    const body = adsBody();
    $("ads-send").disabled = !body.question || !body.tables.length;
    if (!body.tables.length) { $("ads-preview").textContent = "Позначте хоча б одну таблицю."; return; }
    const d = await api("/api/ads/preview", adsPost(body)).catch((e) => ({ parts: [{ text: e.message }] }));
    $("ads-preview").textContent = d.parts.map((p) => p.text).join("");
  }, 250);
}

function openAds() {
  const open = GeoMap.current(), rows = [...GeoMap.datasets()].sort((a, b) => (b.dataset_id === open) - (a.dataset_id === open));
  $("ads-tables").innerHTML = rows.map((d) => `<label class="small chk"><input type="checkbox" value="${esc(d.dataset_id)}" data-file="${esc(d.filename)}"${d.dataset_id === open ? " checked" : ""}> ` +
    `${esc(d.filename)} · рівень ${esc(d.resolution)}${d.dataset_id === open ? " · відкрита на мапі" : ""}</label>`).join("");
  $("ads-zone").textContent = `Зона: ${plural(GeoMap.zone().length, "комірка", "комірки", "комірок")} мапи.`;
  $("ads-question").value = $("question").value;
  $("ads-dialog").showModal();
  previewAds();
}

async function askAds() {
  const body = adsBody();
  $("ads-dialog").close();
  const turn = { question: body.question, zone: body.cells.length, since: new Date().toISOString(), ads: { stage: "search", timings: {} } };
  S.chat.push(turn); renderThread();
  const fail = (e) => { turn.ads = { stage: "error", timings: {}, error: { code: e.code || "ads_unavailable", detail: e.message } }; renderThread(); };
  const started = await api("/api/ads/jobs", adsPost(body)).catch(fail);
  for (let failures = 0; started && S.chat.includes(turn);) {  // New chat drops the turn: its job is read no more
    const job = await api(`/api/ads/jobs/${started.job_id}`).catch((e) => (++failures > ADS_POLL_RETRIES ? fail(e) : null));
    if (failures > ADS_POLL_RETRIES) return;
    if (job) {  // a read that failed is read again: the job runs on
      const moved = job.stage !== turn.ads.stage;
      failures = 0; turn.ads = job;
      if (moved) { turn.since = new Date().toISOString(); renderThread(); }
      if (job.stage === "done" || job.stage === "error") return;
    }
    await new Promise((r) => setTimeout(r, ADS_POLL_MS));
  }
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
$("ads-tables").addEventListener("change", previewAds);
$("ads-cancel").addEventListener("click", () => $("ads-dialog").close());
$("ads-form").addEventListener("submit", (e) => { e.preventDefault(); askAds(); });
