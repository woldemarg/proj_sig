/* SIG UI — the chat: the answer card with its evidence, the insight card from the map, asking (docs/08_interface.md §8.2, §8.5). */
"use strict";

function md(text, keyTo) {
  /* small, safe markdown: escape first, then bold/italic/code, headings, lists, paragraphs, citations */
  const cite = (s) => s.replace(/\[(P\d+(?:\s*[,;]\s*P\d+)*)\]/g, (m, g) => g.split(/\s*[,;]\s*/).map((k) => keyTo[k] ? `<a class="cite" data-pid="${esc(keyTo[k])}" tabindex="0" role="button" title="${esc(keyTo[k])}">${k}</a>` : k).join(""));
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
  if (item.role.startsWith("vertical_")) return `<div class="chain"><span class="n">${esc(item.path_text)}</span></div>`;  // a level up or down
  if (item.role === "seed" || !item.path.length) return `<div class="chain"><span class="n">matched your question</span></div>`;
  let html = `<span class="n">${esc(item.path[0].source)}</span>`;
  for (const st of item.path) {
    const label = { ACTIVATES: st.reverse ? "member" : "theme", RELATED_TO: "related theme", CO_OCCURS: "co-occurring theme", SPECIALIZES: st.reverse ? "narrower" : "broader", CONTRASTS: "contrast", CO_LOCATED: "same place" }[st.edge_type] || st.edge_type;
    html += `<span class="e">→ ${esc(label)} ${num(st.weight)} →</span><span class="n ${st.target.startsWith("A-") ? "A" : ""}">${esc(st.target)}</span>`;
  }
  return `<div class="chain">${html}</div>`;
}

function itemTrust(it) {
  /* an evidence item's trust badges; the tooltip is the trust line the model saw */
  if (!it.trust) return "";
  const integrity = (it.trust.match(/integrity (\w+)$/) || [])[1] || "unverified";
  const [ccls, cwords] = cred(it.credibility);
  return `<div class="ev-trust" title="${esc(it.trust)}">` + (it.reliability ? chip("", `${word("reliability")} ${it.reliability}`) : chip("bad", `${word("reliability")}: невідома`)) +
    (it.credibility != null ? chip(ccls, cwords) : "") + chip(INTEGRITY[integrity] ?? "warn", word(`integrity_${integrity}`)) + `</div>`;
}

function chainLine(chain, items) {
  /* res 7: P1 +1.60 sd → res 5: P7 +1.10 sd → region R:3:…, finest first, on the host's target */
  const by = Object.fromEntries(items.map((i) => [i.pattern_id, i])), host = by[chain.pattern_id];
  const steps = [...chain.down].reverse().map((s) => by[s.pattern_id]).concat([host], chain.up.map((s) => by[s.pattern_id])).filter(Boolean);
  const level = (it) => { const sp = it.provenance.spatial; return sp ? `res ${sp.resolution}` : "table"; };
  const parts = steps.map((it) => { const x = it.statistics.shifts.find((sh) => sh.metric === host.target); return `<span class="n">${esc(level(it))}: ${esc(it.key)}${x ? ` ${signed(x.robust_z)} sd` : ""}</span>`; });
  if (chain.regions[0]) parts.push(`<span class="n">region ${esc(chain.regions[0].id)}</span>`);
  return `<div class="chain" title="The same phenomenon at finer and coarser levels"><span class="e">scale</span>${parts.join('<span class="e">→</span>')}</div>`;
}

function botCard(turn) {
  const qa = turn.qa;
  const ev = qa.view.evidence || {};
  const items = ev.items || [];
  const keyTo = Object.fromEntries(items.map((i) => [i.key, i.pattern_id]));
  const area = ev.area || null;  // an asked map zone: item key -> place, 0 outside
  const chains = Object.fromEntries((ev.scale_chains || []).filter((c) => c.up.length || c.down.length).map((c) => [c.pattern_id, c]));
  const forged = (ev.datasets || []).filter((d) => d.integrity === "mismatch").map((d) => d.dataset_id);
  const llm = qa.llm || {};
  const answer = qa.answer || "";
  const forgedNote = forged.length ? `<div class="notice bad">The metadata of ${esc(forged.join(", "))} failed its integrity check: its source, reliability and indicators are ignored.</div>` : "";
  const notice = qa.answer_mode !== "fallback" ? ""
    : llm.error === "disabled" ? "Showing the evidence only (explanation turned off)."
    : "The language model is unavailable — showing the evidence only.";
  const cit = qa.citations || {};
  const b = (qa.view.traversal && qa.view.traversal.baselines) || {};
  const cross = items.filter((i) => i.transversal_only).length;
  const model = String(llm.model || "").split("/").pop();
  const meta = [
    qa.answer_mode === "llm" ? chip("", `✨ ${model} · ${num(llm.latency_s, 1)} s`) : chip("warn", "Evidence only"),
    cit.grounded ? chip("ok", `✓ ${(cit.cited || []).length} sources cited`) : chip("warn", "No citations"),
    cross ? chip("run", `⤳ ${cross} cross-segment`, "Reached only through a theme, in a different part of the data") : "",
  ].join("");
  // literal grounding (docs/07_question_answering.md §7.1.1): which words of the question were read as which data literals
  const grounded = (qa.view.evidence && qa.view.evidence.parsed && qa.view.evidence.parsed.grounding) || [];
  const understood = grounded.length ? `<div class="understood" title="Question words matched to data literals (by characters or by meaning)">Understood: ${grounded.map((g) => `<b>${esc(g.span)}</b> → ${esc(g.literal)}`).join(" · ")}</div>` : "";
  const evid = items.map((it, i) => {
    if (it.folded_into) return "";  // another level of the same finding: shown on its host's card
    const s = it.statistics;
    return `<div class="ev" data-i="${i}" tabindex="0" role="button"><div class="ev-head"><span class="ev-key">${esc(it.key)}</span><span class="role ${esc(it.role)}">${esc({ seed: "match", structural: "lattice", transversal: "via theme", vertical_up: "coarser level", vertical_down: "finer level" }[it.role] || it.role)}</span>` +
      (it.transversal_only ? `<span class="role cross">other segment</span>` : "") +
      (it.also && it.also.length ? `<span class="role" title="The same finding at another level of this upload">also ${esc(it.also.join(", "))}</span>` : "") +
      (area && it.key in area ? `<span class="role" title="Its map cells against the zone the question asked about">${area[it.key] ? "in the asked area" : "outside the asked area"}</span>` : "") +
      (it.scale && it.scale !== "insufficient" && TERMS[`scale_${it.scale}`] ? `<span class="role">${term(`scale_${it.scale}`)}</span>` : "") + `</div>` +
      `<div class="ev-scope">${esc(it.reading.where)}</div><div class="ev-what">${esc(it.reading.what)}</div>` +
      `<div class="ev-stats">${term("support")}: ${num(s.support, 0)} · ${term("weight")}: ${num(s.weight)}</div>${itemTrust(it)}${renderChain(it)}${chains[it.pattern_id] ? chainLine(chains[it.pattern_id], items) : ""}</div>`;
  }).join("");
  const hypo = b.transversal_only ? `<div class="hypo"><b>Трансверсальна перевірка:</b> ${b.transversal_only.length} із цих інсайтів не мають жодної спільної умови з відправною точкою і знайдені лише через тему.</div>` : "";
  const keyOf = Object.fromEntries(items.map((i) => [i.pattern_id, i.key]));
  const zones = (ev.zones || []).length ? `<div class="hypo"><b>Zones</b>${ev.zones.map((z, n) => `<div>Z${n + 1} · res ${z.resolution} · ${z.holds.map((p) => keyOf[p]).join(", ")} — ${picture(z)}</div>`).join("")}</div>` : "";
  const notes = (ev.notes || []).length ? `<div class="hypo"><b>Notes</b>${ev.notes.map((n) => `<div>${esc(n)}</div>`).join("")}</div>` : "";
  const panes = [
    items.length ? ["Evidence & how it was found", `<span class="count chip">${items.length}</span>`, `<p class="ev-prefix">Click an item to isolate its path in the graph or on the sphere.</p>${hypo}${zones}${notes}${evid}`] : null,
    ["Sources", "", `<div class="muted" style="font-size:12.5px">${esc(qa.provenance_footer || "")}</div>`],
    qa.prompt ? ["Prompt", "", `<p class="ev-prefix">The exact evidence the model was shown.</p><pre class="canon">${esc(qa.prompt)}</pre>`] : null,
  ].filter(Boolean);
  return `<div class="bot-card">${forgedNote}${notice ? `<div class="notice">${esc(notice)}</div>` : ""}<div class="answer-box">${md(answer, keyTo)}</div>` +
    `<div class="bot-meta">${meta}</div>${understood}` +
    `<div class="bot-tabs" role="tablist">${panes.map(([t, c], i) => `<button type="button" role="tab" data-pane="${i}" aria-selected="${String(i) === turn.pane}"${String(i) === turn.pane ? ' class="active"' : ""}>${esc(t)}${c}</button>`).join("")}</div>` +
    panes.map(([, , body], i) => `<div class="bot-pane" data-pane="${i}"${String(i) === turn.pane ? "" : " hidden"}>${body}</div>`).join("") +
    `</div>`;
}

function followUps(turn) {
  const items = (turn.qa.view.evidence && turn.qa.view.evidence.items) || [];
  const geo = items.length && (GeoMap.has(items[0].pattern_id) || items[0].provenance.dataset_id === GeoMap.current());  // the map need not be open
  if (geo) return GeoMap.suggestions().filter((q) => q !== turn.question).slice(0, 3);
  const atts = (turn.qa.view.evidence && turn.qa.view.evidence.attractors) || [];
  const out = [];
  if (atts[0]) out.push(`Де ще трапляється «${atts[0].label}»?`);
  const cross = items.find((i) => i.transversal_only);
  if (cross) out.push(`Розкажи про ${cross.target} для ${cross.scope.map((c) => c.split("=")[1]).join(" ")}`);
  const seed = items[0];
  if (seed && seed.statistics.covariance && seed.statistics.covariance.pair) out.push(`Як пов'язані ${seed.statistics.covariance.pair.join(" і ")}?`);
  return out.slice(0, 3);
}

const chipButtons = (questions) => questions.map((q) => `<button type="button" class="chip-q">${esc(q)}</button>`).join("");

function suggestions() {
  /* the welcome card's questions: only for data that is there — a demo dataset's own set, another geo dataset's
     questions from its insights; the dataset filter (whose options are the READY datasets) narrows them to one
     dataset; no dataset, no questions */
  const filter = $("dataset-filter").value;
  const out = [];
  for (const o of [...$("dataset-filter").options].filter((o) => o.value && (!filter || o.value === filter))) {
    if (SUGGESTIONS[o.dataset.file]) out.push(...SUGGESTIONS[o.dataset.file]);
    else if (o.value === GeoMap.current()) out.push(...GeoMap.suggestions());
  }
  return out;
}

// an insight picked on the map (GeoMap.pick): its card in the chat, with its place to ask about and its graph links to walk on
const WALK = { CONTRASTS: "contrasts with", "CONTRASTS (in)": "contrasts with", "SPECIALIZES": "narrower version of", "SPECIALIZES (in)": "broader than",
  SIBLING: "sibling", "SIBLING (in)": "sibling", CO_LOCATED: "same place", "CO_LOCATED (in)": "same place" };

async function pickInsight(pid, view) {
  let d;
  try { d = await api(`/api/nodes/${encodeURIComponent(pid)}`); } catch (e) { return; }
  S.chat.push({ question: d.node.label, picked: true, insight: d, view });
  renderThread();
}

function askAbout(p) {
  const where = p.conditions.map((c) => c.value).join(" ");
  return p.phenomenon_type === "covariance" && p.covariance && p.covariance.pair
    ? `Як пов'язані ${p.covariance.pair.map(human).join(" і ")} для ${where}?` : `Розкажи про ${human(p.target)} для ${where}`;
}

function insightCard(turn) {
  const d = turn.insight, p = d.node.props, w = turn.view;
  const theme = d.theme, seen = new Set([p.id]);  // primary_anchor, as the map's list shows it: never a weak membership
  const chain = d.scale_chain || { up: [], down: [] }, level = (id) => (d.levels[id] != null ? `res ${d.levels[id]}` : "another level");
  const steps = [...chain.up.map((st) => [st, "↑"]), ...chain.down.map((st) => [st, "↓"])].map(([st, arrow]) =>  // the scale chain's levels first
    ({ id: st.pattern_id, label: `${arrow} ${level(st.pattern_id)}: ${nodeLabel(st.pattern_id, d.labels)}`, word: st.relation === "corresponds" ? "same shift" : "opposite shift" }));
  const links = [...steps, ...Object.entries(WALK).flatMap(([type, word]) => (d.neighbors[type] || []).filter((it) => it.integrity !== "mismatch").map((it) => ({ ...it, word })))]
    .filter((it) => !seen.has(it.id) && seen.add(it.id)).slice(0, 8);
  return `<div class="bot-card insight-card"><div class="answer-box">` +
    `<div class="ic-tags">${w ? chip("", w.level) : ""}${theme ? `<a class="ic-theme" data-id="${esc(theme.id)}" tabindex="0" role="button" title="Open the theme">theme · ${esc(theme.label)}</a>` : ""}</div>` +
    `<p><b>${esc(d.reading.where)}</b></p><p>${esc(d.reading.what)}</p>` +
    `<div class="ev-stats">${term("support")}: ${num(p.support, 0)} (${num(p.support_fraction * 100, 1)}%) · ${term("weight")}: ${num(p.weight)}` +
    (w ? ` · ${term("cells_in_view")}: ${num(w.cells, 0)} · ${term("points")}: ${num(w.points, 0)} (${num(w.share * 100, 0)}%) · ${term("window_lift")}: ${esc(w.lift_words)}` : "") + `</div>` +
    (links.length ? `<h4>Walk on</h4>${links.map((it) => nbRow(it.id, it.label, it.word)).join("")}` : "") +
    `</div><div class="followups ic-actions"><button type="button" class="ic-ask">Запитати граф про це</button><button type="button" class="ic-graph">Показати в графі</button></div></div>`;
}

function botTurn(turn, isLast) {
  if (turn.insight) return insightCard(turn);
  if (turn.pending) {
    return `<div class="bot-card"><div class="typing"><i></i><i></i><i></i><span>Searching the graph${$("use-llm").checked ? " and writing an answer" : ""}… <span class="ds-elapsed" data-since="${turn.since}"></span></span>` +
      `<button type="button" class="linkish ask-cancel">Cancel</button></div></div>`;
  }
  if (turn.error) return `<div class="bot-card${turn.cancelled ? "" : " error"}"><div class="answer-box">${esc(turn.error)}</div></div>`;
  const next = isLast ? followUps(turn) : [];
  return botCard(turn) + (next.length ? `<div class="followups">${chipButtons(next)}</div>` : "");
}

const turnOf = (el) => S.chat[Number(el.closest(".msg-bot").dataset.turn)];  // the turn a clicked element of the thread belongs to

function toggleTab(tab) {
  /* an answer's panel tab: opens its pane, or closes it when it is the open one */
  const box = tab.closest(".msg-bot"), open = !tab.classList.contains("active");
  turnOf(tab).pane = open ? tab.dataset.pane : null;  // on the turn: the thread's next render keeps it open
  box.querySelectorAll(".bot-tabs button").forEach((t) => { t.classList.toggle("active", open && t === tab); t.setAttribute("aria-selected", String(open && t === tab)); });
  box.querySelectorAll(".bot-pane").forEach((pane) => { pane.hidden = !open || pane.dataset.pane !== tab.dataset.pane; });
}

function renderThread() {
  /* the conversation, or the welcome card with the questions the data offers (re-rendered when the data or the filter change) */
  const t = $("thread");
  if (!S.chat.length) {
    const chips = suggestions();
    t.innerHTML = `<div class="welcome"><div class="art"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 5h16v11H8l-4 4z"/></svg></div>` +
      `<h3>Ask about your data</h3><p>Answers are built only from validated insights in the graph, with every claim linked to its source.</p>` +
      (chips.length ? `<div class="suggest">${chipButtons(chips)}</div>` : "") + `</div>`;
  } else {
    t.innerHTML = S.chat.map((turn, idx) =>
      `<div class="msg-user">${esc(turn.question)}${turn.zone ? `<span class="msg-zone">зона на карті · комірок: ${turn.zone}</span>` : ""}${turn.picked ? `<span class="msg-zone">обрано на карті</span>` : ""}</div><div class="msg-bot" data-turn="${idx}">${botTurn(turn, idx === S.chat.length - 1)}</div>`).join("");
  }
  tick(t);
  t.scrollTop = t.scrollHeight;
}

async function ask(question, zone = null) {
  question = (question || "").trim();  // with a map zone ({cells}) the text may be empty; false: asked without one
  if (zone === null && GeoMap.zone().length) zone = { cells: GeoMap.zone() };  // a selected zone goes with every question until cleared
  if ((!question && !zone) || S.busy) return;
  S.busy = true; $("ask-btn").disabled = true;
  $("question").value = ""; autosize();
  const turn = { question, zone: zone ? zone.cells.length : 0, pending: true, since: new Date().toISOString(), ctl: new AbortController() };
  const live = () => S.chat.includes(turn);  // New chat drops the turn: its late answer paints nothing
  const timer = setTimeout(() => turn.ctl.abort("timeout"), CHAT_TIMEOUT_S * 1000);
  S.chat.push(turn); renderThread();
  try {
    const qa = await api("/api/chat/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, ...zone, use_llm: $("use-llm").checked }), signal: turn.ctl.signal });
    Object.assign(turn, { pending: false, qa });
    S.hl = qa.view.highlight;
    const evDatasets = ((qa.view.evidence && qa.view.evidence.datasets) || []).map((d) => d.dataset_id);
    if ($("dataset-filter").value && evDatasets.length && !evDatasets.includes($("dataset-filter").value)) {
      $("dataset-filter").value = evDatasets.length === 1 ? evDatasets[0] : "";  // show where the evidence lives
      await refreshBatches();  // every view, the map and the rail's active card follow the filter
      if (!live()) return;
    }
    renderThread();
    highlight(S.hl);
    try { GeoMap.onAnswer(qa); } catch (err) { console.error("map highlight failed", err); }  // the map never breaks an answer
    if (S.view === "graph") focusNodes([...(qa.view.highlight.traversed || [])]);
    renderSphere();
  } catch (e) {
    const why = turn.ctl.signal.reason, late = why === "timeout" || e.status === 504;
    if (live()) { Object.assign(turn, { pending: false, cancelled: why === "cancel", error: why === "cancel" ? "Cancelled." : late ? ERROR_HELP.chat_timeout.join(" ") : e.message }); renderThread(); }
  } finally {
    clearTimeout(timer);
    S.busy = false; $("ask-btn").disabled = !$("question").value.trim();
  }
}

function autosize() { const q = $("question"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 140) + "px"; $("ask-btn").disabled = S.busy || !q.value.trim(); }
