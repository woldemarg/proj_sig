/* SIG UI — the time axis (docs/08_interface.md §8.2): every insight placed in time on one calendar, a bar for its core,
   a mark for each of its peaks and dips, the as-of date across them; the ruler shows the periods the graph holds at one
   level of the calendar, chosen by its buttons. Positions are percentages of the span, so a resize redraws nothing. */
"use strict";

const Timeline = (() => {
  const T = { hubs: [], rows: [], level: null };
  const ms = (s) => Date.parse(s);

  function load(g) {
    /* the graph payload's periods and the insights with a core or an event, each with its events joined through the hub */
    const data = g.nodes.map((n) => n.data), hubs = data.filter((d) => d.kind === "TemporalNode");
    const byId = Object.fromEntries(hubs.map((h) => [h.id, h])), events = {};
    for (const e of g.edges.map((x) => x.data)) {
      if ((e.type === "PEAKED_AT" || e.type === "DROPPED_AT") && byId[e.target]) (events[e.source] ||= []).push({ kind: e.type, hub: byId[e.target] });
    }
    const from = (r) => (r.core ? ms(r.core.t_start) : Math.min(...r.events.map((e) => ms(e.hub.t_start))));
    T.hubs = hubs.sort((a, b) => ms(a.t_start) - ms(b.t_start) || a.id.localeCompare(b.id));  // the ruler in calendar order
    T.rows = data.filter((d) => d.kind === "Pattern" && (d.core || events[d.id])).map((d) => ({ ...d, events: events[d.id] || [] }))
      .sort((a, b) => from(a) - from(b) || b.weight - a.weight);  // in calendar order, the strongest first on a tie
    $("time-tab").hidden = !hubs.length;
    if (!hubs.length && S.view === "time") setView("graph");
    else if (S.view === "time") render();
  }

  function render() {
    const body = $("time-body"), levels = TIME_LEVELS.filter((l) => T.hubs.some((h) => h.granularity === l));
    if (!levels.length) { body.innerHTML = ""; return; }
    const count = (l) => T.hubs.filter((h) => h.granularity === l).length, room = Math.max(body.clientWidth - 240, 200);
    if (!levels.includes(T.level)) T.level = levels.find((l) => room / count(l) >= 56) || levels[levels.length - 1];  // the finest whose labels fit
    const asOf = asOfDate(), now = asOf ? ms(asOf) : null;
    const t0 = Math.min(...T.hubs.map((h) => ms(h.t_start)), now ?? Infinity), t1 = Math.max(...T.hubs.map((h) => ms(h.t_end)), now ?? -Infinity);
    const at = (t) => ((100 * (t - t0)) / (t1 - t0 || 1)).toFixed(2), span = (a, b) => `left:${at(a)}%;width:${(at(b) - at(a)).toFixed(2)}%`;
    $("time-levels").innerHTML = TIME_LEVELS.map((l) =>
      `<button type="button" class="tl-level${l === T.level ? " active" : ""}" data-level="${l}" aria-pressed="${l === T.level}"${levels.includes(l) ? "" : " disabled"}>${TIME_LEVEL_NAMES[l]}</button>`).join("");
    const ticks = T.hubs.filter((h) => h.granularity === T.level).map((h) =>
      `<button type="button" class="tl-tick" data-id="${esc(h.id)}" style="${span(ms(h.t_start), ms(h.t_end))}" title="${esc(h.t_start.slice(0, 10))} — ${esc(h.t_end.slice(0, 10))}">${esc(h.period)}</button>`).join("");
    const rows = T.rows.map((r) => {
      const c = r.core, active = c && now !== null && ms(c.t_start) <= now && now < ms(c.t_end);
      const marks = r.events.map((e) => `<b class="tl-ev ${e.kind === "PEAKED_AT" ? "up" : "down"}" style="left:${at((ms(e.hub.t_start) + ms(e.hub.t_end)) / 2)}%" title="${esc(word(e.kind))} ${esc(e.hub.period)}">${e.kind === "PEAKED_AT" ? "▲" : "▼"}</b>`).join("");
      return `<div class="tl-row${active ? " active" : ""}" data-id="${esc(r.id)}" tabindex="0" role="button" title="${esc(r.full_label)}">` +
        `<span class="tl-name">${esc(r.full_label)}</span><span class="tl-track">${c ? `<i class="tl-core" style="${span(ms(c.t_start), ms(c.t_end))}"></i>` : ""}${marks}</span></div>`;
    }).join("");
    const line = now !== null ? `<i class="tl-asof" style="left:calc(var(--tl-gutter) + (100% - var(--tl-gutter)) * ${at(now) / 100})" title="${esc(word("as_of"))} ${esc(asOf.slice(0, 10))}"></i>` : "";
    body.innerHTML = `<div class="tl-head"><span class="tl-name muted">${esc(cap(TIME_LEVEL_NAMES[T.level]))}</span><span class="tl-track">${ticks}</span></div>${rows}${line}`;
    $("time-caption").textContent = `${plural(T.rows.length, "інсайт", "інсайти", "інсайтів")} · ${plural(T.hubs.length, "період", "періоди", "періодів")}` + (asOf ? ` · ${word("as_of")} ${asOf.slice(0, 10)}` : "");
  }

  function setLevel(level) { T.level = level; render(); }

  return { load, render, setLevel };
})();
