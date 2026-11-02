/* SIG UI — time (docs/08_interface.md §8.2): the time axis, every insight placed in time on one calendar, a bar for its
   core, a mark for each of its peaks and dips, the as-of date across them; the ruler shows the periods the graph holds
   at one level of the calendar, chosen by its buttons. Positions are percentages of the span, so a resize redraws
   nothing. The period bar above the views picks a stretch of the same calendar (S.period); the axis, the map and the
   questions read it. The pill sets the bar's window, the thumbs a stretch in it, by a step that follows the window. */
"use strict";

const HOUR = 3600000, DAY = 24 * HOUR;
const isoDay = (t) => new Date(t).toISOString().slice(0, 10);
const isoMinute = (t) => new Date(t).toISOString().slice(0, 16);  // the calendar's wall clock, as the period goes to the API
const wall = (s) => Date.parse(s.length > 10 ? `${s.slice(0, 19)}Z` : s);  // a calendar time with its zone dropped: every time the console holds is the data's wall clock
const shortDay = (d) => `${d.slice(8, 10)}.${d.slice(5, 7)}.${d.slice(0, 4)}`;  // 2031-03-24 → 24.03.2031
const PRESETS = [["останній тиждень", 7], ["останній місяць", 30], ["останні 3 місяці", 91], ["увесь період", 0]];  // days to the latest data
const STEP_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720];  // the thumbs' steps: about BAR_STEPS of them over the window
const BAR_STEPS = 30;

function periodWords(lo, hi) {
  /* [lo, hi) in words: whole days as their first and last day, else to the minute */
  const days = lo % DAY === 0 && hi % DAY === 0, at = (t) => `${shortDay(isoDay(t))} ${isoMinute(t).slice(11)}`;
  return days ? `${shortDay(isoDay(lo))} — ${shortDay(isoDay(hi - DAY))}` : `${at(lo)} — ${at(hi)}`;
}

const spanWords = (lo, hi) => ((hi - lo) % DAY === 0 ? plural((hi - lo) / DAY, "день", "дні", "днів") : plural(Math.round((hi - lo) / HOUR), "година", "години", "годин"));

const Timeline = (() => {
  const T = { hubs: [], rows: [], byId: {}, level: null, span: false, lo: 0, hi: 0, wlo: 0, whi: Infinity, plo: 0, phi: 0 };  // the bar's span, window, stretch
  const ms = wall;

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
    T.byId = Object.fromEntries(T.rows.map((r) => [r.id, r]));
    T.span = Boolean(g.span);  // the timed datasets' days: the bar's, [the first's midnight, the midnight after the last)
    $("time-tab").hidden = !hubs.length;
    const was = JSON.stringify(S.period);
    if (T.span) {  // another filter: the window and the stretch it still holds, else the whole span
      const day = (t) => t - (t % DAY);
      [T.lo, T.hi] = [day(ms(g.span[0])), day(ms(g.span[1])) + DAY];
      const [wlo, whi] = [Math.max(T.wlo, T.lo), Math.min(T.whi, T.hi)];
      [T.wlo, T.whi] = wlo < whi ? [wlo, whi] : [T.lo, T.hi];
      const [lo, hi] = S.period ? [Math.max(T.plo, T.wlo), Math.min(T.phi, T.whi)] : [T.lo, T.hi];
      choose(...(lo < hi ? [lo, hi] : [T.wlo, T.whi]), false);
    } else { S.period = null; drawBar(); }
    if (JSON.stringify(S.period) !== was) GeoMap.onPeriod();  // the map hears of a stretch the filter narrowed or dropped
    if (!hubs.length && S.view === "time") setView("graph");
    else if (S.view === "time") render();
  }

  function render() {
    const body = $("time-body"), levels = TIME_LEVELS.filter((l) => T.hubs.some((h) => h.granularity === l));
    if (!levels.length) { body.innerHTML = ""; return; }
    const count = (l) => T.hubs.filter((h) => h.granularity === l).length, room = Math.max(body.clientWidth - 240, 200);
    if (!levels.includes(T.level)) T.level = levels.find((l) => room / count(l) >= 56) || levels[levels.length - 1];  // the finest whose labels fit
    const now = S.asOf ? ms(S.asOf) : null;
    const t0 = Math.min(...T.hubs.map((h) => ms(h.t_start)), now ?? Infinity), t1 = Math.max(...T.hubs.map((h) => ms(h.t_end)), now ?? -Infinity);
    const at = (t) => ((100 * (t - t0)) / (t1 - t0 || 1)).toFixed(2), span = (a, b) => `left:${at(a)}%;width:${(at(b) - at(a)).toFixed(2)}%`;
    $("time-levels").innerHTML = TIME_LEVELS.map((l) =>
      `<button type="button" class="tl-level${l === T.level ? " active" : ""}" data-level="${l}" aria-pressed="${l === T.level}"${levels.includes(l) ? "" : " disabled"}>${TIME_LEVEL_NAMES[l]}</button>`).join("");
    const ticks = T.hubs.filter((h) => h.granularity === T.level).map((h) =>
      `<button type="button" class="tl-tick" data-id="${esc(h.id)}" style="${span(ms(h.t_start), ms(h.t_end))}" title="${esc(h.t_start.slice(0, 10))} — ${esc(h.t_end.slice(0, 10))}">${esc(h.period)}</button>`).join("");
    const [lo, hi] = S.period ? [T.plo, T.phi] : [-Infinity, Infinity];
    const cut = (a, b) => {  // a core's part outside the stretch dark: where the stretch starts and ends along the bar
      if (!S.period) return "";
      const f = (t) => (100 * Math.min(Math.max((t - a) / (b - a || 1), 0), 1)).toFixed(1);
      return `;--in0:${f(lo)}%;--in1:${f(Math.max(hi, lo))}%`;
    };
    const rows = T.rows.map((r) => {
      const c = r.core;
      const marks = r.events.map((e) => `<b class="tl-ev ${e.kind === "PEAKED_AT" ? "up" : "down"}" style="left:${at((ms(e.hub.t_start) + ms(e.hub.t_end)) / 2)}%" title="${esc(word(e.kind))} ${esc(e.hub.period)}">${e.kind === "PEAKED_AT" ? "▲" : "▼"}</b>`).join("");
      return `<div class="tl-row" data-id="${esc(r.id)}" tabindex="0" role="button" title="${esc(r.full_label)}">` +
        `<span class="tl-name">${esc(r.full_label)}</span><span class="tl-track">${c ? `<i class="tl-core${S.period ? " cut" : ""}" style="${span(ms(c.t_start), ms(c.t_end))}${cut(ms(c.t_start), ms(c.t_end))}"></i>` : ""}${marks}</span></div>`;
    }).join("");
    const line = now !== null ? `<i class="tl-asof" style="left:calc(var(--tl-gutter) + (100% - var(--tl-gutter)) * ${at(now) / 100})" title="${esc(word("as_of"))} ${esc(S.asOf.slice(0, 10))}"></i>` : "";
    body.innerHTML = `<div class="tl-head"><span class="tl-name muted">${esc(cap(TIME_LEVEL_NAMES[T.level]))}</span><span class="tl-track">${ticks}</span></div>${rows}${line}`;
    $("time-caption").textContent = `${plural(T.rows.length, "інсайт", "інсайти", "інсайтів")} · ${plural(T.hubs.length, "період", "періоди", "періодів")}` +
      (S.period ? ` · ${word("period")} ${words()}` : S.asOf ? ` · ${word("as_of")} ${S.asOf.slice(0, 10)}` : "");
  }

  function setLevel(level) { T.level = level; render(); }

  function inPeriod(pid) {
    /* whether an insight acts in the chosen stretch: its core meets it, or one of its events lies inside it, as a period
       reads an event (docs/07_question_answering.md §7.2); one without either, or not on this graph, is never set aside */
    const r = T.byId[pid];
    if (!S.period || !r) return true;
    const [lo, hi] = [T.plo, T.phi];
    return Boolean((r.core && ms(r.core.t_start) < hi && lo < ms(r.core.t_end)) || r.events.some((e) => lo <= ms(e.hub.t_start) && ms(e.hub.t_end) <= hi));
  }

  const words = () => S.period && periodWords(T.plo, T.phi);

  function choose(lo, hi, window = true) {
    /* [lo, hi) chosen, the bar's window with it unless a thumb moved; the whole span is no period. Redraws the bar only:
       apply() tells the views */
    if (window) { T.wlo = lo; T.whi = hi; }
    [T.plo, T.phi] = [lo, hi];
    S.period = lo <= T.lo && hi >= T.hi ? null : { start: isoMinute(lo), end: isoMinute(hi) };
    drawBar();
  }

  function setPeriod(start, end) { choose(ms(start), ms(end)); }  // ISO wall clock, the end not in it

  function apply() {
    if (S.view === "time") render();
    GeoMap.onPeriod();
  }

  function step() {
    /* the step nearest to a BAR_STEPS-th of the window, on a log scale: 6 hours over a week, 3 days over three months */
    const want = (T.whi - T.wlo) / BAR_STEPS;
    return HOUR * STEP_HOURS.reduce((a, b) => (Math.abs(Math.log((b * HOUR) / want)) < Math.abs(Math.log((a * HOUR) / want)) ? b : a));
  }

  const thumbAt = (k) => Math.min(T.wlo + k * step(), T.whi);  // a thumb's place: a step of the window, its end the last

  function drawBar() {
    $("period-bar").hidden = !T.span;
    if (!T.span) return;
    const width = T.whi - T.wlo || 1, n = Math.ceil(width / step());
    const [lo, hi] = S.period ? [T.plo, T.phi] : [T.lo, T.hi];
    const k = (t) => Math.min(Math.max(Math.round((t - T.wlo) / step()), 0), n), f = (t) => Math.min(Math.max((t - T.wlo) / width, 0), 1);
    for (const [id, t] of [["pb-from", lo], ["pb-to", hi]]) { $(id).max = n; $(id).value = k(t); }
    Object.assign($("pb-fill").style, { left: `calc(7px + (100% - 14px) * ${f(lo)})`, width: `calc((100% - 14px) * ${f(hi) - f(lo)})` });
    $("pb-text").innerHTML = `<b>${esc(S.period ? words() : periodWords(T.lo, T.hi))}</b><small>${esc(S.period ? spanWords(lo, hi) : "увесь період")}</small>`;
    $("pb-clear").style.visibility = S.period ? "" : "hidden";  // its room kept: the pill never changes width
    for (const [id, v] of [["pb-start", T.wlo], ["pb-end", T.whi - DAY]]) Object.assign($(id), { min: isoDay(T.lo), max: isoDay(T.hi - DAY), value: isoDay(v) });
    const room = $("pb-ticks").clientWidth || 600, levels = TIME_LEVELS.filter((l) => T.hubs.some((h) => h.granularity === l));
    const level = levels.find((l) => T.hubs.filter((h) => h.granularity === l && ms(h.t_end) > T.wlo && ms(h.t_start) < T.whi).length * 64 <= room) || levels[levels.length - 1];
    const days = Array.from({ length: Math.ceil(width / DAY) }, (_, i) => T.wlo + i * DAY);
    const ticks = width <= 14 * DAY ? days.map((t) => [t, shortDay(isoDay(t)).slice(0, 5)])  // a short window: its days
      : T.hubs.filter((h) => h.granularity === level).map((h) => [ms(h.t_start), h.period]);
    let last = -Infinity;
    $("pb-ticks").innerHTML = ticks.map(([t, label]) => {
      const at = (t - T.wlo) / width;
      if (at < 0 || (at - last) * room < 64 || (1 - at) * room < 48) return "";  // before the window, into the last label or off the end
      last = at;
      return `<span style="left:${(100 * at).toFixed(2)}%">${esc(label)}</span>`;
    }).join("");
  }

  function slide(e) {
    /* a thumb moved: never past the other one; the views follow when it is let go (change) */
    const from = $("pb-from"), to = $("pb-to");
    if (+from.value > +to.value) (e.target === from ? to : from).value = e.target.value;
    choose(thumbAt(+from.value), thumbAt(+to.value), false);
  }

  function jump(e) {
    /* a click on the rail: the nearer thumb comes to the step under it */
    if (e.target.matches("input")) return;  // a thumb's own click
    const box = $("pb-ticks").getBoundingClientRect(), from = $("pb-from"), to = $("pb-to");
    const k = Math.round(Math.min(Math.max((e.clientX - box.left) / (box.width || 1), 0), 1) * +to.max);
    const thumb = Math.abs(k - from.value) < Math.abs(k - to.value) || (k < +from.value) ? from : to;
    thumb.value = k;
    choose(thumbAt(+from.value), thumbAt(+to.value), false); apply();
  }

  function openPop(open) {
    $("pb-pop").hidden = !open; $("pb-label").setAttribute("aria-expanded", String(open));
    if (open) $("pb-presets").innerHTML = PRESETS.map(([label, days]) => `<button type="button" class="btn ghost sm" data-days="${days}">${label}</button>`).join("");
  }

  function init() {
    for (const id of ["pb-from", "pb-to"]) { $(id).addEventListener("input", slide); $(id).addEventListener("change", apply); }
    document.querySelector(".pb-track").addEventListener("click", jump);
    $("pb-label").addEventListener("click", () => openPop($("pb-pop").hidden));
    $("pb-clear").addEventListener("click", () => { choose(T.lo, T.hi); apply(); });
    on($("pb-presets"), "button", (b) => {
      const days = +b.dataset.days;  // back from the latest data's day, the span's last
      choose(days ? Math.max(T.hi - days * DAY, T.lo) : T.lo, T.hi); apply(); openPop(false);
    });
    for (const id of ["pb-start", "pb-end"]) $(id).addEventListener("change", () => {
      const [s, e] = [$("pb-start").value, $("pb-end").value];
      if (!s || !e) return;  // a half-typed date: nothing yet
      const [a, b] = [ms(s), ms(e)].sort((x, y) => x - y);
      choose(Math.max(a, T.lo), Math.min(b + DAY, T.hi)); apply();
    });
    document.addEventListener("click", (e) => { if (!$("pb-pop").hidden && !e.target.closest(".pb-pill")) openPop(false); });
    new ResizeObserver(() => T.span && drawBar()).observe($("pb-ticks"));  // the ticks that fit
  }

  return { load, render, setLevel, inPeriod, setPeriod, words, apply, openPop, init };
})();
