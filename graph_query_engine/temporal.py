"""Time in the answer (docs/07_question_answering.md §7.2, §7.4): an insight of a dataset with a time column carries
its time line — the grid, the span, the trend and current level of its metric (its target, unless only a shift has a
series), its core and its age, its events and the bins measured — from the snapshot's ``trajectories``; a calendar
period in the question raises the seed score of the patterns measured in it, and without one the fresher findings —
the closer their core to the reference date — rank first (``recency``)."""

from __future__ import annotations

import functools
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from insight_contracts import wording


def period_interval(text: str) -> tuple[datetime, datetime] | None:
    """The UTC interval [start, end) of the first calendar period written in ``text`` the way the grid names one —
    ``2031-W13`` (an ISO week), ``2031-Q1`` or ``Q1 2031``, ``2031-03-24`` (a time after it allowed), ``2031-03`` —
    (``wording.PERIOD``); a token that only looks like one (``2031-W54``, ``2019-20``) or whose period runs past the
    calendar (``9999-12-31``) is passed over. None when there is none."""
    return next((span for m in wording.PERIOD.finditer(text) if (span := _interval(m.group(0)))), None)


def period_hub(text: str) -> str | None:
    """The time hub id (``T:<unit>:<label>``, as the grid names the period) of the first calendar period
    ``period_interval`` reads in ``text``; None when there is none."""
    for m in wording.PERIOD.finditer(text):
        if _interval(m.group(0)):
            g = m.groups()
            if g[0]:
                return f"T:week:{g[0]}-W{g[1]}"
            if g[2] or g[4]:
                return f"T:quarter:{g[2] or g[5]}-Q{g[3] or g[4]}"
            return f"T:day:{g[6]}-{g[7]}-{g[8]}" if g[6] else f"T:month:{g[9]}-{g[10]}"
    return None


def parse_as_of(text: str) -> datetime:
    """A reference date as asked: an ISO date (its midnight) or an ISO date and time, UTC unless it names a zone;
    ``ValueError`` otherwise."""
    stamp = datetime.fromisoformat(text.strip())
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def age_days(core: dict[str, Any], as_of: datetime) -> float:
    """Days from a core (``[t_start, t_end)``) to ``as_of``: 0 inside it, positive after its end, negative before its start."""
    start, end = datetime.fromisoformat(core["t_start"]), datetime.fromisoformat(core["t_end"])
    if as_of < start:
        return (as_of - start).total_seconds() / 86_400
    return (as_of - end).total_seconds() / 86_400 if as_of >= end else 0.0


def recency(trajectory: dict[str, Any] | None, as_of: datetime | None, half_life_days: float) -> float | None:
    """How fresh a finding is at ``as_of``: 2^(−|age| / half-life), its age the days from its core to that date (1 inside
    the core; the future decays as the past does). None — no bonus, no penalty — without a core, a date or a positive
    half-life (0 switches it off)."""
    core = (trajectory or {}).get("core")
    if not core or as_of is None or half_life_days <= 0:
        return None
    return 0.5 ** (abs(age_days(core, as_of)) / half_life_days)


def strip_periods(text: str) -> str:
    """``text`` without the calendar periods it names (``period_interval``'s), so that none reads as a literal of the
    data (``2031-Q1`` is no band ``q1``); a token that only looks like one stays (``2019-20`` may be a value)."""
    return wording.PERIOD.sub(lambda m: " " if _interval(m.group(0)) else m.group(0), text)


@functools.lru_cache(maxsize=1024)  # ponytail: a grid's labels repeat across patterns and questions
def _interval(token: str) -> tuple[datetime, datetime] | None:
    """One ``wording.PERIOD`` token's interval; None for one that names no date."""
    g = wording.PERIOD.fullmatch(token).groups()
    try:
        if g[0]:
            start = date.fromisocalendar(int(g[0]), int(g[1]), 1)
            stop = start + timedelta(days=7)
        elif g[2] or g[4]:
            year, quarter = (int(g[2]), int(g[3])) if g[2] else (int(g[5]), int(g[4]))
            start = date(year, 3 * quarter - 2, 1)
            stop = _months_on(start, 3)
        elif g[6]:
            start = date(int(g[6]), int(g[7]), int(g[8]))
            stop = start + timedelta(days=1)
        else:
            start = date(int(g[9]), int(g[10]), 1)
            stop = _months_on(start, 1)
    except (ValueError, OverflowError):
        return None
    return tuple(datetime(d.year, d.month, d.day, tzinfo=UTC) for d in (start, stop))


def _months_on(day: date, months: int) -> date:
    """The first day of the month ``months`` after ``day``'s."""
    month = day.month - 1 + months
    return date(day.year + month // 12, month % 12 + 1, 1)


_stamp = functools.lru_cache(maxsize=4096)(datetime.fromisoformat)  # ponytail: a dataset's bins repeat across its patterns


def time_score(trajectory: dict[str, Any], asked: tuple[datetime, datetime]) -> float:
    """Half the share of the asked interval its metric's measured bins cover, plus a half when the bin of one of its
    events lies inside it — an event in the period outranks mere presence through it, and a coarse event (a month's)
    is no event of each finer period it spans (``seeds.SEED_TIME_WEIGHT`` weighs the score). The period is read on
    the dataset's calendar (its ``tz``): the same wall-clock bounds in its zone, against the bounds the trajectory
    stores for its bins."""
    zone = ZoneInfo(trajectory["tz"]) if trajectory.get("tz") else UTC
    lo, hi = (t.replace(tzinfo=zone) for t in asked)
    bins = [(_stamp(a), _stamp(b)) for a, b in trajectory["bins"]]
    measured = [bins[t] for t, e in enumerate(trajectory["series"].get(trajectory["metric"], [])) if e]
    share = sum(max((min(b, hi) - max(a, lo)).total_seconds(), 0.0) for a, b in measured) / (hi - lo).total_seconds()
    inside = any(lo <= bins[e["bin"]][0] and bins[e["bin"]][1] <= hi for e in trajectory["events"])
    return 0.5 * share + (0.5 if inside else 0.0)


def time_facts(trajectory: dict[str, Any] | None, fade_z: float) -> dict[str, Any]:
    """What an insight's time line says (empty for an insight of a dataset without a time column): its grid and span,
    its metric's trend (None when no bin was measured) and current level (``faded`` under ``fade_z``), its core (the
    run of bins holding most of its rows, as the trajectory holds it; its age from the reference date is added by the
    evidence), its own events —
    the strongest first, each with the condition that carries it when one does and ``outside: True`` when its bin is not
    in the core —, the ``echoes`` (the events a broader insight has too, told once, there), the bins measured and all of it
    in words (``wording.time_words``), as the evidence-only answer and the console say it."""
    if not trajectory:
        return {}
    events = sorted(trajectory["events"], key=lambda e: -e["prominence"])
    current, core = trajectory["current"], trajectory.get("core")
    facts = {
        "grid": trajectory["grid"],
        "span": trajectory["span"],
        "metric": trajectory["metric"],
        "trend": trajectory["trend"] if trajectory["bins_measured"] else None,  # a series never measured has no trend
        "current": current,
        "faded": current is not None and abs(current) < fade_z,
        "core": core,
        "events": [
            {k: e[k] for k in ("kind", "period", "metric", "z_smoothed", "carrier") if k in e}
            | ({"outside": True} if core and not core["first"] <= e["bin"] <= core["last"] else {})
            for e in events
            if "echo_of" not in e
        ],
        "echoes": [{k: e[k] for k in ("kind", "period", "metric", "echo_of")} for e in events if "echo_of" in e],
        "bins_measured": trajectory["bins_measured"],
        "of": trajectory["of"],
        "gaps": trajectory["gaps"],
    }
    return facts | {"words": wording.time_words(facts)}


def time_line(facts: dict[str, Any]) -> str:
    """The ``time:`` line of an evidence item: the span's first and last day, the trend with its Ukrainian name (the word
    rule 13 asks the answer to use), the current level, the core and its age from the reference date, the events — each
    with the condition that carries it, and whether it fell outside the core —, the echoes of a broader finding's events
    and the bins."""
    start, end = (t[:10] for t in facts["span"])
    trend = f"{facts['trend']} «{wording.name('trend_' + facts['trend'])}»" if facts["trend"] else f"not measured «{wording.name('time_unmeasured')}»"
    current = "not measured" if facts["current"] is None else f"{facts['current']:+.2f} sd" + (" (below the threshold)" if facts["faded"] else "")
    kinds = {"PEAKED_AT": "peak", "DROPPED_AT": "dip"}

    def where(e: dict[str, Any]) -> str:
        c = e.get("carrier")
        found = f", in {c['attribute']} = {c['value']}: {c['rows']} of its {c['of']} rows, {c['shift']:+.2f} sd against the rest" if c else ""
        return found + (", outside its core" if e.get("outside") else "")

    events = (
        "; ".join(f"{kinds[e['kind']]} {e['period']} ({e['metric']}, {e['z_smoothed']:+.2f} sd{where(e)})" for e in facts["events"][:3]) or "none"
    )
    echoes = "; ".join(f"{kinds[e['kind']]} {e['period']} ({e['metric']})" for e in facts.get("echoes", [])[:3])
    echo = f" | the same as a broader finding's: {echoes}" if echoes else ""
    gaps = f" | unmeasured: {', '.join(a if a == b else f'{a}–{b}' for a, b in facts['gaps'])}" if facts["gaps"] else ""
    return (
        f"grid {facts['grid']} | span [{start}, {end}] | trend {trend} ({facts['metric']}) | current {current}{_core(facts)}"
        f" | events: {events}{echo} | bins {facts['bins_measured']}/{facts['of']} measured{gaps}"
    )


def _core(facts: dict[str, Any]) -> str:
    """The time line's core: its periods, the bins and the share of the finding's timed rows they hold, and its age."""
    core = facts.get("core")
    if not core:
        return " | core: none (no timed rows)" if "core" in facts else ""
    periods = core["start"] if core["start"] == core["end"] else f"{core['start']}–{core['end']}"
    bins = f"{core['bins']} bin{'s' if core['bins'] > 1 else ''}"
    line = f" | core {periods} ({bins}, {core['share']:.0%} of its timed rows)"
    age = facts.get("age_days")
    if age is None:
        return line
    days = math.ceil(abs(age))
    if age > 0:
        return f"{line}, ended {days} days before the reference date"
    return f"{line}, starts {days} days after the reference date" if age < 0 else f"{line}, the reference date inside it"
