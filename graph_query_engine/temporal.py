"""Time in the answer (docs/07_question_answering.md §7.2, §7.4): an insight of a dataset with a time column carries
its time line — the grid, the span, the trend and current level of its metric (its target, unless only a shift has a
series), its events and the bins measured — from the snapshot's ``trajectories``; a calendar period in the question
raises the seed score of the patterns measured in it."""

from __future__ import annotations

import functools
from datetime import UTC, date, datetime, timedelta
from typing import Any

from insight_contracts import wording


@functools.lru_cache(maxsize=8192)  # ponytail: a grid's labels repeat across patterns and questions
def period_interval(text: str) -> tuple[datetime, datetime] | None:
    """The UTC interval [start, end) of the first calendar period written in ``text`` the way the grid names one —
    ``2031-W13`` (an ISO week), ``2031-Q1`` or ``Q1 2031``, ``2031-03-24`` (a time after it allowed), ``2031-03`` —
    (``wording.PERIOD``); a token that only looks like one (``2031-W54``, ``2019-20``) or whose period runs past the
    calendar (``9999-12-31``) is passed over. None when there is none."""
    for m in wording.PERIOD.finditer(text):
        g = m.groups()
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
            continue
        return tuple(datetime(d.year, d.month, d.day, tzinfo=UTC) for d in (start, stop))
    return None


def _months_on(day: date, months: int) -> date:
    """The first day of the month ``months`` after ``day``'s."""
    month = day.month - 1 + months
    return date(day.year + month // 12, month % 12 + 1, 1)


def time_score(trajectory: dict[str, Any], asked: tuple[datetime, datetime]) -> float:
    """How much of the asked interval a pattern was measured in: the share of it its metric's measured bins cover,
    doubled when one of its events falls in it, at most 1 (``seeds.SEED_TIME_WEIGHT`` weighs it)."""

    def overlap(period: str) -> float:
        span = period_interval(period)
        return max((min(span[1], asked[1]) - max(span[0], asked[0])).total_seconds(), 0.0) if span else 0.0

    measured = [e["period"] for e in trajectory["series"].get(trajectory["metric"], []) if e]
    share = sum(map(overlap, measured)) / (asked[1] - asked[0]).total_seconds()
    return min(1.0, share * (2 if any(overlap(e["period"]) for e in trajectory["events"]) else 1))


def time_facts(trajectory: dict[str, Any] | None, fade_z: float) -> dict[str, Any]:
    """What an insight's time line says (empty for an insight of a dataset without a time column): its grid and span,
    its metric's trend (None when no bin was measured) and current level (``faded`` under ``fade_z``), its events — the
    strongest first —, the bins measured and all of it in words (``wording.time_words``), as the evidence-only answer and the console say it."""
    if not trajectory:
        return {}
    events = sorted(trajectory["events"], key=lambda e: -e["prominence"])
    current = trajectory["current"]
    facts = {
        "grid": trajectory["grid"],
        "span": trajectory["span"],
        "metric": trajectory["metric"],
        "trend": trajectory["trend"] if trajectory["bins_measured"] else None,  # a series never measured has no trend
        "current": current,
        "faded": current is not None and abs(current) < fade_z,
        "events": [{k: e[k] for k in ("kind", "period", "metric", "z_smoothed")} for e in events],
        "bins_measured": trajectory["bins_measured"],
        "of": trajectory["of"],
        "gaps": trajectory["gaps"],
    }
    return facts | {"words": wording.time_words(facts)}


def time_line(facts: dict[str, Any]) -> str:
    """The ``time:`` line of an evidence item."""
    start, end = (t[:10] for t in facts["span"])
    trend = facts["trend"] or "not measured"
    current = "not measured" if facts["current"] is None else f"{facts['current']:+.2f} sd" + (" (below the threshold)" if facts["faded"] else "")
    kinds = {"PEAKED_AT": "peak", "DROPPED_AT": "dip"}
    events = "; ".join(f"{kinds[e['kind']]} {e['period']} ({e['metric']}, {e['z_smoothed']:+.2f} sd)" for e in facts["events"][:3]) or "none"
    gaps = f" | unmeasured: {', '.join(a if a == b else f'{a}–{b}' for a, b in facts['gaps'])}" if facts["gaps"] else ""
    return (
        f"grid {facts['grid']} | span [{start}, {end}) | trend {trend} ({facts['metric']}) | current {current}"
        f" | events: {events} | bins {facts['bins_measured']}/{facts['of']} measured{gaps}"
    )
