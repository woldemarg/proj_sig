"""Time in one upload (docs/02_discovery.md §2.10): the calendar grid of its time column and the dynamics of each
admitted insight over it.

Discovery never sees time: ``load_dataset(time=)`` keeps the column apart and the search runs once over every row.
After admission an ``auto`` grid is fitted to the insights (``fit_grid``), each insight's covered rows are binned by the
grid (``calendar_bins``) and every metric it shifts gets a series of robust z against its bin's population
(``annotate``). ``dynamics`` smooths a series (``rts_smooth``: a local-level Kalman filter and its Rauch–Tung–Striebel
pass, each bin's sampling noise its measurement noise), keeps the extrema a deterministic gate passes
(``detect_events``) and names the trend; ``carriers`` names the condition inside an insight that carries an event,
``echoes`` the events a broader insight has too. All functions are pure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import norm

import subgroup_miner.vendor.eda.main_upd as eda
from insight_contracts import TIME_GRIDS, EdgeType
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import _MEDIAN_SE_FACTOR
from subgroup_miner.ingestion import GRID_FREQ, IngestionError

RISING, FALLING, FLAT, SPIKE = "rising", "falling", "flat", "spike"
PEAKED_AT, DROPPED_AT = EdgeType.PEAKED_AT.value, EdgeType.DROPPED_AT.value


def period_label(period: pd.Period, grid: str) -> str:
    """A bin's calendar name: ``2026-08-14``, ``2026-W35`` (ISO year and week), ``2026-08``, ``2026-Q3``."""
    if grid == "week":
        year, week, _ = period.start_time.isocalendar()
        return f"{year}-W{week:02d}"
    if grid == "quarter":
        return f"{period.year}-Q{period.quarter}"
    return period.strftime("%Y-%m-%d" if grid == "day" else "%Y-%m")


def calendar_bins(times: pd.Series, grid: str, max_bins: int | None = None, tz: str | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Each row's bin (-1 for a row without a time) and the bins from the first time's period to the last's, empty ones
    included, each ``{period, t_start, t_end, rows, covered, partial}`` (half-open; the calendar of ``tz``, UTC by
    default, its bounds written with that zone's offset). ``covered``: the share of the
    bin's days the upload's times span (1 but for its first and last bin); ``partial``: under 1. More bins than
    ``max_bins`` is ``invalid_options``, before any is built: one placeholder date (``9999-12-31``) would make hundreds
    of thousands."""
    known = times.notna().to_numpy()
    if not known.any():  # no time reads: no bin (the upload's check refuses it)
        return np.full(len(times), -1, dtype=np.int64), []
    naive = times[known].dt.tz_convert(tz or "UTC").dt.tz_localize(None)  # the wall times of the calendar
    periods = naive.dt.to_period(GRID_FREQ[grid])
    first, last = periods.min(), periods.max()
    if max_bins and (count := last.ordinal - first.ordinal + 1) > max_bins:
        raise IngestionError("invalid_options", f"the time column spans {count} {grid} bins, from {first} to {last} (TEMPORAL_MAX_BINS={max_bins})")
    span = pd.period_range(first, last, freq=GRID_FREQ[grid])
    codes = np.full(len(times), -1, dtype=np.int64)
    codes[known] = periods.array.asi8 - first.ordinal
    rows = np.bincount(codes[known], minlength=len(span))
    lo, hi = naive.min().normalize(), naive.max().normalize() + pd.Timedelta(days=1)  # the days the upload spans
    bins = []
    for i, p in enumerate(span):
        start, end = p.start_time, (p + 1).start_time
        covered = (min(end, hi) - max(start, lo)) / (end - start)
        bins.append(
            {
                "period": period_label(p, grid),
                "t_start": start.tz_localize(tz or "UTC", ambiguous=True, nonexistent="shift_forward").isoformat(),
                "t_end": end.tz_localize(tz or "UTC", ambiguous=True, nonexistent="shift_forward").isoformat(),
                "rows": int(rows[i]),
                "covered": round(float(covered), 3),
                "partial": bool(covered < 1),
            }
        )
    return codes, bins


def fit_grid(times: pd.Series, grid: str, covers: Mapping[str, np.ndarray], min_rows: int, min_bins: int, tz: str | None = None) -> str:
    """``auto``'s unit once the insights are admitted: ``grid`` (``resolve_grid``'s, which weighs the upload's rows, not
    an insight's — 3 % of 20,000 rows over 35 weeks is 17 a week, under ``min_rows``) or the first coarser unit at which
    the median insight is measurable in at least half the bins its span covers by half — ``min_rows`` of its rows there
    and as many of the rest (``annotate``'s floor). A unit with fewer than ``min_bins`` bins holding rows and covered by
    half (``check_time``'s) is never taken: with none fit, the coarsest one that has them. ``covers``: pattern id → its
    rows (positions in ``times``)."""
    fit = grid
    for unit in TIME_GRIDS[TIME_GRIDS.index(grid) :]:
        codes, bins = calendar_bins(times, unit, tz=tz)
        whole, rows = np.array([b["covered"] >= 0.5 for b in bins]), np.array([b["rows"] for b in bins])
        if (whole & (rows > 0)).sum() < max(min_bins, 1):
            break
        fit = unit
        own = [np.bincount(c[c >= 0], minlength=len(bins)) for c in (codes[r] for r in covers.values())]
        if np.median([(whole & (o >= min_rows) & (rows - o >= min_rows)).sum() / whole.sum() for o in own]) >= 0.5:
            break
    return fit


def reference_scale(frame: pd.DataFrame) -> dict[str, float | None]:
    """Each numeric column's robust scale over the whole upload, s_ref = MAD / 0.6745 (the EDA's MAD, with its fallback):
    the unit of every series. None for a constant column."""
    out: dict[str, float | None] = {}
    for name in frame.select_dtypes(include=[np.number]).columns:
        mad = eda.calculate_mad(frame[name].dropna())
        out[str(name)] = mad / 0.6745 if mad > 0 else None
    return out


def _by_bin(codes: np.ndarray, values: np.ndarray, count: int) -> list[np.ndarray]:
    """The finite values of each bin, in bin order (rows without a time left out)."""
    keep = (codes >= 0) & np.isfinite(values)
    c, v = codes[keep], values[keep]
    order = np.argsort(c, kind="stable")
    return np.split(v[order], np.searchsorted(c[order], np.arange(1, count)))


def _level(values: np.ndarray, tied: bool, fallback: float) -> tuple[float, float]:
    """A bin's level and its standard error: the mean and s / √n for a tie-heavy metric, else the median and
    1.2533 · 1.4826 · MAD / √n; ``fallback`` (the upload's s or MAD) stands in for a bin's 0 — never a zero error."""
    n = math.sqrt(len(values))
    if tied:
        return float(np.mean(values)), (float(np.std(values)) or fallback) / n
    med = float(np.median(values))
    return med, _MEDIAN_SE_FACTOR * (float(np.median(np.abs(values - med))) or fallback) / n


def _population(frame: pd.DataFrame, metric: str, scale: float, tie_share: float) -> tuple[np.ndarray, bool, float]:
    """A metric's values over the upload, whether ``tie_share`` of them is one value (a lump: levels are means) and the
    spread that stands in for a bin's zero (``_level``)."""
    values = frame[metric].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    tied = bool(np.unique(finite, return_counts=True)[1].max() >= tie_share * len(finite))  # one value that often: a lump
    return values, tied, (float(np.std(finite)) if tied else scale * 0.6745) or scale


def annotate(
    frame: pd.DataFrame,
    codes: np.ndarray,
    bins: Sequence[Mapping[str, Any]],
    covers: Mapping[str, np.ndarray],
    metrics: Mapping[str, Sequence[str]],
    scale: Mapping[str, float | None],
    min_rows: int,
    tie_share: float,
) -> dict[str, dict[str, Any]]:
    """Each admitted insight's series over the grid. ``covers``: pattern id → its rows (positions in ``frame``);
    ``metrics``: pattern id → the metrics it shifts; ``scale``: ``reference_scale`` of the upload. Per pattern
    ``{rows: [its rows in each bin], untimed: its rows without a time, series: {metric: [{z, n, se} | None, …]}}``;
    ``sum(rows) + untimed`` is its support. Per bin t of a metric k, S the insight's rows in the bin and R the rest of
    the bin's rows (an insight that is most of its bin would otherwise move its own reference):

        z = (c_S(t) − c_R(t)) / s_ref        se = sqrt(SE_S² + SE_R²) / s_ref

    c and SE are a bin's level and its standard error (``_level``): the median's, or, for a metric ``tie_share`` of whose
    values are one value (a discrete one, whose median jumps between the values it takes), the mean's. A bin holding
    fewer than ``min_rows`` rows of the upload, of the insight or of the rest, or covering less than half its period
    (the upload's first or last bin), is None — never a z on an unstable level. A metric without a scale has no series."""
    min_rows = max(min_rows, 1)  # a bin needs a value to have a level
    population = {m: _population(frame, m, scale[m], tie_share) for m in {m for ms in metrics.values() for m in ms if scale.get(m)}}
    sparse = [b["rows"] < min_rows or b["covered"] < 0.5 for b in bins]
    out = {}
    for pid, rows in covers.items():
        own, rest = codes[rows], codes.copy()
        rest[rows] = -1  # the bin's rows outside the insight
        series = {}
        for metric in metrics[pid]:
            if metric not in population:
                continue
            values, tied, fallback = population[metric]
            entries = []
            for t, (sub, ref) in enumerate(zip(_by_bin(own, values[rows], len(bins)), _by_bin(rest, values, len(bins)))):
                if sparse[t] or len(sub) < min_rows or len(ref) < min_rows:
                    entries.append(None)
                    continue
                (c_s, se_s), (c_r, se_r) = _level(sub, tied, fallback), _level(ref, tied, fallback)
                entries.append({"z": (c_s - c_r) / scale[metric], "n": len(sub), "se": math.hypot(se_s, se_r) / scale[metric]})
            series[metric] = entries
        timed = own[own >= 0]
        out[pid] = {"rows": np.bincount(timed, minlength=len(bins)).tolist(), "untimed": int((own < 0).sum()), "series": series}
    return out


def extent_points(point_cell: np.ndarray, covers: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each geo pattern's points: those whose cell (``point_cell``, the point → cell row map the geo ingest stored) is one
    of the cell rows it covers — never a point placed again."""
    return {pid: np.flatnonzero(np.isin(point_cell, rows)) for pid, rows in covers.items()}


def rts_smooth(y: Sequence[float | None], r: Sequence[float | None], q: float) -> tuple[list[float | None], list[float | None]]:
    """The local-level model x_t = x_{t−1} + w (Var w = ``q``), y_t = x_t + v (Var v = ``r[t]``) over the bins from the first
    measured one to the last: a Kalman filter (x₀ = y₀, P₀ = R₀; a missing bin, None, only predicts: P += Q) and the
    Rauch–Tung–Striebel pass back. Returns the smoothed level x̂ and its variance P̂; None outside the measured span."""
    measured = [t for t, v in enumerate(y) if v is not None]
    xs: list[float | None] = [None] * len(y)
    ps: list[float | None] = [None] * len(y)
    if not measured:
        return xs, ps
    lo, hi = measured[0], measured[-1]
    xf, pf, xp, pp = {}, {}, {}, {}
    xf[lo], pf[lo] = float(y[lo]), float(r[lo])
    for t in range(lo + 1, hi + 1):
        xp[t], pp[t] = xf[t - 1], pf[t - 1] + q
        if y[t] is None:
            xf[t], pf[t] = xp[t], pp[t]
        else:
            gain = pp[t] / (pp[t] + r[t])
            xf[t], pf[t] = xp[t] + gain * (y[t] - xp[t]), (1 - gain) * pp[t]
    xs[hi], ps[hi] = xf[hi], pf[hi]
    for t in range(hi - 1, lo - 1, -1):
        c = pf[t] / pp[t + 1]
        xs[t] = xf[t] + c * (xs[t + 1] - xp[t + 1])
        ps[t] = pf[t] + c * c * (ps[t + 1] - pp[t + 1])
    return xs, ps


def detect_events(x: Sequence[float | None], p: Sequence[float | None], measured: Sequence[int], alpha: float, excess: float) -> list[dict[str, Any]]:
    """The extrema of the smoothed level that pass the gate, over the measured bins. A peak (a dip: the same on −x̂) is
    a local maximum, a plateau's middle for several equal bins; the first and the last measured bin never are — what
    follows them is unknown, so a monotone series has no event. Its prominence is topographic (``find_peaks``): on each
    side the base is the lowest bin before a higher one (or the series' end), and it is an event when both sides clear

        x̂_t − x̂_base ≥ max(excess, k · sqrt(P̂_t + P̂_base)),   k = Φ⁻¹(1 − alpha / (T(T − 1))),  T the measured bins

    so a hump of several bins is one event at its top. The base is chosen after looking, so the family is every pair of
    bins, T(T − 1)/2, two-sided. Each: ``{bin, kind (PEAKED_AT | DROPPED_AT), z_smoothed, prominence (the smaller side),
    p (the weaker side's, × T(T − 1), capped at 1)}``."""
    count = len(measured)
    if count < 3:
        return []
    k = float(norm.isf(alpha / (count * (count - 1))))
    level, var = np.array([x[t] for t in measured]), np.array([p[t] for t in measured])
    events = []
    for sign, kind in ((1.0, PEAKED_AT), (-1.0, DROPPED_AT)):
        peaks, props = find_peaks(sign * level, prominence=excess)
        for j, *bases in zip(peaks, props["left_bases"], props["right_bases"]):
            sides = [(sign * (level[j] - level[b]), var[j] + var[b]) for b in bases]
            if all(d >= k * math.sqrt(v) for d, v in sides):
                events.append(
                    {
                        "bin": measured[j],
                        "kind": kind,
                        "z_smoothed": float(level[j]),
                        "prominence": float(min(d for d, _ in sides)),
                        "p": float(min(1.0, count * (count - 1) * max(norm.sf(d / math.sqrt(v)) for d, v in sides))),
                    }
                )
    return sorted(events, key=lambda e: e["bin"])


def trend(x: Sequence[float | None], p: Sequence[float | None], measured: Sequence[int], alpha: float, excess: float, events: bool) -> str:
    """``rising`` / ``falling``: the smoothed level moved from the first measured bin to the last by at least ``excess``,
    significantly at ``alpha`` against sqrt(P̂_first + P̂_last) — a dip that recovers or a hump that falls back moved
    nowhere; else ``spike`` when the series has events, ``flat`` otherwise."""
    first, last = measured[0], measured[-1]
    change = x[last] - x[first]
    if first != last and abs(change) >= excess and abs(change) >= norm.isf(alpha / 2) * math.sqrt(p[first] + p[last]):
        return RISING if change > 0 else FALLING
    return SPIKE if events else FLAT


def dynamics(entry: Mapping[str, Any], bins: Sequence[Mapping[str, Any]], metrics: Sequence[str], config: MinerConfig) -> dict[str, Any]:
    """A pattern's trajectory from its ``annotate`` entry: per metric the measured series and the smoothed level, and the
    events of every metric. Its ``metric`` is the first of ``metrics`` (the insight's target, then its shifts) with a
    measured bin — a geo pattern's counter or share has no series —, and of it the trend, the current level (x̂ of its
    last measured bin), the measured bins and the gaps (runs of bins without a measurement, by period)."""
    th = config.thresholds
    series, smoothed, events = {}, {}, []
    measured_in = {m: [t for t, e in enumerate(entry["series"][m]) if e] for m in entry["series"]}
    primary = next((m for m in metrics if measured_in.get(m)), metrics[0])
    out: dict[str, Any] = {"metric": primary, "trend": FLAT, "current": None, "bins_measured": 0, "of": len(bins), "gaps": []}
    for metric, entries in entry["series"].items():
        y = [e["z"] if e else None for e in entries]
        x, p = rts_smooth(y, [e["se"] ** 2 if e else None for e in entries], config.temporal_process_noise)
        measured = measured_in[metric]
        found = detect_events(x, p, measured, config.temporal_event_p, th.temporal_event_excess)
        events += [{"metric": metric, "period": bins[e["bin"]]["period"], "z_raw": y[e["bin"]], **e} for e in found]
        series[metric] = [{"period": bins[t]["period"], "z": e["z"], "se": e["se"]} if e else None for t, e in enumerate(entries)]
        smoothed[metric] = x
        if metric == primary and measured:
            out |= {
                "trend": trend(x, p, measured, config.temporal_event_p, th.temporal_event_excess, bool(found)),
                "current": x[measured[-1]],
                "bins_measured": len(measured),
                "gaps": _gaps(y, bins),
            }
    return {"series": series, "smoothed": smoothed, "events": sorted(events, key=lambda e: (e["bin"], e["metric"])), **out}


def _gaps(y: Sequence[float | None], bins: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """The runs of bins without a measurement, as ``[first period, last period]``."""
    missing = [t for t, v in enumerate(y) if v is None]
    runs = np.split(np.array(missing, dtype=int), np.flatnonzero(np.diff(missing) != 1) + 1) if missing else []
    return [[bins[r[0]]["period"], bins[r[-1]]["period"]] for r in runs]


def carriers(
    frame: pd.DataFrame,
    codes: np.ndarray,
    covers: Mapping[str, np.ndarray],
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    scale: Mapping[str, float | None],
    config: MinerConfig,
    dimensions: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Per pattern, the condition ``a = v`` of one of the ``dimensions`` (the columns the search groups by) that carries
    each of its events (``dynamics``'): of the insight's rows, those with a = v (in) moved in the event's bin t against
    its other rows (out) beyond how the two differ over its other timed rows (¬t) — a difference in differences of their
    levels (``_level``) — and moved themselves, against the rest of the upload (R, the reference of the series):

        d = ((c_in(t) − c_out(t)) − (c_in(¬t) − c_out(¬t))) / s_ref
        se = sqrt(SE_in(t)² + SE_out(t)² + SE_in(¬t)² + SE_out(¬t)²) / s_ref
        m = ((c_in(t) − c_R(t)) − (c_in(¬t) − c_R(¬t))) / s_ref

    each of the four holding ``temporal_min_rows_bin`` values. Of the conditions whose d and m both clear the least
    event excess in the event's direction — the part moved, and more than the rest of the insight (out moving the other
    way alone moves in by d, never by m) — the one with the largest d / se carries it when d / se clears Φ⁻¹(1 − p / N),
    N every condition tried (Bonferroni: every value of every dimension is). A shift the whole insight shares has none;
    a cohort the one global search could not mine, alive for a bin inside a broader insight, is named through it
    (docs/02_discovery.md §2.10). Each ``{metric, bin, attribute, value, rows (its rows in the bin), of (the insight's),
    shift (d), p}``."""
    dims = {c: pd.factorize(frame[c]) for c in dimensions}
    min_rows, excess = config.temporal_min_rows_bin, config.thresholds.temporal_event_excess
    population: dict[str, tuple[np.ndarray, bool, float]] = {}
    out: dict[str, list[dict[str, Any]]] = {}
    for pid, found in events.items():
        rows = covers[pid]
        own, others = codes[rows], np.ones(len(codes), dtype=bool)
        others[rows] = False
        for e in found:
            metric, sign, s = e["metric"], 1.0 if e["kind"] == PEAKED_AT else -1.0, scale[e["metric"]]
            if metric not in population:
                population[metric] = _population(frame, metric, s, config.temporal_tie_share)
            values, tied, fallback = population[metric]
            v, finite = values[rows], others & np.isfinite(values)
            at, rest = (own == e["bin"]) & np.isfinite(v), (own >= 0) & (own != e["bin"]) & np.isfinite(v)
            ref_t, ref_o = (_level(values[finite & m], tied, fallback)[0] for m in (codes == e["bin"], (codes >= 0) & (codes != e["bin"])))
            tried, kept = 0, []
            for attribute, (labels, uniques) in dims.items():
                label = labels[rows]
                counts = np.bincount(label[at & (label >= 0)], minlength=len(uniques))
                for k in np.flatnonzero((counts >= min_rows) & (at.sum() - counts >= min_rows)):  # a split the bin can level
                    inside = label == k
                    groups = [v[at & inside], v[at & ~inside], v[rest & inside], v[rest & ~inside]]
                    if min(map(len, groups)) < min_rows:
                        continue
                    tried += 1
                    (c1, s1), (c2, s2), (c3, s3), (c4, s4) = (_level(g, tied, fallback) for g in groups)
                    d = ((c1 - c2) - (c3 - c4)) / s
                    if sign * d >= excess and sign * ((c1 - ref_t) - (c3 - ref_o)) / s >= excess:
                        kept.append((sign * d * s / math.sqrt(s1**2 + s2**2 + s3**2 + s4**2), d, attribute, uniques[k], len(groups[0])))
            if not kept:
                continue
            stat, d, attribute, value, n = max(kept, key=lambda t: t[0])
            if (p := min(1.0, tried * float(norm.sf(stat)))) <= config.temporal_event_p:
                found_at = {"metric": metric, "bin": e["bin"], "attribute": attribute, "value": str(value), "rows": n, "of": int(at.sum())}
                out.setdefault(pid, []).append(found_at | {"shift": d, "p": p})
    return out


def echoes(trajectories: Mapping[str, dict[str, Any]], patterns: Mapping[str, Mapping[str, Any]]) -> None:
    """Marks each event a broader insight of the same upload has too — the same metric, bin and kind, of an insight whose
    conditions are a strict subset of this one's, so whose rows hold its rows — with ``echo_of``: the broadest such
    insight (the fewest conditions, then the most rows, then the id). One shift nested cohorts all read is one event to
    a reader, and ``carriers`` says where in the broadest it sits. ``patterns``: id → its ``conditions`` and ``support``."""
    scope = {pid: frozenset((c["attribute"], c["value"]) for c in patterns[pid]["conditions"]) for pid in trajectories}
    fired: dict[tuple[str, int, str], list[str]] = {}
    for pid, t in trajectories.items():
        for e in t["events"]:
            fired.setdefault((e["metric"], e["bin"], e["kind"]), []).append(pid)
    for pid, t in trajectories.items():
        for e in t["events"]:
            if broader := [q for q in fired[(e["metric"], e["bin"], e["kind"])] if scope[q] < scope[pid]]:
                e["echo_of"] = min(broader, key=lambda q: (len(scope[q]), -patterns[q]["support"], q))
