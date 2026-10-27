"""Time in one upload (docs/02_discovery.md §2.10): the calendar grid of its time column and the dynamics of each
admitted insight over it.

Discovery never sees time: ``load_dataset(time=)`` keeps the column apart and the search runs once over every row.
After admission each insight's covered rows are binned by the grid (``calendar_bins``) and every metric it shifts gets
a series of robust z against its bin's population (``annotate``). ``dynamics`` smooths a series (``rts_smooth``: a
local-level Kalman filter and its Rauch–Tung–Striebel pass, each bin's sampling noise its measurement noise), keeps the
extrema a deterministic gate passes (``detect_events``) and names the trend. All functions are pure.
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
from insight_contracts import EdgeType
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
    population = {}
    for metric in {m for ms in metrics.values() for m in ms if scale.get(m)}:
        values = frame[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        tied = bool(np.unique(finite, return_counts=True)[1].max() >= tie_share * len(finite))  # one value that often: a lump
        population[metric] = (values, tied, (float(np.std(finite)) if tied else scale[metric] * 0.6745) or scale[metric])
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
