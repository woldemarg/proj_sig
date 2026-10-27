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
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import _MEDIAN_SE_FACTOR

_FREQ = {"day": "D", "week": "W-SUN", "month": "M", "quarter": "Q"}  # a week runs Monday to Sunday, as ISO 8601's
RISING, FALLING, FLAT, SPIKE = "rising", "falling", "flat", "spike"
PEAKED_AT, DROPPED_AT = "PEAKED_AT", "DROPPED_AT"


def period_label(period: pd.Period, grid: str) -> str:
    """A bin's calendar name: ``2026-08-14``, ``2026-W35`` (ISO year and week), ``2026-08``, ``2026-Q3``."""
    if grid == "week":
        year, week, _ = period.start_time.isocalendar()
        return f"{year}-W{week:02d}"
    if grid == "quarter":
        return f"{period.year}-Q{period.quarter}"
    return period.strftime("%Y-%m-%d" if grid == "day" else "%Y-%m")


def calendar_bins(times: pd.Series, grid: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Each row's bin (-1 for a row without a time) and the bins from the first time's period to the last's, empty ones
    included, each ``{period, t_start, t_end, rows, partial}`` (UTC, half-open). ``partial``: the upload's times begin
    after the bin's first day or end before its last."""
    known = times.notna().to_numpy()
    naive = times[known].dt.tz_convert("UTC").dt.tz_localize(None)
    periods = naive.dt.to_period(_FREQ[grid])
    first = periods.min()
    span = pd.period_range(first, periods.max(), freq=_FREQ[grid])
    codes = np.full(len(times), -1, dtype=np.int64)
    codes[known] = periods.array.asi8 - first.ordinal
    rows = np.bincount(codes[known], minlength=len(span))
    lo, hi = naive.min().normalize(), naive.max().normalize()
    bins = []
    for i, p in enumerate(span):
        start, end = p.start_time, (p + 1).start_time
        bins.append(
            {
                "period": period_label(p, grid),
                "t_start": start.tz_localize("UTC").isoformat(),
                "t_end": end.tz_localize("UTC").isoformat(),
                "rows": int(rows[i]),
                "partial": bool((i == 0 and lo > start) or (i == len(span) - 1 and hi < end - pd.Timedelta(days=1))),
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


def _median_mad(values: np.ndarray, fallback: float) -> tuple[float, float]:
    """A bin's median and MAD, the upload's MAD where the bin's is 0 (a discrete metric: no zero standard error)."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad or fallback


def annotate(
    frame: pd.DataFrame,
    codes: np.ndarray,
    bins: Sequence[Mapping[str, Any]],
    covers: Mapping[str, np.ndarray],
    metrics: Mapping[str, Sequence[str]],
    scale: Mapping[str, float | None],
    min_rows: int,
) -> dict[str, dict[str, Any]]:
    """Each admitted insight's series over the grid. ``covers``: pattern id → its rows (positions in ``frame``);
    ``metrics``: pattern id → the metrics it shifts; ``scale``: ``reference_scale`` of the upload. Per pattern
    ``{rows: [its rows in each bin], untimed: its rows without a time, series: {metric: [{z, n, se} | None, …]}}``;
    ``sum(rows) + untimed`` is its support. Per bin t of a metric k:

        z = (med_S(t) − med_G(t)) / s_ref        se = sqrt(SE_S² + SE_G²) / s_ref,   SE = 1.2533 · 1.4826 · MAD / √n

    G is the bin's whole population; a bin holding fewer than ``min_rows`` rows (the whole upload's) or fewer than
    ``min_rows`` of the insight's values is None — never a z on an unstable median. A metric without a scale has no series."""
    min_rows = max(min_rows, 1)  # a bin needs a value to have a median
    population = {}
    for metric in {m for ms in metrics.values() for m in ms if scale.get(m)}:
        values = frame[metric].to_numpy(dtype=float)
        fallback = scale[metric] * 0.6745
        population[metric] = (values, fallback, [_median_mad(g, fallback) + (len(g),) if len(g) else None for g in _by_bin(codes, values, len(bins))])
    sparse = [b["rows"] < min_rows for b in bins]
    out = {}
    for pid, rows in covers.items():
        own = codes[rows]
        series = {}
        for metric in metrics[pid]:
            if metric not in population:
                continue
            values, fallback, base = population[metric]
            entries = []
            for t, sub in enumerate(_by_bin(own, values[rows], len(bins))):
                if sparse[t] or len(sub) < min_rows or base[t] is None:
                    entries.append(None)
                    continue
                med_s, mad_s = _median_mad(sub, fallback)
                med_g, mad_g, n_g = base[t]
                se = math.hypot(_MEDIAN_SE_FACTOR * mad_s / math.sqrt(len(sub)), _MEDIAN_SE_FACTOR * mad_g / math.sqrt(n_g))
                entries.append({"z": (med_s - med_g) / scale[metric], "n": len(sub), "se": se / scale[metric]})
            series[metric] = entries
        timed = own[own >= 0]
        out[pid] = {"rows": np.bincount(timed, minlength=len(bins)).tolist(), "untimed": int((own < 0).sum()), "series": series}
    return out


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


def dynamics(entry: Mapping[str, Any], bins: Sequence[Mapping[str, Any]], primary: str, config: MinerConfig) -> dict[str, Any]:
    """A pattern's trajectory from its ``annotate`` entry: per metric the measured series and the smoothed level, the
    events of every metric, and of its ``primary`` metric (the insight's target) the trend, the current level (x̂ of its
    last measured bin), the measured bins and the gaps (runs of bins without a measurement, by period)."""
    th = config.thresholds
    series, smoothed, events = {}, {}, []
    out: dict[str, Any] = {"trend": FLAT, "current": None, "bins_measured": 0, "of": len(bins), "gaps": []}
    for metric, entries in entry["series"].items():
        y = [e["z"] if e else None for e in entries]
        x, p = rts_smooth(y, [e["se"] ** 2 if e else None for e in entries], config.temporal_process_noise)
        measured = [t for t, v in enumerate(y) if v is not None]
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
