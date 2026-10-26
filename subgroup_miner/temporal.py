"""Time windows of one upload and the trajectories through them (docs/02_discovery.md §2.10).

An upload with a time column is cut into fixed, non-overlapping windows; each window is mined as a dataset of its own,
and every cohort a window admits opens a *trajectory*: its rule (the shortest selector of its extent) is evaluated
directly in each later window (``run_discovery(known=)``), outside the EDA's ranking and budget. The functions here
are pure: what a window observed of each known rule, the fold of those observations into states, and the transition
between two windows that admitted the same rule. Nothing holds state between calls; the service keeps the
observations on the windows' batch records and the admitted nodes in its journal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import subgroup_miner.vendor.eda.main_upd as eda
from insight_contracts import Condition, Insight, Rejection, stable_hash
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import DiscoveryResult, KnownRule
from subgroup_miner.ingestion import IngestionError, upload_table

EVALUATED, NOT_EVALUATED, WINDOW_UNAVAILABLE = "EVALUATED", "NOT_EVALUATED", "WINDOW_UNAVAILABLE"
ACTIVE, DORMANT, CLOSED = "ACTIVE", "DORMANT", "CLOSED"
TERMINATED, TTL_EXPIRED_UNOBSERVED = "TERMINATED", "TTL_EXPIRED_UNOBSERVED"
CONTINUATION, DRIFT, RECOVERY, REAPPEARANCE = "CONTINUATION", "DRIFT", "RECOVERY", "REAPPEARANCE"
MIN_WINDOW = pd.Timedelta("1s")  # a size without a unit reads as nanoseconds: refused


def parse_window(spec: str) -> pd.Timedelta:
    """A fixed window size (``7D``, ``24h``, ``90min``); calendar units (months, years) have no fixed length."""
    try:
        size = pd.Timedelta(str(spec).strip())
    except (ValueError, TypeError) as exc:
        raise IngestionError("invalid_options", f"window {spec!r}: {exc}") from exc
    if size < MIN_WINDOW:
        raise IngestionError("invalid_options", f"window {spec!r}: give a positive duration with a unit, e.g. 7D or 24h")
    return size


def upload_windows(
    path: Path, config: MinerConfig, *, time: str, window: str, max_windows: int, bins: str | None = None, categories: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """The windows of a stored upload, ``[t0 + w·size, t0 + (w + 1)·size)`` from the day of its first time to its last,
    each ``{index, of, column, size, t_start, t_end, rows}`` (``size`` in ISO 8601, so ``7D`` and ``1W`` are one size), and
    its reference scale (``reference_scale``). The table is
    the one its windows read (``ingestion.upload_table``: bands over the whole upload)."""
    size = parse_window(window)
    table = upload_table(path, config, time=time, bins=bins, categories=categories)
    stamps = table.time.dropna()
    t0 = stamps.min().floor("D")
    count = int((stamps.max() - t0) // size) + 1
    if count > max_windows:
        raise IngestionError("invalid_options", f"window {window}: {count} windows over the upload's time span (MAX_WINDOWS={max_windows})")
    rows = np.bincount(((stamps - t0) // size).to_numpy(dtype=np.int64), minlength=count)
    windows = [
        {
            "index": w,
            "of": count,
            "column": time,
            "size": size.isoformat(),
            "t_start": (t0 + w * size).isoformat(),
            "t_end": (t0 + (w + 1) * size).isoformat(),
            "rows": int(rows[w]),
        }
        for w in range(count)
    ]
    return windows, reference_scale(table.frame)


def reference_scale(frame: pd.DataFrame) -> dict[str, float | None]:
    """Each numeric column's robust scale over the whole upload, s_ref = MAD / 0.6745: the unit every change between
    windows is measured in. None where the MAD is 0 even after the EDA's fallback (a constant column)."""
    out: dict[str, float | None] = {}
    for name in frame.select_dtypes(include=[np.number]).columns:
        mad = eda.calculate_mad(frame[name].dropna())
        out[str(name)] = mad / 0.6745 if mad > 0 else None
    return out


def series_id(upload_id: str, window: Mapping[str, Any]) -> str:
    """The window series a window belongs to: its upload cut on one time column by one size. One file cut otherwise is
    another series, with trajectories of its own."""
    return f"{upload_id}:{window['column']}:{window['size']}"


def trajectory_key(series: str, predicate: Iterable[Condition]) -> str:
    """A trajectory's id: its window series and the conditions its rule selects with."""
    return "T-" + stable_hash(series, [c.expr for c in sorted(predicate)])


def stamp_insights(insights: Iterable[Insight], result: DiscoveryResult, *, series: str, window: int) -> None:
    """Each insight's ``provenance.temporal``: its window series and window, whether its rule was known (``tracked``) or the EDA
    found it (``eda``), its trajectory, the rule and the metrics every node of the trajectory reports. A new rule is the
    shortest selector of the cohort's extent (``Candidate.generator``), not its closed intent: a condition that happens to
    hold on all of a window's rows is no part of the mechanism."""
    cohorts = {c.expression: c for c in (*result.validated, *result.tracked)}
    for ins in insights:
        cohort = cohorts[ins.expression]
        if cohort.known:
            rule, origin = cohort.known[0], "tracked"
        else:
            predicate = cohort.generator
            rule, origin = KnownRule(trajectory_key(series, predicate), predicate, tuple(s.metric for s in ins.shifts)), "eda"
        ins.provenance["temporal"] = {
            "series": series,
            "window": window,
            "origin": origin,
            "trajectory": rule.key,
            "predicate": [asdict(c) for c in rule.predicate],
            "metrics": list(rule.metrics),
        }


@dataclass
class Observation:
    """What one window found of one known rule. ``active``: admitted into the graph (``pattern_id`` its node); an
    evaluated rule that is not active carries the ``rejection`` that refused it; ``reason`` says why it was not
    evaluated. ``p_adjusted`` and ``stability`` are None for a covariance insight, as on the insight."""

    status: str
    active: bool = False
    reason: str | None = None
    rejection: str | None = None
    pattern_id: str | None = None
    support: int | None = None
    effect_size: float | None = None
    p_adjusted: float | None = None
    stability: float | None = None
    phenomenon_type: str | None = None


def observed(ins: Insight, active: bool, rejection: str | None = None) -> Observation:
    """An evaluated rule's observation from the insight its cohort became."""
    return Observation(
        EVALUATED,
        active=active,
        rejection=rejection,
        pattern_id=ins.id if active else None,
        support=ins.support,
        effect_size=ins.effect_size,
        p_adjusted=ins.p_adjusted,
        stability=ins.stability,
        phenomenon_type=ins.phenomenon_type,
    )


def observations(
    known: Iterable[KnownRule], result: DiscoveryResult, insights: Iterable[Insight], admitted: Iterable[Insight], rejections: Iterable[Rejection]
) -> dict[str, Observation]:
    """Each known rule's outcome in a window: not evaluated (and why), or evaluated and admitted or refused (and by which
    rule). ``insights``: what ``build_insights`` made of the window; ``admitted``: the ones the graph took."""
    built = {i.expression: i for i in insights}
    kept = {i.expression: i for i in admitted}
    why = {r.expression: r.reason for r in rejections}
    cohort = {rule.key: c.expression for c in result.tracked for rule in c.known}
    out = {}
    for rule in known:
        if rule.key in result.tracked_absent:
            reason, support = result.tracked_absent[rule.key]
            out[rule.key] = (
                Observation(EVALUATED, rejection=reason, support=support)
                if reason == "no_shift"
                else Observation(NOT_EVALUATED, reason=reason, support=support)
            )
        else:
            expression = cohort[rule.key]
            out[rule.key] = observed(kept[expression], True) if expression in kept else observed(built[expression], False, why.get(expression))
    return out


@dataclass
class WindowFacts:
    """One window as the fold reads it: its observations of the known rules and the nodes it admitted. A window that
    observed nothing (it failed before it was mined, or is gone) is no ``WindowFacts``: the fold is not given it."""

    observations: dict[str, Observation]
    nodes: list[Insight] = field(default_factory=list)


@dataclass
class Trajectory:
    key: str
    predicate: tuple[Condition, ...]
    metrics: tuple[str, ...]
    first_window: int
    state: str = ACTIVE
    closed_reason: str | None = None
    episode: int = 1
    windows: list[dict[str, Any]] = field(default_factory=list)  # one entry per window observed, in order
    edges: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)  # (from node, to node, EVOLVES_TO props)
    inactive_streak: int = 0
    unobserved_streak: int = 0
    last: tuple[int, Insight] | None = None  # the last admitted (window, node)


def transition(
    a: Insight,
    b: Insight,
    scale: Mapping[str, float | None],
    metrics: Iterable[str],
    config: MinerConfig,
    *,
    recovered: bool = False,
    reappeared: bool = False,
) -> tuple[str, dict[str, dict[str, float]], str | None]:
    """The change between two nodes of one trajectory, in reference-scale units, per metric the trajectory reports:
    the subgroup's median shift Δsub, the window population's Δbase and the excess δ = Δsub − Δbase; and its type —
    a new episode (REAPPEARANCE), a return from dormancy (RECOVERY), an excess at least ``TEMPORAL_DRIFT_EXCESS`` between
    two shift insights (DRIFT: a covariance insight's medians are not its phenomenon), else CONTINUATION."""
    shifts = {}
    for metric in metrics:
        s, sa, sb = scale.get(metric), a.shift_for(metric), b.shift_for(metric)
        if s and sa and sb:
            sub, base = (sb.local_median - sa.local_median) / s, (sb.global_median - sa.global_median) / s
            shifts[metric] = {"subgroup": sub, "baseline": base, "excess": sub - base}
    strongest = max(shifts, key=lambda m: abs(shifts[m]["excess"]), default=None)
    drift = (
        strongest is not None
        and a.phenomenon_type == b.phenomenon_type == "shift"
        and abs(shifts[strongest]["excess"]) >= config.temporal_drift_excess
    )
    kind = REAPPEARANCE if reappeared else RECOVERY if recovered else DRIFT if drift else CONTINUATION
    return kind, shifts, strongest if kind == DRIFT else None


def fold_trajectories(windows: Mapping[int, WindowFacts], upto: int, scale: Mapping[str, float | None], config: MinerConfig) -> dict[str, Trajectory]:
    """The trajectories of one upload after its first ``upto`` windows (a window missing from ``windows`` is
    unavailable). Per open trajectory and window: admitted → ACTIVE, linked to its previous node; evaluated but refused →
    DORMANT, closed (TERMINATED) after ``TEMPORAL_DORMANT_WINDOWS`` such windows since it was last admitted; not evaluated
    → unchanged, closed (TTL_EXPIRED_UNOBSERVED) after ``TEMPORAL_UNOBSERVED_WINDOWS`` in a row — a closure for want of
    data, not a finding that the phenomenon ended. A node whose trajectory is unknown opens it; one whose trajectory is
    closed opens its next episode (REAPPEARANCE)."""
    trajectories: dict[str, Trajectory] = {}
    windows = {int(k): v for k, v in windows.items()}  # a map read back from JSON keys its windows by strings
    for index in range(upto):
        seen = _seen(windows.get(index))
        for t in sorted((t for t in trajectories.values() if t.state != CLOSED), key=lambda t: t.key):
            obs, node = seen.get(t.key, (Observation(NOT_EVALUATED, reason=WINDOW_UNAVAILABLE), None))
            _observe(t, index, obs, node, scale, config)
        for key, (obs, node) in sorted(seen.items()):
            t = trajectories.get(key)
            if node is None or (t is not None and t.state != CLOSED):
                continue
            if t is None:
                temporal = node.provenance["temporal"]
                predicate = tuple(Condition(**c) for c in temporal["predicate"])
                t = trajectories[key] = Trajectory(key, predicate, tuple(temporal["metrics"]), first_window=index)
                t.windows.append({"index": index, "episode": 1, **asdict(obs)})
                t.last = (index, node)
            else:
                t.episode, t.state, t.closed_reason, t.inactive_streak, t.unobserved_streak = t.episode + 1, ACTIVE, None, 0, 0
                t.windows.append({"index": index, "episode": t.episode, **asdict(obs)})
                _link(t, index, node, scale, config, reappeared=True)
    return trajectories


def open_rules(trajectories: Mapping[str, Trajectory]) -> list[KnownRule]:
    """The rules a next window evaluates directly: those of the trajectories not closed, by key."""
    return [KnownRule(t.key, t.predicate, t.metrics) for _, t in sorted(trajectories.items()) if t.state != CLOSED]


def _seen(facts: WindowFacts | None) -> dict[str, tuple[Observation, Insight | None]]:
    """A window's observations by trajectory, each with its node when admitted; a node no observation names (a new
    rule) is an admitted observation of its own trajectory. Empty when the window is unavailable."""
    if facts is None:
        return {}
    nodes = {n.id: n for n in facts.nodes}
    out = {key: (obs, nodes.get(obs.pattern_id)) for key, obs in facts.observations.items()}
    for node in facts.nodes:
        key = node.provenance["temporal"]["trajectory"]
        if out.get(key, (None, None))[1] is None:
            out[key] = (observed(node, True), node)
    return out


def _observe(t: Trajectory, index: int, obs: Observation, node: Insight | None, scale: Mapping[str, float | None], config: MinerConfig) -> None:
    t.windows.append({"index": index, "episode": t.episode, **asdict(obs)})
    if node is not None:
        _link(t, index, node, scale, config, recovered=t.state == DORMANT)
        t.state, t.inactive_streak, t.unobserved_streak = ACTIVE, 0, 0
    elif obs.status == EVALUATED:
        t.state, t.inactive_streak, t.unobserved_streak = DORMANT, t.inactive_streak + 1, 0
        if t.inactive_streak >= config.temporal_dormant_windows:
            t.state, t.closed_reason = CLOSED, TERMINATED
    else:
        t.unobserved_streak += 1
        if t.unobserved_streak >= config.temporal_unobserved_windows:
            t.state, t.closed_reason = CLOSED, TTL_EXPIRED_UNOBSERVED


def _link(
    t: Trajectory,
    index: int,
    node: Insight,
    scale: Mapping[str, float | None],
    config: MinerConfig,
    *,
    recovered: bool = False,
    reappeared: bool = False,
) -> None:
    """The EVOLVES_TO edge from the trajectory's last admitted node to ``node``, which becomes the last."""
    window, previous = t.last
    transition_type, shifts, drift_metric = transition(previous, node, scale, t.metrics, config, recovered=recovered, reappeared=reappeared)
    props = {
        "trajectory": t.key,
        "episode": t.episode,
        "from_window": window,
        "to_window": index,
        "gap": index - window - 1,
        "transition": transition_type,
        "shifts": shifts,
        "drift_metric": drift_metric,
    }
    t.edges.append((previous.id, node.id, props))
    t.last = (index, node)
