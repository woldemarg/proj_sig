"""The temporal benchmark (docs/10_verification.md §10.4): planted cohorts in one upload mined once, their series over
the weekly grid, stationary uploads for the false-event rate, and the flash matrix of what one global pass misses.

    python scripts/temporal_benchmark.py [--seeds 1-20] [--stationary-seeds 1-100] [--flash-seeds 1-5] [--engine-seeds 1-3]
                                         [--workers 8] [--label current] [--override TEMPORAL_PROCESS_NOISE=0.1 ...]

``generate(seed)`` writes 20,000 events over 70 days from a Monday (ten ISO weeks) with five categorical columns and
five metrics drawn N(0, 1), and plants four cohorts, each on a metric of its own (the same bytes as the chain model's
benchmark, which ``temporal_ab.py`` compares against):

    S1 stable         segment=B ∧ region_type=urban   value +2.5 every week                  trend flat, no event
    S2 strengthening  segment=C ∧ channel=online      load +(1 + w) in week w                trend rising, no peak
    S3 dormant        segment=D ∧ region_type=rural   rate +2.5, but 0 in weeks 3 and 4      a dip in x̂, no event at its edges
    S4 macro trend    segment=E ∧ channel=retail      score +2.5; every row +2 from week 5   trend flat (the bin's population
                                                                                             carries the trend)

``spike=True`` adds the metric ``pulse`` (S6): segment=A ∧ region_type=suburban +2.5 every week and +3 more in week 6 — one
PEAKED_AT, in that ISO week. ``stationary=True`` (S5) plants the four cohorts as a constant +2.5 with no time in
them: every admitted insight's series is stationary, so any event is false — the share of uploads with one, with its
Wilson interval. ``flash=(k, m)`` adds the metric ``flash`` (S7): segment=A ∧ channel=partner +m in k consecutive weeks
only, over a table without the cohorts. Over durations k and magnitudes m the matrix of how often the one global pass admits that cohort (or a
specialisation of it), and how often a peak marks it, is the measured boundary of Flash Event Blindness — a research boundary, not an acceptance
threshold (docs/extra_funcs/temporal_analytics/temporal_event_graph_architecture.md §2.9).

Each upload runs what a batch runs of the miner — ``load_dataset(time=, grid="week")``, ``run_discovery``,
``build_insights``, ``select_insights``, the graph's admission — then ``annotate`` and ``dynamics``, without vectors,
anchors or a workspace. The engine mode (``--engine-seeds``) sends a planted upload through the graph service as well
(``time=when``, ``grid=week``, the hashing embedder): its snapshot's trajectories must equal the library's — trend,
current level, measured bins and events of every pattern —, and the batch's stage times are recorded. Results:
``.scratch/results/temporal_benchmark_<label>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from attractor_benchmark import parse_seeds
from scipy.stats import binomtest
from scratch import build_engine, parse_overrides, save_result, scratch_dir, settings

from insight_contracts import Condition, Insight
from insight_graph_service.core.batch import admit_insights
from insight_graph_service.core.settings import Settings
from subgroup_miner.discovery import DiscoveryError, build_insights, covers_of, run_discovery
from subgroup_miner.ingestion import load_dataset
from subgroup_miner.selection import select_insights
from subgroup_miner.temporal import FLAT, PEAKED_AT, RISING, annotate, calendar_bins, dynamics, reference_scale

EPOCH = pd.Timestamp("2031-03-03", tz="UTC")  # a Monday: the weeks are ISO weeks (the time is synthetic, AGENTS.md)
WEEKS, ROWS, DAYS = 10, 20_000, 70
METRICS = ("value", "load", "rate", "score", "noise")
PLANTED = {
    "S1": ({"segment": "B", "region_type": "urban"}, "value"),
    "S2": ({"segment": "C", "channel": "online"}, "load"),
    "S3": ({"segment": "D", "region_type": "rural"}, "rate"),
    "S4": ({"segment": "E", "channel": "retail"}, "score"),
    "S6": ({"segment": "A", "region_type": "suburban"}, "pulse"),
}
FLASH = ({"segment": "A", "channel": "partner"}, "flash")
QUIET = (3, 4)  # S3's weeks without its effect
TREND_FROM = 5  # the week from which every row's score rises (S4)
SPIKE_WEEK = 6  # S6's week
FLASH_WEEKS = (1, 2, 3, 5, 8, 10)  # S7's durations k
FLASH_SIZES = (1.0, 2.0, 3.0, 5.0)  # S7's magnitudes m, in sd


def generate(
    seed: int,
    *,
    rows: int = ROWS,
    days: int = DAYS,
    stationary: bool = False,
    spike: bool = False,
    flash: tuple[int, float] | None = None,
) -> pd.DataFrame:
    """One upload: ``rows`` events uniform over ``days`` days from ``EPOCH`` with the planted cohorts (``stationary``:
    each a constant +2.5; ``flash``: none); ``spike`` and ``flash`` each add a metric column of their own, so the default
    file stays the chain benchmark's."""
    rng = np.random.default_rng(seed)
    when = EPOCH + pd.to_timedelta(rng.uniform(0, days * 86_400, rows), unit="s")
    week = ((when - EPOCH) // pd.Timedelta("7D")).to_numpy(dtype=int)
    frame = pd.DataFrame(
        {
            "when": when.strftime("%Y-%m-%d %H:%M:%S"),
            "segment": rng.choice(list("ABCDE"), rows),
            "region_type": rng.choice(["urban", "suburban", "rural"], rows),
            "channel": rng.choice(["online", "retail", "partner"], rows),
            "tier": rng.choice(["t1", "t2", "t3", "t4"], rows),
            "source": rng.choice(["s1", "s2", "s3", "s4"], rows),
        }
    )
    for metric in METRICS:
        frame[metric] = rng.normal(0.0, 1.0, rows)
    scope = {name: np.logical_and.reduce([frame[c] == v for c, v in conds.items()]) for name, (conds, _) in PLANTED.items()}
    if stationary:
        for name, (_, metric) in list(PLANTED.items())[:4]:
            frame.loc[scope[name], metric] += 2.5
    elif not flash:
        frame.loc[scope["S1"], "value"] += 2.5
        frame.loc[scope["S2"], "load"] += 1.0 + week[scope["S2"]]
        frame.loc[scope["S3"] & ~np.isin(week, QUIET), "rate"] += 2.5
        frame.loc[week >= TREND_FROM, "score"] += 2.0
        frame.loc[scope["S4"], "score"] += 2.5
    if spike:
        frame["pulse"] = rng.normal(0.0, 1.0, rows) + 2.5 * scope["S6"] + 3.0 * (scope["S6"] & (week == SPIKE_WEEK))
    if flash:
        k, m = flash
        start = (WEEKS - k) // 2
        cohort = np.logical_and.reduce([frame[c] == v for c, v in FLASH[0].items()])
        frame["flash"] = rng.normal(0.0, 1.0, rows) + m * (cohort & (week >= start) & (week < start + k))
    return frame


def mine(path: Path, config: Settings) -> tuple[list[Insight], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The library pipeline over one upload: its admitted insights, each one's trajectory by pattern id, and the bins."""
    cfg = config.miner
    loaded = load_dataset(path, cfg, time="when", grid="week")
    try:
        result = run_discovery(loaded.frame, cfg)
    except DiscoveryError:
        return [], {}, []
    insights = build_insights(result, cfg, dataset_id=loaded.dataset_id, batch_id="bench", filename=path.name)
    kept, _ = admit_insights(select_insights(insights, cfg).kept, config.min_insight_weight, config.max_insights_per_batch)
    codes, bins = calendar_bins(loaded.time, loaded.grid)
    covers = covers_of(result)
    series = annotate(
        loaded.frame,
        codes,
        bins,
        {i.id: covers[i.expression] for i in kept},
        {i.id: [s.metric for s in i.shifts] for i in kept},
        reference_scale(loaded.frame),
        cfg.temporal_min_rows_bin,
    )
    return kept, {i.id: dynamics(series[i.id], bins, [i.target, *(s.metric for s in i.shifts)], cfg) for i in kept}, bins


def cohort(kept: list[Insight], scope: dict[str, str], metric: str) -> Insight | None:
    """The admitted insight of a planted scope that shifts ``metric``: the scope itself, else its narrowest specialisation."""
    want = {Condition(c, v) for c, v in scope.items()}
    found = [i for i in kept if want <= set(i.conditions) and any(s.metric == metric for s in i.shifts)]
    return min(found, key=lambda i: len(i.conditions), default=None)


def score_planted(kept: list[Insight], trajectories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each planted cohort: found, its trend, its events on its own metric and on the others, and S3's dip depth."""
    out = {}
    for name, (scope, metric) in PLANTED.items():
        ins = cohort(kept, scope, metric)
        if ins is None:
            out[name] = {"found": False}
            continue
        t = trajectories[ins.id]
        own = [e for e in t["events"] if e["metric"] == metric]
        x = [v for v in t["smoothed"].get(metric, []) if v is not None]
        row = {"found": True, "trend": t["trend"] if ins.target == metric else None, "events": [(e["kind"], e["bin"]) for e in own]}
        row["other_events"] = sum(e["metric"] != metric for e in t["events"])
        if name == "S3" and len(x) == WEEKS:
            row["dip"] = float(np.mean([v for w, v in enumerate(x) if w not in QUIET]) - np.mean([x[w] for w in QUIET]))
        out[name] = row
    return out


def run_seed(seed: int, kind: str, overrides: dict[str, Any], flash: tuple[int, float] | None = None) -> dict[str, Any]:
    """One upload through the library pipeline: planted (S1–S4, S6), stationary (S5) or one flash cell (S7)."""
    folder = scratch_dir(f"temporal_benchmark/{kind}_{seed}" + (f"_{flash[0]}x{flash[1]:g}" if flash else ""), fresh=True)
    path = folder / "events.csv"
    generate(seed, stationary=kind == "stationary", spike=kind == "planted", flash=flash).to_csv(path, index=False, lineterminator="\n")
    start = time.perf_counter()
    kept, trajectories, bins = mine(path, settings(**overrides))
    row = {"seed": seed, "kind": kind, "seconds": time.perf_counter() - start, "patterns": len(kept), "bins": len(bins)}
    row["events"] = sum(len(t["events"]) for t in trajectories.values())
    if kind == "planted":
        row["planted"] = score_planted(kept, trajectories)
    if flash:
        ins = cohort(kept, *FLASH)
        events = trajectories[ins.id]["events"] if ins else []
        row |= {"flash": list(flash), "found": ins is not None, "evented": any(e["metric"] == FLASH[1] and e["kind"] == PEAKED_AT for e in events)}
    return row


def parity(library: dict[str, dict[str, Any]], served: dict[str, dict[str, Any]]) -> list[str]:
    """The patterns whose trajectory the library pipeline and the graph service tell apart — any field ``dynamics``
    gives, as the snapshot stores it (series, smoothed levels, events, trend, current level, bins, gaps) — or that only
    one of them has."""

    def same(p: str) -> bool:
        return json.loads(json.dumps(library[p], default=float)) == {k: served[p].get(k) for k in library[p]}

    return sorted(p for p in library.keys() | served.keys() if p not in library or p not in served or not same(p))


def run_engine(seed: int, overrides: dict[str, Any], embedder: Any) -> dict[str, Any]:
    """One planted upload through the library pipeline and through the graph service: their parity, the batch's stages."""
    folder = scratch_dir(f"temporal_benchmark/engine_{seed}", fresh=True)
    path = folder / "events.csv"
    generate(seed, spike=True).to_csv(path, index=False, lineterminator="\n")
    _, library, _ = mine(path, settings(**overrides))
    engine = build_engine(folder / "ws", embedder=embedder, **overrides)
    with path.open("rb") as f:
        (record,) = [engine.process(r["batch_id"]) for r in engine.upload(path.name, f, time="when", grid="week")]
    served = (engine.ws.load_graph() or {}).get("trajectories", {})
    timings = record.get("metrics", {}).get("timings", {})
    return {
        "seed": seed,
        "kind": "engine",
        "status": record["status"],
        "patterns": len(served),
        "differing": parity(library, served),
        "stages": timings,
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    """The Wilson score 95 % interval of a proportion k / n."""
    interval = binomtest(k, n).proportion_ci(method="wilson") if n else None
    return (float(interval.low), float(interval.high)) if interval else (0.0, 1.0)


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per planted cohort the share of uploads meeting its expectation; the stationary false-event rate; the flash matrix."""
    planted = [r["planted"] for r in rows if r["kind"] == "planted"]
    stationary = [r for r in rows if r["kind"] == "stationary" and r["patterns"]]  # an upload that admits nothing tests no series
    expect = {
        "S1": lambda s: s["trend"] == FLAT and not s["events"],
        "S2": lambda s: s["trend"] == RISING and not any(k == PEAKED_AT for k, _ in s["events"]),
        "S3": lambda s: s.get("dip", 0.0) > 1.0 and not any(b in (QUIET[0] - 1, QUIET[-1] + 1) for _, b in s["events"]),
        "S4": lambda s: s["trend"] == FLAT and not s["events"],
        "S6": lambda s: s["events"] == [(PEAKED_AT, SPIKE_WEEK)],
    }
    summary: dict[str, Any] = {"planted_uploads": len(planted)}
    for name, ok in expect.items():
        runs = [p[name] for p in planted]
        summary[name] = {
            "found": float(np.mean([r["found"] for r in runs])) if runs else None,
            "as_expected": float(np.mean([r["found"] and ok(r) for r in runs])) if runs else None,
            "other_events": int(sum(r.get("other_events", 0) for r in runs)),
        }
    dips = [p["S3"]["dip"] for p in planted if "dip" in p["S3"]]
    summary["S3"]["dip_median"] = float(np.median(dips)) if dips else None
    hits = sum(r.get("events", 0) > 0 for r in stationary)
    summary["S5"] = {
        "uploads": len(stationary),
        "patterns": int(sum(r.get("patterns", 0) for r in stationary)),
        "with_false_event": hits,
        "rate": hits / len(stationary) if stationary else None,
        "wilson95": wilson(hits, len(stationary)),
    }
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["kind"] == "flash" and "flash" in r:
            cells.setdefault(f"{r['flash'][0]}x{r['flash'][1]:g}", []).append(r)
    summary["S7"] = {
        cell: {"found": float(np.mean([r["found"] for r in rs])), "evented": float(np.mean([r["evented"] for r in rs]))} for cell, rs in cells.items()
    }
    seconds = [r["seconds"] for r in rows if "seconds" in r]
    summary["seconds_per_upload"] = float(np.mean(seconds)) if seconds else None
    served = [r for r in rows if r["kind"] == "engine"]
    summary["engine"] = {
        "uploads": len(served),
        "identical": sum(r["status"] == "READY" and r["patterns"] > 0 and not r["differing"] for r in served),  # never on nothing
        "patterns": int(sum(r["patterns"] for r in served)),
        "stages": {k: float(np.mean([r["stages"].get(k, 0.0) for r in served])) for k in sorted({k for r in served for k in r["stages"]})},
    }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"temporal benchmark — {summary['planted_uploads']} planted uploads, {summary['S5']['uploads']} stationary uploads"]
    for name in ("S1", "S2", "S3", "S4", "S6"):
        s = summary[name]
        if s["found"] is not None:
            lines.append(f"  {name}: found {s['found']:.3f}, as expected {s['as_expected']:.3f}, events on other metrics {s['other_events']}")
    if summary["S3"].get("dip_median") is not None:
        lines.append(f"  S3 dip of the smoothed level (median): {summary['S3']['dip_median']:.2f} sd")
    low, high = summary["S5"]["wilson95"]
    lines.append(
        f"  S5 uploads with a false event: {summary['S5']['with_false_event']}/{summary['S5']['uploads']} [{low:.3f}, {high:.3f}]"
        f" over {summary['S5']['patterns']} stationary insights"
    )
    if summary["S7"]:
        lines.append("  S7 found / evented by (weeks k × magnitude m):")
        for k in FLASH_WEEKS:
            cells = [summary["S7"].get(f"{k}x{m:g}") for m in FLASH_SIZES]
            lines.append(f"    k={k:>2}: " + "  ".join(f"m={m:g} {c['found']:.2f}/{c['evented']:.2f}" for m, c in zip(FLASH_SIZES, cells) if c))
    if summary["seconds_per_upload"] is not None:
        lines.append(f"  {summary['seconds_per_upload']:.2f} s per upload")
    if summary["engine"]["uploads"]:
        e = summary["engine"]
        lines.append(f"  engine: {e['identical']}/{e['uploads']} uploads READY with the library's trajectories ({e['patterns']} patterns)")
        lines.append("  stage seconds (mean): " + ", ".join(f"{k} {v:.2f}" for k, v in e["stages"].items()))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1-20", help="planted upload seeds, a range a-b or a list a,b,c")
    parser.add_argument("--stationary-seeds", default="1-100", help="stationary upload seeds (S5)")
    parser.add_argument("--flash-seeds", default="1-5", help="seeds of each flash cell (S7); empty to skip the matrix")
    parser.add_argument("--engine-seeds", default="1-3", help="planted seeds also sent through the graph service; empty to skip")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label", default="current")
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE", help="a settings field by its upper-case name")
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    jobs = [(s, "planted", None) for s in parse_seeds(args.seeds)] if args.seeds else []
    jobs += [(s, "stationary", None) for s in parse_seeds(args.stationary_seeds)] if args.stationary_seeds else []
    if args.flash_seeds:
        jobs += [(s, "flash", (k, m)) for k in FLASH_WEEKS for m in FLASH_SIZES for s in parse_seeds(args.flash_seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_seed, *zip(*[(s, kind, overrides, flash) for s, kind, flash in jobs])))
    if args.engine_seeds:
        from tests.doubles import HashingEmbedder

        rows += [run_engine(s, overrides, HashingEmbedder()) for s in parse_seeds(args.engine_seeds)]
    summary = analyse(rows)
    print(format_summary(summary))
    print(save_result("temporal_benchmark", args.label, {"summary": summary, "overrides": overrides, "per_upload": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
