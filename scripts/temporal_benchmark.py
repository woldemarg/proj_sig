"""The temporal benchmark (docs/10_verification.md §10.4): planted cohorts followed through the weekly windows of one
upload, and pure-noise uploads for the rate of false trajectories.

    python scripts/temporal_benchmark.py [--seeds 1-20] [--null-seeds 1-100] [--workers 8] [--label current]
                                         [--engine-seeds 1-3] [--encoder hashing|model]
                                         [--override TEMPORAL_DRIFT_EXCESS=0.5 ...]

``generate(seed)`` writes 20,000 events over 70 days (``window=7D``: ten windows of about 2,000 rows) with five
categorical columns and five metrics drawn N(0, 1), and plants four cohorts, each on a metric of its own:

    S1 stable         segment=B ∧ region_type=urban   value +2.5 in every window
    S2 strengthening  segment=C ∧ channel=online      load +(1 + w) in window w: twice the drift threshold a window, so
                                                      the noise of the change (sd ≈ 0.15 at ~130 rows) never decides it
    S3 dormant        segment=D ∧ region_type=rural   rate +2.5, but 0 in W3 and W4
    S4 macro trend    segment=E ∧ channel=retail      score +2.5 over its window, while from W5 every row's score is +2

``generate(seed, null=True)`` is the same table without them (S5). The library mode runs, window by window, what a
batch runs of the miner — ``load_dataset(window=)``, ``run_discovery(known=)``, ``build_insights``, ``select_insights``,
the graph's admission — and folds the observations as the service does, without vectors, anchors or a workspace.
Scores: each planted trajectory's expected edges found (recall; a missing edge is a false break, with its cause) and
created edges expected (precision), each edge's transition, the excess shift δ (S2's sign, S4's size at the trend),
and over the null uploads the share with a trajectory active in at least three windows (a Wilson 95 % interval).

The engine mode (``--engine-seeds``) uploads the same planted files to the graph service on scratch workspaces (``time``,
``window=7D``; the hashing double unless ``--encoder model``): its snapshot's trajectories and EVOLVES_TO edges must
equal the library pipeline's (parity, by rule), its stage times per window are recorded, and the same file uploaded
without a time column measures what the windows add to the latent plane (anchors, members per anchor, the heaviest
anchor's mass, CO_OCCURS links and their shared members). Results: ``.scratch/results/temporal_benchmark_<label>.json``.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from attractor_benchmark import parse_seeds
from scipy.stats import binomtest
from scratch import build_engine, parse_overrides, save_result, scratch_dir, settings

from insight_contracts import Condition
from insight_graph_service.core.batch import admit_insights
from insight_graph_service.core.settings import Settings
from subgroup_miner.discovery import DiscoveryError, build_insights, run_discovery
from subgroup_miner.ingestion import IngestionError, forget_tables, load_dataset
from subgroup_miner.selection import select_insights
from subgroup_miner.temporal import (
    CONTINUATION,
    DRIFT,
    RECOVERY,
    Trajectory,
    WindowFacts,
    fold_trajectories,
    observations,
    open_rules,
    series_id,
    stamp_insights,
    upload_windows,
)

EPOCH = pd.Timestamp("2031-03-03", tz="UTC")  # any epoch: the time is synthetic (AGENTS.md, the data rule)
WINDOWS, ROWS, DAYS = 10, 20_000, 70
METRICS = ("value", "load", "rate", "score", "noise")
PLANTED = {
    "S1": ({"segment": "B", "region_type": "urban"}, "value"),
    "S2": ({"segment": "C", "channel": "online"}, "load"),
    "S3": ({"segment": "D", "region_type": "rural"}, "rate"),
    "S4": ({"segment": "E", "channel": "retail"}, "score"),
}
QUIET = (3, 4)  # S3's windows without its effect
TREND_FROM = 5  # the window from which every row's score rises


def generate(seed: int, *, null: bool = False, rows: int = ROWS, days: int = DAYS) -> pd.DataFrame:
    """One upload: ``rows`` events uniform over ``days`` days from ``EPOCH``, the planted cohorts unless ``null``."""
    rng = np.random.default_rng(seed)
    when = EPOCH + pd.to_timedelta(rng.uniform(0, days * 86_400, rows), unit="s")
    window = ((when - EPOCH) // pd.Timedelta("7D")).to_numpy(dtype=int)
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
    if null:
        return frame
    scope = {name: np.logical_and.reduce([frame[c] == v for c, v in conds.items()]) for name, (conds, _) in PLANTED.items()}
    frame.loc[scope["S1"], "value"] += 2.5
    frame.loc[scope["S2"], "load"] += 1.0 + window[scope["S2"]]
    frame.loc[scope["S3"] & ~np.isin(window, QUIET), "rate"] += 2.5
    frame.loc[window >= TREND_FROM, "score"] += 2.0
    frame.loc[scope["S4"], "score"] += 2.5
    return frame


def expected_edges() -> dict[str, list[tuple[int, int, str]]]:
    """Per planted cohort, the EVOLVES_TO edges (from window, to window, transition) its trajectory should have."""
    steady = [(w, w + 1, CONTINUATION) for w in range(WINDOWS - 1)]
    dormant = [(0, 1, CONTINUATION), (1, 2, CONTINUATION), (2, 5, RECOVERY)] + [(w, w + 1, CONTINUATION) for w in range(5, WINDOWS - 1)]
    return {"S1": steady, "S2": [(w, w + 1, DRIFT) for w in range(WINDOWS - 1)], "S3": dormant, "S4": steady}


def mine_upload(path: Path, upload_id: str, config: Settings) -> tuple[dict[str, Trajectory], list[float]]:
    """The library pipeline over one upload's windows: its trajectories and the seconds each window took."""
    cfg = config.miner
    windows, scale = upload_windows(path, cfg, time="when", window="7D", max_windows=WINDOWS)
    facts: dict[int, WindowFacts] = {}
    seconds = []
    for window in windows:
        start = time.perf_counter()
        known = open_rules(fold_trajectories(facts, window["index"], scale, cfg))
        try:
            loaded = load_dataset(path, cfg, window=window)
            result = run_discovery(loaded.frame, cfg, known=known)
        except (IngestionError, DiscoveryError):  # nothing to mine: the window observed nothing
            continue
        insights = build_insights(result, cfg, dataset_id=loaded.dataset_id, batch_id=f"W{window['index']}", filename=path.name)
        stamp_insights(insights, result, series=series_id(upload_id, window), window=window["index"])
        selection = select_insights(insights, cfg)
        kept, refused = admit_insights(selection.kept, config.min_insight_weight, config.max_insights_per_batch)
        facts[window["index"]] = WindowFacts(observations(result, insights, kept, [*result.rejections, *selection.rejections, *refused]), kept)
        seconds.append(time.perf_counter() - start)
    forget_tables()
    return fold_trajectories(facts, len(windows), scale, cfg), seconds


def score_planted(trajectories: dict[str, Trajectory]) -> dict[str, Any]:
    """Each planted cohort's trajectory against its expected edges, with the excess shift of its own metric per edge."""
    out = {}
    for name, (scope, metric) in PLANTED.items():
        want = tuple(sorted(Condition(c, v) for c, v in scope.items()))
        found = next((t for t in trajectories.values() if t.predicate == want), None)
        edges = {(p["from_window"], p["to_window"]): p for _, _, p in (found.edges if found else [])}
        expected = expected_edges()[name]
        rows = [
            {
                "from": a,
                "to": b,
                "want": kind,
                "got": edges[a, b]["transition"] if (a, b) in edges else None,
                "cause": None if (a, b) in edges else _cause(found, b),
            }
            for a, b, kind in expected
        ]
        out[name] = {
            "recall": sum(r["got"] is not None for r in rows) / len(rows),
            "precision": sum((a, b) in edges for a, b, _ in expected) / len(edges) if edges else None,
            "typed": sum(r["got"] == r["want"] for r in rows) / len(rows),
            "edges": rows,
            "excess": {f"{a}-{b}": p["shifts"].get(metric, {}).get("excess") for (a, b), p in sorted(edges.items())},
            "state": found.state if found else None,
            "trajectories": sum(t.predicate == want for t in trajectories.values()),
        }
    return out


def _cause(found: Trajectory | None, window: int) -> str:
    """Why an expected edge into ``window`` is missing: no trajectory for the scope, or what the window observed."""
    if found is None:
        return "no_trajectory"
    entry = next((w for w in found.windows if w["index"] == window), None)
    if entry is None:
        return "match"  # the trajectory began later, or was closed
    return entry["reason"] or entry["rejection"] or "inactive"


def parity(library: dict[str, Trajectory], snapshot: dict[str, Any]) -> dict[str, Any]:
    """The library pipeline's trajectories against the graph service's snapshot of the same upload, by rule (the keys
    differ: a series names its upload): state, each window's status and admission, each edge's windows and transition."""

    def shape(state: str, windows: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple:
        return (
            state,
            [(w["index"], w["status"], w["active"]) for w in windows],
            sorted((p["from_window"], p["to_window"], p["transition"]) for p in edges),
        )

    edges: dict[str, list[dict[str, Any]]] = {}
    for e in snapshot["edges"]:
        if e["type"] == "EVOLVES_TO":
            edges.setdefault(e["props"]["trajectory"], []).append(e["props"])
    lib = {tuple((c.attribute, c.value) for c in t.predicate): shape(t.state, t.windows, [p for _, _, p in t.edges]) for t in library.values()}
    svc = {
        tuple((c["attribute"], c["value"]) for c in t["predicate"]): shape(t["state"], t["windows"], edges.get(k, []))
        for k, t in snapshot["trajectories"].items()
    }
    differing = sorted(str(r) for r in lib.keys() | svc.keys() if lib.get(r) != svc.get(r))
    return {"same": not differing, "trajectories": len(svc), "differing": differing}


def latent(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The latent plane of a snapshot: anchors, their members, the heaviest anchor's mass, CO_OCCURS links and members."""
    anchors = [n["props"] for n in snapshot["nodes"] if n["kind"] == "Attractor"]
    shared = [e["props"]["shared"] for e in snapshot["edges"] if e["type"] == "CO_OCCURS"]
    return {
        "patterns": sum(n["kind"] == "Pattern" for n in snapshot["nodes"]),
        "anchors": len(anchors),
        "patterns_per_anchor": float(np.mean([a["n_patterns"] for a in anchors])) if anchors else 0.0,
        "max_mass": max((a["mass"] for a in anchors), default=0.0),
        "co_occurs": len(shared),
        "shared_mean": float(np.mean(shared)) if shared else 0.0,
    }


def run_engine(seed: int, overrides: dict[str, Any], embedder: Any) -> dict[str, Any]:
    """One planted upload through the graph service, windowed and plain: parity with the library run, stage seconds per
    window, and the latent plane of both."""
    folder = scratch_dir(f"temporal_benchmark/engine_{seed}", fresh=True)
    path = folder / "events.csv"
    generate(seed).to_csv(path, index=False, lineterminator="\n")
    library, _ = mine_upload(path, f"up-{seed}", settings(**overrides))
    engine = build_engine(folder / "windowed", embedder=embedder, **overrides)
    with path.open("rb") as f:
        records = [engine.process(r["batch_id"]) for r in engine.upload(path.name, f, time="when", window="7D")]
    timings = [r["metrics"].get("timings", {}) for r in records if r["status"] == "READY"]
    plain = build_engine(folder / "plain", embedder=engine.encoder.embedder, **overrides)
    plain.ingest_file(path)
    windowed = engine.ws.load_graph()
    return {
        "seed": seed,
        "parity": parity(library, windowed),
        "seconds": {
            stage: float(np.mean([t.get(stage, 0.0) for t in timings])) for stage in ("discover_s", "select_s", "embed_s", "ontology_s", "persist_s")
        },
        "latent": {"windowed": latent(windowed), "plain": latent(plain.ws.load_graph())},
    }


def run_seed(seed: int, null: bool, overrides: dict[str, Any]) -> dict[str, Any]:
    """One upload through the library pipeline: its planted trajectories' scores, or (null) its long-lived trajectories."""
    folder = scratch_dir(f"temporal_benchmark/{'null' if null else 'planted'}_{seed}", fresh=True)
    path = folder / "events.csv"
    generate(seed, null=null).to_csv(path, index=False, lineterminator="\n")
    trajectories, seconds = mine_upload(path, f"up-{seed}", settings(**overrides))
    lasting = sum(sum(w["active"] for w in t.windows) >= 3 for t in trajectories.values())
    row = {"seed": seed, "null": null, "trajectories": len(trajectories), "lasting": lasting, "seconds": seconds}
    return row | ({} if null else {"planted": score_planted(trajectories)})


def wilson(k: int, n: int) -> tuple[float, float]:
    """The Wilson score 95 % interval of a proportion k / n."""
    interval = binomtest(k, n).proportion_ci(method="wilson") if n else None
    return (float(interval.low), float(interval.high)) if interval else (0.0, 1.0)


def analyse(rows: list[dict[str, Any]], drift_excess: float) -> dict[str, Any]:
    """Recall, precision, transitions and false-break causes per planted cohort; S2's sign and S4's size of the excess
    shift; the null uploads' false-trajectory rate; seconds per window."""
    planted, null = [r for r in rows if not r["null"]], [r for r in rows if r["null"]]
    summary: dict[str, Any] = {"planted_uploads": len(planted)}
    for name in PLANTED:
        runs = [r["planted"][name] for r in planted]
        causes: dict[str, int] = {}
        for edge in (e for run in runs for e in run["edges"] if e["cause"]):
            causes[edge["cause"]] = causes.get(edge["cause"], 0) + 1
        summary[name] = {
            key: float(np.mean([r[key] for r in runs if r[key] is not None])) if runs else None for key in ("recall", "precision", "typed")
        } | {"single_trajectory": float(np.mean([r["trajectories"] == 1 for r in runs])) if runs else None, "false_breaks": causes}
    s2 = [x for r in planted for x in r["planted"]["S2"]["excess"].values() if x is not None]
    summary["S2"]["excess_positive"] = float(np.mean([x > 0 for x in s2])) if s2 else None
    trend = [x for r in planted if (x := r["planted"]["S4"]["excess"].get(f"{TREND_FROM - 1}-{TREND_FROM}")) is not None]
    summary["S4"]["trend_excess"] = (
        {"median": float(np.median(trend)), "within": float(np.mean([abs(x) < drift_excess for x in trend]))} if trend else None
    )
    hits = sum(r["lasting"] > 0 for r in null)
    summary["S5"] = {
        "uploads": len(null),
        "with_false_trajectory": hits,
        "rate": hits / len(null) if null else None,
        "wilson95": wilson(hits, len(null)),
    }
    summary["planted_trajectories"] = float(np.mean([r["trajectories"] for r in planted])) if planted else None
    summary["seconds_per_window"] = float(np.mean([s for r in rows for s in r["seconds"]])) if rows else None
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"temporal benchmark — {summary['planted_uploads']} planted uploads, {summary['S5']['uploads']} null uploads"]
    for name in PLANTED:
        s = summary[name]
        lines.append(
            f"  {name}: recall {s['recall']:.3f} precision {s['precision']:.3f} typed {s['typed']:.3f} one trajectory {s['single_trajectory']:.2f} breaks {s['false_breaks']}"
        )
    lines.append(f"  S2 excess > 0 on {summary['S2']['excess_positive']:.3f} of its edges; S4 trend excess {summary['S4']['trend_excess']}")
    low, high = summary["S5"]["wilson95"]
    lines.append(
        f"  S5 false trajectories (≥ 3 active windows): {summary['S5']['with_false_trajectory']}/{summary['S5']['uploads']} [{low:.3f}, {high:.3f}]"
    )
    lines.append(f"  {summary['planted_trajectories']:.1f} trajectories per planted upload; {summary['seconds_per_window']:.2f} s per window")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1-20", help="planted upload seeds, a range a-b or a list a,b,c")
    parser.add_argument("--null-seeds", default="1-100", help="pure-noise upload seeds (S5)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label", default="current")
    parser.add_argument("--engine-seeds", default="", help="planted seeds also run through the graph service (parity, timings, latent plane)")
    parser.add_argument("--encoder", choices=("hashing", "model"), default="hashing", help="the engine mode's embedder")
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE", help="a settings field by its upper-case name")
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    jobs = [(s, null) for spec, null in ((args.seeds, False), (args.null_seeds, True)) if spec for s in parse_seeds(spec)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_seed, [s for s, _ in jobs], [n for _, n in jobs], [overrides] * len(jobs)))
    summary = analyse(rows, settings(**overrides).miner.temporal_drift_excess)
    print(format_summary(summary))
    engine_rows = []
    if args.engine_seeds:
        from tests.doubles import HashingEmbedder

        embedder = HashingEmbedder() if args.encoder == "hashing" else None
        for seed in parse_seeds(args.engine_seeds):
            engine_rows.append(run_engine(seed, overrides, embedder))
            print(f"engine, seed {seed}: {engine_rows[-1]}", flush=True)
    result = {"summary": summary, "engine": engine_rows, "overrides": overrides, "per_upload": rows}
    print(save_result("temporal_benchmark", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
