"""How far an upload's base insights find their place through their events (docs/10_verification.md §10.3), on a scratch
workspace: one upload with ``--geo <locator>@auto`` and its table level, then every insight's events projected on the
regions of ``GEO_REGION_LEVELS`` as docs/06_graph_and_storage.md §6.1 places them.

    python scripts/geo_coverage.py <file | synthetic | california> --geo <locator>@auto [--bins SPEC] [--encoder hashing|model]
        [--label NAME]

``--bins`` goes to both levels, each applying the columns it has (``california``: its median income in quartiles).

An event is located when the finest geo level holds its point (``point_cell >= 0``); its region at a level is the cell
holding the centre of its finest cell. A base insight (the table level's) has its covered rows as events, a geo
insight the points of its cells. Three coverages, each over its own denominator:
    C_members   per base insight, |P_loc| / |P|: how complete its events' geography is (median and minimum)
    C_insights  per rule and level, the share of base insights with C_members >= GEO_LOCATED_MIN and at least one link
    C_map       the share of the finest level's located points (and cells) inside the union of the base footprints
Per level, over each group (base, and each geo level's own insights): the regions an insight's events touch, k50 / k80
(the fewest regions holding 50 / 80 % of |P_loc|), the largest share in one region, and the regions with
lift = (x / |P_loc|) / (n_R / N) >= 1.5 and >= 2 at x >= EVENTS_MIN, where x are P's events in region R, n_R the map's
and N all located events. The candidate rules for #119, every share over |P_loc|:
    S   x / |P_loc| >= SHARE_MIN                                  R holds a fifth of P
    E   x / min(|P_loc|, n_R) >= SHARE_MIN, x >= EVENTS_MIN       IN_REGION's rule, counted in events
    F   |fp ∩ R| / min(|fp|, occupied cells of R) >= SHARE_MIN     IN_REGION's rule on the GEO_FOOTPRINT_SHARE footprint
    L   the binomial tail P(X >= x | n_R, |P_loc| / N) (where small and x >= 5, never below the exact hypergeometric one), under BH at FDR_Q over the level's
        regions holding a located event, lift >= LIFT_MIN, x >= EVENTS_MIN
    P   the placement of #119 (place_table): L on a level of at least LIFT_MIN_REGIONS regions holding a located event, S with
        lift >= 1 below, at x >= link_min_events(|P_loc|)
A real export prints aggregates only; its result stays in ``.scratch/results/geo_coverage_<label>.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h3
import numpy as np
from scipy.stats import binom, false_discovery_control
from scratch import build_engine, save_result, scratch_dir

from insight_graph_service.core.engine import Engine
from insight_graph_service.core.geo_views import member_rows
from insight_graph_service.core.settings import PROJECT_ROOT
from subgroup_miner.spatial import GEO_LIFT_MIN as LIFT_MIN  # the rules' bounds are the graph's
from subgroup_miner.spatial import GEO_LIFT_MIN_REGIONS as LIFT_MIN_REGIONS
from subgroup_miner.spatial import GEO_LINK_FDR as FDR_Q
from subgroup_miner.spatial import GEO_LINK_MIN_EVENTS as EVENTS_MIN
from subgroup_miner.spatial import GEO_LOCATED_MIN, event_regions, footprint, link_min_events
from subgroup_miner.spatial import GEO_REGION_MIN_SHARE as SHARE_MIN

RULES = ("S", "E", "F", "L", "P")


CALIFORNIA_BINS = "median_income:4,median_income_median:4"  # the table's and the cells' median income, in quartiles


def ingest(engine: Engine, path: Path, geo: str, bins: str | None = None) -> list[dict[str, Any]]:
    """The upload with its table level, every level processed; the levels' records."""
    with path.open("rb") as stream:
        levels = engine.upload(path.name, stream, geo=geo, table=True, bins=bins)
    return [engine.process(level["batch_id"]) for level in levels]


def upload_insights(engine: Engine, records: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """The finest geo level's ``cells`` and ``point_cell``, and each committed insight's events by group: ``base`` (the
    table level) and ``r<res>`` per geo level, each linked to the finest level's points by ``member_rows`` (one
    ``upload_id`` and one row per point, contract 6)."""
    ready = [r for r in records if r["status"] == "READY"]
    geo = [r for r in ready if r["level"] != "table"]
    if not geo:
        raise SystemExit("no geo level of the upload is READY")
    top = max(geo, key=lambda r: int(r["level"]))
    finest = engine.ws.load_geo(top["dataset_id"], ("cells", "point_cell"))
    graph = engine.committed().graph
    groups: dict[str, dict[str, np.ndarray]] = {}
    for r in sorted(ready, key=lambda r: r["level"] != "table"):  # the base first
        members = member_rows(engine.ws, r, top, len(finest["point_cell"]))
        if members is None:
            continue
        groups["base" if r["level"] == "table" else f"r{r['level']}"] = {
            pid: members(rows) for pid, rows in engine.ws.load_covers(r["dataset_id"]).items() if pid in graph.nodes and not graph.tampered(pid)
        }
    if "base" not in groups:
        raise SystemExit("no READY table level of the upload with one row per point")
    return finest, groups


def measure(point_cell: np.ndarray, cells: np.ndarray, insights: dict[str, np.ndarray], levels: list[int]) -> dict[str, Any]:
    """Every insight's coverage, distribution and rule links per level; ``map``: the located events and cells."""
    regions = {lv: np.unique(names, return_inverse=True)[1] for lv, names in event_regions(cells, tuple(levels)).items()}  # by centre
    located = point_cell[point_cell >= 0]
    n_map = len(located)
    per_level = {lv: (np.bincount(idx[located]), np.bincount(idx)) for lv, idx in regions.items()}  # n_R, occupied cells
    out = {}
    for pid, rows in insights.items():
        loc = rows[point_cell[rows] >= 0]
        entry: dict[str, Any] = {"events": len(rows), "located": len(loc), "c_members": len(loc) / len(rows) if len(rows) else 0.0}
        if len(loc):
            fp = footprint(point_cell, loc)[0]
            entry["footprint"] = fp
            entry["levels"] = {lv: level_stats(point_cell[loc], fp, regions[lv], *per_level[lv], n_map) for lv in levels}
        out[pid] = entry
    return {"insights": out, "events": n_map, "n_R": {lv: v[0] for lv, v in per_level.items()}}


def level_stats(cell_rows: np.ndarray, fp: np.ndarray, idx: np.ndarray, n_r: np.ndarray, occupied: np.ndarray, n_map: int) -> dict[str, Any]:
    x = np.bincount(idx[cell_rows], minlength=len(n_r))
    n_loc = len(cell_rows)
    share = x / n_loc
    with_events = n_r > 0  # the regions holding a located event: the graph's placement switches and adjusts over these
    lift = np.divide(share, n_r / n_map, out=np.zeros(len(n_r)), where=with_events)
    held = np.cumsum(np.sort(x)[::-1])
    pv = np.where(x > 0, binom.sf(x - 1, n_r, n_loc / n_map), 1.0)  # as the graph's placement (place_table)
    fp_in = np.bincount(idx[fp], minlength=len(n_r))
    q = np.ones(len(n_r))
    q[with_events] = false_discovery_control(pv[with_events])
    x_min = link_min_events(n_loc)
    links = {
        "S": share >= SHARE_MIN,
        "E": (x / np.minimum(n_loc, n_r) >= SHARE_MIN) & (x >= EVENTS_MIN),
        "F": fp_in / np.minimum(len(fp), occupied) >= SHARE_MIN,
        "L": (q <= FDR_Q) & (lift >= LIFT_MIN) & (x >= EVENTS_MIN),
        "P": ((q <= FDR_Q) & (lift >= LIFT_MIN) if with_events.sum() >= LIFT_MIN_REGIONS else (share >= SHARE_MIN) & (lift >= 1)) & (x >= x_min),
    }
    strong = (lift >= 2) & (x >= EVENTS_MIN)
    return {
        "touched": int((x > 0).sum()),
        "k50": int(np.searchsorted(held, 0.5 * n_loc) + 1),
        "k80": int(np.searchsorted(held, 0.8 * n_loc) + 1),
        "top_share": float(share.max()),
        "lift_15": int(((lift >= 1.5) & (x >= EVENTS_MIN)).sum()),
        "lift_2": int(strong.sum()),
        "lift_2_share": float(x[strong].sum() / n_loc),
        "lift_max": float(lift[x > 0].max()),
        "lift_min": float(lift[x > 0].min()),
        "links": {rule: np.flatnonzero(mask) for rule, mask in links.items()},
    }


def summarise(measured: dict[str, Any], levels: list[int], in_region: dict[tuple[str, int], int] | None, per_cell: np.ndarray) -> dict[str, Any]:
    """The aggregates of one group: C_members, the distributions' medians, each rule's links, C_insights and C_map."""
    insights = measured["insights"]
    placed = {pid: e for pid, e in insights.items() if "levels" in e}
    c_members = [e["c_members"] for e in insights.values()]
    out: dict[str, Any] = {
        "insights": len(insights),
        "c_members": {"median": _median(c_members), "min": min(c_members, default=None)},
    }
    for lv in levels:
        stats = [e["levels"][lv] for e in placed.values()]
        row: dict[str, Any] = {k: _median([s[k] for s in stats]) for k in ("touched", "k50", "k80", "top_share", "lift_2", "lift_2_share")}
        row["in_region"] = None if in_region is None else _median([in_region.get((pid, lv), 0) for pid in insights])  # its own edges at lv
        for rule in RULES:
            counts = [len(s["links"][rule]) for s in stats]
            union = np.unique(np.concatenate([s["links"][rule] for s in stats])) if stats else np.array([], int)
            linked = sum(1 for pid, e in placed.items() if e["c_members"] >= GEO_LOCATED_MIN and len(e["levels"][lv]["links"][rule]))
            row[rule] = {
                "with_link": sum(c > 0 for c in counts),
                "links_median": _median(counts),
                "links_max": max(counts, default=0),
                "regions": len(union),
                "map_share": float(measured["n_R"][lv][union].sum() / measured["events"]),
                "c_insights": linked / len(insights) if insights else None,
            }
        out[lv] = row
    fp = np.unique(np.concatenate([e["footprint"] for e in placed.values()])) if placed else np.array([], int)
    out["c_map"] = {"cells": len(fp) / len(per_cell), "events": float(per_cell[fp].sum() / per_cell.sum())}
    return out


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def run(engine: Engine, path: Path, geo: str, bins: str | None = None) -> dict[str, Any]:
    """The upload, then every group's measures and aggregates."""
    records = ingest(engine, path, geo, bins)
    finest, groups = upload_insights(engine, records)
    point_cell, cells = finest["point_cell"], finest["cells"]
    res = h3.get_resolution(str(cells[0]))
    levels = [lv for lv in engine.settings.region_levels if lv <= res]
    in_region: dict[tuple[str, int], int] = {}
    for e in engine.committed().graph.edges:
        if e["type"] == "IN_REGION":  # R:<level>:<h3>
            key = (e["source"], int(e["target"].split(":")[1]))
            in_region[key] = in_region.get(key, 0) + 1
    per_cell = np.bincount(point_cell[point_cell >= 0], minlength=len(cells))
    out: dict[str, Any] = {
        "levels": [{"level": r["level"], "status": r["status"], "insights": r["metrics"].get("validated_insights", 0)} for r in records]
    }
    out["map"] = {"resolution": res, "cells": len(cells), "events": int(per_cell.sum()), "rows": len(point_cell)}
    out["region_levels"], out["groups"] = levels, {}
    for name, insights in groups.items():
        measured = measure(point_cell, cells, insights, levels)
        out["map"]["regions"] = {lv: len(n_r) for lv, n_r in measured["n_R"].items()}
        out["groups"][name] = summarise(measured, levels, in_region, per_cell)
    return out


def format_tables(result: dict[str, Any]) -> str:
    """The distributions per group and level, then the rules per level, both from ``run``."""
    levels, groups = result["region_levels"], result["groups"]
    m = result["map"]
    lines = [
        f"map: resolution {m['resolution']}, {m['cells']} cells, {m['events']} located of {m['rows']} rows; regions "
        + ", ".join(f"r{lv} {n}" for lv, n in m["regions"].items())
        + "; levels "
        + ", ".join(f"{lv['level']} {lv['status']} {lv['insights']}" for lv in result["levels"])
    ]
    lines.append(f"{'insights':<10}{'n':>4}  {'C_members':<12}{'C_map':<14} level  touched    k50    k80      top  lift≥2 (share)  IN_REGION")
    lines.append(f"{'':<10}{'':>4}  {'med / min':<12}{'cells / ev':<14}{'':<55}  median")
    for name, g in groups.items():
        cm, cmap = g["c_members"], g["c_map"]
        head = f"{_group(name):<10}{g['insights']:>4}  {_fmt(cm['median']) + ' / ' + _fmt(cm['min']):<12}"
        head += f"{_fmt(cmap['cells']) + ' / ' + _fmt(cmap['events']):<14}"
        for i, lv in enumerate(levels):
            r = g[lv]
            stats = f" r{lv:<4}{_fmt(r['touched']):>8}{_fmt(r['k50']):>7}{_fmt(r['k80']):>7}{_fmt(r['top_share']):>9}"
            lift = f"{_fmt(r['lift_2']):>8} ({_fmt(r['lift_2_share'])})"
            lines.append((head if i == 0 else " " * len(head)) + stats + f"{lift:<17}{_fmt(r['in_region']):>9}")
    lines.append("")
    lines.append("rule  level  insights: with a link / links median (max) / regions in the union (their share of the map's events) / C_insights")
    for rule in RULES:
        for lv in levels:
            cells = [
                f"{_group(name)} {g[lv][rule]['with_link']}/{g['insights']} {_fmt(g[lv][rule]['links_median'])} ({g[lv][rule]['links_max']}) "
                f"{g[lv][rule]['regions']} ({_fmt(g[lv][rule]['map_share'])}) {_fmt(g[lv][rule]['c_insights'])}"
                for name, g in groups.items()
            ]
            lines.append(f"{rule:<6}r{lv:<5}" + " | ".join(cells))
    return "\n".join(lines)


def _group(name: str) -> str:
    return name if name == "base" else f"own {name}"


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:.3g}"


def main() -> int:
    from perf_benchmark import write_synthetic_points

    from tests.doubles import HashingEmbedder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a file, 'synthetic' or 'california'")
    parser.add_argument("--geo", required=True, help="the geo option, e.g. lat,lon@auto")
    parser.add_argument("--bins", default=None, help="BIN_COLUMNS for both levels (california: its median income)")
    parser.add_argument("--encoder", choices=("hashing", "model"), default="model")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    root = scratch_dir(f"geo_coverage_{args.label}", fresh=True)
    path = {
        "synthetic": lambda: write_synthetic_points(scratch_dir("perf_benchmark_data")),
        "california": lambda: PROJECT_ROOT / "data" / "demo" / "california_housing.csv",
    }.get(args.source, lambda: Path(args.source))()
    engine = build_engine(root / "ws", embedder=HashingEmbedder() if args.encoder == "hashing" else None)
    result = {
        "source": args.source if args.source in ("synthetic", "california") else "file",
        "geo": args.geo,
        **run(engine, path, args.geo, CALIFORNIA_BINS if args.source == "california" and args.bins is None else args.bins),
    }
    print(format_tables(result))
    print(save_result("geo_coverage", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
