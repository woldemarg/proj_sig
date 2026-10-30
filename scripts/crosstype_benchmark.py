"""Cross-type context benchmark (docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md §8):
target points and context layers drawn on one smooth density over a box of California, a dependence between them planted
or not, mined through the library with the layers (``load_dataset(context=)``), and each upload's admitted cross-type
insights read under three gates over replicate seeds — **A** the median test alone (the concept's pipeline: the same
insights selected with ``p_context`` left out, which is what ``CONTEXT_PERMUTATIONS=0`` admits), **B0** the context
null per cohort (its ``context_family`` left out: the gate before the family adjustment) and **B** the gate as built —
the false rates and the found rates with Wilson 95 % intervals.

    python scripts/crosstype_benchmark.py                                   # X1–X6b: planted 1-20, null 1-100
    python scripts/crosstype_benchmark.py --scenarios X2 --override CONTEXT_RING=0 --label ring0
    python scripts/crosstype_benchmark.py --scenarios X3 --override CONTEXT_RING=2 --label ring2

Scenarios (X, as ``temporal_benchmark`` names its own S1–S7): X1 the layer uniform, the target on the density, nothing
planted; X2 the target denser (× 1 + c) and ``load`` higher in the cells holding the layer; X3 the same around them (their
ring, the cell itself excluded); X4 two layers, planted only where both are in the cell's disk; X6 both types on the
density, independent given it; X6b the same with ``load`` rising with the density (as injuries downtown).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import h3
import numpy as np
import pandas as pd
from attractor_benchmark import parse_seeds
from geo_benchmark import CENTRE
from perf_benchmark import EPOCH
from scratch import build_engine, parse_overrides, save_result, scratch_dir, settings
from temporal_benchmark import wilson

from insight_contracts import Insight
from insight_graph_service.core.batch import admit_insights
from insight_graph_service.core.settings import Settings
from subgroup_miner.discovery import DiscoveryError, build_insights, covers_of, run_discovery
from subgroup_miner.ingestion import load_dataset
from subgroup_miner.selection import select_insights

RES = 8
BOX = (CENTRE[0] - 0.15, CENTRE[1] - 0.2, 0.3, 0.4)  # lat0, lon0, dlat, dlon: about 1,600 cells at RES
SIGMA_KM = 8.0  # the density's bump: λ = 0.2 + exp(−d² / 2σ²) around the box's centre
N_TARGET, N_LAYER = 20_000, 600
LIFT, BETA = 4.0, 1.0  # load +LIFT where planted; the target's intensity × (1 + BETA · the planted count)
FOUND = 0.5  # share of the planted cells the plant's finders must cover together (geo_benchmark's rule): a finder raises a
# planted metric and draws at least FOUND of its own cells from the planted ones
NULL, PLANTED = ("X1", "X6", "X6b"), ("X2", "X3", "X4")
GATES = ("A", "B0", "B")
SCENARIOS = ("X1", "X2", "X3", "X4", "X6", "X6b")


def density(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0, lon0, dlat, dlon = BOX
    dy = (lat - (lat0 + dlat / 2)) * 111.0
    dx = (lon - (lon0 + dlon / 2)) * 111.0 * np.cos(np.radians(lat0 + dlat / 2))
    return 0.2 + np.exp(-(dx**2 + dy**2) / (2 * SIGMA_KM**2))


def sample(rng: np.random.Generator, n: int, weight) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points in the box with intensity ∝ ``weight(lat, lon)`` (rejection against its maximum on a grid)."""
    lat0, lon0, dlat, dlon = BOX
    grid = np.meshgrid(np.linspace(lat0, lat0 + dlat, 60), np.linspace(lon0, lon0 + dlon, 60))
    top = float(np.max(weight(grid[0].ravel(), grid[1].ravel()))) * 1.05
    lats, lons = [], []
    while sum(map(len, lats)) < n:
        lat, lon = rng.uniform(lat0, lat0 + dlat, 4 * n), rng.uniform(lon0, lon0 + dlon, 4 * n)
        keep = rng.random(4 * n) * top < weight(lat, lon)
        lats.append(lat[keep]), lons.append(lon[keep])
    return np.concatenate(lats)[:n], np.concatenate(lons)[:n]


def cells_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.array([h3.latlng_to_cell(a, b, RES) for a, b in zip(lat, lon)], dtype=object)


def counted(cells: np.ndarray) -> dict[str, int]:
    keys, n = np.unique(cells, return_counts=True)
    return dict(zip(keys, n.tolist()))


def layer_frame(lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"lat": lat.round(6), "lon": lon.round(6)})


def generate(name: str, seed: int) -> dict[str, Any]:
    """One upload of scenario ``name``: the target frame, its layers by name and the planted cells (empty for a null)."""
    rng = np.random.default_rng(seed)
    uniform = lambda lat, lon: np.ones_like(lat)  # noqa: E731
    layers = {"schools": layer_frame(*sample(rng, N_LAYER, density if name in ("X6", "X6b") else uniform))}
    if name == "X4":
        layers = {"schools": layer_frame(*sample(rng, N_LAYER // 2, uniform)), "shops": layer_frame(*sample(rng, N_LAYER // 2, uniform))}
    counts = {k: counted(cells_of(f["lat"].to_numpy(), f["lon"].to_numpy())) for k, f in layers.items()}
    school = counts["schools"]

    def plant(cell: str) -> float:  # the planted count of a target point's cell (0: none)
        if name == "X2":
            return school.get(cell, 0)
        if name == "X3":
            return float(np.mean([school.get(c, 0) for c in h3.grid_ring(cell, 1)]))
        if name == "X4":
            disk = h3.grid_disk(cell, 1)
            return 2.0 * all(any(counts[k].get(c, 0) for c in disk) for k in counts)
        return 0.0

    memo: dict[str, float] = {}
    planted = lambda cells: np.array([memo.setdefault(c, plant(c)) for c in cells])  # noqa: E731
    lat, lon = sample(rng, N_TARGET, lambda a, b: density(a, b) * (1 + BETA * planted(cells_of(a, b))))
    lift = planted(cells_of(lat, lon)) > 0
    base = 8 + 4 * density(lat, lon) / 1.2 if name == "X6b" else 10.0  # 1.2: the density at the centre
    frame = pd.DataFrame(
        {
            "lat": lat.round(6),
            "lon": lon.round(6),
            "road_type": rng.choice(["local", "collector", "arterial"], size=len(lat), p=[0.5, 0.3, 0.2]),
            "when": np.datetime64(EPOCH) + rng.integers(0, 31, len(lat)).astype("timedelta64[D]"),
            "load": (rng.normal(base, 2.0, len(lat)) + LIFT * lift).round(2),
            "severity": rng.normal(10.0, 2.0, len(lat)).round(2),
        }
    )
    return {"frame": frame, "layers": layers, "planted": sorted({c for c, p in memo.items() if p > 0})}


def without(insight: Insight, *keys: str) -> Insight:
    """The insight with these ``provenance.spatial`` keys left out (gate A: ``p_context``; B0: ``context_family``)."""
    spatial = {k: v for k, v in insight.provenance.get("spatial", {}).items() if k not in keys}
    return replace(insight, provenance={**insight.provenance, "spatial": spatial})


def cross_rows(insights: list[Insight], covers: dict[str, np.ndarray], cells: np.ndarray, planted: set[str]) -> dict[str, Any]:
    """The admitted insights with a context condition (target, sign, ``lisa_points`` in the scope or not) and whether
    they find the plant: the finders (a planted metric raised, at least ``FOUND`` of their cells planted) together
    cover ``FOUND`` of the planted cells."""
    rows, reached = [], set()
    for i in insights:
        if not any(c.attribute.startswith("context_") and c.value != "none nearby" for c in i.conditions):
            continue
        own = set(cells[covers[i.expression]])
        if i.target in ("points", "load_median") and i.effect_size > 0 and len(own & planted) >= FOUND * len(own):
            reached |= own & planted
        rows.append(
            {
                "target": i.target,
                "z": round(i.effect_size, 2),
                "lisa": any(c.attribute == "lisa_points" for c in i.conditions),
                "conditions": len(i.conditions),
                "p_context": i.provenance["spatial"].get("p_context"),
            }
        )
    return {"insights": rows, "found": bool(planted) and len(reached) >= FOUND * len(planted)}


def run_seed(name: str, seed: int, overrides: dict[str, Any], family: bool) -> dict[str, Any]:
    """One upload through the library: both gates' cross-type insights, the search's size, with the layer and (``family``)
    without it."""
    config: Settings = settings(**overrides)
    cfg = config.miner
    folder = scratch_dir(f"crosstype/{name}/{seed}", fresh=True)
    upload = generate(name, seed)
    target = folder / "target.csv"
    upload["frame"].to_csv(target, index=False)
    paths = []
    for layer, frame in upload["layers"].items():
        frame.to_csv(path := folder / f"{layer}.csv", index=False)
        paths.append((layer, path, "lat,lon"))
    t0 = time.perf_counter()
    loaded = load_dataset(target, cfg, geo=f"lat,lon@{RES}", context=paths)
    row = {"scenario": name, "seed": seed, "cells": len(loaded.frame), "bands": [c for c in loaded.frame if c.startswith("context_")]}
    try:
        result = run_discovery(loaded.frame, cfg, spatial=loaded.spatial)
    except DiscoveryError as exc:
        return row | {"error": exc.code, **dict.fromkeys(GATES, {"insights": [], "found": False}), "seconds": time.perf_counter() - t0}
    insights = build_insights(result, cfg, dataset_id=loaded.dataset_id, batch_id="bench", filename=target.name)
    admit = lambda found: admit_insights(select_insights(found, cfg).kept, config.min_insight_weight, config.max_insights_per_batch)[0]  # noqa: E731
    covers, cells, planted = covers_of(result), loaded.spatial.cells, set(upload["planted"])
    marginal = [i for i in insights if i.p_adjusted is not None and i.p_adjusted <= cfg.max_p_adjusted and abs(i.effect_size) >= cfg.min_effect_z]
    row |= {
        "A": cross_rows(admit([without(i, "p_context") for i in insights]), covers, cells, planted),
        "B0": cross_rows(admit([without(i, "context_family") for i in insights]), covers, cells, planted),
        "B": cross_rows(admit(insights), covers, cells, planted),
        "p_context": [i.provenance["spatial"]["p_context"] for i in marginal if "p_context" in i.provenance["spatial"]],
        "selected": [d for d in result.profile.selected_dimensions if d.startswith("context_")],
        "space": result.profile.search_space_size,
        "longest": max((len(c.conditions) for c in result.candidates), default=0),
        "n_tests": result.n_tests,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if family:  # the same file without its layers: what the forced column costs the Bonferroni family and the search
        bare = load_dataset(target, cfg, geo=f"lat,lon@{RES}")
        plain = run_discovery(bare.frame, cfg, spatial=bare.spatial)
        row |= {"n_tests_plain": plain.n_tests, "space_plain": plain.profile.search_space_size}
    return row


def rate(hits: int, n: int) -> dict[str, Any]:
    return {"hits": hits, "n": n, "rate": round(hits / n, 3) if n else None, "wilson95": [round(x, 3) for x in wilson(hits, n)]}


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per scenario and gate: a null's false rate (any cross-type insight; X6 by target), a plant's found rate; the null's
    p_context calibration; the composition with ``lisa_points``; the search's size and the family with and without the layer."""
    out: dict[str, Any] = {}
    for name in sorted({r["scenario"] for r in rows}):
        runs = [r for r in rows if r["scenario"] == name]
        s: dict[str, Any] = {"uploads": len(runs), "errors": sum("error" in r for r in runs)}
        for gate in GATES:
            if name in NULL:
                s[gate] = {
                    "any": rate(sum(bool(r[gate]["insights"]) for r in runs), len(runs)),
                    "points": rate(sum(any(x["target"] == "points" for x in r[gate]["insights"]) for r in runs), len(runs)),
                    "attribute": rate(sum(any(x["target"] != "points" for x in r[gate]["insights"]) for r in runs), len(runs)),
                }
            else:
                s[gate] = {"found": rate(sum(r[gate]["found"] for r in runs), len(runs))}
        p = [x for r in runs for x in r.get("p_context", [])]
        cross = [x for r in runs for x in r["B"]["insights"]]
        s["p_context_le_05"] = rate(sum(x <= 0.05 for x in p), len(p))
        s["lisa_composition"] = rate(sum(x["lisa"] for x in cross), len(cross))
        s["selected_by_step2"] = rate(sum(bool(r.get("selected")) for r in runs), len(runs))
        for key in ("space", "longest", "n_tests", "n_tests_plain", "space_plain", "seconds", "cells"):
            values = [r[key] for r in runs if key in r]
            s[key] = round(float(np.mean(values)), 2) if values else None
        out[name] = s
    return out


def format_summary(summary: dict[str, Any]) -> str:
    lines = ["scenario  gate  measure     rate   [wilson95]          n"]  # gates A, B0, B
    for name, s in summary.items():
        for gate in GATES:
            for measure, r in s[gate].items():
                lines.append(f"{name:<9} {gate:<5} {measure:<10} {r['rate'] if r['rate'] is not None else '-':>6}  {r['wilson95']}  {r['n']:>4}")
        r = s["p_context_le_05"]
        lines.append(f"{name:<9} null  p<=0.05    {r['rate'] if r['rate'] is not None else '-':>6}  {r['wilson95']}  {r['n']:>4}")
        lines.append(
            f"{name:<9} space {s['space']} (plain {s['space_plain']}), longest {s['longest']}, n_tests {s['n_tests']} (plain {s['n_tests_plain']}),"
            f" lisa∧band {s['lisa_composition']['rate']}, {s['seconds']} s, {s['cells']} cells, errors {s['errors']}"
        )
    return "\n".join(lines)


def run_engine(name: str, seed: int, overrides: dict[str, Any], library: dict[str, Any]) -> dict[str, Any]:
    """The same upload through the graph service (``Engine.upload`` with its layers as context files): the admitted
    patterns' ``p_context`` must equal the library's gate B."""
    from tests.doubles import HashingEmbedder

    folder = scratch_dir(f"crosstype_engine/{name}/{seed}", fresh=True)
    engine = build_engine(folder / "ws", embedder=HashingEmbedder(), **overrides)
    upload = generate(name, seed)
    (folder / "target.csv").write_bytes(upload["frame"].to_csv(index=False).encode())
    for layer, frame in upload["layers"].items():
        (folder / f"{layer}.csv").write_bytes(frame.to_csv(index=False).encode())
    streams = [(f"{layer}.csv", (folder / f"{layer}.csv").open("rb")) for layer in upload["layers"]]
    try:
        with (folder / "target.csv").open("rb") as target:
            (record,) = [
                engine.process(r["batch_id"])
                for r in engine.upload("target.csv", target, geo=f"lat,lon@{RES}", context=streams, context_geo="lat,lon")
            ]
    finally:
        for _, s in streams:
            s.close()
    served = sorted(p["provenance"]["spatial"].get("p_context") for p in engine.ws.patterns() if "p_context" in p["provenance"].get("spatial", {}))
    mined = sorted(x["p_context"] for x in library["B"]["insights"] if x["p_context"] is not None)
    return {"scenario": name, "seed": seed, "status": record["status"], "served": served, "library": mined, "identical": served == mined}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--seeds", default="1-20", help="planted scenarios' seeds (X2, X3, X4)")
    parser.add_argument("--null-seeds", default="1-100", help="null scenarios' seeds (X1, X6, X6b)")
    parser.add_argument("--family-seeds", type=int, default=5, help="the first seeds also mined without the layer")
    parser.add_argument("--engine-seeds", default="1", help="seeds also sent through the graph service; empty to skip")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label", default="current")
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE", help="a settings field by its upper-case name")
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    names = args.scenarios.split(",")
    jobs = [(n, s) for n in names for s in parse_seeds(args.null_seeds if n in NULL else args.seeds)]
    first = {n: min(s for m, s in jobs if m == n) for n in names}
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_seed, *zip(*[(n, s, overrides, s < first[n] + args.family_seeds) for n, s in jobs])))
    summary = analyse(rows)
    if args.engine_seeds:
        by = {(r["scenario"], r["seed"]): r for r in rows}
        summary["engine"] = [run_engine(n, s, overrides, by[(n, s)]) for n in names for s in parse_seeds(args.engine_seeds) if (n, s) in by]
    print(format_summary({k: v for k, v in summary.items() if k != "engine"}))
    for e in summary.get("engine", []):
        print(f"engine {e['scenario']} seed {e['seed']}: {e['status']}, p_context identical to the library: {e['identical']}")
    print(save_result("crosstype_benchmark", args.label, {"summary": summary, "overrides": overrides, "per_upload": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
