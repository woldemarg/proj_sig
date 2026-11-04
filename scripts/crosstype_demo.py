"""The Chicago crimes and bars demonstration measured on the console's path (the M18 specification,
docs/extra_funcs/spatial_analytics_multi_type/cross_dataset_analysis_architecture.md §12, holds the decision rule written
before this ran). On a scratch workspace with the hashing embedder: the shipped crimes of 2025 uploaded with their
manifest's options (``@auto`` reads H3 r9) and once more at every other level a setting names, the shipped bars and CTA 'L'
stations kept as layer-only uploads, and one cross-type run of the crimes with each layer per setting — r9 over the cell
and its six neighbours (the console's default), r9 over the cell alone, r10 over the cell alone — through ``Engine.cross``,
as ``POST /api/crosstype``; then the literature's frequency question as the console asks it — the crimes read again
under ``where=crime=robbery`` (a reading of its own at the level ``@auto`` gives its rows) and crossed with the bars over
both disks. Per run: the cross-type insights admitted (target, z, the layer's verdict ``p_context`` times its family), the
refusals by reason, the seconds.

    python scripts/crosstype_demo.py            # the eight runs; the result in .scratch/results
    python scripts/crosstype_demo.py --m19      # the M19 grid pre-registered in the concept's §14a (#281)

The M19 grid: the crimes of 2025 with per-type counters (``M19_COUNTS``), each layer on its own (1:1), two supports —
the finest level over the cell alone and r9 over the cell and its six neighbours, eval settings, not the console's —, the
``map`` and the ``zone`` baseline, event weights off and on, and the counters off at the console's setting (the family
they add); every admitted finding then looked for in the same run of 2024 (the same expression and target, the same
sign); per counted type the map median of its count at the finest level and one coarser (item 12); the co-location
quotient of each counted type to each layer on the raw points and the model references (``models``: needs
``statsmodels``, the ``eval`` dependency group; absent: skipped, said so).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scratch import build_engine, save_result, scratch_dir, upload_all

from insight_graph_service.core.demo import DEMO_DIR, demo_datasets
from insight_graph_service.core.engine import Engine

TARGET = "chicago_crimes"
LAYERS = ("chicago_bars", "chicago_l_stations")
SETTINGS = ((9, 1), (9, 0), (10, 0))  # (H3 level, disk radius), the console's default first
FREQUENCY = ("chicago_bars", "crime=robbery")  # the frequency of one class near the layer: the target under a filter, crossed
HELD_OUT = "chicago_crimes_2024"  # the replication year (concept §14a)
M19_COUNTS = "crime=robbery,crime=deceptive,crime=theft"  # the counted types: a rare one, the stations' one, a common one
M19_SUPPORTS = ((9, 0), (9, 1))  # (level, disk): the finest @auto gives the crimes, over the cell alone; r9 and its six neighbours


def summary(engine: Engine, record: dict[str, Any], layer: str, level: int, ring: int, seconds: float) -> dict[str, Any]:
    """One run as the spike reports it: its admitted insights with the layer's verdict, its refusals, its seconds."""
    admitted = []
    for p in engine.ws.patterns():
        if p["dataset_id"] == record["dataset_id"]:
            spatial = p["provenance"]["spatial"]
            verdict = min(1.0, spatial["p_context"] * spatial["context_family"]) if spatial.get("p_context") is not None else None
            admitted.append({"target": p["target"], "z": round(p["effect_size"], 2), "p_context_adjusted": verdict, "scope": p["expression"]})
    return {
        "layer": layer,
        "level": level,
        "ring": ring,
        "status": record["status"],
        "error": (record.get("error") or {}).get("code"),
        "admitted": admitted,
        "refused": dict(record.get("metrics", {}).get("pruned", {})),  # a run that admits none counts its refusals too
        "seconds": round(seconds, 1),
    }


def measure(
    engine: Engine, target: Path, options: dict[str, Any], layers: dict[str, Path], settings: Sequence[tuple[int, int]], frequency: tuple[str, str]
) -> dict[str, Any]:
    """``target`` uploaded with ``options`` and again at each level of ``settings`` its own reading lacks (the same file:
    one upload id), each of ``layers`` kept as a layer-only upload, and one cross-type run per layer and setting; then
    ``target`` read under ``frequency``'s filter (its own upload at the level ``@auto`` gives it) crossed with
    ``frequency``'s layer over each disk."""
    own = upload_all(engine, target, **options)
    locator = options["geo"].partition("@")[0]
    for level in sorted({str(lv) for lv, _ in settings} - {r["level"] for r in own}):
        own += upload_all(engine, target, **options | {"geo": f"{locator}@{level}"})
    kept = {name: upload_all(engine, path, geo=locator, layer=True)[0]["upload_id"] for name, path in layers.items()}
    runs = []
    for name, layer in kept.items():
        for level, ring in settings:
            t0 = time.perf_counter()
            for record in engine.cross(own[0]["upload_id"], [layer], ring=ring, levels=[level], allow_coarse=True):  # eval: any level
                runs.append(summary(engine, engine.process(record["batch_id"]), name, level, ring, time.perf_counter() - t0))
    layer, where = frequency
    reading = upload_all(engine, target, **options | {"where": where})
    for ring in (1, 0):
        t0 = time.perf_counter()
        for record in engine.cross(reading[0]["upload_id"], [kept[layer]], ring=ring, allow_coarse=True):  # the reading's own finest level
            runs.append(
                summary(engine, engine.process(record["batch_id"]), f"{layer} · {where}", int(record["level"]), ring, time.perf_counter() - t0)
            )
    return {"target": [{"level": r["level"], "where": r.get("where"), "status": r["status"]} for r in own + reading], "runs": runs}


def run(engine: Engine, target: str, layer: str, level: int, ring: int, **options: Any) -> dict[str, Any]:
    """One cross-type run processed, summarized as ``summary`` does, with its family and the admitted findings' keys."""
    t0 = time.perf_counter()
    (record,) = [engine.process(r["batch_id"]) for r in engine.cross(target, [layer], ring=ring, levels=[level], allow_coarse=True, **options)]
    out = summary(engine, record, "", level, ring, time.perf_counter() - t0)
    family = {p["provenance"]["spatial"].get("context_family") for p in engine.ws.patterns() if p["dataset_id"] == record["dataset_id"]}
    return out | {"family": max(family - {None}, default=None), "keys": sorted((a["scope"], a["target"], a["z"] > 0) for a in out["admitted"])}


def type_medians(path: Path, locator: str, counts: str, levels: tuple[int, int]) -> dict[str, dict[str, float]]:
    """Item 12: per counted type the map median of its count over the target's occupied cells, at each level."""
    import h3

    from subgroup_miner.ingestion import read_rows

    frame = read_rows(path)
    lat, lon = (frame[c].to_numpy(float) for c in locator.split(","))
    out = {}
    for spec in counts.split(","):
        column, _, value = spec.partition("=")
        typed = (frame[column].astype("string") == value).to_numpy()
        out[value] = {}
        for res in levels:
            cells = pd.Series([h3.latlng_to_cell(a, b, res) for a, b in zip(lat, lon)])
            out[value][f"r{res}"] = float(pd.Series(typed).groupby(cells).sum().median())
    return out


def clq(path: Path, locator: str, counts: str, layer: Path, layer_locator: str) -> dict[str, float | str]:
    """The co-location quotient of each counted type to the layer (Leslie & Kronenfeld): of the type's events, the share
    whose nearest other point of the pooled events and layer points is a layer point, over the layer's share of the pool;
    on the raw points, with the cell run's locator and rows. A layer without raw points is refused."""
    from scipy.spatial import cKDTree

    from subgroup_miner.ingestion import read_rows

    frame, points = read_rows(path), read_rows(layer)
    if not set(layer_locator.split(",")) <= set(points.columns):
        return {"refused": "the layer has no raw points under its locator"}
    xy = lambda f, loc: np.c_[[f[c].to_numpy(float) for c in loc.split(",")]].T * [1.0, np.cos(np.radians(41.8))]  # noqa: E731
    events, spots = xy(frame, locator), xy(points, layer_locator)
    pool = np.vstack([events, spots])
    _, nearest = cKDTree(pool).query(events, k=2)
    to_layer = nearest[:, 1] >= len(events)
    out = {}
    for spec in counts.split(","):
        column, _, value = spec.partition("=")
        typed = (frame[column].astype("string") == value).to_numpy()
        out[value] = round(float(to_layer[typed].mean() / (len(spots) / (len(pool) - 1))), 3)
    return out


def models(path: Path, locator: str, counts: str, layers: dict[str, Path], level: int = 9, ring: int = 1) -> dict[str, Any]:
    """The model references (concept §14a) on the cells the run reads — the target's occupied H3 cells at ``level``, its
    rows and locator —, per counted type and layer, each on «the layer within the disk» (a point of it in the cell or
    its ``ring`` neighbours, the band's *none nearby* or not): a negative-binomial regression (NB2, its dispersion
    estimated) of the type's count with the cell's own area as offset (log km²) — the incidence-rate ratio, the
    frequency question — and a quasi-binomial regression of the type's share of the cell's events (the overdispersed
    binomial: the beta-binomial's usual stand-in) — the odds ratio, the composition question —, each with its p."""
    import h3
    import statsmodels.api as sm

    from subgroup_miner.ingestion import read_rows

    def cells_of(frame: pd.DataFrame, loc: str) -> tuple[pd.Series, np.ndarray]:
        lat, lon = (frame[c].to_numpy(float) for c in loc.split(","))
        ok = np.isfinite(lat) & np.isfinite(lon)
        return pd.Series([h3.latlng_to_cell(a, b, level) for a, b in zip(lat[ok], lon[ok])]), ok

    frame = read_rows(path)
    cells, ok = cells_of(frame, locator)
    rows, events = frame.loc[ok].reset_index(drop=True), cells.value_counts()
    offset = np.log([h3.cell_area(c, "km^2") for c in events.index])
    out: dict[str, Any] = {}
    for name, layer in layers.items():
        spots = set(cells_of(read_rows(layer), locator)[0])
        near = np.array([any(n in spots for n in h3.grid_disk(c, ring)) for c in events.index], dtype=float)
        x = sm.add_constant(near)
        for spec in counts.split(","):
            column, _, value = spec.partition("=")
            k = (rows[column].astype("string") == value).groupby(cells).sum().reindex(events.index, fill_value=0).to_numpy(float)
            nb = sm.NegativeBinomial(k, x, offset=offset).fit(disp=0, maxiter=200)
            qb = sm.GLM(np.c_[k, events.to_numpy(float) - k], x, family=sm.families.Binomial()).fit(scale="X2")
            out.setdefault(name, {})[value] = {
                "cells": len(k),
                "near": int(near.sum()),
                "irr": round(float(np.exp(nb.params[1])), 3),
                "p_irr": float(nb.pvalues[1]),
                "or": round(float(np.exp(qb.params[1])), 3),
                "p_or": float(qb.pvalues[1]),
            }
    return out


def m19(engine: Engine, weighted: Engine, shipped: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered grid (concept §14a) on the 2025 crimes, its findings looked for in 2024, the medians, the CLQ."""
    options = shipped[TARGET]["options"]
    locator = options["geo"].partition("@")[0]
    runs = []
    for which, eng in (("unweighted", engine), ("weighted", weighted)):
        own = {key: upload_all(eng, DEMO_DIR / shipped[key]["file"], **options)[0]["upload_id"] for key in (TARGET, HELD_OUT)}
        kept = {name: upload_all(eng, DEMO_DIR / shipped[name]["file"], geo=locator, layer=True)[0]["upload_id"] for name in LAYERS}
        for name, layer in kept.items():
            if which == "unweighted":  # the counters' cost at the console's setting: the run without them (the reading's own)
                runs.append(run(eng, own[TARGET], layer, 9, 1) | {"layer": name, "weights": which, "baseline": "map", "counters": False})
            for level, ring in M19_SUPPORTS:
                for baseline in ("map", "zone"):  # the counters asked for the run, checked against the year's own rows
                    found = run(eng, own[TARGET], layer, level, ring, baseline=baseline, count_values=M19_COUNTS)
                    again = run(eng, own[HELD_OUT], layer, level, ring, baseline=baseline, count_values=M19_COUNTS)
                    replicated = sorted(set(map(tuple, found["keys"])) & set(map(tuple, again["keys"])))
                    runs.append(found | {"layer": name, "weights": which, "baseline": baseline, "replicated": replicated})
    target, layers = DEMO_DIR / shipped[TARGET]["file"], {name: DEMO_DIR / shipped[name]["file"] for name in LAYERS}
    try:
        references: Any = {f"r{level} disk {ring}": models(target, locator, M19_COUNTS, layers, level, ring) for level, ring in M19_SUPPORTS}
    except ImportError:
        references = "skipped: statsmodels is not installed (the eval dependency group)"
    return {
        "runs": runs,
        "medians": type_medians(target, locator, M19_COUNTS, (9, 8)),
        "clq": {name: clq(target, locator, M19_COUNTS, path, locator) for name, path in layers.items()},
        "models": references,
    }


def main() -> int:
    from tests.doubles import HashingEmbedder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current")
    parser.add_argument("--m19", action="store_true", help="the M19 grid pre-registered in the concept's §14a")
    args = parser.parse_args()
    shipped = {d["key"]: d for d in demo_datasets()}
    if args.m19:
        t0, base = time.perf_counter(), scratch_dir("crosstype_m19", fresh=True)
        result = m19(
            build_engine(base / "ws", embedder=HashingEmbedder()),
            build_engine(base / "wsw", embedder=HashingEmbedder(), weight_by_events=True),
            shipped,
        )
        for r in result["runs"]:
            tag = f"{r['layer']:<20} {r['weights']:<10} r{r['level']} disk {r['ring']} {r['baseline']:<4}{'' if r.get('counters', True) else ' no counters'}"
            print(f"{tag}: admitted {len(r['admitted'])}, family {r['family']}, replicated {len(r.get('replicated', []))}, refused {r['refused']}")
            for a in r["admitted"]:
                print(f"    {a['target']:<28} z={a['z']:+.2f} | {a['scope']}")
        print("medians:", result["medians"], "\nclq:", result["clq"], "\nmodels:", result["models"])
        print(save_result("crosstype_demo", f"{args.label}_m19", {"seconds": round(time.perf_counter() - t0, 1), **result}))
        return 0
    engine = build_engine(scratch_dir("crosstype_demo", fresh=True) / "ws", embedder=HashingEmbedder())
    t0 = time.perf_counter()
    layers = {key: DEMO_DIR / shipped[key]["file"] for key in LAYERS}
    result = measure(engine, DEMO_DIR / shipped[TARGET]["file"], shipped[TARGET]["options"], layers, SETTINGS, FREQUENCY)
    print("target:", ", ".join(f"r{t['level']}{' ' + t['where'] if t['where'] else ''} {t['status']}" for t in result["target"]))
    for r in result["runs"]:
        print(
            f"{r['layer']:<36} r{r['level']} disk {r['ring']}: {r['status']:<6} admitted {len(r['admitted'])}, refused {r['refused']}, {r['seconds']} s"
        )
        for a in r["admitted"]:
            print(f"    {a['target']:<28} z={a['z']:+.2f} p_context·m={a['p_context_adjusted']:.3g} | {a['scope']}")
    print(save_result("crosstype_demo", args.label, {"seconds": round(time.perf_counter() - t0, 1), **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
