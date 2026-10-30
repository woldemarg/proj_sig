"""Cross-type context on real data (docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md
§8.2a-8.2b): Chicago cases, each a hypothesis the literature states, mined on one period through the library with the
context null off (A) and on (B), and every cross-type finding re-tested on a held-out period.

    python scripts/crosstype_validation.py                    # R1, R2, R3 on the shipped tables (data/demo, in git)
    python scripts/crosstype_validation.py --cases R1c,R2c    # the second round: violent crimes, their count
    python scripts/crosstype_validation.py --full             # every crime of the year (the pinned portal exports)

R1 robbery and assault around alcohol outlets (two layers: ``chicago_bars``, ``chicago_liquor_stores``); R2 street
robbery around CTA 'L' stations (``chicago_l_stations``); both on the city's crimes of 2025 (``chicago_crimes``, a sixth
by a stable hash), held out on those of 2024 (``chicago_crimes_2024``). R3, the negative control: the crashes around the
schools, the first half-year mined and the second held out. ``--full`` reads every crime of both years from the exports
``scripts/demo_datasets.py`` pins (fetched into ``data/raw/`` when missing).

A finding replicates when its scope, read on the held-out period (the bands cut on that period's cells), shifts its
metric the same way by at least MIN_EFFECT_Z, and passes the median test and the context null, each adjusted over the
findings re-tested.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import demo_datasets
import pandas as pd
from demo_data import chicago_crimes
from scratch import save_result, settings

from insight_contracts import Insight
from insight_graph_service.core.batch import admit_insights
from insight_graph_service.core.demo import DEMO_DIR
from insight_graph_service.core.settings import PROJECT_ROOT
from subgroup_miner.discovery import DiscoveryError, build_insights, recompute_statistics, run_discovery
from subgroup_miner.ingestion import load_dataset
from subgroup_miner.selection import select_insights

WORK = PROJECT_ROOT / "data" / "benchmarks" / "crosstype" / "prepared"  # the prepared targets (git-ignored); never beside raw exports
LAYERS = {"bars": "chicago_bars", "liquor_stores": "chicago_liquor_stores", "l_stations": "chicago_l_stations", "schools": "chicago_schools"}
CRIMES = {2025: "chicago_crimes", 2024: "chicago_crimes_2024"}
CASES = {  # name -> the target's and the hold-out's table, the layers, the geo option, the outcomes the hypothesis names
    "R1": {
        "target": "crimes_2025",
        "holdout": "crimes_2024",
        "layers": ("bars", "liquor_stores"),
        "geo": "lat,lon@9",
        "outcomes": ("share_crime_robbery", "share_crime_assault"),
    },
    "R2": {"target": "crimes_2025", "holdout": "crimes_2024", "layers": ("l_stations",), "geo": "lat,lon@9", "outcomes": ("share_crime_robbery",)},
    # the second round (§8.2b): the literature states frequencies — violent crimes alone, so points is their count
    "R1c": {"target": "violent_2025", "holdout": "violent_2024", "layers": ("bars", "liquor_stores"), "geo": "lat,lon@9", "outcomes": ("points",)},
    "R2c": {"target": "violent_2025", "holdout": "violent_2024", "layers": ("l_stations",), "geo": "lat,lon@9", "outcomes": ("points",)},
    "R3": {"target": "crashes_first", "holdout": "crashes_second", "layers": ("schools",), "geo": "lat,lon@8", "outcomes": ()},
}


def crimes(year: int, full: bool) -> pd.DataFrame:
    """A year of crimes without its time column: the shipped sixth, or (``full``) every crime of the pinned export."""
    key = CRIMES[year]
    if not full:
        return pd.read_csv(DEMO_DIR / f"{key}.csv").drop(columns="crime_date")
    module, folder = importlib.import_module(f"demo_data.{key}"), demo_datasets.RAW / key
    for name, url in module.URLS.items():
        if not (folder / name).is_file():
            demo_datasets.fetch(url, folder, [name])
    return chicago_crimes.prepare(folder, year=year, thin=1).drop(columns="crime_date")


def tables(full: bool) -> dict[str, pd.DataFrame]:
    """The targets and hold-outs by name: two years of crimes (all, violent), the shipped crashes split at its half-year."""
    crashes = pd.read_csv(DEMO_DIR / "chicago_crashes.csv")
    first = pd.to_datetime(crashes["crash_date"]) < pd.Timestamp("2024-04-01")
    out = {f"crimes_{y}": crimes(y, full) for y in CRIMES}
    out |= {f"violent_{y}": out[f"crimes_{y}"].query("crime in ('robbery', 'assault')") for y in CRIMES}
    return out | {"crashes_first": crashes[first].drop(columns="crash_date"), "crashes_second": crashes[~first].drop(columns="crash_date")}


def toward(insight: Insight, names: tuple[str, ...]) -> str:
    """``for`` when the scope says "more of the layer nearby, more of an outcome" or "none nearby, less", ``against`` the
    other way, ``-`` for an outcome the hypothesis does not name."""
    band = next(c.value for c in insight.conditions if c.attribute.startswith("context_"))
    if insight.target not in names:
        return "-"
    return "for" if (band != "none nearby") == (insight.effect_size > 0) else "against"


def run_case(name: str, case: dict[str, Any], files: dict[str, Path], cfg_all) -> dict[str, Any]:
    """One case: the target mined with its layers, the cross-type findings of both gates, each re-tested on the hold-out."""
    cfg = cfg_all.miner
    context = [(layer, files[layer], "lat,lon") for layer in case["layers"]]
    t0 = time.perf_counter()
    loaded = load_dataset(files[case["target"]], cfg, geo=case["geo"], context=context)
    try:
        result = run_discovery(loaded.frame, cfg, spatial=loaded.spatial)
    except DiscoveryError as exc:
        return {"case": name, "error": exc.code}
    insights = build_insights(result, cfg, dataset_id=loaded.dataset_id, batch_id="validation", filename=name)
    admit = lambda found: admit_insights(select_insights(found, cfg).kept, cfg_all.min_insight_weight, cfg_all.max_insights_per_batch)[0]  # noqa: E731
    off = [
        replace(i, provenance={**i.provenance, "spatial": {k: v for k, v in i.provenance["spatial"].items() if k != "p_context"}}) for i in insights
    ]
    cross = lambda found: [i for i in found if any(c.attribute.startswith("context_") for c in i.conditions)]  # noqa: E731
    gate_a, gate_b = cross(admit(off)), cross(admit(insights))
    mined_s = time.perf_counter() - t0
    held = load_dataset(files[case["holdout"]], cfg, geo=case["geo"], context=context)
    by_expression = {i.expression: i for i in insights}  # with its p_context, whichever gate admitted it
    in_a, in_b = {i.expression for i in gate_a}, {i.expression for i in gate_b}
    scopes = list(dict.fromkeys([*in_a, *in_b]))  # every finding either gate admits, once
    family = len(scopes)
    rows = []
    for expression in scopes:
        ins = by_expression[expression]
        conditions = [(c.attribute, c.value) for c in ins.conditions]
        fresh = recompute_statistics(held.frame, conditions, expression, [ins.target], held.spatial, max(family, 1), cfg, max(family, 1))
        z = fresh["shifts"][0]["robust_z"] if fresh.get("shifts") else 0.0
        p_ctx = fresh.get("spatial", {}).get("p_context")
        replicated = (
            fresh["support"] >= cfg.min_support_rows
            and z * ins.effect_size > 0
            and abs(z) >= cfg.min_effect_z
            and fresh.get("p_adjusted", 1.0) <= cfg.max_p_adjusted
            and (p_ctx is None or min(1.0, p_ctx * family) <= cfg.max_p_adjusted)
        )
        spatial = ins.provenance["spatial"]
        rows.append(
            {
                "scope": " ∧ ".join(f"{a}={v}" for a, v in conditions),
                "target": ins.target,
                "z": round(ins.effect_size, 2),
                "gates": ("A" if expression in in_a else "") + ("B" if expression in in_b else ""),
                "p_context_adjusted": round(min(1.0, spatial["p_context"] * spatial["context_family"]), 4) if "p_context" in spatial else None,
                "toward": toward(ins, case["outcomes"]),
                "holdout": {
                    "support": fresh["support"],
                    "z": round(z, 2),
                    "p_adjusted": fresh.get("p_adjusted"),
                    "p_context": p_ctx,
                    "replicated": bool(replicated),
                },
            }
        )
    return {
        "case": name,
        "cells": len(loaded.frame),
        "layers": {layer: int(len(pd.read_csv(files[layer]))) for layer in case["layers"]},
        "bands": [c for c in loaded.frame if c.startswith("context_")],
        "admitted": {"A": len(gate_a), "B": len(gate_b), "all": len(admit(insights))},
        "n_tests": result.n_tests,
        "space": result.profile.search_space_size,
        "context_family": next((i.provenance["spatial"]["context_family"] for i in insights if "context_family" in i.provenance["spatial"]), 0),
        "findings": rows,
        "mined_s": round(mined_s, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="R1,R2,R3")  # the first round; R1c,R2c the second (§8.2b)
    parser.add_argument("--full", action="store_true", help="every crime of both years, not the shipped sixth")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    files = {name: DEMO_DIR / f"{key}.csv" for name, key in LAYERS.items()}  # the layers as shipped: complete
    for name, frame in tables(args.full).items():
        frame.to_csv(path := WORK / f"{name}.csv", index=False)
        files[name] = path
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files.values()))}
    results = [run_case(name, CASES[name], files, settings()) for name in args.cases.split(",")]
    for r in results:
        print(
            f"\n{r['case']}: {r.get('cells')} cells, layers {r.get('layers')}, admitted {r.get('admitted')}, n_tests {r.get('n_tests')}, family {r.get('context_family')}"
        )
        for f in r.get("findings", []):
            h = f["holdout"]
            print(
                f"  [{f['gates']:<2}] {f['target']:<26} z={f['z']:+.2f} p_ctx·m={f['p_context_adjusted']} {f['toward']:<7} | hold-out z={h['z']:+.2f} rep={h['replicated']} | {f['scope']}"
            )
    print(save_result("crosstype_validation", args.label, {"full": args.full, "sources": digests, "cases": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
