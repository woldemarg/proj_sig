"""How the cross-type context scales with the layers of one upload (docs/extra_funcs/spatial_analytics_multi_type/
cross_type_context_architecture.md §14.6): a target of the shipped tables mined through the library with none, one, …
six layers, and for each count the search (selectors, 2- and 3-conjunctions, whether the 3-conjunctions fit the
``COMPUTE_BUDGET``), the validation budget's context cohorts, the Bonferroni family, the cross-type findings each gate
admits (A the median test alone, B the context null) and the seconds of the null.

    python scripts/crosstype_scaling.py                 # Chicago crimes @9 and crashes @8, 0..6 layers

The layers, in the order they are added: for the crimes the bars, liquor stores, 'L' stations, schools, the crashes
(another type of events as a layer) and 1,000 uniform points; for the crashes the schools first and the crimes fifth.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scratch import save_result, settings

from insight_graph_service.core.batch import admit_insights
from insight_graph_service.core.demo import DEMO_DIR
from insight_graph_service.core.settings import PROJECT_ROOT
from subgroup_miner.discovery import build_insights, run_discovery
from subgroup_miner.ingestion import load_dataset
from subgroup_miner.selection import select_insights

WORK = PROJECT_ROOT / "data" / "benchmarks" / "crosstype" / "prepared"  # git-ignored, never beside raw exports
TARGETS = {  # name -> the shipped table, its time column (left out), the geo option, the layers in the order added
    "crimes@9": ("chicago_crimes", "crime_date", "lat,lon@9", ("bars", "liquor_stores", "l_stations", "schools", "crashes", "random")),
    "crashes@8": ("chicago_crashes", "crash_date", "lat,lon@8", ("schools", "bars", "liquor_stores", "l_stations", "crimes", "random")),
}


def levels(column: pd.Series) -> int:
    """The selectors step 3 keeps of a dimension: every level up to the one crossing 95 % of the rows."""
    share = column.value_counts(normalize=True)
    return int(((share.cumsum().shift(1, fill_value=0.0) < 0.95) & ~share.index.astype(str).isin(["nan", "<NA>", "None"])).sum())


def run(name: str, target, geo: str, layers: tuple[str, ...], files: dict, cfg_all) -> dict:
    cfg = cfg_all.miner
    loaded = load_dataset(target, cfg, geo=geo, context=[(k, files[k], "lat,lon") for k in layers])
    result = run_discovery(loaded.frame, cfg, spatial=loaded.spatial)
    t0 = time.perf_counter()
    insights = build_insights(result, cfg, dataset_id=loaded.dataset_id, batch_id="scale", filename=name)
    null_s = time.perf_counter() - t0
    sizes = [levels(loaded.frame[d]) for d in result.profile.selected_dimensions]
    pairs, triples = (sum(math.prod(c) for c in itertools.combinations(sizes, r)) for r in (2, 3))

    def cross(found):
        kept = admit_insights(select_insights(found, cfg).kept, cfg_all.min_insight_weight, cfg_all.max_insights_per_batch)[0]
        return [i for i in kept if any(c.attribute.startswith("context_") for c in i.conditions)]

    off = [
        replace(i, provenance={**i.provenance, "spatial": {k: v for k, v in i.provenance["spatial"].items() if k != "p_context"}}) for i in insights
    ]
    family = next((i.provenance["spatial"]["context_family"] for i in insights if "context_family" in i.provenance["spatial"]), 0)
    return {
        "target": name,
        "layers": len(layers),
        "cells": len(loaded.frame),
        "selectors": sum(sizes),
        "pairs": pairs,
        "triples": triples,
        "space": result.profile.search_space_size,
        "triples_kept": pairs + triples <= cfg.compute_budget,
        "validated": len(result.validated),
        "context_cohorts": family,
        "n_tests": result.n_tests,
        "A": len(cross(off)),
        "B": [f"{i.target} z={i.effect_size:+.2f} | {i.expression}" for i in cross(insights)],
        "null_s": round(null_s, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    files = {k: DEMO_DIR / f"chicago_{k}.csv" for k in ("bars", "liquor_stores", "l_stations", "schools")}
    tables = {key: pd.read_csv(DEMO_DIR / f"{key}.csv").drop(columns=when) for key, when, _, _ in TARGETS.values()}
    for key, frame in tables.items():
        files[key.removeprefix("chicago_")] = WORK / f"{key}.csv"
        frame.to_csv(files[key.removeprefix("chicago_")], index=False)
    crimes = tables["chicago_crimes"]
    lo, hi, rng = crimes[["lat", "lon"]].quantile(0.01), crimes[["lat", "lon"]].quantile(0.99), np.random.default_rng(0)
    uniform = pd.DataFrame({"lat": rng.uniform(lo.lat, hi.lat, 1000), "lon": rng.uniform(lo.lon, hi.lon, 1000)})
    files["random"] = WORK / "random_points.csv"
    uniform.to_csv(files["random"], index=False)
    rows, cfg_all = [], settings()
    for name, (key, _, geo, order) in TARGETS.items():
        for k in range(len(order) + 1):
            row = run(name, files[key.removeprefix("chicago_")], geo, order[:k], files, cfg_all)
            rows.append(row)
            print(
                f"{name} layers={k} cells={row['cells']} selectors={row['selectors']} pairs={row['pairs']} triples={row['triples']} "
                f"space={row['space']} triples_kept={row['triples_kept']} validated={row['validated']} context={row['context_cohorts']} "
                f"n_tests={row['n_tests']} A={row['A']} B={len(row['B'])} null_s={row['null_s']}",
                flush=True,
            )
            for finding in row["B"]:
                print(f"    B: {finding}")
    print(save_result("crosstype_scaling", args.label, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
