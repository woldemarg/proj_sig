"""Cross-type context on real data (docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md
§8.3): three Chicago cases, each a hypothesis the literature states, mined on one period through the library with the
context null off (A) and on (B), and every cross-type finding re-tested on a held-out period.

    python scripts/crosstype_validation.py                # R1, R2, R3; downloads into data/benchmarks/crosstype/
    python scripts/crosstype_validation.py --offline      # the copies already downloaded
    python scripts/crosstype_validation.py --cases R2

R1 robbery and assault around alcohol outlets (two layers: on-premise licences — tavern, consumption on premises, late
hour — and package-goods stores); R2 street robbery around CTA 'L' stations; both on the city's crimes of 2025, held out
on those of 2024. R3, the negative control: the shipped crashes around the shipped schools, its first half-year mined
and its second held out — no school effect beyond the shared density is expected.

The City of Chicago Data Portal re-publishes its tables; each download's SHA-256 is recorded with the results, not
pinned. A finding replicates when its scope, read on the held-out period (the bands cut on that period's cells), shifts
its metric the same way by at least MIN_EFFECT_Z, and passes the median test and the context null, each adjusted over
the findings re-tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from scratch import save_result, settings

from insight_contracts import Insight
from insight_graph_service.core.batch import admit_insights
from insight_graph_service.core.demo import DEMO_DIR
from insight_graph_service.core.settings import PROJECT_ROOT
from subgroup_miner.discovery import DiscoveryError, build_insights, recompute_statistics, run_discovery
from subgroup_miner.ingestion import load_dataset
from subgroup_miner.selection import select_insights

FOLDER = PROJECT_ROOT / "data" / "benchmarks" / "crosstype"  # git-ignored, as data/ is
PORTAL = "https://data.cityofchicago.org/resource"
CRIME_FIELDS = "id,date,primary_type,arrest,domestic,location_description,latitude,longitude"
SOURCES = {  # file -> (Socrata resource, query)
    **{
        f"crimes_{y}.csv": (
            "ijzp-q8t2.csv",
            {"$select": CRIME_FIELDS, "$where": f"year={y} AND latitude IS NOT NULL", "$order": "id", "$limit": 500000},
        )
        for y in (2024, 2025)
    },
    "liquor_licenses.csv": (
        "nrmj-3kcf.csv",
        {"$select": "license_id,license_description,latitude,longitude", "$where": "latitude IS NOT NULL", "$limit": 50000},
    ),
    "l_stops.json": ("8pix-ypme.json", {"$select": "map_id,station_name,location", "$order": "map_id", "$limit": 5000}),
}
CRIME = {  # primary type -> class (at most 8 classes: each becomes a share of the cells' crimes)
    **dict.fromkeys(["ROBBERY"], "robbery"),
    **dict.fromkeys(["BATTERY", "ASSAULT", "HOMICIDE", "CRIMINAL SEXUAL ASSAULT"], "assault"),
    **dict.fromkeys(["THEFT"], "theft"),
    **dict.fromkeys(["MOTOR VEHICLE THEFT", "BURGLARY", "CRIMINAL DAMAGE", "ARSON"], "property"),
    **dict.fromkeys(["DECEPTIVE PRACTICE"], "deceptive"),
    **dict.fromkeys(["NARCOTICS", "WEAPONS VIOLATION", "CONCEALED CARRY LICENSE VIOLATION"], "drugs_weapons"),
}
STREET = {"STREET", "SIDEWALK", "ALLEY"}
ON_PREMISE = {"Tavern", "Consumption on Premises - Incidental Activity", "Late Hour"}
CASES = {  # name -> the target's and the hold-out's file, the layers, the geo option, the outcomes the hypothesis names
    "R1": {
        "target": "crimes_2025",
        "holdout": "crimes_2024",
        "layers": ("bars", "liquor_stores"),
        "geo": "lat,lon@9",
        "outcomes": ("share_crime_robbery", "share_crime_assault"),
    },
    "R2": {"target": "crimes_2025", "holdout": "crimes_2024", "layers": ("l_stations",), "geo": "lat,lon@9", "outcomes": ("share_crime_robbery",)},
    "R3": {"target": "crashes_first", "holdout": "crashes_second", "layers": ("schools",), "geo": "lat,lon@8", "outcomes": ()},
}


def fetch(name: str, offline: bool) -> Path:
    """One source table into ``FOLDER`` (kept: an offline run reads the copy)."""
    path = FOLDER / name
    if not path.is_file() and not offline:
        resource, query = SOURCES[name]
        url = f"{PORTAL}/{resource}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"User-Agent": "sig-crosstype-validation"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            FOLDER.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.read())
    return path


def crimes(raw: Path) -> pd.DataFrame:
    """A year of crimes: the point, the crime's class, street or not, arrest, domestic, the hour band."""
    df = pd.read_csv(raw, dtype=str, keep_default_na=False, na_values=[""])
    when = pd.to_datetime(df["date"], errors="coerce")
    out = pd.DataFrame({"lat": pd.to_numeric(df["latitude"]).round(6), "lon": pd.to_numeric(df["longitude"]).round(6)})
    out["crime"] = df["primary_type"].map(CRIME).fillna("other")
    out["street"] = df["location_description"].isin(STREET).map({True: "yes", False: "no"})
    out["arrest"], out["domestic"] = df["arrest"].str.lower(), df["domestic"].str.lower()
    out["hour_band"] = pd.cut(when.dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out


def layers(offline: bool) -> dict[str, pd.DataFrame]:
    """The context layers by name, one row per place (a business's several licences, a station's platforms: once)."""
    lic = pd.read_csv(fetch("liquor_licenses.csv", offline), dtype=str)
    points = lambda d: d.assign(lat=pd.to_numeric(d["latitude"]).round(6), lon=pd.to_numeric(d["longitude"]).round(6))[
        ["lat", "lon"]
    ].drop_duplicates()  # noqa: E731
    stops = pd.DataFrame(json.loads(fetch("l_stops.json", offline).read_text(encoding="utf-8"))).drop_duplicates("map_id")
    lonlat = stops["location"].map(lambda p: p["coordinates"])
    return {
        "bars": points(lic[lic["license_description"].isin(ON_PREMISE)]),
        "liquor_stores": points(lic[lic["license_description"] == "Package Goods"]),
        "l_stations": pd.DataFrame({"lat": lonlat.map(lambda c: round(float(c[1]), 6)), "lon": lonlat.map(lambda c: round(float(c[0]), 6))}),
        "schools": pd.read_csv(DEMO_DIR / "chicago_schools.csv")[["lat", "lon"]],
    }


def tables(offline: bool) -> dict[str, pd.DataFrame]:
    """The targets and hold-outs by name: two years of crimes, the shipped crashes split at its half-year."""
    crashes = pd.read_csv(DEMO_DIR / "chicago_crashes.csv")
    first = pd.to_datetime(crashes["crash_date"]) < pd.Timestamp("2024-04-01")
    out = {f"crimes_{y}": crimes(fetch(f"crimes_{y}.csv", offline)) for y in (2024, 2025)}
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
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--offline", action="store_true", help="read the copies already downloaded")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    work = FOLDER / "prepared"
    work.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for name, frame in {**tables(args.offline), **layers(args.offline)}.items():
        frame.to_csv(path := work / f"{name}.csv", index=False)
        files[name] = path
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FOLDER.glob("*.*"))}
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
    print(save_result("crosstype_validation", args.label, {"sources": digests, "cases": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
