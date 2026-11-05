"""Controls for a layer link on real data (the cross-layer experiments of 2026-11-04, R-9): what the layer's null does with
points that carry no effect, run the console's way (``Engine.cross``) on any target and layer, each repeated over seeds:

* ``shuffled``  — the target's type labels (``--column``) shuffled over all its events: no composition structure is left,
  so an admitted share finding is a false one (the small-cell artefact the event weights remove);
* ``fine``      — a layer of the target's own events at random, jittered 30 m: co-located with the events at the cell's
  scale, no link of its own (the null tests the layer's arrangement within parent cells, not this);
* ``coarse``    — the same jittered 1 km: the events' density at the parent's scale, which the null must refuse;
* ``uniform``   — as many points as the layer, uniform over the target's extent;
* ``halves``    — the real layer against two random halves of the target: an admitted finding found again on both.

Per control the share of runs that admitted a layer-link finding (and of share findings for ``shuffled``), each run's
admitted findings and refusals. The shipped Chicago crimes and bars by default.

    python scripts/crosstype_controls.py [--target KEY] [--layer KEY] [--column crime] [--seeds 3] [--ring 0] [--label NAME]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from crosstype_demo import summary
from scratch import build_engine, save_result, scratch_dir, upload_all

from insight_graph_service.core.demo import DEMO_DIR, demo_datasets
from insight_graph_service.core.engine import Engine
from tests.doubles import HashingEmbedder

M_PER_DEGREE = 111_320


def jittered(frame: pd.DataFrame, locator: str, n: int, metres: float, seed: int) -> pd.DataFrame:
    """``n`` of ``frame``'s points at random, each moved ``metres`` (normal, per axis)."""
    lat, lon = locator.split(",")
    rng = np.random.default_rng(seed)
    picked = frame.iloc[rng.choice(len(frame), size=min(n, len(frame)), replace=False)]
    out = pd.DataFrame({"lat": picked[lat].to_numpy(float), "lon": picked[lon].to_numpy(float)})
    shift = rng.normal(0, metres, size=(len(out), 2))
    out["lat"] += shift[:, 0] / M_PER_DEGREE
    out["lon"] += shift[:, 1] / (M_PER_DEGREE * np.cos(np.radians(out["lat"].mean())))
    return out.round(6)


def uniform(frame: pd.DataFrame, locator: str, n: int, seed: int) -> pd.DataFrame:
    lat, lon = (frame[c].to_numpy(float) for c in locator.split(","))
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lat": rng.uniform(np.nanmin(lat), np.nanmax(lat), n),
            "lon": rng.uniform(np.nanmin(lon), np.nanmax(lon), n),
        }
    ).round(6)


def run(
    engine: Engine,
    target: Path,
    options: dict[str, Any],
    layer: Path,
    ring: int,
    tag: str,
) -> dict[str, Any]:
    """``target`` read with ``options`` at ``@9``-like settings of its manifest, ``layer`` kept as a layer, one run."""
    own = upload_all(engine, target, **options)
    kept = upload_all(engine, layer, geo="lat,lon", layer=True)[0]["upload_id"]
    t0 = time.perf_counter()
    (asked,) = engine.cross(own[0]["upload_id"], [kept], ring=ring, allow_coarse=True)  # eval: any level
    out = summary(
        engine,
        engine.process(asked["batch_id"]),
        tag,
        int(own[0].get("level") or 0),
        ring,
        time.perf_counter() - t0,
    )
    return out | {"control": tag}


def controls(
    target: Path,
    options: dict[str, Any],
    layer: Path,
    column: str,
    seeds: int,
    ring: int,
    work: Path,
) -> dict[str, Any]:
    locator = options["geo"].partition("@")[0]
    frame, points = (
        pd.read_csv(target, dtype=str, keep_default_na=False),
        pd.read_csv(layer),
    )
    events = frame.assign(**{c: pd.to_numeric(frame[c]) for c in locator.split(",")})
    runs = []
    for seed in range(1, seeds + 1):
        engine = build_engine(work / f"ws{seed}", embedder=HashingEmbedder())
        shuffled = frame.copy()
        shuffled[column] = np.random.default_rng(seed).permutation(shuffled[column].to_numpy())
        files = {
            "shuffled": (shuffled, layer),
            "fine": (frame, jittered(events, locator, len(points), 30, seed)),
            "coarse": (frame, jittered(events, locator, len(points), 1000, seed)),
        }
        files["uniform"] = (frame, uniform(events, locator, len(points), seed))
        half = np.random.default_rng(100 + seed).permutation(len(frame)) < len(frame) // 2
        files["half a"], files["half b"] = (frame[half], layer), (frame[~half], layer)
        for tag, (rows, points_of) in files.items():
            t_path = work / f"{tag.replace(' ', '_')}_{seed}_target.csv"
            rows.to_csv(t_path, index=False)
            if isinstance(points_of, Path):
                l_path = points_of
            else:
                l_path = work / f"{tag}_{seed}_layer.csv"
                points_of.to_csv(l_path, index=False)
            runs.append(run(engine, t_path, options, l_path, ring, tag) | {"seed": seed})
    rate = {}
    for tag in ("shuffled", "fine", "coarse", "uniform", "half a", "half b"):
        mine = [r for r in runs if r["control"] == tag]
        rate[tag] = {
            "runs": len(mine),
            "admitting": sum(bool(r["admitted"]) for r in mine),
        }
        if tag == "shuffled":
            rate[tag]["admitting_a_share"] = sum(any(a["target"].startswith("share_") for a in r["admitted"]) for r in mine)
    return {"rates": rate, "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="chicago_crimes")
    parser.add_argument("--layer", default="chicago_bars")
    parser.add_argument(
        "--column",
        default="crime",
        help="the target's type column the shuffled control permutes",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    shipped = {d["key"]: d for d in demo_datasets()}
    t0, work = time.perf_counter(), scratch_dir("crosstype_controls", fresh=True)
    result = controls(
        DEMO_DIR / shipped[args.target]["file"],
        shipped[args.target]["options"],
        DEMO_DIR / shipped[args.layer]["file"],
        args.column,
        args.seeds,
        args.ring,
        work,
    )
    for tag, r in result["rates"].items():
        print(f"{tag:<9} {r}")
    print(
        save_result(
            "crosstype_controls",
            args.label,
            {
                "seconds": round(time.perf_counter() - t0, 1),
                "args": vars(args),
                **result,
            },
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
