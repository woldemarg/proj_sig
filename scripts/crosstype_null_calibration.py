"""The layer null calibrated on synthetic null fields (docs/extra_funcs/spatial_analytics_multi_type/
cross_layer_analysis_concept.md §14a, items 7 and 10). Per repeat: a target over the 271 H3 cells around a point — its
metric a spatially autocorrelated field (white noise smoothed over each cell's ring, the smoothing weight drawn so the
cells' Moran's I spans about 0–0.7), two event kinds whose share follows a second such field — and a layer of a dozen
points placed with no regard to either; the cells read with the layer as context (one forced band, the default disk),
mined and admitted as a batch is (``run_discovery``, ``build_insights``, selection, the weight floor). A repeat with at
least one admitted insight on the layer's band is a family-wise error: the layer cannot matter. Per repeat also the
share of the null's draws that fall below ``MIN_SUPPORT_ROWS`` for each band value's cohort (item 10), replayed with
the null's own deal (``context.deal``).

    python scripts/crosstype_null_calibration.py [--reps 200] [--weights]     # the result in .scratch/results
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd
from scratch import admitted, save_result, settings, without

from subgroup_miner.config import MinerConfig
from subgroup_miner.context import NONE_NEARBY, deal, dealt_band
from subgroup_miner.discovery import build_insights, run_discovery
from subgroup_miner.ingestion import load_dataset

RES, ORIGIN = 7, h3.latlng_to_cell(48.0, 11.0, 7)  # a neutral place: the calibration is about geometry, not a city


def smooth(cells: list[str], index: dict[str, int], weight: float, rng: np.random.Generator) -> np.ndarray:
    """White noise over ``cells`` pulled toward each cell's ring mean ``weight`` of the way, five times: autocorrelated."""
    field = rng.normal(size=len(cells))
    rings = [[index[n] for n in h3.grid_ring(c, 1) if n in index] for c in cells]
    for _ in range(5):
        field = (1 - weight) * field + weight * np.array([field[r].mean() if r else field[i] for i, r in enumerate(rings)])
    return (field - field.mean()) / (field.std() or 1.0)


def moran(values: np.ndarray, cells: list[str], index: dict[str, int]) -> float:
    """Moran's I of ``values`` over the cells' first rings (binary weights)."""
    z = values - values.mean()
    pairs = [(i, index[n]) for i, c in enumerate(cells) for n in h3.grid_ring(c, 1) if n in index]
    return float(len(z) / len(pairs) * sum(z[i] * z[j] for i, j in pairs) / (z @ z))


def null_field(rng: np.random.Generator, folder: Path) -> tuple[Path, Path, float]:
    """One repeat's target and layer files and the target metric's Moran's I over its cells."""
    cells = [str(c) for c in h3.grid_disk(ORIGIN, 9)]
    index = {c: i for i, c in enumerate(cells)}
    weight = rng.uniform(0.0, 0.32)  # Moran's I of the cells about 0–0.7 (0.1 → 0.25, 0.2 → 0.47, 0.3 → 0.67)
    load, mix = smooth(cells, index, weight, rng), smooth(cells, index, weight, rng)
    parts = []
    for c in cells:
        n = rng.poisson(12) + 1
        lat, lon = h3.cell_to_latlng(c)
        share = 1 / (1 + np.exp(-mix[index[c]]))
        parts.append(pd.DataFrame({
            "lat": lat + rng.normal(0, 1e-4, n), "lon": lon + rng.normal(0, 1e-4, n),
            "load": 10 + 2 * load[index[c]] + rng.normal(0, 1, n), "kind": np.where(rng.random(n) < share, "a", "b"),
        }))  # fmt: skip
    target, layer = folder / "target.csv", folder / "layer.csv"
    pd.concat(parts, ignore_index=True).to_csv(target, index=False)
    spots = rng.choice(len(cells), 12, replace=False)
    pd.DataFrame({"lat": [h3.cell_to_latlng(cells[i])[0] for i in spots], "lon": [h3.cell_to_latlng(cells[i])[1] for i in spots]}).to_csv(
        layer, index=False
    )
    return target, layer, moran(load, cells, index)


def support_dropped(loaded: Any, config: MinerConfig, rng: np.random.Generator) -> float:
    """The share of the null's draws whose cohort — one band value at a time — falls below ``MIN_SUPPORT_ROWS``."""
    layer = next(iter(loaded.spatial.context.values()))
    values = [v for v in pd.unique(loaded.frame[layer.column]) if v != NONE_NEARBY]
    dropped = total = 0
    for _ in range(config.context_permutations):
        dealt = dealt_band(layer, deal(layer, rng))
        for v in values:
            total += 1
            dropped += int((dealt == v).sum() < config.min_support_rows)
    return dropped / total if total else 0.0


def repeat(seed: int, miner: MinerConfig, config: Any) -> dict[str, Any]:
    """One null repeat: its Moran's I, the admitted insights on the layer's band, the draws dropped for support."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        target, layer, morans_i = null_field(rng, Path(tmp))
        loaded = load_dataset(target, miner, geo=f"lat,lon@{RES}", context=[("layer", layer, "lat,lon")])
        result = run_discovery(loaded.frame, miner, spatial=loaded.spatial)
        found = build_insights(result, miner, dataset_id="ds", batch_id="b", filename="target.csv")
        band = next(iter(loaded.spatial.context.values())).column
        on_band = lambda kept: [i for i in kept if any(c.attribute == band and c.value != NONE_NEARBY for c in i.conditions)]  # noqa: E731
        hits, median_only = on_band(admitted(found, config)), on_band(admitted([without(i, "p_context") for i in found], config))
        dropped = round(support_dropped(loaded, miner, rng), 4)
        return {"seed": seed, "moran_i": round(morans_i, 3), "errors": len(hits), "median_only": len(median_only), "dropped": dropped}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The empirical FWER overall and by Moran's I band, the mean share of draws dropped for support."""

    def fwer(part: list[dict[str, Any]]) -> float | None:
        return round(sum(r["errors"] > 0 for r in part) / len(part), 3) if part else None

    bands = {f"{lo:.1f}–{hi:.1f}": [r for r in rows if lo <= r["moran_i"] < hi] for lo, hi in ((-1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0))}
    return {
        "reps": len(rows),
        "fwer": fwer(rows),
        "fwer_by_moran": {k: {"reps": len(v), "fwer": fwer(v)} for k, v in bands.items()},
        "dropped_mean": round(float(np.mean([r["dropped"] for r in rows])), 4) if rows else None,
        "fwer_median_only": round(sum(r["median_only"] > 0 for r in rows) / len(rows), 3) if rows else None,  # the median test alone
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--weights", action="store_true", help="WEIGHT_BY_EVENTS on: the null under event weights")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    config = settings()
    config = replace(config, miner=replace(config.miner, weight_by_events=args.weights))
    t0, rows = time.perf_counter(), []
    for seed in range(args.reps):
        rows.append(repeat(seed, config.miner, config))
        if (seed + 1) % 20 == 0:
            print(f"{seed + 1} repeats: FWER {summarize(rows)['fwer']}, {time.perf_counter() - t0:.0f} s", flush=True)
    result = summarize(rows) | {"weights": args.weights, "seconds": round(time.perf_counter() - t0, 1), "rows": rows}
    print({k: v for k, v in result.items() if k != "rows"})
    print(save_result("crosstype_null_calibration", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
