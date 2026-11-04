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
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scratch import build_engine, save_result, scratch_dir, upload_all

from insight_graph_service.core.demo import DEMO_DIR, demo_datasets
from insight_graph_service.core.engine import Engine

TARGET = "chicago_crimes"
LAYERS = ("chicago_bars", "chicago_l_stations")
SETTINGS = ((9, 1), (9, 0), (10, 0))  # (H3 level, disk radius), the console's default first
FREQUENCY = ("chicago_bars", "crime=robbery")  # the frequency of one class near the layer: the target under a filter, crossed


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
            for record in engine.cross(own[0]["upload_id"], [layer], ring=ring, levels=[level]):
                runs.append(summary(engine, engine.process(record["batch_id"]), name, level, ring, time.perf_counter() - t0))
    layer, where = frequency
    reading = upload_all(engine, target, **options | {"where": where})
    for ring in (1, 0):
        t0 = time.perf_counter()
        for record in engine.cross(reading[0]["upload_id"], [kept[layer]], ring=ring):  # the reading's own finest level
            runs.append(
                summary(engine, engine.process(record["batch_id"]), f"{layer} · {where}", int(record["level"]), ring, time.perf_counter() - t0)
            )
    return {"target": [{"level": r["level"], "where": r.get("where"), "status": r["status"]} for r in own + reading], "runs": runs}


def main() -> int:
    from tests.doubles import HashingEmbedder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    shipped = {d["key"]: d for d in demo_datasets()}
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
