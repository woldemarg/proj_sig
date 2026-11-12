"""The cost of cross-type analysis in a workspace of several uploads (the M18 specification,
docs/extra_funcs/spatial_analytics_multi_type/cross_dataset_analysis_architecture.md §8, §13): the shipped crimes of 2025
and 2024 and the crashes uploaded with their manifests' options, the bars and the CTA 'L' stations kept as layer-only
uploads, then four cross-type runs between them, in the order a user would make them — each batch's seconds, its
commit's (``persist_s``: the journal, the map's files and the snapshot) and the snapshot's size after it. On a scratch
workspace with the hashing embedder; the result in ``.scratch/results``.

    python scripts/crosstype_workspace.py
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from scratch import build_engine, save_result, scratch_dir

from insight_graph_service.core.demo import DEMO_DIR, demo_datasets
from insight_graph_service.core.engine import Engine

TARGETS = ("chicago_crimes", "chicago_crimes_2024", "chicago_crashes")
LAYERS = ("chicago_bars", "chicago_l_stations")
RUNS = (
    ("chicago_crimes", "chicago_bars"),
    ("chicago_crimes", "chicago_l_stations"),
    ("chicago_crimes_2024", "chicago_bars"),
    ("chicago_crashes", "chicago_bars"),
)


def timed(engine: Engine, step: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each record processed in turn: its seconds, its commit's and the snapshot's size after it."""
    rows = []
    for r in records:
        t0 = time.perf_counter()
        done = engine.process(r["batch_id"]) if r["status"] == "UPLOADED" else r
        timings = (done.get("metrics") or {}).get("timings") or {}
        rows.append(
            {
                "step": step,
                "level": done.get("level"),
                "status": done["status"],
                "seconds": round(time.perf_counter() - t0, 1),
                "persist_s": timings.get("persist_s"),
                "snapshot_kb": round(engine.ws.graph_path.stat().st_size / 1024) if engine.ws.graph_path.is_file() else 0,
            }
        )
    return rows


def main() -> int:
    from tests.doubles import HashingEmbedder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    shipped = {d["key"]: d for d in demo_datasets()}
    engine = build_engine(scratch_dir("crosstype_workspace", fresh=True) / "ws", embedder=HashingEmbedder())
    rows, ids = [], {}
    for key in (*TARGETS, *LAYERS):
        options = shipped[key]["options"] if key in TARGETS else {"geo": shipped[key]["options"]["geo"], "layer": True}
        with (DEMO_DIR / shipped[key]["file"]).open("rb") as stream:
            records = engine.upload(shipped[key]["file"], stream, **options)
        ids[key] = records[0]["upload_id"]
        rows += timed(engine, key, records)
    for target, layer in RUNS:
        rows += timed(engine, f"{target} × {layer}", engine.cross(ids[target], [ids[layer]]))
    for r in rows:
        print(f"{r['step']:<44} {str(r['level']):<6} {r['status']:<7} {r['seconds']:>6} s  persist {r['persist_s']}  snapshot {r['snapshot_kb']} KB")
    print(save_result("crosstype_workspace", args.label, {"rows": rows, "batches": len(engine.ws.list_batches())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
