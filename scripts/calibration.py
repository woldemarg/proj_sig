"""The calibration sequence of docs/05_latent_anchors.md §5.10: the demo, then a same-domain batch (demo seed 8), then
housing.csv, on a scratch workspace under the current attractor settings. Per batch: the insights, the orphan rate, the
anchors before and after, the warnings; then every anchor with its patterns, scopes and datasets.

    python scripts/calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the packages live one level up

from scratch import build_engine, ingest_ready, scratch_dir, write_demo_csv  # noqa: E402

from insight_graph_service.core.demo import generate_retail_dataset  # noqa: E402
from insight_graph_service.core.settings import PROJECT_ROOT  # noqa: E402

HOUSING_BINS = "median_income:4,housing_median_age:4"  # housing.csv has one native categorical: the docs' bands (docs/02_discovery.md)


def main() -> None:
    root = scratch_dir("calibration", fresh=True)
    engine = build_engine(root / "ws")
    same_domain = root / "retail_synthetic_seed8.csv"
    generate_retail_dataset(5000, 8).to_csv(same_domain, index=False)
    for path in (write_demo_csv(root), same_domain, PROJECT_ROOT / "data" / "housing.csv"):
        before = set(engine.ontology().attractor_ids)
        if path.name == "housing.csv":
            with path.open("rb") as stream:
                record = engine.process(engine.upload(path.name, stream, bins=HOUSING_BINS)["batch_id"])
            if record["status"] != "READY":
                raise SystemExit(f"housing: {record['status']} {record.get('error')}")
        else:
            record = ingest_ready(engine, path)
        metrics = record["metrics"]
        after = engine.ontology().attractor_ids
        print(
            f"{path.name}: insights {metrics.get('activation_count')}, orphan rate {metrics.get('orphan_rate'):.2f}, "
            f"anchors {len(before)} -> {len(after)} (new {len(set(after) - before)}), warnings {record.get('warnings')}",
            flush=True,
        )
    for node in engine.graph().of_kind("Attractor"):
        props = node["props"]
        print(f"  {node['id']} {node['label']!r}: {props['n_patterns']} patterns, {props['distinct_scopes']} scopes, datasets {props['datasets']}")


if __name__ == "__main__":
    main()
