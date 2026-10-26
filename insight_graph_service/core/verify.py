"""An insight recomputed from its stored upload (docs/06_graph_and_storage.md §6.7): the same code over the same bytes,
compared with its journal record. Reads only; the engine guards the call (a degraded workspace verifies nothing)."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from insight_contracts import Condition, pattern_id
from insight_graph_service.core.errors import PipelineError
from insight_graph_service.core.workspace import Workspace
from subgroup_miner.discovery import NOT_RECOMPUTED, recompute_statistics
from subgroup_miner.ingestion import IngestionError, LoadedDataset


def verify_diff(record: dict[str, Any], fresh: dict[str, Any], covers: np.ndarray | None, family: int) -> list[dict[str, Any]]:
    """What of the journal record the recomputation (``recompute_statistics``) does not reproduce, each as ``{field,
    journal, recomputed}``: its id and selector against its conditions, the rows against ``covers.npz`` and the record's
    row hash, its target as its first shift (on a covariance insight, the stronger of its pair), a missing p only on a
    covariance insight, its test family against its dataset's (a known rule's: the window's metrics), then every number; floats agree within a relative 1e-9 (the same code over the same bytes)."""
    rows = fresh["rows"]
    out = []

    def differs(field: str, journal: Any, recomputed: Any) -> None:
        out.append({"field": field, "journal": journal, "recomputed": recomputed})

    if record["id"] != pattern_id(record["dataset_id"], tuple(Condition(c["attribute"], c["value"]) for c in record["conditions"])):
        differs("id", record["id"], "another id for these conditions")
    if fresh["expression_rows"] is None or not np.array_equal(fresh["expression_rows"], rows):
        differs("expression", record["expression"], "selects other rows than the conditions")
    if covers is not None and not np.array_equal(np.sort(covers), rows):
        differs("rows", f"{len(covers)} rows in covers.npz", f"{len(rows)} rows by the conditions")
    if record.get("row_hash") != hashlib.sha1(rows.astype(np.int64).tobytes()).hexdigest()[:16]:
        differs("row_hash", record.get("row_hash"), hashlib.sha1(rows.astype(np.int64).tobytes()).hexdigest()[:16])
    if record["provenance"].get("multiple_testing_family") != family:
        differs("multiple_testing_family", record["provenance"].get("multiple_testing_family"), f"{family} for its dataset")
    covariance = record.get("phenomenon_type") == "covariance" and bool((record.get("covariance") or {}).get("pair"))
    if record["p_value"] is None and not covariance:
        differs("p_value", None, fresh.get("p_value"))
    if "shifts" not in fresh:
        return out + [{"field": "support", "journal": record["support"], "recomputed": 0}]
    primary = next((s for s in fresh["shifts"] if s["metric"] == record["target"]), None)
    if primary is None:
        differs("target", record["target"], "not among its shifts")
        primary = fresh["shifts"][0]
    elif not covariance and record["shifts"][0]["metric"] != record["target"]:  # its p and spatial test are of its first shift
        differs("target", record["target"], f"not its first shift ({record['shifts'][0]['metric']}, whose test its p is)")
    pair = [s for s in fresh["shifts"] if s["metric"] in (record.get("covariance") or {}).get("pair", [])]
    if covariance and pair and record["target"] != (chosen := min(pair, key=lambda s: (-abs(s["robust_z"]), s["metric"]))["metric"]):
        differs("target", record["target"], f"{chosen} (the stronger metric of its pair)")
    pairs = [
        ("support", record["support"], fresh["support"]),
        ("support_fraction", record["support_fraction"], fresh["support_fraction"]),
        ("baseline", record["baseline"], primary["global_median"]),
        ("local", record["local"], primary["local_median"]),
        ("effect_size", record["effect_size"], primary["robust_z"]),
    ]
    if record["p_value"] is not None:
        pairs += [("p_value", record["p_value"], fresh["p_value"]), ("p_adjusted", record["p_adjusted"], fresh["p_adjusted"])]
    for stored, again in zip(record["shifts"], fresh["shifts"], strict=True):
        pairs += [(f"shifts.{stored['metric']}.{k}", stored[k], again[k]) for k in ("robust_z", "local_median", "global_median", "global_mad")]
    spatial = record["provenance"].get("spatial") or {}
    pairs += [(f"spatial.{k}", spatial[k], v) for k, v in fresh.get("spatial", {}).items() if k in spatial]
    for name, a, b in pairs:
        if not (a == b or (isinstance(a, int | float) and isinstance(b, int | float) and math.isclose(a, b, rel_tol=1e-9))):
            differs(name, a, b)
    return out


def verify_pattern(ws: Workspace, pattern_id: str, load: Callable[[dict[str, Any]], LoadedDataset]) -> dict[str, Any]:
    """``pattern_id`` recomputed from its stored upload (docs/06_graph_and_storage.md §6.7): ``reproduced`` when its rows
    and every recomputed number match the journal, ``differs`` with what does not, ``source_changed`` when the file
    read with its options (under the current miner settings) no longer gives the dataset id, ``source_missing`` when
    it is gone. Reads only, without the writer lock: a file removed meanwhile is ``source_missing``."""
    records = ws.patterns()
    record = next((r for r in records if r["id"] == pattern_id), None)
    if record is None:
        raise PipelineError("unknown_pattern", pattern_id)
    ds = record["dataset_id"]
    batch = ws.ready_batch_for(ds)
    out = {"pattern_id": pattern_id, "dataset_id": ds, "diff": [], "not_checked": list(NOT_RECOMPUTED)}
    try:
        loaded = load(batch) if batch else None
    except IngestionError as exc:  # unreadable now, or refused by the current settings
        return out | {"status": "source_missing" if exc.code == "unsupported_file" else "source_changed", "detail": str(exc)}
    if loaded is None:
        return out | {"status": "source_missing"}
    if loaded.dataset_id != ds:
        return out | {"status": "source_changed", "detail": f"the file read with its options now gives {loaded.dataset_id}"}
    if (record["provenance"].get("temporal") or {}).get("origin") == "tracked":  # a known rule's family: the window's metrics (§2.10)
        family = len(batch["profile"]["numerics"])
    else:  # the search's, shared by its dataset's other cohorts
        searched = (r["provenance"] for r in records if r["dataset_id"] == ds and (r["provenance"].get("temporal") or {}).get("origin") != "tracked")
        family = Counter(p.get("multiple_testing_family") for p in searched).most_common(1)[0][0]
    fresh = recompute_statistics(
        loaded.frame,
        [(c["attribute"], c["value"]) for c in record["conditions"]],
        record["expression"],
        [s["metric"] for s in record["shifts"]],
        loaded.spatial,
        family,
    )
    try:
        covers = ws.load_covers(ds).get(pattern_id)
    except (OSError, ValueError):  # rewritten meanwhile: the conditions decide the rows
        covers = None
    if covers is None:
        out["not_checked"].append("its rows in covers.npz (none stored)")
    diff = verify_diff(record, fresh, covers, family)
    return out | {"status": "differs" if diff else "reproduced", "diff": diff}
