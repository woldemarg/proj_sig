"""The time windows of a workspace's uploads as the miner's trajectory fold reads them (docs/02_discovery.md §2.10,
docs/06_graph_and_storage.md §6.2): which batch records are the windows of which series, what each window observed,
and the trajectories and EVOLVES_TO edges they make. A function of the batch records and the journal records it is
given — the engine asks it for a window's known rules, and for the temporal plane it hands the snapshot compiler.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from insight_contracts import EdgeType, GraphEdge, Insight
from insight_graph_service.core.snapshot import compute_metadata_integrity
from insight_graph_service.core.uploads import EMPTY_LEVEL
from insight_graph_service.core.workspace import TERMINAL
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import KnownRule
from subgroup_miner.temporal import NOT_EVALUATED, WINDOW_UNAVAILABLE, Observation, Trajectory, WindowFacts, fold_trajectories, open_rules, series_id


def series_of(record: Mapping[str, Any]) -> str | None:
    """The window series a batch record is a window of (``temporal.series_id``); None for any other batch."""
    return series_id(record["upload_id"], record["window"]) if record.get("window") else None


def _facts(record: Mapping[str, Any], nodes: list[Insight]) -> WindowFacts | None:
    """What one window's record and its journal nodes tell the fold; None when the window observed nothing a trajectory
    may read: it failed before it was mined, or for another reason than having nothing to mine (a rolled-back commit),
    or its record is not verified by its digest. An admitted observation whose node is not in the journal reads as not
    evaluated."""
    observations = record["temporal"]["observations"]  # None until the window is mined
    mined = record["status"] == "READY" or (record.get("error") or {}).get("code") in EMPTY_LEVEL
    if observations is None or not mined or compute_metadata_integrity(record) != "ok":
        return None
    ids = {n.id for n in nodes}
    lost = Observation(NOT_EVALUATED, reason=WINDOW_UNAVAILABLE)
    return WindowFacts({k: lost if v["active"] and v["pattern_id"] not in ids else Observation(**v) for k, v in observations.items()}, nodes)


def _fold(
    records: list[Mapping[str, Any]], nodes: Mapping[str, list[Insight]], config: MinerConfig, before: int | None = None
) -> dict[str, Trajectory]:
    """One series' trajectories from its windows' records and the journal's nodes by dataset id: over the windows before
    ``before``, or up to the first window none of whose records has ended. A SKIPPED record is left out: the window it
    duplicates is read in its own series (this one for the same file again; for the same table saved as another file,
    that file's). A window uploaded twice reads its READY record first."""
    by_index: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r["status"] != "SKIPPED":
            by_index[r["window"]["index"]].append(r)
    if not by_index:
        return {}
    any_record = next(iter(by_index.values()))[0]
    count = any_record["window"]["of"] if before is None else before
    upto = next((i for i in range(count) if by_index.get(i) and all(r["status"] not in TERMINAL for r in by_index[i])), count)
    windows = {}
    for index in range(upto):
        found = (_facts(r, nodes.get(r.get("dataset_id"), [])) for r in sorted(by_index.get(index, []), key=lambda r: r["status"] != "READY"))
        if (facts := next((f for f in found if f is not None), None)) is not None:
            windows[index] = facts
    return fold_trajectories(windows, upto, any_record["temporal"]["reference_scale"], config)


def _nodes(journal: Iterable[Mapping[str, Any]], datasets: set[str]) -> dict[str, list[Insight]]:
    """The admitted insights of ``datasets`` with their ``provenance.temporal``, by dataset id."""
    out: dict[str, list[Insight]] = defaultdict(list)
    for rec in journal:
        if rec["dataset_id"] in datasets and "temporal" in rec.get("provenance", {}):
            out[rec["dataset_id"]].append(Insight.from_record(rec))
    return out


def known_rules(
    batches: Iterable[Mapping[str, Any]], journal: Iterable[Mapping[str, Any]], record: Mapping[str, Any], config: MinerConfig
) -> list[KnownRule]:
    """The rules a window evaluates directly: those of its series' trajectories still open after its earlier windows."""
    series = series_of(record)
    if series is None:
        return []
    windows = [b for b in batches if series_of(b) == series]
    nodes = _nodes(journal, {b["dataset_id"] for b in windows if b.get("dataset_id")})
    return open_rules(_fold(windows, nodes, config, before=record["window"]["index"]))


def temporal_plane(
    batches: Iterable[Mapping[str, Any]], journal: Iterable[Mapping[str, Any]], config: MinerConfig
) -> tuple[list[GraphEdge], dict[str, Any]]:
    """Every series' EVOLVES_TO edges and trajectories: the snapshot's ``trajectories``, key -> its series, rule, metrics,
    state (``closed_reason`` when closed), episode, first window and one entry per window it was observed in."""
    series: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for b in batches:
        if (s := series_of(b)) is not None:
            series[s].append(b)
    if not series:
        return [], {}
    nodes = _nodes(journal, {b["dataset_id"] for windows in series.values() for b in windows if b.get("dataset_id")})
    edges, trajectories = [], {}
    for name, windows in sorted(series.items()):
        for key, t in _fold(windows, nodes, config).items():
            edges += [GraphEdge(a, b, EdgeType.EVOLVES_TO, 1.0, props) for a, b, props in t.edges]
            trajectories[key] = {
                "series": name,
                "predicate": [asdict(c) for c in t.predicate],
                "metrics": list(t.metrics),
                "state": t.state,
                "closed_reason": t.closed_reason,
                "episode": t.episode,
                "first_window": t.first_window,
                "windows": t.windows,
            }
    return edges, trajectories
