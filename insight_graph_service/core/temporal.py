"""The time windows of a workspace's uploads (docs/02_discovery.md §2.10, docs/06_graph_and_storage.md §6.2): the
lifecycle of a window batch the engine runs — its cut at upload, its known rules before discovery, what it observed
after admission, whether its failure moves its series — and the temporal plane derived from the batch records and the
journal, which ``snapshot`` turns into EVOLVES_TO edges and the ``trajectories`` section, as the spatial plane.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from insight_contracts import Insight, Rejection
from insight_graph_service.core.snapshot import compute_metadata_integrity
from insight_graph_service.core.uploads import EMPTY_LEVEL
from insight_graph_service.core.workspace import TERMINAL
from subgroup_miner.config import MinerConfig
from subgroup_miner.discovery import DiscoveryResult, KnownRule
from subgroup_miner.temporal import (
    NOT_EVALUATED,
    WINDOW_UNAVAILABLE,
    Observation,
    Trajectory,
    WindowFacts,
    fold_trajectories,
    observations,
    open_rules,
    series_id,
    stamp_insights,
    upload_windows,
)


@dataclass
class TemporalStructure:
    """The derived temporal plane: every window series' trajectories, folded from its windows' records and the journal
    (``snapshot._temporal_plane`` makes them EVOLVES_TO edges and the ``trajectories`` section)."""

    series: dict[str, dict[str, Trajectory]] = field(default_factory=dict)  # series -> trajectory key -> trajectory


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


def derive_temporal(batches: Iterable[Mapping[str, Any]], journal: Iterable[Mapping[str, Any]], config: MinerConfig) -> TemporalStructure:
    """Every window series' trajectories, from its windows' batch records and the journal's nodes."""
    series: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for b in batches:
        if (s := series_of(b)) is not None:
            series[s].append(b)
    if not series:
        return TemporalStructure()
    nodes = _nodes(journal, {b["dataset_id"] for windows in series.values() for b in windows if b.get("dataset_id")})
    return TemporalStructure({name: _fold(windows, nodes, config) for name, windows in series.items()})


def window_cuts(
    path: Path, config: MinerConfig, *, time: str | None, window: str | None, max_windows: int, bins: str | None, categories: str | None
) -> list[dict[str, Any]]:
    """What each batch of an upload adds to its record: one ``{window, temporal}`` per time window of the file
    (``upload_windows``, its reference scale on each, no observation yet), or one empty cut for an upload without time."""
    if not time:
        return [{}]
    cuts, scale = upload_windows(path, config, time=time, window=window, max_windows=max_windows, bins=bins, categories=categories)
    return [{"window": w, "temporal": {"reference_scale": scale, "observations": None}} for w in cuts]


def record_window(
    record: dict[str, Any], result: DiscoveryResult, insights: list[Insight], admitted: list[Insight], rejections: list[Rejection]
) -> None:
    """After admission, on a time window's batch: each insight's ``provenance.temporal`` (``stamp_insights``) and, on the
    record, what the window observed of each known rule — written before a window that admits nothing fails."""
    if (series := series_of(record)) is None:
        return
    stamp_insights(insights, result, series=series, window=record["window"]["index"])
    record["temporal"]["observations"] = {k: asdict(o) for k, o in observations(result, insights, admitted, rejections).items()}


def moves_series(record: Mapping[str, Any], code: str, batches: Iterable[Mapping[str, Any]]) -> bool:
    """Whether a failed batch changes the temporal plane though it commits nothing: a time window with nothing to mine
    whose series has a READY window (a node, so maybe a trajectory its refusal or its missing data moves)."""
    series = series_of(record)
    return series is not None and code in EMPTY_LEVEL and any(b["status"] == "READY" and series_of(b) == series for b in batches)
