"""The vertical zoom of a geo pattern over the snapshot (docs/07_question_answering.md §7.4).

From a pattern up to a coarser pattern of the same phenomenon (a shared non-weak anchor) in the region that holds it,
down to a finer one inside its region, and above the top pattern the regions alone (PARENT_OF). No new edge: the
chain reads ACTIVATES, IN_REGION and PARENT_OF of the snapshot.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from graph_query_engine.graph import DualGraph
from insight_contracts import dataset_node_id

VERTICAL_MAX_PER_PATTERN = 3  # steps up and steps down from one pattern
VERTICAL_MAX_TOTAL = 10  # vertical patterns over every chain of one question
VERTICAL_MIN_SHARE = 0.5  # the finer pattern's IN_REGION share of the region the two meet in


@dataclass
class VerticalStep:
    pattern_id: str  # the pattern reached
    origin: str  # the pattern it was reached from
    direction: str  # up | down
    relation: str  # corresponds (the shared metric shifts the same way) | contrasts
    metric: str
    region: str
    attractor: str
    score: float  # alignment(origin) · alignment(pattern) · the finer pattern's share


@dataclass
class ScaleChain:
    pattern_id: str
    up: list[VerticalStep] = field(default_factory=list)
    down: list[VerticalStep] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)  # above the top pattern, finest first
    scale: str = ""  # regional | scale-dependent | not_found_coarser | insufficient; "" outside an upload


def regions_above(graph: DualGraph, pattern_ids: list[str]) -> set[str]:
    """The regions coarser than each pattern's finest region level that it lies in, with their ancestors that hold an
    intact pattern: what the console draws (a region at that level, the pattern's own or the next coarser, is one of its
    cells or a handful of them, hundreds per dataset; a region holding tampered patterns only holds nothing). A base
    pattern (no resolution) draws none: its regions, placed by its events, do not nest by PARENT_OF."""
    geo = [pid for pid in pattern_ids if graph.facts(pid).resolution is not None]
    found = {r for pid in geo for r in graph.facts(pid).regions.keys() - graph.facts(pid).finest}
    frontier = set(found)
    while frontier:
        frontier = {e["target"] for r in frontier for e in graph.out.get(r, []) if e["type"] == "PARENT_OF"} - found
        found |= frontier
    return _held(graph, found)


def shown_regions(graph: DualGraph, pattern_ids: list[str]) -> set[str]:
    """The regions the console draws for ``pattern_ids`` (the map's overlay, the graph's regions): ``regions_above`` and
    every region an intact pattern's scale chain names (the regions coarser than its top pattern, walked up from the
    region its last step up met in, or from its home region when it has none: level 7 for cells at 8; never one at the
    top pattern's own level, which is one of its cells), so the map holds each region a SCALE CHAIN names; a region
    holding tampered patterns only is none."""
    intact = [pid for pid in pattern_ids if not graph.tampered(pid)]
    return regions_above(graph, pattern_ids) | _held(graph, {r for pid in intact for chain in scale_chain(graph, [pid]) for r in chain.regions})


def _held(graph: DualGraph, regions: set[str]) -> set[str]:
    """The regions that hold an intact pattern: an ancestor no intact pattern reaches, or a region of tampered patterns
    only, holds nothing."""
    tampered = Counter(e["target"] for pid in graph.insights if graph.tampered(pid) for e in graph.out.get(pid, []) if e["type"] == "IN_REGION")
    return {r for r in regions if graph.nodes[r]["props"]["patterns"] > tampered[r]}


def _home(graph: DualGraph, pid: str) -> str | None:
    """One finest region of the pattern, the largest share first (where the regions above a chain start)."""
    facts = graph.facts(pid)
    return min(facts.finest, key=lambda r: (-facts.regions[r], r), default=None)


def _relation(graph: DualGraph, a: str, b: str) -> tuple[str, str] | None:
    """(corresponds | contrasts, metric) on the first of the two targets both patterns shift."""
    ia, ib = graph.insight(a), graph.insight(b)
    za, zb = {s.metric: s.robust_z for s in ia.shifts}, {s.metric: s.robust_z for s in ib.shifts}
    metric = next((m for m in (ia.target, ib.target) if m in za and m in zb), None)
    if metric is None:
        return None
    return ("corresponds" if za[metric] * zb[metric] > 0 else "contrasts"), metric


def _affinity(graph: DualGraph, a: str, b: str) -> float:
    """The Jaccard overlap of two patterns' scope conditions: 1 for the same subgroup at another level."""
    ca, cb = set(graph.insight(a).conditions), set(graph.insight(b).conditions)
    return len(ca & cb) / (len(ca | cb) or 1)


def _step(graph: DualGraph, pid: str, direction: str, seen: set[str]) -> VerticalStep | None:
    """The coarser (up) or finer (down) pattern of the same phenomenon in a region the two share: one of the same upload
    first, then the nearest level (no level skipped), then the closest scope (the pattern's own subgroup before another
    pattern of the region), then the score."""
    facts = graph.facts(pid)
    res, anchors, upload, own = facts.resolution, facts.anchors, facts.upload, facts.finest
    kin = {e["source"] for a in anchors for e in graph.inc.get(a, []) if e["type"] == "ACTIVATES" and not e["props"].get("weak")}
    best: tuple[tuple[bool, int, float, float], VerticalStep] | None = None
    for region, share in graph.facts(pid).regions.items():
        for e in graph.inc.get(region, []):
            other = e["source"]  # only a pattern of the same phenomenon (a shared non-weak anchor) can be a step
            if e["type"] != "IN_REGION" or other in seen or other not in kin or graph.tampered(other):
                continue
            if (other_res := graph.facts(other).resolution) is None:
                continue
            coarser, finer_share = (other_res < res, share) if direction == "up" else (other_res > res, e["weight"])
            if not coarser or finer_share < VERTICAL_MIN_SHARE or region not in (graph.facts(other).finest if direction == "up" else own):
                continue
            shared = [(anchors[a] * w * finer_share, a) for a, w in graph.facts(other).anchors.items() if a in anchors]
            relation = _relation(graph, pid, other)
            if not shared or relation is None:
                continue
            score, anchor = max(shared)
            step = VerticalStep(other, pid, direction, relation[0], relation[1], region, anchor, round(score, 4))
            key = (upload is not None and graph.facts(other).upload == upload, -abs(other_res - res), _affinity(graph, pid, other), score)
            if best is None or key > best[0]:
                best = (key, step)
    return best[1] if best else None


def scale_chain(graph: DualGraph, pattern_ids: list[str]) -> list[ScaleChain]:
    """The vertical chain of every geo pattern in ``pattern_ids``: up to ``VERTICAL_MAX_PER_PATTERN`` steps each way,
    at most ``VERTICAL_MAX_TOTAL`` distinct vertical patterns in all (one already in a chain costs nothing again), then
    the regions above the top pattern (from the region the last step met in); the ``scale`` label compares the pattern
    with its first step up, budget or not."""
    chains, reached = [], set()
    for pid in pattern_ids:
        if graph.facts(pid).resolution is None or not graph.facts(pid).regions:
            continue
        chain, seen = ScaleChain(pid), {pid}
        nearest = _step(graph, pid, "up", seen)
        for direction, steps in (("up", chain.up), ("down", chain.down)):
            step = nearest if direction == "up" else _step(graph, pid, "down", seen)
            while step and (step.pattern_id in reached or len(reached) < VERTICAL_MAX_TOTAL):
                steps.append(step)
                seen.add(step.pattern_id)
                reached.add(step.pattern_id)
                step = _step(graph, step.pattern_id, direction, seen) if len(steps) < VERTICAL_MAX_PER_PATTERN else None
        top = chain.up[-1].pattern_id if chain.up else pid
        region, top_res = (chain.up[-1].region if chain.up else _home(graph, pid)), graph.facts(top).resolution
        while region:
            if graph.nodes[region]["props"]["resolution"] < top_res:
                chain.regions.append(region)
            region = next((e["target"] for e in graph.out.get(region, []) if e["type"] == "PARENT_OF"), None)
        chain.scale = _scale(graph, pid, nearest)
        chains.append(chain)
    return chains


def _scale(graph: DualGraph, pid: str, nearest: VerticalStep | None) -> str:
    """The pattern against its nearest coarser level of the same upload: ``nearest`` is its first step up, which ranks
    the same upload and the nearest level first, so a reversal at the next level is never hidden by a farther one."""
    upload = graph.facts(pid).upload
    if upload is None:
        return ""
    if nearest and graph.facts(nearest.pattern_id).upload == upload:
        return "regional" if nearest.relation == "corresponds" else "scale-dependent"
    planned = graph.nodes[dataset_node_id(graph.insight(pid).dataset_id)]["props"].get("upload_levels", [])
    coarser = [lv for lv in planned if lv.isdigit() and int(lv) < graph.facts(pid).resolution]  # mined, even if it found nothing
    return "not_found_coarser" if coarser else "insufficient"


def scale_label(graph: DualGraph, pid: str) -> str:
    """The ``scale`` that ``scale_chain`` gives one pattern, from its first step up alone (the map labels every pattern)."""
    if graph.facts(pid).resolution is None or not graph.facts(pid).regions:
        return ""
    return _scale(graph, pid, _step(graph, pid, "up", {pid}))


def picture(graph: DualGraph, region: str) -> dict[str, int]:
    """What a region holds: its intact IN_REGION patterns over every dataset (a tampered one counts for nothing), and how
    many datasets and anchors (themes) they span."""
    members = [e["source"] for e in graph.inc.get(region, []) if e["type"] == "IN_REGION" and not graph.tampered(e["source"])]
    return {
        "patterns": len(members),
        "datasets": len({graph.insight(p).dataset_id for p in members}),
        "themes": len({a for p in members for a in graph.facts(p).anchors}),
    }


ZONES_MAX = 5  # zones in the evidence


def rank_zones(graph: DualGraph, geo_ids: list[str], answer_ids: list[str]) -> list[dict[str, Any]]:
    """The ZONES of a question (docs/07_question_answering.md §7.4). A region holds a pattern with an IN_REGION share of at
    least ``VERTICAL_MIN_SHARE``. The level is the coarsest where the evidence's geo patterns lie in three or more regions
    and one holds an answering pattern (else the finest that holds one); its regions holding answering patterns rank by
    their sum of insight weight × share; at most ``ZONES_MAX``, each with its ``picture``. Which patterns answer is the
    caller's to say."""
    held: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    score: dict[str, float] = defaultdict(float)
    answers = set(answer_ids)
    for pid in geo_ids:
        for region, share in graph.facts(pid).regions.items():
            if share >= VERTICAL_MIN_SHARE:
                held[graph.nodes[region]["props"]["resolution"]][region].append(pid)
                if pid in answers:
                    score[region] += graph.insight(pid).weight * share
    answered = [lv for lv in sorted(held) if any(r in score for r in held[lv])]
    if not answered:
        return []
    level = next((lv for lv in answered if len(held[lv]) >= 3), answered[-1])
    ranked = sorted((r for r in held[level] if r in score), key=lambda r: (-score[r], r))[:ZONES_MAX]
    return [
        {"id": r, "resolution": level, "score": round(score[r], 4), "holds": [p for p in held[level][r] if p in answers], **picture(graph, r)}
        for r in ranked
    ]
