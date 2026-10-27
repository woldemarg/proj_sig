"""Dual-layer graph assembly: the snapshot compiler (docs/06_graph_and_storage.md); the read model is
``graph_query_engine.graph.DualGraph``.

The snapshot is *derived* data: ``build_snapshot`` turns the journal records, activations and
vectors, the batch records (a READY one with its dataset's profile) and the ontology state into nodes and edges (the
co-memberships and the CO_OCCURS links between anchors are compiled here), so
it can always be regenerated (the engine rebuilds a snapshot of another version when it opens the workspace). It reads no files: the caller gathers the
inputs. ACTIVATES alignments are recomputed against the *current* centroids so graph weights
match the living ontology.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from insight_contracts import (
    SNAPSHOT_VERSION,
    EdgeType,
    GraphEdge,
    GraphNode,
    Insight,
    PhenomenonThresholds,
    attractor_id_of,
    attractor_node_id,
    batch_node_id,
    dataset_node_id,
    dimension_node_id,
    metric_node_id,
    record_digest,
    temporal_node_id,
)
from insight_contracts.text import component_label, headline, humanize
from insight_graph_service.core.settings import Settings
from insight_graph_service.core.uploads import declared_glossary
from insight_graph_service.core.workspace import utc_now
from subgroup_miner.lattice import structural_edges
from subgroup_miner.spatial import GEO_REGION_MIN_SHARE, SpatialStructure
from subgroup_miner.temporal import dynamics

if TYPE_CHECKING:  # the ontology is passed in, never constructed here
    from attractor_topology.ontology import LatentOntology

Member = tuple[dict[str, Any], float, float]  # (pattern record, alignment, strength)


def attractor_signature(members: list[tuple[dict[str, Any], float]]) -> list[tuple[str, float]]:
    """Aggregate signed phenomenon components of member patterns, weighted by strength."""
    agg: dict[str, float] = defaultdict(float)
    for record, strength in members:
        comps = record["canonical"]["components"]
        scale = max((abs(c) for _, c in comps), default=1.0) or 1.0
        for label, coef in comps:
            agg[label] += strength * coef / scale
    total = sum(s for _, s in members) or 1.0
    return sorted(((k, v / total) for k, v in agg.items()), key=lambda kv: -abs(kv[1]))


def _schema_plane(
    batches: dict[str, dict[str, Any]], insights: list[Insight], settings: Settings, integrity: dict[str, str]
) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    """Dataset / Batch / Dimension / Metric nodes of READY batches, plus the edges into them; ``integrity``: batch id ->
    its metadata's check."""
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    scope_attributes: dict[str, set[str]] = defaultdict(set)  # closed intents may use non-selected columns
    for ins in insights:
        scope_attributes[ins.dataset_id].update(c.attribute for c in ins.conditions)
    levels: dict[str, set[str]] = defaultdict(set)  # an upload's levels: those its batch records stand for, of any status
    for b in batches.values():
        if b.get("upload_id"):
            levels[b["upload_id"]].add(b["level"])
    for bid, b in batches.items():
        if b.get("status") != "READY":
            continue
        ds = b["dataset_id"]
        profile = b.get("profile") or {}
        nodes[dataset_node_id(ds)] = GraphNode(
            dataset_node_id(ds),
            "Dataset",
            b.get("filename", ds),
            {
                "dataset_id": ds,
                "filename": b.get("filename"),
                "rows": profile.get("rows"),
                "columns": profile.get("columns"),
                **({"source": b["source"]} if b.get("source") else {}),
                "reliability": b.get("reliability", "F"),
                "integrity": integrity[bid],
                **{k: b[k] for k in ("upload_id", "level", "supersedes") if k in b},
                **({"upload_levels": sorted(levels[b["upload_id"]], key=lambda lv: -int(lv) if lv.isdigit() else 1)} if "upload_id" in b else {}),
                **({"glossary": profile["geo"]["glossary"]} if "glossary" in profile.get("geo", {}) else {}),
                **({"user_glossary": declared} if (declared := declared_glossary(b)) else {}),
                **({"glossary_columns": n} if (n := len({k.split("=", 1)[0] for k in b.get("user_glossary") or {}})) else {}),
                **({"metadata_doc": b["metadata_doc"]} if b.get("metadata_doc") else {}),
            },
        )
        nodes[batch_node_id(bid)] = GraphNode(
            batch_node_id(bid),
            "Batch",
            bid,
            {"batch_id": bid, "batch_seq": b.get("batch_seq"), "created_at": b.get("created_at"), "status": b.get("status")},
        )
        edges.append(GraphEdge(batch_node_id(bid), dataset_node_id(ds), EdgeType.OF_DATASET))
        selected = profile.get("selected_dimensions", [])
        for name in [*selected, *sorted(scope_attributes[ds] - set(selected))]:
            nid = dimension_node_id(ds, name)
            nodes[nid] = GraphNode(
                nid,
                "Dimension",
                name,
                {
                    "dataset_id": ds,
                    "name": name,
                    "cardinality": profile.get("dimension_cardinality", {}).get(name),
                    "entropy": profile.get("dimension_entropy", {}).get(name),
                },
            )
        for name in profile.get("numerics", []):
            nid = metric_node_id(ds, name)
            nodes[nid] = GraphNode(
                nid,
                "Metric",
                humanize(name),
                {
                    "dataset_id": ds,
                    "name": name,
                    "global_median": profile.get("global_medians", {}).get(name),
                    "global_mad": profile.get("global_mads", {}).get(name),
                    **b.get("indicators", {}).get(name, {}),
                },
            )

    for ins in insights:
        if batch_node_id(ins.batch_id) in nodes:
            edges.append(GraphEdge(ins.id, batch_node_id(ins.batch_id), EdgeType.DISCOVERED_IN))
        for c in ins.conditions:
            nid = dimension_node_id(ins.dataset_id, c.attribute)
            if nid in nodes:
                edges.append(GraphEdge(ins.id, nid, EdgeType.HAS_SCOPE, 1.0, {"value": c.value}))
        for s in ins.shifts:
            nid = metric_node_id(ins.dataset_id, s.metric)
            if nid in nodes and (s.metric == ins.target or settings.thresholds.material(s)):
                role = "primary" if s.metric == ins.target else "secondary"
                edges.append(
                    GraphEdge(
                        ins.id,
                        nid,
                        EdgeType.TARGETS,
                        float(min(1.0, s.magnitude / 3.0)),
                        {"role": role, "z": s.robust_z, "local_median": s.local_median, "global_median": s.global_median},
                    )
                )
    return nodes, edges


def _activation_edges(
    activations: list[dict[str, Any]], vectors: np.ndarray, ontology: LatentOntology, by_pid: dict[str, dict[str, Any]], settings: Settings
) -> tuple[list[GraphEdge], dict[int, list[Member]]]:
    """ACTIVATES edges with alignments recomputed against the current centroids, grouped by attractor."""
    live = set(ontology.attractor_ids)
    edges: list[GraphEdge] = []
    members: dict[int, list[Member]] = defaultdict(list)
    for act in activations:
        pid, aid = act["pattern_id"], int(act["attractor_id"])
        if pid not in by_pid or aid not in live:
            continue
        rec = by_pid[pid]
        alignment = float(vectors[rec["row_id"]].astype(np.float64) @ ontology.centroid(aid).astype(np.float64))
        w = float(rec["weight"])
        # one predicate for coverage and retrieval: below the ontology's alignment floor (rerouted at
        # ingest, or drifted below it since) a membership is coverage only and is not walked
        weak = bool(act["weak"]) or alignment < settings.topology.min_activation_alignment
        members[aid].append((rec, alignment, alignment * w))
        edges.append(
            GraphEdge(
                pid,
                attractor_node_id(aid),
                EdgeType.ACTIVATES,
                alignment,
                {
                    "alignment": alignment,
                    "strength": alignment * w,
                    "insight_weight": w,
                    "engine_weight": act["engine_weight"],
                    "alignment_at_ingest": act["alignment"],
                    "source": act["source"],
                    "batch_id": act["batch_id"],
                    "weak": weak,
                },
            )
        )
    return edges, members


def _co_memberships(
    activations: list[dict[str, Any]], by_pid: dict[str, dict[str, Any]], vectors: np.ndarray, ontology: LatentOntology, settings: Settings
) -> list[dict[str, Any]]:
    """The memberships the assignment rule gives each insight against the current centroids — its top ``TOP_K_ASSIGN``
    anchors, each within ``MIXTURE_RATIO`` of its best alignment and above the floor — that its journal activations do not
    already name, in the journal's activation shape (``source: co_membership``; compiled, never journaled). Extraction keeps
    one atom per insight, so an insight's second theme comes from here (docs/05_latent_anchors.md §5.7)."""
    topo, ids = settings.topology, sorted(ontology.attractor_ids)
    if not ids:
        return []
    have = {(a["pattern_id"], int(a["attractor_id"])) for a in activations}
    recs = list(by_pid.values())
    centroids = np.stack([ontology.centroid(a) for a in ids]).astype(np.float64)
    cos = vectors[[r["row_id"] for r in recs]].astype(np.float64) @ centroids.T
    return [
        {
            "pattern_id": rec["id"],
            "attractor_id": ids[j],
            "engine_weight": None,
            "alignment": None,
            "source": "co_membership",
            "batch_id": rec["batch_id"],
            "weak": False,
        }
        for rec, row in zip(recs, cos)
        for j in np.argsort(-row)[: topo.top_k_assign]
        if row[j] >= max(topo.min_activation_alignment, topo.mixture_ratio * row.max()) and (rec["id"], ids[j]) not in have
    ]


def co_occurrence_edges(activates: list[GraphEdge]) -> list[GraphEdge]:
    """CO_OCCURS links between anchors that share non-weak members (docs/05_latent_anchors.md §5.7). With ``a`` the
    current alignments of insight p's non-weak memberships, ``W[p, j] = a_j · a_j / Σ_k a_k``: lac v3's combined-graph
    weight with the assignment rule's membership weight (the cosine), one scale for journal and compiled memberships alike.
    The pair weight is ``min(1, Σ_p W[p, j]·W[p, k])`` — the cap keeps traversal factors ≤ 1 — and ``shared`` counts the insights."""
    rows: dict[str, dict[int, float]] = defaultdict(dict)
    for e in activates:
        if not e.props["weak"] and e.weight > 0:
            rows[e.source][attractor_id_of(e.target)] = e.weight
    pairs: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows.values():
        total = sum(row.values())
        for j, k in itertools.combinations(sorted(row), 2):
            pairs[(j, k)].append(row[j] ** 2 * row[k] ** 2 / total**2)
    return [
        GraphEdge(attractor_node_id(j), attractor_node_id(k), EdgeType.CO_OCCURS, min(1.0, sum(v)), {"kind": "co_occurrence", "shared": len(v)})
        for (j, k), v in sorted(pairs.items())
    ]


def _common(a: dict[int, frozenset[str]], b: dict[int, frozenset[str]]) -> tuple[frozenset[str], frozenset[str]]:
    """Both places' cells at the finest resolution both have (nothing without one)."""
    level = max(a.keys() & b.keys(), default=None)
    return (a[level], b[level]) if level is not None else (frozenset(), frozenset())


def credibility(
    insights: list[Insight],
    batches: dict[str, dict[str, Any]],
    activates: list[GraphEdge],
    tampered: set[str],
    thresholds: PhenomenonThresholds,
    spatial: SpatialStructure,
) -> dict[str, dict[str, Any]]:
    """Each pattern's ``credibility`` and ``credibility_basis`` from independent confirmation (docs/06_graph_and_storage.md
    §6.2). Another source is another declared ``source`` (compared casefolded) outside the pattern's own upload; metrics
    compare by the ``indicators`` registry, and by unit when both give one. Phase A (``rule: anchor``, a pattern without a
    place): 5: a pattern of another declared source shifts a metric of the same indicator materially the opposite way in
    the same subgroup (one's conditions contain the other's); 1: one shares a non-weak anchor and shifts a metric of the
    same indicator the same way; 2: another dataset (declared or not, not of the same upload) shifts a metric of the same
    name; 6: none. Phase B (``rule: place``, a geo pattern), its cells against the other's at the finest resolution both
    datasets have (``spatial.places``): only a pattern whose place overlaps by ``GEO_REGION_MIN_SHARE`` of the smaller counts; 5 needs that share of
    the larger (a comparable place, in any subgroup), and 2 also comes from another dataset's events over that share of
    the place. 5 wins over 1. A version and the one it ``supersedes`` are one source.
    Never read: p, stability, weight; a tampered pattern confirms nothing."""
    batch = {ins.id: batches.get(ins.batch_id, {}) for ins in insights}
    anchors: dict[str, set[str]] = defaultdict(set)
    for e in activates:
        if not e.props["weak"]:
            anchors[e.source].add(e.target)
    of = {ins.id: ins for ins in insights}
    shifts = {ins.id: {s.metric: s for s in thresholds.phenomenon_shifts(ins)} for ins in insights}
    source = {pid: (batch[pid].get("source") or "").casefold() or None for pid in of}
    indicator = {  # per phenomenon metric: (indicator, unit or None), compared casefolded like the source
        pid: {
            m: (e["indicator"].casefold(), (e.get("unit") or "").casefold() or None)
            for m in shifts[pid]
            if (e := (batch[pid].get("indicators") or {}).get(m, {})).get("indicator")
        }
        for pid in of
    }
    conditions = {pid: frozenset(c.expr for c in ins.conditions) for pid, ins in of.items()}
    own = {pid: {ins.dataset_id, batch[pid].get("upload_id")} - {None} for pid, ins in of.items()}  # what a supersedes names
    place_at, occupied = spatial.places, spatial.occupied
    dataset_batch = {b.get("dataset_id"): b for b in batches.values() if b.get("status") == "READY"}
    by_name, by_indicator = defaultdict(set), defaultdict(set)
    for pid in of:
        if pid not in tampered:
            for m in shifts[pid]:
                by_name[m].add(pid)
            for i, _ in indicator[pid].values():
                by_indicator[i].add(pid)

    def independent(pid: str, other: dict[str, Any], dataset: str) -> bool:
        """``dataset`` (batch record ``other``) is outside the pattern's dataset, upload, declared source and versions."""
        upload, theirs = batch[pid].get("upload_id"), (other.get("source") or "").casefold() or None
        return not (
            dataset == of[pid].dataset_id
            or (upload and other.get("upload_id") == upload)
            or (source[pid] is not None and source[pid] == theirs)
            or other.get("supersedes") in own[pid]
            or batch[pid].get("supersedes") in {dataset, other.get("upload_id")} - {None}
        )

    def overlap(pid: str, q: str, over) -> bool:
        a, b = _common(place_at[pid], place_at.get(q, {}))
        return bool(a and b) and len(a & b) >= GEO_REGION_MIN_SHARE * over(len(a), len(b))

    out = {}
    # ponytail: every pair sharing a metric is compared (4000 such patterns: ~6 s a compile); index by condition past a few thousand
    for pid in of:
        mine, geo = shifts[pid], pid in place_at
        found: dict[int, set[str]] = defaultdict(set)
        for q in set().union(*(by_name[m] for m in mine), *(by_indicator[i] for i, _ in indicator[pid].values())):
            if not independent(pid, batch[q], of[q].dataset_id) or (geo and not overlap(pid, q, min)):
                continue  # one source in two datasets, a version and the one it replaces, or (with a place) elsewhere
            declared = source[pid] is not None and source[q] is not None
            pairs = [  # read by 1 and 5 only, which need both sources declared
                (mine[m], shifts[q][n])
                for m, (i, u) in indicator[pid].items()
                for n, (j, v) in indicator[q].items()
                if declared and i == j and (u is None or v is None or u == v)
            ]
            if (
                declared
                and (overlap(pid, q, max) if geo else conditions[pid] <= conditions[q] or conditions[q] <= conditions[pid])
                and any(a.robust_z * b.robust_z < 0 and thresholds.material(a) and thresholds.material(b) for a, b in pairs)
            ):
                found[5].add(q)
            elif declared and anchors[pid] & anchors[q] and any(a.robust_z * b.robust_z > 0 for a, b in pairs):
                found[1].add(q)
            elif set(mine) & set(shifts[q]):
                found[2].add(q)
        level = next((lv for lv in (5, 1, 2) if found[lv]), 6)
        patterns = sorted(found.get(level, ()))
        names = {source[q] or of[q].dataset_id: batch[q].get("source") or of[q].dataset_id for q in reversed(patterns)}  # one per source
        if geo and level in (2, 6):  # another dataset's events over the place
            for ds in sorted(occupied):
                a, b = _common(place_at[pid], occupied[ds])
                other = dataset_batch.get(ds, {})
                if a and len(a & b) >= GEO_REGION_MIN_SHARE * len(a) and independent(pid, other, ds):
                    level = 2
                    names.setdefault((other.get("source") or "").casefold() or ds, other.get("source") or ds)
        basis: dict[str, Any] = {"rule": "place" if geo else "anchor", "patterns": patterns, "sources": sorted(names.values())}
        basis |= {"comparability": c} if (c := {1: "indicator", 5: "indicator", 2: "nominal"}.get(level)) else {}
        if level == 5 and found[1]:
            basis["confirmed_by"] = sorted(found[1])
        out[pid] = {"credibility": level, "credibility_basis": basis}
    return out


def _attractor_nodes(
    ontology: LatentOntology, members: dict[int, list[Member]], seq_to_batch: dict[Any, str], rep: dict[str, Any]
) -> dict[str, GraphNode]:
    """Attractor nodes: label from the member signature, mass, scopes, dispersion, centroid contract."""
    nodes: dict[str, GraphNode] = {}
    for att in ontology.attractors():
        aid = att.id
        mem = members.get(aid, [])
        sig = attractor_signature([(r, s) for r, _, s in mem])
        nid = attractor_node_id(aid)
        nodes[nid] = GraphNode(
            nid,
            "Attractor",
            " · ".join(component_label(k, v) for k, v in sig[:2]) or f"Attractor {aid}",
            {
                "attractor_id": aid,
                "mass": round(float(att.mass), 3),
                "evidence_mass": float(sum(s for _, _, s in mem)),
                "n_patterns": len(mem),
                "distinct_scopes": len({frozenset((c["attribute"], c["value"]) for c in r["conditions"]) for r, _, _ in mem}),
                "dimensions": sorted({c["attribute"] for r, _, _ in mem for c in r["conditions"]}),
                "targets": sorted({r["target"] for r, _, _ in mem}),
                "datasets": sorted({r["dataset_id"] for r, _, _ in mem}),
                "signature": [{"component": k, "value": v} for k, v in sig[:6]],
                "dispersion": float(1.0 - np.mean([a for _, a, _ in mem])) if mem else None,
                "last_updated_batch": seq_to_batch.get(att.last_updated_seq, att.last_updated_seq),
                "created_at": att.created_at,
                "centroid": {
                    "dim": int(att.centroid.shape[0]),
                    "norm": float(np.linalg.norm(att.centroid)),
                    "representation_version": rep.get("representation_version"),
                    "fingerprint": rep.get("fingerprint"),
                },
            },
        )
    return nodes


def _spatial_plane(spatial: SpatialStructure, nodes: dict[str, GraphNode], structural: list[GraphEdge]) -> list[GraphEdge]:
    """Region nodes (added to ``nodes``), IN_REGION (``basis`` cells for a geo pattern, events with their props for a base
    one), PARENT_OF and CO_LOCATED edges, each geo pattern's ``compactness`` and a base pattern's ``geo: partial``; a
    CO_LOCATED pair the lattice already links is left out."""
    for rid, props in spatial.regions.items():
        nodes[rid] = GraphNode(rid, "Region", f"res {props['resolution']} {props['h3']}", dict(props))
    for pid, value in spatial.compactness.items():
        if pid in nodes:
            nodes[pid].props["compactness"] = value
    for pid in spatial.partial & nodes.keys():  # a base pattern too little located to be placed
        nodes[pid].props["geo"] = "partial"
    linked = {frozenset((e.source, e.target)) for e in structural}
    edges = [GraphEdge(p, r, EdgeType.IN_REGION, share, spatial.edge_props[(p, r)]) for p, r, share in spatial.in_region if p in nodes]
    edges += [GraphEdge(child, parent, EdgeType.PARENT_OF) for child, parent in spatial.parent_of]
    edges += [
        GraphEdge(a, b, EdgeType.CO_LOCATED, weight, {"relation": relation})
        for a, b, relation, weight in spatial.co_located
        if a in nodes and b in nodes and frozenset((a, b)) not in linked
    ]
    return edges


def _temporal_plane(
    temporal: dict[str, tuple[dict[str, Any] | None, str]],
    batches: dict[str, dict[str, Any]],
    integrity: dict[str, str],
    nodes: dict[str, GraphNode],
    settings: Settings,
) -> tuple[list[GraphEdge], dict[str, Any]]:
    """TemporalNode hubs (added to ``nodes``), PEAKED_AT / DROPPED_AT edges and the ``trajectories`` section from each
    READY dataset's signed time record and its series file (``temporal``: batch id → (series, the file's SHA-256)). A
    record that fails its metadata digest, or a file other than the one it signed (one that no longer reads among them),
    gives its dataset no temporal plane; a pattern that fails its record digest gets no trajectory. A trajectory is
    ``subgroup_miner.temporal.dynamics`` of the pattern's series; its short form is the pattern's ``temporal_summary``,
    the dataset's ``temporal`` its column, grid and span. A hub exists only with an event in it, and a pattern has one
    edge per hub and kind: its most prominent event there, ``metrics`` naming every metric that peaked (or dipped) then."""
    edges, trajectories = [], {}
    for bid, (series, digest) in sorted(temporal.items()):
        signed = batches[bid]["temporal"]
        if integrity.get(bid) == "mismatch" or digest != signed["series_digest"] or series is None:
            continue
        head = {k: signed[k] for k in ("column", "grid", "span")}
        nodes[dataset_node_id(batches[bid]["dataset_id"])].props["temporal"] = head
        for pid, entry in sorted(series.items()):
            if pid not in nodes or nodes[pid].props["integrity"] == "mismatch":
                continue
            props = nodes[pid].props
            metrics = [props["target"], *(s["metric"] for s in props["shifts"])]
            traj = trajectories[pid] = head | dynamics(entry, signed["bins"], metrics, settings.miner)
            nodes[pid].props["temporal_summary"] = {k: traj[k] for k in ("metric", "trend", "current", "bins_measured", "of")} | {
                "events": len(traj["events"])
            }
            by_period: dict[tuple[str, str], list[dict[str, Any]]] = {}  # (period, kind) -> its events, the most prominent first
            for e in sorted(traj["events"], key=lambda e: -e["prominence"]):
                by_period.setdefault((e["period"], e["kind"]), []).append(e)
            for (period, kind), same in by_period.items():
                hub, cell = temporal_node_id(signed["grid"], period), signed["bins"][same[0]["bin"]]
                props = {"granularity": signed["grid"], "period": period, "t_start": cell["t_start"], "t_end": cell["t_end"]}
                nodes.setdefault(hub, GraphNode(hub, "TemporalNode", period, props))
                first = {k: same[0][k] for k in ("metric", "period", "z_raw", "z_smoothed", "prominence", "p")}
                edges.append(GraphEdge(pid, hub, EdgeType(kind), 1.0, first | {"metrics": [e["metric"] for e in same]}))
    return edges, trajectories


def _checked(record: dict[str, Any], key: str, digest: Callable[[dict[str, Any]], str]) -> str:
    """``ok`` when ``record[key]`` matches the digest of the record, ``mismatch`` when it does not, ``unverified`` for a
    record stored before digests."""
    if key not in record:
        return "unverified"
    return "ok" if record[key] == digest(record) else "mismatch"


def compute_integrity(record: dict[str, Any]) -> str:
    """A pattern's ``integrity``: its journal record against its ``record_digest``."""
    return _checked(record, "record_digest", record_digest)


BATCH_DIGEST_FIELDS = (
    "dataset_id",
    "upload_id",
    "level",
    "filename",
    "source",
    "reliability",
    "indicators",
    "user_glossary",
    "metadata_doc",
    "supersedes",
    "profile",
    "temporal",  # the time column's grid and bins, and the SHA-256 of its series file (docs/02_discovery.md §2.10)
    "window",  # a chain-model time record's window: no new record has it; kept so such a record stays verifiable (§6.1)
)


def metadata_digest(batch: dict[str, Any]) -> str:
    """SHA-256 of a batch record's canonical JSON over what the graph reads of its dataset (``BATCH_DIGEST_FIELDS``), as
    the record is stored: a source, rating, registry or profile changed in the record, or removed, no longer matches."""
    covered = json.loads(json.dumps({k: batch[k] for k in BATCH_DIGEST_FIELDS if k in batch}, default=str))  # as atomic_write_json stores it
    return hashlib.sha256(json.dumps(covered, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def compute_metadata_integrity(batch: dict[str, Any]) -> str:
    """``ok`` / ``mismatch`` / ``unverified`` for a batch record's ``metadata_digest``, as ``compute_integrity`` for a pattern."""
    return _checked(batch, "metadata_digest", metadata_digest)


def _distrust(batch: dict[str, Any]) -> dict[str, Any]:
    """A batch record whose metadata failed its check, without what it declares: source, rating, registry, glossaries,
    metadata document."""
    out = {k: v for k, v in batch.items() if k not in ("source", "reliability", "indicators", "user_glossary", "metadata_doc")}
    geo = out.get("profile", {}).get("geo", {})
    if "glossary" in geo:
        out["profile"] = {**out["profile"], "geo": {k: v for k, v in geo.items() if k != "glossary"}}
    return out


def build_snapshot(
    *,
    records: list[dict[str, Any]],
    activations: list[dict[str, Any]],
    vectors: np.ndarray,
    batches: dict[str, dict[str, Any]],
    representation: dict[str, Any],
    ontology: LatentOntology,
    settings: Settings,
    spatial: SpatialStructure | None = None,
    temporal: dict[str, tuple[dict[str, Any], str]] | None = None,
) -> dict[str, Any]:
    """The dual graph of the committed state. ``records``: journal Pattern records (row = ``row_id`` of
    ``vectors``); ``batches``: batch id -> record (a READY one gives its dataset's schema nodes from its
    ``profile``); ``representation``: the recorded vector contract; ``spatial``: the derived spatial plane of the geo
    patterns (``subgroup_miner.spatial.derive_spatial``); ``temporal``: batch id → (its dataset's series, the series file's
    SHA-256) of the READY datasets with a time column (``_temporal_plane``)."""
    seq_to_batch = {b.get("batch_seq"): bid for bid, b in batches.items()}
    by_pid = {rec["id"]: rec for rec in records}
    insights = [Insight.from_record(rec) for rec in by_pid.values()]
    # The journal record is the pattern contract. Readers rehydrate it with Insight.from_record.
    nodes = {
        ins.id: GraphNode(ins.id, "Pattern", headline(ins), {**by_pid[ins.id], "integrity": compute_integrity(by_pid[ins.id])}) for ins in insights
    }
    integrity = {bid: compute_metadata_integrity(b) for bid, b in batches.items() if b.get("status") == "READY"}
    batches = {bid: _distrust(b) if integrity.get(bid) == "mismatch" else b for bid, b in batches.items()}  # declares nothing
    schema_nodes, edges = _schema_plane(batches, insights, settings, integrity)
    nodes.update(schema_nodes)
    structural = structural_edges(insights, settings.miner)
    spatial = spatial or SpatialStructure()
    edges += structural + _spatial_plane(spatial, nodes, structural)
    events, trajectories = _temporal_plane(temporal or {}, batches, integrity, nodes, settings)
    edges += events
    compiled = _co_memberships(activations, by_pid, vectors, ontology, settings)
    activates, members = _activation_edges(activations + compiled, vectors, ontology, by_pid, settings)
    edges += activates + co_occurrence_edges(activates)
    tampered = {pid for pid in by_pid if nodes[pid].props["integrity"] == "mismatch"}
    for pid, trust in credibility(insights, batches, activates, tampered, settings.thresholds, spatial).items():
        nodes[pid].props.update(trust)
    intact = {aid: [m for m in mem if nodes[m[0]["id"]].props["integrity"] != "mismatch"] for aid, mem in members.items()}
    nodes.update(_attractor_nodes(ontology, intact, seq_to_batch, representation))  # a tampered member describes no anchor
    for rel in ontology.topology():
        edges.append(
            GraphEdge(
                attractor_node_id(rel["source"]), attractor_node_id(rel["target"]), EdgeType.RELATED_TO, float(rel["weight"]), {"kind": "mutual_knn"}
            )
        )

    counts = Counter(e.type.value for e in edges)
    n_attr = len(ontology.attractor_ids)
    return {
        "version": SNAPSHOT_VERSION,
        "created_at": utc_now(),
        "representation": representation,
        "nodes": [{"id": n.id, "kind": n.kind, "label": n.label, "props": n.props} for n in nodes.values()],
        "edges": [e.to_dict() for e in edges],
        "own_cells": {pid: sorted(cells) for pid, cells in spatial.own_cells.items() if pid in nodes},
        "trajectories": trajectories,
        "stats": {
            "patterns": len(insights),
            "attractors": n_attr,
            "edges": len(edges),
            "edge_counts": dict(counts),
            "avg_attractor_degree": (2 * counts["RELATED_TO"] / n_attr) if n_attr else 0.0,
        },
    }
