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

import itertools
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from insight_contracts import (
    SNAPSHOT_VERSION,
    EdgeType,
    GraphEdge,
    GraphNode,
    Insight,
    attractor_id_of,
    attractor_node_id,
    batch_node_id,
    dataset_node_id,
    dimension_node_id,
    metric_node_id,
)
from insight_contracts.text import component_label, headline, humanize
from insight_graph_service.core.settings import Settings
from insight_graph_service.core.workspace import utc_now
from subgroup_miner.lattice import colocation_edges, structural_edges

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


def _schema_plane(batches: dict[str, dict[str, Any]], insights: list[Insight], settings: Settings) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    """Dataset / Batch / Dimension / Metric nodes of READY batches, plus the edges into them."""
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    scope_attributes: dict[str, set[str]] = defaultdict(set)  # closed intents may use non-selected columns
    for ins in insights:
        scope_attributes[ins.dataset_id].update(c.attribute for c in ins.conditions)
    for bid, b in batches.items():
        if b.get("status") != "READY":
            continue
        ds = b["dataset_id"]
        profile = b.get("profile") or {}
        nodes[dataset_node_id(ds)] = GraphNode(
            dataset_node_id(ds),
            "Dataset",
            b.get("filename", ds),
            {"dataset_id": ds, "filename": b.get("filename"), "rows": profile.get("rows"), "columns": profile.get("columns")},
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
                "mass": att.mass,
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


def build_snapshot(
    *,
    records: list[dict[str, Any]],
    activations: list[dict[str, Any]],
    vectors: np.ndarray,
    batches: dict[str, dict[str, Any]],
    representation: dict[str, Any],
    ontology: LatentOntology,
    settings: Settings,
) -> dict[str, Any]:
    """The dual graph of the committed state. ``records``: journal Pattern records (row = ``row_id`` of
    ``vectors``); ``batches``: batch id -> record (a READY one gives its dataset's schema nodes from its
    ``profile``); ``representation``: the recorded vector contract."""
    seq_to_batch = {b.get("batch_seq"): bid for bid, b in batches.items()}
    by_pid = {rec["id"]: rec for rec in records}
    insights = [Insight.from_record(rec) for rec in by_pid.values()]
    # The journal record is the pattern contract. Readers rehydrate it with Insight.from_record.
    nodes = {ins.id: GraphNode(ins.id, "Pattern", headline(ins), dict(by_pid[ins.id])) for ins in insights}
    schema_nodes, edges = _schema_plane(batches, insights, settings)
    nodes.update(schema_nodes)
    structural = structural_edges(insights, settings.miner)
    edges += structural + colocation_edges(insights, structural)
    compiled = _co_memberships(activations, by_pid, vectors, ontology, settings)
    activates, members = _activation_edges(activations + compiled, vectors, ontology, by_pid, settings)
    edges += activates + co_occurrence_edges(activates)
    nodes.update(_attractor_nodes(ontology, members, seq_to_batch, representation))
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
        "stats": {
            "patterns": len(insights),
            "attractors": n_attr,
            "edges": len(edges),
            "edge_counts": dict(counts),
            "avg_attractor_degree": (2 * counts["RELATED_TO"] / n_attr) if n_attr else 0.0,
        },
    }
