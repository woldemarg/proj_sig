"""Dual-layer graph assembly + in-memory index (docs/06_graph_and_storage.md).

The snapshot is *derived* data: it is rebuilt from the journals, dataset
artifacts and ontology state after every batch, so it can always be
regenerated (``ltir rebuild-graph``). ACTIVATES alignments are recomputed
against the *current* centroids so graph weights match the living ontology.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from ltir.canonical import headline, humanize
from ltir.config import Config
from ltir.models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    Insight,
    attractor_node_id,
    batch_node_id,
    dataset_node_id,
    dimension_node_id,
    metric_node_id,
)
from ltir.ontology import LatentOntology
from ltir.store import Workspace, read_json, utc_now
from ltir.structural import structural_edges

SNAPSHOT_VERSION = 2  # Pattern props are the journal Insight record (not a projected schema)

Member = tuple[dict[str, Any], float, float]  # (pattern record, alignment, strength)


def _component_word(label: str, value: float) -> str:
    if label.startswith("correlation between"):
        pair = label.removeprefix("correlation between ").replace(" and ", "~")
        return f"corr({pair}) {'strengthens' if value > 0 else 'weakens'}"
    return f"{label} {'↑' if value > 0 else '↓'}"


def describe_component(label: str, value: float) -> str:
    """Readable signature entry: 'discount up', 'delivery days down', 'correlation between a and b weakens'."""
    if label.startswith("correlation between"):
        return f"{label} {'strengthens' if value > 0 else 'weakens'}"
    return f"{label} {'up' if value > 0 else 'down'}"


def describe_components(signature: list[dict[str, Any]]) -> str:
    """Two strongest entries of a stored signature as prose (how the LLM prompt names an anchor)."""
    return " and ".join(describe_component(e["component"], e["value"]) for e in signature[:2])


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
    ws: Workspace, batches: dict[str, dict[str, Any]], insights: list[Insight], config: Config
) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
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
        profile = read_json(ws.datasets_dir / ds / "profile.json", {}) or {}
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
            if nid in nodes and (s.metric == ins.target or s.magnitude >= config.min_component_z):
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


def _activation_edges(ws: Workspace, ontology: LatentOntology, by_pid: dict[str, dict[str, Any]]) -> tuple[list[GraphEdge], dict[int, list[Member]]]:
    """ACTIVATES edges with alignments recomputed against the current centroids, grouped by attractor."""
    vectors = ws.vectors()
    live = set(ontology.store.concept_ids)
    edges: list[GraphEdge] = []
    members: dict[int, list[Member]] = defaultdict(list)
    for act in ws.activations():
        pid, aid = act["pattern_id"], int(act["attractor_id"])
        if pid not in by_pid or aid not in live:
            continue
        rec = by_pid[pid]
        alignment = float(vectors[rec["row_id"]].astype(np.float64) @ ontology.centroid(aid).astype(np.float64))
        w = float(rec["weight"])
        # one predicate for coverage and retrieval: below the ontology's alignment floor (rerouted at
        # ingest, or drifted below it since) a membership is coverage only and is not walked
        weak = bool(act["weak"]) or alignment < ontology.config.min_activation_alignment
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


def _attractor_nodes(
    ontology: LatentOntology, members: dict[int, list[Member]], seq_to_batch: dict[Any, str], rep: dict[str, Any]
) -> dict[str, GraphNode]:
    """Attractor nodes: label from the member signature, mass, scopes, dispersion, centroid contract."""
    st = ontology.store
    nodes: dict[str, GraphNode] = {}
    for idx, aid in enumerate(st.concept_ids):
        mem = members.get(int(aid), [])
        sig = attractor_signature([(r, s) for r, _, s in mem])
        nid = attractor_node_id(aid)
        nodes[nid] = GraphNode(
            nid,
            "Attractor",
            " · ".join(_component_word(k, v) for k, v in sig[:2]) or f"Attractor {aid}",
            {
                "attractor_id": int(aid),
                "mass": int(st.chunk_counts[idx]),
                "evidence_mass": float(sum(s for _, _, s in mem)),
                "n_patterns": len(mem),
                "distinct_scopes": len({frozenset((c["attribute"], c["value"]) for c in r["conditions"]) for r, _, _ in mem}),
                "dimensions": sorted({c["attribute"] for r, _, _ in mem for c in r["conditions"]}),
                "targets": sorted({r["target"] for r, _, _ in mem}),
                "datasets": sorted({r["dataset_id"] for r, _, _ in mem}),
                "signature": [{"component": k, "value": v} for k, v in sig[:6]],
                "dispersion": float(1.0 - np.mean([a for _, a, _ in mem])) if mem else None,
                "last_updated_batch": seq_to_batch.get(int(st.last_updated_batch[idx]), int(st.last_updated_batch[idx])),
                "created_at": st.created_at[idx],
                "centroid": {
                    "dim": int(st.embeddings.shape[1]),
                    "norm": float(np.linalg.norm(st.embeddings[idx])),
                    "representation_version": rep.get("representation_version"),
                    "fingerprint": rep.get("fingerprint"),
                },
            },
        )
    return nodes


def build_snapshot(ws: Workspace, ontology: LatentOntology, config: Config, pending: dict[str, Any] | None = None) -> dict[str, Any]:
    """``pending``: the batch being committed; treated as READY for schema nodes."""
    batches = {b["batch_id"]: b for b in ws.list_batches()}
    if pending is not None:
        batches[pending["batch_id"]] = {**pending, "status": "READY"}
    seq_to_batch = {b.get("batch_seq"): bid for bid, b in batches.items()}
    rep = ws.representation() or {}

    by_pid = {rec["id"]: rec for rec in ws.patterns()}
    insights = [Insight.from_record(rec) for rec in by_pid.values()]
    # The journal record is the pattern contract. Readers rehydrate it with Insight.from_record.
    nodes = {ins.id: GraphNode(ins.id, "Pattern", headline(ins), dict(by_pid[ins.id])) for ins in insights}
    schema_nodes, edges = _schema_plane(ws, batches, insights, config)
    nodes.update(schema_nodes)
    edges += structural_edges(insights, config)
    activates, members = _activation_edges(ws, ontology, by_pid)
    edges += activates
    nodes.update(_attractor_nodes(ontology, members, seq_to_batch, rep))
    for rel in ontology.topology():
        edges.append(
            GraphEdge(
                attractor_node_id(rel["source"]), attractor_node_id(rel["target"]), EdgeType.RELATED_TO, float(rel["weight"]), {"kind": "mutual_knn"}
            )
        )

    counts: dict[str, int] = defaultdict(int)
    for e in edges:
        counts[e.type.value] += 1
    n_attr = len(ontology.store.concept_ids)
    return {
        "version": SNAPSHOT_VERSION,
        "created_at": utc_now(),
        "representation": rep,
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


class DualGraph:
    """Read-only adjacency index over a snapshot (used by traversal, UI, evidence).

    Pattern nodes carry the journal record. ``insight()`` is the object retrieval
    uses; ``canonical_document()`` is the text sitting beside that record.
    """

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self._catalog: Any = None  # the literal catalog of docs/07 §7.1.1, attached lazily by the engine (None on hand-built graphs)
        self._catalog_loader: Any = None
        self.nodes: dict[str, dict[str, Any]] = {n["id"]: n for n in snapshot.get("nodes", [])}
        self.edges: list[dict[str, Any]] = snapshot.get("edges", [])
        self.insights: dict[str, Insight] = {n["id"]: Insight.from_record(n["props"]) for n in self.nodes.values() if n["kind"] == "Pattern"}
        self.out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.inc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in self.edges:
            self.out[e["source"]].append(e)
            self.inc[e["target"]].append(e)

    @property
    def catalog(self) -> Any:
        """``query.LiteralCatalog`` for grounding, built on first use so read-only commands never load the model."""
        if self._catalog is None and self._catalog_loader is not None:
            self._catalog, self._catalog_loader = self._catalog_loader(), None
        return self._catalog

    def set_catalog_loader(self, loader: Any) -> None:
        self._catalog, self._catalog_loader = None, loader

    def insight(self, node_id: str) -> Insight:
        return self.insights[node_id]

    def canonical_document(self, node_id: str) -> str:
        return self.nodes[node_id]["props"]["canonical"]["document"]

    @classmethod
    def empty(cls) -> DualGraph:
        return cls({"nodes": [], "edges": []})

    def kind(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        return node["kind"] if node else None

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["kind"] == kind]

    def incident(self, node_id: str, types: Iterable[str]) -> list[tuple[dict[str, Any], str]]:
        """(edge, other endpoint) for edges of the given types touching node_id."""
        wanted = set(types)
        found = [(e, e["target"]) for e in self.out.get(node_id, []) if e["type"] in wanted]
        found += [(e, e["source"]) for e in self.inc.get(node_id, []) if e["type"] in wanted]
        return found
