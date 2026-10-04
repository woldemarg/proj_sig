"""The console's data: graph elements, node details and the latent sphere (docs/08_interface.md)."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
from fastapi import HTTPException

from graph_query_engine import CommittedState, DualGraph
from insight_contracts import Insight, attractor_id_of
from insight_contracts.text import humanize


def _short_label(graph: DualGraph, node: dict[str, Any]) -> str:
    if node["kind"] != "Pattern":
        return node["label"]
    ins = graph.insight(node["id"])
    scope = " · ".join(c.value for c in ins.conditions)
    if ins.phenomenon_type == "covariance" and ins.covariance:
        a, b = sorted(ins.covariance["pair"])
        return f"{scope}\ncorr({humanize(a)}, {humanize(b)})"
    return f"{scope}\n{humanize(ins.target)} {ins.effect_size:+.2f} sd"


def primary_anchor(graph: DualGraph, pattern_id: str) -> dict[str, Any] | None:
    """The pattern's strongest ACTIVATES edge: the theme it is shown under."""
    return max((e for e, _ in graph.incident(pattern_id, ["ACTIVATES"])), key=lambda e: e["weight"], default=None)


def cytoscape_elements(graph: DualGraph, dataset: str | None = None) -> dict[str, Any]:
    keep: set[str] = set()
    nodes = []
    for n in graph.nodes.values():
        p = n["props"]
        if dataset and n["kind"] in {"Pattern", "Dimension", "Metric", "Dataset"} and p.get("dataset_id") != dataset:
            continue
        if dataset and n["kind"] == "Attractor" and dataset not in p.get("datasets", []):
            continue
        keep.add(n["id"])
        data = {"id": n["id"], "kind": n["kind"], "label": _short_label(graph, n), "full_label": n["label"]}
        if n["kind"] == "Pattern":
            ins = graph.insight(n["id"])
            anchor = primary_anchor(graph, n["id"])
            data.update(
                weight=round(ins.weight, 3),
                direction=1 if ins.effect_size > 0 else -1,
                ptype=ins.phenomenon_type,
                dataset=ins.dataset_id,
                support=ins.support,
                target=ins.target,
                effect=round(ins.effect_size, 3),
                scope=[c.expr for c in ins.conditions],
                anchor=anchor["target"] if anchor else None,
                anchor_label=graph.nodes[anchor["target"]]["label"] if anchor else None,
            )
        elif n["kind"] == "Attractor":
            data.update(mass=p["mass"], n_patterns=p["n_patterns"])
        nodes.append({"data": data})
    edges = [
        {
            "data": {
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "type": e["type"],
                "plane": e["plane"],
                "weight": round(float(e["weight"]), 3),
                "weak": bool(e["props"].get("weak", False)),  # a coverage-only membership: drawn, not walked
            }
        }
        for e in graph.edges
        if e["source"] in keep and e["target"] in keep
    ]
    return {"nodes": nodes, "edges": edges, "stats": graph.snapshot.get("stats", {})}


def node_details(graph: DualGraph, node_id: str) -> dict[str, Any]:
    node = graph.nodes.get(node_id)
    if node is None:
        raise HTTPException(404, f"unknown node {node_id}")
    neighbors: dict[str, list[dict[str, Any]]] = {}
    for e in graph.out.get(node_id, []) + graph.inc.get(node_id, []):
        other = e["target"] if e["source"] == node_id else e["source"]
        key = e["type"] + ("" if e["source"] == node_id else " (in)")
        neighbors.setdefault(key, []).append(
            {"id": other, "label": graph.nodes[other]["label"], "kind": graph.nodes[other]["kind"], "weight": e["weight"], "props": e["props"]}
        )
    for items in neighbors.values():
        items.sort(key=lambda x: -float(x["weight"]))
    return {"node": node, "neighbors": neighbors}


# --- the latent sphere: pattern vectors and attractor centroids in the unit ball; the console draws them (§8.4)


def project_to_sphere(vectors: np.ndarray, seed: int) -> np.ndarray:
    """Points in the unit ball, as lac's prosphera projector places them: robust-scaled vectors -> cosine KernelPCA
    to 3D -> centred; the direction is the PCA's, the radius ``log ‖y‖²`` min-max scaled to [0.1, 1]."""
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import minmax_scale, robust_scale

    y = KernelPCA(n_components=3, kernel="cosine", random_state=seed, n_jobs=-1).fit_transform(robust_scale(vectors, quantile_range=(5, 95)))
    y = y - y.mean(axis=0)
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    return y / norms * minmax_scale(np.log(norms**2), feature_range=(0.1, 1))


def _pattern_hover(ins: Insight, anchor_label: str, alignment: float) -> str:
    # the data's names are escaped: plotly renders hover text as HTML
    shifts = "<br>".join(
        f"&nbsp;&nbsp;{escape(s.metric)}: {s.local_median:.4g} vs {s.global_median:.4g} (z {s.robust_z:+.2f})" for s in ins.shifts[:3]
    )
    return (
        f"<b>Insight</b> {ins.id}<br>"
        f"<b>Scope:</b> {escape(' ∧ '.join(c.expr for c in ins.conditions))}<br>"
        f"<b>Target:</b> {escape(ins.target)} ({ins.phenomenon_type})<br>{shifts}<br>"
        f"<b>Support:</b> {ins.support} rows · <b>evidence</b> {ins.weight:.2f}<br>"
        f"<b>Theme:</b> {escape(anchor_label)} (alignment {alignment:.2f})<br>"
        f"<b>Dataset:</b> {escape(ins.provenance.get('filename', ''))}"
    )


def _attractor_hover(node: dict[str, Any]) -> str:
    p = node["props"]
    return (
        f"<b>Theme</b> {node['id']}<br><b>{escape(node['label'])}</b><br>"
        f"<b>Insights:</b> {p['n_patterns']} over {p['distinct_scopes']} scopes<br>"
        f"<b>Mass:</b> {p['mass']} · <b>evidence mass</b> {p['evidence_mass']:.2f}<br>"
        f"<b>Columns:</b> {escape(', '.join(p['dimensions']))}"
    )


def sphere_points(state: CommittedState, seed: int, dataset: str | None = None) -> dict[str, Any]:
    """The sphere as data: every pattern and the attractors they activate, placed by one projection of one commit
    (vectors and centroids share the frame), plus the edges between them; ``message`` when there is nothing to draw."""
    g, frame = state.graph, state.frame
    pats = [n for n in g.of_kind("Pattern") if not dataset or n["props"]["dataset_id"] == dataset]
    drawn = {n["id"] for n in pats}
    acts = [e for e in g.edges if e["type"] == "ACTIVATES" and e["source"] in drawn]
    anchors = sorted({e["target"] for e in acts}, key=attractor_id_of)
    if not pats:
        return {"points": [], "edges": [], "message": "No insights to project — upload a dataset first."}
    if len(pats) + len(anchors) < 4:
        return {"points": [], "edges": [], "message": "A 3D projection needs at least 4 points (insights + themes)."}
    vectors = [frame.patterns[n["id"]] for n in pats] + [frame.attractors[attractor_id_of(a)] for a in anchors]
    coords = project_to_sphere(np.stack(vectors).astype(np.float64), seed)
    points = []
    for n, xyz in zip(pats, coords):
        ins = g.insight(n["id"])
        anchor = primary_anchor(g, n["id"])
        cls = "cov" if ins.phenomenon_type == "covariance" else "up" if ins.effect_size > 0 else "down"
        points.append(
            {
                "id": n["id"],
                "kind": "Pattern",
                "cls": cls,
                "weight": ins.weight,
                "xyz": xyz.round(5).tolist(),
                "hover": _pattern_hover(ins, g.nodes[anchor["target"]]["label"], float(anchor["weight"])),
            }
        )
    for a, xyz in zip(anchors, coords[len(pats) :]):
        node = g.nodes[a]
        points.append(
            {"id": a, "kind": "Attractor", "n_patterns": node["props"]["n_patterns"], "xyz": xyz.round(5).tolist(), "hover": _attractor_hover(node)}
        )
    placed = drawn | set(anchors)  # between placed nodes run only lattice, membership and anchor-anchor edges
    edges = [
        {"id": e["id"], "type": e["type"], "source": e["source"], "target": e["target"], "weak": bool(e["props"].get("weak"))}
        for e in g.edges
        if e["source"] in placed and e["target"] in placed
    ]
    return {"points": points, "edges": edges, "message": None}
