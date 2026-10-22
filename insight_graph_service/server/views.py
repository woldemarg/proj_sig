"""The console's data: graph elements, node details and the latent sphere (docs/08_interface.md)."""

from __future__ import annotations

from collections import Counter
from html import escape
from typing import Any

import numpy as np
from fastapi import HTTPException

from graph_query_engine import CommittedState, DualGraph
from graph_query_engine.evidence import chain_dict
from graph_query_engine.graph import insight_reading, primary_anchor
from graph_query_engine.vertical import drawn_regions, picture, place_key, places, scale_chain, shown_regions
from insight_contracts import Insight, PhenomenonThresholds, attractor_id_of, dataset_node_id, wording
from insight_contracts.text import humanize


def _short_label(graph: DualGraph, node: dict[str, Any], members: int = 0) -> str:
    if node["kind"] == "Region":
        return f"рівень {node['props']['resolution']}\n{members} {wording.plural(members, 'інсайт', 'інсайти', 'інсайтів')}"
    if node["kind"] != "Pattern":
        return node["label"]
    ins = graph.insight(node["id"])
    scope = " · ".join(c.value for c in ins.conditions)
    if ins.phenomenon_type == "covariance" and ins.covariance:
        a, b = sorted(ins.covariance["pair"])
        return f"{scope}\ncorr({humanize(a)}, {humanize(b)})"
    return f"{scope}\n{humanize(ins.target)} {ins.effect_size:+.2f} sd"


def cytoscape_elements(graph: DualGraph, dataset: str | None = None, regions: bool = False) -> dict[str, Any]:
    """The graph view of one dataset (or all), derived once per commit: without its Region nodes and their edges unless
    ``regions`` (docs/08_interface.md §8.2), their count given either way."""
    return graph.memoised(("cytoscape", dataset or "", regions), lambda: _elements(graph, dataset, regions))


def _elements(graph: DualGraph, dataset: str | None, with_regions: bool) -> dict[str, Any]:
    keep: set[str] = set()
    nodes = []
    pids = {n["id"] for n in graph.of_kind("Pattern") if not dataset or n["props"]["dataset_id"] == dataset}
    members = Counter({r["id"]: sum(p in pids for p in graph.holders(r["id"])) for r in graph.of_kind("Region")})  # the insights in view: a size
    regions = shown_regions(graph, list(pids))  # the regions the map draws for them, a scale chain's above with none in view too
    for n in graph.nodes.values():
        p = n["props"]
        if n["kind"] == "Region" and (not with_regions or n["id"] not in regions):
            continue
        if dataset and n["kind"] in {"Pattern", "Dimension", "Metric", "Dataset"} and p.get("dataset_id") != dataset:
            continue
        if dataset and n["kind"] == "Attractor" and dataset not in p.get("datasets", []):
            continue
        keep.add(n["id"])
        data = {"id": n["id"], "kind": n["kind"], "label": _short_label(graph, n, members[n["id"]]), "full_label": n["label"]}
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
        elif n["kind"] == "Region":
            data.update(resolution=p["resolution"], patterns=members[n["id"]])
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
        if e["source"] in keep and e["target"] in keep and (e["type"] != "IN_REGION" or e["target"] in drawn_regions(graph, e["source"]))
    ]  # an insight's IN_REGION to the regions drawn for it: a broad one lies in hundreds more
    trust = {
        n["props"]["dataset_id"]: {k: n["props"][k] for k in ("source", "reliability", "integrity") if k in n["props"]}
        for n in graph.of_kind("Dataset")
    }  # every dataset's trust, whatever the filter
    return {"nodes": nodes, "edges": edges, "regions": len(regions), "stats": graph.snapshot.get("stats", {}), "datasets": trust}


IN_REGION_SHOWN = 25  # a pattern's regions in its panel
NEIGHBOURS_SHOWN = 50  # any other group's strongest: a theme or a region of many insights stays a small panel


def node_details(graph: DualGraph, node_id: str, thresholds: PhenomenonThresholds) -> dict[str, Any]:
    node = graph.nodes.get(node_id)
    if node is None:
        raise HTTPException(404, f"unknown node {node_id}")
    neighbors: dict[str, list[dict[str, Any]]] = {}
    for e in graph.out.get(node_id, []) + graph.inc.get(node_id, []):
        other = e["target"] if e["source"] == node_id else e["source"]
        key = e["type"] + ("" if e["source"] == node_id else " (in)")
        on = graph.nodes[other]  # its integrity: Walk on offers intact insights only (docs/08_interface.md §8.7)
        neighbors.setdefault(key, []).append(
            {
                "id": other,
                "label": on["label"],
                "kind": on["kind"],
                "weight": e["weight"],
                "props": e["props"],
                "integrity": on["props"].get("integrity"),
            }
        )
    if node["kind"] == "Region":  # the patterns whose own cell it is hold it too (``DualGraph.holders``)
        members = {it["id"] for it in neighbors.get("IN_REGION (in)", [])}
        for pid in (p for p in graph.holders(node_id) if p not in members):
            item = {
                "id": pid,
                "label": graph.nodes[pid]["label"],
                "kind": "Pattern",
                "weight": 1.0,
                "props": {"basis": "cells", "share": 1.0, "own": True},
                "integrity": graph.nodes[pid]["props"].get("integrity"),
            }
            neighbors.setdefault("IN_REGION (in)", []).append(item)
    for key, items in neighbors.items():  # a pattern's regions as its places rank them, the rest by weight
        items.sort(key=(lambda x: tuple(-k for k in place_key(x))) if key == "IN_REGION" else (lambda x: -float(x["weight"])))
    out: dict[str, Any] = {"node": node, "neighbors": neighbors, "neighbor_counts": {k: len(v) for k, v in neighbors.items()}}
    for key, items in neighbors.items():  # the strongest of each group, its count beside them (a broad pattern lies in hundreds of regions)
        neighbors[key] = items[: IN_REGION_SHOWN if key == "IN_REGION" else NEIGHBOURS_SHOWN]
    if node["kind"] == "Pattern":  # its dataset's trust, its places and, for a geo pattern, its scale chain (docs/08_interface.md §8.2)
        out["reading"] = insight_reading(graph, node_id, thresholds)
        out["dataset"] = graph.nodes.get(dataset_node_id(node["props"]["dataset_id"]), {"props": {}})["props"]
        edges = {e["target"]: e for e in graph.out.get(node_id, []) if e["type"] == "IN_REGION"}
        out["places"] = [
            {"id": r, **{k: edges[r]["props"][k] for k in ("share", "lift", "n") if k in edges[r]["props"]}} for r, _ in places(graph, node_id)
        ]
        chain = scale_chain(graph, [node_id])
        steps = chain[0].up + chain[0].down if chain else []
        out["scale_chain"] = chain_dict(chain[0], graph) if chain else None
        out["levels"] = {s.pattern_id: graph.facts(s.pattern_id).resolution for s in steps}
        anchor = primary_anchor(graph, node_id)  # the theme it is shown under, as the map's list shows it
        out["theme"] = {"id": anchor["target"], "label": graph.nodes[anchor["target"]]["label"]} if anchor else None
        basis = node["props"].get("credibility_basis", {})
        linked = [*basis.get("patterns", []), *basis.get("confirmed_by", [])]
        linked += [s.pattern_id for s in steps]
        out["labels"] = {pid: graph.nodes[pid]["label"] for pid in linked if pid in graph.nodes}  # of other datasets too
    elif node["kind"] == "Region":
        out["picture"] = picture(graph, node_id)
        out["reading"] = {"what": wording.region_reading(node_id, out["picture"])}
    elif node["kind"] == "Attractor":
        props = node["props"]
        out["reading"] = {"what": wording.theme_reading(props["signature"], props["n_patterns"], props["distinct_scopes"])}
    return out


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


def _pattern_hover(ins: Insight, reading: dict[str, str], theme: str) -> str:
    # the data's names are escaped: plotly renders hover text as HTML
    return (
        f"<b>{wording.name('insight')}</b> {ins.id}<br>"
        f"<b>{wording.name('scope')}:</b> {escape(reading['where'])}<br>{escape(reading['what'])}<br>"
        f"<b>{wording.name('support')}:</b> {ins.support} · <b>{wording.name('weight')}</b> {ins.weight:.2f}<br>"
        f"<b>{wording.name('theme')}:</b> {theme}<br>"
        f"<b>датасет:</b> {escape(ins.provenance.get('filename', ''))}"
    )


def _attractor_hover(node: dict[str, Any]) -> str:
    p = node["props"]
    return (
        f"<b>{wording.name('theme')}</b> {node['id']}<br><b>{escape(node['label'])}</b><br>"
        f"{escape(wording.theme_reading(p['signature'], p['n_patterns'], p['distinct_scopes']))}<br>"
        f"<b>{wording.name('mass')}:</b> {p['mass']} · <b>{wording.name('evidence_mass')}</b> {p['evidence_mass']:.2f}<br>"
        f"<b>колонки:</b> {escape(', '.join(p['dimensions']))}"
    )


def sphere_points(state: CommittedState, seed: int, thresholds: PhenomenonThresholds, dataset: str | None = None) -> dict[str, Any]:
    """The sphere as data: every pattern and the attractors they activate, placed by one projection of one commit
    (vectors and centroids share the frame), plus the edges between them; ``message`` when there is nothing to draw."""
    g, frame = state.graph, state.frame
    pats = [n for n in g.of_kind("Pattern") if not dataset or n["props"]["dataset_id"] == dataset]
    drawn = {n["id"] for n in pats}
    acts = [e for e in g.edges if e["type"] == "ACTIVATES" and e["source"] in drawn]
    anchors = sorted({e["target"] for e in acts}, key=attractor_id_of)
    if not pats:
        return {"points": [], "edges": [], "message": "Немає інсайтів для проєкції — спершу завантажте датасет."}
    if len(pats) + len(anchors) < 4:
        return {"points": [], "edges": [], "message": "Для 3D-проєкції потрібно щонайменше 4 точки (інсайти й теми)."}
    vectors = [frame.patterns[n["id"]] for n in pats] + [frame.attractors[attractor_id_of(a)] for a in anchors]
    coords = project_to_sphere(np.stack(vectors).astype(np.float64), seed)
    points = []
    for n, xyz in zip(pats, coords):
        ins = g.insight(n["id"])
        anchor = primary_anchor(g, n["id"])
        theme = (
            f"{escape(g.nodes[anchor['target']]['label'])} (відповідність {float(anchor['weight']):.2f})"
            if anchor
            else "немає (лише слабкі зв'язки з темами)"
        )
        cls = "cov" if ins.phenomenon_type == "covariance" else "up" if ins.effect_size > 0 else "down"
        points.append(
            {
                "id": n["id"],
                "kind": "Pattern",
                "cls": cls,
                "weight": ins.weight,
                "xyz": xyz.round(5).tolist(),
                "hover": _pattern_hover(ins, insight_reading(g, n["id"], thresholds), theme),
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
