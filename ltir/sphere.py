"""3D latent sphere: insight vectors + attractors projected on a sphere (SDD 13 §Sphere).

Reuses lac's visualisation stack unchanged (vendored as ``ltir/engines/lac/projector.py``:
prosphera ``KernelPCA(cosine)`` -> sphere scaling, dark Plotly figure, ACTIVATES
lines, ``save_html``), the way lac's ``v2_orchestrator/viz_export.py`` feeds it chunks
+ concepts. SIG feeds Pattern vectors (the 1152-d journal rows) +
Attractor centroids and adds: one legend group per latent anchor, RELATED_TO
(mutual kNN) lines, optional lattice/contrast lines, and query-path highlighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ltir.engines.lac import projector as viz
from ltir.models import Insight, attractor_id_of

if TYPE_CHECKING:
    from ltir.pipeline import Engine

PLOTLY_LOCAL = "/vendor/plotly.min.js"
PATH_COLOR = "#f2b705"


class SphereError(RuntimeError):
    pass


def _pattern_hover(ins: Insight, anchor_label: str, alignment: float) -> str:
    shifts = "<br>".join(f"&nbsp;&nbsp;{s.metric}: {s.local_median:.4g} vs {s.global_median:.4g} (z {s.robust_z:+.2f})" for s in ins.shifts[:3])
    return (
        f"<b>🧩 Pattern</b> {ins.id}<br>"
        f"<b>Scope:</b> {' ∧ '.join(c.expr for c in ins.conditions)}<br>"
        f"<b>Target:</b> {ins.target} ({ins.phenomenon_type})<br>{shifts}<br>"
        f"<b>Support:</b> {ins.support} rows · <b>weight</b> {ins.weight:.2f}<br>"
        f"<b>Anchor:</b> {anchor_label} (alignment {alignment:.2f})<br>"
        f"<b>Dataset:</b> {ins.provenance.get('filename', '')}"
    )


def _attractor_hover(node: dict[str, Any]) -> str:
    p = node["props"]
    return (
        f"<b>🧠 Latent anchor</b> {node['id']}<br><b>{node['label']}</b><br>"
        f"<b>Patterns:</b> {p['n_patterns']} over {p['distinct_scopes']} scopes<br>"
        f"<b>Mass:</b> {p['mass']} · <b>evidence mass</b> {p['evidence_mass']:.2f}<br>"
        f"<b>Dimensions:</b> {', '.join(p['dimensions'])}"
    )


def _segments(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for a, b in pairs:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return xs, ys, zs


def sphere_figure(engine: Engine, *, dataset: str | None = None, highlight: dict[str, list[str]] | None = None, draw_edges: bool = True):
    """Plotly figure of the latent insight sphere (optionally with a retrieval highlight)."""
    import plotly.graph_objs as go

    g = engine.graph()
    pats = [n for n in g.of_kind("Pattern") if not dataset or n["props"]["dataset_id"] == dataset]
    if not pats:
        raise SphereError("no patterns to project — upload a dataset first")
    pidx = {n["id"]: i for i, n in enumerate(pats)}
    acts = [e for e in g.edges if e["type"] == "ACTIVATES" and e["source"] in pidx]
    att_ids = sorted({e["target"] for e in acts}, key=attractor_id_of)
    aidx = {a: i for i, a in enumerate(att_ids)}
    if len(pats) + len(att_ids) < 4:
        raise SphereError("need at least 4 points (patterns + attractors) for a 3D projection")

    frame = engine.frame()  # committed vectors: patterns and centroids share one space
    P = np.stack([frame.patterns[n["id"]] for n in pats]).astype(np.float64)
    A = np.stack([frame.attractors[attractor_id_of(a)] for a in att_ids]).astype(np.float64)

    best: dict[str, tuple[str, float]] = {}
    for e in acts:
        if e["source"] not in best or e["weight"] > best[e["source"]][1]:
            best[e["source"]] = (e["target"], float(e["weight"]))
    labels = [f"{best[n['id']][0]} · {g.nodes[best[n['id']][0]]['label']}" for n in pats]
    hovers = [_pattern_hover(g.insight(n["id"]), g.nodes[best[n["id"]][0]]["label"], best[n["id"]][1]) for n in pats]
    activations = [{"chunk_id": pidx[e["source"]], "concept_id": aidx[e["target"]], "weight": e["weight"]} for e in acts]

    projector = viz.OntologyProjector(random_state=engine.config.random_seed)
    coords, _ = projector._scale_vectors_on_sphere(projector._apply_pca(np.vstack([P, A])))
    pc, ac = coords[: len(pats)], coords[len(pats) :]
    fig = projector._build_figure(
        pc,
        ac,
        activations,
        chunk_labels=labels,
        chunk_hovertext=hovers,
        concept_hovertext=[_attractor_hover(g.nodes[a]) for a in att_ids],
        draw_edges=draw_edges,
    )

    # SIG adaptation of lac's figure
    colors = dict(zip(labels, viz._chunk_colors(labels)))
    hl = highlight or {}
    focus = set(hl.get("traversed", [])) | set(hl.get("evidence", [])) | set(hl.get("seeds", []))
    kept = []
    for tr in fig.data:
        if tr.name == "Chunks":
            continue  # replaced by one trace per latent anchor (legend-toggleable)
        if tr.name == "Concepts (L0)":
            tr.name = "Latent anchors (attractors)"
            tr.mode = "markers+text"
            tr.text = att_ids
            tr.textposition = "top center"
            tr.textfont = dict(color="#e9d5ff", size=11)
            tr.marker.size = [8 + 1.2 * g.nodes[a]["props"]["n_patterns"] for a in att_ids]
            tr.marker.color = "#c084fc"
            tr.marker.line = dict(color="#f8fafc", width=1)
        if tr.name == "ACTIVATES" and focus:
            tr.line.color = "rgba(255,255,255,0.04)"
        kept.append(tr)
    fig.data = tuple(kept)

    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    for lab in sorted(groups, key=lambda s: attractor_id_of(s.split(" · ")[0])):
        idx = groups[lab]
        fig.add_trace(
            go.Scatter3d(
                x=pc[idx, 0],
                y=pc[idx, 1],
                z=pc[idx, 2],
                mode="markers",
                legendgroup=lab,
                name=f"{lab if len(lab) <= 36 else lab[:35] + '…'} ({len(idx)})",  # full label in hover
                opacity=0.35 if focus else viz.STYLE["chunk_opacity"],
                marker=dict(size=[3 + 6 * g.insight(pats[i]["id"]).weight for i in idx], color=colors[lab], line=dict(width=0)),
                hovertext=[hovers[i] for i in idx],
                hovertemplate=viz.HOVER,
            )
        )

    def coord(node_id: str) -> np.ndarray | None:
        if node_id in pidx:
            return pc[pidx[node_id]]
        if node_id in aidx:
            return ac[aidx[node_id]]
        return None

    t = np.linspace(0, 2 * np.pi, 97)
    circles = [np.c_[np.cos(t), np.sin(t), 0 * t], np.c_[np.cos(t), 0 * t, np.sin(t)], np.c_[0 * t, np.cos(t), np.sin(t)]]
    fig.add_trace(
        go.Scatter3d(  # faint unit-sphere great circles for depth (not in lac's figure)
            x=np.concatenate([np.r_[c[:, 0], np.nan] for c in circles]),
            y=np.concatenate([np.r_[c[:, 1], np.nan] for c in circles]),
            z=np.concatenate([np.r_[c[:, 2], np.nan] for c in circles]),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color="rgba(148,163,184,0.16)", width=1),
        )
    )

    rel = [
        (ac[aidx[e["source"]]], ac[aidx[e["target"]]]) for e in g.edges if e["type"] == "RELATED_TO" and e["source"] in aidx and e["target"] in aidx
    ]
    if rel:
        x, y, z = _segments(rel)
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode="lines", name="RELATED_TO (mutual kNN)", line=dict(color="rgba(192,132,252,0.85)", width=4), hoverinfo="skip"
            )
        )
    for etype, color, name in (
        ("SPECIALIZES", "rgba(148,163,184,0.5)", "SPECIALIZES (lattice)"),
        ("CONTRASTS", "rgba(248,113,113,0.7)", "CONTRASTS"),
    ):
        segs = [(coord(e["source"]), coord(e["target"])) for e in g.edges if e["type"] == etype and e["source"] in pidx and e["target"] in pidx]
        if segs:
            x, y, z = _segments(segs)
            fig.add_trace(
                go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, visible="legendonly", line=dict(color=color, width=2), hoverinfo="skip")
            )

    if hl:
        by_id = {e["id"]: e for e in g.edges}
        path = [(coord(by_id[i]["source"]), coord(by_id[i]["target"])) for i in hl.get("edges", []) if i in by_id]
        path = [(a, b) for a, b in path if a is not None and b is not None]
        if path:
            x, y, z = _segments(path)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Retrieval path", line=dict(color=PATH_COLOR, width=7), hoverinfo="skip"))
        for key, name, symbol, color, size in (
            ("evidence", "Evidence", "circle-open", "#f8fafc", 8),
            ("transversal_only", "Cross-scope evidence", "circle-open", "#f87171", 10),
            ("seeds", "Seed", "diamond", PATH_COLOR, 7),
        ):
            ids = [i for i in hl.get(key, []) if i in pidx]
            if ids:
                pts = np.stack([pc[pidx[i]] for i in ids])
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="markers",
                        name=name,
                        marker=dict(symbol=symbol, size=size, color=color, line=dict(color=color, width=2)),
                        hovertext=[hovers[pidx[i]] for i in ids],
                        hovertemplate=viz.HOVER,
                    )
                )
        anchors = [a for a in hl.get("anchors", []) if a in aidx]
        if anchors:
            pts = np.stack([ac[aidx[a]] for a in anchors])
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    name="Anchors visited",
                    marker=dict(symbol="circle-open", size=15, color=PATH_COLOR, line=dict(color=PATH_COLOR, width=3)),
                    hoverinfo="skip",
                )
            )

    subtitle = (
        f"{len(pats)} insight vectors ({P.shape[1]}-d) + {len(att_ids)} attractors · KernelPCA(cosine) → sphere (prosphera) · colour = latent anchor"
    )
    fig.update_layout(
        title=dict(text=f"<b>Latent Insight Sphere</b><br><sup>{subtitle}</sup>"),
        legend=dict(
            title=dict(text="Latent anchors · layers (click to toggle)", font=dict(size=11)),
            font=dict(size=10),
            x=0.995,
            xanchor="right",
            y=0.9,
            yanchor="top",
            bgcolor="rgba(15, 23, 42, 0.55)",
            tracegroupgap=0,
        ),
        uirevision="sig-sphere",
    )
    fig.update_scenes(domain=dict(x=[0.0, 0.7], y=[0.0, 1.0]))  # legend lives in the right-hand strip
    return fig


def sphere_html(engine: Engine, *, dataset: str | None = None, highlight: dict | None = None, plotly_src: str = PLOTLY_LOCAL) -> str:
    """Full HTML page; ``plotly_src`` = local vendored path (served) or ``"cdn"``."""
    fig = sphere_figure(engine, dataset=dataset, highlight=highlight)
    html = fig.to_html(
        full_html=True, include_plotlyjs=plotly_src, default_width="100%", default_height="100%", config={"displaylogo": False, "responsive": True}
    )
    css = f"<style>html, body {{ margin: 0; height: 100%; overflow: hidden; background-color: {viz.BG}; }}</style>"
    return html.replace("<head>", f"<head>\n{css}", 1)


def export_sphere(engine: Engine, output: Path | None = None, dataset: str | None = None) -> Path:
    """Standalone copy like lac's ``ontology_sphere.html`` (plotly from CDN, opens from disk)."""
    out = output or engine.ws.root / "graph" / "sphere.html"
    viz.save_html(sphere_figure(engine, dataset=dataset), out)
    return out


def plotly_js_path() -> Path:
    import plotly

    return Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
