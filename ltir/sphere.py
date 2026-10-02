"""3D latent sphere: insight vectors + attractors projected on a sphere (docs/08_interface.md §8.4).

Reuses lac's visualisation stack (vendored as ``ltir/engines/lac/projector.py``: prosphera
``KernelPCA(cosine)`` -> sphere scaling, the figure skeleton, ``save_html``), the way lac's
``v2_orchestrator/viz_export.py`` feeds it chunks + concepts. SIG feeds Pattern vectors (the
journal rows) + Attractor centroids and draws them with the *same* visual language as the 2D
graph (docs/08 §8.3): node colour = metric higher / lower / correlation change, theme = diamond,
link layers named like the UI's legend toggles, answer markers identical to the graph's rings.
The web UI passes its theme palette and layer state; the standalone export uses the dark palette.
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
# the dark-theme tokens of ltir/web/static/style.css (the UI sends its live palette; the export has none)
DEFAULT_PALETTE = {
    "bg": "#0e1118",
    "ink": "#c4c9d8",
    "muted": "#7f879d",
    "anchor": "#a78bfa",
    "up": "#5aa2ff",
    "down": "#ff9a52",
    "cov": "#3ccb92",
    "path": "#ffc53d",
    "bad": "#ff6b6b",
}
# the UI's layer toggles and their initial state (index.html); a trace is a layer when it carries one of these names
LAYER_TRACES = {
    "lattice": ("Hierarchy",),
    "contrast": ("Contrasts",),
    "sibling": ("Siblings",),
    "latent": ("Theme links",),
    "activates": ("Memberships", "Memberships (weak)"),
}
LAYER_DEFAULTS = {"lattice": True, "contrast": False, "sibling": False, "latent": True, "activates": False}


class SphereError(RuntimeError):
    pass


def _pattern_hover(ins: Insight, anchor_label: str, alignment: float) -> str:
    shifts = "<br>".join(f"&nbsp;&nbsp;{s.metric}: {s.local_median:.4g} vs {s.global_median:.4g} (z {s.robust_z:+.2f})" for s in ins.shifts[:3])
    return (
        f"<b>Insight</b> {ins.id}<br>"
        f"<b>Scope:</b> {' ∧ '.join(c.expr for c in ins.conditions)}<br>"
        f"<b>Target:</b> {ins.target} ({ins.phenomenon_type})<br>{shifts}<br>"
        f"<b>Support:</b> {ins.support} rows · <b>evidence</b> {ins.weight:.2f}<br>"
        f"<b>Theme:</b> {anchor_label} (alignment {alignment:.2f})<br>"
        f"<b>Dataset:</b> {ins.provenance.get('filename', '')}"
    )


def _attractor_hover(node: dict[str, Any]) -> str:
    p = node["props"]
    return (
        f"<b>Theme</b> {node['id']}<br><b>{node['label']}</b><br>"
        f"<b>Insights:</b> {p['n_patterns']} over {p['distinct_scopes']} scopes<br>"
        f"<b>Mass:</b> {p['mass']} · <b>evidence mass</b> {p['evidence_mass']:.2f}<br>"
        f"<b>Columns:</b> {', '.join(p['dimensions'])}"
    )


def _segments(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for a, b in pairs:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return xs, ys, zs


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _is_dark(hex_color: str) -> bool:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b < 128


def sphere_figure(
    engine: Engine,
    *,
    dataset: str | None = None,
    highlight: dict[str, list[str]] | None = None,
    palette: dict[str, str] | None = None,
    layers: dict[str, bool] | None = None,
):
    """Plotly figure of the latent insight sphere, in the graph's visual language (optionally highlighted)."""
    import plotly.graph_objs as go

    pal = {**DEFAULT_PALETTE, **(palette or {})}
    on = {**LAYER_DEFAULTS, **(layers or {})}
    g = engine.graph()
    pats = [n for n in g.of_kind("Pattern") if not dataset or n["props"]["dataset_id"] == dataset]
    if not pats:
        raise SphereError("no insights to project — upload a dataset first")
    pidx = {n["id"]: i for i, n in enumerate(pats)}
    acts = [e for e in g.edges if e["type"] == "ACTIVATES" and e["source"] in pidx]
    att_ids = sorted({e["target"] for e in acts}, key=attractor_id_of)
    aidx = {a: i for i, a in enumerate(att_ids)}
    if len(pats) + len(att_ids) < 4:
        raise SphereError("need at least 4 points (insights + themes) for a 3D projection")

    frame = engine.frame()  # committed vectors: patterns and centroids share one space
    P = np.stack([frame.patterns[n["id"]] for n in pats]).astype(np.float64)
    A = np.stack([frame.attractors[attractor_id_of(a)] for a in att_ids]).astype(np.float64)

    best: dict[str, tuple[str, float]] = {}
    for e in acts:
        if e["source"] not in best or e["weight"] > best[e["source"]][1]:
            best[e["source"]] = (e["target"], float(e["weight"]))
    hovers = [_pattern_hover(g.insight(n["id"]), g.nodes[best[n["id"]][0]]["label"], best[n["id"]][1]) for n in pats]

    projector = viz.OntologyProjector(random_state=engine.config.random_seed)
    coords, _ = projector._scale_vectors_on_sphere(projector._apply_pca(np.vstack([P, A])))
    pc, ac = coords[: len(pats)], coords[len(pats) :]
    fig = projector._build_figure(pc, ac, [], concept_hovertext=[_attractor_hover(g.nodes[a]) for a in att_ids], draw_edges=False)

    hl = highlight or {}
    focus = set(hl.get("traversed", [])) | set(hl.get("evidence", [])) | set(hl.get("seeds", []))
    for tr in fig.data:
        if tr.name == "Concepts (L0)":  # lac's concept markers become the themes, drawn like the graph's hexagons
            tr.name = "Themes"
            tr.mode = "markers+text"
            tr.text = att_ids
            tr.customdata = att_ids  # node ids: a click opens the same details drawer as in the graph
            tr.textposition = "top center"
            tr.textfont = dict(color=pal["ink"], size=11)
            tr.marker.symbol = "diamond"
            tr.marker.size = [8 + 1.2 * g.nodes[a]["props"]["n_patterns"] for a in att_ids]
            tr.marker.color = pal["anchor"]
            tr.marker.line = dict(color=pal["ink"], width=1)

    def coord(node_id: str) -> np.ndarray | None:
        if node_id in pidx:
            return pc[pidx[node_id]]
        if node_id in aidx:
            return ac[aidx[node_id]]
        return None

    def lines(name: str, pairs: list, color: str, width: float, visible: bool, dash: str | None = None) -> None:
        if pairs:
            x, y, z = _segments(pairs)
            line = dict(color=color, width=width, **({"dash": dash} if dash else {}))
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, visible=visible, line=line, hoverinfo="skip"))

    # memberships (the graph's lilac lines; weak = dashed), drawn first so nodes sit on top
    member_color = _rgba(pal["anchor"], 0.08 if focus else 0.3)
    for name, weak in (("Memberships", False), ("Memberships (weak)", True)):
        pairs = [(pc[pidx[e["source"]]], ac[aidx[e["target"]]]) for e in acts if bool(e["props"].get("weak")) == weak]
        lines(name, pairs, member_color, 1.5, on["activates"], "dash" if weak else None)

    # insights: colour and shape by what they say, exactly as in the graph
    classes = {"Metric higher": ("circle", pal["up"]), "Metric lower": ("circle", pal["down"]), "Correlation change": ("square", pal["cov"])}
    by_class: dict[str, list[int]] = {k: [] for k in classes}
    for i, n in enumerate(pats):
        ins = g.insight(n["id"])
        by_class["Correlation change" if ins.phenomenon_type == "covariance" else "Metric higher" if ins.effect_size > 0 else "Metric lower"].append(
            i
        )
    for name, idx in by_class.items():
        if not idx:
            continue
        symbol, color = classes[name]
        fig.add_trace(
            go.Scatter3d(
                x=pc[idx, 0],
                y=pc[idx, 1],
                z=pc[idx, 2],
                mode="markers",
                name=name,
                opacity=0.35 if focus else 0.9,
                marker=dict(symbol=symbol, size=[3 + 6 * g.insight(pats[i]["id"]).weight for i in idx], color=color, line=dict(width=0)),
                hovertext=[hovers[i] for i in idx],
                customdata=[pats[i]["id"] for i in idx],
                hovertemplate=viz.HOVER,
            )
        )

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
            line=dict(color=_rgba(pal["muted"], 0.2), width=1),
        )
    )

    # the link layers, named like the legend toggles
    lines(
        "Theme links",
        [
            (ac[aidx[e["source"]]], ac[aidx[e["target"]]])
            for e in g.edges
            if e["type"] == "RELATED_TO" and e["source"] in aidx and e["target"] in aidx
        ],
        pal["anchor"],
        4,
        on["latent"],
    )
    for etype, name, color, width, key, dash in (
        ("SPECIALIZES", "Hierarchy", pal["muted"], 2, "lattice", None),
        ("CONTRASTS", "Contrasts", pal["bad"], 2, "contrast", "dash"),
        ("SIBLING", "Siblings", pal["muted"], 1.5, "sibling", "dot"),
    ):
        pairs = [(coord(e["source"]), coord(e["target"])) for e in g.edges if e["type"] == etype and e["source"] in pidx and e["target"] in pidx]
        lines(name, pairs, color, width, on[key], dash)

    if hl:  # the answer, with the graph's markers: gold seed and path, ink evidence ring, red cross-segment ring
        by_id = {e["id"]: e for e in g.edges}
        path = [(coord(by_id[i]["source"]), coord(by_id[i]["target"])) for i in hl.get("edges", []) if i in by_id]
        lines("Answer path", [(a, b) for a, b in path if a is not None and b is not None], pal["path"], 7, True)
        for key, name, color, size, width in (
            ("evidence", "Evidence", pal["ink"], 8, 2),
            ("transversal_only", "Other segment", pal["bad"], 10, 3),
            ("seeds", "Seed", pal["path"], 9, 4),
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
                        marker=dict(symbol="circle-open", size=size, color=color, line=dict(color=color, width=width)),
                        hovertext=[hovers[pidx[i]] for i in ids],
                        customdata=ids,
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
                    name="Themes visited",
                    marker=dict(symbol="circle-open", size=16, color=pal["path"], line=dict(color=pal["path"], width=4)),
                    hoverinfo="skip",
                )
            )

    fig.update_layout(  # no title and no legend of its own: the UI's shared legend strip explains both views
        template="plotly_dark" if _is_dark(pal["bg"]) else "plotly",
        title=None,
        showlegend=False,
        paper_bgcolor=pal["bg"],
        plot_bgcolor=pal["bg"],
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="sig-sphere",
    )
    return fig


def sphere_html(
    engine: Engine,
    *,
    dataset: str | None = None,
    highlight: dict | None = None,
    palette: dict[str, str] | None = None,
    layers: dict[str, bool] | None = None,
    plotly_src: str = PLOTLY_LOCAL,
) -> str:
    """Full HTML page; ``plotly_src`` = local vendored path (served) or ``"cdn"``."""
    fig = sphere_figure(engine, dataset=dataset, highlight=highlight, palette=palette, layers=layers)
    html = fig.to_html(
        full_html=True, include_plotlyjs=plotly_src, default_width="100%", default_height="100%", config={"displaylogo": False, "responsive": True}
    )
    bg = {**DEFAULT_PALETTE, **(palette or {})}["bg"]
    css = f"<style>html, body {{ margin: 0; height: 100%; overflow: hidden; background-color: {bg}; }}</style>"
    return html.replace("<head>", f"<head>\n{css}", 1)


def export_sphere(engine: Engine, output: Path | None = None, dataset: str | None = None) -> Path:
    """Standalone copy like lac's ``ontology_sphere.html`` (plotly from CDN, opens from disk)."""
    out = output or engine.ws.root / "graph" / "sphere.html"
    viz.save_html(sphere_figure(engine, dataset=dataset), out)
    return out


def plotly_js_path() -> Path:
    import plotly

    return Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
