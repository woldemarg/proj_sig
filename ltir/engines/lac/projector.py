"""3D sphere visualisation: L0 concepts + optional ACTIVATES edges on a prosphera sphere.

The chunk points are drawn by the caller (``ltir/sphere.py``: one trace per latent anchor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objs as go
from prosphera.projector import Projector

HOVER = "%{hovertext}<extra></extra>"
BG = "#0f172a"  # Deep slate (modern, soft dark mode)

STYLE = {
    "concept_color": "#f8fafc",  # Crisp white — concepts stand out as anchors
    "concept_size": 6,
    "chunk_opacity": 0.9,
    "edge_color": "rgba(255, 255, 255, 0.08)",
    "edge_width": 0.5,
    "axis_color": "rgba(148, 163, 184, 0.35)",
}


def save_html(fig: go.Figure, filepath: Path | str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_str = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100%",
    )
    css = f"<style>body {{ margin: 0; overflow: hidden; background-color: {BG}; }}</style>"
    path.write_text(html_str.replace("<head>", f"<head>\n{css}", 1), encoding="utf-8")


# ColorBrewer "Set3", the 12 colours lac took from seaborn.color_palette("Set3"); cycled beyond 12 labels like seaborn
SET3 = (
    "rgb(141, 211, 199)",
    "rgb(255, 255, 179)",
    "rgb(190, 186, 218)",
    "rgb(251, 128, 114)",
    "rgb(128, 177, 211)",
    "rgb(253, 180, 98)",
    "rgb(179, 222, 105)",
    "rgb(252, 205, 229)",
    "rgb(217, 217, 217)",
    "rgb(188, 128, 189)",
    "rgb(204, 235, 197)",
    "rgb(255, 237, 111)",
)


def _chunk_colors(labels: list[str]) -> list[str]:
    """One Set3 colour per distinct label, assigned in sorted label order."""
    color_map = {label: SET3[i % len(SET3)] for i, label in enumerate(sorted(set(labels)))}
    return [color_map[label] for label in labels]


def _edge_segments(
    chunk_coords: np.ndarray,
    concept_coords: np.ndarray,
    activations: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    xs, ys, zs = [], [], []
    for edge in activations:
        c = chunk_coords[edge["chunk_id"]]
        p = concept_coords[edge["concept_id"]]
        xs.extend([c[0], p[0], None])
        ys.extend([c[1], p[1], None])
        zs.extend([c[2], p[2], None])
    return xs, ys, zs


class OntologyProjector(Projector):
    """Joint prosphera projection of chunk + concept embeddings (chunk coordinates feed the edges)."""

    def _build_figure(
        self,
        chunk_coords: np.ndarray,
        concept_coords: np.ndarray,
        activations: list[dict[str, Any]],
        *,
        concept_hovertext: list[str],
        draw_edges: bool,
    ) -> go.Figure:
        fig = go.Figure()

        if draw_edges:
            ex, ey, ez = _edge_segments(chunk_coords, concept_coords, activations)
            if ex:
                fig.add_trace(
                    go.Scatter3d(
                        x=ex,
                        y=ey,
                        z=ez,
                        mode="lines",
                        name="ACTIVATES",
                        line=dict(color=STYLE["edge_color"], width=STYLE["edge_width"]),
                        hoverinfo="skip",
                    )
                )

        fig.add_trace(
            go.Scatter3d(
                x=concept_coords[:, 0],
                y=concept_coords[:, 1],
                z=concept_coords[:, 2],
                mode="markers",
                name="Concepts (L0)",
                marker=dict(
                    size=STYLE["concept_size"],
                    symbol="circle",
                    color=STYLE["concept_color"],
                ),
                hovertext=concept_hovertext,
                hovertemplate=HOVER,
            )
        )

        for axis in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            fig.add_trace(
                go.Scatter3d(
                    x=[0, axis[0]],
                    y=[0, axis[1]],
                    z=[0, axis[2]],
                    mode="lines",
                    line=dict(color=STYLE["axis_color"], width=0.5),
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            margin=dict(l=0, r=0, b=0, t=60),
            title=dict(
                text=("<b>Latent Ontology Manifold</b><br><sup>Document distribution across discovered semantic concepts</sup>"),
                font=dict(size=22, color="#f8fafc", family="Inter, system-ui, sans-serif"),
                x=0.02,
                y=0.96,
            ),
            legend=dict(
                title=dict(text="Knowledge Domains", font=dict(color="#94a3b8")),
                bgcolor="rgba(15, 23, 42, 0.7)",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(color="#f8fafc", size=12),
                yanchor="top",
                y=0.9,
                xanchor="left",
                x=0.02,
                itemsizing="constant",
            ),
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1, 1]),
                yaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1, 1]),
                zaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1, 1]),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.1, y=1.1, z=1.1)),
            ),
        )
        return fig
