"""3D sphere visualisation: chunks + L0 concepts + optional ACTIVATES edges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objs as go
import seaborn as sns
from prosphera.projector import Projector

HOVER = "%{hovertext}<extra></extra>"
BG = "#0f172a"  # Deep slate (modern, soft dark mode)

STYLE = {
    "concept_color": "#f8fafc",  # Crisp white — concepts stand out as anchors
    "concept_size": 6,
    "chunk_size": 4,
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


def _chunk_colors(labels: list[str]) -> list[str]:
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("Set3", n_colors=len(unique_labels))
    color_map = {label: f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})" for label, (r, g, b) in zip(unique_labels, palette)}
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
    """Joint prosphera projection of chunk + concept embeddings."""

    def _build_figure(
        self,
        chunk_coords: np.ndarray,
        concept_coords: np.ndarray,
        activations: list[dict[str, Any]],
        *,
        chunk_labels: list[str],
        chunk_hovertext: list[str],
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
                x=chunk_coords[:, 0],
                y=chunk_coords[:, 1],
                z=chunk_coords[:, 2],
                mode="markers",
                name="Chunks",
                opacity=STYLE["chunk_opacity"],
                marker=dict(
                    size=STYLE["chunk_size"],
                    color=_chunk_colors(chunk_labels),
                    line=dict(width=0),
                ),
                hovertext=chunk_hovertext,
                hovertemplate=HOVER,
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
