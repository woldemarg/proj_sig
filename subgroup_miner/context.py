"""A context layer: the points of another type counted on the target's H3 grid and read as one forced band per layer
(``context_<name>_band``), and the within-parent deal of those counts the cross-type null draws from
(docs/02_discovery.md §2.9; docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md §6).

Pure functions over arrays; the statistics (the median test of a draw, the permutation p) live in ``discovery``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import h3
import numpy as np

NONE_NEARBY = "none nearby"  # a cell whose disk holds no point of the layer; two words, as NO_CLUSTER (question._values)
BANDS = 4  # quarters of the positive disk sums: the labels of apply_bins
BAND_WORDS = {"q1": "the fewest", "q2": "fewer than the middle", "q3": "more than the middle", "q4": "the most"}


@dataclass
class ContextSource:
    """The located points of one layer, as ``load_dataset`` reads them."""

    name: str
    lat: np.ndarray
    lon: np.ndarray


@dataclass
class ContextLayer:
    """A layer on the target's grid, in memory only (rebuilt from its file by every read, never stored): its counts over
    every child of every parent a target cell's disk touches — so a deal within a parent can move a count into a cell
    without target points —, each such cell's parent, and each target cell's disk over them (CSR, built once: a draw
    only re-sums it)."""

    name: str
    counts: np.ndarray  # the layer's points per cell
    parents: np.ndarray  # per cell, its parent one resolution coarser (int codes): the deal's unit
    disk_ptr: np.ndarray  # CSR: target cell row -> positions of its disk among the cells
    disk_idx: np.ndarray
    edges: np.ndarray  # the quarter edges of the positive disk sums, as observed: every draw is read by them

    @property
    def column(self) -> str:
        return f"context_{self.name}_band"


def context_layer(source: ContextSource, target_cells: np.ndarray, resolution: int, ring: int) -> ContextLayer:
    """``source`` counted at ``resolution`` over the children of every parent that a target cell's ``ring``-disk touches."""
    disks = [h3.grid_disk(c, ring) for c in target_cells]
    parent = (lambda c: h3.cell_to_parent(c, resolution - 1)) if resolution else (lambda c: c)
    parents = {parent(c) for disk in disks for c in disk}
    cells = sorted({ch for p in parents for ch in (h3.cell_to_children(p, resolution) if resolution else [p])})
    pos = {c: i for i, c in enumerate(cells)}
    ok = np.isfinite(source.lat) & np.isfinite(source.lon)
    counts = np.zeros(len(cells))
    for c, n in Counter(h3.latlng_to_cell(a, b, resolution) for a, b in zip(source.lat[ok], source.lon[ok])).items():
        if c in pos:
            counts[pos[c]] = n
    codes = np.unique([parent(c) for c in cells], return_inverse=True)[1]
    ptr = np.cumsum([0, *map(len, disks)])
    idx = np.array([pos[c] for disk in disks for c in disk], dtype=np.int64)
    layer = ContextLayer(source.name, counts, codes, ptr, idx, np.zeros(0))
    layer.edges = fit_edges(disk_sums(layer))
    return layer


def disk_sums(layer: ContextLayer, counts: np.ndarray | None = None) -> np.ndarray:
    """Per target cell, the layer's points in its disk (absent cells count 0); ``counts`` stands in for the layer's own."""
    counts = layer.counts if counts is None else counts
    return np.add.reduceat(counts[layer.disk_idx], layer.disk_ptr[:-1])  # every disk holds its own cell: no empty segment


def fit_edges(sums: np.ndarray) -> np.ndarray:
    """The quarter edges of the positive sums (fewer where quantiles coincide); empty without a positive sum."""
    positive = sums[sums > 0]
    return np.unique(np.quantile(positive, np.linspace(0, 1, BANDS + 1))) if len(positive) else np.zeros(0)


def band(sums: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``q1..qk`` of each positive sum by ``edges`` (right-closed, as pd.qcut; a sum past the last edge in the top band),
    ``NONE_NEARBY`` for zero."""
    labels = np.array([f"q{i + 1}" for i in range(max(len(edges) - 1, 1))], dtype=object)
    return np.where(sums > 0, labels[np.searchsorted(edges[1:-1], sums, side="left")], NONE_NEARBY)


def dealt_counts(layer: ContextLayer, rng: np.random.Generator) -> np.ndarray:
    """The layer's counts dealt anew among the cells of each parent: every parent keeps its total."""
    order = np.argsort(layer.parents, kind="stable")
    out = np.empty_like(layer.counts)
    out[order] = layer.counts[np.lexsort((rng.random(len(order)), layer.parents))]  # both run parent by parent
    return out


def dealt_band(layer: ContextLayer, rng: np.random.Generator) -> np.ndarray:
    """One draw of the null: every target cell's band after a deal, read by the observed edges."""
    return band(disk_sums(layer, dealt_counts(layer, rng)), layer.edges)


def context_bands(
    sources: Sequence[ContextSource], cells: np.ndarray, resolution: int, ring: int
) -> tuple[dict[str, np.ndarray], dict[str, ContextLayer], dict[str, str], list[str]]:
    """Each layer of ``sources`` on the target's ``cells`` -> (its band column, the layers, their glossary, warnings); a
    layer that leaves one band at this resolution (too sparse) is left out with a warning."""
    columns, layers, words, warnings = {}, {}, {}, []
    for source in sources:
        layer = context_layer(source, cells, resolution, ring)
        bands = band(disk_sums(layer), layer.edges)
        if len(np.unique(bands)) < 2:
            warnings.append(f"geo: context layer {source.name!r} gives one band at resolution {resolution} (too sparse): not added")
            continue
        columns[layer.column], layers[source.name] = bands, layer
        words |= glossary(layer, ring)
    return columns, layers, words, warnings


def glossary(layer: ContextLayer, ring: int) -> dict[str, str]:
    """The plain English of the layer's column and values (``SpatialContext.glossary``)."""
    name = layer.name.replace("_", " ")
    around = "itself only" if ring == 0 else "its six neighbours" if ring == 1 else f"its neighbours within {ring} rings"
    words = {layer.column: f"number of {name} points in the cell and {around}, in quarters of the cells that have any"}
    words[f"{layer.column}={NONE_NEARBY}"] = f"no {name} point in the cell or {around}"
    for q, how in BAND_WORDS.items():
        words[f"{layer.column}={q}"] = f"the cell and {around} hold {how} {name} points (of the cells that have any)"
    return words
