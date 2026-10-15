"""The geo benchmark (docs/10_verification.md §10.3): point datasets with a planted effect, ingested with the geo option on
scratch workspaces (no LLM), scored against the truth they were generated from.

    python scripts/geo_benchmark.py [--scenarios S-regional,S-local,...] [--seed 1] [--k 5] [--encoder hashing|model|both]
                                    [--label baseline]

Every scenario raises the events' ``load`` by ``LIFT`` on its planted part (``load`` in a table, ``load_median`` in cells).
A focus is a ring-3 disk of ``FINE`` cells (37 cells): the smallest disk above ``MIN_SUPPORT_ROWS`` (30), the floor below
which no set of cells is an insight. Three foci fit in one ``REGION`` cell, three resolutions coarser than ``FINE``. Each
focus also raises its own signature metric, so that it is its own insight. A part is found when the patterns with an upward
shift of one of its planted metrics that draw at least half their events from it hold at least ``FOUND`` of its events.

    S-regional  three foci in one region cell, centres 10-11 cells apart (the nearest cells of two foci 4 apart), so no
                CO_LOCATED links them
    S-control   three foci in three different region cells
    S-local     one focus, uploaded at FINE and 7; the rest of its region is ordinary
    S-levels    a broad zone (the eastern half of a large area) in one file, uploaded at 7 and 5 with its table level
    S-parts     the large area as three files split by longitude, the split lines cutting region cells; the zone its
                northern half, across both splits
    S-size      about 10⁴ fine cells in three datasets of ten foci each

Metrics, per encoder (the hashing double; the bundled model when present):
    found        per file and planted part (a focus or the zone): the patterns that draw at least half their events from
                 it, the share of its events they hold together (coverage) and whether that reaches FOUND
    recall@k     the walk from each found focus's best pattern (seed forced, as in attractor_benchmark) to the other planted
                 foci (an unfound one is a miss): at k, anywhere ("reached"), and the mean rank of those reached
    shared       pairs of distinct best patterns (of different foci, levels or files) that activate a common non-weak attractor
    cross_coloc  CO_LOCATED edges between datasets (derive_spatial links patterns of one resolution from any datasets)
    scale        the scale label of each best geo pattern (scale_chain): reg(ional), dep(endent), nfc (not found coarser),
                 ins(ufficient)
    size         per batch the processing and graph-compile seconds; the snapshot's nodes and edges by type; the walk's
                 visited nodes per seed; the evidence prompt's length for one question

The result goes to ``.scratch/results/geo_benchmark_<label>.json``, a table to stdout. The same seed writes the same files.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the packages live one level up

from graph_query_engine.seeds import SeedMatch  # noqa: E402
from graph_query_engine.traversal import traverse  # noqa: E402
from graph_query_engine.vertical import scale_chain  # noqa: E402
from insight_graph_service.core.engine import Engine  # noqa: E402

FINE, REGION = 8, 5
RADIUS = 3  # a focus: the ring-3 disk, 37 cells
LIFT = 4.0  # load is N(10, 2) everywhere, +LIFT on the planted events
CENTRE = (48.45, 35.3)  # lat, lon of the focus scenarios' region cell
BROAD = (47.0, 33.0, 2.4, 3.4)  # lat0, lon0, dlat, dlon of S-levels and S-parts: about 270 cells at resolution 5
SIZE_BOX = (48.0, 34.6, 0.8, 1.2)  # about 10⁴ cells at FINE
SCENARIOS = ("S-regional", "S-control", "S-local", "S-levels", "S-parts", "S-size")
QUESTION = "Where is load higher?"
FOUND = 0.5  # share of a part's events its patterns must hold for the part to count as found
SIGNATURES = ("wait", "cost", "size")  # focus i also raises SIGNATURES[i % 3]: its own phenomenon


@dataclass
class Upload:
    name: str
    frame: pd.DataFrame
    geo: str | None  # the geo option (levels as @r1,r2); None ingests the table
    parts: dict[str, np.ndarray]  # planted part (a focus or "zone") -> mask over the frame's rows
    targets: dict[str, set[str]] = field(default_factory=dict)  # part -> its planted metrics besides load
    table: bool = False  # the table level too (its point columns left out)


@dataclass
class Scenario:
    name: str
    uploads: list[Upload]
    truth: dict[str, Any] = field(default_factory=dict)  # facts of the generated geometry


def _events(
    rng: np.random.Generator,
    lat: np.ndarray,
    lon: np.ndarray,
    planted: np.ndarray,
    box: tuple[float, ...],
    own: np.ndarray | None = None,
    grid: tuple[int, int] = (6, 6),
) -> pd.DataFrame:
    """Event rows: the point, a district (a ``grid`` of rows × columns over ``box``), a noise category and date, ``load``
    (+LIFT where planted) and the ``SIGNATURES`` (N(10, 2); +LIFT on ``SIGNATURES[own % 3]`` where ``own`` >= 0)."""
    frame = {
        "lat": lat.round(6),
        "lon": lon.round(6),
        "district": _district(lat, lon, box, grid),
        "kind": rng.choice(["a", "b", "c"], size=len(lat), p=[0.5, 0.3, 0.2]),
        "when": np.datetime64("2026-08-01") + rng.integers(0, 31, len(lat)).astype("timedelta64[D]"),
        "load": (rng.normal(10.0, 2.0, len(lat)) + LIFT * planted).round(2),
    }
    own = np.full(len(lat), -1) if own is None else own
    for i, name in enumerate(SIGNATURES):
        frame[name] = (rng.normal(10.0, 2.0, len(lat)) + LIFT * ((own >= 0) & (own % 3 == i))).round(2)
    return pd.DataFrame(frame)


def _district(lat: np.ndarray, lon: np.ndarray, box: tuple[float, ...], grid: tuple[int, int]) -> list[str]:
    lat0, lon0, dlat, dlon = box
    row = np.clip(((lat - lat0) / dlat * grid[0]).astype(int), 0, grid[0] - 1)
    col = np.clip(((lon - lon0) / dlon * grid[1]).astype(int), 0, grid[1] - 1)
    return [f"d{r}{c}" for r, c in zip(row, col, strict=True)]


def _uniform(rng: np.random.Generator, n: int, box: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    lat0, lon0, dlat, dlon = box
    return rng.uniform(lat0, lat0 + dlat, n), rng.uniform(lon0, lon0 + dlon, n)


def _in_cells(rng: np.random.Generator, cells: list[str], per_cell: int) -> tuple[np.ndarray, np.ndarray]:
    """``per_cell`` points inside each cell (rejection around its centre)."""
    lat, lon = [], []
    for cell in cells:
        clat, clon = h3.cell_to_latlng(cell)
        got = 0
        while got < per_cell:
            a, b = clat + rng.uniform(-0.004, 0.004), clon + rng.uniform(-0.006, 0.006)
            if h3.latlng_to_cell(a, b, FINE) == cell:
                lat.append(a)
                lon.append(b)
                got += 1
    return np.asarray(lat), np.asarray(lon)


def _box_of(cells: list[str]) -> tuple[float, float, float, float]:
    pts = np.array([p for c in cells for p in h3.cell_to_boundary(c)])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))


def _fits(region: str, box: tuple[float, ...]) -> list[str]:
    """Fine cells whose focus disk lies inside ``region`` and inside one district, nearest the region's centre first."""
    kids = set(h3.cell_to_children(region, FINE))
    centre = h3.cell_to_center_child(region, FINE)
    fits = []
    for c in sorted(kids):
        disk = h3.grid_disk(c, RADIUS)
        lat, lon = np.array([h3.cell_to_latlng(d) for d in disk]).T
        if set(disk) <= kids and len(set(_district(lat, lon, box, (6, 6)))) == 1:
            fits.append(c)
    return sorted(fits, key=lambda c: (h3.grid_distance(c, centre), c))


def _spread(cells: list[str], n: int) -> list[str]:
    """The ``n`` cells with the largest smallest pairwise grid distance (first such set in sorted order)."""
    dist = np.array([[h3.grid_distance(a, b) for b in cells] for a in cells])
    return [
        cells[i] for i in max(itertools.combinations(range(len(cells)), n), key=lambda t: min(dist[a, b] for a, b in itertools.combinations(t, 2)))
    ]


def _foci_upload(rng: np.random.Generator, name: str, centres: list[str], box: tuple[float, ...], background: int) -> Upload:
    """Uniform background events plus 25 events per cell of each focus, as one geo upload at FINE. Every focus raises load,
    and focus i also its own signature metric."""
    blat, blon = _uniform(rng, background, box)
    foci = [h3.grid_disk(c, RADIUS) for c in centres]
    points = [_in_cells(rng, sorted(cells), 25) for cells in foci]
    lat, lon = np.concatenate([blat, *(p[0] for p in points)]), np.concatenate([blon, *(p[1] for p in points)])
    sizes = [background, *(len(p[0]) for p in points)]
    owner = np.repeat(np.arange(len(sizes)) - 1, sizes)  # -1: background, i: focus i
    parts = {f"focus{i}": owner == i for i in range(len(foci))}
    targets = {f"focus{i}": {SIGNATURES[i % 3], f"{SIGNATURES[i % 3]}_median"} for i in range(len(foci))}
    return Upload(name, _events(rng, lat, lon, owner >= 0, box, owner), f"lat,lon@{FINE}", parts, targets)


def generate(name: str, seed: int) -> Scenario:
    """One scenario, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    region = h3.latlng_to_cell(*CENTRE, REGION)
    area = _box_of(h3.grid_disk(region, 1))
    if name == "S-regional":
        centres = _spread(_fits(region, area), 3)
        return Scenario(name, [_foci_upload(rng, "s_regional.csv", centres, area, 8000)], {"region": region})
    if name == "S-control":
        others = h3.grid_ring(region, 1)[::2]  # three region cells, no two adjacent
        centres = [_fits(r, area)[0] for r in others]
        return Scenario(name, [_foci_upload(rng, "s_control.csv", centres, area, 8000)])
    if name == "S-local":
        centres = _fits(region, area)[:1]
        return Scenario(name, [replace(_foci_upload(rng, "s_local.csv", centres, area, 8000), geo=f"lat,lon@{FINE},7")])
    if name in ("S-levels", "S-parts"):
        lat, lon = _uniform(rng, 80000, BROAD)
        lat0, lon0, dlat, dlon = BROAD
        if name == "S-levels":
            zone = lon > lon0 + dlon / 2
            frame = _events(rng, lat, lon, zone, BROAD, grid=(1, 2))  # two districts: west, east
            uploads = [Upload("s_levels.csv", frame, "lat,lon@7,5", {"zone": zone}, table=True)]
            return Scenario(name, uploads)
        cuts = lon0 + dlon * np.array([1 / 3, 2 / 3])
        zone = lat > lat0 + dlat / 2  # the northern half, across both cuts
        frame = _events(rng, lat, lon, zone, BROAD, grid=(2, 1))  # two districts: south, north
        part = np.searchsorted(cuts, frame["lon"].to_numpy())
        uploads = [Upload(f"s_parts_{i}.csv", frame[part == i].reset_index(drop=True), "lat,lon@7", {"zone": zone[part == i]}) for i in range(3)]
        regions = [{h3.latlng_to_cell(a, b, REGION) for a, b in zip(frame["lat"][part == i], frame["lon"][part == i], strict=True)} for i in range(3)]
        return Scenario(name, uploads, {"shared_regions": len((regions[0] & regions[1]) | (regions[1] & regions[2]))})
    if name == "S-size":
        uploads = []
        for i in range(3):
            pool = sorted({h3.latlng_to_cell(a, b, FINE) for a, b in zip(*_uniform(rng, 2000, SIZE_BOX), strict=True)})
            cells = [pool[rng.integers(len(pool))]]
            while len(cells) < 10:  # farthest-point: foci spread over the area
                cells.append(max(pool, key=lambda c: min(h3.grid_distance(c, o) for o in cells)))
            uploads.append(_foci_upload(rng, f"s_size_{i}.csv", cells, SIZE_BOX, 20000))
        return Scenario(name, uploads)
    raise ValueError(f"unknown scenario {name}")


def _patterns(engine: Engine, record: dict[str, Any], upload: Upload) -> dict[str, tuple[dict[str, Any], np.ndarray]]:
    """pattern id -> (record, mask over the upload's rows: the events inside its extent)."""
    ds = record["dataset_id"]
    covers = engine.ws.load_covers(ds)
    geo = engine.ws.load_geo(ds)
    out = {}
    for p in engine.ws.patterns():
        if p["dataset_id"] == ds:
            rows = covers[p["id"]]
            out[p["id"]] = (p, np.isin(geo["point_cell"], rows) if geo else np.isin(np.arange(len(upload.frame)), rows))
    return out


def _best(patterns: dict[str, tuple[dict[str, Any], np.ndarray]], part: np.ndarray, targets: set[str]) -> dict[str, Any]:
    """The part as found: the patterns with an upward shift (target or secondary) of one of its planted metrics, load and
    ``targets``, that draw at least half their events from it. ``coverage``: the share of the part's events inside any of
    them; ``pattern``: the one holding most."""
    inside = []
    for pid, (p, mask) in patterns.items():
        hits = int((mask & part).sum())
        raises = any(sh["metric"] in {"load", "load_median", *targets} and sh["robust_z"] > 0 for sh in p["shifts"])
        if raises and hits >= 0.5 * mask.sum():
            inside.append((hits, pid, mask))
    if not inside:
        return {"pattern": None, "patterns": 0, "coverage": 0.0, "found": False}
    union = np.logical_or.reduce([m for _, _, m in inside])
    coverage = float((union & part).sum() / part.sum())
    return {
        "pattern": max(inside, key=lambda x: (x[0], x[1]))[1],
        "patterns": len(inside),
        "coverage": round(coverage, 3),
        "found": coverage >= FOUND,
    }


def _recall(engine: Engine, foci: list[str | None], k: int) -> dict[str, Any]:
    """The walk from each found focus (its best pattern; None when not found) to the other planted foci: recall at k and
    the share reached, an unfound focus counting as a miss and one sharing the seed's pattern as a hit; the mean rank of
    those reached."""
    state = engine.committed()
    wide = replace(engine.settings.query, max_retrieved=len(state.graph.of_kind("Pattern")))
    rows, ranks = [], []
    for i, seed in enumerate(foci):
        if seed is None:
            continue
        others = [f for j, f in enumerate(foci) if j != i]
        ranked = [r.node_id for r in traverse(state.graph, [SeedMatch(seed, 1.0, {})], wide).patterns if r.node_id != seed]
        ranks += [ranked.index(o) + 1 for o in others if o in ranked]
        rows.append(
            {
                "recall": sum(o == seed or o in ranked[:k] for o in others) / len(others),
                "reached": sum(o == seed or o in ranked for o in others) / len(others),
            }
        )
    mean = {m: round(float(np.mean([r[m] for r in rows])), 3) if rows else None for m in ("recall", "reached")}
    return {"cases": len(rows), **mean, "rank": round(float(np.mean(ranks)), 3) if ranks else None}


def _graph(engine: Engine, best: list[str]) -> dict[str, Any]:
    """Attractors shared by the best patterns, CO_LOCATED across datasets, and the size of the graph, the walk and the prompt."""
    state = engine.committed()
    g = state.graph
    members: dict[str, set[str]] = {}
    for e in g.edges:
        if e["type"] == "ACTIVATES" and not e["props"]["weak"]:
            members.setdefault(e["source"], set()).add(e["target"])
    dataset = {n["id"]: n["props"]["dataset_id"] for n in g.of_kind("Pattern")}
    pairs = list(itertools.combinations(sorted(set(best)), 2))
    walks = [traverse(g, [SeedMatch(p, 1.0, {})], engine.settings.query).visited_count for p in sorted(set(best))]
    return {
        "shared": f"{sum(bool(members.get(a, set()) & members.get(b, set())) for a, b in pairs)}/{len(pairs)}",
        "cross_coloc": sum(e["type"] == "CO_LOCATED" and dataset[e["source"]] != dataset[e["target"]] for e in g.edges),
        "nodes": {kind: len(g.of_kind(kind)) for kind in ("Pattern", "Attractor")},
        "edges": dict(sorted(g.snapshot["stats"]["edge_counts"].items())),
        "visited_per_seed": round(float(np.mean(walks)), 1) if walks else None,
        "prompt_chars": len(engine.evidence(QUESTION).evidence_prompt),
        "scale": {c.pattern_id: c.scale for c in scale_chain(g, sorted(set(best)))},
    }


def run_scenario(scenario: Scenario, root: Path, engine: Engine, *, k: int) -> dict[str, Any]:
    """Ingest the scenario's uploads (written to ``root``) into ``engine``'s empty workspace and score them."""
    files, best = [], []
    for upload in scenario.uploads:
        path = root / upload.name
        upload.frame.to_csv(path, index=False)
        with path.open("rb") as stream:
            levels = engine.upload(upload.name, stream, geo=upload.geo, table=upload.table)
        for level in levels:
            record = engine.process(level["batch_id"])
            if record["status"] == "FAILED" and record["error"]["code"] == "no_viable_insights" and level is not levels[0]:
                files.append({"file": upload.name, "level": record["level"], "found": {}, "seconds": 0.0, "status": "no_viable_insights"})
                continue  # a coarser level mined and found nothing: a level, not a failure of the run
            if record["status"] != "READY":
                raise SystemExit(f"{scenario.name} {upload.name} {record['level']}: {record['status']} {record.get('error')}")
            patterns = _patterns(engine, record, upload)
            found = {name: _best(patterns, part, upload.targets.get(name, set())) for name, part in upload.parts.items()}
            best += [f["pattern"] for f in found.values() if f["found"]]
            metrics = record["metrics"]
            files.append(
                {
                    "file": upload.name,
                    "level": record["level"],
                    "cells": record["profile"].get("geo", {}).get("cells"),
                    "patterns": len(patterns),
                    "found": found,
                    "seconds": round(metrics["processing_duration_s"], 2),
                    "graph_seconds": round(metrics["timings"].get("graph_s", 0.0), 2),
                }
            )
    foci = [f["pattern"] if f["found"] else None for f in files[0]["found"].values()] if len(files) == 1 else []
    result = {"scenario": scenario.name, "truth": scenario.truth, "files": files, "recall": _recall(engine, foci, k) if len(foci) > 1 else None}
    return result | _graph(engine, best)


def format_table(results: list[dict[str, Any]]) -> str:
    head = ("encoder", "scenario", "found", "recall@k", "reached", "rank", "shared", "x-coloc", "scale", "visited", "prompt", "sec")
    short = {"regional": "reg", "scale-dependent": "dep", "not_found_coarser": "nfc", "insufficient": "ins"}
    lines = [" ".join(f"{h:>9}" for h in head)]
    for r in results:
        parts = [f for file in r["files"] for f in file["found"].values()]
        rec = r["recall"] or {}
        row = (
            r["encoder"],
            r["scenario"],
            f"{sum(f['found'] for f in parts)}/{len(parts)}",
            *(rec.get(m) for m in ("recall", "reached", "rank")),
            r["shared"],
            r["cross_coloc"],
            ",".join(f"{short[k]}{n}" for k, n in sorted(Counter(r["scale"].values()).items())) or None,
            r["visited_per_seed"],
            r["prompt_chars"],
            round(sum(f["seconds"] for f in r["files"]), 1),
        )
        lines.append(" ".join(f"{'-' if v is None else v:>9}" for v in row))
    return "\n".join(lines)


def main() -> int:
    from scratch import build_engine, save_result, scratch_dir

    from attractor_topology.encoder import model_folder
    from insight_graph_service.core.model_store import problems
    from insight_graph_service.core.settings import load_settings
    from tests.doubles import HashingEmbedder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--encoder", choices=("hashing", "model", "both"), default="both")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    encoders = ["hashing", "model"] if args.encoder == "both" else [args.encoder]
    if "model" in encoders and problems(model_folder(load_settings().topology.model_dir), hashes=False):
        print("the embedding model is missing: the model run is skipped")
        encoders.remove("model")
    results = []
    for encoder in encoders:
        embedder = HashingEmbedder() if encoder == "hashing" else None
        for name in args.scenarios.split(","):
            t0 = time.perf_counter()
            root = scratch_dir(f"geo_benchmark/{encoder}/{name}", fresh=True)
            engine = build_engine(root / "ws", embedder=embedder)
            embedder = engine.encoder.embedder  # the model loads once
            results.append({"encoder": encoder, **run_scenario(generate(name, args.seed), root, engine, k=args.k)})
            print(f"{encoder} {name}: {time.perf_counter() - t0:.1f} s", flush=True)
    print(format_table(results))
    print(save_result("geo_benchmark", args.label, {"seed": args.seed, "k": args.k, "results": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
