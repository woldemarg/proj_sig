"""The attractor benchmark (docs/10_verification.md §10.3): the anchor walk over the living attractors against the same
walk over a static clustering of the same insight vectors, on planted datasets with enough cases for statistics.

    python scripts/attractor_benchmark.py [--seeds 201-230] [--per-workspace 5] [--k 5] [--label current]
                                          [--override MIN_ASSIGN_THRESHOLD=0.75 ...]

Each workspace ingests ``--per-workspace`` planted datasets (``generate``: three cross-scope mechanisms out of eight,
each in three scopes that share no condition, one distractor anomaly) as successive batches. A case is a pattern of a
planted mechanism; its analogues are the mechanism's other patterns, in any dataset of the workspace, that share no
condition with it. Rankers:

    transversal   the walk of graph_query_engine.traversal over the engine's attractors (seed forced to the case)
    static        the same snapshot builder and walk with the attractors replaced by k-means centroids of the same vectors
                  (as many clusters as the engine formed; nearest-cluster memberships; the same RELATED_TO and
                  co-membership rules)
    random        the walk over random memberships of the same cluster sizes, co-memberships not compiled: the null
    vector_nn     cosine of the insight vectors (no graph)
    text_nn       cosine of the canonical-document embeddings
    weight_only   every other pattern by its insight weight (no seed, no graph)

The result (``.scratch/results/attractor_benchmark_<label>.json``) holds the per-case scores and the paired comparisons of
``transversal`` against the others on recall@k: a cluster bootstrap over the source datasets (1,000 resamples, seed 42),
Holm over the comparisons. Labels are planted scopes, never measured shifts.
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from experiment import Truth, cosine_rank, label_of, rank_scores
from scratch import build_engine, ingest_ready, parse_overrides, save_result, scratch_dir
from sklearn.cluster import KMeans

import insight_graph_service.core.snapshot as snapshot_module
from attractor_topology.ontology import Attractor
from attractor_topology.vendor.lac import ontology_engine as eng
from graph_query_engine.graph import DualGraph
from graph_query_engine.seeds import SeedMatch
from graph_query_engine.traversal import traverse
from insight_graph_service.core.engine import Engine
from insight_graph_service.core.snapshot import build_snapshot

DIMENSIONS = {
    "region": ["EU", "US", "APAC", "LATAM"],
    "category": ["laptops", "phones", "tablets", "accessories"],
    "channel": ["online", "retail", "partner"],
    "segment": ["consumer", "smb", "enterprise"],
}
# driver column -> (shift up, shift down); the base relations below turn a driver shift into an outcome shift
SHIFTS = {"discount": (9.0, -6.0), "delivery_days": (3.0, -1.2), "wait_minutes": (15.0, -8.0), "price_index": (15.0, -10.0)}


def draw_scopes(rng: np.random.Generator, mechanisms: int, per: int, tries: int = 20000) -> list[list[dict[str, str]]]:
    """``mechanisms`` lists of ``per`` two-condition scopes: no shared condition within a mechanism, no scope twice."""
    pairs = list(itertools.combinations(DIMENSIONS, 2))
    for _ in range(tries):
        scopes = []
        for _ in range(mechanisms * per):
            a, b = pairs[rng.integers(len(pairs))]
            scopes.append({a: DIMENSIONS[a][rng.integers(len(DIMENSIONS[a]))], b: DIMENSIONS[b][rng.integers(len(DIMENSIONS[b]))]})
        groups = [scopes[i * per : (i + 1) * per] for i in range(mechanisms)]
        within = all(not (set(x.items()) & set(y.items())) for g in groups for x, y in itertools.combinations(g, 2))
        if within and len({frozenset(x.items()) for x in scopes}) == len(scopes):
            return groups
    raise RuntimeError("no scope draw met the constraints")


def generate(seed: int, n_rows: int = 8000) -> tuple[pd.DataFrame, Truth]:
    """One planted dataset: (the table, mechanism name -> planted scopes). The base relations hold everywhere:
    margin = 26 − 0.6·discount, return_rate = 0.02 + 0.01·delivery, satisfaction = 9 − 0.08·wait, churn = 0.03 + 0.002·price."""
    rng = np.random.default_rng(seed)
    cols = {d: rng.choice(v, size=n_rows) for d, v in DIMENSIONS.items()}
    drivers = {
        "discount": np.clip(rng.normal(10.0, 3.0, n_rows), 0.0, 40.0),
        "delivery_days": 2.0 + rng.gamma(2.0, 0.7, n_rows),
        "wait_minutes": rng.gamma(4.0, 5.0, n_rows),
        "price_index": rng.normal(100.0, 8.0, n_rows),
    }
    groups = draw_scopes(rng, 4, 3)  # three mechanisms, and the distractor's scope from the fourth group
    truth: Truth = {}
    for column, sign, scopes in zip(rng.choice(list(SHIFTS), size=3, replace=False), rng.choice(["up", "down"], size=3), groups[:3], strict=True):
        up, down = SHIFTS[column]
        for scope in scopes:
            mask = np.logical_and.reduce([cols[k] == v for k, v in scope.items()])
            drivers[column][mask] += up if sign == "up" else down
        truth[f"{column}_{sign}"] = scopes
    margin = 26.0 - 0.6 * drivers["discount"] + rng.normal(0.0, 3.0, n_rows)
    margin[np.logical_and.reduce([cols[k] == v for k, v in groups[3][0].items()])] += 5.0  # the distractor: a local anomaly with no analogue
    frame = {
        "order_id": np.arange(100000, 100000 + n_rows),
        **{d: c.astype(str) for d, c in cols.items()},
        "payment": rng.choice(["card", "cash", "wallet"], size=n_rows),
        "weekday": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], size=n_rows),
        "discount": drivers["discount"].round(2),
        "margin": margin.round(2),
        "delivery_days": np.clip(drivers["delivery_days"], 0.5, None).round(1),
        "return_rate": np.clip(0.02 + 0.01 * drivers["delivery_days"] + rng.normal(0.0, 0.008, n_rows), 0.0, 1.0).round(4),
        "wait_minutes": np.clip(drivers["wait_minutes"], 0.0, None).round(1),
        "satisfaction": np.clip(9.0 - 0.08 * drivers["wait_minutes"] + rng.normal(0.0, 0.6, n_rows), 0.0, 10.0).round(2),
        "price_index": drivers["price_index"].round(1),
        "churn_rate": np.clip(0.03 + 0.002 * drivers["price_index"] + rng.normal(0.0, 0.01, n_rows), 0.0, 1.0).round(4),
    }
    return pd.DataFrame(frame), truth


@dataclass
class Case:
    seed: str
    dataset: int
    phenomenon: str
    analogues: set[str]


def build_cases(engine: Engine, truth_by_batch: dict[str, Truth], seed_by_batch: dict[str, int]) -> list[Case]:
    """A pattern belongs to the mechanism of its own dataset whose planted scope its scope contains (none when ambiguous);
    the mechanism's name is the phenomenon across datasets."""
    records = engine.ws.patterns()
    conds = {r["id"]: {(c["attribute"], c["value"]) for c in r["conditions"]} for r in records}
    label: dict[str, str | None] = {}
    for r in records:
        label[r["id"]] = label_of(conds[r["id"]], truth_by_batch[r["batch_id"]])
    batch_of = {r["id"]: r["batch_id"] for r in records}
    cases = []
    for pid, ph in label.items():
        if ph is None:
            continue
        analogues = {q for q, lab in label.items() if lab == ph and q != pid and not (conds[q] & conds[pid])}
        if analogues:
            cases.append(Case(pid, seed_by_batch[batch_of[pid]], ph, analogues))
    return cases


@dataclass
class _Store:
    embeddings: np.ndarray
    concept_ids: list[int]


class StaticOntology:
    """The ontology interface the snapshot builder reads, over fixed centroids: no lifecycle."""

    def __init__(self, centroids: np.ndarray, config: Any) -> None:
        self.store = _Store(centroids / np.linalg.norm(centroids, axis=1, keepdims=True), list(range(len(centroids))))
        self.config = config

    @property
    def attractor_ids(self) -> list[int]:
        return list(self.store.concept_ids)

    def attractors(self) -> list[Attractor]:
        return [Attractor(i, c.copy(), 0, "static", 0) for i, c in zip(self.store.concept_ids, self.store.embeddings, strict=True)]

    def centroid(self, attractor_id: int) -> np.ndarray:
        return self.store.embeddings[attractor_id]

    def topology(self) -> list[dict[str, Any]]:
        return eng.calculate_knn_topology(self.store, self.config)


def static_graph(engine: Engine, assign: np.ndarray, centroids: np.ndarray, *, compile_co_memberships: bool = True) -> DualGraph:
    """The engine's committed graph rebuilt with the attractor layer replaced (``assign[row]``: the cluster of a pattern)."""
    ws, settings = engine.ws, engine.settings
    records, vectors = ws.patterns(), ws.vectors()
    activations = [
        {
            "pattern_id": r["id"],
            "attractor_id": int(assign[r["row_id"]]),
            "engine_weight": None,
            "alignment": None,
            "source": "static",
            "batch_id": r["batch_id"],
            "weak": False,
        }
        for r in records
    ]
    compile_ = snapshot_module._co_memberships
    if not compile_co_memberships:
        snapshot_module._co_memberships = lambda *args, **kwargs: []  # the walk sees only the memberships it is given
    try:
        snapshot = build_snapshot(
            records=records,
            activations=activations,
            vectors=vectors,
            batches={b["batch_id"]: b for b in ws.list_batches()},
            representation=ws.representation() or {},
            ontology=StaticOntology(centroids, settings.topology),
            settings=settings,
        )
    finally:
        snapshot_module._co_memberships = compile_
    return DualGraph(snapshot)


def rank_cases(engine: Engine, cases: list[Case], *, k: int, seed: int) -> list[dict[str, Any]]:
    """Every ranker's scores per case, on the engine's committed state."""
    state = engine.committed()
    g, cfg = state.graph, engine.settings.query
    ids = [n["id"] for n in g.of_kind("Pattern")]
    wide = replace(cfg, max_retrieved=len(ids))
    vectors = engine.ws.vectors().astype(np.float64)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    n_attr = len(state.frame.attractors)
    km = KMeans(n_clusters=n_attr, random_state=seed, n_init=10).fit(unit)
    rng = np.random.default_rng(seed)
    graphs = {
        "transversal": g,
        "static": static_graph(engine, km.labels_, km.cluster_centers_),
        "random": static_graph(engine, rng.permutation(km.labels_), km.cluster_centers_, compile_co_memberships=False),
    }
    weight = {p: g.insight(p).weight for p in ids}
    by_weight = sorted(ids, key=lambda p: -weight[p])
    documents = dict(zip(ids, engine.encoder.embedder.embed([g.canonical_document(p) for p in ids])))  # the naive baseline's
    rankers = {
        name: (lambda s, gr=gr: [r.node_id for r in traverse(gr, [SeedMatch(s, 1.0, {})], wide).patterns if r.node_id != s])
        for name, gr in graphs.items()
    }
    rankers |= {
        "vector_nn": lambda s: cosine_rank(s, state.frame.patterns),
        "text_nn": lambda s: cosine_rank(s, documents),
        "weight_only": lambda s: [p for p in by_weight if p != s],
    }
    rows = []
    for case in cases:
        row: dict[str, Any] = {
            "seed": case.seed,
            "dataset": case.dataset,
            "phenomenon": case.phenomenon,
            "analogues": len(case.analogues),
            "patterns": len(ids),
            "attractors": n_attr,
        }
        for name, fn in rankers.items():
            ranked = fn(case.seed)
            row[name] = rank_scores(ranked, case.analogues, k) | {"reached": sum(1 for p in ranked if p in case.analogues) / len(case.analogues)}
        rows.append(row)
    return rows


def cluster_bootstrap(rows: list[dict[str, Any]], a: str, b: str, metric: str, *, resamples: int = 1000, seed: int = 42) -> dict[str, float]:
    """Mean of metric(a) − metric(b) over the cases, with a percentile 95 % interval and a two-sided p-value from
    resampling the source datasets (the cases of one dataset are not independent)."""
    by: dict[int, list[float]] = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r[a][metric] - r[b][metric])
    datasets = sorted(by)
    rng = np.random.default_rng(seed)
    means = np.array([np.mean([x for d in rng.choice(datasets, size=len(datasets)) for x in by[d]]) for _ in range(resamples)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, max(2 / resamples, 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))))
    return {"delta": float(np.mean([x for v in by.values() for x in v])), "ci_low": float(lo), "ci_high": float(hi), "p": p}


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in input order."""
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def analyse(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Means per ranker and the paired comparisons of the walk against the others, Holm-corrected."""
    rankers = ["transversal", "static", "random", "vector_nn", "text_nn", "weight_only"]
    means = {m: {key: round(float(np.mean([r[m][key] for r in rows])), 3) for key in ("recall", "precision", "mrr", "reached")} for m in rankers}
    comparisons = {other: cluster_bootstrap(rows, "transversal", other, "recall") for other in rankers[1:]}
    for other, adjusted in zip(comparisons, holm([c["p"] for c in comparisons.values()]), strict=True):
        comparisons[other]["p_holm"] = adjusted
        comparisons[other]["mrr"] = cluster_bootstrap(rows, "transversal", other, "mrr")
    return {"k": k, "cases": len(rows), "datasets": len({r["dataset"] for r in rows}), "summary": means, "comparisons": comparisons}


def parse_seeds(text: str) -> list[int]:
    """``201-230`` or ``1,2,3``."""
    if m := re.fullmatch(r"(\d+)-(\d+)", text):
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(s) for s in text.split(",")]


def run_benchmark(
    seeds: list[int], *, per_workspace: int, k: int, overrides: dict[str, Any], scratch: Path, embedder: Any = None
) -> list[dict[str, Any]]:
    """Ingest the planted datasets, ``per_workspace`` consecutive seeds per fresh scratch workspace, and rank every case."""
    rows = []
    for start in range(0, len(seeds), per_workspace):
        group = seeds[start : start + per_workspace]
        root = scratch_dir(f"{scratch}/ws_{group[0]}", fresh=True)
        engine = build_engine(root / "ws", embedder=embedder, **overrides)
        embedder = engine.encoder.embedder
        truth_by_batch: dict[str, Truth] = {}
        seed_by_batch: dict[str, int] = {}
        for seed in group:
            frame, truth = generate(seed)
            csv = root / f"planted_{seed}.csv"
            frame.to_csv(csv, index=False)
            batch = ingest_ready(engine, csv)["batch_id"]
            truth_by_batch[batch], seed_by_batch[batch] = truth, seed
        group_rows = rank_cases(engine, build_cases(engine, truth_by_batch, seed_by_batch), k=k, seed=group[0])
        rows += [{"workspace": group[0], **r} for r in group_rows]
        print(f"workspace {group}: {len(group_rows)} cases, {group_rows[0]['attractors'] if group_rows else 0} attractors", flush=True)
    return rows


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"attractor benchmark — {result['cases']} cases from {result['datasets']} datasets, k={result['k']}",
        f"{'ranker':<12} {'recall@k':>9} {'MRR':>6} {'reached':>8}",
    ]
    lines += [f"{m:<12} {s['recall']:>9.3f} {s['mrr']:>6.3f} {s['reached']:>8.3f}" for m, s in result["summary"].items()]
    lines.append("transversal against (recall@k, cluster bootstrap over datasets, Holm):")
    for other, c in result["comparisons"].items():
        lines.append(f"  {other:<12} {c['delta']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  p_holm {c['p_holm']:.3f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="201-230", help="dataset seeds, a range a-b or a list a,b,c")
    parser.add_argument("--per-workspace", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label", default="current")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a settings field by its upper-case name, e.g. MIN_ASSIGN_THRESHOLD=0.75",
    )
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    rows = run_benchmark(
        parse_seeds(args.seeds), per_workspace=args.per_workspace, k=args.k, overrides=overrides, scratch=Path("attractor_benchmark")
    )
    result = analyse(rows, args.k) | {"overrides": overrides, "per_case": rows}
    print(format_summary(result))
    print(save_result("attractor_benchmark", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
