"""Hypothesis experiment (docs/10_verification.md §10.3): cross-dimensional analogue retrieval.

    python scripts/experiment.py [--k 5] [--label current]


For every pattern S that expresses a planted phenomenon, the *analogues* of S are
the other patterns expressing the same phenomenon whose scope shares no condition
with S. Four retrievers rank all other patterns from S:

    transversal   graph_query_engine.traversal (seed forced to S)
    structural    BFS over all structural edges (SPECIALIZES/GENERALIZES/SIBLING/CONTRASTS)
    text_nn       cosine of canonical-document text embeddings (naive vector RAG)
    vector_nn     cosine of the insight vectors (same space, no attractor graph)

and are scored with recall@k, precision@k (hits over min(k, |analogues|), so a
case with fewer analogues than k can reach 1.0) and MRR of the first analogue.

Phenomenon membership comes from the *planted* ground truth (``insight_graph_service.core.demo.GROUND_TRUTH``):
a pattern belongs to a mechanism when its scope lies inside one of that mechanism's
planted scopes. Labels must not be derived from the measured shifts: those are exactly
what the representation encodes, so shift-based labels would favour vector retrieval
by construction.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the packages live one level up

from graph_query_engine.search import CommittedState  # noqa: E402
from graph_query_engine.seeds import SeedMatch  # noqa: E402
from graph_query_engine.traversal import structural_closure, traverse  # noqa: E402
from insight_contracts import Insight  # noqa: E402
from insight_graph_service.core.demo import GROUND_TRUTH  # noqa: E402
from insight_graph_service.core.engine import Engine  # noqa: E402
from insight_graph_service.core.workspace import utc_now  # noqa: E402

Truth = dict[str, list[dict[str, str]]]  # mechanism -> planted scopes


def planted_scopes() -> Truth:
    """Mechanisms with their planted scopes (only multi-scope mechanisms have analogues)."""
    out: Truth = {}
    for name, spec in GROUND_TRUTH.items():
        scopes = spec.get("scopes") or ([spec["scope"]] if "scope" in spec else [])
        if scopes:
            out[name] = scopes
    return out


@dataclass
class Case:
    seed: str
    phenomenon: str
    analogues: set[str]


def phenomenon_of(ins: Insight, truth: Truth) -> str | None:
    """Planted mechanism whose scope contains the pattern's scope; None if none or ambiguous."""
    conds = {(c.attribute, c.value) for c in ins.conditions}
    hits = [name for name, scopes in truth.items() if any(set(s.items()) <= conds for s in scopes)]
    return hits[0] if len(hits) == 1 else None


def build_cases(state: CommittedState, truth: Truth) -> list[Case]:
    pats = list(state.graph.insights.values())
    labels = {ins.id: phenomenon_of(ins, truth) for ins in pats}
    conds = {ins.id: {c.expr for c in ins.conditions} for ins in pats}
    cases = []
    for ins in pats:
        ph = labels[ins.id]
        if ph is None:
            continue
        analogues = {m for m, lab in labels.items() if lab == ph and m != ins.id and not (conds[m] & conds[ins.id])}
        if analogues:
            cases.append(Case(ins.id, ph, analogues))
    return cases


def _cosine_rank(seed: str, vecs: dict[str, np.ndarray]) -> list[str]:
    q = vecs[seed]
    return [pid for pid, _ in sorted(((p, float(v @ q)) for p, v in vecs.items() if p != seed), key=lambda x: -x[1])]


def _score(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = ranked[:k]
    hits = sum(1 for p in top if p in relevant)
    first = next((i for i, p in enumerate(ranked, start=1) if p in relevant), None)
    return {"recall": hits / len(relevant), "precision": hits / min(k, len(relevant)), "mrr": 1.0 / first if first else 0.0}


def run_experiment(engine: Engine, *, k: int = 5) -> dict[str, Any]:
    state = engine.prepared()  # one commit for the graph, both vector sets and the cases
    g, cfg = state.graph, engine.settings.query
    truth = planted_scopes()
    cases = build_cases(state, truth)
    ids = [n["id"] for n in g.of_kind("Pattern")]
    vec_nn = state.frame.patterns  # the insight representation itself, no attractor graph
    text = {p: state.frame.documents[p] for p in ids}
    wide = replace(cfg, max_retrieved=len(ids))
    weight = {p: g.insight(p).weight for p in ids}

    methods: dict[str, Callable[[str], list[str]]] = {
        "transversal": lambda s: [r.node_id for r in traverse(g, [SeedMatch(s, 1.0, {})], wide).patterns if r.node_id != s],
        "structural": lambda s: [
            p
            for p, _ in sorted(
                ((p, h) for p, h in structural_closure(g, [s], cfg.traversal_max_depth).items() if p != s), key=lambda x: (x[1], -weight[x[0]])
            )
        ],
        "text_nn": lambda s: _cosine_rank(s, text),
        "vector_nn": lambda s: _cosine_rank(s, vec_nn),
    }
    per_case = []
    totals = {m: {"recall": 0.0, "precision": 0.0, "mrr": 0.0} for m in methods}
    for case in cases:
        row: dict[str, Any] = {
            "seed": case.seed,
            "seed_label": g.nodes[case.seed]["label"],
            "phenomenon": case.phenomenon,
            "analogues": sorted(case.analogues),
        }
        for name, fn in methods.items():
            sc = _score(fn(case.seed), case.analogues, k)
            row[name] = sc
            for key, val in sc.items():
                totals[name][key] += val
        per_case.append(row)
    n = max(len(cases), 1)
    summary = {m: {key: round(v / n, 3) for key, v in t.items()} for m, t in totals.items()}
    result = {
        "at": utc_now(),
        "k": k,
        "cases": len(cases),
        "summary": summary,
        "per_case": per_case,
        "labels": "planted ground truth (scope containment)",
        "phenomena": {ph: sum(1 for c in cases if c.phenomenon == ph) for ph in truth},
    }
    return result


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"cross-dimensional analogue retrieval — {result['cases']} seed cases, k={result['k']}  {result['phenomena']}",
        f"{'method':<12} {'recall@k':>9} {'precision@k':>12} {'MRR':>6}",
    ]
    for m, s in result["summary"].items():
        lines.append(f"{m:<12} {s['recall']:>9.3f} {s['precision']:>12.3f} {s['mrr']:>6.3f}")
    return "\n".join(lines)


def main() -> int:
    """The hypothesis benchmark on a scratch workspace holding the synthetic demo; the result is saved under .scratch/results."""
    import argparse

    from scratch import demo_engine, save_result

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    result = run_experiment(demo_engine("experiment"), k=args.k)
    print(format_summary(result))
    print(save_result("experiment", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
