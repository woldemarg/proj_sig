"""Hypothesis experiment (SDD 15 §Experiment): cross-dimensional analogue retrieval.

For every pattern S that expresses a planted phenomenon, the *analogues* of S are
the other patterns expressing the same phenomenon whose scope shares no condition
with S. Four retrievers rank all other patterns from S:

    transversal   ltir.traversal (seed forced to S)
    structural    BFS over all structural edges (SPECIALIZES/GENERALIZES/SIBLING/CONTRASTS)
    text_nn       cosine of canonical-document text embeddings (naive vector RAG)
    vector_nn     cosine of LTIR insight vectors (same space, no attractor graph)

and are scored with recall@k, precision@k (hits over min(k, |analogues|), so a
case with fewer analogues than k can reach 1.0) and MRR of the first analogue.

Phenomenon membership comes from the *planted* ground truth (``ltir.synth.GROUND_TRUTH``):
a pattern belongs to a mechanism when its scope lies inside one of that mechanism's
planted scopes. Labels must not be derived from the measured shifts: those are exactly
what the representation encodes, so shift-based labels would favour vector retrieval
by construction.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ltir.models import Insight
from ltir.pipeline import Engine
from ltir.query import SeedMatch
from ltir.store import utc_now
from ltir.synth import GROUND_TRUTH
from ltir.traversal import structural_closure, traverse

Truth = dict[str, list[dict[str, str]]]  # mechanism -> planted scopes


def planted_scopes(truth: dict[str, Any] = GROUND_TRUTH) -> Truth:
    """Mechanisms with their planted scopes (only multi-scope mechanisms have analogues)."""
    out: Truth = {}
    for name, spec in truth.items():
        scopes = spec.get("scopes") or ([spec["scope"]] if "scope" in spec else [])
        if scopes:
            out[name] = scopes
    return out


@dataclass
class Case:
    seed: str
    phenomenon: str
    analogues: set[str]


def phenomenon_of(ins: Insight, truth: Truth | None = None) -> str | None:
    """Planted mechanism whose scope contains the pattern's scope; None if none or ambiguous."""
    truth = truth if truth is not None else planted_scopes()
    conds = {(c.attribute, c.value) for c in ins.conditions}
    hits = [name for name, scopes in truth.items() if any(set(s.items()) <= conds for s in scopes)]
    return hits[0] if len(hits) == 1 else None


def build_cases(engine: Engine, dataset_id: str | None = None, truth: Truth | None = None) -> list[Case]:
    g = engine.graph()
    pats = [ins for ins in g.insights.values() if dataset_id is None or ins.dataset_id == dataset_id]
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


def run_experiment(engine: Engine, *, k: int = 5, dataset_id: str | None = None, truth: Truth | None = None) -> dict[str, Any]:
    g = engine.graph()
    cfg = engine.config
    truth = truth if truth is not None else planted_scopes()
    cases = build_cases(engine, dataset_id, truth)
    ids = [n["id"] for n in g.of_kind("Pattern")]
    vec_nn = engine.frame().patterns  # the LTIR representation itself, no attractor graph
    text = dict(zip(ids, engine.encoder.embedder.embed([g.canonical_document(p) for p in ids])))
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
    out = engine.ws.root / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"hypothesis_{result['at'][:19].replace(':', '')}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"cross-dimensional analogue retrieval — {result['cases']} seed cases, k={result['k']}  {result['phenomena']}",
        f"{'method':<12} {'recall@k':>9} {'precision@k':>12} {'MRR':>6}",
    ]
    for m, s in result["summary"].items():
        lines.append(f"{m:<12} {s['recall']:>9.3f} {s['precision']:>12.3f} {s['mrr']:>6.3f}")
    return "\n".join(lines)
