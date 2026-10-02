"""Grounded question answering (docs/07_question_answering.md).

question -> parse -> seed resolution -> transversal traversal -> evidence
-> Gemma 4 (or the evidence-only summary) -> citation check -> highlight groups
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from ltir.evidence import Evidence, build_evidence
from ltir.graph import DualGraph
from ltir.llm import SYSTEM_PROMPT
from ltir.query import SeedMatch, parse_query, resolve_seeds
from ltir.store import utc_now
from ltir.traversal import TraversalResult, naive_nearest, structural_closure, traverse

if TYPE_CHECKING:
    from ltir.pipeline import Engine

CITATION = re.compile(r"\[(P\d+(?:\s*[,;]\s*P\d+)*)\]")  # [P1] and grouped [P1, P3]
HIGHLIGHT_KEYS = ("seeds", "traversed", "anchors", "evidence", "edges", "transversal_only")


@dataclass
class QAResult:
    question: str
    answer: str
    answer_mode: str  # llm | fallback | empty
    llm: dict[str, Any]
    citations: dict[str, Any]
    evidence: dict[str, Any]
    traversal: dict[str, Any]
    highlight: dict[str, list[str]]
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance_footer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_citations(answer: str, key_to_pattern: dict[str, str]) -> dict[str, Any]:
    keys = {k for group in CITATION.findall(answer) for k in re.split(r"\s*[,;]\s*", group)}
    cited = sorted(keys, key=lambda k: int(k[1:]))
    return {
        "cited": [{"key": k, "pattern_id": key_to_pattern[k]} for k in cited if k in key_to_pattern],
        "unknown": [k for k in cited if k not in key_to_pattern],
        "uncited": [k for k in key_to_pattern if k not in cited],
        "grounded": bool(cited) and all(k in key_to_pattern for k in cited),
    }


def compute_baselines(engine: Engine, graph: DualGraph, question: str, seeds: list[SeedMatch], result: TraversalResult) -> dict[str, Any]:
    """What structural-only and naive text retrieval would have returned (docs/07_question_answering.md §7.6)."""
    config = engine.config
    closure = structural_closure(graph, [s.pattern_id for s in seeds], config.traversal_max_depth, config.lattice_edges)
    documents = engine.document_vectors()  # embedded at ingest: the question is the only text embedded here
    doc_vecs = {n["id"]: documents[n["id"]] for n in graph.of_kind("Pattern")}
    naive = naive_nearest(engine.encoder.embedder.embed_queries([question])[0], doc_vecs, config.max_retrieved)
    retrieved = {r.node_id for r in result.patterns}
    return {
        "structural_only": sorted(closure),
        "naive_nearest": [{"pattern_id": p, "cosine": round(c, 3)} for p, c in naive],
        "transversal_only": [r.node_id for r in result.patterns if r.transversal_only],
        "not_in_naive_topk": sorted(retrieved - {p for p, _ in naive}),
        "not_structurally_reachable": sorted(retrieved - set(closure)),
    }


def _explain(engine: Engine, evidence: Evidence, prompt: str, use_llm: bool) -> tuple[str, str, dict[str, Any]]:
    """LLM answer when enabled and reachable, otherwise the evidence-only summary -> (answer, mode, llm meta)."""
    meta: dict[str, Any] = {"model": None, "ok": False, "latency_s": 0.0, "error": None if use_llm else "disabled"}
    if use_llm and evidence.items:
        resp = engine.llm.generate(SYSTEM_PROMPT, prompt)
        meta = {"model": resp.model, "ok": resp.ok, "latency_s": round(resp.latency_s, 3), "error": resp.error, "usage": resp.usage}
        if resp.ok:
            return resp.text, "llm", meta
    return evidence.summary(), "fallback", meta


def answer_question(engine: Engine, question: str, *, use_llm: bool = True) -> QAResult:
    t0 = time.perf_counter()
    graph = engine.graph()
    config = engine.config
    if not graph.of_kind("Pattern"):
        return QAResult(
            question,
            "The knowledge graph is empty — upload a dataset first.",
            "empty",
            {},
            {},
            {},
            {},
            {k: [] for k in HIGHLIGHT_KEYS},
            {"error": "empty_graph"},
        )

    engine.ws.check_representation(engine.encoder.spec)  # query and stored vectors must share one frame
    parsed = parse_query(question, graph, config)
    seeds = resolve_seeds(parsed, graph, engine.encoder, engine.frame().patterns, config)
    result = traverse(graph, seeds, config)
    result.baselines = compute_baselines(engine, graph, question, seeds, result)
    t_retrieval = time.perf_counter() - t0

    evidence = build_evidence(parsed, result, graph, config)
    prompt = evidence.to_prompt()
    answer, mode, llm_meta = _explain(engine, evidence, prompt, use_llm)

    citations = check_citations(answer, evidence.key_to_pattern)
    footer = "Sources: " + "; ".join(
        f"[{p['key']}] {p['pattern_id']} = {p['expression']} (dataset {p['dataset_id']}, {p['filename']}, batch {p['batch_id']})"
        for p in evidence.provenance
    )
    highlight = {
        "seeds": [s.pattern_id for s in seeds],
        "traversed": result.traversed_nodes,
        "anchors": [a.node_id for a in result.attractors],
        "evidence": [it.pattern_id for it in evidence.items],
        "edges": result.used_edges,
        "transversal_only": [it.pattern_id for it in evidence.items if it.transversal_only],
    }
    metrics = {
        "retrieval_s": round(t_retrieval, 3),
        "total_s": round(time.perf_counter() - t0, 3),
        "llm_latency_s": llm_meta.get("latency_s"),
        "seed_count": len(seeds),
        "traversal_depth": result.max_depth,
        "visited_states": result.visited_count,
        "retrieved_evidence": len(evidence.items),
        "anchors_visited": len(result.attractors),
        "prompt_chars": len(prompt),
    }
    qa = QAResult(question, answer, mode, llm_meta, citations, {**evidence.to_dict(), "prompt": prompt}, result.to_dict(), highlight, metrics, footer)
    engine.ws.log_query(
        {
            "at": utc_now(),
            "question": question,
            "mode": mode,
            "metrics": metrics,
            "seeds": highlight["seeds"],
            "evidence": highlight["evidence"],
            "citations": citations,
        }
    )
    return qa
