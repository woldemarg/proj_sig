"""Chat answers (docs/07_question_answering.md §7.5): the evidence of a search, verbalised by the LLM — or
summarised without it — with every citation checked against the evidence keys.

The LLM only verbalises evidence; it never retrieves or computes statistics. It is any
``llm_client.ChatModel`` (``OpenAICompatibleLLM`` in production, a fake in the tests).
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ltir.llm_client import ChatModel
from ltir.retrieval.evidence import Evidence
from ltir.retrieval.search import HIGHLIGHT_KEYS, SearchResult

SYSTEM_PROMPT = """You are a careful data analyst answering questions about a tabular dataset.
You receive EVIDENCE: statistically validated subgroup findings retrieved from a knowledge graph.
Shifts are robust standard deviations ("sd": the median difference scaled by the MAD).
Rules:
1. Use ONLY the evidence. Do not invent numbers, subgroups, metrics or datasets.
2. Cite every factual statement with its key, e.g. [P1] or [P2][P4].
3. Structure the answer in two labelled parts:
   "Спостереження:" - what the verified statistics show (medians, shifts in sd, support).
   "Інтерпретація (гіпотези):" - possible explanations, explicitly marked as hypotheses.
4. These are observational subgroup statistics. Do not claim causation; say "is associated with".
5. If items were reached through a latent anchor and are scope-disjoint from the seeds (no shared
   condition), point out that the same phenomenon recurs in a different part of the data.
6. If the evidence does not answer the question, say so plainly.
7. LANGUAGE: write the answer in Ukrainian. Copy every data literal byte-for-byte from the evidence, in its
   original script - column names, category values, dataset and file names, ids, "sd" and the [P#] keys.
   Never translate or transliterate them (write `margin`, `phones`, `US`, not their Ukrainian equivalents).
Be concise (at most ~250 words)."""

CITATION = re.compile(r"\[(P\d+(?:\s*[,;]\s*P\d+)*)\]")  # [P1] and grouped [P1, P3]


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


def empty_answer(question: str) -> QAResult:
    """The answer while the knowledge graph holds no insight."""
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


def _explain(llm: ChatModel | None, evidence: Evidence, prompt: str) -> tuple[str, str, dict[str, Any]]:
    """LLM answer when a model is given and answers, otherwise the evidence-only summary -> (answer, mode, llm meta)."""
    meta: dict[str, Any] = {"model": None, "ok": False, "latency_s": 0.0, "error": None if llm is not None else "disabled"}
    if llm is not None and evidence.items:
        resp = llm.generate(SYSTEM_PROMPT, prompt)
        meta = {"model": resp.model, "ok": resp.ok, "latency_s": round(resp.latency_s, 3), "error": resp.error, "usage": resp.usage}
        if resp.ok:
            return resp.text, "llm", meta
    return evidence.summary(), "fallback", meta


def answer(found: SearchResult, llm: ChatModel | None) -> QAResult:
    """Verbalise a search: the LLM (``None``: the evidence-only summary), the citation check, the highlight groups."""
    start = time.perf_counter()
    evidence, result = found.evidence, found.traversal
    prompt = evidence.to_prompt()
    text, mode, llm_meta = _explain(llm, evidence, prompt)
    citations = check_citations(text, evidence.key_to_pattern)
    footer = "Sources: " + "; ".join(
        f"[{p['key']}] {p['pattern_id']} = {p['expression']} (dataset {p['dataset_id']}, {p['filename']}, batch {p['batch_id']})"
        for p in evidence.provenance
    )
    metrics = {
        "retrieval_s": round(found.seconds, 3),
        "total_s": round(found.seconds + time.perf_counter() - start, 3),
        "llm_latency_s": llm_meta.get("latency_s"),
        "seed_count": len(found.seeds),
        "traversal_depth": result.max_depth,
        "visited_states": result.visited_count,
        "retrieved_evidence": len(evidence.items),
        "anchors_visited": len(result.attractors),
        "prompt_chars": len(prompt),
    }
    return QAResult(
        found.question,
        text,
        mode,
        llm_meta,
        citations,
        {**evidence.to_dict(), "prompt": prompt},
        result.to_dict(),
        found.highlight(),
        metrics,
        footer,
    )
