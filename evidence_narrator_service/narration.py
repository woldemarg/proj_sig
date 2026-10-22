"""Narration (docs/07_question_answering.md §7.5): the evidence of one question, verbalised by the LLM — or
summarised without it — with every citation checked against the evidence keys.

The LLM only verbalises evidence; it never retrieves or computes statistics. It is any
``llm_client.ChatModel`` (``OpenAICompatibleLLM`` in production, a fake in the tests).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from evidence_narrator_service.llm_client import ChatModel
from insight_contracts import EvidencePayload

SYSTEM_PROMPT = """You are a careful data analyst answering questions about a tabular dataset.
You receive EVIDENCE: statistically validated subgroup findings retrieved from a knowledge graph.
Shifts are robust standard deviations ("sd": the median difference scaled by the MAD).
Rules:
1. Use ONLY the evidence. Do not invent numbers, subgroups, metrics or datasets.
2. Cite every factual statement with its key, e.g. [P1] or [P2][P4].
3. Structure the answer in two labelled parts:
   "Спостереження:" - what the statistics show (medians, shifts in sd, support).
   "Інтерпретація (гіпотези):" - possible explanations, explicitly marked as hypotheses.
4. These are observational subgroup statistics. Do not claim causation; say "is associated with".
5. If items were reached through a latent anchor and are scope-disjoint from the seeds (no shared
   condition), point out that the same phenomenon recurs in a different part of the data.
6. Items reached only by location lie in the same area of the map; that does not show the findings are related.
   Say they are in the same area; do not explain one by the other, not even as a hypothesis.
   AREA: the question is about a map area the user selected. Answer it from the findings marked "in the asked area";
   if none is, say so first. Findings marked "outside the asked area" are not in it: keep them apart, after the
   area's own findings, as the same picture elsewhere. Findings with neither mark are from other levels or tables.
7. SCALE: a "scale:" line and the SCALE CHAIN follow a finding to coarser and finer levels of the same phenomenon.
   Say whether it holds at the coarser level or reverses there; "not found at the coarser level" means it is not
   confirmed there on its own, never that it is local.
8. If the evidence does not answer the question, say so plainly.
9. LANGUAGE: write the answer in Ukrainian. Copy every data literal byte-for-byte from the evidence, in its
   original script - column names, category values, dataset and file names, ids, "sd" and the [P#] keys.
   Never translate or transliterate them (write `margin`, `phones`, `US`, not their Ukrainian equivalents).
   The one exception is rule 10.
10. PLAIN LANGUAGE: when the evidence has a PLAIN LANGUAGE section, the reader is not an analyst. The evidence
   already describes its columns in words: retell those descriptions in everyday Ukrainian and keep category
   values as written. Describe shifts with the medians as plain ratios or percentages ("у 8 разів частіше:
   9% проти 1%"), not in sd. Do not use the words sd, median, MAD, quartile, q1-q4, LISA, subgroup, seed,
   transversal, latent or anchor; say "зони" for map cells, "схожа картина в інших зонах" for recurrences and "в тому самому районі" for items
   reached only by location, "тримається і на рівні району" for a finding that holds at the coarser level,
   "на рівні району зсув протилежний" for one that reverses there and "на рівні району окремо не підтверджено"
   for one not found there. The ZONES are larger areas: call them "зона Z1", "зона Z2" and say which findings
   each contains. Findings that lie in one region of the SCALE CHAIN are "в тому самому районі", which says nothing
   about their scale. The asked area is "виділена зона", never "район" (a district is about scale). Never write a region id.
11. TRUST: a "trust:" line gives three separate things: the source's reliability (A-F, as the uploader rated it),
   the credibility (whether another independent source confirms the finding) and the integrity check. None of them
   follows from the statistics on the "validation:" line: a strong p-value from one source is still unconfirmed.
   Say plainly when a finding is contradicted (credibility 5) or no other source covers it (6), or when its source
   is rated D, E or F. When sources disagree, describe the disagreement; do not resolve it. Name the levels in words,
   e.g. "підтверджено іншим джерелом", "суперечить іншому джерелу", "немає іншого джерела для перевірки", not by
   their codes.
Be concise (at most ~250 words)."""

CITATION = re.compile(r"\[(P\d+(?:\s*[,;]\s*P\d+)*)\]")  # [P1] and grouped [P1, P3]


@dataclass
class QAResult:
    question: str
    answer: str
    answer_mode: str  # llm | fallback | empty
    llm: dict[str, Any]
    citations: dict[str, Any]
    prompt: str  # the evidence the model was shown
    view: dict[str, Any]  # the graph service's view (evidence cards, traversal, highlight), as received
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance_footer: str = ""


def check_citations(answer: str, key_to_pattern: dict[str, str]) -> dict[str, Any]:
    keys = {k for group in CITATION.findall(answer) for k in re.split(r"\s*[,;]\s*", group)}
    cited = sorted(keys, key=lambda k: int(k[1:]))
    return {
        "cited": [{"key": k, "pattern_id": key_to_pattern[k]} for k in cited if k in key_to_pattern],
        "unknown": [k for k in cited if k not in key_to_pattern],
        "uncited": [k for k in key_to_pattern if k not in cited],
        "grounded": bool(cited) and all(k in key_to_pattern for k in cited),
    }


EMPTY_GRAPH = "Граф знань порожній — спершу завантажте датасет."


def _explain(llm: ChatModel | None, payload: EvidencePayload) -> tuple[str, str, dict[str, Any]]:
    """LLM answer when a model is given, there is evidence and it answers; otherwise the evidence-only answer: the
    cited observation lines in the two labelled parts of rule 3 -> (answer, mode, llm meta)."""
    meta: dict[str, Any] = {"model": None, "ok": False, "latency_s": 0.0, "error": None if llm is not None else "disabled"}
    if llm is not None and payload.citations:
        resp = llm.generate(SYSTEM_PROMPT, payload.evidence_prompt)
        meta = {"model": resp.model, "ok": resp.ok, "latency_s": round(resp.latency_s, 3), "error": resp.error, "usage": resp.usage}
        if resp.ok:
            return resp.text, "llm", meta
    fallback = "Спостереження:\n" + payload.evidence_summary + "\nІнтерпретація (гіпотези): не сформовано (відповідь мовної моделі недоступна)."
    return fallback, "fallback", meta


def answer(payload: EvidencePayload, llm: ChatModel | None) -> QAResult:
    """Verbalise the evidence of one question: the LLM (``None``: the evidence-only summary), the citation check
    against the manifest, the sources footer. Reads only the payload; its ``view`` goes to the console untouched."""
    if payload.graph_empty:
        return QAResult(payload.question, EMPTY_GRAPH, "empty", {}, {}, "", payload.view, {"error": "empty_graph"})
    start = time.perf_counter()
    text, mode, llm_meta = _explain(llm, payload)
    citations = check_citations(text, {key: c["pattern_id"] for key, c in payload.citations.items()})
    footer = "Джерела: " + "; ".join(
        f"[{key}] {c['pattern_id']} = {c['expression']} (датасет {c['dataset_id']}, {c['filename']}, батч {c['batch_id']})"
        for key, c in payload.citations.items()
    )
    metrics = {
        **payload.metrics,
        "total_s": round(payload.metrics["retrieval_s"] + time.perf_counter() - start, 3),
        "llm_latency_s": llm_meta.get("latency_s"),
        "prompt_chars": len(payload.evidence_prompt),
    }
    return QAResult(payload.question, text, mode, llm_meta, citations, payload.evidence_prompt, payload.view, metrics, footer)
