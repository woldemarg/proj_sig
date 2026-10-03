"""Live answer check against the configured LLM (docs/07_question_answering.md §7.7): grounding, citations, latency, tokens.

    python scripts/eval_answers.py --label baseline

Ingests the demo into a scratch workspace and asks five fixed questions through ``LLM_BASE_URL`` (by default the
LLM gateway, configured in ``.env.gemma``; values are never printed). About five LLM calls per run.
"""

from __future__ import annotations

import argparse

from scratch import demo_engine, save_result

QUESTIONS = [
    "Why is margin lower for phones in the US?",
    "What drives higher return rates?",
    "Tell me about EU laptops margin",
    "Where does the correlation between discount and margin break down?",
    "Why are delivery days longer for online orders in APAC?",
]


def evaluate(engine, question: str) -> dict:
    """One question: answer mode, grounding, citations, latency and token usage."""
    qa = engine.ask(question, use_llm=True)
    usage = qa.llm.get("usage") or {}
    prompt = qa.evidence.get("prompt", "")
    return {
        "question": question,
        "mode": qa.answer_mode,
        "grounded": bool(qa.citations.get("grounded")),
        "cited": len(qa.citations.get("cited", [])),
        "unknown": len(qa.citations.get("unknown", [])),
        "evidence": len(qa.evidence.get("items", [])),
        "latency_s": qa.llm.get("latency_s"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_chars": len(prompt),
        "prompt_non_ascii": sum(1 for ch in prompt if ord(ch) > 127),
        "error": qa.llm.get("error"),
        "answer": qa.answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current", help="name of this run (e.g. baseline, canon3, qwen3)")
    args = parser.parse_args()
    engine = demo_engine("eval_answers")
    print(f"LLM: {engine.llm.health()['model']} | embedder: {engine.config.embedding_model}")
    rows = [evaluate(engine, q) for q in QUESTIONS]
    cols = ["mode", "grounded", "cited", "unknown", "evidence", "latency_s", "prompt_tokens", "completion_tokens", "prompt_chars", "prompt_non_ascii"]
    print(f"{'#':<3}" + "".join(f"{c:>18}" for c in cols))
    for i, r in enumerate(rows, 1):
        print(f"{i:<3}" + "".join(f"{str(r[c]):>18}" for c in cols) + (f"   error: {r['error']}" if r["error"] else ""))
    answered = [r for r in rows if r["mode"] == "llm"]
    summary = {
        "llm_answers": len(answered),
        "grounded_rate": sum(r["grounded"] for r in rows) / len(rows),
        "unknown_citations": sum(r["unknown"] for r in rows),
        "mean_latency_s": round(sum(r["latency_s"] or 0 for r in answered) / max(len(answered), 1), 2),
        "prompt_tokens_total": sum(r["prompt_tokens"] or 0 for r in rows),
        "completion_tokens_total": sum(r["completion_tokens"] or 0 for r in rows),
    }
    print("summary:", summary)
    print("saved:", save_result("eval_answers", args.label, {"summary": summary, "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
