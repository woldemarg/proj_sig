"""Live answer check against the configured LLM (docs/07_question_answering.md §7.7): grounding, citations, analyst words,
literal fidelity, latency, tokens.

    python scripts/eval_answers.py --label m12

Asks through ``LLM_BASE_URL`` (by default the LLM model broker, configured in ``.env``; values are never printed), on
throwaway workspaces of public data only: the five demo questions on the plain demo and on the demo with an invented
glossary and metadata document, three questions with a California zone and two of the map's suggested questions on
the California table at resolutions 6 and 5. Fifteen LLM calls per run.
"""

from __future__ import annotations

import argparse
import re

from scratch import build_engine, ingest_ready, save_result, scratch_dir, write_demo_csv

from evidence_narrator_service.llm_client import OpenAICompatibleLLM
from evidence_narrator_service.narration import answer
from evidence_narrator_service.settings import NarratorSettings
from insight_graph_service.core.settings import PROJECT_ROOT

QUESTIONS = [
    "Why is margin lower for phones in the US?",
    "What drives higher return rates?",
    "Tell me about EU laptops margin",
    "Where does the correlation between discount and margin break down?",
    "Why are delivery days longer for online orders in APAC?",
]
GLOSSARY = '{"region=US": "США", "channel": "канал продажу"}'  # invented for the demo, as the document below
DOC = """Демо: замовлення роздрібної мережі

Description: Синтетичні замовлення роздрібної мережі, по рядку на замовлення.

Columns:

margin
  Title (uk): маржа замовлення (%)
  Description: Валова маржа замовлення у відсотках.
  Assistant context: Вища маржа означає вигідніше замовлення.

discount
  Title (uk): знижка (%)
  Description: Знижка від ціни за прайсом.

category
  Title (uk): товарна категорія

region
  Title (uk): регіон продажу
"""
OAKLAND = {"geometry": {"center": [-122.2712, 37.8044], "radius_m": 20000}, "resolution": 5}
FRESNO = {"geometry": {"center": [-119.7871, 36.7378], "radius_m": 8000}, "resolution": 5}  # an inland circle
CALIFORNIA = [
    ("", OAKLAND),
    ("Чи вищі тут ціни на житло?", OAKLAND),
    ("Що відбувається з цінами в цій зоні?", FRESNO),
    ("Що особливого в найгарячіших зонах, де подій найбільше?", {}),  # GeoMap.suggestions
    ("Що відбувається там, де подій найменше?", {}),
]
TRANSLATED = {  # a demo literal the model may write in Ukrainian, counted as kept (rule 9 asks for it as stored)
    "phones": ("телефон", "фони"),
    "US": ("США",),
    "EU": ("ЄС",),
    "laptops": ("ноутбук",),
    "tablets": ("планшет",),
    "retail": ("роздріб",),
}


def fidelity(qa) -> tuple[int, int, int]:
    """Over the cited items with a scope value no glossary phrases (an item whose every condition is phrased says
    nothing of rule 9): (those whose every such value stands in the answer as stored, those that need a listed
    Ukrainian form of a demo value for one, those checked)."""
    evidence = qa.view["evidence"]
    items = {it["key"]: it for it in evidence.get("items", [])}
    phrased = {k for d in evidence.get("datasets", []) for k in (d.get("glossary", {}) | d.get("user_glossary", {}))}
    exact = translated = checked = 0
    for key in (c["key"] for c in qa.citations.get("cited", [])):
        if not (values := [c.split("=", 1)[1] for c in items[key]["scope"] if c not in phrased]):
            continue
        checked += 1
        if all(re.search(rf"(?<![^\W_]){re.escape(v)}(?![^\W_])", qa.answer) for v in values):
            exact += 1
        elif all(v in qa.answer or any(t in qa.answer for t in TRANSLATED.get(v, ())) for v in values):
            translated += 1
    return exact, translated, checked


def evaluate(engine, llm, question: str, **zone) -> dict:
    """One question: answer mode, grounding, citations, analyst words, literal fidelity, latency and token usage."""
    qa = answer(engine.evidence(question, **zone), llm)
    usage = qa.llm.get("usage") or {}
    exact, translated, checked = fidelity(qa)
    return {
        "question": question or "(the zone alone)",
        "mode": qa.answer_mode,
        "grounded": bool(qa.citations.get("grounded")),
        "cited": len(qa.citations.get("cited", [])),
        "unknown": len(qa.citations.get("unknown", [])),
        "evidence": len(qa.view["evidence"].get("items", [])),
        "jargon": qa.metrics.get("jargon", {}),
        "literal_ok": exact,
        "literal_translated": translated,
        "literal_checked": checked,
        "words": {w: len(re.findall(rf"(?<![^\W_]){w}", qa.answer, re.IGNORECASE)) for w in ("сегмент", "зон", "комірк", "регіон")},
        "latency_s": qa.llm.get("latency_s"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_chars": len(qa.prompt),
        "plain_language": "PLAIN LANGUAGE" in qa.prompt,
        "error": qa.llm.get("error"),
        "answer": qa.answer,
    }


def summarise(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["mode"] == "llm"]
    words: dict[str, int] = {}
    for r in rows:
        for w, n in r["jargon"].items():
            words[w] = words.get(w, 0) + n
    tokens = [r["prompt_tokens"] for r in rows if r["prompt_tokens"]]
    return {
        "llm_answers": len(answered),
        "grounded": f"{sum(r['grounded'] for r in rows)}/{len(rows)}",
        "unknown_citations": sum(r["unknown"] for r in rows),
        "citations": sum(r["cited"] for r in rows),
        "jargon": sum(words.values()),
        "jargon_words": words,
        "words": {w: sum(r["words"][w] for r in rows) for w in rows[0]["words"]},
        "literal_fidelity": round(sum(r["literal_ok"] for r in rows) / max(sum(r["literal_checked"] for r in rows), 1), 3),
        "literal_translated": sum(r["literal_translated"] for r in rows),
        "literal_checked": sum(r["literal_checked"] for r in rows),
        "prompt_tokens": [min(tokens, default=0), max(tokens, default=0), sum(tokens)],
        "completion_tokens_total": sum(r["completion_tokens"] or 0 for r in rows),
        "mean_latency_s": round(sum(r["latency_s"] or 0 for r in answered) / max(len(answered), 1), 2),
    }


def engines() -> dict:
    """The three workspaces, one embedding model: the demo, the demo with its declared language, California at
    resolutions 6 and 5 (its scale chains)."""
    plain = build_engine(scratch_dir("eval_answers_demo", fresh=True) / "ws")
    ingest_ready(plain, write_demo_csv(scratch_dir("eval_answers_demo")))
    root = scratch_dir("eval_answers_declared", fresh=True)
    declared = build_engine(root / "ws", embedder=plain.encoder.embedder)
    record = declared.process(declared.submit(write_demo_csv(root), glossary=GLOSSARY, metadata=DOC)["batch_id"])
    if record["status"] != "READY":
        raise SystemExit(f"declared demo: {record['status']} {record.get('error')}")
    california = build_engine(scratch_dir("eval_answers_california", fresh=True) / "ws", embedder=plain.encoder.embedder)
    with (PROJECT_ROOT / "data" / "demo" / "california_housing.csv").open("rb") as f:
        levels = california.upload("california_housing.csv", f, geo="auto@6,5")
    for level in levels:
        if (record := california.process(level["batch_id"]))["status"] != "READY":
            raise SystemExit(f"california {level['level']}: {record['status']} {record.get('error')}")
    return {"demo": plain, "demo_declared": declared, "california": california}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current", help="name of this run (e.g. baseline, m12)")
    args = parser.parse_args()
    built = engines()
    llm = OpenAICompatibleLLM(NarratorSettings.from_env())
    print(f"LLM: {llm.health()['model']} | embedder: {built['demo'].encoder.embedder.model_id}")
    runs = {
        "demo": [evaluate(built["demo"], llm, q) for q in QUESTIONS],
        "demo_declared": [evaluate(built["demo_declared"], llm, q) for q in QUESTIONS],
        "california": [evaluate(built["california"], llm, q, **zone) for q, zone in CALIFORNIA],
    }
    cols = [
        "mode",
        "grounded",
        "cited",
        "unknown",
        "literal_ok",
        "literal_checked",
        "latency_s",
        "prompt_tokens",
        "completion_tokens",
        "plain_language",
    ]
    for name, rows in runs.items():
        print(f"\n{name}")
        print(f"{'#':<3}" + "".join(f"{c:>18}" for c in cols) + "   jargon")
        for i, r in enumerate(rows, 1):
            print(
                f"{i:<3}" + "".join(f"{str(r[c]):>18}" for c in cols) + f"   {r['jargon'] or ''}" + (f"   error: {r['error']}" if r["error"] else "")
            )
        print("summary:", summarise(rows))
    summary = {name: summarise(rows) for name, rows in runs.items()}
    print("saved:", save_result("eval_answers", args.label, {"summary": summary, "runs": runs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
