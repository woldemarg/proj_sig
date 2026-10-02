"""The 60-question multilingual retrieval benchmark (docs/07_question_answering.md §7.7, §10.4).

Four buckets of 15 questions each, on two scratch workspaces built from the synthetic demo:

* ``demo``    — the demo as shipped (English literals): B1 English, B2 code-switched (Ukrainian
                around the literals as stored), B3 fully translated into Ukrainian;
* ``demo_uk`` — the same rows with Ukrainian category / city / channel values: B4 inflected
                values (``у Харкові`` for ``Харків``) and, on ``demo``, typos (``fones``).

Every question has a *twin*: the question with every literal typed exactly as stored. The twin's
seeds and evidence are the gold standard (docs §7.7): Seed Recall@3, Cross-lingual Consistency
(Jaccard of seed sets), Evidence Overlap, FPGR (grounded spans whose symbol the twin does not
hold), direction / relationship agreement and the retrieval latency.

    python scripts/multilingual_benchmark.py --label full
    python scripts/multilingual_benchmark.py --label neo4j --neo4j fake      # retrieval must not change

``--neo4j fake`` publishes every batch through the tests' in-memory driver (never a live database).
Results: ``.scratch/results/multilingual_<label>.json``. Never touches ``workspace/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from scratch import build_engine, ingest_ready, save_result, scratch_dir, write_demo_csv

from ltir.synth import generate_retail_dataset

UK_VALUES = {
    "category": {"laptops": "ноутбуки", "phones": "телефони", "tablets": "планшети", "accessories": "аксесуари"},
    "region": {"EU": "Львів", "US": "Харків", "APAC": "Київ", "LATAM": "Одеса"},
    "channel": {"online": "онлайн", "retail": "роздріб", "partner": "партнер"},
}

# (bucket, workspace, question, twin with every literal as stored)
QUESTIONS: list[tuple[str, str, str, str]] = []
B1 = [
    "Why is margin lower for phones in the US?",
    "What drives higher return_rate?",
    "Where is margin higher for laptops in the EU?",
    "Why is discount higher for tablets in APAC?",
    "Where does the correlation between discount and margin break down?",
    "Which segments have longer delivery_days?",
    "Why is return_rate higher for online orders in APAC?",
    "Where is margin lower for laptops in retail?",
    "Tell me about discount for phones in the US",
    "Where is delivery_days longer for laptops in retail in the US?",
    "How are delivery_days and return_rate related?",
    "Where is discount higher?",
    "Why is margin lower for tablets in the EU retail channel?",
    "What is associated with higher discount?",
    "Where is return_rate lower?",
]
B2 = [
    "Чому margin нижчий для phones у US?",
    "Що спричиняє вищий return_rate?",
    "Де margin вищий для laptops у EU?",
    "Чому discount вищий для tablets у APAC?",
    "Де руйнується кореляція між discount і margin?",
    "Які сегменти мають довші delivery_days?",
    "Чому return_rate вищий для online у APAC?",
    "Де margin нижчий для laptops у retail?",
    "Розкажи про discount для phones у US",
    "Де delivery_days довші для laptops у retail у US?",
    "Як пов'язані delivery_days і return_rate?",
    "Де discount вищий?",
    "Чому margin нижчий для tablets у EU у каналі retail?",
    "Що пов'язано з вищим discount?",
    "Де return_rate нижчий?",
]
B3 = [
    "Чому маржа нижча для телефонів у США?",
    "Що спричиняє вищу частку повернень?",
    "Де маржа вища для ноутбуків у ЄС?",
    "Чому знижка вища для планшетів в APAC?",
    "Де руйнується кореляція між знижкою та маржею?",
    "Які сегменти мають довші дні доставки?",
    "Чому частка повернень вища для онлайн-замовлень в APAC?",
    "Де маржа нижча для ноутбуків у роздробі?",
    "Розкажи про знижку для телефонів у США",
    "Де дні доставки довші для ноутбуків у роздробі в США?",
    "Як пов'язані дні доставки та частка повернень?",
    "Де знижка вища?",
    "Чому маржа нижча для планшетів у ЄС у роздрібному каналі?",
    "Що пов'язано з вищою знижкою?",
    "Де частка повернень нижча?",
]
for q1, q2, q3 in zip(B1, B2, B3):
    QUESTIONS += [("B1", "demo", q1, q1), ("B2", "demo", q2, q1), ("B3", "demo", q3, q1)]
B4 = [
    ("demo_uk", "Чому маржа нижча для телефонів у Харкові?", "Why is margin lower for телефони in Харків?"),
    ("demo_uk", "Що спричиняє вищий return_rate у Києві?", "What drives higher return_rate in Київ?"),
    ("demo_uk", "Де margin вищий для ноутбуків у Львові?", "Where is margin higher for ноутбуки in Львів?"),
    ("demo_uk", "Чому discount вищий для планшетів у Києві?", "Why is discount higher for планшети in Київ?"),
    ("demo_uk", "Де margin нижчий для ноутбуків у роздробі?", "Where is margin lower for ноутбуки in роздріб?"),
    ("demo_uk", "Розкажи про discount для телефонів у Харкові", "Tell me about discount for телефони in Харків"),
    ("demo_uk", "Де delivery_days довші для ноутбуків у роздробі у Харкові?", "Where is delivery_days longer for ноутбуки in роздріб in Харків?"),
    ("demo_uk", "Чому return_rate вищий для онлайну у Києві?", "Why is return_rate higher for онлайн in Київ?"),
    ("demo", "Why is margn lower for fones in the US?", "Why is margin lower for phones in the US?"),
    ("demo", "What drives higher retrun_rate?", "What drives higher return_rate?"),
    ("demo", "Where is margin higher for laptps in the EU?", "Where is margin higher for laptops in the EU?"),
    ("demo", "Why is dicount higher for tablets in APAC?", "Why is discount higher for tablets in APAC?"),
    ("demo", "Where is margin lower for laptops in retial?", "Where is margin lower for laptops in retail?"),
    ("demo", "Where is delivry_days longer for laptops in retail in the US?", "Where is delivery_days longer for laptops in retail in the US?"),
    ("demo", "Where is return_rate lower for phoens?", "Where is return_rate lower for phones?"),
]
QUESTIONS += [("B4", ws, q, twin) for ws, q, twin in B4]
assert len(QUESTIONS) == 60


def write_uk_csv(folder: Path) -> Path:
    """The demo rows with Ukrainian values (and ``region`` renamed ``city``), as the document's B4 bucket needs."""
    path = folder / "retail_synthetic_uk.csv"
    if not path.exists():
        df = generate_retail_dataset(5000, 7)
        for col, mapping in UK_VALUES.items():
            df[col] = df[col].map(mapping)
        df.rename(columns={"region": "city"}).to_csv(path, index=False)
    return path


def workspaces(args: argparse.Namespace) -> dict[str, Any]:
    root = scratch_dir("multilingual")
    engines: dict[str, Any] = {}
    shared = None
    for name, writer in (("demo", write_demo_csv), ("demo_uk", write_uk_csv)):
        engine = build_engine(root / name, embedder=shared)
        shared = engine.encoder.embedder  # one loaded model for both workspaces
        if not engine.graph().of_kind("Pattern"):
            ingest_ready(engine, writer(root))
        if args.neo4j == "fake":
            _fake_neo4j(engine)
        engines[name] = engine
    return engines


def _fake_neo4j(engine: Any) -> None:
    """Turn the Neo4j mirror on through the tests' in-memory driver and publish the snapshot: the mirror path
    runs, no database is touched, and retrieval must come out identical (the document's step 6)."""
    from dataclasses import replace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_persistence import _Driver  # noqa: PLC0415

    from ltir import neo4j_sink

    real = neo4j_sink.publish_snapshot
    neo4j_sink.publish_snapshot = lambda snapshot, config, driver=None: real(snapshot, config, driver=_Driver())
    engine.config = replace(engine.config, neo4j_enabled=True)
    assert engine.sync_neo4j()["status"] == "ok"


def symbols(parsed: dict[str, Any]) -> set[str]:
    return set(parsed["targets"]) | set(parsed["conditions"])


def run(engines: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cache: dict[tuple[str, str], Any] = {}

    def ask(ws: str, q: str):
        if (ws, q) not in cache:
            cache[(ws, q)] = engines[ws].ask(q, use_llm=False)
        return cache[(ws, q)]

    for bucket, ws, q, twin in QUESTIONS:
        gold, qa = ask(ws, twin), ask(ws, q)
        gold_seeds = {s["pattern_id"] for s in gold.traversal["seeds"]}
        seeds = {s["pattern_id"] for s in qa.traversal["seeds"]}
        gold_ev = {i["pattern_id"] for i in gold.evidence["items"]}
        ev = {i["pattern_id"] for i in qa.evidence["items"]}
        parsed, gold_parsed = qa.evidence["parsed"], gold.evidence["parsed"]
        wanted = symbols(gold_parsed)
        grounding = parsed.get("grounding", [])  # a condition grounds to one symbol per column
        false = [g for g in grounding if not any(s in wanted for s in (g["symbol"] if isinstance(g["symbol"], list) else [g["symbol"]]))]
        rows.append(
            {
                "bucket": bucket,
                "workspace": ws,
                "question": q,
                "twin": twin,
                "recall3": len(seeds & gold_seeds) / max(len(gold_seeds), 1),
                "jaccard": len(seeds & gold_seeds) / max(len(seeds | gold_seeds), 1),
                "evidence_overlap": len(ev & gold_ev) / max(len(gold_ev), 1),
                "direction_ok": parsed["direction"] == gold_parsed["direction"],
                "covariance_ok": parsed["covariance"] == gold_parsed["covariance"],
                "symbols_ok": symbols(parsed) == wanted,
                "grounded": len(grounding),
                "false_grounded": len(false),
                "grounding": grounding,
                "retrieval_s": qa.metrics["retrieval_s"],
                "seeds": sorted(seeds),
                "gold_seeds": sorted(gold_seeds),
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bucket in ("B1", "B2", "B3", "B4"):
        part = [r for r in rows if r["bucket"] == bucket]
        grounded = sum(r["grounded"] for r in part)
        out[bucket] = {
            "recall3": sum(r["recall3"] for r in part) / len(part),
            "jaccard": sum(r["jaccard"] for r in part) / len(part),
            "evidence_overlap": sum(r["evidence_overlap"] for r in part) / len(part),
            "direction_acc": sum(r["direction_ok"] for r in part) / len(part),
            "symbols_acc": sum(r["symbols_ok"] for r in part) / len(part),
            "fpgr": (sum(r["false_grounded"] for r in part) / grounded) if grounded else 0.0,
            "retrieval_ms": 1000 * sum(r["retrieval_s"] for r in part) / len(part),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="full")
    parser.add_argument("--neo4j", default="off", choices=["off", "fake"])
    args = parser.parse_args()
    engines = workspaces(args)
    t0 = time.perf_counter()
    rows = run(engines)
    summary = summarise(rows)
    print(f"{'bucket':<7}{'recall@3':>9}{'jaccard':>9}{'evid.ovl':>9}{'dir.acc':>9}{'sym.acc':>9}{'FPGR':>7}{'ms':>8}")
    for b, m in summary.items():
        print(
            f"{b:<7}{m['recall3']:>9.2f}{m['jaccard']:>9.2f}{m['evidence_overlap']:>9.2f}{m['direction_acc']:>9.2f}{m['symbols_acc']:>9.2f}{m['fpgr']:>7.2f}{m['retrieval_ms']:>8.0f}"
        )
    misses = [r for r in rows if r["recall3"] < 1 or not r["direction_ok"] or r["false_grounded"]]
    for r in misses:
        print(
            f"  [{r['bucket']}] {r['question']}  recall={r['recall3']:.2f} dir={'ok' if r['direction_ok'] else 'X'} false={r['false_grounded']} seeds={r['seeds']} gold={r['gold_seeds']}"
        )
    path = save_result("multilingual", args.label, {"args": vars(args), "summary": summary, "rows": rows, "elapsed_s": time.perf_counter() - t0})
    print(f"({len(rows)} questions in {time.perf_counter() - t0:.1f}s)  saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
