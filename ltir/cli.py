"""Command line entry point: ``python -m ltir <command>``.

demo                 write the synthetic dataset and ingest it
ingest PATH          run the dataset lifecycle on a CSV/TSV/Parquet file (--bins col:q,... --categories a,b)
query "QUESTION"     grounded answer (--no-llm, --json)
status               batches + graph statistics
experiment           cross-dimensional analogue retrieval benchmark (--k)
rebuild-graph        regenerate graph/snapshot.json from journals + state
sphere               write the 3D latent sphere HTML (-o FILE, --dataset ID)
neo4j-sync           make the Neo4j mirror equal to the snapshot (NEO4J_* settings)
llm-check            probe the LLM endpoint (LLM_BASE_URL: the gateway by default)
reset                delete the SIG workspace (--yes)
migrate              re-ingest every READY batch with the current code; old workspace kept as a backup (--yes)
serve                start the web UI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ltir.config import load_config


def _print_batch(rec: dict) -> None:
    m = rec.get("metrics", {})
    print(f"{rec['batch_id']}  {rec['status']:<8} {rec.get('filename')}  dataset={rec.get('dataset_id')}")
    if rec.get("error"):
        print(f"  error: {rec['error']['code']}: {rec['error']['message']}")
    for w in rec.get("warnings", []):
        print(f"  warning: {w}")
    if m.get("validated_insights") is not None:
        print(
            f"  rows={m['input_rows']} candidates={m['candidate_patterns']} validated_insights={m['validated_insights']} "
            f"pruned={m['pruned_total']} attractors={m['attractors_total']} (+{m['attractors_new']}) orphan_rate={m['orphan_rate']:.2f} "
            f"edges={m['graph_edges']} avg_attractor_degree={m['avg_attractor_degree']:.2f} duration={m['processing_duration_s']:.1f}s"
        )


def main(argv: list[str] | None = None) -> int:
    """Run one command; a busy workspace (another writer holds its lock) is a clean error, exit code 2."""
    from ltir.storage.workspace import WorkspaceBusy

    try:
        return _run(argv)
    except WorkspaceBusy as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog="ltir", description="Latent Transversal Insight Representation (SIG)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    p = sub.add_parser("ingest")
    p.add_argument("path")
    p.add_argument("--bins", default=None, help="derive quantile band dimensions, e.g. median_income:4")
    p.add_argument("--categories", default=None, help="treat code-like columns as dimensions, e.g. Store,Holiday_Flag")
    p = sub.add_parser("query")
    p.add_argument("question")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--json", action="store_true")
    sub.add_parser("status")
    p = sub.add_parser("experiment")
    p.add_argument("--k", type=int, default=5)
    sub.add_parser("rebuild-graph")
    p = sub.add_parser("sphere")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--dataset", default=None)
    sub.add_parser("neo4j-sync")
    sub.add_parser("llm-check")
    p = sub.add_parser("reset")
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("migrate")
    p.add_argument("--yes", action="store_true")
    sub.add_parser("serve")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        from ltir.web.app import main as serve

        serve()
        return 0
    if args.cmd == "llm-check":
        from ltir.llm_client import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(load_config())
        print(json.dumps(llm.health(fresh=True), indent=1))
        resp = llm.generate("Reply with one word.", "Say OK.")
        print(
            json.dumps(
                {"ok": resp.ok, "model": resp.model, "latency_s": round(resp.latency_s, 2), "text": resp.text[:200], "error": resp.error}, indent=1
            )
        )
        return 0 if resp.ok else 1
    if args.cmd == "migrate":
        from ltir.migrate import migrate_workspace

        cfg = load_config()
        if not args.yes:
            print(
                f"This re-ingests every READY batch of {cfg.workspace_dir} with the current code; the old workspace is kept as a backup. Re-run with --yes.",
                file=sys.stderr,
            )
            return 1
        print(json.dumps(migrate_workspace(cfg), indent=1))
        return 0

    from ltir.engine import Engine

    engine = Engine(load_config(), recover=args.cmd in {"demo", "ingest", "reset", "rebuild-graph"})  # only writers recover
    if args.cmd == "demo":
        from ltir.evaluation.synthetic import write_demo

        rec = engine.ingest_file(write_demo())
        _print_batch(rec)
        return 0 if rec["status"] in {"READY", "SKIPPED"} else 1
    if args.cmd == "ingest":
        rec = engine.ingest_file(args.path, bins=args.bins, categories=args.categories)
        _print_batch(rec)
        return 0 if rec["status"] in {"READY", "SKIPPED"} else 1
    if args.cmd == "query":
        qa = engine.ask(args.question, use_llm=not args.no_llm)
        if args.json:
            print(json.dumps(qa.to_dict(), indent=1, ensure_ascii=False, default=str))
            return 0
        print(qa.answer)
        print("\n---")
        print(
            f"mode={qa.answer_mode} llm={qa.llm.get('model')} ok={qa.llm.get('ok')} err={qa.llm.get('error')} grounded={qa.citations.get('grounded')}"
        )
        print(
            f"seeds={qa.highlight['seeds']} anchors={qa.highlight['anchors']} evidence={len(qa.highlight['evidence'])} "
            f"cross_scope={qa.highlight['transversal_only']} metrics={qa.metrics}"
        )
        for it in qa.evidence.get("items", []):
            print(f"  [{it['key']}] {it['role']:<11} {' AND '.join(it['scope'])}  via {it['path_text']}")
        print(qa.provenance_footer)
        return 0
    if args.cmd == "status":
        for rec in engine.ws.list_batches():
            _print_batch(rec)
        print("graph:", json.dumps(engine.graph().snapshot.get("stats", {})))
        return 0
    if args.cmd == "experiment":
        from ltir.evaluation.experiment import format_summary, run_experiment

        print(format_summary(run_experiment(engine, k=args.k)))
        return 0
    if args.cmd == "rebuild-graph":
        snapshot = engine.rebuild_graph()
        print(json.dumps({**snapshot["stats"], "neo4j": engine.sync_neo4j(snapshot)["status"]}))
        return 0
    if args.cmd == "sphere":
        from pathlib import Path

        from ltir.web.sphere import export_sphere

        print(export_sphere(engine, Path(args.output) if args.output else None, dataset=args.dataset))
        return 0
    if args.cmd == "neo4j-sync":
        from ltir.storage.neo4j_mirror import publish_snapshot

        snap = engine.ws.load_graph()
        if not snap:
            print("no graph snapshot yet", file=sys.stderr)
            return 1
        print(json.dumps(publish_snapshot(snap, engine.config), indent=1))
        return 0
    if args.cmd == "reset":
        if not args.yes:
            print(f"This deletes {engine.config.workspace_dir}. Re-run with --yes.", file=sys.stderr)
            return 1
        neo4j = engine.reset()["neo4j"]["status"]
        print(f"workspace reset (Neo4j mirror: {neo4j})")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
