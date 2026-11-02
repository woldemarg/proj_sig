"""A/B of the two temporal models across worktrees (docs/10_verification.md §10.4): the chain model (a batch per time
window, EVOLVES_TO between them) against the event model (one global pass, a series per pattern), on the planted
uploads of ``temporal_benchmark.generate`` — the same bytes on both branches (``compare`` checks their digests).

    python scripts/temporal_ab.py capture --label chain|event [--seeds 1-3] [--encoder hashing|model]
    python scripts/temporal_ab.py compare .scratch/results/temporal_ab_chain.json .scratch/results/temporal_ab_event.json

``capture`` uploads each seed's file twice through the graph service, on scratch workspaces: with its time column (the
chain branch cuts it into ``window=7D``, the event branch bins it by ``grid=week``) and without. Per upload it records
the Pattern nodes, the edges by type, the journal rows, the seconds (processing and discovery), the latent plane, the
scopes admitted with their targets, and the event edges (type, period, scope). ``compare`` prints one table: graph and journal size, time, the latent plane against the plain
upload's, the plain upload's patterns recalled, the temporal edges, and the chain's window-only finds — scopes no plain
upload admits — by class: *nested* (a specialisation or a generalisation of a plain scope), *overlapping* (sharing a
condition with one) and *new* (sharing none), then each of them by name: its class, the seeds that admitted it in
that class, its targets and what its rows are to the planted cohorts S1–S4 of the file — it narrows one, widens one,
holds part of one (no column of it conflicts), or none (it holds no planted row) —, and each capture's event edges.
The event model loses those finds by design (Flash Event Blindness; its boundary is the S7 matrix of
``temporal_benchmark``). Results: ``.scratch/results/temporal_ab_<label>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from attractor_benchmark import parse_seeds
from scratch import build_engine, save_result, scratch_dir
from temporal_benchmark import PLANTED, generate

TEMPORAL_EDGES = ("EVOLVES_TO", "PEAKED_AT", "DROPPED_AT")


def latent(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The latent plane of a snapshot: its anchors and their members per anchor."""
    anchors = [n["props"] for n in snapshot["nodes"] if n["kind"] == "Attractor"]
    return {"anchors": len(anchors), "patterns_per_anchor": float(np.mean([a["n_patterns"] for a in anchors])) if anchors else 0.0}


def measure(engine: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    """One upload's graph, journal, time and scopes after its batches ran."""
    snapshot = engine.ws.load_graph() or {"nodes": [], "edges": []}
    patterns = [n["props"] for n in snapshot["nodes"] if n["kind"] == "Pattern"]
    scope = {
        n["id"]: " ∧ ".join(f"{c['attribute']}={c['value']}" for c in n["props"]["conditions"]) for n in snapshot["nodes"] if n["kind"] == "Pattern"
    }
    targets: dict[str, set[str]] = {}
    for n in snapshot["nodes"]:
        if n["kind"] == "Pattern":
            targets.setdefault(scope[n["id"]], set()).add(n["props"]["target"])
    return {
        "batches": dict(Counter(r["status"] for r in records)),
        "patterns": len(patterns),
        "edges": dict(Counter(e["type"] for e in snapshot["edges"])),
        "journal_rows": len(engine.ws.patterns()),
        "seconds": sum(r["metrics"].get("processing_duration_s", 0.0) for r in records),
        "discover_seconds": sum(r["metrics"].get("timings", {}).get("discover_s", 0.0) for r in records),
        "latent": latent(snapshot),
        "scopes": sorted({tuple(f"{c['attribute']}={c['value']}" for c in p["conditions"]) for p in patterns}),
        "targets": {s: sorted(t) for s, t in targets.items()},
        "events": sorted([e["type"], e["target"], scope[e["source"]]] for e in snapshot["edges"] if e["type"] in TEMPORAL_EDGES[1:]),
    }


def capture_seed(seed: int, label: str, embedder: Any) -> dict[str, Any]:
    folder = scratch_dir(f"temporal_ab/{label}_{seed}", fresh=True)
    path = folder / "events.csv"
    generate(seed).to_csv(path, index=False, lineterminator="\n")
    timed = build_engine(folder / "timed", embedder=embedder)
    option = {"window": "7D"} if "window" in inspect.signature(timed.upload).parameters else {"grid": "week"}  # chain | event
    with path.open("rb") as f:
        uploaded = timed.upload(path.name, f, time="when", **option)
    records = [timed.process(r["batch_id"]) for r in uploaded]
    plain = build_engine(folder / "plain", embedder=timed.encoder.embedder)
    plain_record = plain.ingest_file(path)
    return {
        "seed": seed,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "option": option,
        "timed": measure(timed, records),
        "plain": measure(plain, [plain_record]),
    }


def classify(scope: tuple[str, ...], plain: list[tuple[str, ...]]) -> str:
    """A window-only scope against the plain upload's scopes (conditions as sets)."""
    own = set(scope)
    if any(own < set(p) or set(p) < own for p in plain):
        return "nested"
    return "overlapping" if any(own & set(p) for p in plain) else "new"


def planted(scope: tuple[str, ...]) -> str:
    """What a scope's rows are to the planted cohorts S1–S4 of the A/B file (independent columns): it narrows one, widens
    one, holds part of one (no column of it conflicts with the cohort's), or none — it holds no planted row."""
    own = dict(c.split("=", 1) for c in scope)
    for verb, test in (
        ("narrows", lambda p: p.items() <= own.items()),
        ("widens", lambda p: own.items() < p.items()),
        ("holds part of", lambda p: all(own.get(c, v) == v for c, v in p.items())),
    ):
        hit = [name for name, (conds, _) in list(PLANTED.items())[:4] if test(conds)]
        if hit:
            return f"{verb} {', '.join(hit)}"
    return "none"


def window_only(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[int], str, str]]:
    """The window-only scopes of a capture by name: class, scope, the seeds that admitted it in that class, its targets
    and its planted relation."""
    seen: dict[tuple[tuple[str, ...], str], list[int]] = {}
    targets: dict[tuple[str, ...], set[str]] = {}
    for r in rows:
        plain = [tuple(s) for s in r["plain"]["scopes"]]
        for s in map(tuple, r["timed"]["scopes"]):
            if s not in plain:
                seen.setdefault((s, classify(s, plain)), []).append(r["seed"])
                targets.setdefault(s, set()).update(r["timed"]["targets"].get(" ∧ ".join(s), []))
    return sorted((cls, " ∧ ".join(s), seeds, ", ".join(sorted(targets[s])), planted(s)) for (s, cls), seeds in seen.items())


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over the seeds, the plain patterns the timed upload recalls, the window-only finds by class."""

    def mean(f) -> float:
        return float(np.mean([f(r) for r in rows]))

    only = Counter()
    for r in rows:
        plain = [tuple(s) for s in r["plain"]["scopes"]]
        only.update(classify(tuple(s), plain) for s in r["timed"]["scopes"] if tuple(s) not in plain)
    return {
        "seeds": [r["seed"] for r in rows],
        "patterns": mean(lambda r: r["timed"]["patterns"]),
        "plain_patterns": mean(lambda r: r["plain"]["patterns"]),
        "edges": mean(lambda r: sum(r["timed"]["edges"].values())),
        "temporal_edges": mean(lambda r: sum(r["timed"]["edges"].get(t, 0) for t in TEMPORAL_EDGES)),
        "journal_rows": mean(lambda r: r["timed"]["journal_rows"]),
        "seconds": mean(lambda r: r["timed"]["seconds"]),
        "discover_seconds": mean(lambda r: r["timed"]["discover_seconds"]),
        "anchors": mean(lambda r: r["timed"]["latent"]["anchors"]),
        "plain_anchors": mean(lambda r: r["plain"]["latent"]["anchors"]),
        "patterns_per_anchor": mean(lambda r: r["timed"]["latent"]["patterns_per_anchor"]),
        "plain_recall": mean(
            lambda r: len({tuple(s) for s in r["plain"]["scopes"]} & {tuple(s) for s in r["timed"]["scopes"]}) / max(len(r["plain"]["scopes"]), 1)
        ),
        "window_only": {k: v / len(rows) for k, v in sorted(only.items())},
    }


def compare(chain: dict[str, Any], event: dict[str, Any]) -> str:
    """The table of the two captures (the same seeds, the same files)."""
    files = {r["seed"]: r["file_sha256"] for r in chain["per_seed"]}
    if any(files.get(r["seed"]) != r["file_sha256"] for r in event["per_seed"]):
        raise SystemExit("the captures did not read the same files: compare captures of the same seeds")
    a, b = chain["summary"], event["summary"]
    lines = [f"| measure (mean of seeds {b['seeds']}) | chain | event |", "|---|---|---|"]
    for key in (
        "patterns",
        "plain_patterns",
        "edges",
        "temporal_edges",
        "journal_rows",
        "seconds",
        "discover_seconds",
        "anchors",
        "plain_anchors",
        "patterns_per_anchor",
        "plain_recall",
    ):
        lines.append(f"| {key} | {a[key]:.2f} | {b[key]:.2f} |")
    for cls in ("nested", "overlapping", "new"):
        lines.append(f"| window-only, {cls} | {a['window_only'].get(cls, 0.0):.2f} | {b['window_only'].get(cls, 0.0):.2f} |")
    for side, capture in (("chain", chain), ("event", event)):
        found = window_only(capture["per_seed"])
        lines += ["", f"window-only scopes of the {side} model: {len({s for _, s, *_ in found})}, admitted {sum(len(f[2]) for f in found)} times"]
        lines += [f"  {c:<11} {s}  (seeds {seeds}; target {t}; {p})" for c, s, seeds, t, p in found]
        lines += [f"  seed {r['seed']} event: {' '.join(e)}" for r in capture["per_seed"] for e in r["timed"].get("events", [])]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--label", required=True)
    cap.add_argument("--seeds", default="1-3")
    cap.add_argument("--encoder", choices=("hashing", "model"), default="hashing")
    cmp = sub.add_parser("compare")
    cmp.add_argument("chain", type=Path)
    cmp.add_argument("event", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        print(compare(*(json.loads(p.read_text(encoding="utf-8")) for p in (args.chain, args.event))))
        return 0
    embedder = None
    if args.encoder == "hashing":
        from tests.doubles import HashingEmbedder

        embedder = HashingEmbedder()
    rows = [capture_seed(seed, args.label, embedder) for seed in parse_seeds(args.seeds)]
    result = {"summary": summarise(rows), "per_seed": rows}
    print(json.dumps(result["summary"], indent=1))
    print(save_result("temporal_ab", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
