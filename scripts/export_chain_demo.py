"""Export the demo graph for an external chain evaluation (the lac repository's ``eval_multihop`` suite reads it as its
``sig_demo`` corpus): one JSON with the patterns (id, headline, canonical document, target, scope literals, the committed
insight vector, planted mechanism), mechanism -> pattern ids (the ground truth of ``experiment.planted_scopes``) and the
snapshot's edges. Builds (or reuses) a scratch workspace with the demo ingested; ``workspace/`` is never touched.

    python scripts/export_chain_demo.py <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the packages live one level up

from experiment import phenomenon_of, planted_scopes  # noqa: E402
from scratch import build_engine, demo_engine, scratch_dir  # noqa: E402

EDGE_KINDS = {"SPECIALIZES", "GENERALIZES", "SIBLING", "CONTRASTS", "CO_LOCATED", "ACTIVATES", "RELATED_TO", "CO_OCCURS"}


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    ws = scratch_dir("export_chain_demo") / "ws"
    engine = build_engine(ws) if (ws / "graph" / "snapshot.json").exists() else demo_engine("export_chain_demo")
    state = engine.prepared()
    g, frame = state.graph, state.frame
    truth = planted_scopes()
    patterns = [
        {
            "id": pid,
            "headline": g.nodes[pid]["label"],
            "document": g.canonical_document(pid),
            "target": g.insight(pid).target,
            "conditions": [c.expr for c in g.insight(pid).conditions],
            "vector": [float(v) for v in frame.patterns[pid]],
            "mechanism": phenomenon_of(g.insight(pid), truth),
        }
        for pid in sorted(g.insights)
    ]
    mechanisms: dict[str, list[str]] = {}
    for p in patterns:
        if p["mechanism"]:
            mechanisms.setdefault(p["mechanism"], []).append(p["id"])
    edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "type": e["type"],
            "plane": e.get("plane"),
            "weak": bool(e.get("props", {}).get("weak", False)),
        }
        for e in g.edges
        if e["type"] in EDGE_KINDS
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"patterns": patterns, "mechanisms": mechanisms, "edges": edges}, indent=1), encoding="utf-8")
    print(
        f"{len(patterns)} patterns, {len(patterns[0]['vector'])}-d vectors, sets { ({k: len(v) for k, v in mechanisms.items()}) }, {len(edges)} edges -> {out}"
    )


if __name__ == "__main__":
    main()
