"""Golden regression: without new options the demo graph and its evidence prompt stay exactly as recorded.

The demo (retail, no geo) is ingested with the hashing double; its snapshot is summarised (pattern ids and their
statistics, the props each node kind carries, every edge) and compared with ``tests/golden/``, and so is the
evidence prompt for one question. A change that alters either regenerates them explicitly and explains the diff
in its PR: ``python -m pytest tests/test_golden.py --update-golden``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

GOLDEN = Path(__file__).parent / "golden"
QUESTION = "Why is margin lower for phones in the US?"
BATCH_ID = re.compile(r"B\d{8}T\d{6}-[0-9a-f]{6}")  # a batch id carries its creation time
VOLATILE = {"created_at"}  # props that differ between two builds of the same data
FLOAT = re.compile(r"(-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+e[-+]?\d+)")


def _round(value: Any) -> Any:
    """Floats to 6 significant digits, for readable golden lines."""
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def _edge(e: dict[str, Any]) -> str:
    props = " ".join(f"{k}={_round(v)}" for k, v in sorted(e["props"].items()))
    return f"{e['source']} -{e['type']}-> {e['target']} {_round(e['weight'])} {props}".rstrip()


def _pattern(p: dict[str, Any]) -> str:
    scope = " & ".join(f"{c['attribute']}={c['value']}" for c in p["conditions"])
    stats = " | ".join(f"{k} {_round(p[k])}" for k in ("support", "p_value", "p_adjusted", "stability", "weight"))
    shifts = "; ".join(f"{s['metric']} {_round(s['robust_z']):+} @{_round(s['local_median'])}" for s in p["shifts"])
    return f"{p['id']} {scope} | target {p['target']} | {stats} | {shifts}"


def _same(a: str, b: str) -> bool:
    """The same text with floats equal to 1e-4: the float32 anchor maths differs in its last bits between BLAS builds."""
    pa, pb = FLOAT.split(a), FLOAT.split(b)
    return len(pa) == len(pb) and all(
        x == y if i % 2 == 0 else math.isclose(float(x), float(y), rel_tol=1e-4, abs_tol=1e-12) for i, (x, y) in enumerate(zip(pa, pb))
    )


def summarise(snapshot: dict[str, Any]) -> dict[str, Any]:
    """What the golden file holds: the snapshot without volatile fields, vectors or free text."""
    snapshot = json.loads(BATCH_ID.sub("<batch>", json.dumps(snapshot)))
    nodes = snapshot["nodes"]
    kinds = sorted({n["kind"] for n in nodes})
    return {
        "version": snapshot["version"],
        "props": {k: sorted(set().union(*(n["props"] for n in nodes if n["kind"] == k)) - VOLATILE) for k in kinds},
        "datasets": _round([n["props"] for n in nodes if n["kind"] == "Dataset"]),
        "edge_counts": dict(sorted(snapshot["stats"]["edge_counts"].items())),
        "patterns": sorted(_pattern(n["props"]) for n in nodes if n["kind"] == "Pattern"),
        "edges": sorted(_edge(e) for e in snapshot["edges"]),
    }


def test_demo_graph_and_prompt_match_the_golden_files(hashed_engine, request):
    summary = summarise(json.loads(hashed_engine.ws.graph_path.read_text(encoding="utf-8")))
    prompt = BATCH_ID.sub("<batch>", hashed_engine.evidence(QUESTION).evidence_prompt) + "\n"
    graph_file, prompt_file = GOLDEN / "retail_demo.json", GOLDEN / "retail_demo_prompt.txt"
    if request.config.getoption("--update-golden"):
        GOLDEN.mkdir(exist_ok=True)
        graph_file.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        prompt_file.write_text(prompt, encoding="utf-8")
        pytest.skip("golden files rewritten; review and commit the diff")
    golden = json.loads(graph_file.read_text(encoding="utf-8"))
    assert summary.keys() == golden.keys()
    for section in ("version", "props", "datasets", "edge_counts"):
        assert summary[section] == golden[section], f"golden section {section!r} changed"
    for section in ("patterns", "edges"):
        now, then = summary[section], golden[section]
        assert len(now) == len(then), f"golden section {section!r}: {len(now)} lines, recorded {len(then)}"
        changed = [(a, b) for a, b in zip(now, then) if not _same(a, b)]
        assert not changed, f"golden section {section!r} changed, first line:\n now:      {changed[0][0]}\n recorded: {changed[0][1]}"
    assert prompt == prompt_file.read_text(encoding="utf-8")
