"""Golden regression: the demo graph (retail, no geo, hashing double) and one evidence prompt against ``tests/golden/``.

Regenerate after an intended change: ``python -m pytest tests/test_golden.py --update-golden``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

GOLDEN = Path(__file__).parent / "golden"
QUESTION = "Why is margin lower for phones in the US?"
BATCH_ID = re.compile(r"B\d{8}T\d{6}-[0-9a-f]{6}")  # a batch id carries its creation time
FLOAT = re.compile(r"(?<![\w.-])(-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+e[-+]?\d+)(?![\w.])")  # not inside an id like P-4684602e780d
EXACT = ("version", "nodes", "props", "datasets", "edge_counts")


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
    """The same text with floats equal to 1e-4 relative: the float32 anchor maths differs in its last bits between BLAS builds."""
    pa, pb = FLOAT.split(a), FLOAT.split(b)
    return len(pa) == len(pb) and all(
        x == y if i % 2 == 0 else math.isclose(float(x), float(y), rel_tol=1e-4) for i, (x, y) in enumerate(zip(pa, pb))
    )


def summarise(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Node counts, how many nodes of each kind carry each prop, the dataset, every pattern's statistics and every edge."""
    snapshot = json.loads(BATCH_ID.sub("<batch>", json.dumps(snapshot)))
    nodes = snapshot["nodes"]
    kinds = sorted({n["kind"] for n in nodes})
    return {
        "version": snapshot["version"],
        "nodes": dict(sorted(Counter(n["kind"] for n in nodes).items())),
        "props": {k: dict(sorted(Counter(p for n in nodes if n["kind"] == k for p in n["props"]).items())) for k in kinds},
        "datasets": _round([n["props"] for n in nodes if n["kind"] == "Dataset"]),
        "edge_counts": dict(sorted(snapshot["stats"]["edge_counts"].items())),
        "patterns": sorted(_pattern(n["props"]) for n in nodes if n["kind"] == "Pattern"),
        "edges": sorted(_edge(e) for e in snapshot["edges"]),
    }


def test_demo_graph_and_prompt_match_the_golden_files(hashed_engine, request):
    summary = summarise(hashed_engine.ws.load_graph())
    prompt = BATCH_ID.sub("<batch>", hashed_engine.evidence(QUESTION).evidence_prompt) + "\n"
    graph_file, prompt_file = GOLDEN / "retail_demo.json", GOLDEN / "retail_demo_prompt.txt"
    if request.config.getoption("--update-golden"):
        graph_file.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        prompt_file.write_text(prompt, encoding="utf-8")
        pytest.skip("golden files rewritten; review and commit the diff")
    golden = json.loads(graph_file.read_text(encoding="utf-8"))
    assert summary.keys() == golden.keys() == {*EXACT, "patterns", "edges"}  # a new section must be compared somewhere
    assert {k: summary[k] for k in EXACT} == {k: golden[k] for k in EXACT}
    for section in ("patterns", "edges"):
        now, then = summary[section], golden[section]
        assert len(now) == len(then), f"golden section {section!r}: {len(now)} lines, recorded {len(then)}"
        changed = [(a, b) for a, b in zip(now, then) if not _same(a, b)]
        assert not changed, f"golden section {section!r} changed, first line:\n now:      {changed[0][0]}\n recorded: {changed[0][1]}"
    assert prompt == prompt_file.read_text(encoding="utf-8")
