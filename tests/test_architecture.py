"""Layer boundaries (docs/12_architecture.md §12.2): the import table below is the whole set of allowed
cross-layer imports; every one points down the stack, and a new one is a design decision, not a side effect.

The analytical core (analysis, retrieval, storage) imports no presentation, chat or LLM code and none of their
third-party frameworks; only the Neo4j mirror imports the driver; the LLM gateway imports nothing from ``ltir``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = {"config", "models"}  # everyone may import these
ALLOWED = {  # layer -> the other layers it may import, besides SHARED
    "config": set(),
    "models": set(),
    "fileio": set(),
    "engines": {"fileio"},  # the vendored journal writes atomically
    "analysis": {"engines"},
    "storage": {"engines", "fileio"},
    "retrieval": {"analysis"},
    "llm_client": set(),
    "answering": {"retrieval", "llm_client"},
    "engine": {"analysis", "retrieval", "storage", "answering", "llm_client"},
    "migrate": {"engine", "storage"},
    "evaluation": {"engine", "retrieval"},
    "web": {"engine", "storage", "analysis", "evaluation"},  # storage: its errors; analysis: labels; evaluation: the demo data
    "cli": {"engine", "storage", "evaluation", "migrate", "web", "llm_client"},  # the entry point: one command per layer
}
CORE = {"analysis", "retrieval", "storage", "models", "config", "engines", "fileio"}
FRAMEWORKS = {"fastapi", "uvicorn", "starlette", "plotly", "httpx"}  # presentation and HTTP; the core needs none


def _imports(path: Path) -> set[str]:
    """Every absolute module name a file imports, relative imports resolved, ``from ltir import x`` as ``ltir.x``."""
    module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    package = module.rsplit(".", 1)[0]  # a package's __init__ too: its module path ends in ".__init__"
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
            found.add(base)
            if base in {"ltir", "llm_gateway"}:
                found |= {f"{base}.{a.name}" for a in node.names}
    return found


def _sources(folder: str) -> list[Path]:
    return [p for p in (ROOT / folder).rglob("*.py") if p.name != "__main__.py" or p.parent.name != "ltir"]


def test_imports_point_down_the_stack():
    wrong = []
    for path in _sources("ltir"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if len(parts) == 2 and parts[1] == "__init__":
            continue
        layer = parts[1]
        for module in _imports(path):
            top = module.split(".")[0]
            if top == "ltir" and module != "ltir":
                target = module.split(".")[1]
                if target != layer and target not in SHARED | ALLOWED[layer]:
                    wrong.append(f"{path.relative_to(ROOT)} imports {module}")
            elif layer in CORE and top in FRAMEWORKS:
                wrong.append(f"{path.relative_to(ROOT)} imports {module}")
    assert wrong == []


def test_the_neo4j_driver_stays_in_the_mirror_and_the_gateway_stays_independent():
    users = {p.relative_to(ROOT).as_posix() for p in _sources("ltir") if any(m.split(".")[0] == "neo4j" for m in _imports(p))}
    assert users <= {"ltir/storage/neo4j_mirror.py"}
    assert not [p for p in _sources("llm_gateway") if any(m.split(".")[0] == "ltir" for m in _imports(p))]
