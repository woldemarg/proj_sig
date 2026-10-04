"""Package boundaries (docs/12_architecture.md): which repository packages each package imports, and which
third-party ones. The table below is the whole set of allowed dependencies; a new one is a design decision.

* the shared kernel ``insight_contracts`` imports nothing outside the standard library;
* the three libraries import only the kernel (each is usable alone);
* the graph service's core imports no web or HTTP framework; only its Neo4j mirror imports the driver;
* the narrator and the model broker are separate deployables: the narrator imports only the kernel, the broker nothing.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = {
    "insight_contracts",
    "subgroup_miner",
    "attractor_topology",
    "graph_query_engine",
    "insight_graph_service",
    "evidence_narrator_service",
    "llm_model_broker",
}
KERNEL = {"insight_contracts"}
# package -> (repository packages it may import besides itself, third-party top-level modules it may import)
RULES = {
    "insight_contracts": (set(), set()),
    "subgroup_miner": (KERNEL, {"numpy", "pandas", "scipy", "pysubgroup"}),
    "attractor_topology": (KERNEL, {"numpy", "sklearn", "sentence_transformers", "torch"}),
    "graph_query_engine": (KERNEL, {"numpy", "sklearn"}),
    "insight_graph_service": (set(), set()),  # the package itself: its two parts below
    "insight_graph_service.core": (
        KERNEL | {"subgroup_miner", "attractor_topology", "graph_query_engine"},
        {"numpy", "pandas", "neo4j", "huggingface_hub"},
    ),
    "insight_graph_service.server": (
        KERNEL | {"insight_graph_service.core", "attractor_topology", "graph_query_engine"},
        {"numpy", "sklearn", "fastapi", "pydantic", "uvicorn", "torch"},
    ),
    "evidence_narrator_service": (KERNEL, {"fastapi", "pydantic", "uvicorn", "httpx"}),
    "llm_model_broker": (set(), {"fastapi", "pydantic", "uvicorn", "httpx"}),
}


def _module(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _imports(path: Path) -> set[str]:
    """Every absolute module a file imports, relative imports resolved."""
    package = _module(path).rsplit(".", 1)[0]  # a package's __init__ too: its module path ends in ".__init__"
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add(importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module)
    return found


def _owner(module: str) -> str:
    return max((p for p in RULES if module == p or module.startswith(p + ".")), key=len)


def _within(module: str, packages: set[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in packages)


def test_every_import_is_in_the_table():
    wrong = []
    for top in sorted(REPO):
        for path in (ROOT / top).rglob("*.py"):
            module = _module(path)
            owner = _owner(module)
            repo_ok, third_ok = RULES[owner]
            for imported in _imports(path):
                head = imported.split(".")[0]
                if head in REPO:
                    allowed = _within(imported, repo_ok | {owner})
                elif head in sys.stdlib_module_names or head == "__future__":
                    allowed = True
                else:
                    allowed = head in third_ok
                if not allowed:
                    wrong.append(f"{path.relative_to(ROOT)} imports {imported}")
    assert wrong == []


def test_the_neo4j_driver_stays_in_the_mirror():
    users = {_module(p) for top in REPO for p in (ROOT / top).rglob("*.py") if any(m.split(".")[0] == "neo4j" for m in _imports(p))}
    assert users <= {"insight_graph_service.core.neo4j_mirror"}
