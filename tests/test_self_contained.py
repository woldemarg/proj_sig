"""The repository is self-contained: code, models and paths resolve inside it or inside the Python environment."""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import site
import sys
import sysconfig
from dataclasses import fields
from pathlib import Path

import pytest
from conftest import ask

from insight_graph_service.core.settings import PROJECT_ROOT, load_settings

ROOT = PROJECT_ROOT.resolve()
PACKAGES = [
    "insight_contracts",
    "subgroup_miner",
    "attractor_topology",
    "graph_query_engine",
    "insight_graph_service",
    "evidence_narrator_service",
    "llm_model_broker",
]
# the interpreter, its standard library and site-packages (also of a base environment the venv is layered on)
ENVIRONMENT = sorted(
    {
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        *site.getsitepackages(),
        site.getusersitepackages(),
        *sysconfig.get_paths().values(),
    }
)
ABSOLUTE = re.compile(r"\b[A-Za-z]:[\\/]|(?<![\w.:/])/(?:home|Users|mnt|media)/")  # a machine path
PARENT = re.compile(r"(?:^|(?<=[\s\"'`(\\/]))\.\.[\\/][^\s\"'`),;]*")  # a relative path that climbs a folder


def _inside(path: str | Path, roots: list[Path]) -> bool:
    p = Path(path).resolve()
    return any(p.is_relative_to(r) for r in roots)


def _docstrings(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def test_loaded_code_comes_from_the_repository_or_the_environment(hashed_engine):
    import attractor_topology.vendor.lac.ontology_engine as ontology_engine
    import subgroup_miner.vendor.eda.main_upd as eda

    for name in PACKAGES:
        package = importlib.import_module(name)
        for mod in pkgutil.walk_packages(package.__path__, f"{name}."):
            if not mod.name.endswith("__main__"):
                importlib.import_module(mod.name)
    ask(hashed_engine, "Why is margin lower for phones in the US?")  # exercises discovery->ontology->evidence->narration
    allowed = [ROOT, *(Path(r).resolve() for r in ENVIRONMENT if r)]
    loaded = [m.__name__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and not _inside(m.__file__, allowed)]
    assert loaded == []
    assert not [p for p in sys.path if p and not _inside(p, allowed)]
    for module in (eda, ontology_engine):  # the vendored engines, not an installed copy
        assert Path(module.__file__).resolve().is_relative_to(ROOT)


def test_config_paths_live_inside_the_project():
    settings = load_settings()
    for section in (settings, settings.miner, settings.topology, settings.query):
        for f in fields(section):
            value = getattr(section, f.name)
            if isinstance(value, Path):
                assert Path(value).resolve().is_relative_to(ROOT), f.name


def test_sources_hold_no_path_outside_the_repository():
    """No machine path in any string; a climbing relative path only in a docstring, and inside the repository."""
    offenders = []
    for path in [p for name in PACKAGES for p in (ROOT / name).rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstrings(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            where = f"{path.relative_to(ROOT)}:{node.lineno}"
            if ABSOLUTE.search(node.value):
                offenders.append(where)
            for ref in PARENT.findall(node.value):
                if id(node) not in docstrings or not (path.parent / ref).resolve().is_relative_to(ROOT):
                    offenders.append(f"{where} {ref}")
    assert offenders == []


@pytest.mark.model
def test_embedding_model_loads_from_bundled_copy():
    from attractor_topology.encoder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(load_settings().topology)
    emb.embed(["self-contained"])
    assert emb.local.resolve().is_relative_to(ROOT / "models") and emb.dim == 384
