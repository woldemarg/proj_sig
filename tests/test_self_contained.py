"""sig/ is self-contained: no code, model or config path resolves into the sibling eda/ or lac/ repos."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import fields
from pathlib import Path

import pytest

import ltir
from ltir.config import PROJECT_ROOT, Config, load_config

SIBLINGS = [PROJECT_ROOT.parent / "eda", PROJECT_ROOT.parent / "lac"]


def _inside(path: str | Path, roots: list[Path]) -> bool:
    p = Path(path).resolve()
    return any(p.is_relative_to(r.resolve()) for r in roots)


def test_no_module_loaded_from_sibling_repos(hashed_engine):
    import ltir.engines.eda.main_upd as eda
    import ltir.engines.lac.ontology_engine as ontology_engine
    import ltir.engines.lac.projector as projector

    for mod in pkgutil.walk_packages(ltir.__path__, "ltir."):
        if not mod.name.endswith("__main__"):
            importlib.import_module(mod.name)
    hashed_engine.ask("Why is margin lower for phones in the US?")  # exercises discovery->ontology->QA code paths
    loaded = [m.__name__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and _inside(m.__file__, SIBLINGS)]
    assert loaded == []
    assert not [p for p in sys.path if p and _inside(p, SIBLINGS)]
    for module in (eda, ontology_engine, projector):
        assert Path(module.__file__).resolve().is_relative_to(PROJECT_ROOT)


def test_config_paths_live_inside_the_project():
    cfg = load_config()
    for f in fields(Config):
        value = getattr(cfg, f.name)
        if isinstance(value, Path):
            assert Path(value).resolve().is_relative_to(PROJECT_ROOT), f.name


def test_sources_do_not_reference_sibling_repos():
    offenders = []
    for path in (PROJECT_ROOT / "ltir").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in ("../lac", "../eda", "sig_proj", "..\\lac", "..\\eda")):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


@pytest.mark.model
def test_embedding_model_loads_from_bundled_copy():
    from ltir.encoder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(load_config())
    emb.embed(["self-contained"])
    assert Path(emb.source).resolve().is_relative_to(PROJECT_ROOT / "models")
