"""Service settings: the root that builds every package's config from one environment (docs/09_operations.md §9.3;
every field: docs/11_reference.md §11.3).

Each package owns its config (``MinerConfig``, ``TopologyConfig``, ``QueryConfig``); ``load_settings`` fills them all,
and this service's own fields, from the process environment (``NAME`` = the field name in upper case). The
``PhenomenonThresholds`` the three packages must agree on are read once and shared. Nothing here reads a file:
host entry points call ``load_env_file()`` first; containers get their environment from compose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from attractor_topology.config import TopologyConfig
from graph_query_engine.config import QueryConfig
from insight_contracts import PhenomenonThresholds
from subgroup_miner.config import MinerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # the repository root; relative paths in settings resolve against it


@dataclass(frozen=True)
class Settings:
    workspace_dir: Path = Path("workspace")
    max_upload_mb: int = 200

    # admission into the graph (docs/03_insights.md §3.2, rules R4 and R7)
    min_insight_weight: float = 0.2
    max_insights_per_batch: int = 200

    # Neo4j mirror (docs/06_graph_and_storage.md §6.6)
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "sigv1"
    neo4j_load_batch_size: int = 5000

    # web
    web_host: str = "127.0.0.1"
    web_port: int = 8765

    miner: MinerConfig = MinerConfig()
    topology: TopologyConfig = TopologyConfig()
    query: QueryConfig = QueryConfig()

    def __post_init__(self) -> None:
        if not self.miner.thresholds == self.topology.thresholds == self.query.thresholds:
            raise ValueError("miner, topology and query must share one PhenomenonThresholds (build them with load_settings)")

    @property
    def thresholds(self) -> PhenomenonThresholds:
        return self.topology.thresholds


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """``KEY=value`` lines into the process environment (variables already set win); a missing file is no error.
    ``GEMMA_*`` lines are skipped: the provider settings and key belong to the LLM model broker, not this process."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if not key.strip().startswith("GEMMA_"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _coerce(raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        return Path(raw).expanduser()
    if isinstance(default, tuple):
        return tuple(float(x) for x in raw.split(","))
    return raw


def _build(cls: type, overrides: dict[str, Any], used: set[str], **parts: Any) -> Any:
    """One config: defaults ← environment ← overrides; a relative path resolves against the repository root.
    An empty environment value clears a string field and leaves any other field at its default."""
    values: dict[str, Any] = dict(parts)
    for f in fields(cls):
        if f.name in parts:
            continue
        if f.name in overrides:
            values[f.name] = overrides[f.name]
            used.add(f.name)
        else:
            raw = os.environ.get(f.name.upper())
            if raw is not None and (raw != "" or isinstance(f.default, str)):
                values[f.name] = _coerce(raw, f.default)
        value = values.get(f.name, f.default)
        if isinstance(value, Path) and not value.is_absolute():
            values[f.name] = (PROJECT_ROOT / value).resolve()
    return cls(**values)


SECTIONS = (("miner", MinerConfig), ("topology", TopologyConfig), ("query", QueryConfig))
_NAMES = [
    f.name
    for cls in (Settings, PhenomenonThresholds, *(c for _, c in SECTIONS))
    for f in fields(cls)
    if f.name not in {"thresholds", *dict(SECTIONS)}
]
assert len(_NAMES) == len(set(_NAMES)), "a setting name must be unique across sections: it is its environment variable"


def load_settings(**overrides: Any) -> Settings:
    """Settings from the process environment; ``overrides`` set any field of any section by its name."""
    used: set[str] = set()
    thresholds = _build(PhenomenonThresholds, overrides, used)
    sections = {name: _build(cls, overrides, used, thresholds=thresholds) for name, cls in SECTIONS}
    settings = _build(Settings, overrides, used, **sections)
    unknown = set(overrides) - used
    if unknown:
        raise TypeError(f"unknown settings: {sorted(unknown)}")
    return settings
