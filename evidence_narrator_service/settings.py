"""The narrator's settings, from the process environment (``NAME`` = the field name in upper case)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class NarratorSettings:
    insight_graph_url: str = "http://127.0.0.1:8765"  # the graph service: its POST /api/search gives the evidence
    insight_graph_timeout_s: float = 300.0  # the first question after a commit builds the literal catalog (CPU: tens of seconds)
    llm_base_url: str = "http://127.0.0.1:8080/v1"  # the LLM model broker: it holds the upstream, the model and the key
    llm_timeout_s: float = 120.0
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1200

    @classmethod
    def from_env(cls, **overrides: Any) -> NarratorSettings:
        """Defaults <- environment <- overrides; an empty value clears a string and keeps any other default."""
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f.name.upper())
            if raw is not None and (raw != "" or isinstance(f.default, str)):
                values[f.name] = type(f.default)(raw)
        return cls(**{**values, **overrides})
