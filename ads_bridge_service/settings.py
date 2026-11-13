"""The bridge's settings, from the process environment (``NAME`` = the field name in upper case)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class AdsSettings:
    ads_url: str = "http://127.0.0.1:8000"  # the ADS orchestrator: sessions, datasets, jobs
    ads_llm_url: str = "http://127.0.0.1:8080"  # ADS's LLM gateway: the final report's model
    ads_graph_url: str = "http://127.0.0.1:8765"  # the graph service: POST /api/geo/{dataset}/aoi gives a table's rows
    ads_narrator_url: str = "http://127.0.0.1:8765"  # the narrator: POST /api/chat/query gives the zone's evidence
    ads_job_timeout_s: float = 1200.0  # how long an ADS job may run
    ads_poll_s: float = 5.0  # between two reads of an ADS job's status

    @classmethod
    def from_env(cls, **overrides: Any) -> AdsSettings:
        """Defaults <- environment <- overrides; an empty value clears a string and keeps any other default."""
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f.name.upper())
            if raw is not None and (raw != "" or isinstance(f.default, str)):
                values[f.name] = type(f.default)(raw)
        return cls(**{**values, **overrides})
