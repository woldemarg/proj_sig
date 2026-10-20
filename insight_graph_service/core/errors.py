"""The graph service's refusal (docs/09_operations.md §9.4): a code, which the HTTP API maps to a status, and a message."""

from __future__ import annotations


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
