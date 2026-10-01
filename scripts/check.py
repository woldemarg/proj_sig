"""Local quality gate (AGENTS.md Rule 0): ruff check, ruff format --check, pytest.

    python scripts/check.py          # the gate: everything, including model + browser tests
    python scripts/check.py --quick  # mid-change feedback: lint + the fast test subset

Runs every step even after a failure and exits non-zero if any step failed, so one
run reports the whole picture. Works the same on Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINT_PATHS = ["ltir", "tests", "scripts"]


def run_step(name: str, cmd: list[str]) -> bool:
    """Run one step from the repo root; prints its outcome and returns success."""
    print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
    start = time.perf_counter()
    code = subprocess.call(cmd, cwd=ROOT, env={**os.environ, "LTIR_NO_DOTENV": "1"})
    print(f"=== {name}: {'ok' if code == 0 else f'FAILED (exit {code})'} in {time.perf_counter() - start:.1f}s", flush=True)
    return code == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the model-backed and browser tests")
    args = parser.parse_args(argv)
    py = sys.executable
    pytest_cmd = [py, "-m", "pytest", "-q", "-p", "no:warnings"]
    if args.quick:
        pytest_cmd += ["-m", "not model and not browser"]
    steps = [
        ("ruff check", [py, "-m", "ruff", "check", *LINT_PATHS]),
        ("ruff format", [py, "-m", "ruff", "format", "--check", *LINT_PATHS]),
        ("pytest", pytest_cmd),
    ]
    results = [(name, run_step(name, cmd)) for name, cmd in steps]
    print("\n" + " | ".join(f"{name}: {'ok' if ok else 'FAILED'}" for name, ok in results))
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
