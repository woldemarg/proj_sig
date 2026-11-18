"""Local quality gate (AGENTS.md Rule 0): the checks CI runs (.github/workflows/ci.yml), stage by stage.

    python scripts/check.py                  # the gate: every stage, including the model + browser tests
    python scripts/check.py lint types       # chosen stages: lint, types, test, security
    python scripts/check.py --quick          # mid-change feedback: lint, types and the fast tests

Stages: lint (ruff check, ruff format --check, uv lock --check), types (mypy), test (pytest), security (gitleaks over
the git history, trivy over the lockfile, dev groups included, and the Dockerfiles: any finding of any severity fails). A tool missing from PATH is reported as skipped, never as
passed (in CI, as failed). Runs every step even after a failure and exits non-zero if any failed. Works on Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("lint", "types", "test", "security")
TRIVY_SKIP = [".venv", "models", "workspace", "data", ".scratch", ".pytest_tmp"]  # not ours or not shipped


def run_step(name: str, cmd: list[str]) -> bool | None:
    """Run one step from the repo root; returns success, or None when its tool is not installed."""
    if shutil.which(cmd[0]) is None:
        if os.environ.get("CI"):  # CI installs every tool: a missing one fails
            print(f"\n=== {name}: FAILED ({cmd[0]} not on PATH)", flush=True)
            return False
        print(f"\n=== {name}: SKIPPED ({cmd[0]} not on PATH)", flush=True)
        return None
    print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
    start = time.perf_counter()
    code = subprocess.call(cmd, cwd=ROOT)
    print(f"=== {name}: {'ok' if code == 0 else f'FAILED (exit {code})'} in {time.perf_counter() - start:.1f}s", flush=True)
    return code == 0


def build_steps(stages: list[str], quick: bool) -> list[tuple[str, list[str]]]:
    """The (name, command) steps of the chosen stages; CI runs the same commands."""
    py = sys.executable
    pytest_cmd = [py, "-m", "pytest"]  # options in pytest.ini
    if quick:
        pytest_cmd += ["-m", "not model and not browser"]
    trivy_skip = [arg for d in TRIVY_SKIP for arg in ("--skip-dirs", d)]
    by_stage = {
        "lint": [
            ("ruff check", [py, "-m", "ruff", "check", "."]),  # pyproject.toml excludes what is not ours
            ("ruff format", [py, "-m", "ruff", "format", "--check", "."]),
            ("uv lock", ["uv", "lock", "--check"]),
        ],
        "types": [("mypy", [py, "-m", "mypy"])],  # files and settings in pyproject.toml
        "test": [("pytest", pytest_cmd)],
        "security": [
            ("gitleaks", ["gitleaks", "git", "--redact", "--no-banner", "."]),
            (
                "trivy fs",
                [
                    "trivy",
                    "fs",
                    "--scanners",
                    "vuln,misconfig",
                    "--include-dev-deps",
                    "--exit-code",
                    "1",
                    *trivy_skip,
                    ".",
                ],
            ),
        ],
    }
    return [step for stage in stages for step in by_stage[stage]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", metavar="stage", help=f"any of {', '.join(STAGES)} (default: all)")
    parser.add_argument("--quick", action="store_true", help="lint, types and the tests without the model and browser ones")
    args = parser.parse_args(argv)
    if unknown := set(args.stages) - set(STAGES):
        parser.error(f"unknown stage(s): {', '.join(sorted(unknown))}")
    stages = args.stages or (["lint", "types", "test"] if args.quick else list(STAGES))
    results = [(name, run_step(name, cmd)) for name, cmd in build_steps(stages, args.quick)]
    print("\n" + " | ".join(f"{name}: {'skipped' if ok is None else 'ok' if ok else 'FAILED'}" for name, ok in results))
    return 0 if all(ok is not False for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
