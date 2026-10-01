# CLAUDE.md

Read [`AGENTS.md`](AGENTS.md) first; it holds the rules (Rule 0: `scripts/check.py` must pass). This file only adds what is specific to Claude Code on this machine.

- Interpreter: `.venv\Scripts\python.exe` (a venv layered on `D:/conda_envs/env_ont`; never `python` from PATH, never the conda env directly).
- Working directory for commands: `D:/llm/sig_proj/sig`. The sibling folders `../eda`, `../lac` and `D:/llm/spectr/*` are reference only — read, never edit, never import.
- Temporary scripts, dumps and scratch workspaces go to the session scratchpad directory, not into the repo and never into `workspace/`.
- Shell: both Bash (Git Bash) and PowerShell are available. Heredocs that contain backslashes or `\n` are mangled by the shell layer — put edit scripts in a file and run them with the interpreter.
- Long runs (the full gate ≈ 90 s with the model and browser tests; `python -m ltir demo` ≈ 20 s cold) go in the background; read the log when it finishes rather than polling.
- The first model load warms CUDA (~10 s). Tests that need it carry the `model` marker; the suite skips them only when `models/paraphrase-multilingual-MiniLM-L12-v2/` is missing.
- Before touching statistics, vectors or attractor mechanics, open SDD 16 (`docs/sdd/16_core_mathematics.md`); before touching any string a user, the LLM or the encoder sees, open SDD 17 (`docs/sdd/17_textual_contracts.md`).
- The user's `workspace/` and `.env` are off limits unless asked (AGENTS.md §7). Experiments use a scratch `WORKSPACE_DIR`.
- When a representation or canonical version bump is unavoidable, say in the summary that the user's existing workspace will be refused and must be reset and re-ingested.
