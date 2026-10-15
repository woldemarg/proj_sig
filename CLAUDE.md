# CLAUDE.md

@AGENTS.md

The rules are in AGENTS.md (imported above). This file only adds what is specific to Claude Code.

- Temporary scripts, dumps and scratch workspaces go to the session scratchpad directory, not into the repo and never into `workspace/`.
- On Windows (Git Bash or PowerShell) heredocs that contain backslashes or `\n` are mangled by the shell layer: put edit scripts in a file and run them with the venv interpreter.
- Long runs go in the background; read the log when they finish rather than polling. That covers the full gate (≈ 2 min with the model and browser tests), `scripts/multilingual_benchmark.py` (several minutes), and `scripts/compose_check.py` (a first image build takes several minutes; `--tests` adds the gate inside the image on CPU, ≈ 5 min).
- `scripts/compose_check.py` runs as compose project `sig-check` with its own volumes and host ports. Never `down -v` the user's project `sig`: its volumes hold their containerised workspace, model copy and Neo4j data.
- The first model load takes ~10 s. Tests that need it carry the `model` marker; the suite skips them only when the configured model folder (`models/Qwen3-Embedding-0.6B/` by default) is missing.
- `scripts/eval_answers.py` and a chat question with `use_llm` call the LLM behind the broker, which is a paid call. Run them only within an agreed live evaluation, like `scripts/compose_check.py --live-llm`.
