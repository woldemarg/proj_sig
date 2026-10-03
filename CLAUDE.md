# CLAUDE.md

Read [`AGENTS.md`](AGENTS.md) first; it holds the rules (Rule 0: `scripts/check.py` must pass). This file only adds what is specific to Claude Code on this machine.

- Interpreter: `.venv\Scripts\python.exe` (never `python` from PATH, never the base environment the venv is layered on).
- Working directory for commands: the repository root. Anything outside the repository is out of scope: never edit it, never import from it.
- Temporary scripts, dumps and scratch workspaces go to the session scratchpad directory, not into the repo and never into `workspace/`.
- Shell: both Bash (Git Bash) and PowerShell are available. Heredocs that contain backslashes or `\n` are mangled by the shell layer — put edit scripts in a file and run them with the interpreter.
- Long runs (the full gate ≈ 85 s with the model and browser tests; `python -m ltir demo` ≈ 15 s cold; `scripts/compare_embedders.py` several minutes) go in the background; read the log when it finishes rather than polling.
- Docker Desktop (with the NVIDIA runtime) is available. `scripts/compose_check.py` runs as compose project `sig-check` with its own volumes and host ports — never `down -v` the user's project `sig` (its volumes hold their containerised workspace and Neo4j data). A first image build takes several minutes, `--tests` adds the gate inside the image on CPU (≈ 5 min): run both in the background.
- The first model load warms CUDA (~10 s). Tests that need it carry the `model` marker; the suite skips them only when the configured model folder (`models/Qwen3-Embedding-0.6B/` by default) is missing.
- Measurement scripts in `scripts/` write to `.scratch/`; `scripts/eval_answers.py` and `python -m ltir query` without `--no-llm` call the LLM behind `LLM_BASE_URL` (by default the gateway with `.env.gemma`; a paid call) — run them only within an agreed live evaluation, like `scripts/compose_check.py --live-llm`.
- Before touching a stage, open its chapter (`docs/README.md` maps chapters to code): statistics in `docs/02_discovery.md` and `03_insights.md`, vectors and every embedded string in `04_representation.md`, anchor mechanics in `05_latent_anchors.md`, the prompt in `07_question_answering.md`; `docs/11_reference.md` indexes every formula and parameter.
- The user's `workspace/`, `.env` and `.env.gemma` are off limits unless asked (AGENTS.md §7). Experiments use a scratch `WORKSPACE_DIR`.
- When a representation or canonical version bump is unavoidable, say in the summary that the user's existing workspace will be refused until `python -m ltir migrate --yes` rebuilds it (the old copy is kept). Never run that on `workspace/` unless asked.
