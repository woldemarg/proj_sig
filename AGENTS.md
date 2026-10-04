# AGENTS.md — Working rules for coding agents in `sig/`

This file defines the default behaviour of AI coding agents (and is good advice for people) working in this repository. Priorities: correctness, statistical honesty, maintainability, fast feedback. Scope: **this folder (`sig/`) only** — see §Boundaries.

---

## Rule 0 — `python scripts/check.py` must pass before you report done (NON-NEGOTIABLE)

Every task that changes a line of Python ends with the gate passing:

```powershell
.venv\Scripts\python.exe scripts\check.py          # Windows
.venv/bin/python scripts/check.py                  # macOS / Linux
```

The gate is `ruff check` + `ruff format --check` + the **whole** pytest suite (≈ 2 min; model-backed and browser tests included when the local model and Chromium are present on the host).

- `scripts/check.py --quick` (lint + the fast tests) is for feedback **mid-change**. It is not the gate.
- A targeted `pytest tests/test_x.py` is not the gate either: the E2E, persistence, narrator and UI tests are where cross-package regressions show up.
- "Small / obvious / comment-only" is not an exemption; formatting and import order are exactly what small edits break.
- Do not silence the gate with `# noqa`, `# type: ignore`, new `ignore` entries in `pyproject.toml`, or skipped tests. Fix the cause.
- If the gate cannot run (no venv, no model, no Chromium), say so in the summary and name the stages that were skipped. Never imply it passed.

Docs-only changes (`docs/`, `README.md`, `AGENTS.md`, the packages' `README.md`) may skip the gate; any Python, `pyproject.toml`, `pytest.ini` or `requirements*.txt` change may not. A change to a `Dockerfile`, `compose.yaml`, `nginx.conf` or `constraints.txt` also runs `python scripts/compose_check.py` (Docker must be running).

---

## Repository layout

| Path | Role |
|---|---|
| `insight_contracts/` | the shared kernel, **standard library only**: `insight` (the `Insight` and its parts, `PhenomenonThresholds`), `graph` (edge types, node ids, the snapshot contract), `text` (how an insight reads), `payload` (`EvidencePayload`, the graph service → narrator contract) |
| `subgroup_miner/` | table → validated, weighted insights: `ingestion`, `discovery` (the EDA adapter), `selection` (validity rules, weight), `lattice`, `describe`; **vendored** `vendor/eda/main_upd.py` (divergences in `vendor/PROVENANCE.md`) |
| `attractor_topology/` | insights → vectors → latent attractors: `canonical`, `encoder` (Qwen3 only), `ontology` (the lac adapter), `models`; **vendored** `vendor/lac/` (`vendor/PROVENANCE.md`) |
| `graph_query_engine/` | question → evidence over the dual graph: `graph` (`DualGraph`, `LatentFrame`), `question`, `seeds`, `traversal`, `evidence`, `search`, `ports` |
| `insight_graph_service/` | the graph service. `core/` (no web framework): `settings`, `engine`, `batch`, `snapshot`, `workspace`, `chunk_journal` (vendored, `core/PROVENANCE.md`), `fileio`, `neo4j_mirror` + `cypher/`, `demo`, `model_store`. `server/`: `app`, `views`. Its `Dockerfile` (targets `runtime`, `test`) |
| `evidence_narrator_service/` | the chat service: `narration`, `llm_client`, `settings`, `app`; imports only `insight_contracts` |
| `llm_model_broker/` | the LLM service: one OpenAI-compatible endpoint in front of the upstream model; imports nothing from the repository |
| `sig_web_console/` | the static UI (`static/`: `index.html`, `app.js`, `style.css`, vendored browser libraries in `vendor/`, `PROVENANCE.md`), `nginx.conf`, its `Dockerfile` |
| `compose.yaml`, `.dockerignore`, `constraints.txt` | the service suite in start order (`docs/12_architecture.md` §12.5); the exact package versions every image installs with |
| `tests/` | pytest suite (`conftest.py` holds the fixtures, `FakeLLM`, `make_engine`, `ask`; `doubles.py` the deterministic 384-d test embedder); markers `model`, `browser` |
| `docs/` | the specification: eleven chapters read in pipeline order and the architecture chapter (`docs/README.md` is the reading guide; `11_reference.md` holds the glossary, parameters and formula index) |
| `docs/init_concepts/` | the theory documents the project started from; read-only |
| `scripts/check.py` | the quality gate (Rule 0) |
| `scripts/compose_check.py` | the Docker check: builds and starts the suite on its own volumes and ports, runs the workflow through the console (`--tests`: the gate inside the image; `--fresh`: clean room) |
| `scripts/dev.py` | host development: console, graph service and narrator on one port, routed as `nginx.conf` routes them |
| `scripts/` (others) | measurements on throwaway workspaces under `.scratch/` (never `workspace/`): `multilingual_benchmark.py`, `eval_answers.py` (live LLM through the broker), `experiment.py` (the hypothesis benchmark); shared helpers in `scratch.py`; results in `.scratch/results/` |
| `data/` | local data: `housing.csv` |
| `models/` | the embedding model of host runs, `Qwen3-Embedding-0.6B/`, provisioned and verified by `python -m insight_graph_service.core.model_store`; the stack's seed for its `sig-models` volume |
| `workspace/` | the knowledge base of host runs. Gitignored. **The user's working knowledge base — never delete or reset it without being asked** |
| `.env` | the user's settings and secrets (the provider key `GEMMA_API_KEY`, the Neo4j password). Gitignored. Never print, copy or commit its values; template `.env.sample` |

## Boundaries

- **Only this repository is in scope.** Code outside it is reference material at most: do not edit it, do not import from it, do not add it to `sys.path`. `tests/test_self_contained.py` enforces that every loaded module and `sys.path` entry lies in the repository or the Python environment, that no source holds a machine path or a relative path leading out of the repository, and that every path setting resolves inside the repository.
- Reuse upstream code by editing the vendored copy and recording the change in its `PROVENANCE.md`. Never re-copy an upstream file over the vendored one (it carries the signed-attractor repair and the numerical fixes).
- The Python environment that `.venv` may be layered on is not ours to modify; install into `.venv` only and add the dependency to the owning package's `requirements.txt` (each library and service has one; `requirements-dev.txt` collects them plus the test tools).
- Keep the bounded contexts (`docs/12_architecture.md`): the libraries import only `insight_contracts`; graph construction (`core/snapshot.py`) and retrieval read no files — the engine passes them their data; the workspace layout belongs to `core/workspace.py` (no other module builds a path below the workspace root); the core imports no web or HTTP framework; the narrator reaches the graph service only through `POST /api/search` (the `EvidencePayload`) and its health probe `GET /api/health`. `tests/test_architecture.py` lists who may import whom — extend that table only with a reason.
- The configured Neo4j database and the LLM provider account belong to the user. Settings classes read only the process environment, so tests never see `.env`; tests run with `neo4j_enabled=False` and the LLM stub.

---

## 1) Operating principles

- Small, reversible changes; root causes, not symptoms.
- Read the owning chapter before changing a module (table in §6). The docs describe the current code; if code and doc disagree, fix both in the same change.
- Statistics are the product. A change that alters what is counted as an insight, how a shift is measured, how vectors are composed or how attractors form is a **mathematical change**: write the formula and its reason into the owning chapter, test it against the planted phenomena (`insight_graph_service/core/demo.py`), and re-measure the numbers quoted in the README and in the chapters' measured-behaviour sections.
- Prefer deleting a mechanism over adding a flag. One frame for vectors, one id scheme, one place per tunable.
- Ask one focused question before a risky change; otherwise decide and state the decision.

## 2) Implementation standards

- Python 3.12 (`from __future__ import annotations`, `X | None`, dataclasses). pandas objects never leave `subgroup_miner`; everything downstream speaks `insight_contracts`.
- Typed contracts over ad-hoc dicts where the shape matters (`Insight`, `CanonicalInsight`, `EmbeddingSpec`, `LatentFrame`, `EvidencePayload`, `QAResult`). Journal and snapshot records are plain JSON by design — read them through `Insight.from_record`.
- Minimal English docstrings on public functions and non-trivial helpers; state the contract or the non-obvious intent, not the control flow. Action-oriented helper names (`build_`, `select_`, `compute_`, `resolve_`, `record_`).
- No decorative section banners, no review scaffolding, no "temporary" compatibility layers, no wrappers around a single call.
- Every tunable is a field of a package config (`MinerConfig`, `TopologyConfig`, `QueryConfig`), of `PhenomenonThresholds` or of the service's `Settings` (`insight_graph_service/core/settings.py`), settable from the environment by its upper-case name (names are unique across sections) and listed with its default in `docs/11_reference.md` §11.3; `.env.sample` documents the commonly changed ones. lac parameters keep lac's names. The narrator's settings are `NarratorSettings`; the broker's are `GEMMA_*`.
- Vocabulary: one name per concept per layer (glossary: `docs/11_reference.md` §11.1). Code says Pattern / Attractor / scope / target; the UI says insight / theme; vendored lac code says chunk / concept. Do not mix layers.
- Errors are codes, not crashes: a batch ends `FAILED` with an `error.code` from `docs/09_operations.md` §9.4; a workspace that cannot answer gives 409 with a `code`; an answer degrades to `answer_mode=fallback`. Never swallow an exception without recording it on the batch or the result.

## 3) Environment and commands

| Task | Command (Windows paths; use `.venv/bin/python` elsewhere) |
|---|---|
| gate | `.venv\Scripts\python.exe scripts\check.py` |
| quick feedback | `.venv\Scripts\python.exe scripts\check.py --quick` |
| one test | `.venv\Scripts\python.exe -m pytest tests\test_traversal.py -q` |
| the model (once) | `.venv\Scripts\python.exe -m insight_graph_service.core.model_store` (verified against the pinned manifest) |
| the stack on the host | `.venv\Scripts\python.exe -m llm_model_broker` (port 8080), then `.venv\Scripts\python.exe scripts\dev.py` → http://127.0.0.1:8765 |
| containers | `docker compose up -d --build --wait`; verification `.venv\Scripts\python.exe scripts\compose_check.py [--tests] [--fresh]` |
| demo, upload, ask, delete, reset | the HTTP API (`README.md` lists the `curl` calls); `POST /api/reset` only on request for the user's workspace |
| hypothesis benchmark | `.venv\Scripts\python.exe scripts\experiment.py --k 3` |
| multilingual benchmark | `.venv\Scripts\python.exe scripts\multilingual_benchmark.py --label <name>` |
| live answers | `.venv\Scripts\python.exe scripts\eval_answers.py` (calls the LLM behind the broker: a paid call) |

- Use `pathlib.Path` and repo-relative forward-slash paths in code and docs; the platform-specific code is the writer lock (`workspace._lock_byte`: `msvcrt` on Windows, `fcntl` elsewhere) and `fileio`'s Windows sharing retry (`fileio.retry_sharing`).
- The embedding model (Qwen3-Embedding-0.6B, ≈ 1.2 GB VRAM) loads once (~10 s, CUDA when available; the containers use the CPU). The LLM is never loaded in-process; it is only reached through `LLM_BASE_URL`. Tests use the 384-d hashing double (`tests/doubles.py`) unless marked `model`.
- `pytest.ini` pins `--basetemp=.pytest_tmp -p no:cacheprovider` (Windows temp-dir permissions); do not remove it.
- Processing a batch is sequential and checkpointed (`docs/09_operations.md`); when debugging a stuck or failed batch, read `workspace/registry/batches/<id>.json` first (`error`, `failed_stage`, `warnings`).

## 4) Testing

- Order: targeted tests for the changed area → `check.py --quick` → **`check.py`** (Rule 0).
- Behaviour changes, regressions and non-trivial edge cases get a test in the file that owns the transition (test map: `docs/10_verification.md` §10.1). Prefer extending an existing fixture over inventing a new workspace.
- Planted-phenomenon tests (`test_discovery_contract.py`, `test_e2e.py`, `test_hypothesis_apparatus`) are the statistical regression suite: a change that makes them fail is a change to the mathematics and needs the owning chapter updated and a justification, not a loosened assertion.
- Each library has a standalone test (`tests/test_*_standalone.py`) that uses it without the services (`tests/test_architecture.py` keeps its imports to the kernel); keep it passing when a library's API changes.
- Browser test: Playwright with the locally installed Chromium (`tests/test_ui_smoke.py::_chromium` looks under `%LOCALAPPDATA%\ms-playwright` for the Windows or Linux build; elsewhere the test skips); it runs the console against the `scripts/dev.py` composite. UI changes keep its selectors working or update them in the same change.
- Never add network calls to tests. The LLM endpoint is the stub server, Neo4j the fake driver.

## 5) Statistical and representational changes

| If you change… | Then also… |
|---|---|
| any formula in `subgroup_miner/vendor/eda/main_upd.py`, `subgroup_miner/discovery.py`, `subgroup_miner/selection.py` | update chapters 2/3; re-run the demo and `scripts/experiment.py --k 3`; update the numbers in the README and in chapters 2, 3, 5, 9 and 10; list the edit in `PROVENANCE.md` if it is in the vendored file |
| canonical text, components, what is embedded, block weights, the model or its revision | bump `CANONICAL_VERSION` / `REPRESENTATION_VERSION` in `attractor_topology/models.py`; update chapter 4 (and 7 if the prompt changes); existing workspaces start degraded and must be reset — say so in the summary. A new input that shapes vectors belongs in `EmbeddingSpec` (the fingerprint) |
| the embedding model | cosine thresholds belong to the embedder: recalibrate `MIN_ASSIGN_THRESHOLD` and `GROUNDING_MIN_COSINE` (`docs/05_latent_anchors.md` §5.10, `docs/07_question_answering.md` §7.1.1) and update the pinned revision and manifest in `attractor_topology/encoder.py` and `insight_graph_service/core/model_store.py` |
| attractor extraction, assignment, EMA, thresholds (`attractor_topology/ontology.py`, `vendor/lac/`) | update chapter 5; run `tests/test_ontology.py` and the E2E tests with the real model |
| traversal grammar, seed scoring, evidence limits | update chapter 7; keep `insight_graph_service/core/cypher/queries/transversal.cypher` in step |
| a settings field | the owning chapter, the parameter table in `docs/11_reference.md` §11.3, and `.env.sample` when users are expected to change it |
| error codes, batch record fields, API responses, `EvidencePayload` | chapter 9 (codes, record), chapter 8 (API), chapter 12 (the contract), `app.js` (`ERROR_HELP`, data contract) |

## 6) Documentation rules

- Docs describe the current state in the present tense. No changelog prose ("previously", "was removed"); history belongs in commits and in the summary you hand back.
- One canonical home per topic: each chapter owns the behaviour, formulas and text contracts of its stage (table below); `docs/11_reference.md` owns the glossary, identifier formats, the parameter table and the formula index; the `PROVENANCE.md` files own the vendored-code divergences. Other docs link instead of restating.
- Cite docs from code as `docs/<chapter>.md §<section>` (for example `docs/05_latent_anchors.md §5.10`). Section numbers are handles: when you add a section, append it or update every citation (`grep -rnE "docs/[01][0-9]_" --include=*.py .`).
- Every path, function, setting, test file and number named in a doc must exist in the repo and be current. Public docs name nothing outside the repository.

Code surface → doc owner:

| Surface | Owner |
|---|---|
| package map, invariants, deviations from the theory | `01_overview.md` |
| bounded contexts and their import rule, the services and their contracts, `compose.yaml`, the Dockerfiles, storage decision, observability | `12_architecture.md` |
| `subgroup_miner/` (`ingestion`, `discovery`, `vendor/eda/`) | `02_discovery.md`, `PROVENANCE.md` |
| `insight_contracts/insight.py`, `subgroup_miner/selection.py`, admission (`core/batch.py`) | `03_insights.md` |
| `attractor_topology/canonical.py`, `encoder.py`, `models.py`, `insight_contracts/text.py` | `04_representation.md` |
| `attractor_topology/ontology.py`, `vendor/lac/`, the anchor descriptions in `core/snapshot.py` | `05_latent_anchors.md`, `PROVENANCE.md` |
| `subgroup_miner/lattice.py`, `core/snapshot.py`, `core/workspace.py`, `fileio.py`, `neo4j_mirror.py`, `cypher/` | `06_graph_and_storage.md` |
| `graph_query_engine/`, `evidence_narrator_service/` | `07_question_answering.md` |
| `sig_web_console/`, `insight_graph_service/server/` | `08_interface.md` |
| `core/engine.py`, `core/settings.py`, `core/model_store.py` | `09_operations.md` (+ `11_reference.md` §11.3 for fields) |
| `core/demo.py`, `scripts/`, `tests/` | `10_verification.md` |

## 7) Data, secrets and the user's state

- `workspace/` is the user's knowledge base. `POST /api/reset` deletes it; never call it against that workspace, nor delete `workspace/`, without an explicit request. Experiments go to a scratch workspace (`scripts/scratch.py`, or `make_config(tmp_path)` in tests).
- Never print, log or paste the contents of `.env`. Do not edit it without approval.
- Uploaded files are kept under `workspace/uploads/`; treat that folder as the user's data.
- The evidence prompt sends subgroup statistics (not rows) to the configured LLM endpoint; do not widen what `Evidence.to_prompt()` includes without saying so (`docs/07_question_answering.md` §7.4).

## 8) Agent workflow

Before coding: read the owning chapter(s) and the code they name; find the smallest set of files; write down the assumptions that could break behaviour; for multi-step work, end the plan with a **Consolidation & cleanup** section.

While coding: small edits; `check.py --quick` after each meaningful step; keep contracts (`insight_contracts`, batch record, API shapes, journal formats) stable or update every consumer in the same change.

Consolidation pass (after the implementation todos, before handoff), scoped to the touched surfaces and widened to the repo when the change crosses packages:
- remove dead code, obsolete aliases, stale imports, unused settings, compatibility shims and wrappers the change made unnecessary;
- strip banners and redundant comments; confirm docstrings and action-oriented names on new helpers;
- grep for references to anything you removed or renamed (code, tests, docs, README, `.env.sample`, `app.js`);
- sweep legacy terms (glossary: `docs/11_reference.md` §11.1) and re-measure quoted numbers if the statistics changed;
- run **`scripts/check.py`** and confirm it passed.

Before handoff: Rule 0 passed in this session; docs updated per §5–6; summary states what changed, why, what was verified (with the real command output, not a paraphrase of success), and any residual risk — including whether the user's `workspace/` will start degraded after a version bump.

## 9) Output style

Concise and direct. Report failures with the output. Name skipped steps. Do not pad with restated context.
