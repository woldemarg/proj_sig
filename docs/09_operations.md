# 9. Operations — lifecycle, command line, configuration, errors

> **In one paragraph.** One `Engine` object is the stateful core of the graph service, behind its HTTP API. Every uploaded file runs through it as one *batch*: an explicit sequence of stages from validation to the committed graph, recorded in a batch record that is rewritten atomically at each stage. A batch ends `READY`, `FAILED` (with an error code and the stage it failed in) or `SKIPPED` (the same content and options are already in the workspace). A workspace the engine cannot serve does not stop the service: it starts degraded, refuses writes and questions with a code, and a reset recovers it. Every tunable lives in one settings tree built from the process environment; failures are codes, never crashes of the service.

**Code** `insight_graph_service/core/engine.py` (`Engine`), `insight_graph_service/core/batch.py`, `insight_graph_service/core/settings.py`, `insight_graph_service/core/model_store.py`, `scripts/dev.py` · **Tests** `tests/test_e2e.py`, `tests/test_persistence.py`, `tests/test_model_store.py`, `tests/test_canonical_embedding.py` (settings from `.env`), `tests/test_ui_smoke.py` · **Previous** [8. Interface](08_interface.md) · **Next** [10. Verification](10_verification.md)

---

## 9.1 The batch lifecycle

`Engine(settings, *, embedder=None, recover=True)` (`insight_graph_service/core/engine.py`) works on one workspace:
- **Upload.** `Engine.upload(name, stream, bins=, categories=)` stores an uploaded file, refused with `file_too_large` above `MAX_UPLOAD_MB`, and registers it. `Engine.submit(path, filename=, bins=, categories=)` creates the `UPLOADED` record (`None` = workspace defaults).
- **Processing.** `Engine.process(batch_id)` runs a batch; `Engine.ingest_file()` submits and processes in one call (tests and scripts). The HTTP app queues `process` on its single worker thread ([8.1](08_interface.md#81-http-api)). Read-only callers pass `recover=False` ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)).
- **Questions.** Over the committed state, `Engine.search(question)` returns the `SearchResult` and `Engine.evidence(question)` its `EvidencePayload` (the empty payload while the graph holds no insight) ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)). No language model runs in this process; the narrator verbalises the payload.
- **Other writes.** `Engine.delete_dataset(id)` and `Engine.reset()`. A reset is refused with `busy` at once while a batch or a deletion runs or an upload waits.
- **Readers.** Readers see the committed state through `Engine.committed()`: graph, vectors and literal catalog of one commit, loaded when the engine starts ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)). `graph()` and `frame()` read from it, and `prepared()` returns it with the literal catalog and every document vector filled.

**Start.** The server first checks the model files, before anything touches the workspace (exit 3 when one is missing; [9.3](#93-configuration)). A writer engine then takes the workspace's writer lock — another writer process gets `WorkspaceBusy`, and the server exits with code 2 — and:
1. it recovers interrupted writes ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery));
2. it compares the stored canonical and representation versions with the code's (`Workspace.check_versions`, no model needed);
3. it loads the committed state; a snapshot of another layout version is rebuilt from the journals and saved.

A workspace the engine cannot serve does not stop the start: the engine starts degraded and refuses writes and questions with `workspace_degraded` until `Engine.reset()` (`POST /api/reset`) ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).

After the start, the app queues `Engine.startup_sync()` on its worker thread. It brings the Neo4j mirror up to the committed snapshot, holding the write lock so it never interleaves with a deletion's or a reset's own sync. It never syncs from a degraded or empty workspace: that would wipe a mirror without an operator action.

```text
UPLOADED → VALIDATING → PROFILING → DISCOVERING → VALIDATING_INSIGHTS → EMBEDDING
         → UPDATING_ONTOLOGY → BUILDING_GRAPH → PERSISTING → READY            terminal: READY | FAILED | SKIPPED
```

| Stage | Work | Chapter |
|---|---|---|
| VALIDATING | load and validate the file, bands and categories, dataset id; a READY batch with the same id → `SKIPPED` (`duplicate_of`) | [2.1](02_discovery.md#21-ingestion) |
| PROFILING | EDA steps 1–3 (an empty search space fails here) | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) |
| DISCOVERING | pass 1, deduplication, pass 2 (the bootstrap) | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps), [2.3](02_discovery.md#23-deduplication-before-validation) |
| VALIDATING_INSIGHTS | `build_insights` + `select_insights` (validity rules R1–R3, weight) + `batch.admit_insights` (admission: R4 weight floor, R7 batch budget); rejections persisted | [3](03_insights.md) |
| EMBEDDING | canonicalise and encode; embed the canonical documents for the text baseline; representation check | [4](04_representation.md) |
| UPDATING_ONTOLOGY | **transaction** (checkpoint + `pending.json`), batch sequence, `LatentOntology.ingest` | [5](05_latent_anchors.md) |
| BUILDING_GRAPH | pattern records (+ canonical form, embedding metadata), covers | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| PERSISTING | duplicate guard, journal append, state save, representation record, batch sequence commit, snapshot, committed graph and frame loaded | [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery) |
| READY | record saved (the commit point), marker and checkpoint removed; then readers switch to the new graph and frame and the optional Neo4j mirror is synced (a failure there is a warning) | [6.6](06_graph_and_storage.md#66-neo4j-mirror) |

One writer process works on a workspace at a time — it holds the workspace's writer lock ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)) — and within it batches run one at a time (a re-entrant lock; the HTTP app uses a single worker thread). Any exception before the READY save rolls back (`Workspace.transaction`, [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)) and marks the batch `FAILED` with `error {code, message[, trace]}` and `failed_stage`. If the rollback itself fails, the record also carries the warning `rollback failed`, and later batches fail with `rollback_pending` until a writer restart has finished it. Nothing after the READY save can turn the batch FAILED: the committed graph and frame were loaded before it, and switching them in is an assignment. A failure to save the record after the Neo4j sync propagates out of `process()`; the batch stays READY on disk.

**Batch record** (`registry/batches/<batch_id>.json`): `{batch_id, batch_seq, dataset_id, filename, source_path, bins, categories, status, stage_times {STAGE: iso}, created_at, updated_at, profile {rows, columns, numerics, categoricals, dropped_columns, selected_dimensions, search_space_size, global_medians, global_mads, dimension_cardinality, dimension_entropy, derived_columns, bins, categorical_overrides}, metrics {…}, warnings [], error, failed_stage, duplicate_of, neo4j {status, …}}`. The checkpoint of a committing batch lives in `pending.json`, not in the record.

## 9.2 Command line

SIG has no command-line program of its own. A command line starts the processes, and everything else goes through the HTTP API.

| Command | Does |
|---|---|
| `python scripts/dev.py` | the console, the graph service and the narrator on one port (`WEB_HOST:WEB_PORT`, http://127.0.0.1:8765), routed as the console's nginx routes them; reads `.env` |
| `python -m insight_graph_service.server` | the graph service alone on `WEB_HOST:WEB_PORT` (its API only); reads `.env` |
| `python -m evidence_narrator_service` | the narrator alone on `NARRATOR_HOST:NARRATOR_PORT` (127.0.0.1:8766), reaching the graph service at `INSIGHT_GRAPH_URL`; reads only the process environment |
| `python -m llm_model_broker` | the broker on `BROKER_HOST:BROKER_PORT` (127.0.0.1:8080), reading `GEMMA_*` from `.env` ([12.4](12_architecture.md#124-services-and-contracts)) |
| `python -m insight_graph_service.core.model_store [MODEL_DIR] [--seed DIR]` | provisions the embedding model and verifies it file by file; without a folder, the service's `MODEL_DIR` (environment, then `.env`); exit 1 unless the result verifies |
| `docker compose up -d --build --wait` | the whole suite in dependency order ([12.5](12_architecture.md#125-containers-and-start-order)) |
| `python scripts/experiment.py`, `scripts/multilingual_benchmark.py`, `scripts/eval_answers.py` | measurements on scratch workspaces ([10.4](10_verification.md#104-measurement-scripts)) |
| `python scripts/check.py`, `scripts/compose_check.py` | the quality gate and the container check ([10.5](10_verification.md#105-quality-gate-and-container-check)) |

Headless operations go through the console's port; `README.md` lists them as `curl` calls. The graph service's own port serves the same calls except `/api/chat/`.

| Operation | Request | Answer |
|---|---|---|
| ingest the demo | `POST /api/demo` | 202 + the `UPLOADED` record |
| ingest a file | `POST /api/upload`, multipart `file`, optional `bins` (`median_income:4,housing_median_age:4`) and `categories` (`Store,Holiday_Flag`) | 202 + the `UPLOADED` record |
| status | `GET /api/batches`, `GET /api/batches/<id>`; `GET /api/health` (graph statistics, `problem`) | batch records; health |
| ask | `POST /api/chat/query` `{question, use_llm}` | the answer (`QAResult`, [7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| evidence only | `POST /api/search` `{question}` | the `EvidencePayload` |
| delete a dataset | `DELETE /api/datasets/<dataset id>` (or the batch id of an upload that failed before it had one) | what was removed |
| reset | `POST /api/reset` | the workspace deleted, the mirror cleared |

A batch is queued and runs on the service's worker thread; poll `GET /api/batches/<id>` until it is READY, FAILED or SKIPPED. The Neo4j mirror follows by itself: after every commit, deletion and reset, and at start ([6.6](06_graph_and_storage.md#66-neo4j-mirror)). The sphere is served as data (`GET /api/sphere`, [8.4](08_interface.md#84-the-latent-sphere)).

**Writers.** The running service (or `scripts/dev.py`) holds the workspace's writer lock for its whole life. Another writer process on the same workspace — a second server, or a script's `Engine` with `recover=True` — is refused with `WorkspaceBusy`; while the service runs, upload through it. The measurement scripts build their own scratch workspaces under `.scratch/`. Use a separate `WORKSPACE_DIR` for experiments — the default `workspace/` is the knowledge base people work with.

## 9.3 Configuration

`insight_graph_service/core/settings.py::Settings` (a frozen dataclass) is the root of every tunable of the graph service:
- **Its own fields:** paths, upload limit, admission, Neo4j, web.
- **One section per package**, which owns it: `miner: MinerConfig` (`subgroup_miner/config.py`), `topology: TopologyConfig` (`attractor_topology/config.py`), `query: QueryConfig` (`graph_query_engine/config.py`).
- **Shared thresholds.** The three sections share one `PhenomenonThresholds` (`insight_contracts/insight.py`: `MIN_EMM_SCORE`, `WEIGHT_EMM_REF`, `MIN_COMPONENT_Z`), the values mining, representation and retrieval must agree on; `Settings` refuses sections whose thresholds differ.

`load_settings(**overrides)` builds the tree in three layers: the defaults, then the process environment (`NAME` = the field name in upper case), then the overrides.
- **One name, one variable.** An override is routed by field name to the section that has the field. Names are unique across all sections, checked at import, so each one is a single environment variable; an unknown override is a `TypeError`.
- **Coercion.** Values are coerced to the field's type: booleans from `1/true/yes/on`, tuples from comma-separated numbers, relative paths resolved against the repository root. An empty value clears a text field and leaves any other field at its default.

**Where values come from.** Values come only from the process environment; no settings class reads a file.
- **Host runs.** The host entry points call `load_env_file()` first: `python -m insight_graph_service.server`, `scripts/dev.py`, `python -m insight_graph_service.core.model_store` without a folder, and the measurement scripts through `scripts/scratch.py`. It copies the repository's `.env` into the environment, where variables already set win, and skips `GEMMA_*` lines: the provider settings and key belong to the broker.
- **Containers** get their environment from compose (per service, [12.5](12_architecture.md#125-containers-and-start-order)).
- **Tests** never load the repository's `.env`, so no developer credential reaches a test.

The narrator's settings are `NarratorSettings` (`evidence_narrator_service/settings.py`, `from_env`, the same naming rule; [7.8](07_question_answering.md#78-configuration-failure-modes-and-limitations)); the broker's are its `GEMMA_*` variables ([12.4](12_architecture.md#124-services-and-contracts)). `.env.sample` documents the commonly changed fields; [11.3](11_reference.md#113-parameters) lists them all with defaults.

**Deployment.**

| Part | Requirement |
|---|---|
| Python | 3.12+ (the EDA engine uses PEP 701 f-strings); `pip install -r requirements-dev.txt` for host development (every service's runtime plus pytest, ruff and playwright; torch first for CUDA); each library and service has its own `requirements.txt`, and the images install with `constraints.txt` |
| embedding model | `python -m insight_graph_service.core.model_store` once: Qwen3-Embedding-0.6B at the pinned revision into `models/` (1.19 GB), checked file by file against the manifest (sizes and SHA-256). A verified copy is kept, a verified seed (`--seed`) is copied, otherwise the revision is downloaded; it is loaded offline afterwards ([4.4](04_representation.md#44-the-embedding-model)). In the stack, `model-init` fills the `sig-models` volume the same way. Before torch loads, the service checks presence and sizes against the same manifest and exits with code 3, naming this command, when a file is missing or has the wrong size |
| GPU | optional; the embedder takes ≈ 1.2 GB resident, ≈ 1.7 GB peak; set `EMBEDDING_DEVICE` (the containers use `cpu`) |
| LLM | the broker (`python -m llm_model_broker`, or the `llm-model-broker` container; [12.4](12_architecture.md#124-services-and-contracts)) or any other OpenAI-compatible endpoint behind the narrator's `LLM_BASE_URL`; without one, answers are evidence-only ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| Neo4j | optional mirror on the host (`NEO4J_ENABLED=true`, a running DBMS); the stack runs its own container ([6.6](06_graph_and_storage.md#66-neo4j-mirror)) |
| Docker | `docker compose up -d --build --wait`: `model-init`, `neo4j`, `llm-model-broker`, `insight-graph`, `evidence-narrator`, `sig-web-console` in dependency order ([12.5](12_architecture.md#125-containers-and-start-order)); `scripts/compose_check.py` verifies them end to end ([10.5](10_verification.md#105-quality-gate-and-container-check)) |

## 9.4 Error codes

| Code | Meaning | Stage |
|---|---|---|
| `unsupported_file`, `unreadable_file`, `invalid_options` | missing file or unsupported type, parse error, malformed or unknown column options | VALIDATING |
| `invalid_schema` | too few rows or columns; no categorical dimension after profiling | VALIDATING / PROFILING |
| `no_numeric_targets` | no numeric column | VALIDATING / PROFILING |
| `no_candidates` | empty search space (PROFILING) or nothing passed the pass-1 size screen (DISCOVERING) | PROFILING / DISCOVERING |
| `no_viable_insights` | selection and admission kept nothing | VALIDATING_INSIGHTS |
| `embedding_failure`, `representation_mismatch` | encoder error; fingerprint differs from the workspace's | EMBEDDING |
| `ontology_failure` | an ontology invariant was violated | UPDATING_ONTOLOGY |
| `duplicate_patterns` | a pattern id is already journaled (the append-only journal would be corrupted) | PERSISTING |
| `rollback_pending` | an earlier rollback has not finished; no transaction starts until a writer restart finishes it (`failed_stage` is the last stage entered) | the commit |
| `internal_error` | anything else (the trace is kept) | any |
| `interrupted` | crash recovery found the batch unfinished; the record keeps its last stage in `stage_times` but gets no `failed_stage` | any |
| `file_too_large` (not a batch code) | an upload above `MAX_UPLOAD_MB`; the file is not kept (HTTP 413) | upload |
| `WorkspaceBusy` (not a batch code) | another writer process holds the workspace; the server exits with code 2 | engine start |
| `workspace_degraded` (not a batch code) | the engine serves a workspace it cannot use ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)); writes and questions are refused (HTTP 409) until a reset | any call |
| `representation_mismatch` on a question (not a batch code) | the encoder's fingerprint differs from the workspace's; `POST /api/search` answers 409 | question |
| `busy` (not a batch code) | a reset while a batch or a deletion runs or an upload waits; a deletion while one of the dataset's batches runs (HTTP 409) | — |
| `unknown_batch` (not a batch code) | `Engine.process` was given an id with no record; over HTTP, `GET /api/batches/{id}` answers a plain 404 `{detail}` for an unknown id | — |
| `unknown_dataset` (not a batch code) | `Engine.delete_dataset` was given an id no batch carries (HTTP 404) | — |
| warning `graph persistence (Neo4j) failed` | the mirror failed; the batch stays READY | READY |
| QA `answer_mode = fallback` / `empty` | LLM unavailable or switched off / empty graph | question |

Over HTTP every refusal is `{detail, code}` with the status of [12.4](12_architecture.md#124-services-and-contracts). The console maps the batch codes a person can act on to a plain-language title, cause and tip (`ERROR_HELP` in `app.js`); any other code shows a generic title and the raw message.

## 9.5 Batch metrics

`record["metrics"]`: `input_rows, columns, numeric_targets, dimensions, search_space, pass1_subgroups, candidate_patterns (distinct cohorts), validated_candidates, validated_insights, pruned {reason: n}, pruned_total, avg_insight_support, avg_insight_weight, embedding_count, embedding_dim, attractors_total, attractors_new, orphan_rate, activation_count, soft_merged, centroid_drift, adaptive_thresh, max_concept_density_pct, density_threshold, damped_attractors, max_centroid_step, clamped_attractors, avg_attractor_degree, graph_edges, edge_counts, graph_patterns, ontology_warnings, timings {validate_s, discover_s, select_s, embed_s, ontology_s, graph_s}, processing_duration_s` (`batch.batch_metrics`). The full lac metrics per batch are in `state/ontology_metrics.csv` ([5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics)); query-side metrics are in each answer's `metrics` and in the narrator's query log (logger `evidence_narrator_service.queries`, [7.5](07_question_answering.md#75-the-language-model-and-citation-check)).

> **Running example.** The demo batch: 5,000 rows, 11 columns, 4 metrics, 3 dimensions, 88 conjunctions, 84 pass-1 subgroups and distinct cohorts, 50 validated, 28 insights, `pruned {weak_effect: 22}`, 4 anchors, 281 graph edges, READY in 13.1 s — of which 10.9 s embedding (the one-off model load, the three blocks and the documents), 1.1 s discovery and 0.9 s ontology. `housing.csv` with two bands (20,640 rows) is READY in about 17 s from a cold start.

## 9.6 Guarantees

* Terminal states are final: nothing after the READY save can turn a batch FAILED.
* `READY` implies that journals, state and snapshot are mutually consistent.
* Idempotent: identical content and options are processed once.
* A workspace the engine cannot serve never stops the service: it starts degraded, says why, and `POST /api/reset` recovers it.
* Every failure inside a batch is a batch state and every failure inside a question an answer mode or an HTTP error; the service keeps running. Engine-level refusals (`WorkspaceBusy`, `workspace_degraded`, `busy`, `file_too_large`) are refusals of the call, not batch states.
