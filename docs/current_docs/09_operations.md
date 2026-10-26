# 9. Operations — lifecycle, command line, configuration, errors

> **In one paragraph.** One `Engine` object is the stateful core of the graph service, behind its HTTP API. Every uploaded file runs through it as one *batch*: an explicit sequence of stages from validation to the committed graph, recorded in a batch record that is rewritten atomically at each stage. A batch ends `READY`, `FAILED` (with an error code and the stage it failed in) or `SKIPPED` (the same content and options are already in the workspace). A workspace the engine cannot serve does not stop the service: it starts degraded, refuses writes and questions with a code, and a reset recovers it. Every tunable lives in one settings tree built from the process environment; failures are codes, never crashes of the service.

**Code** `insight_graph_service/core/engine.py` (`Engine`), `insight_graph_service/core/batch.py`, `insight_graph_service/core/settings.py`, `insight_graph_service/core/model_store.py`, `scripts/dev.py` · **Tests** `tests/test_e2e.py`, `tests/test_persistence.py`, `tests/test_model_store.py`, `tests/test_canonical_embedding.py` (settings from `.env`), `tests/test_ui_smoke.py` · **Previous** [8. Interface](08_interface.md) · **Next** [10. Verification](10_verification.md)

---

## 9.1 The batch lifecycle

`Engine(settings, *, embedder=None, recover=True)` (`insight_graph_service/core/engine.py`) works on one workspace:
- **Upload.** `Engine.upload(name, stream, bins=, categories=, geo=, table=)` stores an uploaded file, refused with `file_too_large` above `MAX_UPLOAD_MB`, and registers one batch per level ([2.9](02_discovery.md#29-spatial-datasets-the-geo-option)), returning their records. `Engine.submit(path, filename=, bins=, categories=)` creates the `UPLOADED` record (`None` = workspace defaults).
- **Levels.** The levels of one upload run in the order `resolve_levels` lists them: the finest geo level, the coarser ones, the table level last. The table level mines the most rows and takes longest, so the map comes first and the places of the base insights arrive with the table's commit, which the open console applies as any other ([8.7](08_interface.md#87-the-map)). Each level is a batch of its own: a failed level, such as a coarse resolution with too few cells (`invalid_schema`) or nothing to mine (`no_viable_insights`), does not stop the next; the table level is placed by the geo levels that are READY, and a failed table level leaves the geo levels as they are (`test_a_failed_coarse_level_leaves_the_table_placed_by_the_others`, `test_a_failed_table_level_leaves_the_geo_levels_as_they_are`). The console's rail says on the levels' group how many are ready, how many found nothing to mine and how many failed ([8.2](08_interface.md#82-the-single-page-ui)).
- **Processing.** `Engine.process(batch_id)` runs a batch; `Engine.ingest_file()` submits and processes in one call (tests and scripts). The HTTP app queues `process` on its single worker thread ([8.1](08_interface.md#81-http-api)). Read-only callers pass `recover=False` ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)).
- **Questions.** Over the committed state, `Engine.search(question)` returns the `SearchResult` and `Engine.evidence(question)` its `EvidencePayload` (the empty payload while the graph holds no insight) ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)). No language model runs in this process; the narrator verbalises the payload.
- **Other writes.** `Engine.delete_dataset(id)`, `Engine.delete_upload(upload_id)` (every level of one upload) and `Engine.reset()`. Each of them, and a `PATCH /api/datasets`, is refused with `busy` at once while a batch or another write runs (`Engine._writing`), never queued behind it: the web request would wait for the whole batch. A reset is also refused while an upload waits.
- **Readers.** Readers see the committed state through `Engine.committed()`: graph, vectors and literal catalog of one commit, loaded when the engine starts ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)). `graph()` and `frame()` read from it; it holds the literal catalog from the moment it is committed.

**Start.** The server first checks the model files, before anything touches the workspace (exit 3 when one is missing; [9.3](#93-configuration)). A writer engine then takes the workspace's writer lock — another writer process gets `WorkspaceBusy`, and the server exits with code 2 — and:
1. it recovers interrupted writes ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery));
2. it compares the stored canonical and representation versions with the code's (`Workspace.check_versions`, no model needed);
3. it loads the committed state; a snapshot of another layout version is rebuilt from the journals and saved.

A workspace the engine cannot serve does not stop the start: the engine starts degraded and refuses writes and questions with `workspace_degraded` until `Engine.reset()` (`POST /api/reset`) ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).

After the start, the app queues `Engine.startup_sync()` on its worker thread. It brings the Neo4j mirror up to the committed snapshot, holding the write lock so it never interleaves with a deletion's or a reset's own sync. It never syncs from a degraded or empty workspace: that would wipe a mirror without an operator action.

The worker also builds a commit's first reads before a console asks, at the start and after an upload's last level, a `PATCH /api/datasets` and a deletion. It builds the graph view first (`GET /api/graph`, no filter), then the map's reads for each READY geo dataset (`Engine.warm_map`, `geo_views.warm`):
- **Which datasets:** newest first, and at most as many as the cell-package cache holds (8).
- **What it reads:** the cell package, which goes to its per-file cache; the overlay, the projected insights and the whole map's window, whose projection and per-insight facts go to the commit's memo.
- **When it stops:** while a batch waits, whose commit would throw them away. It builds nothing on a degraded workspace.
- **With the Neo4j mirror on,** it starts once the mirror's sync has finished, because the sync runs in the commit's job. A console that reads in between builds what it reads itself, as before.

Measured without the mirror, on the 300,000-point synthetic upload (`@auto` and `table`): the first Map open one poll after READY (the cells, `/projected` and `/overlay` together, then a window) takes 0.03–0.09 s over four runs, against 2.0 s when every read was built on its request. A graph read that lands while the worker builds the graph view waits for that one build (up to 0.43 s, against 0.27 s for a cold build), and is a memo hit otherwise.

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
| EMBEDDING | canonicalise and encode the three blocks; representation check | [4](04_representation.md) |
| UPDATING_ONTOLOGY | **transaction** (checkpoint + `pending.json`), batch sequence, `LatentOntology.ingest` | [5](05_latent_anchors.md) |
| BUILDING_GRAPH | pattern records (+ canonical form, embedding metadata), covers | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| PERSISTING | duplicate guard, journal append, state save, representation record, batch sequence commit, snapshot, committed graph and frame loaded | [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery) |
| READY | record saved (the commit point), marker and checkpoint removed; then readers switch to the new graph and frame and the optional Neo4j mirror is synced (a failure there is a warning). A poll (`GET /api/batches`, `Engine.polled`) reads the batch PERSISTING until readers switch, so READY there means the new graph | [6.6](06_graph_and_storage.md#66-neo4j-mirror) |

One writer process works on a workspace at a time — it holds the workspace's writer lock ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)) — and within it batches run one at a time (a re-entrant lock; the HTTP app uses a single worker thread). Any exception before the READY save rolls back (`Workspace.transaction`, [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)) and marks the batch `FAILED` with `error {code, message[, trace]}` and `failed_stage`. If the rollback itself fails, the record also carries the warning `rollback failed`, and later batches fail with `rollback_pending` until a writer restart has finished it. Nothing after the READY save can turn the batch FAILED: the committed graph and frame were loaded before it, and switching them in is an assignment. A failure to save the record after the Neo4j sync propagates out of `process()`; the batch stays READY on disk.

**Batch record** (`registry/batches/<batch_id>.json`): `{batch_id, batch_seq, dataset_id, filename, source_path, bins, categories, geo?, upload_id?, level?, strict?, locator? (the table level of a geo upload: its point columns left out), source?, reliability?, indicators?, user_glossary?, metadata_doc? (the declared glossary and the parsed metadata document, [2.1](02_discovery.md#21-ingestion)), supersedes? (the dataset or upload this version replaces), window? and temporal? (a time window's bounds and `{reference_scale, observations}`: `observations` null until the window's selection writes what it saw of each known rule, [2.10](02_discovery.md#210-time-windows-and-trajectories)), metadata_digest? (of what the graph reads of the dataset, set at READY, at every `PATCH` and when a time window fails, [6.7](06_graph_and_storage.md#67-configuration-and-guarantees)), superseded? (what its commit removed: `{datasets, batches, patterns_removed, anchors_removed}`, [6.8](06_graph_and_storage.md#68-deleting-a-dataset)), status, stage_times {STAGE: iso}, created_at, updated_at, profile {rows, columns, numerics, categoricals, dropped_columns, selected_dimensions, search_space_size, global_medians, global_mads, dimension_cardinality, dimension_entropy, derived_columns, bins, categorical_overrides, points? (a table level's left-out point columns, what places it: [6.1](06_graph_and_storage.md#61-the-structural-plane)), geo? (a geo level's, [2.9](02_discovery.md#29-spatial-datasets-the-geo-option); its `levels`, the `@auto` selection, is there from the upload on, whatever the batch's end)}, metrics {…}, warnings [], error, failed_stage, duplicate_of, neo4j {status, sync (`full` or `diff`, [6.6](06_graph_and_storage.md#66-neo4j-mirror)), written (rows), deleted (a diff's), …}}`. The checkpoint of a committing batch lives in `pending.json`, not in the record.

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
| ingest the map demo | `POST /api/demo/geo` | 202 + the `UPLOADED` record (the shipped housing table as H3 cells) |
| ingest a file | `POST /api/upload`, multipart `file`, optional `bins` (`median_income:4,housing_median_age:4`) and `categories` (`Store,Holiday_Flag`) | 202 + the `UPLOADED` record, with `levels`: every level's batch id |
| status | `GET /api/batches[?upload_id=]`, `GET /api/batches/<id>`; `GET /api/health` (graph statistics, `problem`) | batch records; health |
| ask | `POST /api/chat/query` `{question, use_llm}`, a map zone optional (`cells`, `geometry`, `resolution`) | the answer (`QAResult`, [7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| evidence only | `POST /api/search` `{question}`, the same zone fields | the `EvidencePayload` |
| delete a dataset | `DELETE /api/datasets/<dataset id>` (or the batch id of an upload that failed before it had one); every level of one upload: `DELETE /api/uploads/<upload id>` | what was removed |
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
| GPU | optional: without one the embedder runs on the CPU. In Docker the default image has the CPU torch build (`sig/insight-graph:cpu`); `compose.gpu.yaml` builds the CUDA 13.0 one (`sig/insight-graph:gpu`) and attaches an NVIDIA GPU (driver 580 or newer; Docker Desktop on WSL 2, or the NVIDIA Container Toolkit on Linux). Docker Desktop on a Mac attaches no GPU: there the graph service runs on the host to use MPS (`python scripts/dev.py` with the broker, README *Mac: GPU*); the console's health pill shows the device `cpu` in amber. The embedder takes ≈ 1.2 GB resident, ≈ 1.7 GB peak; on the host, set `EMBEDDING_DEVICE` |
| LLM | the broker (`python -m llm_model_broker`, or the `llm-model-broker` container; [12.4](12_architecture.md#124-services-and-contracts)) or any other OpenAI-compatible endpoint behind the narrator's `LLM_BASE_URL`; without one, answers are evidence-only ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| Neo4j | optional mirror on the host (`NEO4J_ENABLED=true`, a running DBMS); the stack runs its own container ([6.6](06_graph_and_storage.md#66-neo4j-mirror)) |
| Docker | `docker compose up -d --build --wait`: `model-init`, `neo4j`, `llm-model-broker`, `insight-graph`, `evidence-narrator`, `sig-web-console` in dependency order ([12.5](12_architecture.md#125-containers-and-start-order)); `scripts/compose_check.py` verifies them end to end ([10.5](10_verification.md#105-quality-gate-and-container-check)) |

## 9.4 Error codes

| Code | Meaning | Stage |
|---|---|---|
| `unsupported_file`, `unreadable_file`, `invalid_options` | missing file or unsupported type, parse error, malformed or unknown column options (a bad `source`, `reliability`, `indicators`, `glossary` or metadata document is refused before the file is stored or a batch exists: HTTP 400); `invalid_options` also refuses a question's map zone that cannot be read (`cells`, `geometry`; HTTP 400, no batch) and a `supersedes` that names nothing, the same file, or (by `PATCH`) the dataset itself or its own upload ([6.8](06_graph_and_storage.md#68-deleting-a-dataset)) | VALIDATING |
| `invalid_schema` | too few rows or columns; no categorical dimension after profiling | VALIDATING / PROFILING |
| `no_numeric_targets` | no numeric column | VALIDATING / PROFILING |
| `no_candidates` | empty search space (PROFILING) or nothing passed the pass-1 size screen (DISCOVERING), and no known rule to report (a time window's known rules are reported instead, [2.10](02_discovery.md#210-time-windows-and-trajectories)) | PROFILING / DISCOVERING |
| `no_viable_insights` | selection and admission kept nothing; a time window ends so with its observations recorded, and its series' trajectories read it as evaluated ([2.10](02_discovery.md#210-time-windows-and-trajectories)) | VALIDATING_INSIGHTS |
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
| `busy` (not a batch code) | a deletion, a `PATCH` or a reset while a batch or another write runs; a reset while an upload waits; a deletion, or a `PATCH supersedes`, while one of the target's batches is still to run (HTTP 409) | — |
| `unknown_batch` (not a batch code) | `Engine.process` was given an id with no record; over HTTP, `GET /api/batches/{id}` answers a plain 404 `{detail}` for an unknown id | — |
| `metadata_mismatch` (not a batch code) | a `PATCH /api/datasets/{id}` on a dataset whose batch record no longer matches its `metadata_digest` (HTTP 409): signing it again would pass the edit off as genuine; delete the dataset and upload it again ([6.7](06_graph_and_storage.md#67-configuration-and-guarantees)) | — |
| `unknown_dataset` (not a batch code) | `Engine.delete_dataset` was given an id no batch carries (HTTP 404) | — |
| `unknown_pattern` (not a batch code) | `Engine.verify_pattern` was given an id no journal record carries (HTTP 404) | — |
| `unknown_upload` (not a batch code) | `Engine.delete_upload` was given an upload id no batch carries (HTTP 404) | — |
| warning `graph persistence (Neo4j) failed` | the mirror failed; the batch stays READY | READY |
| QA `answer_mode = fallback` / `empty` | LLM unavailable or switched off / empty graph | question |

Over HTTP every refusal is `{detail, code}` with the status of [12.4](12_architecture.md#124-services-and-contracts). The console maps the batch codes a person can act on to a plain-language title, cause and tip (`ERROR_HELP` in `app.js`); any other code shows a generic title and the raw message. `ERROR_HELP` also holds the console's own `chat_timeout`, the text of a question without an answer in time ([8.2](08_interface.md#82-the-single-page-ui)).

## 9.5 Batch metrics

`record["metrics"]`: `input_rows, columns, numeric_targets, dimensions, search_space, pass1_subgroups, candidate_patterns (distinct cohorts), validated_candidates, validated_insights, pruned {reason: n}, pruned_total, avg_insight_support, avg_insight_weight, embedding_count, embedding_dim, attractors_total, attractors_new, orphan_rate, activation_count, soft_merged, centroid_drift, adaptive_thresh, max_concept_density_pct, density_threshold, damped_attractors, max_centroid_step, clamped_attractors, avg_attractor_degree, graph_edges, edge_counts, graph_patterns, ontology_warnings, timings {validate_s, discover_s, select_s, embed_s, ontology_s, graph_s, persist_s, neo4j_s} (one per stage block: `graph_s` the records and covers, `persist_s` the journal append, the map's files, a replaced version's removal and the snapshot), processing_duration_s` (`batch.batch_metrics`). The full lac metrics per batch are in `state/ontology_metrics.csv` ([5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics)); query-side metrics are in each answer's `metrics` and in the narrator's query log (logger `evidence_narrator_service.queries`, [7.5](07_question_answering.md#75-the-language-model-and-citation-check)).

`neo4j_s` is the Neo4j sync after the commit point, written with the record's last save (with `neo4j`), so it is absent while the record says READY and the mirror is still being written; `processing_duration_s` ends at the commit point and does not include it.

> **Running example.** The demo batch: 5,000 rows, 11 columns, 4 metrics, 3 dimensions, 88 conjunctions, 84 pass-1 subgroups and distinct cohorts, 50 validated, 28 insights, `pruned {weak_effect: 22}`, 4 anchors, 281 graph edges, READY in 13.8 s on a host with an NVIDIA GPU — of which 11.3 s embedding (the one-off model load and the three blocks), 1.4 s discovery and 0.8 s ontology; on the same host's CPU 23.4 s, of which 19.9 s embedding (35.3 s and 32.5 s while every canonical document was embedded too, [10.4](10_verification.md#104-measurement-scripts)). `housing.csv` with two bands (20,640 rows) is READY in about 17 s from a cold start.

## 9.6 Guarantees

* Terminal states are final: nothing after the READY save can turn a batch FAILED.
* `READY` implies that journals, state and snapshot are mutually consistent.
* Idempotent: identical content and options are processed once.
* A workspace the engine cannot serve never stops the service: it starts degraded, says why, and `POST /api/reset` recovers it.
* Every failure inside a batch is a batch state and every failure inside a question an answer mode or an HTTP error; the service keeps running. Engine-level refusals (`WorkspaceBusy`, `workspace_degraded`, `busy`, `file_too_large`) are refusals of the call, not batch states.
