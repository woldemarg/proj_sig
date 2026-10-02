# 9. Operations — lifecycle, command line, configuration, errors

> **In one paragraph.** One `Engine` object is the stateful service behind both the CLI and the web app. Every uploaded file runs through it as one *batch*: an explicit sequence of stages from validation to the committed graph, recorded in a batch record that is rewritten atomically at each stage. A batch ends `READY`, `FAILED` (with an error code and the stage it failed in) or `SKIPPED` (the same content and options are already in the workspace). Every tunable lives in one configuration object that reads the environment; failures are codes, never crashes of the service.

**Code** `ltir/pipeline.py` (`Engine`), `ltir/cli.py`, `ltir/config.py` · **Tests** `tests/test_e2e.py`, `tests/test_persistence.py`, `tests/test_ui_smoke.py` · **Previous** [8. Interface](08_interface.md) · **Next** [10. Verification](10_verification.md)

---

## 9.1 The batch lifecycle

`Engine(config, *, embedder=None, llm=None, recover=True)`; `Engine.submit(path, filename, bins=None, categories=None)` creates the `UPLOADED` record (`None` = workspace defaults), `Engine.process(batch_id)` runs it, `Engine.ingest_file()` does both. Read-only callers pass `recover=False` ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)).

```text
UPLOADED → VALIDATING → PROFILING → DISCOVERING → VALIDATING_INSIGHTS → EMBEDDING
         → UPDATING_ONTOLOGY → BUILDING_GRAPH → PERSISTING → READY            terminal: READY | FAILED | SKIPPED
```

| Stage | Work | Chapter |
|---|---|---|
| VALIDATING | load and validate the file, bands and categories, dataset id; a READY batch with the same id → `SKIPPED` (`duplicate_of`); copy the source | [2.1](02_discovery.md#21-ingestion) |
| PROFILING | EDA steps 1–3 (an empty search space fails here) | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) |
| DISCOVERING | pass 1, deduplication, pass 2 (the bootstrap) | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps), [2.3](02_discovery.md#23-deduplication-before-validation) |
| VALIDATING_INSIGHTS | `build_insights` + `select_insights`; rejections persisted | [3](03_insights.md) |
| EMBEDDING | canonicalise and encode; embed the canonical documents for the text baseline; representation check | [4](04_representation.md) |
| UPDATING_ONTOLOGY | **checkpoint**, batch sequence, `LatentOntology.ingest` | [5](05_latent_anchors.md) |
| BUILDING_GRAPH | pattern records (+ canonical form, embedding metadata), covers | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| PERSISTING | duplicate guard, journal append, state save, representation record, batch sequence commit, snapshot | [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery) |
| READY | record saved, checkpoint discarded, committed caches swapped in; then the optional Neo4j publish and the sphere export, queued on a background thread (a failure there is a warning) | [6.6](06_graph_and_storage.md#66-neo4j-mirror), [8.4](08_interface.md#84-the-latent-sphere) |

One writer process works on a workspace at a time — it holds the workspace's writer lock ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)) — and within it batches run one at a time (a re-entrant lock; the web app uses a single worker thread). Any exception before the READY save rolls back, discards the checkpoint and marks the batch `FAILED` with `error {code, message[, trace]}` and `failed_stage`. An exception after the READY save but before the caches are swapped (discarding the checkpoint, rebuilding the caches) overwrites READY with `FAILED`; it rolls back only while the checkpoint still exists, so a failure of the cache swap leaves the batch's rows committed under a FAILED record — a narrow window, kept visible rather than hidden. A failure to save the record after the Neo4j publish propagates out of `process()`; the batch stays READY on disk. The sphere export runs after `process()` has returned, so the returned record may not yet have its `sphere` key; its failures, including the record save, never reach the caller.

**Batch record** (`registry/batches/<batch_id>.json`): `{batch_id, batch_seq, dataset_id, filename, source_path, bins, categories, status, stage_times {STAGE: iso}, created_at, updated_at, profile {rows, columns, numerics, categoricals, dropped_columns, selected_dimensions, search_space_size, global_medians, global_mads, dimension_cardinality, dimension_entropy, derived_columns, bins, categorical_overrides}, metrics {…}, warnings [], error, failed_stage, duplicate_of, neo4j {status, …}, sphere}`; a temporary `checkpoint` key exists while a batch is committing (the batch list endpoint strips it).

## 9.2 Command line

`python -m ltir <command>`:

| Command | Does | Writes |
|---|---|---|
| `demo` | writes the synthetic dataset and ingests it (workspace-default options) | workspace |
| `ingest PATH [--bins col:q,…] [--categories a,b]` | runs the lifecycle on a file | workspace |
| `query "QUESTION" [--no-llm] [--json]` | grounded answer, evidence, paths, footer | query log |
| `status` | batches and graph statistics | — |
| `experiment [--k K]` (default 5) | the hypothesis benchmark ([10.3](10_verification.md#103-hypothesis-benchmark)) | `experiments/*.json` |
| `rebuild-graph` | regenerates `graph/snapshot.json` from journals and state; syncs Neo4j when enabled | snapshot, Neo4j |
| `sphere [-o FILE] [--dataset ID]` | writes the 3D sphere page | the file |
| `neo4j-sync` | makes the Neo4j mirror equal to the snapshot | Neo4j |
| `llm-check` | probes the configured LLM endpoint | — |
| `migrate --yes` | rebuilds an outdated workspace; the old one is kept as `<workspace>.bak-<time>`; syncs Neo4j when enabled ([6.5](06_graph_and_storage.md#65-versions-and-migration)) | workspace, Neo4j |
| `reset --yes` | deletes the workspace and clears the Neo4j mirror when enabled | workspace, Neo4j |
| `serve` | starts the web UI (same as `python -m ltir.web`) | — |

Writers (`demo`, `ingest`, `reset` — also without `--yes` —, `rebuild-graph`, `migrate` and the web app) take the workspace's writer lock; all but `migrate` then recover interrupted batches (`migrate` refuses while a batch is unfinished instead). While another writer holds the lock they stop with `error: workspace … is in use by another writer process (pid …)` and exit code 2 — while the web app runs, upload through it. The others open read-only and run alongside a writer. Use a separate `WORKSPACE_DIR` for experiments — the default `workspace/` is the knowledge base people work with.

Exit codes: 0 on success (`SKIPPED` included); 1 for a FAILED `demo` or `ingest`, `migrate` or `reset` without `--yes`, `neo4j-sync` without a snapshot, and a failed `llm-check` (which sends one live request); 2 for a busy workspace. Other errors end with a Python traceback and exit code 1: a `migrate` refusal or failure (`PipelineError`), a `query` against an outdated workspace (`RepresentationMismatch`), a Neo4j error in `neo4j-sync`.

## 9.3 Configuration

`ltir/config.py::Config` (a frozen dataclass) is the single source of tunables. `load_config(env_file=None, **overrides)` builds it from the defaults, then the environment (`NAME` = the field name in upper case, optionally loaded from `.env`; values already in the process environment win over the file), then explicit overrides. Values are coerced to the field's type (booleans from `1/true/yes/on`, tuples from comma-separated numbers, paths relative to the repository root). An empty value clears a text field (`EMBEDDING_QUERY_INSTRUCTION=`) and leaves any other field at its default. `LTIR_NO_DOTENV` set to any non-empty value skips `.env` (an explicit `env_file` argument is loaded anyway); the test suite sets it, so no developer credential ever reaches a test. `Config.public_dict()` masks passwords and API keys for logs and the UI. `.env.sample` documents the commonly changed fields; [11.3](11_reference.md#113-parameters) lists them all with defaults.

**Deployment.**

| Part | Requirement |
|---|---|
| Python | 3.12+ (the EDA engine uses PEP 701 f-strings); `pip install -r requirements.txt` (torch first for CUDA) |
| embedding model | `python scripts/download_model.py` once (Qwen3-Embedding-0.6B into `models/`, pinned revision); loaded offline afterwards ([4.4](04_representation.md#44-the-embedding-model)) |
| GPU | optional; the embedder takes ≈ 1.2 GB resident, ≈ 1.7 GB peak; set `EMBEDDING_DEVICE` |
| LLM | any OpenAI-compatible endpoint behind `LLM_BASE_URL` (OpenRouter, Ollama, LM Studio); without one, answers are evidence-only ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| Neo4j | optional mirror (`NEO4J_ENABLED=true`, a running DBMS) ([6.6](06_graph_and_storage.md#66-neo4j-mirror)) |

## 9.4 Error codes

| Code | Meaning | Stage |
|---|---|---|
| `unsupported_file`, `unreadable_file`, `invalid_options` | missing file, type, size, parse error, malformed or unknown column options | VALIDATING |
| `invalid_schema` | too few rows or columns; no categorical dimension after profiling | VALIDATING / PROFILING |
| `no_numeric_targets` | no numeric column | VALIDATING / PROFILING |
| `no_candidates` | empty search space (PROFILING) or nothing passed the pass-1 size screen (DISCOVERING) | PROFILING / DISCOVERING |
| `no_viable_insights` | selection kept nothing | VALIDATING_INSIGHTS |
| `embedding_failure`, `representation_mismatch` | encoder error; fingerprint differs from the workspace's | EMBEDDING |
| `ontology_failure` | an ontology invariant was violated | UPDATING_ONTOLOGY |
| `duplicate_patterns` | a pattern id is already journaled (the append-only journal would be corrupted) | PERSISTING |
| `internal_error` | anything else (the trace is kept) | any |
| `interrupted` | crash recovery found the batch unfinished; the record keeps its last stage in `stage_times` but gets no `failed_stage` | any |
| `WorkspaceBusy` (not a batch code) | another writer process holds the workspace; the CLI exits with code 2, the web app does not start | engine start |
| `busy`, `nothing_to_migrate`, `migration_failed` (not batch codes) | `migrate` refused (a batch is unfinished, or there is no READY batch) or a re-ingest did not end READY; the workspace is unchanged | migrate |
| `unknown_batch`, `unknown_dataset` (not batch codes) | `Engine.process` was given an id with no record; `Engine.delete_dataset` an id no batch carries (HTTP 404) | — |
| `busy` (not a batch code) | `delete_dataset` while one of the dataset's batches runs (HTTP 409); also a `migrate` refusal, above | — |
| warning `graph persistence (Neo4j) failed` | the mirror failed; the batch stays READY | READY |
| QA `answer_mode = fallback` / `empty` | LLM unavailable or switched off / empty graph | query |

The UI maps every code to a plain-language title, cause and tip.

## 9.5 Batch metrics

`record["metrics"]`: `input_rows, columns, numeric_targets, dimensions, search_space, pass1_subgroups, candidate_patterns (distinct cohorts), validated_candidates, validated_insights, pruned {reason: n}, pruned_total, avg_insight_support, avg_insight_weight, embedding_count, embedding_dim, attractors_total, attractors_new, orphan_rate, activation_count, soft_merged, centroid_drift, adaptive_thresh, max_concept_density_pct, density_threshold, damped_attractors, max_centroid_step, clamped_attractors, avg_attractor_degree, graph_edges, edge_counts, graph_patterns, ontology_warnings, timings {validate_s, discover_s, select_s, embed_s, ontology_s, graph_s}, processing_duration_s`. The full lac metrics per batch are in `state/ontology_metrics.csv` ([5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics)); query-side metrics are in `QAResult.metrics` and `logs/queries.jsonl`.

> **Running example.** The demo batch: 5,000 rows, 11 columns, 4 metrics, 3 dimensions, 88 conjunctions, 84 pass-1 subgroups and distinct cohorts, 50 validated, 28 insights, `pruned {weak_effect: 22}`, 4 anchors, 279 graph edges, READY in 13.1 s — of which 10.9 s embedding (the one-off model load, the three blocks and the documents), 1.1 s discovery and 0.9 s ontology. `housing.csv` with two bands (20,640 rows) is READY in about 17 s from a cold CLI start.

## 9.6 Guarantees

* Terminal states are final, with the one exception above: a failure between the READY save and the cache swap rewrites READY as FAILED.
* `READY` implies that journals, state and snapshot are mutually consistent.
* Idempotent: identical content and options are processed once.
* Every failure inside a batch is a batch state and every failure inside a question an answer mode or an HTTP error; the service keeps running. Engine-level refusals (`WorkspaceBusy`, migration refusals) are errors of the command, not batch states.
