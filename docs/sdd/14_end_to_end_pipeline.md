# SDD 14 — End-to-end pipeline (dataset lifecycle)

## Purpose
Run every uploaded dataset as one processing unit (batch) through an explicit, observable lifecycle that commits atomically.

## Scope
`ltir/pipeline.py::Engine` (the single stateful service behind the CLI and the web app).

## Inputs
`Engine(config, *, embedder=None, llm=None, recover=True)`; `Engine.submit(path, filename, bins=None, categories=None)` → record (`UPLOADED`; `None` = workspace defaults); `Engine.process(batch_id)`; `Engine.ingest_file()` = submit + process. Read-only callers pass `recover=False` (SDD 09).

## Outputs
A batch record (`registry/batches/<batch_id>.json`, atomically rewritten at each stage):
`{batch_id, batch_seq, dataset_id, filename, source_path, bins, categories, owner_pid, status, stage_times{STAGE: iso}, created_at, updated_at, profile{rows, columns, numerics, categoricals, dropped_columns, selected_dimensions, search_space_size, global_medians, global_mads, dimension_cardinality, dimension_entropy, derived_columns, bins, categorical_overrides}, metrics{…}, warnings[], error{code, message[, trace]}, failed_stage, duplicate_of, neo4j{status, …}}`.

## Lifecycle
```text
UPLOADED → VALIDATING → PROFILING → DISCOVERING → VALIDATING_INSIGHTS → EMBEDDING
         → UPDATING_ONTOLOGY → BUILDING_GRAPH → PERSISTING → READY        (terminal: READY | FAILED | SKIPPED)
```
| Stage | Work | Module |
|---|---|---|
| VALIDATING | load/validate file, bands, dataset id, idempotency check (READY batch for the same id → `SKIPPED`), copy source | 02 |
| PROFILING / DISCOVERING | EDA step1–3 / step4 + step4b | 03 |
| VALIDATING_INSIGHTS | `build_insights` + `select_insights`; rejections persisted | 03, 04 |
| EMBEDDING | canonicalise + encode; representation guard | 05, 06 |
| UPDATING_ONTOLOGY | **checkpoint**, `LatentOntology.ingest` | 07, 09 |
| BUILDING_GRAPH | pattern records (+ canonical, embedding meta), covers | 08 |
| PERSISTING | duplicate guard, journal append, state save, representation record, batch sequence, snapshot | 09 |
| READY | batch saved, then checkpoint discarded and the committed caches (`graph()`, `frame()`) swapped in; optional Neo4j publish and sphere export (failure = warning) | 09 |

Batches run one at a time (`RLock`; the web app uses a single worker thread). Any exception before the batch is saved as `READY` triggers rollback, discards the checkpoint and marks the batch `FAILED` with `failed_stage`. Neo4j publish and sphere export run after that save; a failure there leaves the batch `READY`. On start, a writer engine's `_recover_interrupted()` rolls back and fails batches left in a non-terminal state whose `owner_pid` is dead (SDD 09).

## Metrics (observability, requirement §29)
`input_rows, columns, numeric_targets, dimensions, search_space, candidate_patterns, validated_candidates, validated_insights, pruned{reason: n}, pruned_total, avg_insight_support, avg_insight_weight, embedding_count, embedding_dim, attractors_total, attractors_new, orphan_rate, activation_count, soft_merged, centroid_drift, adaptive_thresh, max_concept_density_pct, avg_attractor_degree, graph_edges, edge_counts, graph_patterns, ontology_warnings, timings{validate_s, discover_s, select_s, embed_s, ontology_s, graph_s}, processing_duration_s`. Query-side metrics (traversal depth, evidence count, LLM latency) are in `QAResult.metrics` and `logs/queries.jsonl`.

## Error codes (requirement §30)
| Code | Meaning | Stage |
|---|---|---|
| `unsupported_file`, `unreadable_file`, `invalid_options` | file type/size/parse/band options | VALIDATING |
| `invalid_schema` | too few rows/columns, or no categorical dimension | VALIDATING / PROFILING |
| `no_numeric_targets` | no numeric columns | VALIDATING / PROFILING |
| `no_candidates` | empty search space / nothing passed pass 1 | DISCOVERING |
| `no_viable_insights` | selection kept nothing | VALIDATING_INSIGHTS |
| `embedding_failure`, `representation_mismatch` | encoder error, fingerprint change | EMBEDDING |
| `ontology_failure` | lac invariant violation | UPDATING_ONTOLOGY |
| `duplicate_patterns` | a pattern id is already journaled (would corrupt the append-only journal) | PERSISTING |
| `internal_error` | anything else (trace kept) | any |
| `interrupted` | crash recovery | any |
| warning: `graph persistence (Neo4j) failed` | mirror failure | READY |
| QA `answer_mode=fallback` / `empty` | LLM unavailable / empty graph | query |

## Configuration
All of `Config` (SDD 01); lifecycle-specific: `WORKSPACE_DIR`, `NEO4J_ENABLED`.

## Invariants
Terminal states are final. `READY` implies journals, state and snapshot are mutually consistent. Idempotent: identical content + options is processed once.

## Testing requirements
`tests/test_e2e.py` (full flow; all stages recorded; metrics keys; grounded answer), `tests/test_persistence.py` (rollback, recovery, idempotency, failure codes), `tests/test_ui_smoke.py` (upload → READY via the API).

## Integration points
CLI (`demo`, `ingest`, `query`, `status`, `experiment`, `rebuild-graph`, `neo4j-sync`, `llm-check`, `reset`, `serve`) and the web API (SDD 13).

## Current implementation status
Implemented. Measured on the dev machine (RTX 4060, CUDA embeddings): synthetic 5 000 × 11 → 84 candidates, 28 insights, 7 themes, READY in 16–19 s including the one-off model load (the 20-resample bootstrap and the chi-square driver test add ≈ 1 s). `housing.csv` 20 640 × 10 + 2 bands → READY in ≈ 12 s.
