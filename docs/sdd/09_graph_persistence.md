# SDD 09 — Persistence (journals, state, snapshot, Neo4j mirror)

## Purpose
Persist all validated insights, activations, attractors, relationships and embedding metadata so the ontology can be restored and every batch is repeatable, recoverable and idempotent.

## Scope
`ltir/store.py::Workspace`, `ltir/graph.py::build_snapshot/DualGraph`, `ltir/neo4j_sink.py`, `ltir/cypher/`.

## Inputs
Per batch: pattern records, unit vectors, activation records, embedding blocks, covers, profile, rejections, and `ConceptStore` state.

## Outputs
The `WORKSPACE_DIR` layout (below); `graph/snapshot.json`; optionally the Neo4j database `NEO4J_DATABASE`.

## Dependencies
lac `ChunkJournal` (reused via a 3-line subclass that renames `chunks.jsonl` → `patterns.jsonl`) and lac `ConceptStore.save/load`; numpy; the optional `neo4j` driver.

## Layout (`WORKSPACE_DIR`, default `sig/workspace/`)
| Path | Content | Writer |
|---|---|---|
| `registry/batches/<batch_id>.json` | lifecycle record: status, stage_times, profile, metrics, warnings, error, neo4j status | `save_batch` (atomic) |
| `datasets/<id>/source.<ext>` | copy of the uploaded file | pipeline |
| `datasets/<id>/profile.json`, `covers.npz`, `rejections.json` | schema summary; `pattern_id → covered rows`; pruned candidates | pipeline |
| `journal/patterns.jsonl` | append-only `Insight.to_record()` + `row_id` + `canonical{…, document, components}` + `embedding{fingerprint, model_id, dim, representation_version}` | lac `ChunkJournal.append_batch` |
| `journal/activations.jsonl` | append-only activation records (SDD 07) | same |
| `journal/embeddings.mmap` + `embeddings_meta.json` | float32 `(rows, 1152)`; row = `row_id` | same |
| `journal/blocks/<batch_id>.npz` | scope / target / phenomenon blocks (the tripartite parts of each row) | `Workspace.append` |
| `state/concepts.npz`, `state.json`, `orphan_buffer.npz` | ontology state | lac `ConceptStore.save` |
| `state/representation.json` | `EmbeddingSpec` of the workspace | `record_representation` (at the first commit, after `ConceptStore.save`); validated by `check_representation` |
| `state/sig_state.json`, `state/ontology_metrics.csv` | batch sequence; lac `BatchMetrics` | pipeline / lac `MetricsRecorder` |
| `graph/snapshot.json` | derived dual-layer graph `{version, created_at, representation, nodes, edges, stats}` | `save_graph` (atomic) |
| `graph/sphere.html` | standalone 3D latent sphere (lac projector; SDD 13) | `Engine._export_sphere` after READY |
| `logs/queries.jsonl` | query observability | `log_query` |
| `experiments/*.json` | hypothesis benchmark results | `experiment.py` |

## Algorithms
* **Commit order**: checkpoint → ontology ingest → duplicate guard (`duplicate_patterns` if any pattern id is already journaled) → journal append (mmap growth is a `.tmp` write + atomic rename) → `ConceptStore.save` → `record_representation` → batch sequence commit → snapshot build + atomic write → batch saved `READY` → discard checkpoint. The checkpoint outlives the READY write, so a crash between the two still recovers.
* **Checkpoint/rollback**: `checkpoint()` copies `state/` to `checkpoint/` and stores `ChunkJournal.mark()` (record/activation log sizes, vector rows). `rollback()` calls `ChunkJournal.truncate(mark)` (the journal owns its files) and restores `state/`. It runs on any failure after the checkpoint.
* **Recovery**: a *writer* `Engine` (`recover=True`: the web app, `demo`, `ingest`, `reset`, `rebuild-graph`) rolls back every non-terminal batch whose `owner_pid` is no longer alive and marks it `FAILED/interrupted`. Read-only engines (`query`, `status`, `experiment`, `sphere`, …) open with `recover=False`, so a second process never rolls back a batch that is still running.
* **Committed caches**: `Engine.graph()` and `Engine.frame()` (`LatentFrame`: pattern id → unit vector, attractor id → centroid) are rebuilt from committed state after READY, so readers never see a half-appended journal or a half-saved ontology.
* **Idempotency**: `dataset_id` is content-addressed (SDD 02), so a READY batch for the same id makes a new upload `SKIPPED` (`duplicate_of`). Pattern ids are deterministic, and Neo4j writes are MERGE-only.
* **Snapshot as derived data**: `build_snapshot()` reads journals + profiles + registry + `ConceptStore`. ACTIVATES alignment is recomputed against current centroids, and RELATED_TO from `calculate_knn_topology`. `ltir rebuild-graph` regenerates it.
* **Migration**: `python -m ltir migrate --yes` (`ltir/migrate.py::migrate_workspace`) rebuilds a workspace with the current code: it refuses while a batch is in flight, re-ingests every READY batch's stored `datasets/<id>/source.<ext>` with its recorded `bins`/`categories` in `batch_seq` order into `<workspace>.migrating` (Neo4j off), and on success renames the old workspace to `<workspace>.bak-<UTC timestamp>` (never deleted) and the new one into place. Dataset ids are stable (content + options); pattern ids change only where closure adds conditions. A failure leaves the old workspace untouched.
* **Representation guard**: the first batch stores the spec; any later fingerprint mismatch raises `RepresentationMismatch`. `rebuild-graph` checks the stored canonical/representation versions first (`check_versions`, no model load), so a journal written by an older version is refused with a reset hint instead of being re-derived.
* **Neo4j mirror** (`NEO4J_ENABLED=true`): `CREATE DATABASE <db> IF NOT EXISTS` (ignored on Community), unique constraints on all six labels, parameterised `UNWIND … MERGE (n:Label {id})` / `MERGE (s)-[r:TYPE]->(t)` with `SET += props`, and RELATED_TO replaced wholesale per publish (lac's MERGE-only stale-edge trade-off is removed). Nested properties are stored as `<key>_json` strings. `ltir neo4j-sync` backfills from the snapshot. The Browser query lives in `cypher/queries/transversal.cypher`.

## Configuration
`WORKSPACE_DIR`, `NEO4J_ENABLED` (false), `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` (`sigv1`), `NEO4J_LOAD_BATCH_SIZE`.

## Failure modes
| Failure | Effect |
|---|---|
| any exception after checkpoint | rollback; FAILED with `internal_error`/specific code; no partial knowledge |
| crash mid-batch | recovered at next start (`interrupted`) |
| `representation_mismatch` | FAILED before any state change |
| Neo4j unreachable/error | batch stays READY; `neo4j.status=failed` + warning ("graph persistence (Neo4j) failed") |
| snapshot missing/corrupt | `rebuild-graph` regenerates from journals |

## Invariants
`journal rows == mmap rows == ConceptStore.next_chunk_id`; every activation references a journaled pattern and a live attractor; snapshot ids are identical after reload and after rebuild.

## Testing requirements
`tests/test_persistence.py`: write/reload/rebuild equality, idempotent re-ingestion, graph consistency, rollback after a simulated failure, representation mismatch, interrupted-batch recovery, ingestion failure codes, MERGE-only Neo4j publish with prop flattening (fake driver).

## Integration points
The UI and traversal read `DualGraph(snapshot)`; the CLI covers `status`, `rebuild-graph`, `neo4j-sync`, `migrate`, `reset`.

## Current implementation status
Implemented and **validated live** on the local Neo4j Desktop Enterprise DBMS. `sigv1` was created next to the existing databases, which were not touched. The stored graph equalled the snapshot at that time (60 Pattern, 12 Attractor, 6 Dimension, 10 Metric, 2 Dataset, 2 Batch; all 10 relationship types with the same counts; counts differ after the discovery repairs). Republishing leaves the counts unchanged (idempotent). `cypher/queries/transversal.cypher` returns one best path per target with the same ranking as `ltir/traversal.py`. Tests use a fake driver and never read `sig/.env`, so they cannot write to `sigv1`.
