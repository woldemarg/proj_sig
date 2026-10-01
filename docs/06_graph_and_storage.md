# 6. The dual graph and its storage

> **In one paragraph.** The structural plane is derived exactly from the insights' scopes: a narrower subgroup SPECIALIZES a broader one, two subgroups that differ in one value are SIBLINGs, and overlapping subgroups whose shifts point in opposite directions CONTRAST. Together with the latent plane of [5](05_latent_anchors.md) and a small schema of datasets, batches, dimensions and metrics, this forms one typed graph. Everything is persisted in a workspace folder: append-only journals hold the insights, their vectors and their memberships; the ontology state holds the anchors; the graph snapshot is derived from both. A batch commits atomically behind a checkpoint, and a workspace built under another representation is refused, then rebuilt by migration. Neo4j can mirror the graph; the files remain the source of truth.

**Code** `ltir/structural.py`, `ltir/graph.py`, `ltir/store.py`, `ltir/migrate.py`, `ltir/neo4j_sink.py`, `ltir/cypher/` · **Tests** `tests/test_structural.py`, `tests/test_persistence.py` · **Previous** [5. Latent anchors](05_latent_anchors.md) · **Next** [7. Question answering](07_question_answering.md)

---

## 6.1 The structural plane

`structural_edges(insights, config)` reads only the conditions, shifts, targets and supports of the insights of **one dataset** at a time — scopes of different datasets are incomparable — and never an embedding (a test enforces that the module needs no numpy). Let `C(P)` be the condition set of pattern `P`:

| Relation | Rule | Edge, weight | Properties |
|---|---|---|---|
| SPECIALIZES | `C(B) ⊂ C(A)` strictly and **no present pattern `M` with `C(B) ⊂ C(M) ⊂ C(A)`** — the covering relation (Hasse diagram) of the present patterns | `A → B` (the narrower subgroup specialises the broader), 1.0 | `added_conditions`, `support_ratio = n_A / n_B`, `metric` (the parent's target), `parent_z`, `child_z` |
| GENERALIZES | the exact inverse of every SPECIALIZES edge | `B → A`, 1.0 | same |
| SIBLING | `|C(A)| = |C(B)|`, and the two differ in exactly one condition, on the same attribute (same parent scope, another value of one partition attribute) | smaller id → larger id, 1.0 | `parent_scope`, `partition_attribute`, `values` |
| CONTRASTS | `overlap = |C(A) ∩ C(B)| / min(|C(A)|, |C(B)|) ≥ CONTRAST_MIN_OVERLAP` (0.5), and some metric has opposite-signed shifts with `min(|z_A|, |z_B|) ≥ CONTRAST_MIN_SHIFT` (0.5); the strongest such metric is recorded | smaller id → larger id, weight = overlap | `metric`, `z_source`, `z_target`, `scope_overlap`, `relation` ∈ {`specialization_reversal` (a SPECIALIZES pair), `sibling` (a SIBLING pair), `overlap`} |

Because conditions are closed intents ([2.3](02_discovery.md#23-deduplication-before-validation)), this is the Hasse diagram of a concept lattice restricted to the present patterns: a strictly smaller extent always has a strictly larger intent, and SPECIALIZES follows extent containment exactly. The EDA enumerates 2- and 3-conjunctions, but implied conditions can make intents longer.

> **Running example.** `P-bc4657a04746` (phones ∧ US) is specialised by its three channel refinements (`phones ∧ US ∧ online`, `∧ retail`, `∧ partner`, each `added_conditions: ["channel=…"]`), has two SIBLINGs (phones in the other regions) and contrasts with nothing. A typical contrast on the demo is `laptops ∧ EU: margin +1.13 sd` ↔ `laptops ∧ EU ∧ retail: margin −0.86 sd`, a `specialization_reversal` with weight 1.0 — the planted contrasting subgroup.

## 6.2 The graph schema

`graph.build_snapshot(ws, ontology, config)` assembles the snapshot from the journals, the dataset profiles, the batch registry and the concept store; `DualGraph(snapshot)` is the read-only adjacency index that traversal, evidence and the UI use.

| Node | Id | Label | Key properties |
|---|---|---|---|
| Pattern | `P-<hash>` | the headline ([8.5](08_interface.md#85-text-shown-to-people)) | the journal record unchanged: `Insight` fields + `row_id`, `canonical`, `embedding` |
| Attractor | `A-<k>` | the anchor label | [5.8](05_latent_anchors.md#58-how-an-anchor-is-described) |
| Dimension | `D:<dataset>:<column>` | raw column name | `name, cardinality, entropy` — the EDA-selected dimensions plus every attribute a closed intent uses |
| Metric | `M:<dataset>:<column>` | `humanize(name)` | `name, global_median, global_mad` |
| Dataset | `DS:<dataset_id>` | file name | `filename, rows, columns` |
| Batch | `B:<batch_id>` | batch id | `batch_seq, created_at, status` |

| Edge | From → to | Plane | Weight and properties |
|---|---|---|---|
| SPECIALIZES, GENERALIZES, SIBLING, CONTRASTS | Pattern → Pattern | structural | [6.1](#61-the-structural-plane) |
| ACTIVATES | Pattern → Attractor | bridge | alignment with the **current** centroid; `strength`, `insight_weight`, `engine_weight`, `alignment_at_ingest`, `source`, `weak`, `batch_id` |
| RELATED_TO | Attractor → Attractor (smaller → larger id) | latent | mutual-kNN cosine; `kind: mutual_knn` |
| HAS_SCOPE | Pattern → Dimension | schema | 1.0; `value` |
| TARGETS | Pattern → Metric | schema | `min(1, |z| / 3)` for the target and every shift with `|z| ≥ MIN_COMPONENT_Z`; `role` primary / secondary, `z`, medians |
| DISCOVERED_IN | Pattern → Batch | provenance | 1.0 |
| OF_DATASET | Batch → Dataset | provenance | 1.0 |

Edge ids are `<TYPE>:<source>-><target>`. Ids are dataset-scoped, because datasets reuse column names with different meanings. Activations that point to an unjournaled pattern or a dead anchor are skipped when the snapshot is built.

> **Running example.** The demo snapshot holds 28 Pattern, 4 Attractor and the schema nodes, and 279 edges: SPECIALIZES 24, GENERALIZES 24, SIBLING 37, CONTRASTS 16, ACTIVATES 28, RELATED_TO 3, HAS_SCOPE 70, TARGETS 48, DISCOVERED_IN 28, OF_DATASET 1.

## 6.3 The workspace on disk

`WORKSPACE_DIR` (default `workspace/`) holds one knowledge base. `store.Workspace` owns the layout; lac's `ChunkJournal` (constructed with `records_name="patterns.jsonl"`) owns the journal files.

| Path | Format | Content | Written by |
|---|---|---|---|
| `registry/batches/<batch_id>.json` | JSON | the batch record ([9.1](09_operations.md#91-the-batch-lifecycle)) | `save_batch`, atomically at every stage |
| `uploads/` | files | uploads waiting to be processed | web app |
| `datasets/<ds>/source.<ext>` | the uploaded bytes | provenance, migration input | pipeline |
| `datasets/<ds>/profile.json` | JSON | the batch `profile` (columns, selected dimensions, global medians and MADs, cardinality, entropy, bands, overrides) | pipeline |
| `datasets/<ds>/covers.npz` | npz: pattern id → int64 row positions | the `rows_ref` target | pipeline |
| `datasets/<ds>/rejections.json` | JSON list `{expression, reason, detail}` | discovery merges, then selection rejections (`detail` e.g. `same rows as …`, `|z|=0.17 p_adj=1 emm=0.07`) | pipeline |
| `journal/patterns.jsonl` | one JSON record per line | `Insight.to_record()` + `row_id`, `canonical` (incl. the document), `embedding` | `ChunkJournal.append_batch` |
| `journal/activations.jsonl` | one JSON record per line | activation records ([5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics)) | same |
| `journal/embeddings.mmap` + `embeddings_meta.json` | float32 `(rows, 1152)`; `{rows, dim}` | the insight vectors; row = `row_id` | same (grown by a `.tmp` write and an atomic rename) |
| `journal/blocks/<batch_id>.npz` | `scope`, `target`, `phenomenon`, float32 `(n, 384)` | the three blocks of each vector | `Workspace.append` |
| `state/concepts.npz`, `state.json`, `orphan_buffer.npz` | lac `ConceptStore` | centroids, counts, ids, timestamps (no text) | `ConceptStore.save` |
| `state/representation.json` | JSON | `EmbeddingSpec` + fingerprint | `record_representation`, at the first commit |
| `state/sig_state.json` | JSON | the next batch sequence | pipeline |
| `state/ontology_metrics.csv` | CSV | lac `BatchMetrics` rows | `MetricsRecorder` |
| `graph/snapshot.json` | `{version, created_at, representation, nodes[{id, kind, label, props}], edges[{id, source, target, type, plane, weight, props}], stats}` | the derived dual graph (`SNAPSHOT_VERSION` 2) | `save_graph`, atomically |
| `graph/sphere.html` | standalone Plotly page | the latent sphere ([8.4](08_interface.md#84-the-latent-sphere)) | after every READY batch (`SPHERE_EXPORT`) |
| `logs/queries.jsonl` | one JSON per question | `{at, question, mode, metrics, seeds, evidence, citations}` | `log_query` (empty-graph questions are not logged) |
| `experiments/*.json` | JSON | benchmark results | `python -m ltir experiment` |

The snapshot is **derived data**: `ltir rebuild-graph` regenerates it from the journals, the profiles, the registry and the concept store, recomputing ACTIVATES alignments against the current centroids and RELATED_TO from the topology. Nothing in the workspace stores raw rows other than the source copy.

## 6.4 Commit, rollback and recovery

**Commit order** (stages UPDATING_ONTOLOGY and PERSISTING):

```text
checkpoint → ontology ingest → duplicate guard (duplicate_patterns if a pattern id is already journaled)
→ journal append (records, activations, vectors) → ConceptStore.save → record_representation → batch sequence
→ snapshot build + atomic write → batch saved READY → checkpoint discarded → committed caches swapped in
→ optional: Neo4j publish, sphere export (failures there are warnings; the batch stays READY)
```

**Checkpoint and rollback.** `checkpoint()` copies `state/` to `checkpoint/` and stores `ChunkJournal.mark()` (the sizes of both logs and the record count). `rollback()` truncates the journal to the mark and restores `state/`; it runs on any failure after the checkpoint. The checkpoint outlives the READY write, so a crash between the two still recovers. Files outside the journal and the state survive a failed batch — `journal/blocks/<batch>.npz`, `covers.npz`, `profile.json`, `rejections.json` and the source copy — and so does a snapshot written just before a failure of the READY save; the next successful batch or `rebuild-graph` replaces it.

**Committed caches.** `Engine.graph()` and `Engine.frame()` (`LatentFrame`: pattern id → unit vector, anchor id → centroid) are rebuilt from committed state after READY, so readers never see a half-appended journal or a half-saved ontology. A reader that finds an outdated snapshot version rebuilds and writes the snapshot.

**Recovery.** A *writer* engine (the web app; CLI `demo`, `ingest`, `reset`, `rebuild-graph`) starts by rolling back and failing (`interrupted`) every non-terminal batch that no live process owns — including batches still `UPLOADED` and therefore unowned. Read-only engines (`query`, `status`, `experiment`, `sphere`, `neo4j-sync`, `migrate`'s target) open with `recover=False` and never roll back. Batches run one at a time inside a process (a re-entrant lock; the web app uses a single worker thread), but there is no lock between processes: a CLI writer started while the web app has uploads queued fails those uploads, and two writer processes must not process batches in one workspace at the same time.

**Idempotency.** The dataset id is content-addressed, so a READY batch for the same id makes a new upload `SKIPPED` (`duplicate_of`); pattern ids are deterministic; Neo4j writes are MERGE-only. A failed batch keeps its `batch_seq`, and the next batch reuses that number.

## 6.5 Versions and migration

A workspace records the `EmbeddingSpec` of its first batch ([4.6](04_representation.md#46-representation-identity-and-versions)). Any later batch or query under another fingerprint raises `RepresentationMismatch` before touching the journal or the ontology; `rebuild-graph` first runs `check_versions`, which compares the stored canonical and representation versions without loading a model.

`python -m ltir migrate --yes` (`migrate.migrate_workspace`) rebuilds a workspace with the current code:

1. refuses while any batch is non-terminal;
2. re-ingests every READY batch's stored `datasets/<id>/source.<ext>` with its recorded `bins` and `categories`, in `batch_seq` order, into `<workspace>.migrating` (Neo4j off);
3. on success renames the old workspace to `<workspace>.bak-<UTC timestamp>` — never deleted — and the new one into place, and prints old → new batch ids and pattern counts.

Dataset ids stay stable (content + options); pattern ids change only where closure adds conditions. A failure leaves the old workspace untouched. `python -m ltir reset --yes` deletes a workspace instead; it is never needed for a version change.

## 6.6 Neo4j mirror

With `NEO4J_ENABLED=true`, every READY batch is published to `NEO4J_DATABASE` (`neo4j_sink`):

* `CREATE DATABASE <db> IF NOT EXISTS` (ignored on editions without multi-database support), unique constraints on all six labels;
* parameterised `UNWIND … MERGE (n:Label {id}) SET n += props` and `MERGE (s)-[r:TYPE]->(t) SET r += props, r.weight = …`, batched by `NEO4J_LOAD_BATCH_SIZE`;
* RELATED_TO deleted and rewritten in the same transaction on every publish (the topology is recomputed each batch);
* nested properties stored as JSON strings in `<key>_json` (`conditions_json`, `shifts_json`, `canonical_json`, `signature_json`, `centroid_json`, …).

`python -m ltir neo4j-sync` backfills from the snapshot (regardless of `NEO4J_ENABLED`). A Neo4j failure leaves the batch READY with `neo4j.status = failed` and the warning `graph persistence (Neo4j) failed`. The mirror only adds: nodes and relationships other than RELATED_TO are never deleted (also not after `reset`, `migrate` or `rebuild-graph`), `SET +=` never removes a property, and properties that become `None` keep their old value — clear the database when the workspace is rebuilt. `ltir/cypher/queries/transversal.cypher` is the Browser equivalent of the traversal of [7.3](07_question_answering.md#73-transversal-traversal). Validated against a live Neo4j Enterprise instance: the stored graph equalled the snapshot (counts per label and per relationship type) and republishing left the counts unchanged.

## 6.7 Configuration and guarantees

| Parameter | Default |
|---|---|
| `WORKSPACE_DIR` | `workspace` |
| `CONTRAST_MIN_OVERLAP`, `CONTRAST_MIN_SHIFT` | 0.5, 0.5 |
| `NEO4J_ENABLED`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `NEO4J_LOAD_BATCH_SIZE` | false, `bolt://localhost:7687`, `neo4j`, "", `sigv1`, 5000 |

Guarantees (tests: `test_structural.py`; `test_persistence.py`: write → reload → rebuild equality, idempotent re-ingest, graph consistency, rollback after a simulated failure, representation mismatch, interrupted-batch recovery, migration with backup, a batch that stays READY when a post-commit save fails, MERGE-only Neo4j publish with property flattening against a fake driver):

* GENERALIZES = inverse(SPECIALIZES); no transitive SPECIALIZES edges; structural edges connect patterns of one dataset; RELATED_TO connects only anchors; the latent plane has at most `RELATED_TO_PEER_COUNT · N / 2` edges.
* `journal rows == vector rows == ConceptStore.next_chunk_id`; snapshot ids are identical after reload and after rebuild.
* `READY` implies that journals, state and snapshot are mutually consistent.
