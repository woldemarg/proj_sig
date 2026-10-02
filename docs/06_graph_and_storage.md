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
| ACTIVATES | Pattern → Attractor | bridge | `alignment` with the **current** centroid (also the weight); `strength`, `insight_weight`, `engine_weight`, `alignment_at_ingest`, `source`, `batch_id`, `weak` — the membership was rerouted below `MIN_ACTIVATION_ALIGNMENT` at ingest, or its current alignment is below that floor; it counts for coverage and is not walked |
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
| `<workspace>.writer.lock` (beside the folder) | the writer's pid, under an OS lock | the exclusive writer lock ([6.4](#64-commit-rollback-and-recovery)) | `store.acquire_writer_lock` |
| `registry/batches/<batch_id>.json` | JSON | the batch record ([9.1](09_operations.md#91-the-batch-lifecycle)) | `save_batch`, atomically at every stage |
| `uploads/` | files | the web app's uploads, kept after processing until a reset | web app |
| `datasets/<ds>/source.<ext>` | the uploaded bytes | provenance, migration input | pipeline |
| `datasets/<ds>/profile.json` | JSON | the batch `profile` (columns, selected dimensions, global medians and MADs, cardinality, entropy, bands, overrides) | pipeline |
| `datasets/<ds>/covers.npz` | npz: pattern id → int64 row positions | the `rows_ref` target | pipeline |
| `datasets/<ds>/rejections.json` | JSON list `{expression, reason, detail}` | discovery merges, then selection rejections (`detail` e.g. `same rows as …`, `|z|=0.17 p_adj=1 emm=0.07`) | pipeline |
| `journal/patterns.jsonl` | one JSON record per line | `Insight.to_record()` + `row_id`, `canonical` (incl. the document), `embedding` | `ChunkJournal.append_batch` |
| `journal/activations.jsonl` | one JSON record per line | activation records ([5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics)) | same |
| `journal/embeddings.mmap` + `embeddings_meta.json` | float32 `(rows, 1152)`; `{rows, dim}` | the insight vectors; row = `row_id` | same (grown by a `.tmp` write and an atomic rename) |
| `journal/blocks/<batch_id>.npz` | `scope`, `target`, `phenomenon`, `document`, float32 `(n, 384)`; `pattern_ids` | the three blocks of each vector, and the canonical-document embedding the naive text baseline compares questions with | `Workspace.append`; `add_document_vectors` back-fills `document` for an older batch ([7.6](07_question_answering.md#76-baselines)) |
| `state/concepts.npz`, `state.json`, `orphan_buffer.npz` | lac `ConceptStore` | centroids, counts, ids, timestamps (no text) | `ConceptStore.save` |
| `state/representation.json` | JSON | `EmbeddingSpec` + fingerprint | `record_representation`, at the first commit |
| `state/sig_state.json` | JSON | the next batch sequence | pipeline |
| `state/ontology_metrics.csv` | CSV | lac `BatchMetrics` rows | `MetricsRecorder` |
| `graph/snapshot.json` | `{version, created_at, representation, nodes[{id, kind, label, props}], edges[{id, source, target, type, plane, weight, props}], stats}` | the derived dual graph (`SNAPSHOT_VERSION` 2) | `save_graph`, atomically |
| `graph/literals.npz` | npz: `texts`, `vectors` float32 `(N, 384)`, `fingerprint` | the embeddings of the literal catalog ([7.1.1](07_question_answering.md#711-literal-grounding)) by text — a cache: the catalog itself is rebuilt from the graph, a text it lacks (or a file of another fingerprint) is embedded again; outside `state/`, so no rollback restores it | `Engine.catalog` of a writer, on the first question after a commit that added literals |
| `pending.json` | `{batch_id` or `dataset_id, checkpoint}` | the unfinished transaction ([6.4](#64-commit-rollback-and-recovery)) | `Workspace.transaction`; removed at its end |
| `graph/sphere.html` | standalone Plotly page | the latent sphere ([8.4](08_interface.md#84-the-latent-sphere)) | on a background thread after a READY batch (`SPHERE_EXPORT`; only the latest of several queued exports runs, and one that finishes after a reset is discarded); `python -m ltir sphere` |
| `logs/queries.jsonl` | one JSON per question | `{at, question, mode, metrics, seeds, evidence, citations}` | `log_query` (empty-graph questions are not logged) |
| `experiments/*.json` | JSON | benchmark results | `python -m ltir experiment` |

**Atomic replacement on Windows.** Batch records, the snapshot and the other JSON files, the blocks files and the vector matrix are written beside the target and renamed over it (`fileio.replace_file`). On Windows that rename fails while another handle has the target open, and opening the target fails while it is being renamed over — both as `PermissionError` (WinError 5). The UI polls the batch records while the worker rewrites them, so the rename and the readers (`read_json`, the blocks and matrix loaders) retry with a short backoff for up to `fileio.SHARING_RETRY_S` (2 s, a module constant); a reader holds a file for microseconds. Elsewhere a `PermissionError` is raised at once. Without the retry, a batch failed with that error in the middle of a stage.

The snapshot is **derived data**: `ltir rebuild-graph` regenerates it from the journals, the profiles, the registry and the concept store, recomputing ACTIVATES alignments against the current centroids and RELATED_TO from the topology. Raw rows are stored only in the source copy and, for web uploads, in `uploads/`; both stay until a reset.

## 6.4 Commit, rollback and recovery

**Commit order** (stages UPDATING_ONTOLOGY and PERSISTING):

```text
transaction: checkpoint + marker pending.json → ontology ingest → duplicate guard (duplicate_patterns if a pattern
id is already journaled) → journal append (records, activations, vectors) → ConceptStore.save → record_representation
→ batch sequence → snapshot build + atomic write → committed graph and frame loaded → batch saved READY (the commit
point) → marker removed, checkpoint discarded
then: readers switched to the new graph and frame → sphere export queued → optional Neo4j sync (failures are warnings)
```

**One transaction protocol.** A batch commit and a dataset deletion ([6.8](#68-deleting-a-dataset)) each run in `Workspace.transaction(intent)`: it checkpoints, then writes `pending.json` holding the intent (`batch_id` or `dataset_id`) and the checkpoint. `checkpoint()` copies `state/` and `graph/snapshot.json` to `checkpoint/` and stores `ChunkJournal.mark()` (the sizes of both logs and the record count); a deletion, which shrinks the journal, passes `journal=True` and the journal folder is copied too. An exception in the body rolls back — `rollback()` truncates the journal to the mark (or restores the copied folder), restores `state/`, puts the previous snapshot back (or removes it when there was none) and returns any moved-aside files — then removes the marker and the checkpoint. A rollback that fails itself is logged and leaves both: the batch is recorded FAILED with the warning `rollback failed`, and every later transaction is refused (`rollback_pending`) until a writer restart retries it, so nothing can overwrite the only good copy. The READY record is a batch's commit point, so a failure to write it — for example a Windows sharing violation that outlasts the retry of [6.3](#63-the-workspace-on-disk) — rolls the batch back completely: readers never see a snapshot whose patterns the journal does not hold. Files outside the journal, the state and the snapshot survive a failed batch — `journal/blocks/<batch>.npz`, `covers.npz`, `profile.json`, `rejections.json` and the source copy; nothing reads them for a batch that is not READY, and the next ingest of that dataset overwrites them.

**Committed caches.** `Engine.graph()` and `Engine.frame()` (`LatentFrame`: pattern id → unit vector, anchor id → centroid, pattern id → document vector) are loaded inside the transaction, before the commit point, and switched in together after it — so a committed batch cannot fail afterwards, and readers never see a half-appended journal or a half-saved ontology. The literal catalog (`Engine.catalog()`) belongs to one graph: the first question after a switch builds it from that graph, under its own lock, so a question never waits for a running batch. An engine that finds an outdated snapshot version rebuilds and writes the snapshot.

**One writer per workspace.** A *writer* engine (the web app; CLI `demo`, `ingest`, `reset` — with or without `--yes` —, `rebuild-graph`; `migrate`) first takes the workspace's exclusive writer lock on `<workspace>.writer.lock`, a file beside the workspace folder (so a migration can rename the folder while it is held) that also records the holder's pid: on Windows `msvcrt` locks one byte (at offset 1 MiB, so the pid stays readable), on POSIX `fcntl.flock` locks the whole file. The lock is held until the process exits and released by the operating system, also after a crash; it is re-entrant within a process. A second writer process gets `WorkspaceBusy` — the CLI prints it and exits with code 2, the web app refuses to start — instead of interleaving batches with the first or touching its queue. Read-only engines (`query`, `status`, `experiment`, `sphere`, `neo4j-sync`) take no lock and change no batch, journal or ontology state; they may still write derived files — an outdated snapshot rebuilt, `logs/queries.jsonl`, `experiments/*.json`, `graph/sphere.html`. Within the writer, batches run one at a time (a re-entrant lock; the web app uses a single worker thread).

**Recovery.** Holding the lock, the writer first finishes an interrupted transaction (`Workspace.recover`): a batch whose READY record was written had committed and keeps its writes; anything else — a batch before its commit point, any deletion — is rolled back from the checkpoint in the marker. `rollback()` is repeatable, so a rollback that failed halfway is completed. Then every unfinished batch fails (`interrupted`) — with no other writer alive, any batch that is not terminal was left by a process that died, including uploads that were still queued.

**Idempotency.** The dataset id is content-addressed, so a READY batch for the same id makes a new upload `SKIPPED` (`duplicate_of`); pattern ids are deterministic; a Neo4j publish is idempotent (MERGE on ids, then the same stale deletions). A failed batch keeps its `batch_seq`, and the next batch reuses that number.

## 6.5 Versions and migration

A workspace records the `EmbeddingSpec` of its first batch ([4.6](04_representation.md#46-representation-identity-and-versions)). Any later batch or query under another fingerprint raises `RepresentationMismatch` before touching the journal or the ontology; `rebuild-graph` first runs `check_versions`, which compares the stored canonical and representation versions without loading a model.

`python -m ltir migrate --yes` (`migrate.migrate_workspace`) rebuilds a workspace with the current code:

1. takes the writer lock (refused while the web app or another writer runs) and refuses while any batch is unfinished;
2. re-ingests every READY batch's stored `datasets/<id>/source.<ext>` with its recorded `bins` and `categories`, in `batch_seq` order, into `<workspace>.migrating` (Neo4j and the sphere export off, so no thread writes into the folder being swapped); FAILED and SKIPPED batches do not carry over, and a batch recorded without options (`None`) is resolved against the current defaults again;
3. on success renames the old workspace to `<workspace>.bak-<UTC timestamp>` — never deleted — and the new one into place (if that second rename fails, the old workspace is renamed back), syncs the Neo4j mirror when it is enabled, and prints old → new batch ids and pattern counts.

Dataset ids stay stable (content + options); pattern ids change only where closure adds conditions. A failure leaves the old workspace untouched and the partial `<workspace>.migrating` folder for inspection (the next migration replaces it). `.gitignore` covers these folders and the lock file for the default name `workspace` only. `python -m ltir reset --yes` deletes a workspace instead; it is never needed for a version change.

## 6.6 Neo4j mirror

With `NEO4J_ENABLED=true`, every publish makes `NEO4J_DATABASE` equal to the snapshot (`neo4j_sink.publish_snapshot`):

* before the write, each as its own statement: `CREATE DATABASE <db> IF NOT EXISTS WAIT` (errors ignored: an edition without multi-database support, or a user without the right, uses the database as configured), then unique constraints on all six labels;
* in one write transaction, parameterised `UNWIND … MERGE (n:Label {id}) SET n = props, n.id = id` (the props include the node's `label`) and `MERGE (s)-[r:TYPE]->(t) SET r = props, r.weight = …`, batched by `NEO4J_LOAD_BATCH_SIZE` — properties are **replaced**, so a property that disappeared or became `None` is removed (`None` values are never written);
* then every node of the six labels whose id the snapshot does not hold is deleted with its relationships, and every relationship of the ten types whose (source, target) pair it does not hold is deleted — stale anchors, links and patterns never linger;
* nested properties are stored as JSON strings in `<key>_json` (`conditions_json`, `shifts_json`, `canonical_json`, `signature_json`, `centroid_json`, …).

Publishing happens after every READY batch, after `rebuild-graph` and after a migration; `reset` publishes an empty snapshot, which clears the mirror; `python -m ltir neo4j-sync` syncs on demand (regardless of `NEO4J_ENABLED`). SIG owns its six labels in that database: one workspace per database, and no other data under those labels. A Neo4j failure leaves the batch READY with `neo4j.status = failed` and the warning `graph persistence (Neo4j) failed`; `reset`, `rebuild-graph` and `migrate` report it and go on. `ltir/cypher/queries/transversal.cypher` approximates the traversal of [7.3](07_question_answering.md#73-transversal-traversal) for the Browser: the same walkable edges, one latent hop and one lattice hop, the best path per target, but no budgeted best-first search.

## 6.7 Configuration and guarantees

| Parameter | Default |
|---|---|
| `WORKSPACE_DIR` | `workspace` |
| `CONTRAST_MIN_OVERLAP`, `CONTRAST_MIN_SHIFT` | 0.5, 0.5 |
| `NEO4J_ENABLED`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `NEO4J_LOAD_BATCH_SIZE` | false, `bolt://localhost:7687`, `neo4j`, "", `sigv1`, 5000 |

Guarantees (tests: `test_structural.py`; `test_persistence.py`: write → reload → rebuild equality, idempotent re-ingest, graph consistency, rollback after a simulated failure, a failed rollback that blocks writes until recovered, representation mismatch, a batch killed after its journal append and recovered, a second writer process refused while the first keeps its queue, migration with backup, a batch that stays READY when a post-commit save fails, the Neo4j mirror equal to the snapshot — replaced properties, stale nodes and relationships deleted, an empty snapshot clearing it — against a fake driver):

* GENERALIZES = inverse(SPECIALIZES); no transitive SPECIALIZES edges; structural edges connect patterns of one dataset; RELATED_TO connects only anchors; the latent plane has at most `RELATED_TO_PEER_COUNT · N / 2` edges.
* `journal rows == vector rows == ConceptStore.next_chunk_id`; snapshot ids are identical after reload and after rebuild.
* `READY` implies that journals, state and snapshot are mutually consistent; no transaction starts while `pending.json` exists.

## 6.8 Deleting a dataset

`Engine.delete_dataset(dataset_id)` (web: `DELETE /api/datasets/{id}`, the card's *Delete* button; the id may also be the batch id of an upload that failed before it had a dataset id) removes one dataset from the knowledge base while the rest stays intact — the counterpart of `reset`, which removes everything. It is a writer operation under the engine lock, refused with `busy` while one of the dataset's batches is still running and with `unknown_dataset` when no batch carries that id. It is all or nothing:

```text
transaction (6.4): checkpoint (state, journal, snapshot) + marker pending.json → datasets/<id>/,
journal/blocks/<batch>.npz, the batch records (READY, FAILED and SKIPPED alike) and the web upload moved into the
checkpoint → ontology.forget: memberships recounted, orphan anchors dropped (5.11) → the journal rewritten without
the dataset's rows (row ids renumbered in journal order; next_chunk_id = rows) → state saved → snapshot rebuilt and
saved → committed graph and frame loaded → marker removed (the commit point) → checkpoint discarded (this deletes
the moved files)
then, as after a batch: readers switched → sphere export queued → Neo4j synced
```

Dataset-specific entities (patterns, memberships, blocks, covers, profile, source copy, records) always go; the shared entities — anchors and their links — follow the orphan rule of [5.11](05_latent_anchors.md#511-removing-patterns-the-orphan-rule). Structural edges need no treatment: they are derived from the remaining patterns when the snapshot is rebuilt, and the Neo4j publish deletes whatever the snapshot no longer holds ([6.6](#66-neo4j-mirror)). An exception before the commit point rolls everything back at once — journal, state, snapshot, and the moved files go back where they were. A crash before it leaves the marker: the next writer start (`Workspace.recover`) rolls back the same way, so the journal and the ontology can never be left disagreeing; `migrate` refuses while such a marker exists. After the commit point a crash only leaves the checkpoint folder behind, which the next checkpoint replaces. A deleted dataset can be uploaded again: its READY record is gone, so the upload is not `SKIPPED`.

Tests (`test_persistence.py::test_an_interrupted_deletion_is_undone`: an error and a simulated crash both leave the workspace exactly as before; `test_delete_dataset_removes_its_knowledge_and_orphan_anchors`): the dataset's patterns, folder, blocks and records disappear; every surviving anchor has a member or a prior link and every removed one had neither; journal rows, vector rows, `next_chunk_id` and the frame agree; the snapshot equals a fresh rebuild; a question still answers; the dataset can be re-ingested; deleting the last dataset empties the ontology and a fresh ingest cold-starts.

