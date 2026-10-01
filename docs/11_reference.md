# 11. Reference — glossary, identifiers, parameters, formula index

> **In one paragraph.** Lookup tables for the whole system: one name per concept, the format of every identifier, every tunable with its default and the chapter that explains it, an index of every formula with the code that computes it, and the versioned contracts that decide whether a workspace can be reused.

**Previous** [10. Verification](10_verification.md) · **Start** [Reading guide](README.md)

---

## 11.1 Glossary

One concept, one name per layer. Code and these docs use the first column, the UI the second, the vendored lac modules the third.

| Code / docs | UI | lac (vendored) | Meaning |
|---|---|---|---|
| pattern, insight (`Insight`, node `P-…`) | insight | chunk (a journal row) | a validated subgroup finding: scope + phenomenon + statistics ([3](03_insights.md)) |
| attractor, latent anchor (node `A-k`) | theme | concept | a living unit-norm centroid that recurring phenomena activate ([5](05_latent_anchors.md)) |
| scope, conditions | where | — | the conjunction of `attribute = value` selectors that defines the subgroup |
| cohort, extent | — | — | the set of rows a subgroup covers; selectors with the same extent are one cohort ([2.3](02_discovery.md#23-deduplication-before-validation)) |
| closed intent | scope | — | every `attribute = value` that holds on all of a cohort's rows; the pattern's conditions |
| target, metric | metric | — | the numeric column whose robust median shift (or correlation change) the pattern reports |
| robust z, shift | `±x sd` | — | `0.6745 · Δmedian / MAD`, signed, capped at 10: a shift in robust standard deviations |
| phenomenon | — | — | the signed statistical behaviour of a pattern: its material shifts and, when material, its correlation change |
| component | — | — | one `(label, signed coefficient)` term of the phenomenon, e.g. `("discount", +2.21)` ([4.2](04_representation.md#42-the-canonical-form)) |
| `insight_weight`, weight | evidence | weight (the scaled input magnitude) | statistical strength in `[WEIGHT_FLOOR, 1]` ([3.3](03_insights.md#33-insight-weight)) |
| ACTIVATES | membership | activation | pattern → anchor edge; `alignment` (cosine) and `strength` (alignment × weight) |
| RELATED_TO | theme link | RELATED_TO | mutual nearest-neighbour edge between anchors ([5.7](05_latent_anchors.md#57-links-between-anchors)) |
| seed | match | — | a pattern resolved directly from the question ([7.2](07_question_answering.md#72-seeds)) |
| transversal, `transversal_only` | via theme, other segment | — | reached through the latent plane; `transversal_only` = and sharing no condition with any seed |
| frame (`LatentFrame`) | — | — | the single space of unit insight vectors and centroids; no centering |
| batch | dataset card | batch | one processing run of one uploaded file ([9.1](09_operations.md#91-the-batch-lifecycle)) |
| workspace | — | — | the folder that holds one knowledge base (`WORKSPACE_DIR`) ([6.3](06_graph_and_storage.md#63-the-workspace-on-disk)) |

## 11.2 Identifiers

| Identifier | Format | Example | Defined in |
|---|---|---|---|
| dataset id | `ds-` + 12 hex of `sha256(file bytes ‖ bands ‖ categories)` | `ds-6e53eb7fb0f9` | [2.1](02_discovery.md#21-ingestion) |
| batch id | `B` + UTC `YYYYMMDDTHHMMSS` + `-` + 6 hex | `B20261001T112454-ff7fe6` | [9.1](09_operations.md#91-the-batch-lifecycle) |
| writer lock | `<workspace>.writer.lock` beside the workspace folder, holding the writer's pid | `workspace.writer.lock` | [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery) |
| pattern id | `P-` + 12 hex of `sha1(dataset id, sorted condition expressions)` | `P-bc4657a04746` | [3.1](03_insights.md#31-the-insight-record) |
| row-set identity | `row_hash`: 16 hex of `sha1(sorted covered row positions)` | `cda451bada571c5d` | [3.1](03_insights.md#31-the-insight-record) |
| attractor id | integer `k` from lac; graph node `A-k` | `A-1` | [5.2](05_latent_anchors.md#52-one-batch-through-the-ontology) |
| schema node ids | `D:<dataset>:<column>`, `M:<dataset>:<column>`, `DS:<dataset>`, `B:<batch>` | `M:ds-6e53eb7fb0f9:discount` | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| edge id | `<TYPE>:<source>-><target>` | `ACTIVATES:P-bc4657a04746->A-1` | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| journal row id | position of the vector in `journal/embeddings.mmap` | `1` | [6.3](06_graph_and_storage.md#63-the-workspace-on-disk) |
| representation fingerprint | 10 hex of `sha1(EmbeddingSpec fields)` | `3d08cee697` | [4.6](04_representation.md#46-representation-identity-and-versions) |
| citation key | `P` + 1-based rank in the evidence | `[P1]`, grouped `[P4, P7]` | [7.4](07_question_answering.md#74-the-evidence-object) |
| condition expression | `attribute=value` (no spaces, no quotes) | `category=phones` | [3.1](03_insights.md#31-the-insight-record) |
| EDA selector | pysubgroup's rendering, kept verbatim as provenance | `category=='phones' AND region=='US'` | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) |

Same bytes and options give the same dataset id; the same dataset and conditions give the same pattern id. That is what makes re-ingestion idempotent and lets the pattern id double as the duplicate guard.

## 11.3 Parameters

Every field of `ltir/config.py::Config` can be set from the environment by its upper-case name (or in `.env`, template `.env.sample`); explicit overrides in code win. Loading rules: [9.3](09_operations.md#93-configuration).

| Group | Parameter (default) | Explained in |
|---|---|---|
| paths | `WORKSPACE_DIR` (`workspace`), `MODEL_DIR` (`models`) | [6.3](06_graph_and_storage.md#63-the-workspace-on-disk), [4.4](04_representation.md#44-the-embedding-model) |
| ingestion | `MIN_ROWS` 50, `MAX_UPLOAD_MB` 200, `BIN_COLUMNS` "", `CATEGORICAL_COLUMNS` "" | [2.1](02_discovery.md#21-ingestion) |
| discovery | `COMPUTE_BUDGET` 5000, `VALIDATION_BUDGET` 50, `MIN_SEARCH_DIMENSIONS` 3, `EDA_RANDOM_SEED` 42, `REDUNDANCY_JACCARD` 0.88 | [2](02_discovery.md) |
| selection | `MIN_SUPPORT_ROWS` 30, `MIN_EFFECT_Z` 0.5, `MAX_P_ADJUSTED` 0.05, `MIN_STABILITY` 0.5, `MIN_EMM_SCORE` 0.08, `MIN_INSIGHT_WEIGHT` 0.2, `MAX_INSIGHTS_PER_BATCH` 200 | [3.2](03_insights.md#32-selection-rules) |
| weight | `WEIGHT_EFFECT_REF` 1.5, `WEIGHT_CONFIDENCE_REF` 6, `WEIGHT_EMM_REF` 0.08, `WEIGHT_EXPONENTS` (0.35, 0.25, 0.20, 0.10, 0.10), `WEIGHT_FLOOR` 0.05 | [3.3](03_insights.md#33-insight-weight) |
| canonical text | `MIN_COMPONENT_Z` 0.5, `EMM_COMPONENT_WEIGHT` 0.5 | [4.2](04_representation.md#42-the-canonical-form) |
| embedding | `EMBEDDING_MODEL` `Qwen/Qwen3-Embedding-0.6B`, `EMBEDDING_TRUNCATE_DIM` 384, `EMBEDDING_QUERY_INSTRUCTION` (retrieval task sentence), `EMBEDDING_DEVICE` auto, `EMBEDDING_BACKEND` sentence-transformers, `BLOCK_WEIGHTS` (0.45, 0.55, 1.0) | [4.3](04_representation.md#43-the-tripartite-vector), [4.4](04_representation.md#44-the-embedding-model) |
| extraction | `CONCEPTS_PER_CHUNK` 1, `DICTIONARY_K_MIN` 4, `DICTIONARY_K_STEP` 2, `MAX_CONCEPT_COUNT` 40, `RECONSTRUCTION_ERROR_TOLERANCE` 0.015, `DEAD_CONCEPT_PENALTY` 0.05, `MAX_DEAD_CONCEPT_RATIO` 0.25, `DICTIONARY_BATCH_SIZE` 256, `DICTIONARY_INPUT_SCALE` 10, `RANDOM_SEED` 42, `SOFT_MERGE_LOW` 0.85, `MIN_ACTIVATION_ALIGNMENT` 0.20 | [5.3](05_latent_anchors.md#53-extraction-omp-k-sweep-with-signed-repair) |
| assignment | `MIN_ASSIGN_THRESHOLD` 0.75 (MiniLM 0.55), `MAX_ASSIGN_THRESHOLD` 0.80, `ADAPTIVE_PERCENTILE` 85, `TOP_K_ASSIGN` 2, `MIXTURE_RATIO` 0.90, `CENTROID_ALPHA` 0.05, `ORPHAN_BUFFER_MIN_FACTOR` 3 | [5.4](05_latent_anchors.md#54-assignment-ema-and-orphans), [5.10](05_latent_anchors.md#510-calibration-per-embedder) |
| guards | `DENSITY_FLOOR` 0.25, `DENSITY_MULTIPLE` 3.0, `MAX_CENTROID_STEP` 0.10, `WARN_ORPHAN_RATE` 0.50, `WARN_MIN_EXTRACTION_YIELD` 0.10, `WARN_AVG_DEGREE` (1.0, 8.0) | [5.6](05_latent_anchors.md#56-stability-guards) |
| anchor links | `RELATED_TO_PEER_COUNT` 3, `RELATED_TO_MIN_WEIGHT` 0.30 | [5.7](05_latent_anchors.md#57-links-between-anchors) |
| structural plane | `CONTRAST_MIN_OVERLAP` 0.5, `CONTRAST_MIN_SHIFT` 0.5 | [6.1](06_graph_and_storage.md#61-the-structural-plane) |
| seeds | `SEED_TOP_K` 3, `SEED_MIN_SCORE` 0.25, `SEED_RELATIVE_MIN` 0.75 | [7.2](07_question_answering.md#72-seeds) |
| traversal | `ACTIVATION_THRESHOLD` 0.40, `RELATION_THRESHOLD` 0.40, `MAX_LATENT_HOPS` 1, `STRUCTURAL_HOPS` 1, `TRAVERSAL_MAX_DEPTH` 5, `MAX_RETRIEVED` 12, `STRUCTURAL_EDGE_DECAY` 0.85, `TRAVERSAL_STRUCTURAL_EDGES` `SPECIALIZES,GENERALIZES,CONTRASTS` | [7.3](07_question_answering.md#73-transversal-traversal) |
| evidence | `EVIDENCE_MAX_PATTERNS` 10 | [7.4](07_question_answering.md#74-the-evidence-object) |
| LLM | `LLM_BASE_URL` `http://localhost:11434/v1`, `LLM_MODEL` `gemma4`, `LLM_API_KEY` "", `LLM_PROVIDER_ORDER` "", `LLM_APP_TITLE` `SIG LTIR`, `LLM_TIMEOUT_S` 120, `LLM_TEMPERATURE` 0.1, `LLM_MAX_TOKENS` 1200, `LLM_REASONING_EFFORT` "" | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) |
| Neo4j | `NEO4J_ENABLED` false, `NEO4J_URI` `bolt://localhost:7687`, `NEO4J_USER` `neo4j`, `NEO4J_PASSWORD` "", `NEO4J_DATABASE` `sigv1`, `NEO4J_LOAD_BATCH_SIZE` 5000 | [6.6](06_graph_and_storage.md#66-neo4j-mirror) |
| web | `WEB_HOST` 127.0.0.1, `WEB_PORT` 8765, `SPHERE_EXPORT` true | [8](08_interface.md) |

Fixed constants that are not configurable but enter the mathematics: the EDA's 95 % mass and correlation limits, the 1.10 nesting factor, the size floor `n_min`, `ROBUST_Z_CAP` 10, `BOOTSTRAP_RESAMPLES` 20, the JS 0.15 / χ² `0.01 / #categoricals` confounder gate and the 1.5 hidden-shift limit ([2.2](02_discovery.md#22-the-eda-engine-in-five-steps)); `ENCODE_BATCH_SIZE` 16 ([4.4](04_representation.md#44-the-embedding-model)); `HEALTH_TTL_S` 15 ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)).

## 11.4 Formula index

Every quantity the system computes, where it is explained and which code computes it.

| Quantity | Explained in | Code |
|---|---|---|
| dataset id, pattern id, row hash | [2.1](02_discovery.md#21-ingestion), [3.1](03_insights.md#31-the-insight-record) | `ingestion.dataset_fingerprint`, `models.pattern_id`, `discovery.build_insights` |
| ε² grouping power | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `main_upd.step2_evaluate_macro_groupings` |
| robust scale (MAD with fallback), robust z | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `main_upd.calculate_mad`, `main_upd.robust_z_score` |
| SD score, EMM score, volume utility | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `main_upd.step4_evaluate_micro_slices` |
| bootstrap stability, confounder drivers | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `main_upd.step4b_deep_validation` |
| closed intent, near-duplicate Jaccard, `temp_index` | [2.3](02_discovery.md#23-deduplication-before-validation) | `discovery.closed_intent`, `prune_near_duplicates`, `run_discovery` |
| signed shift, EMM per pair, covariance pair, median test, Bonferroni | [2.4](02_discovery.md#24-what-the-adapter-adds) | `discovery.build_insights` |
| selection predicates | [3.2](03_insights.md#32-selection-rules) | `quality.select_insights` |
| weight factors and weighted geometric mean | [3.3](03_insights.md#33-insight-weight) | `quality.weight_factors`, `quality.insight_weight` |
| phenomenon components | [4.2](04_representation.md#42-the-canonical-form) | `canonical.canonicalize` |
| tripartite vector, cosine decomposition | [4.3](04_representation.md#43-the-tripartite-vector) | `encoder.InsightEncoder.encode`, `encode_query` |
| OMP K-sweep, signed repair | [5.3](05_latent_anchors.md#53-extraction-omp-k-sweep-with-signed-repair) | `ontology_engine.extract_attractors`, `repair_extraction` |
| adaptive threshold, assignment, EMA with inertia | [5.4](05_latent_anchors.md#54-assignment-ema-and-orphans) | `ontology_engine.compute_adaptive_threshold`, `assign_and_update`, `storage.update_concept_centroid` |
| hub threshold, damping, trust region | [5.6](05_latent_anchors.md#56-stability-guards) | `observability.density_threshold`, `LatentOntology._damping`, `_clamp_steps` |
| mutual kNN links | [5.7](05_latent_anchors.md#57-links-between-anchors) | `ontology_engine.calculate_knn_topology` |
| signature, label, dispersion, evidence mass | [5.8](05_latent_anchors.md#58-how-an-anchor-is-described) | `graph._attractor_nodes` |
| alignment, strength | [5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics) | `ontology._activation_records`, `graph._activation_edges` |
| batch metrics and warnings | [5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics) | `LatentOntology` (`BatchMetrics`), `observability.apply_health_warnings` |
| structural edges, contrast overlap, TARGETS weight | [6.1](06_graph_and_storage.md#61-the-structural-plane) | `structural.structural_edges`, `graph.build_snapshot` |
| query components, seed score | [7.1](07_question_answering.md#71-from-question-to-query), [7.2](07_question_answering.md#72-seeds) | `query.parse_query`, `query.score_pattern` |
| path score, node rank, scope overlap | [7.3](07_question_answering.md#73-transversal-traversal) | `traversal.traverse` |
| sphere projection | [8.4](08_interface.md#84-the-latent-sphere) | `sphere.sphere_figure` |
| recall@k, precision@k, MRR | [10.3](10_verification.md#103-hypothesis-benchmark) | `experiment.run_experiment` |

## 11.5 Versioned contracts

Two version strings in `ltir/models.py` and one fingerprint decide whether stored vectors and text can be compared with what the current code produces.

| Version | Value | Covers |
|---|---|---|
| `CANONICAL_VERSION` | `ltir-canon-4` | the canonical form: both text contracts, the closed-intent scope, the phrase grammar, component labels and coefficients (covariance insights cite only their correlation change) ([4.2](04_representation.md#42-the-canonical-form)) |
| `REPRESENTATION_VERSION` | `ltir-rep-3` | what is embedded and how it is composed: block layout, re-normalisation, the uncentred frame ([4.3](04_representation.md#43-the-tripartite-vector)) |
| fingerprint | `sha1` of `EmbeddingSpec` | both versions plus model id and revision, dimensions, compute dtype, truncation, query instruction, normalisation, block weights, EMM component weight and the three component settings (`MIN_COMPONENT_Z`, `MIN_EMM_SCORE`, `WEIGHT_EMM_REF`) ([4.6](04_representation.md#46-representation-identity-and-versions)) |
| `SNAPSHOT_VERSION` | 2 | the layout of `graph/snapshot.json` ([6.3](06_graph_and_storage.md#63-the-workspace-on-disk)) |

A workspace built under another fingerprint is refused (`representation_mismatch`) for ingest and for queries; `python -m ltir migrate --yes` rebuilds it from its stored sources and keeps the old copy ([6.5](06_graph_and_storage.md#65-versions-and-migration)). Renderings — documents, headlines, labels, the prompt, the UI — are derived from the records and can change without a version bump.
