# 11. Reference — glossary, identifiers, parameters, formula index

> **In one paragraph.** Lookup tables for the whole system: one name per concept, the format of every identifier, every tunable with its default and the chapter that explains it, an index of every formula with the code that computes it, and the versioned contracts that decide whether a workspace can be reused.

**Previous** [10. Verification](10_verification.md) · **Next** [12. Architecture](12_architecture.md) · **Start** [Reading guide](README.md)

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
| admission | — | — | the graph's policy over valid insights: the R4 weight floor and the R7 batch budget (`batch.admit_insights`, [3.2](03_insights.md#32-selection-rules)) |
| ACTIVATES | membership | activation | pattern → anchor edge; `alignment` (cosine) and `strength` (alignment × weight) |
| RELATED_TO | theme link | RELATED_TO | mutual nearest-neighbour edge between anchors ([5.7](05_latent_anchors.md#57-links-between-anchors)) |
| CO_OCCURS | co-occurring themes | — | link between anchors that share member insights ([5.7](05_latent_anchors.md#57-links-between-anchors)) |
| co-membership | — | — | a membership the snapshot compiles with the assignment rule (`source: co_membership`, [5.7](05_latent_anchors.md#57-links-between-anchors)) |
| seed | match | — | a pattern resolved directly from the question ([7.2](07_question_answering.md#72-seeds)) |
| transversal, `transversal_only` | via theme, other segment | — | reached through the latent plane; `transversal_only` = and sharing no condition with any seed |
| frame (`LatentFrame`) | — | — | the single space of unit insight vectors and centroids; no centering |
| committed state (`CommittedState`) | — | — | the graph, frame and literal catalog of one commit, replaced whole by the next ([9.1](09_operations.md#91-the-batch-lifecycle)) |
| evidence payload (`EvidencePayload`) | — | — | the evidence of one question as the graph service hands it to the narrator: prompt, summary, citation manifest, view, metrics ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| citation manifest | Sources | — | `P#` → pattern id, expression, dataset id, file name, batch id: what a citation is checked against ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| batch | dataset card | batch | one processing run of one uploaded file ([9.1](09_operations.md#91-the-batch-lifecycle)) |
| workspace | — | — | the folder that holds one knowledge base (`WORKSPACE_DIR`) ([6.3](06_graph_and_storage.md#63-the-workspace-on-disk)) |
| degraded start | — | — | a start on a workspace the engine cannot serve: writes and questions refused with `workspace_degraded` until a reset ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)) |
| bounded context | — | — | a package that owns one part of the model and imports only what [12.2](12_architecture.md#122-dependency-direction) allows ([12.1](12_architecture.md#121-bounded-contexts)) |
| shared kernel (`insight_contracts`) | — | — | the standard-library package all others speak: the insight, the graph vocabulary, how an insight reads, the evidence payload |
| graph service (`insight_graph_service`) | — | — | the service that owns the workspace, the batch lifecycle, the dual graph, the Neo4j mirror and the evidence for a question ([9](09_operations.md)) |
| narrator (`evidence_narrator_service`) | chat | — | the service that verbalises an evidence payload through the LLM and checks every citation ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)) |
| broker (`llm_model_broker`) | — | — | the LLM model broker: one OpenAI-compatible endpoint that holds the upstream model, its key and the provider routing ([12.4](12_architecture.md#124-services-and-contracts)) |
| console (`sig_web_console`) | — | — | the static web UI and its nginx edge, one origin for both APIs ([8](08_interface.md)) |

## 11.2 Identifiers

| Identifier | Format | Example | Defined in |
|---|---|---|---|
| dataset id | `ds-` + 12 hex of `sha256(file bytes ‖ bands ‖ categories)` | `ds-6e53eb7fb0f9` | [2.1](02_discovery.md#21-ingestion) |
| batch id | `B` + UTC `YYYYMMDDTHHMMSS` + `-` + 6 hex | `B20261001T112454-ff7fe6` | [9.1](09_operations.md#91-the-batch-lifecycle) |
| writer lock | `<workspace>.writer.lock` beside the workspace folder, holding the writer's pid | `workspace.writer.lock` | [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery) |
| pattern id | `P-` + 12 hex of `sha1(dataset id, condition expressions of the conditions sorted by (attribute, value))` | `P-bc4657a04746` | [3.1](03_insights.md#31-the-insight-record) |
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

The graph service's settings tree (`insight_graph_service/core/settings.py`, loading rules in [9.3](09_operations.md#93-configuration)) has one section per package, one shared `PhenomenonThresholds` and the service's own fields. Every field is set from the process environment by its upper-case name; host runs copy the repository's `.env` (template `.env.sample`) into the environment first. Explicit overrides to `load_settings` win. The narrator and the broker read their own variables, listed at the end.

| Section, group | Parameter (default) | Explained in |
|---|---|---|
| `MinerConfig` (`subgroup_miner/config.py`) — ingestion | `MIN_ROWS` 50, `BIN_COLUMNS` "", `CATEGORICAL_COLUMNS` "" | [2.1](02_discovery.md#21-ingestion) |
| `MinerConfig` — discovery | `COMPUTE_BUDGET` 5000, `VALIDATION_BUDGET` 50, `MIN_SEARCH_DIMENSIONS` 3, `EDA_RANDOM_SEED` 42, `REDUNDANCY_JACCARD` 0.88 | [2](02_discovery.md) |
| `MinerConfig` — validity rules | `MIN_SUPPORT_ROWS` 30, `MIN_EFFECT_Z` 0.5, `MAX_P_ADJUSTED` 0.05, `MIN_STABILITY` 0.5 | [3.2](03_insights.md#32-selection-rules) |
| `MinerConfig` — weight | `WEIGHT_EFFECT_REF` 1.5, `WEIGHT_CONFIDENCE_REF` 6, `WEIGHT_EXPONENTS` (0.35, 0.25, 0.20, 0.10, 0.10), `WEIGHT_FLOOR` 0.05 | [3.3](03_insights.md#33-insight-weight) |
| `MinerConfig` — structural lattice | `CONTRAST_MIN_OVERLAP` 0.5, `CONTRAST_MIN_SHIFT` 0.5 | [6.1](06_graph_and_storage.md#61-the-structural-plane) |
| `TopologyConfig` (`attractor_topology/config.py`) — representation | `MODEL_DIR` (`models`), `EMBEDDING_DEVICE` auto, `BLOCK_WEIGHTS` (0.45, 0.55, 1.0), `EMM_COMPONENT_WEIGHT` 0.5 | [4.3](04_representation.md#43-the-tripartite-vector), [4.4](04_representation.md#44-the-embedding-model) |
| `TopologyConfig` — extraction | `CONCEPTS_PER_CHUNK` 1, `DICTIONARY_K_MIN` 4, `DICTIONARY_K_STEP` 2, `MAX_CONCEPT_COUNT` 40, `RECONSTRUCTION_ERROR_TOLERANCE` 0.015, `DEAD_CONCEPT_PENALTY` 0.05, `MAX_DEAD_CONCEPT_RATIO` 0.25, `DICTIONARY_BATCH_SIZE` 256, `DICTIONARY_INPUT_SCALE` 10, `RANDOM_SEED` 42, `SOFT_MERGE_LOW` 0.85, `MIN_ACTIVATION_ALIGNMENT` 0.20 | [5.3](05_latent_anchors.md#53-extraction-omp-k-sweep-with-signed-repair) |
| `TopologyConfig` — assignment | `MIN_ASSIGN_THRESHOLD` 0.75, `MAX_ASSIGN_THRESHOLD` 0.80, `ADAPTIVE_PERCENTILE` 85, `TOP_K_ASSIGN` 2, `MIXTURE_RATIO` 0.80, `CENTROID_ALPHA` 0.05, `ORPHAN_BUFFER_MIN_FACTOR` 3 | [5.4](05_latent_anchors.md#54-assignment-ema-and-orphans), [5.10](05_latent_anchors.md#510-calibration-per-embedder) |
| `TopologyConfig` — guards | `DENSITY_FLOOR` 0.25, `DENSITY_MULTIPLE` 3.0, `MAX_CENTROID_STEP` 0.10, `WARN_ORPHAN_RATE` 0.50, `WARN_MIN_EXTRACTION_YIELD` 0.10, `WARN_AVG_DEGREE` (1.0, 8.0) | [5.6](05_latent_anchors.md#56-stability-guards) |
| `TopologyConfig` — anchor links | `RELATED_TO_PEER_COUNT` 3, `RELATED_TO_MIN_WEIGHT` 0.30 | [5.7](05_latent_anchors.md#57-links-between-anchors) |
| `QueryConfig` (`graph_query_engine/config.py`) — literal grounding | `GROUNDING_MIN_COSINE` 0.30 (centred cosine; embedder-specific); fixed in `graph_query_engine/question.py`: `CHAR_MIN` 0.30, `CHAR_MAX_LEN_DIFF` 3, `MARGIN_MIN` 0.15, `MARGIN_K` 5, `LOWE_MAX` 0.85, `MAX_SPAN` 3, `ACRONYM_MAX_LEN` 4 | [7.1.1](07_question_answering.md#711-literal-grounding) |
| `QueryConfig` — seeds | `SEED_TOP_K` 3, `SEED_MIN_SCORE` 0.25, `SEED_RELATIVE_MIN` 0.75 | [7.2](07_question_answering.md#72-seeds) |
| `QueryConfig` — traversal | `MAX_LATENT_HOPS` 1, `STRUCTURAL_HOPS` 1, `TRAVERSAL_MAX_DEPTH` 5, `MAX_RETRIEVED` 12, `STRUCTURAL_EDGE_DECAY` 0.85, `TRAVERSAL_STRUCTURAL_EDGES` `SPECIALIZES,GENERALIZES,CONTRASTS` | [7.3](07_question_answering.md#73-transversal-traversal) |
| `QueryConfig` — evidence | `EVIDENCE_MAX_PATTERNS` 10 | [7.4](07_question_answering.md#74-the-evidence-object) |
| `PhenomenonThresholds` (`insight_contracts/insight.py`; one instance shared by the three sections) | `MIN_EMM_SCORE` 0.08, `WEIGHT_EMM_REF` 0.08, `MIN_COMPONENT_Z` 0.5 | [3.2](03_insights.md#32-selection-rules), [3.3](03_insights.md#33-insight-weight), [4.2](04_representation.md#42-the-canonical-form) |
| `Settings` (`insight_graph_service/core/settings.py`) — paths, upload | `WORKSPACE_DIR` (`workspace`), `MAX_UPLOAD_MB` 200 | [6.3](06_graph_and_storage.md#63-the-workspace-on-disk), [8.1](08_interface.md#81-http-api) |
| `Settings` — admission | `MIN_INSIGHT_WEIGHT` 0.2, `MAX_INSIGHTS_PER_BATCH` 200 | [3.2](03_insights.md#32-selection-rules) |
| `Settings` — Neo4j | `NEO4J_ENABLED` false, `NEO4J_URI` `bolt://localhost:7687`, `NEO4J_USER` `neo4j`, `NEO4J_PASSWORD` "", `NEO4J_DATABASE` `sigv1`, `NEO4J_LOAD_BATCH_SIZE` 5000 | [6.6](06_graph_and_storage.md#66-neo4j-mirror) |
| `Settings` — web | `WEB_HOST` 127.0.0.1, `WEB_PORT` 8765 | [8.6](08_interface.md#86-configuration-failure-modes-and-tests) |
| `NarratorSettings` (`evidence_narrator_service/settings.py`) — graph service | `INSIGHT_GRAPH_URL` `http://127.0.0.1:8765`, `INSIGHT_GRAPH_TIMEOUT_S` 300 | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) |
| `NarratorSettings` — LLM | `LLM_BASE_URL` `http://127.0.0.1:8080/v1` (the broker), `LLM_TIMEOUT_S` 120, `LLM_TEMPERATURE` 0.1, `LLM_MAX_TOKENS` 1200 | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) |
| broker (`llm_model_broker/server.py`, `Settings.from_env`) | `GEMMA_BASE_URL` (required), `GEMMA_MODEL_NAME` (required), `GEMMA_CHAT_ENDPOINT` `/chat/completions`, `GEMMA_API_KEY` "", `GEMMA_PROVIDER` "" (OpenRouter provider slugs, in order, no fallbacks), `GEMMA_CONNECT_TIMEOUT_SEC` 5, `GEMMA_READ_TIMEOUT_SEC` 120 | [12.4](12_architecture.md#124-services-and-contracts) |
| entry points (not settings fields) | `NARRATOR_HOST`, `NARRATOR_PORT` (127.0.0.1, 8766) for `python -m evidence_narrator_service`; `BROKER_HOST`, `BROKER_PORT` (127.0.0.1, 8080) for `python -m llm_model_broker`; compose's `SIG_WEB_PORT` 8765, `SIG_NEO4J_HTTP_PORT` 17474, `SIG_NEO4J_BOLT_PORT` 17687, `SIG_NEO4J_PASSWORD` `sig-local-password` | [9.2](09_operations.md#92-command-line), [12.5](12_architecture.md#125-containers-and-start-order) |

**Fixed constants** that are not configurable but enter the mathematics or the behaviour:
- **In the EDA** ([2.2](02_discovery.md#22-the-eda-engine-in-five-steps)): the 95 % mass and correlation limits, the 1.10 nesting factor, the size floor `n_min`, `ROBUST_Z_CAP` 10, `BOOTSTRAP_RESAMPLES` 20, the JS 0.15 / χ² `0.01 / #categoricals` confounder gate and the 1.5 hidden-shift limit.
- **In `attractor_topology/encoder.py`** ([4.4](04_representation.md#44-the-embedding-model)): the embedding model `QWEN_MODEL` (`Qwen/Qwen3-Embedding-0.6B`) at `QWEN_REVISION`, `TRUNCATE_DIM` 384, the query instruction (`QUERY_TASK` in `QUERY_PROMPT`) and `ENCODE_BATCH_SIZE` 16.
- **In `evidence_narrator_service/llm_client.py`** ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)): `HEALTH_TTL_S` 15.

## 11.4 Formula index

Every quantity the system computes, where it is explained and which code computes it.

| Quantity | Explained in | Code |
|---|---|---|
| dataset id, pattern id, row hash | [2.1](02_discovery.md#21-ingestion), [3.1](03_insights.md#31-the-insight-record) | `subgroup_miner/ingestion.py::dataset_fingerprint`, `insight_contracts/insight.py::pattern_id`, `subgroup_miner/discovery.py::build_insights` |
| ε² grouping power | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `subgroup_miner/vendor/eda/main_upd.py::step2_evaluate_macro_groupings` |
| robust scale (MAD with fallback), robust z | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `main_upd.py`: `calculate_mad`, `robust_z_score` |
| SD score, EMM score, volume utility | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `main_upd.py::step4_evaluate_micro_slices` |
| bootstrap stability, confounder drivers | [2.2](02_discovery.md#22-the-eda-engine-in-five-steps) | `main_upd.py::step4b_deep_validation` |
| closed intent, near-duplicate Jaccard, `temp_index` | [2.3](02_discovery.md#23-deduplication-before-validation) | `subgroup_miner/discovery.py`: `closed_intent`, `prune_near_duplicates`, `run_discovery` |
| signed shift, EMM per pair, covariance pair, median test, Bonferroni | [2.4](02_discovery.md#24-what-the-adapter-adds) | `subgroup_miner/discovery.py`: `run_discovery` (EMM per pair), `_covariance_pair`, `_median_test`, `build_insights` (signed shift, Bonferroni) |
| selection predicates | [3.2](03_insights.md#32-selection-rules) | `subgroup_miner/selection.py::select_insights` |
| admission (R4 weight floor, R7 budget) | [3.2](03_insights.md#32-selection-rules) | `insight_graph_service/core/batch.py::admit_insights` |
| weight factors and weighted geometric mean | [3.3](03_insights.md#33-insight-weight) | `subgroup_miner/selection.py`: `weight_factors`, `insight_weight` |
| phenomenon components | [4.2](04_representation.md#42-the-canonical-form) | `attractor_topology/canonical.py::canonicalize` |
| tripartite vector, cosine decomposition | [4.3](04_representation.md#43-the-tripartite-vector) | `attractor_topology/encoder.py`: `InsightEncoder.encode`, `InsightEncoder.encode_query` |
| OMP K-sweep, signed repair | [5.3](05_latent_anchors.md#53-extraction-omp-k-sweep-with-signed-repair) | `attractor_topology/vendor/lac/ontology_engine.py`: `extract_attractors`, `repair_extraction` |
| adaptive threshold, assignment, EMA with inertia | [5.4](05_latent_anchors.md#54-assignment-ema-and-orphans) | `ontology_engine.py`: `compute_adaptive_threshold`, `assign_and_update`; `attractor_topology/vendor/lac/storage.py::ConceptStore.update_concept_centroid` |
| hub threshold, damping, trust region | [5.6](05_latent_anchors.md#56-stability-guards) | `attractor_topology/vendor/lac/observability.py::density_threshold`; `attractor_topology/ontology.py`: `LatentOntology._damping`, `LatentOntology._clamp_steps` |
| mutual kNN links | [5.7](05_latent_anchors.md#57-links-between-anchors) | `ontology_engine.py::calculate_knn_topology` |
| co-memberships, co-occurrence links | [5.7](05_latent_anchors.md#57-links-between-anchors) | `insight_graph_service/core/snapshot.py`: `_activation_edges`, `co_occurrence_edges` |
| signature, label, dispersion, evidence mass | [5.8](05_latent_anchors.md#58-how-an-anchor-is-described) | `insight_graph_service/core/snapshot.py`: `attractor_signature`, `_attractor_nodes` |
| alignment, strength | [5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics) | `attractor_topology/ontology.py::LatentOntology._activation_records`, `insight_graph_service/core/snapshot.py::_activation_edges` |
| batch metrics and warnings | [5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics) | `attractor_topology/ontology.py::LatentOntology` (`BatchMetrics`), `observability.py::apply_health_warnings` |
| structural edges, contrast overlap, TARGETS weight | [6.1](06_graph_and_storage.md#61-the-structural-plane), [6.2](06_graph_and_storage.md#62-the-graph-schema) | `subgroup_miner/lattice.py::structural_edges`, `insight_graph_service/core/snapshot.py::_schema_plane` (TARGETS) |
| query components, seed score | [7.1](07_question_answering.md#71-from-question-to-query), [7.2](07_question_answering.md#72-seeds) | `graph_query_engine/question.py::parse_query`, `graph_query_engine/seeds.py::score_pattern` |
| literal grounding: centred cosine, local margin, Lowe's ratio, char_wb TF-IDF, wildcard scope | [7.1.1](07_question_answering.md#711-literal-grounding) | `graph_query_engine/question.py`: `build_catalog`, `_ground`, `_bind`; `seeds.py::score_pattern`; `insight_graph_service/core/engine.py::Engine._prepare` (the cached vectors) |
| path score, node rank, scope overlap | [7.3](07_question_answering.md#73-transversal-traversal) | `graph_query_engine/traversal.py::traverse` |
| structural closure, naive text top-k | [7.6](07_question_answering.md#76-baselines) | `graph_query_engine/search.py::compute_baselines`, `traversal.py`: `structural_closure`, `naive_nearest` |
| citation check, provenance footer | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) | `evidence_narrator_service/narration.py`: `check_citations`, `answer` |
| sphere projection | [8.4](08_interface.md#84-the-latent-sphere) | `insight_graph_service/server/views.py::project_to_sphere` |
| recall@k, precision@k, MRR | [10.3](10_verification.md#103-hypothesis-benchmark) | `scripts/experiment.py`: `run_experiment`, `_score` |

## 11.5 Versioned contracts

Two version strings in `attractor_topology/models.py` and one fingerprint decide whether stored vectors and text can be compared with what the current code produces; `SNAPSHOT_VERSION` (`insight_contracts/graph.py`) fixes the snapshot's layout.

| Version | Value | Covers |
|---|---|---|
| `CANONICAL_VERSION` | `ltir-canon-4` | the canonical form: both text contracts, the closed-intent scope, the phrase grammar, component labels and coefficients (covariance insights cite only their correlation change) ([4.2](04_representation.md#42-the-canonical-form)) |
| `REPRESENTATION_VERSION` | `ltir-rep-3` | what is embedded and how it is composed: block layout, re-normalisation, the uncentred frame ([4.3](04_representation.md#43-the-tripartite-vector)) |
| fingerprint | `sha1` of `EmbeddingSpec` | both versions plus model id and revision, dimensions, storage and compute dtype, truncation, query instruction, normalisation, block weights, EMM component weight and the three component settings (`MIN_COMPONENT_Z`, `MIN_EMM_SCORE`, `WEIGHT_EMM_REF`) ([4.6](04_representation.md#46-representation-identity-and-versions)) |
| `SNAPSHOT_VERSION` | 3 | the layout of `graph/snapshot.json` ([6.3](06_graph_and_storage.md#63-the-workspace-on-disk)) |

How the engine treats a stored workspace ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)):
- **Another version.** Journals written under another `CANONICAL_VERSION` or `REPRESENTATION_VERSION` give a degraded start ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).
- **Another fingerprint.** Under the same versions, a fingerprint other than the workspace's — another block weight, threshold or model revision — fails a batch at EMBEDDING and refuses a question with `representation_mismatch` (HTTP 409).
- **No migration.** In both cases `POST /api/reset` starts the workspace over and the data is ingested again (proof-of-concept scope).
- **The snapshot is derived data.** A snapshot of another `SNAPSHOT_VERSION` is rebuilt from the journals when the engine starts.
- **Renderings** — documents, headlines, labels, the prompt, the UI — are derived from the records and can change without a version bump.
- **The payload has no version.** `EvidencePayload` is checked field by field instead: `from_dict` is strict, so a graph service and a narrator that disagree fail with a 502 rather than a silent default.
