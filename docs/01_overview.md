# 1. Overview — what SIG is and how the pieces fit

> **In one paragraph.** SIG (Statistical Insight Graph) turns a tabular file into statistically validated subgroup findings ("insights"), represents each one as text and as a vector, lets recurring phenomena self-organise into *latent anchors*, and stores everything as a dual-layer graph. Questions in natural language are answered by walking that graph — across the exact subgroup lattice **and** across the anchors — and a language model only verbalises the evidence that the walk returned, with citations that are checked.

**Code** the whole repository: four libraries and four services ([1.6](#16-module-map)) · **Tests** `tests/` ([10.1](10_verification.md#101-test-map)) · **Next** [2. Discovery](02_discovery.md)

---

## 1.1 The problem and the hypothesis

A table holds many subgroups that behave unusually: in the demo retail data, *phones sold in the US* carry far higher discounts and lower margins than the rest. A purely structural index of subgroups (a lattice of `attribute = value` conjunctions) relates `phones ∧ US` to its own refinements (`phones ∧ US ∧ online`) and siblings (`phones ∧ EU`). It does **not** relate it to `tablets ∧ APAC ∧ retail`, which shows the *same mechanism* — discount up, margin down — in a part of the data that shares no condition with it. In the demo graph those analogues sit 2–4 lattice hops away, mixed in with about twenty unrelated subgroups. Plain text retrieval over descriptions of the subgroups ranks by wording and returns the near-identical refinements first.

SIG makes one hypothesis testable:

> Structurally different subgroups that exhibit related statistical behaviour can be connected through latent attractor concepts. This enables *transversal* retrieval that purely structural graph traversal or naive nearest-neighbour text retrieval does not reliably achieve.

The benchmark in [10.3](10_verification.md#103-hypothesis-benchmark) measures exactly this on data with planted mechanisms: transversal retrieval reaches MRR 0.581 / recall@3 0.521, against at most 0.321 / 0.111 for the structural and text baselines.

## 1.2 The pipeline at a glance

| # | Stage | What happens | Hands on | Chapter |
|---|---|---|---|---|
| 1 | Ingestion | a CSV/TSV/Parquet file is validated, number-coded columns optionally become categories, numeric columns optionally become quantile bands; a content-addressed dataset id is assigned | `DataFrame` + `dataset_id` | [2](02_discovery.md) |
| 2 | Discovery | the vendored automatic-EDA engine screens 2- and 3-attribute subgroups for robust median shifts and correlation changes; duplicate cohorts are merged before the expensive bootstrap validation | validated candidates | [2](02_discovery.md) |
| 3 | Insights | candidates become typed `Insight` records; a fixed rule set keeps the significant, stable shifts and the material correlation changes; each gets an `insight_weight` in [0.05, 1] | kept insights | [3](03_insights.md) |
| 4 | Representation | each insight gets a canonical text form and a 1152-d vector composed from three embedded blocks: scope, target and a *signed* phenomenon | unit vectors | [4](04_representation.md) |
| 5 | Latent anchors | vectors stream into a self-organising dictionary of centroids (lac): assignment, orphan buffer, OMP extraction, soft merge, mutual-kNN links | anchors + memberships | [5](05_latent_anchors.md) |
| 6 | Graph & storage | an exact structural lattice is derived from the scopes; journals, ontology state and a graph snapshot are committed atomically (optional Neo4j mirror) | dual-layer graph | [6](06_graph_and_storage.md) |
| 7 | Question answering | a question becomes seed insights; a budgeted best-first walk follows lattice edges and anchors; an evidence object is built; the LLM answers from it, or a deterministic summary does | cited answer | [7](07_question_answering.md) |
| 8 | Interface | the web console shows the graph, a 3D latent sphere and an insight table as three views of one knowledge base under one legend, a details drawer and the chat (Ukrainian answers around untouched data literals), with the answer's path highlighted; datasets can be removed again | — | [8](08_interface.md) |

Stages 1–6 run once per uploaded file as one atomic *batch* ([9. Operations](09_operations.md)); stage 7 runs per question.

```text
 file ──► ingestion ──► discovery (vendored EDA) ──► insights: selection + weight
                                                          │
                              canonical text ──► tripartite vector (scope ; target ; signed phenomenon)
                                    │                         │
   structural plane                 │                         ▼
   SPECIALIZES / GENERALIZES /      │           latent plane: anchors (vendored lac)
   SIBLING / CONTRASTS  ◄───────────┘           ACTIVATES (insight → anchor), RELATED_TO / CO_OCCURS (anchor ↔ anchor)
                    │                                         │
                    └──────► graph snapshot + journals + state (+ Neo4j mirror) ◄┘
                                          │
 question ──► seeds ──► transversal walk ──► evidence ──► LLM or evidence-only summary ──► cited answer
                                          │
                              web UI: graph · sphere · table · chat, path highlighted
```

## 1.3 The two planes

| | Structural plane | Latent phenomenon plane |
|---|---|---|
| Nodes | `Pattern` (one validated insight); `Dimension`, `Metric` as schema anchors | `Attractor` = latent anchor = "theme" in the UI: a living unit-norm centroid |
| Edges | SPECIALIZES, GENERALIZES, SIBLING, CONTRASTS — exact, derived only from the scope conditions and signed shifts | RELATED_TO — mutual nearest neighbours among centroids; CO_OCCURS — anchors that share member insights |
| Bridge | ACTIVATES: Pattern → Attractor, weighted by cosine alignment (strength = alignment × insight weight); a *weak* membership (rerouted, or now aligned below `MIN_ACTIVATION_ALIGNMENT`) counts for coverage but is not walked | |
| Schema and provenance | `Dimension`, `Metric`, `Dataset`, `Batch` nodes; HAS_SCOPE, TARGETS (Pattern → schema), DISCOVERED_IN, OF_DATASET (provenance) — never walked by retrieval | |
| Never | embeddings never create structural edges | scope predicates never create latent edges |

The structural plane answers "which subgroups refine or contradict which"; the latent plane answers "which subgroups behave alike, wherever they are". Retrieval combines both ([7.3](07_question_answering.md#73-transversal-traversal)).

## 1.4 The running example

Every chapter follows the same object through its stage. The demo dataset (`retail_synthetic.csv`, 5,000 orders, generated by `insight_graph_service/core/demo.py` with planted mechanisms and ingested by `POST /api/demo`, [10.2](10_verification.md#102-the-synthetic-demo-dataset)) contains this insight:

| | |
|---|---|
| pattern id | `P-bc4657a04746` |
| scope | `category = phones ∧ region = US` (438 rows, 8.8 %) |
| phenomenon | discount +2.21 sd (median 19.19 vs 10.74), margin −1.10 sd (median 14.75 vs 19.45) |
| weight | 0.843 |
| anchor | `A-1 "discount ↑ · margin ↓"` (alignment 0.98), shared with 8 other insights over 9 distinct scopes |
| question | *"Why is margin lower for phones in the US?"* → seed `P-bc4657a04746`, then six scope-disjoint tablet insights through `A-1` |

## 1.5 What is reused and what is new

| Part | Origin | In this repository |
|---|---|---|
| Statistical discovery (profiling, macro screen, search space, robust shifts, EMM correlation divergence, volume utility, bootstrap, confounders) | the eda project: `scripts/main_upd.py` ([provenance](../subgroup_miner/vendor/PROVENANCE.md)) | vendored as `subgroup_miner/vendor/eda/main_upd.py`; called step by step from `subgroup_miner/discovery.py` |
| Dynamic ontology (concept store, EMA with inertia, adaptive threshold, orphan buffer, OMP K-sweep, soft merge, mutual kNN, metrics); the chunk journal; the projection of lac's prosphera sphere | the lac project: `v2_orchestrator/`, `v1_single_pass/visualisation/` ([provenance](../attractor_topology/vendor/PROVENANCE.md)) | the ontology vendored as `attractor_topology/vendor/lac/`, driven by `attractor_topology/ontology.py`; the journal vendored as `insight_graph_service/core/chunk_journal.py`; the projection recomputed in `insight_graph_service/server/views.py` |
| Everything else | new | adapter, insight model, selection, weight and admission, canonical text, encoder, structural lattice, graph compiler, persistence, retrieval, evidence, narrator and citation check, LLM model broker, web console, model provisioning, tests |

Every divergence of the vendored code from its origin is listed in its `PROVENANCE.md`: [`subgroup_miner/vendor/PROVENANCE.md`](../subgroup_miner/vendor/PROVENANCE.md), [`attractor_topology/vendor/PROVENANCE.md`](../attractor_topology/vendor/PROVENANCE.md) and [`insight_graph_service/core/PROVENANCE.md`](../insight_graph_service/core/PROVENANCE.md). The embedding model lives in `models/` (provisioned and verified by `python -m insight_graph_service.core.model_store`), the data in `data/` (`housing.csv` is a local copy); both folders are local and git-ignored. The demo dataset is generated in code (`insight_graph_service/core/demo.py`). At run time the repository needs nothing outside itself (`tests/test_self_contained.py`).

## 1.6 Module map

The repository is split into bounded contexts: four libraries, which import only the shared kernel, and the services that compose them ([12](12_architecture.md) has the import rule, the services, their contracts and how to use a library on its own).

| Package | Module | Role | Chapter |
|---|---|---|---|
| `insight_contracts/` (shared kernel, standard library only) | `insight.py` | `Condition`, `Shift`, `Insight`, `Rejection`, `pattern_id`, `PhenomenonThresholds` (what counts as part of a phenomenon) | [3](03_insights.md) |
| | `graph.py` | edge types and planes, node ids, `GraphNode` / `GraphEdge`, `SNAPSHOT_VERSION` and the snapshot schema | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| | `text.py` | how an insight reads: scope, shift, correlation and validation phrases, number formatting, headlines | [4.2](04_representation.md#42-the-canonical-form) |
| | `payload.py` | `EvidencePayload`, the contract between the graph service and the narrator | [12.4](12_architecture.md#124-services-and-contracts) |
| `subgroup_miner/` | `config.py` | `MinerConfig` | [2.6](02_discovery.md#26-configuration), [3.5](03_insights.md#35-configuration) |
| | `ingestion.py` | file validation, categories, bands, dataset id | [2.1](02_discovery.md#21-ingestion) |
| | `discovery.py`, `vendor/eda/` | EDA adapter: closed intents, deduplication before validation, typed insights | [2](02_discovery.md) |
| | `selection.py` | validity rules R1–R3 and `insight_weight` | [3](03_insights.md) |
| | `lattice.py` | the exact lattice edges | [6.1](06_graph_and_storage.md#61-the-structural-plane) |
| | `describe.py` | insights as numbered lines of LLM context | [12.3](12_architecture.md#123-using-a-library-on-its-own) |
| `attractor_topology/` | `config.py` | `TopologyConfig` | [4.7](04_representation.md#47-configuration), [5.12](05_latent_anchors.md#512-configuration) |
| | `canonical.py` | `canonicalize`: the embedding inputs and the readable sections | [4.1](04_representation.md#41-two-text-contracts) |
| | `encoder.py` | the Qwen3 embedder, `InsightEncoder` | [4](04_representation.md) |
| | `models.py` | `CanonicalInsight`, `EmbeddingSpec`, `CANONICAL_VERSION`, `REPRESENTATION_VERSION` | [4.6](04_representation.md#46-representation-identity-and-versions) |
| | `ontology.py`, `vendor/lac/` | `LatentOntology` over lac's lifecycle, activation records | [5](05_latent_anchors.md) |
| `graph_query_engine/` | `graph.py` | `DualGraph` (the read model of a snapshot), `LatentFrame` | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| | `question.py` | question parsing with multilingual literal grounding | [7.1](07_question_answering.md#71-from-question-to-query) |
| | `seeds.py`, `traversal.py` | seed scoring; the transversal walk, structural closure | [7.2](07_question_answering.md#72-seeds), [7.3](07_question_answering.md#73-transversal-traversal) |
| | `evidence.py` | the evidence object, the LLM prompt and the evidence-only summary | [7.4](07_question_answering.md#74-the-evidence-object) |
| | `search.py`, `ports.py`, `config.py` | one question end to end over a `CommittedState`; the embedder shape it needs; `QueryConfig` | [7](07_question_answering.md) |
| `insight_graph_service/core/` | `settings.py` | `Settings`: every package config built from one environment | [9.3](09_operations.md#93-configuration), [11.3](11_reference.md#113-parameters) |
| | `engine.py`, `batch.py` | `Engine`: batch lifecycle, admission, deletion, reset, recovery, the committed read state, the evidence for a question | [9](09_operations.md) |
| | `snapshot.py` | the dual-graph compiler and the anchor descriptions | [6.2](06_graph_and_storage.md#62-the-graph-schema), [5.8](05_latent_anchors.md#58-how-an-anchor-is-described) |
| | `workspace.py`, `chunk_journal.py`, `fileio.py` | workspace layout, journals, transactions and recovery, the writer lock; atomic file replacement that waits out Windows sharing violations | [6.3](06_graph_and_storage.md#63-the-workspace-on-disk), [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery) |
| | `neo4j_mirror.py`, `cypher/` | the Neo4j mirror | [6.6](06_graph_and_storage.md#66-neo4j-mirror) |
| | `demo.py`, `model_store.py` | the demo data with planted mechanisms; the embedding model provisioned and verified | [10.2](10_verification.md#102-the-synthetic-demo-dataset), [4.4](04_representation.md#44-the-embedding-model) |
| `insight_graph_service/server/` | `app.py`, `views.py` | the graph service's HTTP API; graph, node and sphere JSON for the console | [8](08_interface.md) |
| `evidence_narrator_service/` | `narration.py`, `llm_client.py` | the system prompt, the LLM answer or the evidence-only summary, the citation check; the OpenAI-compatible client | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) |
| `llm_model_broker/` | `server.py` | one OpenAI-compatible endpoint in front of the upstream model | [12.4](12_architecture.md#124-services-and-contracts) |
| `sig_web_console/` | `static/`, `nginx.conf` | the single-page UI; the single origin for both APIs | [8](08_interface.md) |
| `scripts/` | | the quality gate (`check.py`), the Docker check (`compose_check.py`), host development (`dev.py`), measurement scripts on throwaway workspaces | [10.4](10_verification.md#104-measurement-scripts), [10.5](10_verification.md#105-quality-gate-and-container-check) |

## 1.7 System-wide guarantees

1. **Traceable.** Every stored insight traces to its dataset, batch, exact EDA selector and covered rows.
2. **Covered.** Every insight has at least one ACTIVATES edge, and every anchor has at least one member — except an anchor whose last member was deleted while it is still linked to another anchor ([5.11](05_latent_anchors.md#511-removing-patterns-the-orphan-rule)).
3. **Planes stay separate.** Structural edges depend only on insight metadata; latent edges come from centroids (RELATED_TO) and from shared memberships (CO_OCCURS), never from scope predicates.
4. **One frame.** Vectors of different representations (model, composition, versions) never share a workspace; a mismatch is refused, not mixed ([4.6](04_representation.md#46-representation-identity-and-versions) lists what the fingerprint covers), and a workspace written by another version starts degraded until it is reset ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).
5. **Atomic batches, one writer.** A batch either commits completely or is rolled back, and one process at a time writes a workspace (the writer lock, [6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)). The Neo4j mirror runs after the commit, outside it: its failure is a warning, never a rollback.
6. **Grounded answers.** The LLM sees only the evidence object; every citation is checked against it, and a provenance footer is built without the LLM.

## 1.8 Deviations from the architecture document

The design started from [`docs/init_concepts/latent_insight_graph_architecture.md`](init_concepts/latent_insight_graph_architecture.md). The implementation differs where measurement or testability demanded it:

| Architecture document | Implementation | Reason |
|---|---|---|
| traversal in Cypher | traversal over the in-memory `DualGraph` (same schema); an approximate Cypher equivalent (one latent hop, then one lattice hop, no budgeted best-first search) is `insight_graph_service/core/cypher/queries/transversal.cypher` | testable without a database; Neo4j is an optional mirror |
| globally unique `Dimension` / `Metric` names | dataset-scoped ids `D:<dataset>:<name>`, `M:<dataset>:<name>` | datasets reuse column names with different meanings |
| three text embeddings per insight | scope and target are text embeddings; the phenomenon is a signed, magnitude-weighted sum of metric-name embeddings | sentence embeddings barely separate "increases" from "decreases" (cosine 0.56 measured) |
| every pattern equally authoritative | `insight_weight` scales the pattern's magnitude in the ontology and its rank in retrieval | weak evidence should not move anchors as much as strong evidence |
| pattern properties `mad_score`, `bootstrap_ci` | `sd_score`, `stability`, `p_value`, `p_adjusted` | the EDA produces a bootstrap coefficient of variation, not a confidence interval |

## 1.9 Scope and status

In scope: the four libraries and their vendored engines, the graph service with its local persistence and the optional Neo4j mirror, the narrator, the LLM model broker, the web console, model provisioning, and a Docker Compose setup of the services ([12.5](12_architecture.md#125-containers-and-start-order)). Out of scope: authentication, multi-user concurrency, distributed processing, model training, and migrating a workspace to a new representation version (it is reset and ingested again, [6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).

Implemented and tested: the test suite ([10.1](10_verification.md#101-test-map)) passes, including the real embedding model and a headless browser. The demo runs `84 cohorts → 50 validated → 28 insights → 4 anchors` in about 13 s including the one-off model load. The end-to-end flow has been run live against Gemma 4 on OpenRouter (5 of 5 answers grounded, 0 unknown citations, measured on 2026-10-01, [7.7](07_question_answering.md#77-measured-behaviour)) and the Neo4j mirror is checked against a real Neo4j container by `scripts/compose_check.py` ([10.5](10_verification.md#105-quality-gate-and-container-check)).
