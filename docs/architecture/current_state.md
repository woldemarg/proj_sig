# Current state — repository reconnaissance (Phase 0)

> **Update (2026-09-30):** the reused engines, the embedding model and `housing.csv` are now **vendored into `sig/`** (`ltir/engines/`, `models/`, `data/`; see `ltir/engines/PROVENANCE.md`). The paths and APIs below describe the *originals* at reconnaissance time, not runtime dependencies; the vendored copies have diverged (e.g. no running-mean centering, one `extract_attractors`, numerical repairs in `main_upd.py`), and PROVENANCE.md lists every difference. Terms: glossary in `docs/11_reference.md` §11.1.

Snapshot of what existed **before** the LTIR integration work. It records what the integration reuses as-is, what gets an adapter, and what is new.
Paths name the original projects (`eda`, `lac`) and files inside them; nothing here is a runtime dependency.

## 1. Projects

| Project | Role | VCS | Notes |
|--------|------|-----|-------|
| `eda/` | Automatic EDA / statistical insight discovery | none | `scripts/main_upd.py` is the current engine (`main_init.py`, `main_ext.py` are older iterations). Data: `data/housing.csv` (20 640 rows), `data/demo_points.csv` (60 604 rows, operational data — not used here). |
| `lac/` | Dynamic latent ontology ("proj_ontology") | git | `v2_orchestrator/` = Latent Semantic Attractor Graph (streaming, EMA attractors, OMP, mutual k-NN, Neo4j MERGE). `v1_single_pass/` = static POC. |
| `sig/` | **Target integrated project** (Statistical Insight Graph) | none | Before this work it held only `docs/init_concepts/` (the two theory documents). |

## 2. Existing component map

```text
component                     public API                                   data structures                      deps                          tests                      integration suitability
----------------------------  -------------------------------------------  -----------------------------------  ----------------------------  -------------------------  ------------------------------------------
eda/scripts/main_upd.py       step1_profile_data(df)                       profile dict {data_safe, numerics,   pandas, numpy, pysubgroup,    none                       REUSE via import (after 3 minimal edits,
                              step2_evaluate_macro_groupings(profile)       categoricals, global_corr,           scipy                                                    §4); adapter adds typing, signed shifts,
                              step3_generate_search_space(profile, cats)    total_rows}; candidate DataFrame                                                           provenance, significance, selection.
                              step4_evaluate_micro_slices(profile, space)   (dimensions, dimension_attrs,
                              step4b_deep_validation(...)                   row_indices, row_count, top_shifts,
                              calculate_mad / robust_z_score /              sd_aggregate_score, emm_...,
                              volume_preference_score                       volume_utility); validated DataFrame
                              step5_integrate_and_materialize (print only)
lac/v2_orchestrator/          ConceptStore (update_global_mean,            embeddings (k,d) float32 unit,        numpy                         verify_smoke (offline),    REUSE unchanged (ConceptStore state +
  storage.py                  update_concept_centroid, bootstrap,          chunk_counts, last_updated_batch,                                   verify_neo4j (live)        save/load; EMA inertia).
                              append_concepts, push_orphans, save/load)     orphan buffer, global_mean
  ontology_engine.py          compute_adaptive_threshold, assign_and_      activation dicts {chunk_id,          numpy, scikit-learn                                      REUSE unchanged (assign, OMP K-sweep,
                              update, assign_orphans_nearest,              concept_id, weight}                                                                       soft merge, routing, mutual k-NN).
                              cold_start_extract, extract_orphans_omp,
                              soft_merge_orphans, route_absorbed_...,
                              build_kept_local_to_global,
                              remap_activation_edges, calculate_knn_...
  chunk_journal.py            ChunkJournal.append_batch/load_*             chunks.jsonl, activations.jsonl,      numpy                                                    REUSE via 3-line subclass (file names).
                                                                           embeddings.mmap + meta
  observability.py            BatchMetrics, MetricsRecorder, avg_related_  metrics.csv                          numpy                                                    REUSE (ontology health metrics).
                              to_degree, max_concept_density_pct,
                              mean_centroid_drift, apply_health_warnings
  neo4j_uploader.py + cypher/ Neo4jOntologyPublisher (Chunk/Concept)       UNWIND/MERGE Cypher                  neo4j driver                                             PATTERN reused; SIG schema (Pattern/
                                                                                                                                                                     Attractor/...) needs its own Cypher files.
  main.py run_batch           Wikipedia-coupled batch loop                                                       wikipedia, langchain                                     NOT callable (fetches Wikipedia); its
                                                                                                                                                                     stage order is mirrored in ltir/ontology.py.
  settings.py                 Settings (frozen dataclass), load_settings   env/.env                                                                                   REUSE the dataclass; SIG builds an instance
                                                                                                                                                                     from its own config (no lac .env read).
lac/models/sentence-...       paraphrase-multilingual-MiniLM-L12-v2 cache  384-d                                sentence-transformers                                    REUSE as the local embedding model.
```

Theory / architecture: `sig/docs/init_concepts/latent_insight_graph_architecture.md` (full blueprint: SD + EMM + FCA + pattern structures + interestingness + sparse dictionary learning + dual-layer graph + transversal traversal + neuro-symbolic RAG) and `sig/docs/init_concepts/auto_eda_plus_graph_topology.md` (concept discussion; names **SIG** = Statistical Insight Graph, **LIO** = Latent Insight Ontology, **LTIR** = the whole system).

lac docs: `lac/docs/v2-latent-semantic-attractor-graph/` (data-flow, concept inertia, configuration, operations) and `lac/docs/cypher/queries/` (RAG subgraph Cypher).

## 3. Environment and infrastructure found

| Item | Finding | Consequence |
|------|---------|-------------|
| Python | Python 3.12 environments with torch (CUDA 12.6), sentence-transformers 5.6, transformers 5.12, neo4j and scikit-learn, but without pysubgroup or FastAPI. | SIG uses its own `.venv`, optionally layered on such an environment (`--system-site-packages`), plus pysubgroup, FastAPI, uvicorn and pytest. No existing environment is modified. |
| transformers import | `transformers` failed to import there (`regex` missing). | `regex` is listed in `requirements.txt` and installed in the SIG venv only. |
| Embedding model | Cached at `lac/models/sentence-transformers/` (MiniLM-L12, 384-d). Loads offline. | Was the default SIG embedding model (`HF_HUB_OFFLINE=1`); since `ltir-rep-3` it is the documented alternative to `Qwen3-Embedding-0.6B` (`docs/04_representation.md` §4.4). |
| GPU | A laptop GPU with 8 GB. | Embedding on CUDA when available (configurable). |
| Neo4j | A local Neo4j server, **not running**. | Neo4j is an optional mirror (`NEO4J_ENABLED`); the local journal/state/snapshot is the source of truth. |
| Local LLM | Common OpenAI-compatible local endpoints: Ollama `http://localhost:11434/v1` (model `gemma4`) and LM Studio `http://localhost:1234/v1` (model `google/gemma-4-26b-a4b`); **neither was running**. | LLM client targets the OpenAI-compatible API (default Ollama `gemma4`); deterministic evidence-only fallback when unreachable. |

## 4. Defects found in existing code and how they are handled

| Location | Defect | Blocks E2E? | Action |
|----------|--------|-------------|--------|
| `eda/scripts/main_upd.py` module level | Reads `../data/demo_points.csv` and runs the whole workflow at import time. | Yes (cannot import). | Wrapped the data load and the workflow in `if __name__ == "__main__":` (Spyder cell execution unchanged). |
| `step1_profile_data` | Any column whose values are all unique is dropped as an identifier, so continuous **float** targets are dropped. | Yes (typical CSVs lose their measures). | Identifier rule now skips float dtype columns. Unchanged for `housing.csv`/`demo_points.csv`. |
| `step2_evaluate_macro_groupings` | Keeps only categories with power **strictly above the median**. With ≤3 categoricals, ≤1 survives, and since `step3` builds only 2-/3-conjunctions the search space is empty (e.g. `housing.csv`). | Yes for narrow schemas. | Added optional `min_categories=0` (default preserves behaviour); the adapter passes `MIN_SEARCH_DIMENSIONS` (default 3), which backfills by power ranking. |
| `step4b_deep_validation` | Bootstrap uses unseeded `DataFrame.sample`. | Reproducibility only. | Adapter seeds NumPy's global RNG (`EDA_RANDOM_SEED`) before calling it. |
| `robust_z_score` | Returns `abs(...)`, so shift direction is lost. | Yes for CONTRASTS / phenomenon direction. | Adapter recomputes the sign from subgroup vs global medians (magnitude taken from the EDA). |
| `lac/.../ontology_engine._omp_extract` | Activation weights use `abs(OMP coefficient)`, so an anti-aligned pattern can "activate" an atom. | Semantics only. | The vendored `extract_attractors` repairs atom sign and re-validates alignments after extraction (`docs/05_latent_anchors.md` §5.3). |
| `eda/.../main_upd.py` (math audit) | MAD = 0 makes robust z infinite; the 95 %-mass rule drops a column's last level; η² favours high-cardinality columns; JS-only confounders fire on sampling noise; 5 × 90 % "bootstrap"; undefined correlations read as 0. | Wrong rankings and spurious insights. | Repaired in the vendored copy (PROVENANCE.md edits 4–9, `docs/02_discovery.md` §2.2). |
| `lac/.../storage.py` running-mean centering | Batch t is centred by the mean of batches < t (batch 0 uncentred), so the frame moves between batches and query vectors cannot share it. | Retrieval consistency. | Not used: SIG keeps one uncentred frame (since `ltir-rep-2`; `docs/04_representation.md`, `docs/05_latent_anchors.md`). |

## 5. What had to be built

Nothing below existed: typed insight model, quality/redundancy policy, canonicalisation, tripartite encoder, insight weighting, adapter over the lac lifecycle for patterns, structural lattice edges, dual-layer graph + persistence for patterns, traversal engine, evidence builder, LLM interface, web UI, E2E pipeline with status tracking, synthetic dataset, and tests.
See [`docs/01_overview.md`](../01_overview.md) for the resulting architecture.
