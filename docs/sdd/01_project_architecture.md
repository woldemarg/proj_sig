# SDD 01 — Project architecture

## Purpose
Specify the end-to-end architecture of **LTIR / SIG**: a prototype that turns a tabular file into statistically validated insights. The insights live in a dual-layer graph with two planes. The **structural plane** is a deterministic lattice. The **latent phenomenon plane** holds living attractors. The system answers natural-language questions by transversal traversal plus a local Gemma 4 model that sees only retrieved evidence.

## Scope
In scope: every module of the `ltir` package, its reuse of `eda/` and `lac/`, local persistence, the optional Neo4j mirror, the web UI and the CLI.
Out of scope: authentication, multi-user concurrency, distributed processing, and model training.

## Inputs
A CSV/TSV/Parquet file (UI upload or CLI) plus optional band derivations (`BIN_COLUMNS`), and natural-language questions.

## Outputs
Persisted knowledge in `WORKSPACE_DIR`: journals, ontology state, and the graph snapshot. Also an interactive graph, and grounded answers with evidence, retrieval paths and provenance.

## Dependencies
| Kind | What |
|------|------|
| Reused code (vendored in `ltir/engines/`, see `PROVENANCE.md`) | `engines/eda/main_upd.py` (steps 1–4b, robust primitives; from `eda/scripts`); `engines/lac/` (`ConceptStore`, `ontology_engine`, `ChunkJournal`, `observability`, sphere `projector`; from `lac/v2_orchestrator` + `lac/v1_single_pass/visualisation`) |
| Libraries | numpy, pandas, scipy, scikit-learn, pysubgroup, sentence-transformers (torch), fastapi/uvicorn, httpx, neo4j (optional), playwright (optional tests) |
| Models | embedder `Qwen/Qwen3-Embedding-0.6B` (Matryoshka 384-d; default) or `paraphrase-multilingual-MiniLM-L12-v2`, bundled in `sig/models/` and loaded in-process (SDD 06); Gemma 4 only through an OpenAI-compatible endpoint (OpenRouter `google/gemma-4-26b-a4b-it`, or a local Ollama / LM Studio server), never loaded by SIG |

## Architecture

```text
 file ─► ingestion (02) ─► discovery adapter (03) ──► main_upd.step1..step4b   [reused; numerical repairs in PROVENANCE.md]
                                   │
                                   ▼
                     insight model + selection/weight (04)
                                   │
                     canonicalisation (05) ─► tripartite encoder (06)
                                   │                        │
             structural plane (08) │                        ▼
   SPECIALIZES/GENERALIZES/        │          latent ontology adapter (07) ──► lac ConceptStore / ontology_engine [reused]
   SIBLING/CONTRASTS               │            ACTIVATES + RELATED_TO (mutual kNN)
                                   ▼                        │
                        graph snapshot + persistence (09) ◄─┘      (optional Neo4j mirror)
                                   │
 question ─► query parse + seeds ─► transversal traversal (10) ─► evidence (11) ─► Gemma 4 / fallback (12) ─► answer
                                   │                                                        │
                                   └────────────── web UI (13): graph, inspection, chat, path highlight ◄──┘
```

Module map (package `sig/ltir/`):

| Module | Role | SDD |
|--------|------|-----|
| `config.py` | all tunables, env loading | 01 |
| `engines/` | vendored EDA and lac engines; discovery imports `eda.main_upd`, ontology imports `lac` | 03, 07, 13 |
| `ingestion.py` | file validation, band derivation, dataset id | 02 |
| `discovery.py` | EDA adapter → typed candidates → closed intents, duplicate cohorts pruned before validation → `Insight` | 03 |
| `models.py` | shared data contracts | 04 |
| `quality.py` | selection rules R1–R4, R7 + `insight_weight` (R5/R6 run in discovery) | 04 |
| `canonical.py` | canonical form (`ltir-canon-3`): embedding inputs, readable document, signed components, number formatting | 05, 17 |
| `encoder.py` | `TextEmbedder`s, `InsightEncoder`, `EmbeddingSpec` | 06 |
| `ontology.py` | `LatentOntology` over lac's lifecycle | 07 |
| `structural.py` | deterministic lattice edges | 08 |
| `graph.py` | snapshot assembly, `DualGraph` index | 08, 09 |
| `store.py` | workspace layout, journals, checkpoint/rollback | 09 |
| `neo4j_sink.py`, `cypher/` | optional Neo4j mirror | 09 |
| `query.py` | question parse, seed resolution | 10 |
| `traversal.py` | transversal search, baselines | 10 |
| `evidence.py` | `Evidence` object and prompt | 11 |
| `llm.py`, `qa.py` | LLM client, grounded QA flow | 12 |
| `web/` | FastAPI app + static UI | 13 |
| `sphere.py` | 3D latent sphere over lac's prosphera projector | 13 |
| `pipeline.py` | `Engine`, lifecycle, recovery, committed `LatentFrame` | 14 |
| `migrate.py` | `python -m ltir migrate --yes`: rebuild an outdated workspace from its stored sources (old copy kept) | 09 |
| `synth.py`, `experiment.py`, `cli.py` | demo data, hypothesis benchmark, CLI | 15, 14 |
| `scripts/check.py`, `pyproject.toml` | quality gate (ruff check, ruff format, pytest) and its configuration | 15 |
| `scripts/` (others) | measurements on scratch workspaces (`.scratch/`): prompt tokens, embedder comparison, live answer eval, model download | 15 |

Cross-cutting documents: SDD 16 (every formula, in pipeline order) and SDD 17 (every text contract: records, renderings, embedding inputs, LLM prompt). `AGENTS.md` at the repo root is the working agreement for changing any of this.

## Data contracts
Defined once in `ltir/models.py` (SDD 04): `Condition`, `Shift`, `Insight`, `Rejection`, `CanonicalInsight`, `EmbeddingSpec`, `EdgeType`, `GraphNode`, `GraphEdge`, and the activation record. Retrieval contracts (`ParsedQuery`, `SeedMatch`, `PathStep`, `Retrieved`, `TraversalResult`, `Evidence`, `QAResult`) live with their modules (SDD 10–12). pandas objects never cross the discovery-adapter boundary.

## The two planes
| | Structural plane Gs | Latent phenomenon plane Gl |
|---|---|---|
| Nodes | `Pattern` (validated insight); `Dimension`, `Metric` as schema anchors | `Attractor` (living centroid on the unit hypersphere) |
| Edges | SPECIALIZES, GENERALIZES, SIBLING, CONTRASTS (exact, from scope conditions and signed shifts) | RELATED_TO (mutual kNN of centroids) |
| Bridge | ACTIVATES (Pattern → Attractor; alignment, strength = alignment × insight_weight) | |
| Never | embeddings do not create structural edges | scope predicates do not create latent edges |

## Configuration
`ltir/config.py::Config` is the single source of tunables. Values come from environment variables with the upper-case field name (optionally loaded from `sig/.env`), and explicit overrides win over both. Ontology parameters keep lac's names (`CENTROID_ALPHA`, `DICTIONARY_K_MIN`, …) with SIG-sized defaults, documented in SDD 07. `Config.public_dict()` masks secrets.

## Failure modes
Every failure is a batch state or an answer mode, never a crash of the service (SDD 14 lists the codes). LLM failure degrades to an evidence-only answer (SDD 12). Neo4j failure degrades to a warning (SDD 09).

## Invariants
1. Every persisted Pattern traces to dataset, batch, exact EDA selector and covered rows.
2. Every Pattern has ≥ 1 ACTIVATES edge, and every Attractor has ≥ 1 activating Pattern.
3. Structural edges depend only on `Insight` metadata.
4. Vectors of different representation fingerprints never share a workspace.
5. A batch either commits completely or is rolled back.

## Testing requirements
See SDD 15. Each critical transition has a contract test, plus E2E and UI smoke tests.

## Integration points
CLI `python -m ltir …`, web `python -m ltir.web`, the optional Neo4j database `NEO4J_DATABASE`, and the local LLM endpoint `LLM_BASE_URL`.

## Glossary
One concept, one name per layer. Code and SDDs use the first column; the UI uses the plain word; lac's vendored modules keep lac's word.

| Code / SDD | UI | lac (vendored) | Meaning |
|---|---|---|---|
| Pattern, insight (`Insight`) | insight | chunk (a journal row) | a validated subgroup finding: scope + phenomenon + statistics |
| Attractor, latent anchor (`A-k`) | theme | concept | a living centroid on the unit sphere that recurring phenomena activate |
| scope (conditions) | where | — | the conjunction of `attribute = value` selectors defining the subgroup |
| target, metric | metric | — | the numeric column whose robust median shift (or correlation) the pattern reports |
| robust z | ±x sd | — | 0.6745·Δmedian / MAD, signed, capped at 10 (the prompt and UI write "sd") |
| cohort, extent | — | — | the set of rows a subgroup covers; selectors with the same extent are one cohort (SDD 03) |
| closed intent | scope | — | every `attribute = value` constant on a cohort's rows; the pattern's conditions (SDD 03, 16 §1) |
| `insight_weight` | evidence | weight (the scaled input magnitude) | statistical strength in [WEIGHT_FLOOR, 1] (SDD 04) |
| ACTIVATES | membership | activation | pattern → attractor edge (alignment, strength) |
| RELATED_TO | theme link | RELATED_TO | mutual-kNN edge between attractors |
| seed | match | — | a pattern resolved directly from the question (SDD 10) |
| transversal / `transversal_only` | via theme / other segment | — | reached through the latent plane; scope-disjoint from every seed |
| frame (`LatentFrame`) | — | — | the single space of unit insight vectors and centroids (no centering) |
| batch | dataset card | batch | one processing run of one uploaded file (SDD 14) |

## Deviations from `latent_insight_graph_architecture.md`
| Architecture | Implementation | Reason |
|---|---|---|
| Traversal in Cypher | Traversal over the in-memory `DualGraph` (same schema); equivalent Cypher in `ltir/cypher/queries/transversal.cypher` | Testable without Neo4j; Neo4j is an optional mirror |
| Single `Dimension`/`Metric` name uniqueness | Ids are dataset-scoped (`D:<dataset>:<name>`, `M:<dataset>:<name>`) | Several datasets may reuse column names with different semantics |
| Tripartite text embeddings | Scope and target are text embeddings; the phenomenon is a signed, magnitude-weighted composition of metric-name embeddings | Sentence embeddings barely separate "increase" from "decrease" (cos 0.56 measured) |
| Every pattern equally authoritative | `insight_weight` scales the pattern's magnitude in the ontology | Requirement §9 |
| Pattern property `mad_score`, `bootstrap_ci` | `sd_score`, `stability`, `p_value`, `p_adjusted` | The EDA produces a bootstrap CV, not a CI |

## Current implementation status
Implemented and tested (69 tests passing). The E2E flow works in the web UI and the CLI with live Gemma 4 (`google/gemma-4-26b-a4b-it` via OpenRouter) and the live Neo4j mirror (`sigv1`).
