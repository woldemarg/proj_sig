# SIG documentation — reading guide

These chapters describe the system as it is implemented, in the order data flows through it. Read them in sequence for a complete technical overview; each chapter also stands on its own, links back to what it builds on, and names the code and tests it describes. One example — the insight *"phones in the US: discount +2.21 sd, margin −1.10 sd"* from the demo dataset — is followed through every chapter.

| # | Chapter | What it answers | Code |
|---|---|---|---|
| 1 | [Overview](01_overview.md) | What SIG is, the hypothesis, the pipeline, the two graph planes, the module map | the whole repository |
| 2 | [Discovery](02_discovery.md) | How a table becomes validated subgroups: ingestion, the five EDA steps, deduplication before validation, what the adapter adds | `subgroup_miner/ingestion.py`, `subgroup_miner/discovery.py`, `subgroup_miner/vendor/eda/` |
| 3 | [Insights](03_insights.md) | What an insight record holds, which insights are kept and admitted, how the evidence weight is computed and where it acts | `insight_contracts/insight.py`, `subgroup_miner/selection.py`, `insight_graph_service/core/batch.py` (admission) |
| 4 | [Representation](04_representation.md) | The two text contracts, the canonical form, the tripartite vector and its geometry, the embedding model, the representation fingerprint | `attractor_topology/canonical.py`, `attractor_topology/encoder.py`, `attractor_topology/models.py`, `insight_contracts/text.py` |
| 5 | [Latent anchors](05_latent_anchors.md) | How recurring phenomena self-organise into anchors: extraction, assignment, guards, links, descriptions, calibration | `attractor_topology/ontology.py`, `attractor_topology/vendor/lac/` |
| 6 | [The dual graph and its storage](06_graph_and_storage.md) | The structural lattice, the graph schema, the workspace on disk, atomic commits, versions and the degraded start, the Neo4j mirror, dataset deletion | `subgroup_miner/lattice.py`, `insight_graph_service/core/snapshot.py`, `insight_graph_service/core/workspace.py`, `insight_graph_service/core/neo4j_mirror.py` |
| 7 | [Question answering](07_question_answering.md) | From a question to seeds, the transversal walk, the evidence and its prompt, the LLM and the citation check | `graph_query_engine/`, `evidence_narrator_service/` |
| 8 | [Interface](08_interface.md) | The HTTP API, the web console, its visual encoding and highlights, the 3D latent sphere | `sig_web_console/`, `insight_graph_service/server/` |
| 9 | [Operations](09_operations.md) | The batch lifecycle, running the service, configuration, error codes, metrics | `insight_graph_service/core/engine.py`, `insight_graph_service/core/settings.py`, `insight_graph_service/core/model_store.py` |
| 10 | [Verification](10_verification.md) | The test map, the synthetic dataset, the hypothesis benchmark, measurement scripts, the quality gate, known limitations | `tests/`, `insight_graph_service/core/demo.py`, `scripts/` |
| 11 | [Reference](11_reference.md) | Glossary, identifier formats, every parameter, a formula index, the versioned contracts | — |
| 12 | [Architecture](12_architecture.md) | The bounded contexts and their import rule, using a library on its own, the services and their contracts, containers and start order, storage, observability | the packages, `compose.yaml`, the `Dockerfile` of each service |
| 13 | [Trust and integrity](13_trust.md) | Why every finding carries its source's reliability, its credibility from other sources and its statistics apart, the integrity checks, where trust is shown, the audit against knowledge-graph poisoning | `insight_graph_service/core/snapshot.py` (credibility, digests), `graph_query_engine/evidence.py` (trust line), `scripts/poisoning_eval.py` |

**Shorter paths.** To *use* the system: 1, 9, 8 (in containers: 12.5). To *reuse* a library on its own or *change the structure*: 12. To change *statistics*: 2, 3, 10. To change *text or vectors*: 4, 7 (and the versioning rules in 11.5). To change the *ontology*: 5, 6. To judge *how far an answer can be trusted* or change the trust model: 13, then 6.2, 6.7 and 7.4. To look something up: 11.

## Conventions

* **Code references** name a module and a symbol: `discovery.closed_intent` is `subgroup_miner/discovery.py::closed_intent`; vendored modules are named by file (`ontology_engine.extract_attractors` is in `attractor_topology/vendor/lac/ontology_engine.py`).
* **Configuration** names in capitals are fields of a package config (`MinerConfig`, `TopologyConfig`, `QueryConfig`), of `PhenomenonThresholds` or of the graph service's `Settings` (`insight_graph_service/core/settings.py`), set from the environment by their upper-case names ([9.3](09_operations.md#93-configuration)); the value in parentheses is the default.
* **Formulas** are written in plain text blocks, next to the code that computes them; [11.4](11_reference.md#114-formula-index) indexes them all.
* **Running example** blocks show the demo's actual output at that stage.
* **Section numbers** (`§5.6`) are stable handles: code comments cite chapters as `docs/05_latent_anchors.md §5.6`.
* The docs describe the current state. The `PROVENANCE.md` files record how the vendored code differs from its origins: [the EDA engine](../subgroup_miner/vendor/PROVENANCE.md), [the ontology engine](../attractor_topology/vendor/PROVENANCE.md), [the chunk journal](../insight_graph_service/core/PROVENANCE.md) and [the test data](../tests/vendor/PROVENANCE.md). [`AGENTS.md`](../AGENTS.md) holds the rules for changing code and docs together.

## Background material

These documents are context, not specification:

* [`init_concepts/`](init_concepts/) — the theory documents the project started from: the latent insight graph architecture, the automatic-EDA and graph-topology concept, and the statistical-pattern embedding audit. Read-only.
* [`extra/spatial_knowledge_graph_architecture.md`](extra/spatial_knowledge_graph_architecture.md) (Ukrainian) — the target architecture of the Spatial Knowledge Graph epic (#2) as the issues define it on 2026-10-15: what the code already has, what issues #3–#35 add (levels per upload, regions derived at commit, automatic vertical zoom, zone as query, `supersedes`, trust passport), standards, risks and the milestone plan. [`extra/spatial_knowledge_graph_transition.md`](extra/spatial_knowledge_graph_transition.md) records how the plan got there from the 2026-10-13 study: decisions, the levels/zoom additions, the complexity review with the repo audit, the harmonisation and the 23 answered questions. [`extra/spatial_knowledge_graph_story.md`](extra/spatial_knowledge_graph_story.md) tells the same system as one plain-language example. The multi-resolution theory they respond to is [`init_concepts/architecture_hierarchical_multi_resolution_geograph.md`](init_concepts/architecture_hierarchical_multi_resolution_geograph.md). Studies, not specifications: chapters 01–12 describe the code.
