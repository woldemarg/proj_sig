# SDD index

Specification-driven documentation for `sig/ltir`. Each document states purpose, scope, inputs, outputs, dependencies, data contracts, algorithms, configuration, failure modes, invariants, testing requirements, integration points and current implementation status. Where a document has no applicable content for a section, it says so or folds the section into a neighbour. Contracts here, code in `ltir/`, and tests in `tests/` describe the same thing. Any change to one updates the others.

| # | Document | Code | Tests |
|---|---|---|---|
| 01 | [Project architecture](01_project_architecture.md) | `config.py`, package map | all |
| 02 | [Data ingestion](02_data_ingestion.md) | `ingestion.py` | `test_persistence.py` |
| 03 | [Statistical discovery adapter](03_statistical_discovery_adapter.md) | `discovery.py` | `test_discovery_contract.py` |
| 04 | [Insight model, selection, weight](04_insight_model.md) | `models.py`, `quality.py` | `test_quality.py` |
| 05 | [Canonicalisation](05_insight_canonicalization.md) | `canonical.py` | `test_canonical_embedding.py` |
| 06 | [Embedding layer](06_embedding_layer.md) | `encoder.py` | `test_canonical_embedding.py` |
| 07 | [Latent ontology](07_latent_ontology.md) | `ontology.py` | `test_ontology.py` |
| 08 | [Structural plane & graph schema](08_structural_graph.md) | `structural.py`, `graph.py` | `test_structural.py`, `test_persistence.py` |
| 09 | [Persistence](09_graph_persistence.md) | `store.py`, `graph.py`, `neo4j_sink.py`, `cypher/` | `test_persistence.py` |
| 10 | [Query & traversal](10_traversal_engine.md) | `query.py`, `traversal.py` | `test_traversal.py`, `test_e2e.py` |
| 11 | [Evidence builder](11_evidence_builder.md) | `evidence.py` | `test_traversal.py`, `test_e2e.py` |
| 12 | [LLM interface & QA](12_llm_interface.md) | `llm.py`, `qa.py` | `test_llm_client.py`, `test_e2e.py` |
| 13 | [Visualisation UI + 3D sphere](13_visualization_ui.md) | `web/`, `sphere.py` | `test_ui_smoke.py`, `test_sphere.py` |
| 14 | [End-to-end pipeline](14_end_to_end_pipeline.md) | `pipeline.py`, `cli.py` | `test_e2e.py`, `test_persistence.py` |
| 15 | [Testing, demo data, experiment](15_testing_strategy.md) | `synth.py`, `experiment.py`, `scripts/check.py` | all |
| 16 | [Core mathematics as implemented](16_core_mathematics.md) | every formula, in pipeline order, with its code and parameters | planted-phenomenon and formula tests |
| 17 | [Textual contracts](17_textual_contracts.md) | what is stored, rendered, embedded and shown to the LLM; how anchors get their description | canonical / embedding / evidence tests |

Terminology (code ↔ UI ↔ lac): the glossary in [SDD 01](01_project_architecture.md#glossary). Rules for changing the code: [`AGENTS.md`](../../AGENTS.md).
