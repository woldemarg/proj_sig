# SIG documentation — reading guide

These chapters describe the system as it is implemented, in the order data flows through it. Read them in sequence for a complete technical overview; each chapter also stands on its own, links back to what it builds on, and names the code and tests it describes. One example — the insight *"phones in the US: discount +2.21 sd, margin −1.10 sd"* from the demo dataset — is followed through every chapter.

| # | Chapter | What it answers | Code |
|---|---|---|---|
| 1 | [Overview](01_overview.md) | What SIG is, the hypothesis, the pipeline, the two graph planes, the module map | the whole package |
| 2 | [Discovery](02_discovery.md) | How a table becomes validated subgroups: ingestion, the five EDA steps, deduplication before validation, what the adapter adds | `analysis/ingestion.py`, `analysis/discovery.py`, `engines/eda/` |
| 3 | [Insights](03_insights.md) | What an insight record holds, which insights are kept, how the evidence weight is computed and where it acts | `models.py`, `analysis/quality.py` |
| 4 | [Representation](04_representation.md) | The two text contracts, the canonical form, the tripartite vector and its geometry, the embedding model, the representation fingerprint | `analysis/canonical.py`, `analysis/encoder.py` |
| 5 | [Latent anchors](05_latent_anchors.md) | How recurring phenomena self-organise into anchors: extraction, assignment, guards, links, descriptions, calibration | `analysis/ontology.py`, `engines/lac/` |
| 6 | [The dual graph and its storage](06_graph_and_storage.md) | The structural lattice, the graph schema, the workspace on disk, atomic commits, migration, the Neo4j mirror | `analysis/structural.py`, `analysis/graph.py`, `storage/workspace.py`, `migrate.py`, `storage/neo4j_mirror.py` |
| 7 | [Question answering](07_question_answering.md) | From a question to seeds, the transversal walk, the evidence and its prompt, the LLM and the citation check | `retrieval/`, `answering.py`, `llm_client.py` |
| 8 | [Interface](08_interface.md) | The HTTP API, the web UI, its visual encoding and highlights, the 3D latent sphere | `web/`, `web/sphere.py` |
| 9 | [Operations](09_operations.md) | The batch lifecycle, the command line, configuration, error codes, metrics, deployment | `engine.py`, `cli.py`, `config.py` |
| 10 | [Verification](10_verification.md) | The test map, the synthetic dataset, the hypothesis benchmark, measurement scripts, the quality gate, known limitations | `tests/`, `evaluation/synthetic.py`, `evaluation/experiment.py`, `scripts/` |
| 11 | [Reference](11_reference.md) | Glossary, identifier formats, every parameter, a formula index, the versioned contracts | — |
| 12 | [Architecture](12_architecture.md) | The layers and their dependency rule, reusing the analytical core, the LLM gateway, the three containers, observability | `ltir/` packages, `llm_gateway/`, `Dockerfile`, `compose.yaml` |

**Shorter paths.** To *use* the system: 1, 9, 8 (in containers: 12.5). To *reuse* the analytical core or *change the structure*: 12. To change *statistics*: 2, 3, 10. To change *text or vectors*: 4, 7 (and the versioning rules in 11.5). To change the *ontology*: 5, 6. To look something up: 11.

## Conventions

* **Code references** name a module and a symbol: `discovery.closed_intent` is `ltir/analysis/discovery.py::closed_intent`; vendored modules are named by file (`ontology_engine.extract_attractors` is in `ltir/engines/lac/ontology_engine.py`).
* **Configuration** names in capitals are fields of `ltir/config.py::Config`, settable from the environment ([9.3](09_operations.md#93-configuration)); the value in parentheses is the default.
* **Formulas** are written in plain text blocks, next to the code that computes them; [11.4](11_reference.md#114-formula-index) indexes them all.
* **Running example** blocks show the demo's actual output at that stage.
* **Section numbers** (`§5.6`) are stable handles: code comments cite chapters as `docs/05_latent_anchors.md §5.6`.
* The docs describe the current state; [`ltir/engines/PROVENANCE.md`](../ltir/engines/PROVENANCE.md) records how the vendored engines differ from their origins, and [`AGENTS.md`](../AGENTS.md) holds the rules for changing code and docs together.

## Background material

These documents are context, not specification:

* [`init_concepts/`](init_concepts/) — the theory documents the project started from: the latent insight graph architecture, the automatic-EDA and graph-topology concept, and the statistical-pattern embedding audit. Read-only.
