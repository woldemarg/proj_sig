"""LTIR — Latent Transversal Insight Representation of statistically discovered insights.

Pipeline: tabular file -> EDA discovery (ltir/engines/eda) -> validated insights
-> canonical tripartite vectors -> latent attractor ontology (ltir/engines/lac)
-> dual-layer graph -> transversal traversal -> evidence -> the LLM behind LLM_BASE_URL (by default the LLM gateway).
See docs/README.md (reading guide), docs/01_overview.md and docs/12_architecture.md (layers and services).
"""
