"""The analytical core: a table becomes validated insights, their vectors, latent anchors and the dual graph.

Graph construction reads no files: the engine passes it the data. The only state kept here is the ontology's own
(lac's concept store, in the directory the engine passes it). Nothing here calls an LLM or renders anything
(docs/12_architecture.md).
"""
