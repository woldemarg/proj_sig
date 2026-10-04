# graph_query_engine

`graph_query_engine` answers a question over the dual graph with evidence, not prose. The question may be in English, Ukrainian or a mix of the two (data literals in any script). The engine:
1. grounds the question's words in the graph's literals;
2. picks seed patterns;
3. walks the structural lattice and the latent attractor bridges;
4. packs what it found into evidence with its paths and statistics ([docs/07_question_answering.md](../docs/07_question_answering.md)).

The result can be the `EvidencePayload` from `insight_contracts`: an LLM-ready evidence prompt, the deterministic cited observation lines (an answer without an LLM, once framed), a citation manifest (`P#` → pattern and provenance) and the data a UI draws.

It depends only on `insight_contracts`, numpy and scikit-learn (`requirements.txt`). It reads a compiled snapshot (`DualGraph`) and the committed vectors (`LatentFrame`). The question is embedded by any object with the `ports.QueryEncoder` shape; `attractor_topology.InsightEncoder` is one.

```python
from graph_query_engine import CommittedState, DualGraph, LatentFrame, QueryConfig, search

state = CommittedState(DualGraph(snapshot), LatentFrame(patterns, attractors, documents))
found = search("Why is margin lower for phones in the US?", state, encoder, QueryConfig())

payload = found.payload()
payload.evidence_prompt  # put it in the LLM prompt; cite the keys [P1], [P2], ...
payload.citations  # check the keys the model cites
```

`tests/test_graph_query_engine_standalone.py` runs a search and the payload round trip; `tests/test_architecture.py` checks that the package imports only the kernel.
