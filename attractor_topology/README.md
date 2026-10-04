# attractor_topology

`attractor_topology` represents insights as vectors and clusters them into a living set of latent attractors (anchors).

**The representation.** Each `Insight` becomes a tripartite 1152-d unit vector made of a scope block, a target block and a signed phenomenon block, embedded with Qwen3-Embedding-0.6B ([docs/04_representation.md](../docs/04_representation.md)).

**The attractors.** Insights with the same statistical behaviour join the same attractor, even when their scopes share no condition. The attractors are extracted by OMP, move with EMA updates and are linked by a mutual-kNN topology ([docs/05_latent_anchors.md](../docs/05_latent_anchors.md)).

It depends only on `insight_contracts` and on numpy, scikit-learn and sentence-transformers (`requirements.txt`; install torch for your device first). Its input is any list of `Insight` records, from `subgroup_miner` or from another source.

```python
from pathlib import Path

import numpy as np
from attractor_topology import InsightEncoder, LatentOntology, SentenceTransformerEmbedder, TopologyConfig, canonicalize

config = TopologyConfig(model_dir=Path("models"))  # holds Qwen3-Embedding-0.6B/
encoder = InsightEncoder(SentenceTransformerEmbedder(config), config)
vectors = encoder.encode([canonicalize(i, config) for i in insights])["vector"]

ontology = LatentOntology(config, Path("state"))  # the anchors persist in this folder
update = ontology.ingest(vectors, np.array([i.weight for i in insights]), [i.id for i in insights], batch_seq=0, batch_id="run-1")
ontology.save()
update.activations  # pattern -> attractor memberships with alignments
ontology.attractors()  # id, centroid, mass of every attractor
ontology.topology()  # RELATED_TO links between attractors
```

`encoder.spec` is the embedding contract. Its fingerprint changes whenever stored vectors would stop being comparable.

The lac engine under `vendor/lac/` is a modified copy (`vendor/PROVENANCE.md`). `tests/test_attractor_topology_standalone.py` runs the flow above with a deterministic test embedder; `tests/test_architecture.py` checks that the package imports only the kernel.
