# SDD 06 — Embedding layer

## Purpose
Map canonical insights to reproducible vectors in a structured semantic space, using the local embedding model and a model-independent `InsightEncoder` with a tripartite scope/target/phenomenon composition.

## Scope
`ltir/encoder.py`: `TextEmbedder` protocol, `SentenceTransformerEmbedder`, `HashingEmbedder`, `InsightEncoder`, `EmbeddingSpec` (in `models.py`).

## Inputs
`list[CanonicalInsight]` (SDD 05). For queries: scope text, target text, signed components and raw question text.

## Outputs
`encode()` → `{"vector": (n, 3d), "scope": (n, d), "target": (n, d), "phenomenon": (n, d)}`, all `float32`, all rows unit L2 norm. `encode_query()` → `(3d,)`. `spec` → `EmbeddingSpec`.

## Dependencies
sentence-transformers loading the bundled model folder `MODEL_DIR/<EMBEDDING_MODEL>` = `sig/models/paraphrase-multilingual-MiniLM-L12-v2` (offline). If that folder is missing, it falls back to resolving the name and caching the download under `MODEL_DIR`. No hosted embedding service.

## Algorithm
```text
s = E(scope text)                                   (unit)
t = E(target text)                                  (unit)
p = normalize( Σ_k c_k · E(label_k) )               (unit; c_k = signed component, SDD 05)
v = normalize( [ w_s·s ; w_t·t ; w_p·p ] )          w = BLOCK_WEIGHTS = (0.45, 0.55, 1.0)
```
* **Why a composed phenomenon block.** Measured with MiniLM, cos("margin decreases strongly", "margin increases strongly") = 0.56, so sentence embeddings barely encode direction. With signed composition, opposite shifts of the same metric give p·p′ = −1. Co-shift structure (which metrics move together) is kept, and scope never enters p.
* **Why phenomenon dominates.** Attractors should represent recurring behaviour across scopes. With these weights, same phenomenon + disjoint scope gives cos ≈ 0.9 (measured 0.93–0.99 on the demo). The same scope with an opposite shift gives cos ≈ −0.3.
* If all components cancel (norm < 1e-9), p falls back to E(phenomenon text).
* Text embeddings are memoised per text (metric labels repeat).

## Data contract — `EmbeddingSpec`
| Field | Value (default) |
|---|---|
| `model_id` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or `hashing-ngram-256` |
| `block_dim` / `dim` | 384 / 1152 |
| `dtype` | `float32` |
| `normalization` | `l2(block) -> weighted concat -> l2` |
| `block_weights`, `emm_component_weight` | (0.45, 0.55, 1.0), 0.5 |
| `canonical_version`, `representation_version` | `ltir-canon-2` (EMM per-pair scale, capped robust z), `ltir-rep-2` (single uncentred frame) |
| `fingerprint` | sha1 of all of the above [:10] |

The spec is written to `state/representation.json` when the first batch commits and is carried in every pattern record (`embedding`). A later batch with a different fingerprint fails with `representation_mismatch` (SDD 09), so historical vectors are never silently mixed; a version bump is how an incompatible change (e.g. the `ltir-rep-1` running-mean frame) is refused instead of patched over. The composite vectors are the journal rows (`journal/embeddings.mmap`) and the only frame in the system: the ontology, the sphere and seed resolution (`Engine.frame()`) all read them. The per-block vectors are kept in `journal/blocks/<batch>.npz`.

## Configuration
`EMBEDDING_BACKEND` (`sentence-transformers` | `hashing`), `EMBEDDING_MODEL`, `MODEL_DIR` (`models`), `EMBEDDING_DEVICE` (auto|cpu|cuda; `sig/.env` pins `cuda` → `cuda:0` on the RTX 4060, shown in the UI header), `BLOCK_WEIGHTS`, `EMM_COMPONENT_WEIGHT`.

## Failure modes
Model load or encode errors become `embedding_failure` (batch FAILED before any state change). A fingerprint change becomes `representation_mismatch`.

## Invariants
Unit norm for all rows and blocks; `dim = 3·block_dim`; deterministic output for identical input; the fingerprint changes whenever any composition choice changes.

## Testing requirements
`tests/test_canonical_embedding.py::test_embedding_contract`, run for both the hashing and the real model (`model` marker). It checks shape, dtype, norms, stability, direction separation and cross-scope similarity. `::test_fingerprint_changes_with_representation_choices`.

## Integration points
Vectors go to the ontology (SDD 07) and to `journal/embeddings.mmap`. `encode_query` feeds seed resolution (SDD 10), and `encoder.embedder.embed` feeds the naive text-NN baseline.

## Current implementation status
Implemented. The model runs on `cuda:0`. The first-call load ≈ 10 s is import and weight loading, not inference; encoding the 28 demo insights takes ≈ 0.03 s afterwards.
