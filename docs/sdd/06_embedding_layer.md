# SDD 06 — Embedding layer

## Purpose
Map canonical insights to reproducible vectors in a structured semantic space, using a local embedding model and a model-independent `InsightEncoder` with a tripartite scope/target/phenomenon composition.

## Scope
`ltir/encoder.py`: `TextEmbedder` protocol, `SentenceTransformerEmbedder`, `HashingEmbedder`, `InsightEncoder`, `model_folder()`; `EmbeddingSpec` (in `models.py`).

## Inputs
`list[CanonicalInsight]` (SDD 05). For queries: recognised scope text, target text, signed components and the raw question text.

## Outputs
`encode()` → `{"vector": (n, 3d), "scope": (n, d), "target": (n, d), "phenomenon": (n, d)}`, all `float32`, all rows unit L2 norm. `encode_query()` → `(3d,)`. `spec` → `EmbeddingSpec`.

## Dependencies
sentence-transformers loading the bundled folder `model_folder(config)` = `MODEL_DIR/<last path segment of EMBEDDING_MODEL>` offline: `sig/models/Qwen3-Embedding-0.6B` (default; pinned revision `97b0c614…`, fetched once by `scripts/download_model.py`, 1.19 GB bf16) or `sig/models/paraphrase-multilingual-MiniLM-L12-v2` (alternative). If the folder is missing it falls back to resolving the name and caching under `MODEL_DIR`. No hosted embedding service.

## Algorithm
```text
s = E(scope text)                                   (unit)
t = E(target text)                                  (unit)
p = normalize( Σ_k c_k · E(label_k) )               (unit; c_k = signed component, SDD 05)
v = normalize( [ w_s·s ; w_t·t ; w_p·p ] )          w = BLOCK_WEIGHTS = (0.45, 0.55, 1.0)
```
* **Matryoshka truncation.** With `EMBEDDING_TRUNCATE_DIM = 384` the model returns the first 384 of its 1024 dimensions; every row is then **re-normalised** to unit length (`l2_normalize` in `SentenceTransformerEmbedder._encode`) — a slice of a unit vector is shorter than 1, and the cosine decomposition of SDD 16 §11 assumes unit blocks. `d = 384` keeps `dim = 1152`.
* **Why a composed phenomenon block.** Sentence embeddings barely encode direction (MiniLM: cos("margin decreases strongly", "margin increases strongly") = 0.56). With signed composition, opposite shifts of the same metric give p·p′ = −1; scope never enters p.
* **Query instruction** (Qwen3 is instruction-aware). `embed_queries()` prefixes `Instruct: {EMBEDDING_QUERY_INSTRUCTION}\nQuery: `; `embed()` (documents, labels) does not. In `encode_query` the prefix applies **only** to the free question text — the stand-in for an unrecognised scope or target block, the empty-component fallback of p — and to the naive text-NN baseline (`qa.compute_baselines`). Recognised scope/target strings and every component label are embedded exactly as for documents: the signed composition needs the identical `E(label)` on both sides, and a recognised scope string is the documents' own vocabulary. This forgoes the instruction gain on structured seeds by design (`test_query_instruction_only_reaches_free_question_text`).
* Text embeddings are memoised per `(prompt, text)` and encoded in batches of `ENCODE_BATCH_SIZE` = 16 rows (bounds the transient VRAM peak on long documents: Qwen3 1.7 GB at 16 vs 3.4 GB at 64, and no slower).
* On CUDA the checkpoint dtype is used (`dtype="auto"`: bf16 for Qwen3, fp32 for MiniLM); on CPU fp32.

## Data contract — `EmbeddingSpec`
| Field | Value (default) |
|---|---|
| `model_id` | `Qwen/Qwen3-Embedding-0.6B` (or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `hashing-ngram-256` in tests) |
| `truncate_dim`, `query_instruction` | 384, `Instruct: Given a quantitative analysis question, retrieve relevant statistical subgroup patterns\nQuery: ` |
| `block_dim` / `dim` | 384 / 1152 |
| `dtype`, `normalization` | `float32`, `l2(block) -> weighted concat -> l2` |
| `block_weights`, `emm_component_weight` | (0.45, 0.55, 1.0), 0.5 |
| `canonical_version`, `representation_version` | `ltir-canon-3`, `ltir-rep-3` |
| `fingerprint` | sha1 of all of the above [:10] |

The spec is written to `state/representation.json` when the first batch commits and is carried in every pattern record. A different fingerprint fails ingest and queries with `representation_mismatch` (SDD 09, SDD 12); `python -m ltir migrate --yes` rebuilds an outdated workspace with the current code. The composite vectors are the journal rows (`journal/embeddings.mmap`) — the only frame in the system (`Engine.frame()`); the blocks are kept in `journal/blocks/<batch>.npz`.

## Device boundary
The embedder is the only model SIG loads in-process: `EMBEDDING_DEVICE` (`auto` | `cpu` | `cuda` | `cuda:0`); Qwen3 takes ≈ 1.15 GB VRAM resident and peaks at ≈ 1.7 GB (the first question embeds every pattern document for the naive baseline). The 26B–31B LLM is never loaded by SIG; it is reached only through `LLM_BASE_URL` (OpenRouter here; a local Ollama/LM Studio server must run with CPU or partial GPU offload on an 8 GB card). The web app logs free GPU memory after warm-up.

## Configuration
`EMBEDDING_BACKEND` (`sentence-transformers` | `hashing`), `EMBEDDING_MODEL`, `EMBEDDING_TRUNCATE_DIM`, `EMBEDDING_QUERY_INSTRUCTION`, `MODEL_DIR` (`models`), `EMBEDDING_DEVICE`, `BLOCK_WEIGHTS`, `EMM_COMPONENT_WEIGHT`. MiniLM: `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2`, `EMBEDDING_TRUNCATE_DIM=0`, `EMBEDDING_QUERY_INSTRUCTION=` (an empty value clears a string field) and `MIN_ASSIGN_THRESHOLD=0.55` — cosine thresholds belong to the embedder (SDD 07 §Calibration). Switching embedders changes the fingerprint; `python -m ltir migrate --yes` rebuilds the workspace.

## Failure modes
Model load or encode errors become `embedding_failure` (batch FAILED before any state change). A fingerprint change becomes `representation_mismatch`.

## Invariants
Unit norm for all rows and blocks (after truncation too); `dim = 3·block_dim`; deterministic output for identical input; the instruction never reaches labels or recognised scope/target strings; the fingerprint changes whenever any composition choice changes.

## Testing requirements
`tests/test_canonical_embedding.py::test_embedding_contract` (hashing and the real model, `model` marker: shape, dtype, unit norms, stability, direction separation, cross-scope similarity), `::test_fingerprint_changes_with_representation_choices` (incl. `truncate_dim`, `query_instruction`), `::test_query_instruction_only_reaches_free_question_text`. `scripts/compare_embedders.py` measures candidate models.

## Integration points
Vectors go to the ontology (SDD 07) and to `journal/embeddings.mmap`. `encode_query` feeds seed resolution (SDD 10); `embedder.embed_queries` the naive text-NN baseline.

## Current implementation status
Implemented. `scripts/compare_embedders.py` on the demo, then a same-domain batch (demo seed 8) and `housing.csv` (RTX 4060; each model at its calibrated `MIN_ASSIGN_THRESHOLD`):

| metric | MiniLM-L12 (384) | Qwen3-0.6B → 384 | Qwen3-0.6B (1024) |
|---|---|---|---|
| direction cosine (must be < 0) | −0.329 | −0.329 | −0.329 |
| cross-scope, same phenomenon | 0.955 | 0.971 | 0.969 |
| entity (same scope, other phenomenon) | 0.490 | 0.575 | 0.533 |
| label cosine `E(discount)·E(margin)` | 0.432 | 0.707 | 0.683 |
| transversal MRR / recall@3 / recall@5 | 0.567 / 0.333 / 0.729 | **0.581 / 0.521 / 0.729** | 0.581 / – / 0.729 |
| naive text-NN MRR | 0.194 | 0.319 | 0.310 |
| attractors on the demo | 7 | 4 | 4 |
| `MIN_ASSIGN_THRESHOLD` | 0.55 | 0.75 | 0.75 |
| same-domain batch: orphan rate / smallest alignment | 0 / 0.842 | 0 / 0.788 | 0 / 0.761 |
| housing batch: orphan rate (Qwen3 at 0.55 in brackets) | 1.0 | 1.0 (0.53, largest alignment 0.71) | 1.0 |
| RELATED_TO edges retail ↔ housing / evidence items from the other domain (6 questions) | 0 / 0 | 1 / 0 | 3 / 0 |
| peak VRAM over the run | 493 MB | 1,720 MB | 1,721 MB |

Qwen3 keeps the direction contract (the composition fixes it at −0.33 for any model), ranks analogues higher at small k and makes the naive baseline much stronger. Its cosines between unrelated texts sit higher (label cosine 0.71 vs 0.43), with two consequences. Two unique singletons of the demo (the EU∧phones correlation break and the EU∧laptops∧retail margin contrast) share one attractor, while the three recurring mechanisms keep their own (SDD 07). And MiniLM's assignment threshold 0.55 lets half of an unrelated dataset join retail attractors, so Qwen3 runs at 0.75 (SDD 07 §Calibration). Truncation to 384 costs nothing measurable against 1024 here.
