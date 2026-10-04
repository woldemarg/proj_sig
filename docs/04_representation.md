# 4. Representation — canonical text and insight vectors

> **In one paragraph.** Each insight is written down twice, for two different readers. The *embedding inputs* — the scope string, the target name and a list of signed components such as `("discount", +2.21)` — define the vector and contain no measured numbers and no prose. The *readable text* — a Markdown document and phrase helpers shared with the LLM prompt — is for people and the language model; it enters the vector only in one rare fallback (components that cancel out). The encoder embeds scope and target as sentences and builds the phenomenon as a signed, magnitude-weighted sum of metric-name embeddings, so "margin up" and "margin down" point in opposite directions. The three unit blocks are weighted and concatenated into one 1152-d unit vector, and a fingerprint of the model and composition choices guards the workspace against mixing incompatible vectors.

**Code** `attractor_topology/canonical.py`, `attractor_topology/encoder.py`, `attractor_topology/models.py` (`CanonicalInsight`, `EmbeddingSpec`), `insight_contracts/text.py` (the readable phrases), `insight_graph_service/core/model_store.py` (the model on disk) · **Tests** `tests/test_canonical_embedding.py`, `tests/test_model_store.py` · **Previous** [3. Insights](03_insights.md) · **Next** [5. Latent anchors](05_latent_anchors.md)

---

## 4.1 Two text contracts

Text appears in three tiers. Only the third shapes similarity — apart from one fallback: when an insight's components cancel out, its readable phenomenon sentence stands in for the phenomenon block ([4.3](#43-the-tripartite-vector)).

| Tier | What | Where it lives | Who reads it |
|---|---|---|---|
| record | the typed `Insight` as JSON — numbers, selector strings, provenance; the single source of truth ([3.1](03_insights.md#31-the-insight-record)) | journal, snapshot, Neo4j | every module |
| rendering | text derived from the record: the canonical document, headlines and labels, hover text, the evidence prompt and the evidence-only summary | stored next to the record (the document) or computed on demand | people, the LLM, the naive text baseline |
| embedding input | three short strings per insight — scope text, target text and the component labels | the vector in `journal/embeddings.mmap`, its blocks in `journal/blocks/*.npz` | retrieval, the ontology, the sphere |

The canonical document is also embedded once at ingest (`document` in the batch's blocks file; a pattern without a stored document vector is embedded once, on the first question, [7.6](07_question_answering.md#76-baselines)), but only for the naive text baseline and the benchmark's `text_nn` ranker: that vector never enters the ontology or retrieval. Keeping the tiers apart is what lets the prompt be rewritten for readability without moving a single vector, and keeps measured magnitudes out of the embedded strings, where their tokenisation would only add noise. Renderings come in two flavours: the **LLM serializer** (ASCII prose with rounded numbers, [7.4](07_question_answering.md#74-the-evidence-object)) and **visual labels** (compact labels for graph nodes and cards — the headline is ASCII, the UI labels use `·` and arrows, [8.5](08_interface.md#85-text-shown-to-people)).

## 4.2 The canonical form

`canonicalize(insight, config, dataset_rows) → CanonicalInsight(insight_id, version, target, scope, scope_sentence, phenomenon, covariance, confounders, support, components)` (`config` a `TopologyConfig`), version `ltir-canon-4` (`CANONICAL_VERSION`). The readable phrases come from the shared kernel (`insight_contracts/text.py`), so the canonical form, the evidence prompt and the miner's LLM context read alike; which shifts and which correlation change belong to the phenomenon is decided by `PhenomenonThresholds` (`insight_contracts/insight.py`).

**Embedding inputs.**

| Field | Rule | Running example |
|---|---|---|
| `scope` | `attribute = value` joined by `; `, raw column names, values verbatim, in the record's condition order (sorted by attribute, then value) | `category = phones; region = US` |
| `target` | `humanize(target)` (underscores → spaces) | `discount` |
| `components` | `(humanize(metric), signed robust z)` for the target and every shift with `|z| ≥ MIN_COMPONENT_Z` (0.5) — none for a covariance insight, whose median shift failed the shift test; plus `("correlation between a and b", sign · EMM_COMPONENT_WEIGHT · emm_score / WEIGHT_EMM_REF)` when the correlation change is material | `[("discount", 2.21), ("margin", −1.10)]` |

The target and the shifts above `MIN_COMPONENT_Z` are `PhenomenonThresholds.phenomenon_shifts` (a shift is `material` from `|z| ≥ MIN_COMPONENT_Z`). The correlation change is material when a covariance pair exists and the insight is covariance-typed or has `emm_score ≥ MIN_EMM_SCORE` (`PhenomenonThresholds.has_material_covariance`). Its sign is `−1` when the correlation reverses (both `|C_ij|` and `|C_S,ij|` above 0.1 with opposite signs — this test comes first, so `−0.2 → +0.9` is a reversal), otherwise `+1` when `|corr|` grows and `−1` when it does not. With `EMM_COMPONENT_WEIGHT = 0.5`, an EMM score of 0.14 becomes a component of 0.875, comparable to a 0.9 sd shift. The strings carry names and condition values only: no magnitude, median or p-value ever reaches an embedded string — the magnitude lives in the coefficient. (A column name or a condition value can itself contain digits: `Store = 12`, `median_income_band = q4`.) Structural predicates (scope) and statistical behaviour (phenomenon) never share a string.

**Readable text.** Fixed number rules (`format_value`, `format_p`): values below 1,000 with 4 significant digits (`.4g`: trailing zeros dropped, so `19.0` prints as `19`, and values below 0.0001 switch to exponent notation), values from 1,000 rounded to integers with thousands separators; shifts `±x.xx sd`; correlations `±0.xx`; p-values bucketed as `< 0.001`, `< 0.01`, `< 0.05` or two decimals; shares as one-decimal percentages. Magnitude words: `mild < 1 ≤ moderate < 2 ≤ strong < 3 ≤ extreme`. The text is ASCII as long as the data is: column names, values and confounder strings pass through verbatim.

| Field | Built by | Running example |
|---|---|---|
| `scope_sentence` | `describe_scope`: `a is x and b is y`; three or more: `a is x, b is y, and c is z` | `category is phones and region is US` |
| `phenomenon` | `describe_shift` per phenomenon shift, closed by the correlation phrase when material; for a covariance insight only the correlation phrase, followed by `no validated median shift` | `discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall); margin: moderate decrease, -1.10 sd (median 14.75 vs 19.45 overall)` |
| `covariance` | `strongest change: <describe_covariance> (divergence score x.xx)` or `no correlation pair (…)` | `strongest change: correlation between delivery days and return rate weakens from +0.88 overall to +0.73 in the subgroup (divergence score 0.02)` |
| `confounders` | the EDA driver strings, or `none detected` | `none detected` |
| `support` | `n rows (share of N); <validation>` — `bootstrap stability s; adjusted p <bucket>`, or for a covariance insight `correlation change (divergence score x.xx >= MIN_EMM_SCORE); no median test` | `438 rows (8.8% of 5,000); bootstrap stability 0.97; adjusted p < 0.001` |

`CanonicalInsight.document()` renders them as one Markdown block — the text people read in the UI drawer and the text the naive text-retrieval baseline embeds:

```text
### Subgroup finding P-bc4657a04746
* Scope: category is phones and region is US
* Target metric: discount
* Observed shift: discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall); margin: moderate decrease, -1.10 sd (median 14.75 vs 19.45 overall)
* Metric relationships: strongest change: correlation between delivery days and return rate weakens from +0.88 overall to +0.73 in the subgroup (divergence score 0.02)
* Confounders: none detected
* Validation: 438 rows (8.8% of 5,000); bootstrap stability 0.97; adjusted p < 0.001
```

A covariance insight reads differently — it cites only the correlation change, and its validation is that change:

```text
### Subgroup finding P-7b083e5ffddc
* Scope: category is tablets and channel is online
* Target metric: delivery days
* Observed shift: correlation between delivery days and discount strengthens from -0.01 overall to +0.65 in the subgroup; no validated median shift
* Metric relationships: strongest change: correlation between delivery days and discount strengthens from -0.01 overall to +0.65 in the subgroup (divergence score 0.14)
* Confounders: none detected
* Validation: 515 rows (10.3% of 5,000); correlation change (divergence score 0.14 >= 0.08); no median test
```

## 4.3 The tripartite vector

With `E(·)` the unit sentence embedding of [4.4](#44-the-embedding-model):

```text
s = E(scope text)                t = E(target text)                p = normalize( Σ_k coef_k · E(label_k) )
                                                                       (fallback: E(phenomenon sentence) if the sum is 0)
v = normalize( [ w_s · s ; w_t · t ; w_p · p ] )        (w_s, w_t, w_p) = BLOCK_WEIGHTS = (0.45, 0.55, 1.0)        dim = 3 · 384 = 1152
```

Because the three blocks are unit vectors, the cosine of two insight vectors decomposes exactly:

```text
cos(v, v′) = ( w_s² s·s′ + w_t² t·t′ + w_p² p·p′ ) / (w_s² + w_t² + w_p²)  =  0.135 · s·s′ + 0.201 · t·t′ + 0.664 · p·p′
```

The phenomenon carries two thirds of the similarity, the target one fifth, the scope one seventh. Direction lives in the sign of the coefficients: two insights with exactly opposite coefficients have `p·p′ = −1` (opposite signs with other magnitudes give a strongly negative cosine), while their sentence embeddings would be nearly identical (cosine 0.56 between "margin decreases strongly" and "margin increases strongly"). The label vocabulary is small and shared — humanised metric names and `correlation between <a> and <b>` — so insights with the same mechanism land on the same phenomenon direction whatever their scope.

> **Running example.** Cosines of `P-bc4657a04746` (phones ∧ US: discount +2.21, margin −1.10) with three other demo insights:
>
> | Other insight | `s·s′` | `t·t′` | `p·p′` | cosine |
> |---|---|---|---|---|
> | tablets ∧ retail ∧ APAC: discount +2.09, margin −1.00 (no shared condition) | 0.760 | 1.000 | 1.000 | **0.967** |
> | phones ∧ online ∧ US: discount +2.25, margin −1.10 (a refinement) | 0.908 | 1.000 | 1.000 | 0.988 |
> | laptops ∧ EU: margin +1.13 | 0.674 | 0.707 | 0.286 | 0.423 |
>
> The scope-disjoint analogue is almost as close as the refinement — the property the latent plane builds on.

The fallback fires whenever the component sum has a norm below `1e-9` — also when components cancel.

**Queries** use the same composition (`encode_query`): recognised scope conditions and metrics give the scope and target strings, and each named metric becomes a component `±2.0` in the question's direction — a relationship question about two or more metrics gets one correlation component instead ([7.1](07_question_answering.md#71-from-question-to-query)). Where a block has nothing recognised — no condition, no metric, no direction word — the question text stands in for it. A query scope lists conditions in the order they were recognised and a multi-metric target reads `a; b`, so the query strings are close to, but not always identical with, the documents' strings.

## 4.4 The embedding model

| Property | Value |
|---|---|
| model | `Qwen/Qwen3-Embedding-0.6B` (`QWEN_MODEL`) at the pinned revision `QWEN_REVISION`, 1.19 GB bf16 — the only embedder (`attractor_topology/encoder.py::SentenceTransformerEmbedder`) |
| on disk | `model_folder(MODEL_DIR)` = `MODEL_DIR/Qwen3-Embedding-0.6B/`, loaded offline (`HF_HUB_OFFLINE=1` is set by default, so nothing is fetched at run time); loading needs its `modules.json` |
| provisioning | `python -m insight_graph_service.core.model_store [MODEL_DIR] [--seed DIR]` (without a folder: the service's `MODEL_DIR`, read from the environment, then `.env`) checks the folder file by file against a manifest of sizes and SHA-256 values at the pinned revision: a copy that verifies is kept, otherwise a verified seed is copied, otherwise the pinned revision is downloaded; it exits non-zero unless the result verifies. The graph service checks presence and sizes against the same manifest before it opens the workspace or loads the model, and refuses to start (exit code 3) with a message naming that command |
| Matryoshka truncation | `TRUNCATE_DIM = 384` keeps the first 384 of 1024 dimensions; every row is then **re-normalised** (`E(x) = normalize(f(x)_{:384})`) — a slice of a unit vector is shorter than 1, and the decomposition above assumes unit blocks |
| query instruction | `embed_queries` prefixes `Instruct: {QUERY_TASK}\nQuery: `; `embed` (documents, labels, recognised query strings) does not |
| dtype and batching | the checkpoint dtype on every device (`dtype="auto"`: bf16), recorded as `compute_dtype`; batches of `ENCODE_BATCH_SIZE = 16` rows; every text embedding is memoised per `(prompt, text)` for the life of the embedder (one per engine) |
| device | `EMBEDDING_DEVICE` (`auto` · `cpu` · `cuda` · `cuda:0`; the containers use `cpu`); the model takes ≈ 1.15 GB of VRAM resident and peaks at ≈ 1.7 GB; on the CPU the encoder leaves half of the cores to the rest of the batch |

**Where the instruction goes.** Qwen3 is instruction-aware, but the prefix is applied **only** to free question text — the stand-in for an unrecognised scope or target block and for an empty component sum — and to the naive text baseline. Recognised scope and target strings and every component label are embedded exactly as for documents, because the signed composition needs the identical `E(label)` on both sides to keep its exact ±1 geometry. This forgoes the instruction gain on structured queries by design.

**Device boundary.** The embedder is the only model SIG's services load. The language model is never loaded in-process; the narrator reaches it only through `LLM_BASE_URL` (by default the LLM model broker, [12.4](12_architecture.md#124-services-and-contracts); behind it a hosted endpoint, or a local server that must run with CPU or partial GPU offload on a small consumer GPU). The graph service logs the device and free GPU memory after warm-up. Batches of 16 bound the transient peak: the embedding stage also embeds every new pattern's full document for the text baseline ([7.6](07_question_answering.md#76-baselines)), which peaked at 3.4 GB with batches of 64 and runs faster at 16.

**The test double.** Tests inject `tests/doubles.py::HashingEmbedder` through `Engine(settings, embedder=…)`: a deterministic stand-in (word and character-trigram hashing into 384 dimensions, the production block width; model id `hashing-ngram-384`) that needs no model. It maps text without any `[a-z0-9]` character to a zero vector, so it is for tests, not for data. Tests marked `model` use the real checkpoint.

## 4.5 Embedder comparison

The measurement that chose Qwen3: Qwen3-Embedding-0.6B against `paraphrase-multilingual-MiniLM-L12-v2` (native 384-d) on the demo, then a same-domain second batch (demo seed 8) and `housing.csv`, each model at its calibrated assignment threshold:

| Metric | MiniLM-L12 (384) | Qwen3 → 384 (SIG) | Qwen3 (1024) |
|---|---|---|---|
| direction cosine (must be < 0) | −0.329 | −0.329 | −0.329 |
| cross-scope, same phenomenon | 0.955 | 0.971 | 0.969 |
| entity (same scope, other phenomenon) | 0.490 | 0.575 | 0.533 |
| label cosine `E(discount)·E(margin)` | 0.432 | 0.707 | 0.683 |
| transversal MRR / recall@3 / recall@5 | 0.567 / 0.333 / 0.729 | **0.581 / 0.521 / 0.729** | 0.581 / – / 0.729 |
| naive text-NN MRR | 0.194 | 0.321 | 0.311 |
| anchors on the demo | 7 | 4 | 4 |
| `MIN_ASSIGN_THRESHOLD` | 0.55 | 0.75 | 0.75 |
| same-domain batch: orphan rate / smallest alignment | 0 / 0.842 | 0 / 0.788 | 0 / 0.761 |
| housing batch: orphan rate (Qwen3 at 0.55 in brackets) | 1.0 | 1.0 (0.53; largest alignment 0.71) | 1.0 |
| RELATED_TO edges retail ↔ housing / evidence items from the other domain (6 questions) | 0 / 0 | 1 / 0 | 3 / 0 |
| peak VRAM over the run | 493 MB | 1,720 MB | 1,720 MB |

Reading: the composition fixes the direction contract (−0.33 for any model). Qwen3 ranks analogues higher at small `k` and makes the naive baseline much stronger. Its cosines between unrelated texts sit higher (label cosine 0.71 vs 0.43), with two consequences: the two one-off phenomena of the demo share one anchor ([5.13](05_latent_anchors.md#513-guarantees-and-measured-behaviour)), and MiniLM's assignment threshold would let half of an unrelated dataset join retail anchors, so Qwen3 runs at 0.75 ([5.10](05_latent_anchors.md#510-calibration-per-embedder)). Truncation to 384 costs nothing measurable against 1024.

## 4.6 Representation identity and versions

`InsightEncoder.spec → EmbeddingSpec`:

| Field | Value |
|---|---|
| `model_id` | `Qwen/Qwen3-Embedding-0.6B` (the test double: `hashing-ngram-384`) |
| `model_revision` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` — `QWEN_REVISION`, the revision `model_store` provisions and verifies (the test double: `""`) |
| `truncate_dim`, `query_instruction` | 384, `Instruct: Given a quantitative analysis question, retrieve relevant statistical subgroup patterns\nQuery: ` |
| `block_dim`, `dim` | 384, 1152 |
| `dtype`, `compute_dtype`, `normalization` | `float32` (storage), `bfloat16` (the model's computation), `l2(block) -> weighted concat -> l2` |
| `block_weights`, `emm_component_weight` | (0.45, 0.55, 1.0), 0.5 |
| `min_component_z`, `min_emm_score`, `weight_emm_ref` | 0.5, 0.08, 0.08 — the `PhenomenonThresholds` that decide which components exist and how large the correlation component is |
| `canonical_version`, `representation_version` | `ltir-canon-4`, `ltir-rep-3` (`CANONICAL_VERSION`, `REPRESENTATION_VERSION` in `attractor_topology/models.py`) |
| `fingerprint` | `sha1` of all of the above, first 10 hex (`3d08cee697` for the defaults) |

The spec is written to `state/representation.json` when the first batch commits; every pattern record carries `{fingerprint, model_id, dim, representation_version}`. Two checks refuse a mismatch instead of comparing vectors from different spaces:

* `Workspace.check_representation` compares the fingerprint, at ingestion (the batch fails with `representation_mismatch`) and for every question (`POST /api/search` answers 409 with `code: representation_mismatch`).
* `Workspace.check_versions` compares the two version strings without loading a model when the engine opens the workspace; journals of another version make the graph service start degraded ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).

`CANONICAL_VERSION` covers [4.2](#42-the-canonical-form); `REPRESENTATION_VERSION` covers [4.3](#43-the-tripartite-vector). Renderings can change without a bump; anything that changes an embedded string, a coefficient or the composition must bump a version.

**What the fingerprint covers.** Every input that shapes a stored vector: the model and its checkpoint revision, the dtype it computes in and the storage dtype, the truncation, the query instruction, the block weights and the correlation component weight, the normalisation, the three canonicalisation settings that decide the components, and both versions. Changing any of them on an existing workspace is refused with `representation_mismatch`; the workspace is reset (`POST /api/reset`) and the data ingested again ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)). The device is not part of it: the model runs in its checkpoint dtype on CPU and GPU alike, so a workspace moves between them. Renderings — documents, headlines, the prompt — are not part of it either; the phenomenon sentence of the fallback is covered by `CANONICAL_VERSION`.

## 4.7 Configuration

| Parameter | Default | Changes the fingerprint |
|---|---|---|
| `MODEL_DIR` | `models` | no (where the checkpoint is read; its revision is the pinned constant) |
| `EMBEDDING_DEVICE` | `auto` | no (the model runs in its checkpoint dtype on every device) |
| `BLOCK_WEIGHTS`, `EMM_COMPONENT_WEIGHT` | (0.45, 0.55, 1.0), 0.5 | yes |
| `MIN_COMPONENT_Z`, `MIN_EMM_SCORE`, `WEIGHT_EMM_REF` | 0.5, 0.08, 0.08 | yes (they decide the components) |

The model, its revision, the truncation and the query task are constants of `attractor_topology/encoder.py` (`QWEN_MODEL`, `QWEN_REVISION`, `TRUNCATE_DIM`, `QUERY_TASK`), not settings: changing one is a code change that changes the fingerprint, and the checkpoint manifest in `insight_graph_service/core/model_store.py` changes with the revision.

## 4.8 Guarantees and failure modes

* Embedding inputs contain only conditions (scope) and metric names (target, labels): no measured numbers, no prose — except the fallback phenomenon sentence when components cancel. The readable text is a pure function of the insight and the config, and ASCII for ASCII data.
* Every row and every block is unit-norm, also after truncation (the test double excepted for text without `[a-z0-9]`); `dim = 3 · block_dim`; identical input gives identical output.
* The query instruction never reaches component labels or recognised scope/target strings.
* The fingerprint changes whenever any input that shapes a stored vector changes ([4.6](#46-representation-identity-and-versions)).
* Model load or encode errors become `embedding_failure` (the batch fails before the journal or the ontology changes; the dataset folder's `profile.json` and `rejections.json` are already written); a fingerprint change becomes `representation_mismatch`. A model folder that is missing or incomplete stops the graph service at its start, before the workspace is opened.

Tests: `test_canonical_sections_are_separate`, `test_embedding_labels_carry_no_numbers`, `test_text_number_rules`, `test_covariance_canonical_component`, `test_embedding_contract` (the test double and the real model: shape, dtype, unit norms, stability, direction separation, cross-scope similarity), `test_fingerprint_changes_with_representation_choices`, `test_query_instruction_only_reaches_free_question_text`; `tests/test_model_store.py` (a verified seed is copied, a same-size corrupt file is caught by its hash and repaired from the seed while the service's size check passes it; the local checkpoint matches the pinned manifest).
