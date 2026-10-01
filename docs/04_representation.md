# 4. Representation — canonical text and insight vectors

> **In one paragraph.** Each insight is written down twice, for two different readers. The *embedding inputs* — the scope string, the target name and a list of signed components such as `("discount", +2.21)` — define the vector and contain no measured numbers and no prose. The *readable text* — a Markdown document and phrase helpers shared with the LLM prompt — is for people and the language model and never enters the vector. The encoder embeds scope and target as sentences and builds the phenomenon as a signed, magnitude-weighted sum of metric-name embeddings, so "margin up" and "margin down" point in opposite directions. The three unit blocks are weighted and concatenated into one 1152-d unit vector, and a fingerprint of the model and composition choices guards the workspace against mixing incompatible vectors.

**Code** `ltir/canonical.py`, `ltir/encoder.py`, `ltir/models.py` (`CanonicalInsight`, `EmbeddingSpec`) · **Tests** `tests/test_canonical_embedding.py` · **Previous** [3. Insights](03_insights.md) · **Next** [5. Latent anchors](05_latent_anchors.md)

---

## 4.1 Two text contracts

Text appears in three tiers. Only the third shapes similarity.

| Tier | What | Where it lives | Who reads it |
|---|---|---|---|
| record | the typed `Insight` as JSON — numbers, selector strings, provenance; the single source of truth ([3.1](03_insights.md#31-the-insight-record)) | journal, snapshot, Neo4j | every module |
| rendering | text derived from the record: the canonical document, headlines and labels, hover text, the evidence prompt and the evidence-only summary | stored next to the record (the document) or computed on demand | people, the LLM, the naive text baseline |
| embedding input | three short strings per insight — scope text, target text and the component labels | the vector in `journal/embeddings.mmap`, its blocks in `journal/blocks/*.npz` | retrieval, the ontology, the sphere |

Keeping the tiers apart is what lets the prompt be rewritten for readability without moving a single vector, and keeps measured magnitudes out of the embedded strings, where their tokenisation would only add noise. Renderings come in two flavours: the **LLM serializer** (ASCII prose with rounded numbers, [7.4](07_question_answering.md#74-the-evidence-object)) and **visual labels** (compact labels for graph nodes and cards — the headline is ASCII, the UI labels use `·` and arrows, [8.5](08_interface.md#85-text-shown-to-people)).

## 4.2 The canonical form

`canonicalize(insight, config, dataset_rows) → CanonicalInsight(insight_id, version, target, scope, scope_sentence, phenomenon, covariance, confounders, support, components)`, version `ltir-canon-4`.

**Embedding inputs.**

| Field | Rule | Running example |
|---|---|---|
| `scope` | `attribute = value` joined by `; `, raw column names, values verbatim, in closed-intent order | `category = phones; region = US` |
| `target` | `humanize(target)` (underscores → spaces) | `discount` |
| `components` | `(humanize(metric), signed robust z)` for the target and every shift with `|z| ≥ MIN_COMPONENT_Z` (0.5) — none for a covariance insight, whose median shift failed the shift test; plus `("correlation between a and b", sign · EMM_COMPONENT_WEIGHT · emm_score / WEIGHT_EMM_REF)` when the correlation change is material | `[("discount", 2.21), ("margin", −1.10)]` |

The correlation change is material for covariance-typed insights and whenever `emm_score ≥ MIN_EMM_SCORE` (`has_material_covariance`). Its sign is `−1` when the correlation reverses (both `|C_ij|` and `|C_S,ij|` above 0.1 with opposite signs — this test comes first, so `−0.2 → +0.9` is a reversal), otherwise `+1` when `|corr|` grows and `−1` when it does not. With `EMM_COMPONENT_WEIGHT = 0.5`, an EMM score of 0.14 becomes a component of 0.875, comparable to a 0.9 sd shift. The strings carry names and condition values only: no magnitude, median or p-value ever reaches an embedded string — the magnitude lives in the coefficient. (A column name or a condition value can itself contain digits: `Store = 12`, `median_income_band = q4`.) Structural predicates (scope) and statistical behaviour (phenomenon) never share a string.

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

The phenomenon carries two thirds of the similarity, the target one fifth, the scope one seventh. Direction lives in the sign of the coefficients: two insights on the same metrics with opposite signs have `p·p′ = −1`, while their sentence embeddings would be nearly identical (cosine 0.56 between "margin decreases strongly" and "margin increases strongly"). The label vocabulary is small and shared — humanised metric names and `correlation between <a> and <b>` — so insights with the same mechanism land on the same phenomenon direction whatever their scope.

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

**Queries** use the same composition (`encode_query`): recognised scope conditions and metrics give the scope and target strings, and each named metric becomes a component `±2.0` in the question's direction ([7.1](07_question_answering.md#71-from-question-to-query)). Where a block has nothing recognised — no condition, no metric, no direction word — the question text stands in for it. A query scope lists conditions in the order they were recognised and a multi-metric target reads `a; b`, so the query strings are close to, but not always identical with, the documents' strings.

## 4.4 The embedding model

| Property | Value |
|---|---|
| default model | `Qwen/Qwen3-Embedding-0.6B`, loaded offline from `models/Qwen3-Embedding-0.6B/` (pinned revision, fetched once by `scripts/download_model.py`, 1.19 GB bf16) |
| alternative | `paraphrase-multilingual-MiniLM-L12-v2` (native 384-d) from `models/paraphrase-multilingual-MiniLM-L12-v2/`; the hashing backend (`EMBEDDING_BACKEND=hashing`) is a deterministic offline stand-in for tests |
| folder rule | `model_folder(config) = MODEL_DIR / <last path segment of EMBEDDING_MODEL>`; if the folder is missing, the name is resolved through the Hugging Face cache under `MODEL_DIR` — offline (`HF_HUB_OFFLINE=1` is set by default), so only an existing cache resolves |
| Matryoshka truncation | `EMBEDDING_TRUNCATE_DIM = 384` keeps the first 384 of 1024 dimensions; every row is then **re-normalised** (`E(x) = normalize(f(x)_{:384})`) — a slice of a unit vector is shorter than 1, and the decomposition above assumes unit blocks |
| query instruction | `embed_queries` prefixes `Instruct: {EMBEDDING_QUERY_INSTRUCTION}\nQuery: `; `embed` (documents, labels, recognised query strings) does not |
| dtype and batching | the checkpoint dtype on every device (`dtype="auto"`: bf16 for Qwen3, fp32 for MiniLM), recorded as `compute_dtype`; batches of `ENCODE_BATCH_SIZE = 16` rows; every text embedding is memoised per `(prompt, text)` for the life of the process |
| device | `EMBEDDING_DEVICE` (`auto` · `cpu` · `cuda` · `cuda:0`); Qwen3 takes ≈ 1.15 GB of VRAM resident and peaks at ≈ 1.7 GB |

**Where the instruction goes.** Qwen3 is instruction-aware, but the prefix is applied **only** to free question text — the stand-in for an unrecognised scope or target block and for an empty component sum — and to the naive text baseline. Recognised scope and target strings and every component label are embedded exactly as for documents, because the signed composition needs the identical `E(label)` on both sides to keep its exact ±1 geometry. This forgoes the instruction gain on structured queries by design.

**Device boundary.** The embedder is the only model SIG loads. The language model is never loaded in-process; it is reached only through `LLM_BASE_URL` (a hosted endpoint, or a local server that must run with CPU or partial GPU offload on an 8 GB card). The web app logs the device and free GPU memory after warm-up. Batches of 16 bound the transient peak: the first question embeds every pattern document for the text baseline ([7.6](07_question_answering.md#76-baselines)), which peaked at 3.4 GB with batches of 64 and runs faster at 16.

The hashing backend is a deterministic stand-in (word and character-trigram hashing into 256 dimensions); it maps text without any `[a-z0-9]` character to a zero vector, so it is for tests, not for data.

## 4.5 Embedder comparison

`scripts/compare_embedders.py` on the demo, then a same-domain second batch (demo seed 8) and `housing.csv`, each model at its calibrated assignment threshold (RTX 4060):

| Metric | MiniLM-L12 (384) | Qwen3 → 384 (default) | Qwen3 (1024) |
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

Reading: the composition fixes the direction contract (−0.33 for any model). Qwen3 ranks analogues higher at small `k` and makes the naive baseline much stronger. Its cosines between unrelated texts sit higher (label cosine 0.71 vs 0.43), with two consequences: the two one-off phenomena of the demo share one anchor ([5.12](05_latent_anchors.md#512-guarantees-and-measured-behaviour)), and MiniLM's assignment threshold would let half of an unrelated dataset join retail anchors, so Qwen3 runs at 0.75 ([5.10](05_latent_anchors.md#510-calibration-per-embedder)). Truncation to 384 costs nothing measurable against 1024.

Switching to MiniLM: `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2`, `EMBEDDING_TRUNCATE_DIM=0`, `EMBEDDING_QUERY_INSTRUCTION=` (an empty value clears a text setting) and `MIN_ASSIGN_THRESHOLD=0.55`; then migrate the workspace ([6.5](06_graph_and_storage.md#65-versions-and-migration)).

## 4.6 Representation identity and versions

`InsightEncoder.spec → EmbeddingSpec`:

| Field | Default value |
|---|---|
| `model_id` | `Qwen/Qwen3-Embedding-0.6B` (or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `hashing-ngram-256`) |
| `model_revision` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` — read from `models/<folder>/REVISION` (written by `scripts/download_model.py`); `""` when the folder has none |
| `truncate_dim`, `query_instruction` | 384, `Instruct: Given a quantitative analysis question, retrieve relevant statistical subgroup patterns\nQuery: ` |
| `block_dim`, `dim` | 384, 1152 |
| `dtype`, `compute_dtype`, `normalization` | `float32` (storage), `bfloat16` (the model's computation), `l2(block) -> weighted concat -> l2` |
| `block_weights`, `emm_component_weight` | (0.45, 0.55, 1.0), 0.5 |
| `min_component_z`, `min_emm_score`, `weight_emm_ref` | 0.5, 0.08, 0.08 — the canonicalisation settings that decide which components exist and how large the correlation component is |
| `canonical_version`, `representation_version` | `ltir-canon-4`, `ltir-rep-3` |
| `fingerprint` | `sha1` of all of the above, first 10 hex (`3d08cee697` for the defaults) |

The spec is written to `state/representation.json` when the first batch commits; every pattern record carries `{fingerprint, model_id, dim, representation_version}`. Two checks refuse a mismatch instead of comparing vectors from different spaces: `check_representation` compares the fingerprint (ingestion and queries; HTTP 409 on `/api/query`), and `check_versions` compares the two version strings without loading a model (graph rebuilds). `CANONICAL_VERSION` covers [4.2](#42-the-canonical-form); `REPRESENTATION_VERSION` covers [4.3](#43-the-tripartite-vector). Renderings can change without a bump; anything that changes an embedded string, a coefficient or the composition must bump a version.

**What the fingerprint covers.** Every input that shapes a stored vector: the model and its checkpoint revision, the dtype it computes in, the truncation, the query instruction, the block weights, the three canonicalisation settings that decide the components, and both versions. Changing any of them on an existing workspace is refused with `representation_mismatch` until the workspace is migrated ([6.5](06_graph_and_storage.md#65-versions-and-migration)). The device is not part of it: the model runs in its checkpoint dtype on CPU and GPU alike, so a workspace moves between them. Renderings — documents, headlines, the prompt — are not part of it either.

## 4.7 Configuration

| Parameter | Default | Changes the fingerprint |
|---|---|---|
| `EMBEDDING_BACKEND` | `sentence-transformers` | yes (model id) |
| `EMBEDDING_MODEL`, `MODEL_DIR` | `Qwen/Qwen3-Embedding-0.6B`, `models` | model: yes |
| `EMBEDDING_TRUNCATE_DIM` | 384 | yes |
| `EMBEDDING_QUERY_INSTRUCTION` | retrieval task sentence | yes |
| `EMBEDDING_DEVICE` | `auto` | no (the model runs in its checkpoint dtype on every device) |
| `BLOCK_WEIGHTS`, `EMM_COMPONENT_WEIGHT` | (0.45, 0.55, 1.0), 0.5 | yes |
| `MIN_COMPONENT_Z`, `MIN_EMM_SCORE`, `WEIGHT_EMM_REF` | 0.5, 0.08, 0.08 | yes (they decide the components) |

## 4.8 Guarantees and failure modes

* Embedding inputs contain only conditions (scope) and metric names (target, labels): no measured numbers, no prose. The readable text is a pure function of the insight and the config, and ASCII for ASCII data.
* Every row and every block is unit-norm, also after truncation (the hashing backend excepted for text without `[a-z0-9]`); `dim = 3 · block_dim`; identical input gives identical output.
* The query instruction never reaches component labels or recognised scope/target strings.
* The fingerprint changes whenever any input that shapes a stored vector changes ([4.6](#46-representation-identity-and-versions)).
* Model load or encode errors become `embedding_failure` (the batch fails before any state change); a fingerprint change becomes `representation_mismatch`.

Tests: `test_canonical_sections_are_separate`, `test_embedding_labels_carry_no_numbers`, `test_text_number_rules`, `test_covariance_canonical_component`, `test_embedding_contract` (hashing and the real model: shape, dtype, unit norms, stability, direction separation, cross-scope similarity), `test_fingerprint_changes_with_representation_choices`, `test_query_instruction_only_reaches_free_question_text`, `test_env_file_switches_to_the_minilm_block`.
