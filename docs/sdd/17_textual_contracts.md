# SDD 17 — Textual contracts: what is written down, where, and what reaches the embeddings

## Purpose
Every piece of text the system stores, derives or embeds, with its exact format: how a pattern, its scope, target and phenomenon are described; what is persisted verbatim and what is rendered on demand; which strings feed the encoder and which never do; how latent anchors get their description; and what the language model is shown. SDD 16 holds the numbers; this document holds the words.

## Scope
`ltir/models.py` (records), `ltir/canonical.py` (canonical form, headline), `ltir/encoder.py` (what is embedded), `ltir/query.py` (question text), `ltir/graph.py` (node labels and props), `ltir/evidence.py` and `ltir/llm.py` (LLM-facing text), `ltir/web/app.py`, `ltir/web/static/app.js`, `ltir/sphere.py` (UI text), `ltir/neo4j_sink.py` (property flattening), `ltir/store.py` (files).

## Three tiers of text
| Tier | What | Where it lives | Who reads it |
|---|---|---|---|
| **Record** | the typed `Insight` as JSON — numbers, selector strings, provenance; the single source of truth | `journal/patterns.jsonl`, snapshot Pattern props, Neo4j | every module (`Insight.from_record`) |
| **Rendering** | human-readable text derived from the record: canonical document, headline, labels, hover text, evidence prompt, evidence-only summary — two flavours: the **LLM serializer** (ASCII prose, rounded numbers; SDD 05 helpers) and **visual labels** (compact ASCII for graph nodes and cards) | stored next to the record (canonical document) or computed on demand | people, the LLM, the naive text-NN baseline |
| **Embedding input** | three short strings per insight — scope text, target text and the phenomenon labels — composed into the vector | `journal/embeddings.mmap` (the vector), `journal/blocks/*.npz` (the three blocks) | retrieval, the ontology, the sphere |

Only the third tier shapes similarity. Numbers, confounders, support, dataset names and the full document are never embedded (§5).

---

## 1. Identifiers
| Identifier | Format | Example |
|---|---|---|
| dataset id | `ds-` + 12 hex of the content hash (SDD 16 §1) | `ds-6e53eb7fb0f9` |
| batch id | `B` + UTC timestamp `YYYYMMDDTHHMMSS` + `-` + 6 hex | `B20261001T065213-293b1a` |
| pattern id | `P-` + 12 hex of `sha1(dataset_id, sorted condition exprs)` | `P-bc4657a04746` |
| attractor id / node id | integer `k` from lac; graph node `A-k` | `A-0` |
| schema node ids | `D:<dataset_id>:<column>`, `M:<dataset_id>:<column>`, `DS:<dataset_id>`, `B:<batch_id>` | `M:ds-6e53eb7fb0f9:discount` |
| edge id | `<TYPE>:<source>-><target>` | `ACTIVATES:P-a2d74883914a->A-6` |
| citation key | `P` + 1-based rank in the evidence | `[P1]`, grouped `[P4, P7]` |
| journal row id | integer position in `embeddings.mmap` (`row_id`) | `1` |

Condition expression: `attribute=value` (`Condition.expr`, no spaces, no quotes). EDA selector expression (`Insight.expression`): pysubgroup's rendering `category=='phones' AND region=='US'`, kept verbatim as provenance.

## 2. The Insight record (`Insight.to_record`, journal and snapshot)
Field order and meaning (all JSON-native; floats are float64, lists where the dataclass has tuples):

| Field | Content |
|---|---|
| `id`, `dataset_id`, `batch_id` | identifiers (§1) |
| `conditions` | the **closed intent** (SDD 03): `[{"attribute": "category", "value": "phones"}, …]` sorted by attribute then value, including conditions implied by the extent; values are strings even for number-coded columns |
| `expression` | EDA selector string of the cohort (the merged selector with most conditions), verbatim |
| `target` | raw column name of the primary metric (`"discount"`) |
| `shifts` | `[{"metric", "robust_z" (signed), "local_median", "global_median", "global_mad", "source": "eda" \| "covariance_pair"}, …]` sorted by `|robust_z|` descending |
| `support`, `support_fraction` | covered rows, fraction of `N` |
| `baseline`, `local`, `effect_size` | global median, subgroup median, signed robust z of `target` |
| `sd_score`, `sd_raw_score`, `emm_score`, `volume_utility`, `stability`, `integrated_index` | SDD 16 §5–7 |
| `p_value`, `p_adjusted` | median test, Bonferroni |
| `drivers` | EDA confounder strings, e.g. `"[payment] heavily skewed to 'cash' (JS: 0.20)"`, `"[discount] hidden shift (+1.8 robust sigma)"` |
| `row_hash` | 16 hex, identity of the covered row set |
| `phenomenon_type` | `"shift"` or `"covariance"` |
| `covariance` | `{"pair": [a, b], "local_corr", "global_corr", "delta"}` or `{}` |
| `aliases` | selector strings merged into this cohort before validation (R5 identical extent, R6 near duplicate) |
| `weight`, `weight_factors` | `insight_weight`; `{"effect", "stability", "confidence", "support", "emm"}` with `null` for unmeasured factors |
| `provenance` | `{"dataset_id", "filename", "batch_id", "engine": "ltir/engines/eda/main_upd.py", "steps": [...], "expression", "rows_ref": "datasets/<ds>/covers.npz#<pattern id>", "multiple_testing_family"}` |

The journal record adds `row_id`, `canonical` (§3 plus `document`) and `embedding` (`{"fingerprint", "model_id", "dim", "representation_version"}`). The snapshot's Pattern node carries this journal record unchanged as `props`; `label` is the headline (§6). What is **not** stored: the covered rows themselves (only their positions in `covers.npz`), the profiled DataFrame, any raw cell values other than the condition values and the medians.

## 3. The canonical form (`canonical.canonicalize`, `CANONICAL_VERSION = "ltir-canon-3"`)
`CanonicalInsight(insight_id, version, target, scope, scope_sentence, phenomenon, covariance, confounders, support, components)` holds two contracts (SDD 05):

* **embedding inputs** — `scope` (`category = phones; region = US`: `attribute = value` joined by `; `, raw column names, values verbatim), `target` (`humanize(target)`), `components`;
* **readable text** — `scope_sentence`, `phenomenon`, `covariance`, `confounders`, `support`, rendered by `document()` as a Markdown header plus key/value bullets (ASCII; number rules in SDD 05).

```text
### Subgroup finding P-bc4657a04746
* Scope: category is phones and region is US
* Target metric: discount
* Observed shift: discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall); margin: moderate decrease, -1.10 sd (median 14.75 vs 19.45 overall)
* Metric relationships: strongest change: correlation between delivery days and return rate weakens from +0.88 overall to +0.73 in the subgroup (divergence score 0.02)
* Confounders: none detected
* Validation: 438 rows (8.8% of 5,000); bootstrap stability 0.97; adjusted p < 0.001
```
Covariance example (the correlation phrase leads the observed shift; `no material median shift` when no shift reaches `MIN_COMPONENT_Z`):
```text
### Subgroup finding P-7b083e5ffddc
* Scope: category is tablets and channel is online
* Target metric: delivery days
* Observed shift: correlation between delivery days and discount strengthens from -0.01 overall to +0.65 in the subgroup; no material median shift
* Metric relationships: strongest change: correlation between delivery days and discount strengthens from -0.01 overall to +0.65 in the subgroup (divergence score 0.14)
* Confounders: none detected
* Validation: 515 rows (10.3% of 5,000); bootstrap stability 0.46; adjusted p 1.00
```
**Components** (stored as `[[label, coefficient], …]`): `[["discount", 2.2078], ["margin", -1.0958]]` and `[["correlation between delivery days and discount", 0.8723]]`. Labels are `humanize(metric)` and `correlation between <a> and <b>` with `a, b` sorted, and never contain digits; coefficients are the signed robust z or the signed EMM component (SDD 16 §10). Structural predicates (scope) and statistical behaviour (phenomenon) never share a string.

## 4. Short renderings of a pattern
| Surface | Code | Format | Example |
|---|---|---|---|
| headline (graph `label`, Neo4j `label`, evidence `headline`) | `canonical.headline` | compact ASCII: `<attr>=<value>, …: <humanized target> <±z> sd`; covariance: `<scope>: corr(<a>, <b>) <strengthens\|weakens\|reverses>` | `category=phones, region=US: discount +2.21 sd` |
| UI graph label (`/api/graph`, `label`; `full_label` = headline) | `web.app._short_label` | condition **values** joined by ` · `, newline, `<target> <±z> sd` or `corr(<a>, <b>)` | `phones · US⏎discount +2.21 sd` |
| UI table / drawer | `app.js` | scope tags `attr = value`, metric `human(target)`, effect `±z sd` (colour carries the direction), drawer lead sentence `<metric> is higher/lower here — median <local> vs <global> overall (<±z> sd)`; covariance: `The relationship between <a> and <b> changes here: correlation <global> overall → <local> in this subgroup.` | — |
| sphere hover | `sphere._pattern_hover` | HTML: id, `Scope: category=phones ∧ region=US`, `Target: discount (shift)`, up to three shifts `metric: local vs global (z ±x)`, support · weight, anchor + alignment, dataset file | — |
| evidence-only summary line | `Evidence.summary` | `- <scope sentence> \| <top 2 shift phrases> \| n=<support> [P#]` + ` (scope-disjoint from the seed, linked via a latent anchor)` when transversal-only; last line `Interpretation (hypotheses): not generated (no language-model answer is available).` | `- category is phones and region is US \| discount: strong increase, +2.21 sd (…); margin: … \| n=438 [P1]` |
| provenance footer | `qa.answer_question` | `Sources: [P1] <pattern id> = <expression> (dataset <ds>, <file>, batch <batch>); …` | — |

## 5. What the encoder embeds — and what it does not
Per insight exactly three strings reach the sentence model (`InsightEncoder.encode`):

| Block | Text | Example |
|---|---|---|
| scope `s` | `CanonicalInsight.scope` | `category = phones; region = US` |
| target `t` | `CanonicalInsight.target` | `discount` |
| phenomenon `p` | the **labels** of `components`, each embedded separately and summed with its signed coefficient: `p = normalize(Σ coef · E(label))` | `E("discount")·2.21 + E("margin")·(−1.10)` |

Fallback: when the component sum is a zero vector (no components), `p = E(CanonicalInsight.phenomenon)`, the observed-shift sentence. The label vocabulary is small and shared — humanised metric names and `correlation between <a> and <b>` — so patterns on the same metrics with the same signs land on the same phenomenon direction regardless of their scope, and opposite signs land opposite (SDD 16 §11).

Never embedded: the six-section document as a whole (except by the naive text-NN baseline, which embeds `canonical.document` to show what plain RAG would do), numbers (medians, z values enter only through the coefficients), confounders, support, p-values, the covariance section, dataset or batch identifiers, aliases, the EDA selector string.

Dimension and attractor nodes have no embedding of their own: a dimension is text in the graph only; an attractor is a centroid of insight vectors (§8).

## 6. Question text (`query.parse_query`, `resolve_seeds`)
A question is parsed against the graph vocabulary and projected with the same composition as an insight:

| Block | Text | Demo question `Why is margin lower for phones in the US?` |
|---|---|---|
| scope | `attribute = value; …` for every recognised condition (raw attribute names, as in §3) | `category = phones; region = US` |
| target | `humanize(metric)` joined by `; ` | `margin` |
| phenomenon | `(humanize(metric), direction · 2.0)` per recognised metric; relationship questions: `("correlation between a and b", ±2.0)`; no direction word → no components | `[("margin", −2.0)]` |
| fallback | the question text stands in for an empty scope or target block and for an empty component sum — embedded **with** the query instruction (`Instruct: Given a quantitative analysis question, retrieve relevant statistical subgroup patterns\nQuery: <question>`); recognised scope/target strings and component labels are embedded without it (SDD 06) | — |

`ParsedQuery.to_dict()` (stored in the evidence as `parsed`): `{"text", "targets": ["margin"], "direction": -1, "conditions": ["category=phones", "region=US"], "covariance": false}`.

## 7. Schema nodes and edges
| Node | `label` | `props` |
|---|---|---|
| Dimension `D:<ds>:<col>` | raw column name | `dataset_id, name, cardinality, entropy` (selected dimensions and every attribute a closed intent uses) |
| Metric `M:<ds>:<col>` | `humanize(name)` | `dataset_id, name (raw), global_median, global_mad` |
| Dataset `DS:<ds>` | file name | `dataset_id, filename, rows, columns` |
| Batch `B:<id>` | batch id | `batch_id, batch_seq, created_at, status` |

Edge props carrying text: SPECIALIZES/GENERALIZES `added_conditions: ["category=phones"]`, `metric`; SIBLING `parent_scope: ["category=laptops"]`, `partition_attribute`, `values: ["partner", "online"]`; CONTRASTS `metric`, `relation`; HAS_SCOPE `value`; TARGETS `role: primary|secondary`; ACTIVATES `source ∈ {cold_start, assign, omp, absorbed, nearest, reroute}`, `batch_id`, `weak`; RELATED_TO `kind: mutual_knn`.

## 8. How a latent anchor gets its description
An attractor stores **no text**: lac persists only its centroid, `chunk_count`, `last_updated_batch` and `created_at` (`state/concepts.npz`, `state/state.json`). Everything readable about it is derived at snapshot time from its members (`graph._attractor_nodes`) and recomputed after every batch, so the description follows the membership:

| Text | Derivation | Example |
|---|---|---|
| `label` | the two strongest entries of the **signature** rendered by `_component_word`: `<label> ↑` / `<label> ↓`, or `corr(<a>~<b>) strengthens|weakens` for correlation labels, joined by ` · `; `Attractor <k>` if it has no members | `discount ↑ · margin ↓`, `corr(discount~margin) weakens` |
| `signature` | strength-weighted mean of the members' components, each member scaled to `[−1, 1]` (SDD 16 §14); stored as `[{"component", "value"}, …]`, six strongest | `[{"component": "discount", "value": 1.0}, {"component": "margin", "value": -0.487}]` |
| `dimensions`, `targets`, `datasets` | sorted sets over the members' conditions, `target` and `dataset_id` | `["category", "channel", "region"]`, `["discount"]` |
| `n_patterns`, `distinct_scopes`, `mass`, `evidence_mass`, `dispersion` | counts and sums (SDD 16 §14) | `8, 8, 8, 6.25, 0.011` |
| `last_updated_batch` | lac batch sequence mapped back to the batch id | `B20261001T065213-293b1a` |
| `centroid` | `{"dim", "norm", "representation_version", "fingerprint"}` — the vector contract, not the vector | — |
| `description` (evidence only) | `graph.describe_components(signature)`: the two strongest entries as ASCII prose — `<label> up` / `<label> down`, `correlation between a and b strengthens\|weakens`, joined by ` and `; falls back to `label` | `discount up and margin down` |
| evidence prompt line | `- <A-k> "<description>": <n> patterns over <d> distinct scopes; related: <A-j> (<w>), …` | `- A-1 "discount up and margin down": 9 patterns over 9 distinct scopes; related: A-2 (0.32), A-0 (0.61)` |
| sphere hover | `Latent anchor A-k`, label, `Patterns: n over d scopes`, `Mass: m · evidence mass e`, `Dimensions: …` | — |
| UI drawer | `A recurring pattern learned from <n> insights across <d> different subgroups.` + signature rows `<component> ±value` + details | — |
| UI vocabulary | "theme" | — |

Consequence: an anchor's name can change when new members arrive (the signature moves); its id `A-k` and centroid identity do not. Attractor labels are therefore never used as keys — ids are.

## 9. What the language model sees (`Evidence.to_prompt`, `llm.SYSTEM_PROMPT`)
The LLM receives one system prompt and one user message; nothing else (no graph dump, no rows, no history).

System prompt (verbatim, ASCII):
```text
You are a careful data analyst answering questions about a tabular dataset.
You receive EVIDENCE: statistically validated subgroup findings retrieved from a knowledge graph.
Shifts are robust standard deviations ("sd": the median difference scaled by the MAD).
Rules:
1. Use ONLY the evidence. Do not invent numbers, subgroups, metrics or datasets.
2. Cite every factual statement with its key, e.g. [P1] or [P2][P4].
3. Structure the answer in two labelled parts:
   "Observations:" - what the verified statistics show (medians, shifts in sd, support).
   "Interpretation (hypotheses):" - possible explanations, explicitly marked as hypotheses.
4. These are observational subgroup statistics. Do not claim causation; say "is associated with".
5. If items were reached through a latent anchor and are scope-disjoint from the seeds (no shared
   condition), point out that the same phenomenon recurs in a different part of the data.
6. If the evidence does not answer the question, say so plainly.
Be concise (at most ~250 words).
```

The user message is ASCII (`Evidence.to_prompt`, tested with `prompt.isascii()`). Sections, in order: `QUESTION:`; `PARSED:` (`target=… | direction=up|down | scope=…` or `no explicit metric/scope recognised`); `UNITS: shifts are robust standard deviations (sd = median difference scaled by the MAD); medians compare the subgroup with the whole dataset.`; `DATASETS:` (`- <ds> file=<name> batch=<id> rows=<N>`); `METRIC BASELINES (whole dataset):` (`- <metric>: median <m>, MAD <mad>` for the metrics the evidence mentions); `LATENT ANCHORS VISITED (…)` (§8); `EVIDENCE (verified statistical observations; cite as [P#]):` with one block per item, the key first:
```text
[P2] role=transversal | scope: category is phones, channel is online, and region is US | support 205 rows (4.1%)
     shifts: discount: strong increase, +2.25 sd (median 19.35 vs 10.74 overall); margin: moderate decrease, -1.10 sd (median 14.72 vs 19.45 overall)
     validation: bootstrap stability 0.97 | adjusted p < 0.001 | insight weight 0.82 | confounders: none detected
     retrieved via: P-bc4657a04746 -ACTIVATES(0.98)-> A-1 <-ACTIVATES(0.99)- P-4f7921c47717
```
Only phenomenon shifts are listed (the target and \|z\| ≥ `MIN_COMPONENT_Z`); a `relationship:` line (`correlation between a and b weakens from -0.57 overall to +0.09 in the subgroup (divergence 0.09)`) appears only when the correlation change is material. Then `NOTE:` lines (scope-disjoint items; or `No pattern in the graph matched the question.`). The path uses node ids and edge weights, `-TYPE(w)->` forward and `<-TYPE(w)-` against the stored direction, so the model can name the anchor it came through. The evidence-only answer (`Evidence.summary`, used when there is no LLM answer) is built from the same items (§4); citations are validated against the keys (`qa.check_citations`).

## 10. Persisted files and their formats
| File | Format | Text it holds |
|---|---|---|
| `registry/batches/<batch_id>.json` | JSON object | status, stage times, `profile` (column lists, selected dimensions, global medians/MADs, cardinality, entropy, bins, categorical overrides), `metrics`, `warnings[]`, `error {code, message[, trace]}`, `neo4j`, `owner_pid` |
| `datasets/<ds>/source.<ext>` | the uploaded file, byte-for-byte | provenance only |
| `datasets/<ds>/profile.json` | JSON | the batch `profile` |
| `datasets/<ds>/covers.npz` | npz, key = pattern id → int64 row positions | — |
| `datasets/<ds>/rejections.json` | JSON list `{"expression", "reason", "detail"}` (discovery merges first, then selection) | reasons: `cover_equivalent`, `near_duplicate` (before validation), `min_support`, `unstable`, `not_significant`, `weak_effect`, `low_weight`, `budget`; `detail` e.g. `same rows as …`, `\|z\|=0.17 p_adj=1 emm=0.07` |
| `journal/patterns.jsonl` | one JSON record per line (§2 + `row_id`, `canonical`, `embedding`) | the record and the canonical document |
| `journal/activations.jsonl` | one JSON record per line: `{pattern_id, attractor_id, alignment, strength, engine_weight, source, weak, batch_id, row_id}` | — |
| `journal/embeddings.mmap` + `embeddings_meta.json` | float32 matrix `(rows, 1152)`; `{"rows", "dim"}` | — |
| `journal/blocks/<batch_id>.npz` | keys `scope`, `target`, `phenomenon` (float32 `(n, 384)`) | — |
| `state/representation.json` | `EmbeddingSpec` + `fingerprint` | `model_id`, `truncate_dim`, `query_instruction`, `normalization: "l2(block) -> weighted concat -> l2"`, `canonical_version`, `representation_version` |
| `state/concepts.npz`, `state.json`, `orphan_buffer.npz` | lac ConceptStore | `created_at` timestamps only |
| `state/ontology_metrics.csv` | lac `BatchMetrics` rows | warning strings |
| `graph/snapshot.json` | `{version, created_at, representation, nodes[{id, kind, label, props}], edges[{id, source, target, type, plane, weight, props}], stats}` | all labels and props above |
| `graph/sphere.html` | standalone Plotly page | hover texts (§4, §8) |
| `logs/queries.jsonl` | one JSON per question: `{at, question, mode, metrics, seeds, evidence, citations}` | the question text |
| Neo4j | node props = `label` + the snapshot props; nested values become JSON strings in `<key>_json` (`conditions_json`, `shifts_json`, `canonical_json`, `signature_json`, `centroid_json`, …); relationship types are the edge types | same text as the snapshot |

## 11. UI vocabulary
The UI translates graph terms for readers (glossary in SDD 01): attractor → *theme*, pattern → *insight*, `insight_weight` → *evidence*, robust z → `±x sd`, ACTIVATES → *membership*, RELATED_TO → *theme link*, transversal-only → *other segment*; `human()` in `app.js` replaces underscores in column names. Ids (`P-…`, `A-k`) stay visible in path chains and citations so the UI, the prompt and the journal agree.

## 12. Versioning of text contracts
`CANONICAL_VERSION` (`ltir-canon-3`) covers §3 (both contracts, the closed-intent scope, phrase grammar, component labels and coefficients); `REPRESENTATION_VERSION` (`ltir-rep-3`) covers §5 (what is embedded and how it is composed) together with `BLOCK_WEIGHTS`, `EMM_COMPONENT_WEIGHT`, the model id, `truncate_dim` and `query_instruction` (all inside the fingerprint). A change to any of these bumps the version; a workspace built with another version is refused (`representation_mismatch`) rather than mixed, and `python -m ltir migrate --yes` rebuilds it (SDD 09). Renderings (§4, §7–9) can change freely: they are derived from the record.

**Token budget** (`scripts/prompt_tokens.py` on the demo question; XLM-R SentencePiece as a proxy for Gemma's 256k SentencePiece — the Gemma tokenizer is gated — and the Qwen3 byte-level BPE):

| surface | chars | non-ASCII | XLM-R tokens | Qwen tokens |
|---|---|---|---|---|
| evidence prompt, symbol format (`ltir-canon-2`) | 7,009 | 25 | 2,421 | 2,880 |
| evidence prompt, ASCII prose (`ltir-canon-3`) | 6,330 | 0 | 2,102 (−13 %) | 2,341 (−19 %) |
| canonical documents (28), canon-2 → canon-3 | 11,406 → 15,080 | 0 → 0 | 3,455 → 4,367 | 4,224 → 5,017 |
| evidence-only answer, canon-2 → canon-3 | 1,730 → 2,509 | 7 → 0 | 588 → 779 | 760 → 887 |

Live, with Gemma's own counts over five questions, prompt tokens fell 14,983 → 11,705 for the format change alone (SDD 12). The documents and the evidence-only answer grew: they are human-facing (the documents reach no LLM; only the naive text-NN baseline embeds them).

## Invariants
* A pattern's record is written once; every rendering is a pure function of it (and, for anchors, of the current membership).
* The three embedded strings are exactly `canonical.scope`, `canonical.target` and the `components` labels; nothing else reaches the model.
* Every string the LLM can cite (`[P#]`) maps to one pattern id and one selector expression.
* Attribute names are raw column names everywhere a scope is written (canonical scope, query scope, headline, prose scope); metric names are humanised wherever they are read as text (target, components, prompt shifts, UI).
* Everything the LLM reads (system prompt, user message) is ASCII; embedding labels carry no numbers.

## Testing requirements
`tests/test_canonical_embedding.py` (embedding inputs vs document, ASCII, number rules, labels without digits, query instruction placement), `tests/test_traversal.py::test_evidence_object_is_structured_and_traceable` (prompt sections and keys), `tests/test_e2e.py` (provenance on every item, citations), `tests/test_persistence.py` (Pattern props = journal record; Neo4j flattening), `tests/test_sphere.py` (hover and legend text per anchor).

## Integration points
SDD 05 (canonical form), SDD 06 (encoder), SDD 08 (graph schema), SDD 11–12 (evidence and LLM), SDD 13 (UI).

## Current implementation status
Verified against the code on 2026-10-01; the examples above are the demo dataset's output.
