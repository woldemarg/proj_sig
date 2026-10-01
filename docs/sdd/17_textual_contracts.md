# SDD 17 — Textual contracts: what is written down, where, and what reaches the embeddings

## Purpose
Every piece of text the system stores, derives or embeds, with its exact format: how a pattern, its scope, target and phenomenon are described; what is persisted verbatim and what is rendered on demand; which strings feed the encoder and which never do; how latent anchors get their description; and what the language model is shown. SDD 16 holds the numbers; this document holds the words.

## Scope
`ltir/models.py` (records), `ltir/canonical.py` (canonical form, headline), `ltir/encoder.py` (what is embedded), `ltir/query.py` (question text), `ltir/graph.py` (node labels and props), `ltir/evidence.py` and `ltir/llm.py` (LLM-facing text), `ltir/web/app.py`, `ltir/web/static/app.js`, `ltir/sphere.py` (UI text), `ltir/neo4j_sink.py` (property flattening), `ltir/store.py` (files).

## Three tiers of text
| Tier | What | Where it lives | Who reads it |
|---|---|---|---|
| **Record** | the typed `Insight` as JSON — numbers, selector strings, provenance; the single source of truth | `journal/patterns.jsonl`, snapshot Pattern props, Neo4j | every module (`Insight.from_record`) |
| **Rendering** | human-readable text derived from the record: canonical document, headline, labels, hover text, evidence prompt, evidence-only summary | stored next to the record (canonical document) or computed on demand | people, the LLM, the naive text-NN baseline |
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
| `conditions` | `[{"attribute": "category", "value": "phones"}, …]` sorted by attribute then value; values are strings even for number-coded columns |
| `expression` | EDA selector string, verbatim |
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
| `aliases` | selector strings of patterns collapsed into this one (R5/R6) |
| `weight`, `weight_factors` | `insight_weight`; `{"effect", "stability", "confidence", "support", "emm"}` with `null` for unmeasured factors |
| `provenance` | `{"dataset_id", "filename", "batch_id", "engine": "ltir/engines/eda/main_upd.py", "steps": [...], "expression", "rows_ref": "datasets/<ds>/covers.npz#<pattern id>", "multiple_testing_family"}` |

The journal record adds `row_id`, `canonical` (§3 plus `document`) and `embedding` (`{"fingerprint", "model_id", "dim", "representation_version"}`). The snapshot's Pattern node carries this journal record unchanged as `props`; `label` is the headline (§6). What is **not** stored: the covered rows themselves (only their positions in `covers.npz`), the profiled DataFrame, any raw cell values other than the condition values and the medians.

## 3. The canonical form (`canonical.canonicalize`, `CANONICAL_VERSION = "ltir-canon-2"`)
`CanonicalInsight(insight_id, version, target, scope, phenomenon, covariance, confounders, support, components)`; `document()` renders the six sections, one per line, in this fixed order:

```text
TARGET: discount
SCOPE: category = phones; region = US
PHENOMENON: discount strong increase (robust z +2.21; median 19.19 vs 10.74); margin moderate decrease (robust z -1.10; median 14.75 vs 19.45)
COVARIANCE: stabilized correlation divergence 0.022; strongest pair delivery days ~ return rate +0.88 -> +0.73
CONFOUNDERS: none detected
SUPPORT: 438 rows (8.8% of 5,000); bootstrap stability 0.97; adjusted p 0
```

| Section | Rule |
|---|---|
| `TARGET` | `humanize(target)`: underscores → spaces (`delivery_days` → `delivery days`) |
| `SCOPE` | `attribute = value` joined by `; ` in condition order; **attribute names are raw column names** (`median_income_band = q4`), values verbatim |
| `PHENOMENON` | one phrase per phenomenon shift (the target always; others with `\|z\| ≥ MIN_COMPONENT_Z`): `<metric> <mild\|moderate\|strong\|extreme> <increase\|decrease> (robust z ±x.xx; median <local> vs <global>)` with medians to 4 significant digits; a covariance phrase `correlation between <a> and <b> <strengthens\|weakens\|reverses> (<global> -> <local>)` is inserted first for covariance insights and appended otherwise (when `emm_score ≥ MIN_EMM_SCORE`); a covariance insight without material shifts ends with `no material median shift` |
| `COVARIANCE` | `stabilized correlation divergence <emm 3 dp>; strongest pair <a> ~ <b> <global> -> <local>` (`;` part only when a pair exists) |
| `CONFOUNDERS` | the EDA driver strings joined by `; `, or `none detected` |
| `SUPPORT` | `<n> rows (<share>% of <N>); bootstrap stability <s>; adjusted p <p 2 sig>` |

Covariance example:
```text
TARGET: delivery days
SCOPE: category = tablets; channel = online
PHENOMENON: correlation between delivery days and discount strengthens (-0.01 -> +0.65); no material median shift
COVARIANCE: stabilized correlation divergence 0.140; strongest pair discount ~ delivery days -0.01 -> +0.65
CONFOUNDERS: none detected
SUPPORT: 515 rows (10.3% of 5,000); bootstrap stability 0.46; adjusted p 1
```
**Components** (`canonical.components`, stored as `[[label, coefficient], …]`): `[["discount", 2.2078], ["margin", -1.0958]]` for the first example, `[["correlation between delivery days and discount", 0.8723]]` for the second. Labels are `humanize(metric)` and `correlation between <a> and <b>` with `a, b` sorted; coefficients are the signed robust z or the signed EMM component (SDD 16 §10). Structural predicates (scope) and statistical behaviour (phenomenon) never share a string.

## 4. Short renderings of a pattern
| Surface | Code | Format | Example |
|---|---|---|---|
| headline (graph `label`, Neo4j `label`, evidence `headline`) | `canonical.headline` | `<attr>=<value>, …: <humanized target> <↑\|↓> (<±z> z)`; covariance: `<scope>: correlation between a and b <word>` | `category=phones, region=US: discount ↑ (+2.21 z)` |
| UI graph label (`/api/graph`, `label`; `full_label` = headline) | `web.app._short_label` | condition **values** joined by ` · `, newline, `<target> <↑\|↓>` or `corr <a>~<b>` (raw names) | `phones · US⏎discount ↑` |
| UI table / drawer | `app.js` | scope tags `attr = value`, metric `human(target)`, effect `±z σ`, drawer lead sentence `<metric> is higher/lower here — median <local> vs <global> overall (<±z>σ)`; covariance: `The relationship between <a> and <b> changes here: correlation <global> overall → <local> in this subgroup.` | — |
| sphere hover | `sphere._pattern_hover` | HTML: id, `Scope: category=phones ∧ region=US`, `Target: discount (shift)`, up to three shifts `metric: local vs global (z ±x)`, support · weight, anchor + alignment, dataset file | — |
| evidence-only summary line | `Evidence.summary` | `- <scope AND-joined>: <metric> <local> vs <global> (z ±x), … (top 2 shifts); n=<support> [P#]` + ` — scope-disjoint from the seed, linked via a latent anchor` when transversal-only | — |
| provenance footer | `qa.answer_question` | `Sources: [P1] <pattern id> = <expression> (dataset <ds>, <file>, batch <batch>); …` | — |

## 5. What the encoder embeds — and what it does not
Per insight exactly three strings reach the sentence model (`InsightEncoder.encode`):

| Block | Text | Example |
|---|---|---|
| scope `s` | `CanonicalInsight.scope` | `category = phones; region = US` |
| target `t` | `CanonicalInsight.target` | `discount` |
| phenomenon `p` | the **labels** of `components`, each embedded separately and summed with its signed coefficient: `p = normalize(Σ coef · E(label))` | `E("discount")·2.21 + E("margin")·(−1.10)` |

Fallback: when the component sum is a zero vector (no components), `p = E(CanonicalInsight.phenomenon)`, the PHENOMENON sentence. The label vocabulary is small and shared — humanised metric names and `correlation between <a> and <b>` — so patterns on the same metrics with the same signs land on the same phenomenon direction regardless of their scope, and opposite signs land opposite (SDD 16 §11).

Never embedded: the six-section document as a whole (except by the naive text-NN baseline, which embeds `canonical.document` to show what plain RAG would do), numbers (medians, z values enter only through the coefficients), confounders, support, p-values, the covariance section, dataset or batch identifiers, aliases, the EDA selector string.

Dimension and attractor nodes have no embedding of their own: a dimension is text in the graph only; an attractor is a centroid of insight vectors (§8).

## 6. Question text (`query.parse_query`, `resolve_seeds`)
A question is parsed against the graph vocabulary and projected with the same composition as an insight:

| Block | Text | Demo question `Why is margin lower for phones in the US?` |
|---|---|---|
| scope | `attribute = value; …` for every recognised condition (raw attribute names, as in §3) | `category = phones; region = US` |
| target | `humanize(metric)` joined by `; ` | `margin` |
| phenomenon | `(humanize(metric), direction · 2.0)` per recognised metric; relationship questions: `("correlation between a and b", ±2.0)`; no direction word → no components | `[("margin", −2.0)]` |
| fallback | the question text stands in for an empty scope or target block and for an empty component sum | — |

`ParsedQuery.to_dict()` (stored in the evidence as `parsed`): `{"text", "targets": ["margin"], "direction": -1, "conditions": ["category=phones", "region=US"], "covariance": false}`.

## 7. Schema nodes and edges
| Node | `label` | `props` |
|---|---|---|
| Dimension `D:<ds>:<col>` | raw column name | `dataset_id, name, cardinality, entropy` |
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
| evidence prompt line | `- <A-k> "<label>": <n> patterns over <d> distinct scopes; related: <A-j> (<w>), …` | `- A-0 "discount ↑ · margin ↓": 8 patterns over 8 distinct scopes; related: A-6 (0.41), …` |
| sphere hover | `Latent anchor A-k`, label, `Patterns: n over d scopes`, `Mass: m · evidence mass e`, `Dimensions: …` | — |
| UI drawer | `A recurring pattern learned from <n> insights across <d> different subgroups.` + signature rows `<component> ±value` + details | — |
| UI vocabulary | "theme" | — |

Consequence: an anchor's name can change when new members arrive (the signature moves); its id `A-k` and centroid identity do not. Attractor labels are therefore never used as keys — ids are.

## 9. What the language model sees (`Evidence.to_prompt`, `llm.SYSTEM_PROMPT`)
The LLM receives one system prompt and one user message; nothing else (no graph dump, no rows, no history).

System prompt (verbatim):
```text
You are a careful data analyst answering questions about a tabular dataset.
You receive EVIDENCE: statistically validated subgroup findings retrieved from a knowledge graph.
Rules:
1. Use ONLY the evidence. Do not invent numbers, subgroups, metrics or datasets.
2. Cite every factual statement with its key, e.g. [P1] or [P2][P4].
3. Structure the answer in two labelled parts:
   "Observations:" — what the verified statistics show (medians, robust z shifts, support).
   "Interpretation (hypotheses):" — possible explanations, explicitly marked as hypotheses.
4. These are observational subgroup statistics. Do not claim causation; say "is associated with".
5. If items were reached through a latent anchor and are scope-disjoint from the seeds (no shared
   condition), point out that the same phenomenon recurs in a different part of the data.
6. If the evidence does not answer the question, say so plainly.
Be concise (at most ~250 words).
```

User message sections, in order: `QUESTION:`; `PARSED:` (`target=… | direction=up|down | scope=…` or `no explicit metric/scope recognised`); `DATASETS:` (`- <ds> file=<name> batch=<id> rows=<N>`); `METRIC BASELINES (whole dataset):` (`- <metric>: median <m>, MAD <mad>`); `LATENT ANCHORS VISITED (…)` (§8); `EVIDENCE — verified statistical observations (cite as [P#]):` with one block per item:
```text
[P4] role=transversal | scope: category=tablets AND channel=retail AND region=APAC | support 108 rows (2.2%)
     shifts: discount: median 18.74 vs 10.74 (robust z +2.09); margin: median 15.17 vs 19.45 (robust z -1.00); …
     correlation discount~margin: -0.29 in subgroup vs -0.57 overall (EMM 0.02)
     bootstrap stability 0.96 | adjusted p 1.1e-171 | insight weight 0.78 | confounders: none detected
     retrieved via: P-bc4657a04746 -ACTIVATES(0.99)-> A-0 -ACTIVATES⁻¹(0.99)-> P-dc62e58fd403 [scope-disjoint from seeds: no shared condition]
```
then `NOTE:` lines (scope-disjoint items; or `No pattern in the graph matched the question.`). Metric names are humanised in the shifts; scope conditions are `attribute=value`. The path uses node ids and edge weights, so the model can name the anchor it came through. The evidence-only answer (`Evidence.summary`, used when there is no LLM answer) is built from the same items (§4); citations are validated against the keys (`qa.check_citations`).

## 10. Persisted files and their formats
| File | Format | Text it holds |
|---|---|---|
| `registry/batches/<batch_id>.json` | JSON object | status, stage times, `profile` (column lists, selected dimensions, global medians/MADs, cardinality, entropy, bins, categorical overrides), `metrics`, `warnings[]`, `error {code, message[, trace]}`, `neo4j`, `owner_pid` |
| `datasets/<ds>/source.<ext>` | the uploaded file, byte-for-byte | provenance only |
| `datasets/<ds>/profile.json` | JSON | the batch `profile` |
| `datasets/<ds>/covers.npz` | npz, key = pattern id → int64 row positions | — |
| `datasets/<ds>/rejections.json` | JSON list `{"expression", "reason", "detail"}` | reasons: `min_support`, `unstable`, `not_significant`, `weak_effect`, `low_weight`, `cover_equivalent`, `near_duplicate`, `budget`; `detail` e.g. `\|z\|=0.17 p_adj=1 emm=0.07` |
| `journal/patterns.jsonl` | one JSON record per line (§2 + `row_id`, `canonical`, `embedding`) | the record and the canonical document |
| `journal/activations.jsonl` | one JSON record per line: `{pattern_id, attractor_id, alignment, strength, engine_weight, source, weak, batch_id, row_id}` | — |
| `journal/embeddings.mmap` + `embeddings_meta.json` | float32 matrix `(rows, 1152)`; `{"rows", "dim"}` | — |
| `journal/blocks/<batch_id>.npz` | keys `scope`, `target`, `phenomenon` (float32 `(n, 384)`) | — |
| `state/representation.json` | `EmbeddingSpec` + `fingerprint` | `model_id`, `normalization: "l2(block) -> weighted concat -> l2"`, `canonical_version`, `representation_version` |
| `state/concepts.npz`, `state.json`, `orphan_buffer.npz` | lac ConceptStore | `created_at` timestamps only |
| `state/ontology_metrics.csv` | lac `BatchMetrics` rows | warning strings |
| `graph/snapshot.json` | `{version, created_at, representation, nodes[{id, kind, label, props}], edges[{id, source, target, type, plane, weight, props}], stats}` | all labels and props above |
| `graph/sphere.html` | standalone Plotly page | hover texts (§4, §8) |
| `logs/queries.jsonl` | one JSON per question: `{at, question, mode, metrics, seeds, evidence, citations}` | the question text |
| Neo4j | node props = `label` + the snapshot props; nested values become JSON strings in `<key>_json` (`conditions_json`, `shifts_json`, `canonical_json`, `signature_json`, `centroid_json`, …); relationship types are the edge types | same text as the snapshot |

## 11. UI vocabulary
The UI translates graph terms for readers (glossary in SDD 01): attractor → *theme*, pattern → *insight*, `insight_weight` → *evidence*, robust z → `±x σ`, ACTIVATES → *membership*, RELATED_TO → *theme link*, transversal-only → *other segment*; `human()` in `app.js` replaces underscores in column names. Ids (`P-…`, `A-k`) stay visible in path chains and citations so the UI, the prompt and the journal agree.

## 12. Versioning of text contracts
`CANONICAL_VERSION` (`ltir-canon-2`) covers §3 (section templates, phrase grammar, component labels and coefficients); `REPRESENTATION_VERSION` (`ltir-rep-2`) covers §5 (what is embedded and how it is composed) together with `BLOCK_WEIGHTS`, `EMM_COMPONENT_WEIGHT` and the model id (all inside the fingerprint). A change to any of these bumps the version; a workspace built with another version is refused (`representation_mismatch`) rather than mixed. Renderings (§4, §7–9) can change freely: they are derived from the record.

## Invariants
* A pattern's record is written once; every rendering is a pure function of it (and, for anchors, of the current membership).
* The three embedded strings are exactly `canonical.scope`, `canonical.target` and the `components` labels; nothing else reaches the model.
* Every string the LLM can cite (`[P#]`) maps to one pattern id and one selector expression.
* Attribute names are raw column names everywhere a scope is written (canonical scope, query scope, headline, evidence); metric names are humanised wherever they are read as text (target, components, prompt shifts, UI).

## Testing requirements
`tests/test_canonical_embedding.py` (six separate sections; signed components; only the three blocks are composed), `tests/test_traversal.py::test_evidence_object_is_structured_and_traceable` (prompt sections and keys), `tests/test_e2e.py` (provenance on every item, citations), `tests/test_persistence.py` (Pattern props = journal record; Neo4j flattening), `tests/test_sphere.py` (hover and legend text per anchor).

## Integration points
SDD 05 (canonical form), SDD 06 (encoder), SDD 08 (graph schema), SDD 11–12 (evidence and LLM), SDD 13 (UI).

## Current implementation status
Verified against the code on 2026-10-01; the examples above are the demo dataset's output.
