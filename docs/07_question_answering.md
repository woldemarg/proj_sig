# 7. Question answering — seeds, transversal retrieval, evidence, language model

> **In one paragraph.** A question is parsed against the graph's own vocabulary — metric names, condition values, direction words — and projected with the same composition as an insight. The best-matching insights become *seeds*. A budgeted best-first walk then follows lattice edges and, through ACTIVATES, the latent anchors, collecting the insights it reaches with the exact path that reached them. The first ten become an evidence object: the only thing the language model is shown, rendered as an ASCII prompt with citation keys. The model's answer is checked against those keys, a provenance footer is built without the model, and when no model is available the same evidence becomes a deterministic summary.

**Code** `ltir/query.py`, `ltir/traversal.py`, `ltir/evidence.py`, `ltir/llm.py`, `ltir/qa.py` · **Tests** `tests/test_traversal.py`, `tests/test_llm_client.py`, `tests/test_e2e.py` · **Previous** [6. Graph and storage](06_graph_and_storage.md) · **Next** [8. Interface](08_interface.md)

---

The flow of `qa.answer_question(engine, question, use_llm=True)`:

```text
parse_query → resolve_seeds → traverse → compute_baselines → build_evidence
            → LLM: generate(SYSTEM_PROMPT, evidence.to_prompt())   or   evidence.summary()
            → check_citations → provenance footer → highlight groups → log to logs/queries.jsonl
```

Before retrieval the workspace's fingerprint is checked against the encoder's (`RepresentationMismatch` → HTTP 409); an empty graph returns `answer_mode = empty` at once.

## 7.1 From question to query

`parse_query(text, graph)` is deterministic and uses only the graph's vocabulary — no LLM. Tokens are runs of letters and digits in any script (underscores split, so `return_rate` is `return`, `rate`; an apostrophe stays inside a word), lower-cased; two words stem-match when they are equal, or both have at least 5 characters, share the first 5 and differ in length by at most 3. Data literals — metric names, condition values — are matched as the data holds them, whatever the language of the sentence around them: *"Чому margin нижчий для phones у US?"* parses like its English twin.

| Element | Rule | Demo question *"Why is margin lower for phones in the US?"* |
|---|---|---|
| metrics | name parts are the humanised words of at least 3 letters. A metric matches when all its parts match, or at least two parts including a non-generic word match, or its head word matches and that head is non-generic and unique among the metrics (generic: `median, mean, average, avg, total, number, num, count, percent, pct, rate, value, score, index`). Only the best-covered metrics are kept. "returns" finds `return_rate`, "house values" `median_house_value`, "the median" nothing | `margin` |
| conditions | a value matches when every word of it stem-matches a token; values of at most 4 upper-case characters (`US`, `EU`) must appear verbatim, so "tell us" is not `US` | `category=phones`, `region=US` |
| direction | `sign(#positive words − #negative words)` over two lexicons (`lower, drop, erosion, fewer, …` / `higher, rise, delay, slower, …`) plus Ukrainian stems matched as prefixes (`нижч, менш, спад, зниж, …` / `вищ, більш, зрост, затрим, …`); 0 without a direction word | −1 |
| relationship intent | a token starting with `correl, relationship, relation, coupl, decoupl, dependen, covari, linked, associat` or `кореляц, зв'яз, пов'яз, взаємозв, залежн, асоці` makes it a relationship question: the direction is ignored and covariance insights score in the phenomenon slot | no |

The query vector uses the composition of [4.3](04_representation.md#43-the-tripartite-vector): scope `attribute = value; …` for the recognised conditions (in recognition order), target `humanize(metric)` joined by `; `, and components `(humanize(metric), direction · 2.0)` per metric — or, for a relationship question with at least two metrics, `("correlation between a and b", ±2.0)` (−2 when the question speaks of breaking, weakening or decoupling). Without a direction word there are no components; the question text, embedded with the query instruction, stands in for every empty block — the system never assumes "higher". `ParsedQuery.to_dict()` is stored in the evidence: `{"text", "targets": ["margin"], "direction": -1, "conditions": ["category=phones", "region=US"], "covariance": false}`.

## 7.2 Seeds

`score_pattern` gives every pattern a score; for a **lexical** query (a metric or a condition was recognised):

```text
target    = 1 (primary target)  |  0.7 (a shift with |z| ≥ MIN_COMPONENT_Z)  |  0.6 (in the covariance pair)  |  0
scope     = |conditions ∩ wanted| / |wanted|  −  0.5 · #requested attributes with a different value
direction = +1 if the sign of the first material queried metric matches the question, −0.5 if opposite, 0 if unknown
            relationship question: 1 for a covariance insight whose pair contains a queried metric (any pair if none was named), else 0
semantic  = cos(v_pattern, v_query)
score     = 0.35 · target + 0.25 · scope + 0.15 · direction + 0.15 · semantic + 0.10 · w
```

and `score = 0.70 · semantic + 0.30 · w` otherwise. Seeds are the patterns in score order with `score ≥ max(SEED_MIN_SCORE (0.25), SEED_RELATIVE_MIN (0.75) · best)`, at most `SEED_TOP_K` (3), skipping direct SPECIALIZES / GENERALIZES neighbours of seeds already chosen (the walk reaches those anyway); fallback: the best semantic match. A seed's reported score is floored at `1e-3`.

> **Running example.** One seed, `P-bc4657a04746`, score 0.742 = 0.35 · 0.7 (margin is a secondary shift; its target is discount) + 0.25 · 1.0 + 0.15 · 1.0 + 0.15 · 0.087 + 0.10 · 0.843. The semantic term is small for an instructive reason: the question's phenomenon is "margin down" alone, and Qwen3 embeds "discount" and "margin" with a cosine of 0.71, so against the pattern's "discount up, margin down" the phenomenon cosine is −0.29 (`0.135 · 1 + 0.201 · 0.707 + 0.664 · (−0.285) = 0.087`). The lexical terms carry the seed.

## 7.3 Transversal traversal

`traverse(graph, seeds, config)` is a best-first search over a regular path grammar:

```text
P0 (lattice){0,h}  --ACTIVATES-->  A  (RELATED_TO){0,L}  <--ACTIVATES--  P1 (lattice){0,h}
h = STRUCTURAL_HOPS (1)     L = MAX_LATENT_HOPS (1)     total length ≤ TRAVERSAL_MAX_DEPTH (5)
```

Lattice edges are `TRAVERSAL_STRUCTURAL_EDGES` (default SPECIALIZES, GENERALIZES — followed along their stored direction — and CONTRASTS in both directions); SIBLING stays in the graph for exploration and the structural baseline. The path score is the seed score times the product of the edge factors:

| Step | Condition | Factor |
|---|---|---|
| ACTIVATES, pattern → anchor | the membership is not `weak` | alignment |
| RELATED_TO, anchor → anchor | every link the ontology kept (mutual kNN above `RELATED_TO_MIN_WEIGHT`, 0.30) | weight |
| ACTIVATES against its direction, anchor → pattern (not a seed) | not `weak` | alignment |
| lattice hop | SPECIALIZES / GENERALIZES out-edge, CONTRASTS either way | `STRUCTURAL_EDGE_DECAY` (0.85), × overlap for CONTRASTS |

The walk crosses exactly the edges the ontology kept, and the snapshot's `weak` flag says which memberships those are. A membership is `weak` — counted for coverage, drawn dashed, not walked — when it was rerouted below `MIN_ACTIVATION_ALIGNMENT` at ingest or its alignment to the living centroid has since drifted below that floor ([5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics)); a weak path would in any case rank low, since the factors multiply. Every factor is at most 1, so best-first order is meaningful. The search state is `(node, phase, structural hops used, latent hops used)` — Dijkstra over the budgeted grammar — so a higher-scoring arrival with less budget left cannot shadow one that can still expand. Per node the best path wins.

| Output | Definition |
|---|---|
| node rank | path score · `insight_weight` (seeds keep their seed score) |
| `route` | `seed`, `structural` (reached in phase P0), `transversal` (reached in phase P1, i.e. through an anchor) |
| `scope_overlap` | maximum Jaccard similarity of conditions with any seed |
| `transversal_only` | `transversal` **and** `scope_overlap = 0`: the same phenomenon in a part of the data that shares nothing with the question's subgroup |
| `structural_distance` | BFS hops over all structural edges, including SIBLING, up to 5 (`None` = unreachable within 5) |
| result | seeds first, then by rank, up to `|seeds| + MAX_RETRIEVED` (12) patterns, with their `PathStep(source, target, edge_type, weight, hop, edge_id, reverse)` lists; `used_edges` and `traversed_nodes` are the union of the selected paths |

Because the skip rule for seeds covers only SPECIALIZES and GENERALIZES, a second seed that is a CONTRASTS neighbour of a stronger one can be reported with route `structural`.

> **Running example.** From the seed the walk enters `A-1` (alignment 0.98) and descends to its other members: the three channel refinements of phones ∧ US (structural distance 1) and six tablet insights in EU and APAC that share no condition with the seed (`transversal_only`, structural distance 2–4) — among them `tablets ∧ retail ∧ EU` and `tablets ∧ retail ∧ APAC` at rank scores 0.565. One more arrives after a lattice hop: `P-de94f9a092ae (tablets ∧ APAC) -GENERALIZES-> P-33b175b166ec (tablets ∧ online ∧ APAC)`. `A-0` is visited too. 65 search states, 13 retrieved patterns, 2 anchors.

## 7.4 The evidence object

`build_evidence(parsed, result, graph, config)` takes the first `EVIDENCE_MAX_PATTERNS` (10) retrieved patterns in rank order and assigns citation keys `P1 … Pn`:

```python
Evidence(
    query, parsed,                        # question + ParsedQuery.to_dict()
    seed_patterns=["P1", ...],            # keys of the seeds
    items=[EvidenceItem(
        key="P4", pattern_id, role,       # seed | structural | transversal
        headline, scope, target, phenomenon_type,
        statistics={support, support_fraction, baseline, local, effect_size, shifts, emm_score,
                    stability, p_value, p_adjusted, weight, drivers, covariance},
        scope_text="category is tablets, channel is retail, and region is EU",
        shift_text=["discount: strong increase, +2.14 sd (median 18.91 vs 10.74 overall)", ...],
        relationship="",                  # only when the correlation change is material
        validation="bootstrap stability 0.95; adjusted p < 0.001",   # or "correlation change (...); no median test"
        path=[{source, target, edge_type, weight, hop, edge_id, reverse}, ...],   # the PathSteps as dicts
        path_text="P-bc4657a04746 -ACTIVATES(0.98)-> A-1 <-ACTIVATES(0.99)- P-ddfe04dc0882",
        attractors=[{attractor, alignment, label}], transversal_only=True,
        provenance={dataset_id, filename, batch_id, engine, steps, expression, rows_ref, pattern_id, ...})],
    attractors=[{id, label, description, n_patterns, distinct_scopes, related[{id, weight}], signature}],
    paths=[...], metrics=[{metric, dataset_id, global_median, global_mad}],
    datasets=[{dataset_id, filename, rows, batch_id}], provenance=[{key, ...}], notes=[...])
```

Rendering happens here, with the configuration: `shift_text` holds the phenomenon shifts only (the target and `|z| ≥ MIN_COMPONENT_Z`), `relationship` appears only when the correlation change is material, the scope is prose, and an anchor's `description` comes from its signature ([5.8](05_latent_anchors.md#58-how-an-anchor-is-described)). `metrics` holds the global median and MAD of every metric the prompt mentions. A note lists the items that share no scope condition with the seeds (they may still be structurally reachable; `structural_distance` says how far), or says that no pattern matched.

**The prompt** (`Evidence.to_prompt`) is ASCII for ASCII data and has fixed sections, in order: `QUESTION:`; `PARSED:` (`target=… | direction=up|down | scope=…`, or `no explicit metric/scope recognised`); `UNITS:` (shifts are robust standard deviations); `DATASETS:`; `METRIC BASELINES (whole dataset):`; `LATENT ANCHORS VISITED`; `EVIDENCE (verified statistical observations; cite as [P#]):` with one block per item, key first (scope and support; `shifts:`; `relationship:` when material; `validation:` — bootstrap stability and adjusted p, or for a covariance insight `correlation change (divergence score …) | no median test` — with the insight weight and confounders; `retrieved via:`); `NOTE:`. There is no free-form graph dump. From the demo:

```text
QUESTION: Why is margin lower for phones in the US?
PARSED: target=margin | direction=down | scope=category=phones,region=US
UNITS: shifts are robust standard deviations (sd = median difference scaled by the MAD); medians compare the subgroup with the whole dataset.

DATASETS:
- ds-6e53eb7fb0f9 file=retail_synthetic_seed7.csv batch=B20261001T112454-ff7fe6 rows=5000

METRIC BASELINES (whole dataset):
- discount: median 10.74, MAD 2.58
- margin: median 19.45, MAD 2.89
- delivery days: median 3.4, MAD 0.8
- return rate: median 0.0555, MAD 0.0101

LATENT ANCHORS VISITED (recurring phenomena learned across patterns):
- A-1 "discount up and margin down": 9 patterns over 9 distinct scopes; related: A-2 (0.31), A-0 (0.61)
- A-0 "delivery days up and return rate up": 10 patterns over 10 distinct scopes; related: A-2 (0.55), A-1 (0.61)

EVIDENCE (verified statistical observations; cite as [P#]):
[P1] role=seed | scope: category is phones and region is US | support 438 rows (8.8%)
     shifts: discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall); margin: moderate decrease, -1.10 sd (median 14.75 vs 19.45 overall)
     validation: bootstrap stability 0.97 | adjusted p < 0.001 | insight weight 0.84 | confounders: none detected
     retrieved via: seed (matched the question)
[P4] role=transversal | scope: category is tablets, channel is retail, and region is EU | support 134 rows (2.7%)
     shifts: discount: strong increase, +2.14 sd (median 18.91 vs 10.74 overall); margin: moderate decrease, -1.03 sd (median 15.03 vs 19.45 overall)
     validation: bootstrap stability 0.95 | adjusted p < 0.001 | insight weight 0.79 | confounders: none detected
     retrieved via: P-bc4657a04746 -ACTIVATES(0.98)-> A-1 <-ACTIVATES(0.99)- P-ddfe04dc0882 [scope-disjoint from seeds: no shared condition]
...
NOTE: P4, P5, P6, P8, P9, P10 share no scope condition with the seeds; they were reached through latent anchors (the same phenomenon in a different part of the data).
```

Paths use node ids and edge weights — `-TYPE(w)->` along the stored direction, `<-TYPE(w)-` against it — so the model can name the anchor it came through. A `relationship:` line (`correlation between a and b weakens from -0.57 overall to +0.09 in the subgroup (divergence 0.09)`) appears only for a material correlation change, and a covariance insight reads `shifts: no validated median shift`: its median shifts failed the shift test, so only the correlation change is cited. The prompt carries subgroup statistics, never rows.

**The evidence-only summary** (`Evidence.summary`, used whenever there is no LLM answer) follows the chat's language rule — Ukrainian around untouched literals — and is rendered from the numbers rather than from the English phrases: it starts with `Спостереження:` and has one cited line per item, `- category=phones, region=US | discount +2.21 sd (медіана 19.19 проти 10.74); margin -1.10 sd (…) | n=438 [P1]` (the phenomenon shifts, at most two; a covariance insight shows `кореляція a ~ b: +0.88 загалом → +0.73 у підгрупі`), with ` (інший сегмент: без спільної умови із запитом, знайдено через латентну тему)` for transversal-only items, and ends with `Інтерпретація (гіпотези): не сформовано (відповідь мовної моделі недоступна).` Without items it says `- У графі немає відповідних свідчень.`

## 7.5 The language model and citation check

**System prompt** (`llm.SYSTEM_PROMPT`, verbatim):

```text
You are a careful data analyst answering questions about a tabular dataset.
You receive EVIDENCE: statistically validated subgroup findings retrieved from a knowledge graph.
Shifts are robust standard deviations ("sd": the median difference scaled by the MAD).
Rules:
1. Use ONLY the evidence. Do not invent numbers, subgroups, metrics or datasets.
2. Cite every factual statement with its key, e.g. [P1] or [P2][P4].
3. Structure the answer in two labelled parts:
   "Спостереження:" - what the verified statistics show (medians, shifts in sd, support).
   "Інтерпретація (гіпотези):" - possible explanations, explicitly marked as hypotheses.
4. These are observational subgroup statistics. Do not claim causation; say "is associated with".
5. If items were reached through a latent anchor and are scope-disjoint from the seeds (no shared
   condition), point out that the same phenomenon recurs in a different part of the data.
6. If the evidence does not answer the question, say so plainly.
7. LANGUAGE: write the answer in Ukrainian. Copy every data literal byte-for-byte from the evidence, in its
   original script - column names, category values, dataset and file names, ids, "sd" and the [P#] keys.
   Never translate or transliterate them (write `margin`, `phones`, `US`, not their Ukrainian equivalents).
Be concise (at most ~250 words).
```

The language split is deliberate: the prompt (machine-to-machine) stays English, the natural-language answer is Ukrainian, and every literal that exists in the data — in whatever script the data holds it — is reproduced exactly, never translated or transliterated, so a cited value can always be found in the table. The citation check only reads `[P#]` keys, which the rule keeps intact. The same principle holds upstream: ingestion and canonicalisation keep column names and values verbatim ([2.1](02_discovery.md#21-ingestion), [4.2](04_representation.md#42-the-canonical-form)).

**Client** (`OpenAICompatibleLLM`, the only client; tests inject a duck-typed fake with `model`, `generate()` and `health()`): `POST {LLM_BASE_URL}/chat/completions` with `Authorization: Bearer <LLM_API_KEY>` and `X-Title: <LLM_APP_TITLE>` (each header only when set) and the body `{model, messages: [system, user], temperature, max_tokens, stream: false[, reasoning_effort][, provider: {order: LLM_PROVIDER_ORDER, allow_fallbacks: false}]}` — the provider block pins OpenRouter providers. Timeouts: 5 s to connect, `LLM_TIMEOUT_S` (120) for each read, write and pool wait — not a cap on the whole request. Health: `GET {base}/models` (10 s timeout), cached for 15 s — a failed check is cached too, so answers stay evidence-only for up to 15 s after an endpoint recovers; while the endpoint is known to be unreachable, generation fails fast. The code defaults point at a local Ollama (`http://localhost:11434/v1`, `gemma4`); the configured deployment uses OpenRouter (`https://openrouter.ai/api/v1`, `google/gemma-4-26b-a4b-it`, pinned bf16 providers); LM Studio works too.

**Answer modes.** With evidence and a reachable model, the answer is the model's (`answer_mode = llm`). On an LLM failure, an empty completion or `use_llm = false`, the answer is the evidence-only summary (`answer_mode = fallback`; the reason is in `llm.error`, `disabled` when switched off) — the answer text carries no status prefix, clients read the mode. An empty graph gives `answer_mode = empty`.

**Citation check** (`check_citations`): `[P#]` keys and grouped forms (`[P1, P3]`, `[P1; P3]`) are extracted; `grounded` = at least one citation and no unknown key. The provenance footer — `Sources: [P1] <pattern id> = <expression> (dataset <ds>, <file>, batch <batch>); …` — is built from the evidence, independent of the model.

**Result** (`QAResult`): `question, answer, answer_mode, llm {model, ok, latency_s, error[, usage]}, citations {cited [{key, pattern_id}], unknown, uncited, grounded}, evidence (dict + "prompt"), traversal (dict incl. baselines), highlight {seeds, traversed, anchors, evidence, edges, transversal_only}, metrics {retrieval_s, total_s, llm_latency_s, seed_count, traversal_depth, visited_states, retrieved_evidence, anchors_visited, prompt_chars}, provenance_footer`. `traversed` and `edges` are the union of the paths to all retrieved patterns (up to `|seeds| + MAX_RETRIEVED`, 15), while `evidence`, `transversal_only`, the evidence object and the prompt hold the first 10 — the graph can light up paths to a few patterns the model did not see.

## 7.6 Baselines

`compute_baselines` stores, with every answer, what simpler retrieval would have returned: `structural_only` (a BFS from the seeds over the walk's lattice edge types — no SIBLING — in either direction, up to `TRAVERSAL_MAX_DEPTH` hops), `naive_nearest` (the top 12 patterns by `cos(E_query(question), E(canonical document))` — plain text RAG), and the set differences `transversal_only`, `not_in_naive_topk`, `not_structurally_reachable`. The naive baseline compares the question with document vectors embedded once at ingest and stored beside the journal rows (`journal/blocks/<batch>.npz`, read into the committed frame), so an answer embeds only the question and no canonical document, and does not compete with an ingest for the model. A pattern without a stored vector (a batch written before documents were stored) is embedded once, on the first question: `Engine.document_vectors()` keeps the vector on the committed frame and, in a writer process, saves it into that batch's blocks file. The same vectors serve the `text_nn` ranker of the benchmark. For the demo question: the structural closure holds 23 patterns, and 7 of the 13 retrieved patterns are not in the naive top 12. The benchmark of [10.3](10_verification.md#103-hypothesis-benchmark) scores these rankers against planted ground truth.

## 7.7 Measured behaviour

Live against `google/gemma-4-26b-a4b-it` on OpenRouter (`scripts/eval_answers.py`, five fixed demo questions, the model's own token counts):

| Configuration | Grounded | Unknown citations | Citations | Prompt tokens | Completion tokens | Mean latency |
|---|---|---|---|---|---|---|
| symbol-based prompt, MiniLM | 5/5 | 0 | 34 | 14,983 | 1,881 | 9.0 s |
| ASCII prompt, MiniLM | 5/5 | 0 | 38 | 11,705 | 1,668 | 6.0 s |
| ASCII prompt, Qwen3 (current) | 5/5 | 0 | 40 | 11,099 | 1,728 | 6.6 s |

Latency depends on the provider. Answers keep the planted directions (US phones: lower margin, higher discount; the EU∧phones correlation −0.57 → +0.09), separate *Observations* from *Interpretation (hypotheses)*, cite the scope-disjoint analogues reached through `A-1`, and use grouped citations, which are parsed and linked.

Prompt size (`scripts/prompt_tokens.py`, demo question, 10 items, scratch run with the default hashing backend — the documents do not depend on the backend, the evidence prompt does through retrieval; XLM-R SentencePiece as a proxy for Gemma's tokenizer, and Qwen3's byte-level BPE):

| Surface | Characters | Non-ASCII | XLM-R tokens | Qwen tokens |
|---|---|---|---|---|
| evidence prompt, symbol-based format | 7,009 | 25 | 2,421 | 2,880 |
| evidence prompt, ASCII prose (current) | 6,304 | 0 | 2,096 (−13 %) | 2,342 (−19 %) |
| canonical documents (28): symbol-based → current | 11,406 → 15,129 | 0 | 3,455 → 4,378 | 4,224 → 5,029 |
| evidence-only answer: symbol-based → current | 1,730 → 2,509 | 7 → 0 | 588 → 779 | 760 → 887 |

The documents and the summary grew because they are written for people; no LLM reads the documents (only the naive baseline and the `text_nn` ranker use their embeddings). Retrieval for one question takes about 0.1 s with the model loaded, the first question included.

## 7.8 Configuration, failure modes and limitations

| Parameter | Default |
|---|---|
| `SEED_TOP_K`, `SEED_MIN_SCORE`, `SEED_RELATIVE_MIN` | 3, 0.25, 0.75 |
| `STRUCTURAL_HOPS`, `MAX_LATENT_HOPS`, `TRAVERSAL_MAX_DEPTH` | 1, 1, 5 |
| `STRUCTURAL_EDGE_DECAY`, `TRAVERSAL_STRUCTURAL_EDGES` | 0.85, `SPECIALIZES,GENERALIZES,CONTRASTS` |
| `MAX_RETRIEVED`, `EVIDENCE_MAX_PATTERNS` | 12, 10 |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_PROVIDER_ORDER`, `LLM_APP_TITLE` | `http://localhost:11434/v1`, `gemma4`, "", "", `SIG LTIR` |
| `LLM_TIMEOUT_S`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_REASONING_EFFORT` | 120, 0.1, 1200, "" (`none` disables Gemma's thinking where supported) |

| Situation | Behaviour |
|---|---|
| empty graph | `answer_mode = empty`; `llm`, `citations`, `evidence`, `traversal` are empty and `metrics = {"error": "empty_graph"}` |
| no lexical match | semantic seeds |
| no walkable path to an anchor | only seeds and their lattice neighbours; "empty retrieval" shows as zero transversal items |
| LLM unavailable, HTTP error, empty completion | evidence-only summary; graph and statistics untouched |
| representation mismatch | refused before retrieval (HTTP 409) |

Limitations worth knowing:
* The words "associated" and "linked" make a question a relationship question (direction ignored), and the system prompt asks the model to write "is associated with" — a follow-up that reuses the model's wording is parsed as a relationship question.
* Parsing is lexical plus embedding similarity; paraphrases outside the graph's vocabulary rely on the semantic term alone.
* The LLM runs remotely in the configured deployment: the prompt (subgroup statistics, not rows) leaves the machine; a local server keeps it local.

Guarantees: every retrieved node has a path from a seed whose steps are graph edges (their `edge_id`s resolve in the snapshot and the UI); weak memberships are never walked and every kept RELATED_TO is; paths obey the grammar; results are deterministic; citation keys map one-to-one to pattern ids; every item carries dataset, batch, selector and pattern id; every number in the prompt comes from the stored insights.
