# 7. Question answering — seeds, transversal retrieval, evidence, language model

> **In one paragraph.** A question is parsed against the graph's own vocabulary — metric names, condition values, direction words — and projected with the same composition as an insight. The best-matching insights become *seeds*. A budgeted best-first walk then follows lattice edges and, through ACTIVATES, the latent anchors, collecting the insights it reaches with the exact path that reached them. The first ten become an evidence object: the only thing the language model is shown, rendered as an ASCII prompt with citation keys. The graph service hands that prompt, a deterministic summary of the same evidence and a citation manifest to the narrator as one `EvidencePayload`. The narrator has the model verbalise the evidence, checks the answer against the keys, builds a provenance footer without the model, and answers with the summary when no model is available.

**Code** `graph_query_engine/` (`question.py`, `seeds.py`, `traversal.py`, `evidence.py`, `search.py`, `graph.py`, `ports.py`), `insight_contracts/payload.py`, `evidence_narrator_service/` (`narration.py`, `llm_client.py`, `settings.py`, `app.py`), `llm_model_broker/` · **Tests** `tests/test_traversal.py`, `tests/test_multilingual_retrieval.py`, `tests/test_graph_query_engine_standalone.py`, `tests/test_narrator.py`, `tests/test_llm_client.py`, `tests/test_llm_model_broker.py`, `tests/test_e2e.py` · **Previous** [6. Graph and storage](06_graph_and_storage.md) · **Next** [8. Interface](08_interface.md)

---

A question crosses two services. The graph service retrieves (`POST /api/search` → `Engine.evidence` → `search.search`) and returns the `EvidencePayload`; the narrator verbalises it (`POST /api/chat/query` → `narration.answer`):

```text
graph service:  parse_query → resolve_seeds → traverse → compute_baselines → build_evidence → SearchResult.payload()
narrator:       LLM: generate(SYSTEM_PROMPT, evidence_prompt)   or   evidence_summary
                → check_citations → provenance footer → one JSON line on the query log
```

Before retrieval the workspace's fingerprint is checked against the encoder's (`RepresentationMismatch` → HTTP 409, code `representation_mismatch`), and a degraded workspace is refused with `workspace_degraded` ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)); the narrator passes both refusals on as sent. An empty graph returns the empty payload (`graph_empty: true`) at once, and the narrator answers `answer_mode = empty`.

`graph_query_engine` reads a compiled snapshot (`graph.DualGraph`) and the committed vectors (`graph.LatentFrame`) and never imports the representation package: the question is embedded by any object with the `ports.QueryEncoder` shape (`encode_query`, plus an `embedder` with `embed` and `embed_queries`, `ports.Embedder`). In the graph service that object is `attractor_topology.InsightEncoder`; most tests wrap the hashing double of `tests/doubles.py` in it.

## 7.1 From question to query

`parse_query(text, graph, config, catalog)` is deterministic and uses only the graph's vocabulary — no LLM. Tokens are runs of letters and digits in any script (underscores split, so `return_rate` is `return`, `rate`; an apostrophe stays inside a word), lower-cased; two words stem-match when they are equal, or both have at least 5 characters, share the first 5 and differ in length by at most 3. Data literals — metric names, condition values — are matched as the data holds them, whatever the language of the sentence around them: *"Чому margin нижчий для phones у US?"* parses like its English twin.

| Element | Rule | Demo question *"Why is margin lower for phones in the US?"* |
|---|---|---|
| metrics | name parts are the humanised words of at least 3 letters. A metric matches when all its parts match, or at least two parts including a non-generic word match, or its head word matches and that head is non-generic and unique among the metrics (generic: `median, mean, average, avg, total, number, num, count, percent, pct, rate, value, score, index`). Only the best-covered metrics are kept. "returns" finds `return_rate`, "house values" `median_house_value`, "the median" nothing. Words that match nothing are grounded against the literal catalog ([7.1.1](#711-literal-grounding)): *дні доставки* → `delivery_days` | `margin` |
| conditions | a value matches when every word of it stem-matches a token; values of at most 4 upper-case characters (`US`, `EU`) must appear verbatim, so "tell us" is not `US`. A value under several columns binds as a grounded one does ([7.1.1](#711-literal-grounding)): the column named within two tokens, else the wildcard `("*", value)`. Otherwise grounded ([7.1.1](#711-literal-grounding)): *телефонів* → `category=phones`, *США* → `region=US`, *Харкові* → `city=Харків` | `category=phones`, `region=US` |
| direction | counted over the tokens no literal took (a metric `delivery_delay` casts no "delay" vote): `sign(#positive words − #negative words)` over two lexicons (`lower, drop, erosion, fewer, …` / `higher, rise, delay, slower, …`) plus Ukrainian stems matched as prefixes (`нижч, менш, спад, знижен, …` / `вищ, більш, зрост, затрим, …`); a comparative adverb (`більш, більше, менш, менше, more, less`) directly before a direction word casts one vote with it — *більш низький* is one "down", not an "up" and a "down" that cancel; 0 without a direction word | −1 |
| relationship intent | a token (again outside the literals) starting with `correl, relationship, relation, coupl, decoupl, dependen, covari, linked, associat` or `кореляц, зв'яз, пов'яз, взаємозв, залежн, асоці` makes it a relationship question: the direction is ignored and covariance insights score in the phenomenon slot — unless the phrase is a directed driver question, `пов'язано з / associated with / linked to` followed by a direction word (*Що пов'язано з вищим return_rate?*, *What is associated with higher discount?*), which keeps its direction | no |

The query vector uses the composition of [4.3](04_representation.md#43-the-tripartite-vector): scope `attribute = value; …` for the recognised conditions (in recognition order), target `humanize(metric)` joined by `; `, and components `(humanize(metric), direction · 2.0)` per metric — or, for a relationship question with at least two metrics, `("correlation between a and b", ±2.0)` (−2 when a word of the question starts with a breaking, weakening or decoupling stem — *close* is not *lose*). Without a direction word there are no components; the question text, embedded with the query instruction, stands in for every empty block — the system never assumes "higher". `ParsedQuery.to_dict()` is stored in the evidence: `{"text", "targets": ["margin"], "direction": -1, "conditions": ["category=phones", "region=US"], "covariance": false, "grounding": [...]}` (a wildcard condition, [7.1.1](#711-literal-grounding), is written as the bare value).

### 7.1.1 Literal grounding

The lexical rules above read a literal only as the data spells it. A question in another language — *Чому маржа нижча для телефонів у США?* — names the same literals in other words, and without grounding it falls back to the unconditioned semantic score and lands on unrelated subgroups. *Decoupled multilingual literal grounding* (`question._ground`, after the design in `docs/init_concepts/multilingual_graph_retrieval_architecture.md`) resolves the words the lexical layer left over onto the graph's **literal catalog** before seed scoring; nothing is translated, no language model is consulted, and the structural scoring, the traversal and Neo4j are untouched.

**The catalog** (`question.LiteralCatalog`, built by `build_catalog` from the committed graph): every Metric node's name, raw and humanised (`return_rate`, `return rate`); and every distinct condition value with the attributes it occurs under (`phones` → `category`; a value under several columns keeps them all). Values of at most 4 upper-case characters are *case-sensitive* (`US`, `EU`, `Q4`). The texts are embedded once with the document-side embedder (as component labels are, [4.4](04_representation.md#44-the-embedding-model)), **centred by the catalog mean and re-normalised** — the embedding space is anisotropic (every literal sits at cosine 0.6–0.85 of every other; *у* and *для* score 0.75–0.81 against `margin` raw), and centring is what makes the cosines discriminative — and indexed by script with a character n-gram TF-IDF (`char_wb`, 3–5). The graph service builds it from the committed graph on the first question after a commit (`Engine._prepare`), under its own lock (a question never waits for a running batch), and keeps it on the committed state (`CommittedState.catalog`) until the next commit; `search.search` passes it to `parse_query`. A caller of the library that brings no catalog gets one built by `search.search` on first use and kept on its state. Entries always come from the graph; the service caches only the vectors, by text, in `graph/literals.npz` (with the representation fingerprint): a known literal is never embedded twice, a new one is embedded and a writer saves the file again. Only a question builds the catalog: a start, a batch or a deletion embeds no literal.

**Spans.** The question's tokens are grouped into spans of 3, 2 and 1 words, longest first, skipping tokens the lexical pass, the direction lexicons or the relationship words already claimed (so an exact literal never reaches the layers below), spans that start or end with a stop word (≈ 40 English and ≈ 30 Ukrainian question and function words; *для телефонів у* is never tried, *телефонів* is), and one-letter tokens; short upper-case tokens (acronyms) are always tried. An accepted span **claims its tokens**, so overlapping spans are suppressed (non-maximum suppression) and *дні доставки* grounds once, not three times.

**Two layers, the first that accepts wins**:

```text
chars      same script, span ≥ 4 chars, same word count: TF-IDF char_wb(3–5) cosine ≥ CHAR_MIN (0.30),  |len(span) − len(literal)| ≤ 3
dense      other script, |words(span) − words(literal)| ≤ 1, on centred unit vectors c:
           cos = c(span)·c(literal) ≥ GROUNDING_MIN_COSINE (0.30)
           margin = cos − mean of the 5 highest cosines of the span over the catalog ≥ 0.15     (one-sided local scaling: hubs lose)
           Lowe: (1 − cos) / (1 − cos₂) ≤ 0.85, cos₂ = best literal with a *different* symbol    (not the raw/humanised twin)
           an acronym literal needs an upper-case span (США → US; us → nothing)
```

The character layer handles inflection and typos within a script — *Харкові* → `Харків`, *телефонів* → `телефони`, *fones* → `phones`, *retrun rate* → `return rate` — and the length guard is what rejects *marginally* → `margin` (0.89, but four letters longer). The dense layer bridges scripts and never compares a Latin span with a Latin literal (that is the character layer's job, and it is why *sales*, *store*, *profit* ground to nothing). A span of several words grounds only onto a literal of about as many words: *телефонів у США* is not `phones`; *дні доставки* is `delivery days`. Direction stays with the lexicons and the composite rules: the design's multilingual direction anchors are not part of the catalog, since no span can reach them (their 4–5 words against spans of at most 3 and the word-count gate).

**What a grounded symbol does.** A metric joins `targets`; a value joins `conditions` as `(attribute, value)` — when the value lives under several columns, the column named within two tokens of the span wins (the same rule as for a value typed as stored), otherwise the condition is a **wildcard** `("*", value)` that `score_pattern` matches against any column holding the value and never counts as a conflict (so `US` under `origin` and `destination` does not halve the scope score). Every accepted span is recorded in `ParsedQuery.grounding` as `{span, literal, layer, score, symbol}`: the UI shows it as *Understood: маржа → margin · …*, and the benchmark's false-positive rate is computed from it. Because the grounded strings are the canonical literals, a fully grounded Ukrainian question gets the English twin's query vector and the same seeds.

**Measured on Qwen3 → 384 (the demo catalog; `scripts/multilingual_benchmark.py`, §7.7).** Centred cosines of true translations: *телефонів* → `phones` 0.57, *США* → `US` 0.69, *ЄС* → `EU` 0.60, *планшетів* → `tablets` 0.63, *онлайн* → `online` 0.63, *дні доставки* → `delivery days` 0.54, *знижка* → `discount` 0.46, *ноутбуків* → `laptops` 0.48, with margins 0.19–0.45; Ukrainian function words stay ≤ 0.33 with margins ≤ 0.14. Three literals this model cannot bridge are rejected rather than mis-grounded: *маржа* (0.32 to `retail`, `margin` second — Lowe 0.93), *частка повернень* (`return rate` not among the nearest), *роздріб* (`retail` not nearest). Character cosines on the demo catalogs: inflections 0.33–0.83, typos 0.36–0.70; distractors *сегменти* 0.22, *sales* 0.23.

**Constants** (`graph_query_engine/question.py`): `CHAR_MIN` 0.30, `CHAR_MAX_LEN_DIFF` 3, `MARGIN_MIN` 0.15, `MARGIN_K` 5, `LOWE_MAX` 0.85, `MAX_SPAN` 3, `ACRONYM_MAX_LEN` 4, `STOPWORDS`; `GROUNDING_MIN_COSINE` (0.30) is a `QueryConfig` field because it belongs to the embedder, like `MIN_ASSIGN_THRESHOLD` in `TopologyConfig`. The values are calibrated on this embedder and the demo catalog (`scripts/multilingual_benchmark.py`, §7.7).

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
- ds-6e53eb7fb0f9 file=retail_synthetic.csv batch=B20261004T005725-2b9928 rows=5000

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

**The evidence-only summary** (`Evidence.summary`, carried as the payload's `evidence_summary`) follows the chat's language rule — Ukrainian around untouched literals — and is rendered from the numbers rather than from the English phrases: one cited line per item, `- category=phones, region=US | discount +2.21 sd (медіана 19.19 проти 10.74); margin -1.10 sd (…) | n=438 [P1]` (the phenomenon shifts, at most two; a covariance insight shows `кореляція a ~ b: +0.88 загалом → +0.73 у підгрупі`), with ` (інший сегмент: без спільної умови із запитом, знайдено через латентну тему)` for transversal-only items; without items, `- У графі немає відповідних свідчень.` Whenever there is no LLM answer, the narrator frames these lines as the answer: `Спостереження:` before them and `Інтерпретація (гіпотези): не сформовано (відповідь мовної моделі недоступна).` after them (`narration._explain`).

## 7.5 The language model and citation check

Retrieval and the answer run in two services. In the graph service, `Engine.search(question)` runs §7.1–7.4 and §7.6 (`search.search` over the committed graph, frame and literal catalog) and returns a `SearchResult`: the parsed question, the seeds, the walk with its baselines, the evidence object and the retrieval time. No language model is involved. `Engine.evidence(question)` returns its `SearchResult.payload()`, or `empty_payload(question)` while the graph holds no insight. That payload is the body of `POST /api/search` and the whole contract between the two services; its fields are listed in [12.4](12_architecture.md#124-services-and-contracts). This chapter produces them: `evidence_prompt` is `Evidence.to_prompt()` ([7.4](#74-the-evidence-object)), `evidence_summary` is `Evidence.summary()`, `citations` is the manifest the citation check reads, and `view` holds `Evidence.to_dict()` (with the parse `ParsedQuery.to_dict()` under `parsed`), the walk with its baselines ([7.6](#76-baselines)) and `SearchResult.highlight()`.

**The narrator** (`evidence_narrator_service/app.py`) serves `POST /api/chat/query` with `{question, use_llm}`. It asks the graph service for the payload and handles the outcomes:
- an empty question is a 400;
- a 409 is passed on as sent (`{detail, code}`);
- an unreachable graph service, any other status or a payload that breaks the contract is a 502;
- otherwise it answers with `narration.answer(payload, llm)`, where `llm` is `None` when `use_llm` is false.

The answer reads only the payload: the model or the evidence-only summary, the citation check against the manifest, the footer; `view` goes to the console untouched. Every answer on a non-empty graph is logged as one JSON line on the logger `evidence_narrator_service.queries`: `at`, `question`, `mode`, `metrics`, `evidence` (the pattern ids of the manifest) and `citations`. `GET /api/chat/health` always answers 200 with the LLM status (`health()`, below) and whether the graph service's `GET /api/health` answers within 2 s. The narrator reaches the graph service at `INSIGHT_GRAPH_URL` with `INSIGHT_GRAPH_TIMEOUT_S` (300 s: the first question after a commit builds the literal catalog, tens of seconds on a CPU) and a 5 s connect timeout.

**System prompt** (`narration.SYSTEM_PROMPT`, verbatim):

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

**Client** (`llm_client.OpenAICompatibleLLM`, built from `NarratorSettings`, the only client; anything with `generate()` and `health()` — the `llm_client.ChatModel` contract — can stand in, and the tests use a fake): `POST {LLM_BASE_URL}/chat/completions` with the body `{messages: [system, user], temperature, max_tokens, stream: false}`, without a model name or a key. `LLM_BASE_URL` is the LLM model broker ([12.4](12_architecture.md#124-services-and-contracts)), which sets the model and holds the upstream, its key and the OpenRouter provider pinning; a local upstream (Ollama, LM Studio) is configured in the broker too. Timeouts: 5 s to connect, `LLM_TIMEOUT_S` (120) for each read, write and pool wait — not a cap on the whole request. A refused connection is remembered for 15 s (`HEALTH_TTL_S`), during which answers fall back at once instead of retrying; an HTTP error answer keeps its status and the start of its body in `llm.error`. The status display (`health()`, for `GET /api/chat/health` and the console's LLM pill) reads `GET {base}/models`, cached for 15 s: reachable, and the model the broker serves (the first one it lists; `model_available` when it lists one). The configured deployment reaches Gemma 4 on OpenRouter (`google/gemma-4-26b-a4b-it`, pinned bf16 providers) through the broker.

**Answer modes.** With evidence and a reachable model, the answer is the model's (`answer_mode = llm`). On an LLM failure, an empty completion or `use_llm = false`, the answer is the payload's evidence-only summary (`answer_mode = fallback`; the reason is in `llm.error`, `disabled` when switched off) — the answer text carries no status prefix, clients read the mode. An empty graph gives `answer_mode = empty` (HTTP 200).

**Citation check** (`narration.check_citations`): `[P#]` keys and grouped forms (`[P1, P3]`, `[P1; P3]`) are extracted and compared with the citation manifest; `grounded` = at least one citation and no unknown key. The provenance footer — `Sources: [P1] <pattern id> = <expression> (dataset <ds>, <file>, batch <batch>); …` — is built from the manifest, independent of the model.

**Result** (`narration.QAResult`, the body of `POST /api/chat/query`; its fields are listed in [12.4](12_architecture.md#124-services-and-contracts)). The narrator fills `llm` from the model's response, `citations` from the citation check and `prompt` with the evidence prompt the model is shown, passes `view` on as received, and adds `total_s`, `llm_latency_s` and `prompt_chars` to the payload's `metrics`. `retrieval_s` is measured in the graph service by `Engine.evidence`: the representation check, the document back-fill and the literal catalog a first question builds, the parse, seeds, walk, baselines and the evidence object. `total_s` adds the narrator's part (the model, the citation check), not the HTTP hop between the services. The highlight groups come from `SearchResult.highlight()` in the graph service, so the evidence of `POST /api/search` carries them without an answer. `traversed` and `edges` are the union of the paths to all retrieved patterns (up to `|seeds| + MAX_RETRIEVED`, 15), while `evidence`, `transversal_only`, the evidence object and the prompt hold the first 10 — the graph can light up paths to a few patterns the model did not see.

## 7.6 Baselines

`search.compute_baselines` stores, with every search, what simpler retrieval would have returned: `structural_only` (a BFS from the seeds over the walk's lattice edge types — no SIBLING — in either direction, up to `TRAVERSAL_MAX_DEPTH` hops), `naive_nearest` (the top 12 patterns by `cos(E_query(question), E(canonical document))` — plain text RAG), and the set differences `transversal_only`, `not_in_naive_topk`, `not_structurally_reachable`. The naive baseline compares the question with document vectors embedded once at ingest and stored beside the journal rows (`journal/blocks/<batch>.npz`, read into the committed frame), so an answer embeds only the question and no canonical document, and does not compete with an ingest for the model. A pattern without a stored document vector is embedded once, on the first question (`Engine._prepare`): the vector stays on the committed frame and, in a writer process, is saved into that batch's blocks file. A caller of the library alone gets no back-fill: `compute_baselines` leaves a pattern without a document vector out of the naive ranking. The same vectors serve the `text_nn` ranker of the benchmark. For the demo question: the structural closure holds 23 patterns, and 7 of the 13 retrieved patterns are not in the naive top 12. The benchmark of [10.3](10_verification.md#103-hypothesis-benchmark) scores these rankers against planted ground truth.

## 7.7 Measured behaviour

Live against `google/gemma-4-26b-a4b-it` on OpenRouter (`scripts/eval_answers.py`, five fixed demo questions, Qwen3 → 384, the model's own token counts), measured on 2026-10-01 with an earlier wording of the system prompt that asked for English answers:

| Grounded | Unknown citations | Citations | Prompt tokens | Completion tokens | Mean latency |
|---|---|---|---|---|---|
| 5/5 | 0 | 40 | 11,099 | 1,728 | 6.6 s |

The five evidence prompts hold 4,713–6,164 characters, none of them outside ASCII, and 2,010–2,543 prompt tokens each. Latency depends on the provider. Answers keep the planted directions (US phones: lower margin, higher discount; the EU∧phones correlation −0.57 → +0.09), separate the observations from the interpretation as rule 3 asks (with the current prompt under `Спостереження:` and `Інтерпретація (гіпотези):`), cite the scope-disjoint analogues reached through `A-1`, and use grouped citations, which are parsed and linked. `scripts/eval_answers.py` re-measures these numbers with the current prompt; it calls the LLM behind the broker, a paid call.

**Multilingual grounding benchmark** (`scripts/multilingual_benchmark.py`: 60 questions in four buckets of 15, on the demo and on a copy of the demo with Ukrainian category, city and channel values; every question's twin — the same question with each literal as stored — gives the gold seeds and evidence; the model's own embeddings, no LLM). Seed Recall@3 = gold seeds among the question's seeds; Consistency = Jaccard of the seed sets; Evidence overlap over the gold evidence; direction accuracy against the twin; FPGR = grounded spans whose symbol the twin does not hold:

| B1 English | B2 code-switched | B3 translated | B4 inflections / typos | FPGR |
|---|---|---|---|---|
| 1.00 / 1.00 | 1.00 / 1.00 | 0.77 / 0.69, direction 1.00, evidence overlap 0.83 | 0.98 / 0.97, evidence 1.00; symbol accuracy 1.00 (*маржа* → `margin` passes at 0.31 on the Ukrainian-value copy) | 0 |

Cells are Recall@3 / Jaccard. The B3 misses all contain *маржа*, *частка повернень* or *роздріб* (above). Retrieval time per question with the model on a GPU: 70–90 ms for English and code-switched questions, 130–200 ms for a translated one (one embedding pass over its 2–4 unresolved spans).

No LLM reads the canonical documents: only the naive baseline and the `text_nn` ranker use their embeddings. Retrieval for one question takes about 0.1 s with the model loaded (the benchmark's median); the first question after a start or a commit also builds the literal catalog and takes seconds (9 s in the benchmark, on a GPU).

## 7.8 Configuration, failure modes and limitations

| Parameter | Default |
|---|---|
| `GROUNDING_MIN_COSINE` | 0.30 (centred cosine floor of the dense grounding layer, [7.1.1](#711-literal-grounding); embedder-specific) |
| `SEED_TOP_K`, `SEED_MIN_SCORE`, `SEED_RELATIVE_MIN` | 3, 0.25, 0.75 |
| `STRUCTURAL_HOPS`, `MAX_LATENT_HOPS`, `TRAVERSAL_MAX_DEPTH` | 1, 1, 5 |
| `STRUCTURAL_EDGE_DECAY`, `TRAVERSAL_STRUCTURAL_EDGES` | 0.85, `SPECIALIZES,GENERALIZES,CONTRASTS` |
| `MAX_RETRIEVED`, `EVIDENCE_MAX_PATTERNS` | 12, 10 |
| `INSIGHT_GRAPH_URL`, `INSIGHT_GRAPH_TIMEOUT_S` | `http://127.0.0.1:8765` (the host composite of `scripts/dev.py`; compose sets `http://insight-graph:8000`), 300 |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` (the broker) |
| `LLM_TIMEOUT_S`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` | 120, 0.1, 1200 |
| the upstream model, its key and the provider order | the broker's `GEMMA_*` settings ([12.4](12_architecture.md#124-services-and-contracts)) |

The retrieval parameters are fields of `QueryConfig` (`graph_query_engine/config.py`), read by the graph service; `INSIGHT_GRAPH_*` and `LLM_*` are fields of `NarratorSettings` (`evidence_narrator_service/settings.py`), read by the narrator ([9.3](09_operations.md#93-configuration)).

| Situation | Behaviour |
|---|---|
| empty graph | the empty payload; `answer_mode = empty` (HTTP 200); `llm` and `citations` are empty, `prompt` is "", `view` holds an empty evidence object, an empty traversal and empty highlight groups, and `metrics = {"error": "empty_graph"}` |
| no lexical match | semantic seeds |
| no walkable path to an anchor | only seeds and their lattice neighbours; "empty retrieval" shows as zero transversal items |
| LLM unavailable, HTTP error, empty completion | evidence-only summary; graph and statistics untouched |
| representation mismatch, degraded workspace ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)); graph service unreachable, or a payload that breaks the contract | the first two are refused before retrieval and passed on by the narrator as sent, the last two give a 502 from the narrator (statuses and bodies: [12.4](12_architecture.md#124-services-and-contracts)) |

Limitations worth knowing:
* "Associated" and "linked" make a question a relationship question (direction ignored) unless they introduce a direction word (*associated with higher …*); a follow-up that reuses the model's "is associated with" without one is parsed as a relationship question.
* Grounding reaches the literals the embedder knows in the question's language; on the demo, Qwen3 → 384 does not bridge *маржа*, *частка повернень* or *роздріб*, and such a question keeps only the literals it did ground (a wrong literal is never substituted). The thresholds are measured for Qwen3 → 384; another embedder needs `GROUNDING_MIN_COSINE` re-measured (`scripts/multilingual_benchmark.py`).
* Paraphrases outside the graph's vocabulary and its translations rely on the semantic term alone.
* The LLM runs remotely in the configured deployment: the prompt (subgroup statistics, not rows) leaves the machine; a local server keeps it local.

Guarantees: every retrieved node has a path from a seed whose steps are graph edges (their `edge_id`s resolve in the snapshot and the UI); weak memberships are never walked and every kept RELATED_TO is; paths obey the grammar; results are deterministic and identical with the Neo4j mirror on or off; citation keys map one-to-one to pattern ids; every item carries dataset, batch, selector and pattern id; every number in the prompt comes from the stored insights; a grounded literal is always one the data holds, spelled as the data spells it.
