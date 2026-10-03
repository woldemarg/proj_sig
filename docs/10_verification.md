# 10. Verification — tests, benchmark, measurements, limitations

> **In one paragraph.** The test suite guards every transition whose breakage would invalidate the system, from the EDA contract to the highlighted path in a real browser. A deterministic synthetic dataset with planted mechanisms makes the statistics checkable, and a benchmark turns the research hypothesis into a measurement against structural and text baselines. Measurement scripts reproduce every number quoted in these docs on throwaway workspaces. The chapter ends with what the system approximates and what it does not do.

**Code** `tests/`, `ltir/evaluation/synthetic.py`, `ltir/evaluation/experiment.py`, `scripts/` · **Previous** [9. Operations](09_operations.md) · **Next** [11. Reference](11_reference.md)

---

## 10.1 Test map

102 collected tests (101 functions; the embedding contract runs for both the hashing backend and the real model). `python -m pytest` from the repository root; about two minutes with the model and browser tests.

| Transition or contract | File (tests) | What is asserted |
|---|---|---|
| discovery → insight | `test_discovery_contract.py` (10) | profile contract (identifier dropped, float targets kept, EDA dimensions, candidates within the search space); fields and provenance; signed directions of the planted effects; determinism; planted phenomena survive selection; `closed_intent` adds implied conditions; identical extents merge before validation; near duplicates pruned in rank order; extent/intent antitone; `no_numeric_targets` and `invalid_schema` |
| selection and weight | `test_quality.py` (4) | weight bounds and monotonicity, the formula with all five factors measured, reason codes, the R7 budget, covariance retargeting with unmeasured factors and the shift's p-values and stability `None` |
| canonical → embedding | `test_canonical_embedding.py` (9) | embedding inputs vs document (sections, ASCII); labels without numbers; number rules; covariance component; shape, dtype, unit norms, stability, direction separation and cross-scope similarity for the hashing backend **and the real model**; fingerprint sensitivity; query instruction only on free question text; the MiniLM env block applies verbatim |
| insight → ontology | `test_ontology.py` (11) | cold start coverage, at least three anchors, each planted cluster within one anchor; the orphan rule (cases: no members and no link → dropped, no members but linked → kept, members → kept); `forget` recounts, renumbers and drops; assignment, orphans and new anchors; soft merge; weight scales the EMA pull; `τ_density`; damping without membership change; trust region; sign repair; state round-trip |
| structural plane | `test_structural.py` (5) | covering relation only, GENERALIZES inverse, SIBLING rule, CONTRASTS rule and relation type, no embedding dependency |
| persistence | `test_persistence.py` (17) | write → reload → rebuild identical; a failed READY write rolls back the snapshot too; an interrupted deletion (error or crash) is undone; dataset deletion (artefacts gone — the literal cache and a standalone sphere page included —, the orphan rule anchor by anchor, journal/frame/ontology consistent, snapshot = rebuild, re-ingest possible, last dataset empties the ontology); a record rewritten under a concurrent reader (no sharing violation on either side); a reset during a batch refused at once (`busy`) while readers keep the committed state and the batch still commits; idempotent re-ingest; graph consistency (incl. every membership below the alignment floor is weak); rollback; a failed rollback keeps its marker, refuses later writes (`rollback_pending`) and is finished by the next writer start; representation mismatch; a batch killed after its journal append rolled back at the next start; a second writer process refused while the first keeps its queue; migration with backup; READY survives a post-commit save failure; ingestion failure codes; options strict when explicit, lenient as defaults; the Neo4j mirror equals the snapshot |
| literal grounding | `test_multilingual_retrieval.py` (12) | the dense gates on a dictionary embedder (translations accepted, `us` rejected by case, a hub literal demoted, an equidistant span rejected by Lowe); character matches for same-script inflections and typos only, the length guard; longest span claims its tokens and one embedding pass; a value under two columns resolves to the named column or a wildcard that scores without conflict, typed or translated alike; words inside a matched literal cast no direction, and a weakening word is a word, not a substring; Ukrainian comparatives and directed drivers; B1 = B2 seeds (hashing); the catalog vectors persisted and reloaded without re-embedding, the catalog cached until the next commit; B3 translated twin parses and seeds like English, distractors ground to nothing (`model`); B4 inflected Ukrainian value (`model`); identical seeds with the Neo4j mirror on (fake driver) and off |
| traversal and evidence | `test_traversal.py` (4) | the exact pattern → anchor → anchor → pattern path with hops, weights, reversal and score; a weak membership is not walked while a kept 0.35 link is; hop and depth budgets; parsing against the graph vocabulary in English and in Ukrainian around the same literals; the evidence object |
| LLM client | `test_llm_client.py` (8) | the real HTTP client against an OpenAI-compatible stub (a plain OpenAI payload, health, an Ollama tag answering to its name and no other prefix); an unset `LLM_MODEL` takes the one model the endpoint lists, an upstream error keeps its status and reason; an endpoint without `/models` works with `LLM_MODEL` set; an unset model against a multi-model endpoint is refused (`set LLM_MODEL`); a refused connection fails fast and is remembered; a retired setting still set is named in the log; citation validation incl. grouped citations |
| LLM gateway | `test_llm_gateway.py` (3) | the gateway forwards to the configured model with the pinned provider order and its bearer key, replacing the requested model and passing the other fields as sent; an upstream 429 keeps its status and `Retry-After`, a 5xx → 502, empty messages, `stream: true` or a malformed body → an OpenAI-shaped 400, missing settings → 503 with a live `/health`; the whole chain backend client → gateway → upstream over local HTTP, the key only in the gateway |
| end to end | `test_e2e.py` (8) | upload → … → grounded answer with provenance and cross-scope analogues (hashing and real model); the benchmark ordering (`test_hypothesis_apparatus`); an answer embeds the question, never the corpus; missing document vectors are embedded on the first question only and saved; `Engine.search` and `search.search` give the same structured context, `ask` adds the answer, its metrics and the query-log record; LLM failure keeps knowledge; empty graph |
| 3D sphere | `test_sphere.py` (3) | the graph's visual language (classes, theme diamonds, legend layers with their toggle state, every trace tagged with its legend key, node ids for click-to-inspect, no own legend, palette), points within bounds, the answer markers, export, the sphere API with palette and layers, served plotly |
| self-containment | `test_self_contained.py` (4) | every loaded module and `sys.path` entry lies in the repository or the Python environment (interpreter, standard library, site-packages — also of a base environment the venv is layered on), the vendored engines load from `ltir/engines/`; no `ltir` source string holds a machine path, and a relative path that climbs a folder appears only in a docstring and resolves inside the repository; every path setting resolves inside the repository; the model loads from `models/` |
| layer boundaries | `test_architecture.py` (2) | every `ltir` import points down the stack of [12.2](12_architecture.md#122-dependency-direction); the core imports no web, chart or HTTP framework; only the Neo4j mirror imports the driver; the gateway imports nothing from `ltir` |
| UI | `test_ui_smoke.py` (2) | API upload → READY → graph → node → Ukrainian query highlight → delete → empty → a failed upload deleted by batch id; headless Chromium: dataset card, insights table, Ukrainian chat answer with highlight and evidence cards (panels closed by default, open on click), the legend column (defaults, counts, hover spotlight) switching sphere traces and graph edges, an isolated evidence path kept through a layer switch, sphere click → drawer, a dragged column border, citation → drawer, theme toggle, delete from the card, no page errors |

**Isolation.** `conftest.py` sets `LTIR_NO_DOTENV=1` and `neo4j_enabled=False`, so developer credentials never reach a test and no test writes to a real database; the LLM is a deterministic fake or a local stub. **Markers**: `model` (6 tests, skipped automatically when the model folder is missing) and `browser` (skipped without Playwright and an installed Chromium). `pytest.ini` pins `--basetemp=.pytest_tmp -p no:cacheprovider`.

**Known gaps in coverage.** The weight test checks the fully measured case (the renormalisation over measured factors, the `1e-3` clamp, `low_weight` and rule precedence are not asserted separately); `no_candidates` has no test; the delay → returns mechanism is recovered on the demo but only the benchmark guards it; the UI smoke test resolves the highlighted *edges* against the graph; the sphere bound check covers the first anchor's points; the browser test looks for Chromium in the Windows Playwright folder.

## 10.2 The synthetic demo dataset

`ltir/evaluation/synthetic.py` writes `data/demo/retail_synthetic.csv`: 5,000 orders, seed 7. Dimensions `region, category, channel`, plus noise columns `payment, weekday, store_size` that step 2 of the EDA correctly ignores; `order_id` is an identifier. Mechanisms: `margin = 26 − 0.6 · discount + ε`, `return_rate = 0.02 + 0.01 · delivery_days + ε`.

| Planted phenomenon | Scope(s) | Validates |
|---|---|---|
| local anomaly | EU ∧ laptops: margin +5 | anomaly discovery |
| stronger specialisation | EU ∧ laptops ∧ online: +4 more | SPECIALIZES with a larger effect |
| contrasting subgroup | EU ∧ laptops ∧ retail: net −4 | CONTRASTS (specialisation reversal) |
| recurring phenomenon: discount erosion | US ∧ phones, APAC ∧ tablets, EU ∧ tablets ∧ retail: discount +9 | one anchor over structurally disjoint scopes |
| recurring phenomenon: delay → returns | APAC ∧ online, US ∧ laptops ∧ retail: delivery +3 | a second cross-scope anchor |
| correlation break | EU ∧ phones: margin decoupled from discount | EMM / covariance insight |

`GROUND_TRUTH` in `evaluation/synthetic.py` lists these scopes for the tests and the benchmark.

## 10.3 Hypothesis benchmark

`python -m ltir experiment --k K` (`experiment.run_experiment`). **Labels come from the planted ground truth**: a pattern belongs to a mechanism when its scope contains one of that mechanism's planted scopes (ambiguous patterns are skipped). They are deliberately not derived from the measured shifts — the shifts are what the vectors encode, so shift-based labels would favour vector retrieval by construction. For each labelled pattern `S`, the **analogues** are the other patterns of the same mechanism that share no condition with `S`; only the two multi-scope mechanisms yield analogues. Four rankers are compared:

| Ranker | What it does |
|---|---|
| `transversal` | this system: the walk of [7.3](07_question_answering.md#73-transversal-traversal) with the seed forced to `S` and no result cap |
| `structural` | BFS over all structural edges (depth ≤ `TRAVERSAL_MAX_DEPTH`), ranked by hops, then weight |
| `text_nn` | cosine of the canonical-document embeddings stored at ingest — naive vector RAG |
| `vector_nn` | cosine of the raw insight vectors, without the anchor graph |

```text
recall@k = hits / |analogues|        precision@k = hits / min(k, |analogues|)        MRR = 1 / rank of the first analogue (0 if none)
```

Results on the demo (12 seed cases: 8 discount erosion, 4 delay → returns; Qwen3 → 384; 28 patterns):

| Method | recall@3 | precision@3 | MRR | recall@5 |
|---|---|---|---|---|
| transversal | **0.521** | **0.556** | **0.581** | **0.729** |
| structural | 0.000 | 0.000 | 0.079 | 0.000 |
| text_nn | 0.111 | 0.111 | 0.321 | 0.299 |
| vector_nn | 0.028 | 0.028 | 0.229 | 0.507 |

With MiniLM the transversal row is 0.333 / 0.361 / 0.567 / 0.729 ([4.5](04_representation.md#45-embedder-comparison)). Reading: structural traversal never reaches scope-disjoint analogues within its first ranks — they sit 2–4 lattice hops away, mixed with everything else. Text-NN and raw vector kNN rank the seed's own refinements first. The anchor layer puts the analogues at the top (MRR 0.58 vs ≤ 0.32) and reaches most of them by `k = 5` (0.73 vs 0.51). This is an apparatus, not proof: one synthetic dataset, two mechanisms with analogues. `test_hypothesis_apparatus` guards the ordering as a regression.

## 10.4 Measurement scripts

The measurement scripts build throwaway workspaces under `.scratch/` (never `workspace/`), switch Neo4j off, and save their results to `.scratch/results/`; the numbers quoted in chapters 4, 5 and 7 come from them. `scripts/scratch.py` holds the shared helpers; `download_model.py` is a setup script and builds no workspace.

| Script | Measures |
|---|---|
| `multilingual_benchmark.py [--neo4j off|fake]` | the 60-question grounding benchmark of [7.7](07_question_answering.md#77-measured-behaviour): seed recall, consistency, evidence overlap, direction accuracy, FPGR and latency per bucket; `--neo4j fake` publishes through the tests' in-memory driver |
| `prompt_tokens.py [--backend hashing]` | characters, non-ASCII characters and tokens (XLM-R SentencePiece, Qwen BPE) of the evidence prompt, the documents and the evidence-only answer; the default hashing backend leaves the documents unchanged, but the prompt's items depend on retrieval and so on the backend |
| `compare_embedders.py --models minilm,qwen3` | contract cosines, the benchmark, domain separation for `MIN_ASSIGN_THRESHOLD`, cross-domain links and evidence, load and embed time, peak VRAM per embedder |
| `eval_answers.py` | five demo questions through the configured LLM: grounding, unknown citations, latency, token usage — **live**: it calls `LLM_BASE_URL`, by default the gateway, which must be running with `.env.gemma` |
| `download_model.py` | fetches the pinned embedding model into `models/` |

## 10.5 Quality gate and container check

`python scripts/check.py` runs `ruff check`, `ruff format --check` (rules E/F/W/I/UP/B, line length 150; the vendored EDA file keeps its upstream style and is exempt from formatting) and the full pytest run; every step runs even after a failure, and the exit code is non-zero if any failed. `--quick` skips the `model` and `browser` tests for feedback during a change; it is not the gate. A change to a contract (fields, formula, schema, path grammar) updates the owning chapter and its test in the same change; a representation change bumps `REPRESENTATION_VERSION` or `CANONICAL_VERSION` ([11.5](11_reference.md#115-versioned-contracts)). Lint and format cover `ltir/`, `llm_gateway/`, `tests/` and `scripts/`.

`python scripts/compose_check.py` verifies the three containers of [12.5](12_architecture.md#125-containers) as a system, as compose project `sig-check` on its own host ports, Neo4j password and volumes — created for the run and removed after it, so every run ingests from scratch and the user's own stack can keep running: build, start until healthy, the backend's API, the gateway's health and model list (from inside the stack: the gateway publishes no port), the backend reaching the gateway, Neo4j answering; then the demo through `POST /api/demo` to READY (a SKIPPED demo fails), the graph served, the Neo4j mirror holding the snapshot's nodes label by label and all its relationships, and evidence for a question. `--tests` builds the backend image's `test` target and runs `scripts/check.py` inside it (the model mounted read-only, no Chromium, so the browser test skips); `--fresh` rebuilds without the build cache and pulls the pinned base images (the clean-room run); `--live-llm` asks one question through the gateway to the real upstream — a paid call, so it is opt-in.

## 10.6 Known approximations and limitations

**Statistics**
* The EDA searches only 2- and 3-conjunctions of equality selectors — no single selectors, no numeric intervals; numeric dimensions need explicit bands.
* Significance is an asymptotic median test with Bonferroni over distinct cohorts × metrics: conservative (overlapping cohorts count as independent tests), not a permutation test. Pre-validation near-duplicate pruning ranks by the pass-1 `temp_index`, not by the final weight.
* `SD` sums the three largest shifts; `temp_index` adds clipped z-scores of incommensurable quantities (the EDA's design, monotone in each); the EMM shrinkage `sqrt((n − n_min)/(N − n_min))` is a reliability heuristic, not a standard error.
* Weight exponents, block weights, `EMM_COMPONENT_WEIGHT` and the seed-score weights are design choices validated on the synthetic data and the benchmark, not fitted.

**Representation and ontology**
* The phenomenon is linear in label embeddings: with correlated labels (Qwen3: discount · margin = 0.71), a one-metric question and a two-metric insight partly cancel ([7.2](07_question_answering.md#72-seeds)), and two one-off phenomena can share an anchor.
* OMP dictionary learning on tens of rows is far from its intended regime; the signed repair is what makes it usable. The adaptive assignment threshold becomes adaptive only with 10 or more anchors.
* Mutual kNN can leave an anchor without links, and under Qwen3 a few links join anchors of unrelated datasets ([5.7](05_latent_anchors.md#57-links-between-anchors), [5.10](05_latent_anchors.md#510-calibration-per-embedder)). `engine_weight` is not comparable across activation sources.

**Operation**
* Rollback restores the journal, the state and the snapshot; the dataset-folder files of a failed batch remain until the next ingest of that dataset overwrites them.
* The highlighted walk (`traversed`, `edges`) covers up to 15 retrieved patterns; the `evidence` group and the prompt hold the first 10. The LLM runs remotely in the configured deployment, so the prompt — subgroup statistics, not rows — leaves the machine; use a local server for fully local inference.
* One writer process per workspace: while the web app runs, CLI writers (`demo`, `ingest`, `reset`, `rebuild-graph`, `migrate`) are refused; upload through the app.
* SIG owns its six node labels in `NEO4J_DATABASE`: every publish deletes nodes of those labels that the snapshot does not hold, so the database serves one workspace and no other data under those labels.
* The LLM and Neo4j are covered by a stub server and a fake driver in the tests; the live OpenRouter runs are verified by hand ([7.7](07_question_answering.md#77-measured-behaviour)), and the Neo4j publisher against a real Neo4j container by `scripts/compose_check.py` (§10.5; [6.6](06_graph_and_storage.md#66-neo4j-mirror)).
