# 10. Verification — tests, benchmark, measurements, limitations

> **In one paragraph.** The test suite guards every transition whose breakage would invalidate the system, from the EDA contract to the highlighted path in a real browser, and every boundary between the libraries and the services. A deterministic synthetic dataset with planted mechanisms makes the statistics checkable, and a benchmark turns the research hypothesis into a measurement against structural and text baselines. Measurement scripts reproduce the benchmark, grounding and live-answer numbers on throwaway workspaces, and a container check runs the whole suite as deployed. The chapter ends with what the system approximates and what it does not do.

**Code** `tests/`, `insight_graph_service/core/demo.py`, `scripts/` · **Previous** [9. Operations](09_operations.md) · **Next** [11. Reference](11_reference.md)

---

## 10.1 Test map

113 collected tests in 20 files (112 functions; the embedding contract runs for both the hashing double and the real model). `python -m pytest` from the repository root; about two minutes with the model and browser tests.

| Transition or contract | File (tests) | What is asserted |
|---|---|---|
| discovery → insight | `test_discovery_contract.py` (10) | profile contract (identifier dropped, float targets kept, EDA dimensions, candidates within the search space); fields and provenance; signed directions of the planted effects; determinism; planted phenomena survive selection; `closed_intent` adds implied conditions; identical extents merge before validation; near duplicates pruned in rank order; extent/intent antitone; `no_numeric_targets` and `invalid_schema` |
| the miner alone | `test_subgroup_miner_standalone.py` (1) | a table → validated, weighted insights → `describe` (numbered ASCII lines for a prompt) |
| selection, weight and admission | `test_selection.py` (6) | weight bounds and monotonicity, the formula with all five factors measured; reason codes of the validity rules; admission (`batch.admit_insights`): the R4 weight floor (`low_weight`) and the R7 budget (`budget`), heaviest first; covariance retargeting with unmeasured factors and the shift's p-values and stability `None` |
| canonical → embedding | `test_canonical_embedding.py` (9) | embedding inputs vs document (sections, ASCII); labels without numbers; number rules; covariance component; shape, dtype, unit norms, stability, direction separation and cross-scope similarity for the hashing double **and the real model**; fingerprint sensitivity (block weights, width, truncation, query instruction, the three thresholds, compute dtype, revision); a `.env` value read by `load_env_file` overrides its default, an empty non-string value keeps it; query instruction only on free question text |
| the topology alone | `test_attractor_topology_standalone.py` (1) | insights → 1152-d vectors (hashing double) → anchors with unit centroids, every insight activated, the anchors persisting in their folder |
| insight → ontology | `test_ontology.py` (11) | cold start coverage, at least three anchors, each planted cluster within one anchor; the orphan rule (cases: no members and no link → dropped, no members but linked → kept, members → kept); `forget` recounts, renumbers and drops; assignment, orphans and new anchors; soft merge; weight scales the EMA pull; `τ_density`; damping without membership change; trust region; sign repair; state round-trip |
| structural lattice | `test_lattice.py` (5) | covering relation only, GENERALIZES inverse, SIBLING rule, CONTRASTS rule and relation type, no embedding dependency |
| persistence | `test_persistence.py` (20) | write → reload → rebuild identical; a failed READY write rolls back the snapshot too; an interrupted deletion (error or crash) is undone; dataset deletion (artefacts gone, the dataset's literals leaving the literal cache, the orphan rule anchor by anchor, journal/frame/ontology consistent, snapshot = rebuild, re-ingest possible, last dataset empties the ontology); a record rewritten under a concurrent reader (no sharing violation on either side); a reset during a batch refused at once (`busy`) while readers keep the committed state and the batch still commits; idempotent re-ingest; graph consistency (incl. every membership below the alignment floor is weak); rollback; a failed rollback keeps its marker, refuses later writes (`rollback_pending`) and is finished by the next writer start; representation mismatch; a batch killed after its journal append rolled back at the next start; a second writer process refused while the first keeps its queue; journals of another canonical version start degraded (empty state, writes and questions refused with `workspace_degraded`, the Neo4j mirror untouched) and a reset recovers; a failed recovery starts degraded too; READY survives a post-commit save failure; ingestion failure codes; options strict when explicit, lenient as defaults; the Neo4j mirror equals the snapshot |
| literal grounding | `test_multilingual_retrieval.py` (12) | the dense gates on a dictionary embedder (translations accepted, `us` rejected by case, a hub literal demoted, an equidistant span rejected by Lowe); character matches for same-script inflections and typos only, the length guard; longest span claims its tokens and one embedding pass; a value under two columns resolves to the named column or a wildcard that scores without conflict, typed or translated alike; words inside a matched literal cast no direction, and a weakening word is a word, not a substring; Ukrainian comparatives and directed drivers; B1 = B2 seeds (hashing double); the catalog vectors persisted and reloaded without re-embedding, the catalog cached until the next commit; B3 translated twin parses and seeds like English, distractors ground to nothing (`model`); B4 inflected Ukrainian value (`model`); identical seeds with the Neo4j mirror on (fake driver) and off |
| traversal and evidence | `test_traversal.py` (4) | the exact pattern → anchor → anchor → pattern path with hops, weights, reversal and score; a weak membership is not walked while a kept 0.35 link is; hop and depth budgets; parsing against the graph vocabulary in English and in Ukrainian around the same literals; the evidence object |
| the query engine alone | `test_graph_query_engine_standalone.py` (4) | a hand-written snapshot and an encoder with only the `QueryEncoder` shape: the literal catalog built on first use, the seed, the citation manifest equal to the evidence keys, the parse inside `view.evidence` and the prompt only at the top level, the payload's JSON round trip through `EvidencePayload.from_dict`, the empty payload's highlight groups; a snapshot of another version refused |
| narrator | `test_narrator.py` (2) | against a stub graph service serving a recorded payload and the LLM stub: the graph's prompt sent verbatim, a grounded `llm` answer with `view` passed through untouched, `prompt` and the footer, the query-log line; `use_llm: false` gives the payload's summary; a 409 passed on as sent; an empty graph answers `empty` (200); an empty question 400; health without its dependencies still 200, and a query then 502 |
| LLM client | `test_llm_client.py` (4) | the real HTTP client against an OpenAI-compatible stub: a plain OpenAI payload without a model name or a key, health naming the served model, an upstream error keeping its status and reason; a refused connection fails fast and is remembered; citation validation incl. grouped citations |
| LLM model broker | `test_llm_model_broker.py` (3) | the broker forwards to the configured model with the pinned provider order and its bearer key, replacing the requested model and passing the other fields as sent; an upstream 429 keeps its status and `Retry-After`, a 5xx → 502, empty messages, `stream: true` or a malformed body → an OpenAI-shaped 400, missing settings → 503 with a live `/health`; the whole chain narrator client → broker → upstream over local HTTP, the key only in the broker |
| model provisioning | `test_model_store.py` (2) | a verified seed is copied and a present copy kept; a corrupt file of the right size fails the SHA-256 check but passes the service's presence-and-size check, and is repaired from the seed; the bundled checkpoint matches the pinned manifest (`model`) |
| end to end | `test_e2e.py` (8) | upload → … → grounded answer with provenance and cross-scope analogues (hashing double and real model); the benchmark ordering (`test_hypothesis_apparatus`); an answer embeds the question, never the corpus; missing document vectors are embedded on the first question only and saved; `Engine.search` and `search.search` give the same structured context, and the narration adds the answer and its metrics; LLM failure keeps knowledge; empty graph |
| 3D sphere | `test_sphere.py` (3) | every insight and the themes it activates placed inside the unit ball, classes `up` / `down` / `cov`, hover text, edges only between placed nodes (ACTIVATES, RELATED_TO and SPECIALIZES among them); an answer's highlighted edges between placed nodes resolve among the sphere's edges; the projection is deterministic, `GET /api/sphere` serves it, an empty workspace gives the message |
| self-containment | `test_self_contained.py` (4) | every loaded module and `sys.path` entry lies in the repository or the Python environment (interpreter, standard library, site-packages — also of a base environment the venv is layered on), the vendored engines load from the repository; no source string of the seven packages holds a machine path, and a relative path that climbs a folder appears only in a docstring and resolves inside the repository; every path setting of every section resolves inside the repository; the model loads from `models/` |
| package boundaries | `test_architecture.py` (2) | every import of the seven packages is in the allow-list of [12.2](12_architecture.md#122-dependency-direction): the repository packages and the third-party modules each package may import (relative imports resolved); only the Neo4j mirror imports the driver |
| UI | `test_ui_smoke.py` (2) | on the `scripts/dev.py` composite: static files, API upload → READY → graph → node → a grounded Ukrainian chat answer (the narrator asks the graph service over local HTTP) with its highlight → delete → empty → a failed upload deleted by batch id → reset; headless Chromium: dataset card, insights table, Ukrainian chat answer with highlight and evidence cards (panels closed by default, open on click), the legend column (defaults, counts, hover spotlight) switching sphere traces and graph edges, an isolated evidence path kept through a layer switch, sphere click → drawer, a dragged column border, citation → drawer, theme toggle, delete from the card, no page errors |

**Isolation.**
- **No credentials.** Settings read only the process environment, and the tests never load the repository's `.env`, so developer credentials never reach a test.
- **No database.** `conftest.make_config` sets `neo4j_enabled=False`, so no test writes to a real database; the Neo4j tests use a fake driver.
- **No live LLM.** The LLM is a deterministic fake (`conftest.FakeLLM`) or a local OpenAI-compatible stub (`conftest.llm_stub`).
- **The test embedder.** It is the hashing double `tests/doubles.py::HashingEmbedder`, word and character-trigram hashing into 384-d blocks (the production block width). `conftest.make_engine` injects it; `embedder=None` loads the bundled Qwen3.
- **Answers in-process.** `conftest.ask(engine, question, llm)` composes an answer in-process as the two services compose it over HTTP: `narration.answer(engine.evidence(question), llm)`.

**Markers.**
- `model` (7 tests) — skipped automatically when the model folder lacks `modules.json`.
- `browser` (1 test) — skipped without Playwright and an installed Chromium.

`pytest.ini` pins `--basetemp=.pytest_tmp -p no:cacheprovider` and puts `tests` and `scripts` on the import path.

**Known gaps in coverage.**
- The weight test checks the fully measured case; the renormalisation over measured factors, the `1e-3` clamp and rule precedence are not asserted separately.
- `no_candidates` has no test.
- The delay → returns mechanism is recovered on the demo, but only the benchmark guards it.
- The UI smoke test resolves the highlighted *edges* against the graph.
- The browser test looks for Chromium only under the Playwright folder of `%LOCALAPPDATA%` (Windows and Linux builds) and skips elsewhere.
- The containers are verified by `scripts/compose_check.py`, outside pytest ([10.5](#105-quality-gate-and-container-check)).

## 10.2 The synthetic demo dataset

`insight_graph_service/core/demo.py::generate_retail_dataset` builds the demo table: 5,000 orders, seed 7. `POST /api/demo` ingests it as `retail_synthetic.csv`; the tests and the measurement scripts write it to their own folders. Dimensions `region, category, channel`, plus noise columns `payment, weekday, store_size` that step 2 of the EDA correctly ignores; `order_id` is an identifier. Mechanisms: `margin = 26 − 0.6 · discount + ε`, `return_rate = 0.02 + 0.01 · delivery_days + ε`.

| Planted phenomenon | Scope(s) | Validates |
|---|---|---|
| local anomaly | EU ∧ laptops: margin +5 | anomaly discovery |
| stronger specialisation | EU ∧ laptops ∧ online: +4 more | SPECIALIZES with a larger effect |
| contrasting subgroup | EU ∧ laptops ∧ retail: net −4 | CONTRASTS (specialisation reversal) |
| recurring phenomenon: discount erosion | US ∧ phones, APAC ∧ tablets, EU ∧ tablets ∧ retail: discount +9 | one anchor over structurally disjoint scopes |
| recurring phenomenon: delay → returns | APAC ∧ online, US ∧ laptops ∧ retail: delivery +3 | a second cross-scope anchor |
| correlation break | EU ∧ phones: margin decoupled from discount | EMM / covariance insight |

`GROUND_TRUTH` in the same module lists these scopes for the tests and the benchmark.

## 10.3 Hypothesis benchmark

`python scripts/experiment.py --k K` (`run_experiment` in `scripts/experiment.py`; `--k` defaults to 5) ingests the demo into a fresh scratch workspace and saves its result to `.scratch/results/experiment_<label>.json`.
- **Labels come from the planted ground truth.** A pattern belongs to a mechanism when its scope contains one of that mechanism's planted scopes; ambiguous patterns are skipped. They are deliberately not derived from the measured shifts: the shifts are what the vectors encode, so shift-based labels would favour vector retrieval by construction.
- **Analogues.** For each labelled pattern `S`, the analogues are the other patterns of the same mechanism that share no condition with `S`. Only the two multi-scope mechanisms yield analogues.

Four rankers are compared:

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

Reading:
- **Structural traversal** never reaches scope-disjoint analogues within its first ranks: they sit 2–4 lattice hops away, mixed with everything else.
- **Text-NN and raw vector kNN** rank the seed's own refinements first.
- **The anchor layer** puts the analogues at the top (MRR 0.58 vs ≤ 0.32) and reaches most of them by `k = 5` (0.73 vs 0.51).

This is an apparatus, not proof: one synthetic dataset, two mechanisms with analogues. `test_hypothesis_apparatus` guards the ordering as a regression.

## 10.4 Measurement scripts

The measurement scripts build throwaway workspaces under `.scratch/` (never `workspace/`), switch Neo4j off and save their results to `.scratch/results/<name>_<label>.json`. The benchmark, grounding and live-answer numbers of [7.7](07_question_answering.md#77-measured-behaviour) and [10.3](#103-hypothesis-benchmark) come from them. `scripts/scratch.py` holds the shared helpers (`scratch_dir`, `write_demo_csv`, `build_engine`, `ingest_ready`, `demo_engine`, `save_result`) and loads `.env` like the services do.

| Script | Measures |
|---|---|
| `experiment.py [--k K] [--label NAME]` | the hypothesis benchmark of [10.3](#103-hypothesis-benchmark) |
| `multilingual_benchmark.py [--label NAME]` | the 60-question grounding benchmark of [7.7](07_question_answering.md#77-measured-behaviour): seed recall, consistency, evidence overlap, direction accuracy, FPGR and latency per bucket |
| `eval_answers.py [--label NAME]` | five demo questions narrated through the configured LLM (`narration.answer` over `Engine.evidence`): grounding, citations, unknown keys, latency, token usage, prompt size — **live**: it calls the narrator's `LLM_BASE_URL`, by default the broker, which must be running (`python -m llm_model_broker` with `GEMMA_*` in `.env`) |

## 10.5 Quality gate and container check

`python scripts/check.py` runs three steps:
- `ruff check` (rules E/F/W/I/UP/B, line length 150; the vendored EDA file keeps its upstream style and is exempt from formatting);
- `ruff format --check`;
- the full pytest run.

Lint and format cover the seven packages (`insight_contracts/`, `subgroup_miner/`, `attractor_topology/`, `graph_query_engine/`, `insight_graph_service/`, `evidence_narrator_service/`, `llm_model_broker/`), `tests/` and `scripts/`. Every step runs even after a failure, and the exit code is non-zero if any failed. `--quick` skips the `model` and `browser` tests for feedback during a change; it is not the gate. A change to a contract (fields, formula, schema, path grammar, `EvidencePayload`) updates the owning chapter and its test in the same change; a representation change bumps `REPRESENTATION_VERSION` or `CANONICAL_VERSION` ([11.5](11_reference.md#115-versioned-contracts)).

`python scripts/compose_check.py` verifies the suite of [12.5](12_architecture.md#125-containers-and-start-order) as deployed. It builds and starts all six services — `model-init`, `neo4j`, `llm-model-broker`, `insight-graph`, `evidence-narrator`, `sig-web-console` — as compose project `sig-check` on its own host ports, Neo4j password and volumes. The volumes are created for the run and removed after it (`--keep` leaves the stack running), so every run ingests from scratch and the user's own stack can keep running. Everything is reached through the console's nginx, as a user reaches it. It checks:
- **start order:** `model-init` completed before `insight-graph` started; `neo4j` ≤ `insight-graph` ≤ `evidence-narrator`; the broker ≤ the narrator ≤ the console;
- **model provisioning:** `model-init` exited 0 with the model present, copied from the seed or downloaded;
- **the services:** the console serves the UI with its vendored libraries, the graph service answers through it, the narrator reaches the graph service and the broker's model, Neo4j answers;
- **an empty graph** answers `answer_mode: empty`;
- **ingest:** the demo CSV uploaded through nginx (multipart) to READY, where a SKIPPED upload fails; `GET /api/health` stays under 2 s during the ingest;
- **the mirror:** Neo4j holds the snapshot's nodes label by label and all its relationships;
- **an answer:** the evidence narrated (`use_llm: false`) with evidence items, cited keys and no unknown key;
- **the sphere:** the JSON holds every insight and theme;
- **reset:** `POST /api/reset` empties the graph and the mirror, and the next question answers `empty`.

Options:
- `--tests` builds the `test` target of `insight_graph_service/Dockerfile` and runs `scripts/check.py` inside it, with the model mounted read-only and no Chromium, so the browser test skips.
- `--fresh` rebuilds without the build cache and pulls the pinned base images (the clean-room run).
- `--live-llm` asks one question through the broker to the real upstream — a paid call, so it is opt-in.

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
* One writer process per workspace: while the service runs, another writer process on the same workspace is refused (`WorkspaceBusy`); upload through the service. One graph service replica with one worker ([12.6](12_architecture.md#126-storage-files-no-sql-database-no-blob-store)).
* A workspace of another canonical or representation version is not migrated (proof-of-concept scope): it starts degraded, and `POST /api/reset` starts it over ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).
* SIG owns its six node labels in `NEO4J_DATABASE`: every publish deletes nodes of those labels that the snapshot does not hold, so the database serves one workspace and no other data under those labels.
* The tests cover the LLM, the broker's upstream and Neo4j with a stub server and a fake driver. The live OpenRouter runs are verified by hand ([7.7](07_question_answering.md#77-measured-behaviour)) and, opt-in, by `scripts/compose_check.py --live-llm`. The Neo4j publisher is verified against a real Neo4j container by `scripts/compose_check.py` ([10.5](#105-quality-gate-and-container-check); [6.6](06_graph_and_storage.md#66-neo4j-mirror)).
