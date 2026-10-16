# 10. Verification — tests, benchmark, measurements, limitations

> **In one paragraph.** The test suite guards every transition whose breakage would invalidate the system, from the EDA contract to the highlighted path in a real browser, and every boundary between the libraries and the services. A deterministic synthetic dataset with planted mechanisms makes the statistics checkable, and a benchmark turns the research hypothesis into a measurement against structural and text baselines. Measurement scripts reproduce the benchmark, grounding and live-answer numbers on throwaway workspaces, and a container check runs the whole suite as deployed. The chapter ends with what the system approximates and what it does not do.

**Code** `tests/`, `insight_graph_service/core/demo.py`, `scripts/` · **Previous** [9. Operations](09_operations.md) · **Next** [11. Reference](11_reference.md)

---

## 10.1 Test map

143 collected tests in 22 files (142 functions; the embedding contract runs for both the hashing double and the real model). `python -m pytest` from the repository root; about two minutes with the model and browser tests.

| Transition or contract | File (tests) | What is asserted |
|---|---|---|
| discovery → insight | `test_discovery_contract.py` (10) | profile contract (identifier dropped, float targets kept, EDA dimensions, candidates within the search space); fields and provenance; signed directions of the planted effects; determinism; planted phenomena survive selection; `closed_intent` adds implied conditions; identical extents merge before validation; near duplicates pruned in rank order; extent/intent antitone; `no_numeric_targets` and `invalid_schema` |
| geo option | `test_spatial.py` (21) | the geo spec and point parsing (GeoJSON, WKT, junk; `longitude,latitude` read by the column names); cell-table roles from dtypes (counters for every cell, intensive metrics NaN below `MIN_CELL_POINTS`, a place-like categorical as a mode, an identifier dropped); the neighbour lag excludes the cell and counts absent neighbours as 0; LISA marks the planted cluster `hot spot`; `n_eff` widens the median test; the subgroup's Moran's I sees a solid block, not scattered cells; the closure strips the shifts a closed condition restates and the validation never reports a restated metric as a hidden shift; the block bootstrap skips draws without values; a batch that failed after writing its files serves no map; the option is strict and changes the dataset id; a geo batch end to end: SAC counts, `provenance.spatial`, no neighbourhood condition explaining a counter, the planted night share found in hot spots, `geo_cells` / `geo_pattern` / `geo_datasets`; the glossary and a PLAIN LANGUAGE prompt without raw derived column names; `SpatialProvenance` keeps the five ingest fields, round-trips and reads an older record; `derive_spatial` on synthetic cells: the three CO_LOCATED relations, an `adjacent` weight of at most 1 and shared cells below the overlap threshold as `adjacent`, the partner cap with partners of other datasets ranked apart, one region for datasets of resolution 8 and 7, PARENT_OF, each pattern's and each dataset's cells per resolution (a dataset without patterns too); one upload at `@8,7` with `table` makes three READY datasets under one `upload_id` (its 202 over HTTP names every level's batch), the geo levels with the ids of separate uploads, no spatial edge on the table level, a level deleted alone keeps the shared file, `DELETE /api/uploads/{id}` removes the rest, bad levels refused before the file is kept, the table level without its point columns (and not a plain upload's id), credibility by place on the geo levels and by anchor on the table level, all 6 as one source; `@auto` picks 5 for the California table, the same levels twice, a finer level for a denser file, each resolution's cell count as `build_cells` makes it (a box whose parent count passed `MIN_ROWS` at 5 and whose cells do not), the explicit option's dataset id and `profile.geo.levels` from the upload on, and refuses (file not kept) points with too few cells; the scale label reads the levels the upload's batches stand for (a coarser level added by a later upload of the same file counts, one deleted alone does not); a geo dataset ingested before the spatial plane gets it when an older snapshot is rebuilt; a polygon equal to a set of cells is that set (as a Feature and a FeatureCollection too), a circle and a box hold the cells whose centres they hold, a shape smaller than a cell the cell under it, bad shapes (an infinite radius too) and cells refused; a zone's overlap (1 for the zone itself, 0 outside it) counts a finer zone cell's parent and every cell below a coarser one; a zone from the map on an upload at `@8,7`: the question's seed stays first with `place` 1 and the same SCALE CHAIN, the AREA line, a zone alone seeding from inside it, a polygon of the same cells seeding the same, `resolution` 7 keeping level 7 only (a level-8 item neither in nor out of the AREA), `POST /api/search` with cells alone and its 400s |
| the miner alone | `test_subgroup_miner_standalone.py` (1) | a table → validated, weighted insights → `describe` (numbered ASCII lines for a prompt) |
| selection, weight and admission | `test_selection.py` (6) | weight bounds and monotonicity (a p-value change leaves the weight unchanged), the formula with all four factors measured, a five-entry `WEIGHT_EXPONENTS` refused; reason codes of the validity rules; admission (`batch.admit_insights`): the R4 weight floor (`low_weight`) and the R7 budget (`budget`), heaviest first; covariance retargeting with unmeasured factors and the shift's p-values and stability `None` |
| canonical → embedding | `test_canonical_embedding.py` (9) | embedding inputs vs document (sections, ASCII); labels without numbers; number rules; covariance component; shape, dtype, unit norms, stability, direction separation and cross-scope similarity for the hashing double **and the real model**; fingerprint sensitivity (block weights, width, truncation, query instruction, the three thresholds, compute dtype, revision); a `.env` value read by `load_env_file` overrides its default, an empty non-string value keeps it; query instruction only on free question text |
| the topology alone | `test_attractor_topology_standalone.py` (1) | insights → 1152-d vectors (hashing double) → anchors with unit centroids, every insight activated, the anchors persisting in their folder |
| insight → ontology | `test_ontology.py` (12) | cold start coverage, at least three anchors, each planted cluster within one anchor; the orphan rule (cases: no members and no link → dropped, no members but linked → kept, members → kept); `forget` recounts mass in `w`, moves only the anchors that lost a member, renumbers and drops; the journal recount equals the incremental mass and refuses an unknown anchor; assignment, orphans and new anchors; soft merge; weight scales the EMA pull; `τ_density`; damping without membership change; trust region; sign repair; state round-trip |
| structural lattice | `test_lattice.py` (5) | covering relation only, no GENERALIZES (the walk follows SPECIALIZES both ways), SIBLING rule, CONTRASTS rule and relation type, no embedding dependency |
| persistence | `test_persistence.py` (32) | write → reload → rebuild identical; a model failure while the state is loaded leaves the workspace usable (no catalog yet, not degraded), a catalog failure at a commit is a warning on the READY batch and the first question builds it, and a workspace of another representation gets no cache built or saved; the indicator registry on the Metric nodes and quoted in the baselines, an unknown metric dropped with a warning at upload and a 400 at `PATCH`, which replaces the registry without a new batch; a re-upload leaves the registry unchanged and names the `PATCH`; source and reliability outside the dataset id, in the prompt only once declared, changed by `PATCH` without a new batch or new pattern ids, a bad value refused with 400 `invalid_options`; every field the prompt shows is covered by the record digest; a p-value changed in the journal is `integrity: mismatch` after the rebuild, never a seed and left out of the evidence with a NOTE, a record without a digest reads `unverified` (marked in its `validation:` line) and the workspace is not degraded; a failed READY write rolls back the snapshot too; an interrupted deletion (error or crash) is undone; dataset deletion (artefacts gone, the dataset's literals leaving the literal cache, the orphan rule anchor by anchor, journal/frame/ontology consistent, snapshot = rebuild, re-ingest possible, last dataset empties the ontology); a reset during a batch refused at once (`busy`) while readers keep the committed state and the batch still commits; idempotent re-ingest; graph consistency (incl. every membership below the alignment floor is weak and every CO_OCCURS endpoint has a non-weak member); the co-occurrence weights on a hand example (normalised weights, weak memberships excluded, the cap at 1); the Cypher walks both latent edge types; rollback; a failed rollback keeps its marker, refuses later writes (`rollback_pending`) and is finished by the next writer start; representation mismatch; a batch killed after its journal append rolled back at the next start; a second writer process refused while the first keeps its queue; journals of another canonical version start degraded (empty state, writes and questions refused with `workspace_degraded`, the Neo4j mirror untouched) and a reset recovers; a failed recovery starts degraded too; READY survives a post-commit save failure; ingestion failure codes; options strict when explicit, lenient as defaults; the Neo4j mirror equals the snapshot (relationships of a type the schema dropped deleted too); a new version with `supersedes=<upload_id>` at `@8,7`: the first level waits for the last, a commit that fails after the removal rolls it back (both old levels, their journal rows and file intact), and the successful one removes both old levels in its commit, leaves the pattern ids of the new version loaded alone, keeps the shared anchors' ids and a consistent journal; a coarser level with too few cells (`invalid_schema`) still completes the new version, and an old version with a level still queued stays (`busy`, a warning); `PATCH supersedes` deletes the old upload without a new batch or new pattern ids, answers `busy` while a level of it is queued, and refuses the dataset itself, its own upload, its other level, an unknown id and an upload of the very file it would replace, named by its upload or one of its levels (not kept); credibility phase A on hand-made patterns: confirmation 1, contradiction 5 over 1 in one subgroup (not in another or a merely overlapping one), the metric name alone 2, one source 6, a weak membership or a tampered pattern confirming nothing, undeclared or unregistered sources never 1 or 5, one source in two cases, two levels of one upload or a version and the one it supersedes not independent, indicators compared casefolded, a source listed once in the basis, the level unmoved by p, stability and weight; phase B on hand-made places: confirmation only where one place lies mostly inside the other (a pattern elsewhere confirms nothing), contradiction only over a comparable place (not a wide one holding the spot), 2 from another dataset's events over a fifth of the place and 6 below, the coarser dataset's level used, a sparse pattern sharing one cell of a spot (and so one region above it) not the same place, one upload neither confirming nor covering, no registry nominal, units compared only when both give one, a pattern without a place kept in phase A |
| literal grounding | `test_multilingual_retrieval.py` (12) | the dense gates on a dictionary embedder (translations accepted, `us` rejected by case, a hub literal demoted, an equidistant span rejected by Lowe); character matches for same-script inflections and typos only, the length guard; longest span claims its tokens and one embedding pass; a value under two columns resolves to the named column or a wildcard that scores without conflict, typed or translated alike; words inside a matched literal cast no direction, and a weakening word is a word, not a substring; Ukrainian comparatives and directed drivers; B1 = B2 seeds (hashing double); the catalog vectors persisted and reloaded without re-embedding, the catalog cached until the next commit; B3 translated twin parses and seeds like English, distractors ground to nothing (`model`); B4 inflected Ukrainian value (`model`); identical seeds with the Neo4j mirror on (fake driver) and off |
| traversal and evidence | `test_traversal.py` (5) | the exact pattern → anchor → anchor → pattern path with hops, weights, reversal and score; a weak membership is not walked while a kept 0.35 link is; a CO_OCCURS link is walked and named in the evidence; hop and depth budgets; parsing against the graph vocabulary in English and in Ukrainian around the same literals; the evidence object |
| the query engine alone | `test_graph_query_engine_standalone.py` (14) | a hand-written snapshot and an encoder with only the `QueryEncoder` shape: the literal catalog built on first use, the seed, the citation manifest equal to the evidence keys, the parse inside `view.evidence` and the prompt only at the top level, the payload's JSON round trip through `EvidencePayload.from_dict`, the empty payload's highlight groups; a partial `provenance.spatial` (an older record) reads without its missing fields; an item reached only by location is tagged and named in a NOTE, also after a latent step (not marked scope-disjoint); two levels of one upload as one item with its scale (`regional` folded as `also at`, `scale-dependent` a fact of its own, `not_found_coarser` with its NOTE, `insufficient` unprinted), the chain up through the shared anchor and region to the regions above, and `picture`; the scale against a coarser pattern over several regions, with the coarser level ranked first (the group stays `regional` with its `also at`), never folded without an upload, and searched for when the budget is spent; a reversal above a coarser host stays `scale-dependent`; the ZONES of a question without scope (the coarsest level with three regions, a finding mostly outside or shifting the other way adding nothing) and none for a question with scope; a zone without text seeding its best pattern below the floor (`fallback: place`) and nothing when it holds none, no naive baseline, an AREA line naming in and out but never a pattern the zone did not weigh; two anchors joined by CO_OCCURS and RELATED_TO listed once each with both links and the shared count, and the second of one description naming the first with the members they share, or as a separate anchor when none is shared; an evidence item carrying its pattern's credibility (not in the prompt); a snapshot of another version refused |
| narrator | `test_narrator.py` (2) | against a stub graph service serving a recorded payload and the LLM stub: the graph's prompt sent verbatim, a grounded `llm` answer with `view` passed through untouched, `prompt` and the footer, the query-log line; `use_llm: false` gives the payload's summary; a 409 passed on as sent; a zone without text passed on as given and its 400 as sent; an empty graph answers `empty` (200); an empty question 400; health without its dependencies still 200, and a query then 502 |
| LLM client | `test_llm_client.py` (4) | the real HTTP client against an OpenAI-compatible stub: a plain OpenAI payload without a model name or a key, health naming the served model, an upstream error keeping its status and reason; a refused connection fails fast and is remembered; citation validation incl. grouped citations |
| LLM model broker | `test_llm_model_broker.py` (3) | the broker forwards to the configured model with the pinned provider order and its bearer key, replacing the requested model and passing the other fields as sent; an upstream 429 keeps its status and `Retry-After`, a 5xx → 502, empty messages, `stream: true` or a malformed body → an OpenAI-shaped 400, missing settings → 503 with a live `/health`; the whole chain narrator client → broker → upstream over local HTTP, the key only in the broker |
| model provisioning | `test_model_store.py` (2) | a verified seed is copied and a present copy kept; a corrupt file of the right size fails the SHA-256 check but passes the service's presence-and-size check, and is repaired from the seed; the bundled checkpoint matches the pinned manifest (`model`) |
| golden regression | `test_golden.py` (1) | the demo (retail, no geo, hashing double; its CSV written with LF line endings on every OS, since the dataset id hashes the file's bytes) against `tests/golden/`: the node count per kind and how many nodes of each kind carry each prop, the dataset node, the edge counts, every pattern (scope, target, support, p, adjusted p, stability, weight, shifts) and every edge, floats equal to 1e-4 relative (the float32 anchor maths differs in its last bits between BLAS builds), batch ids masked; the evidence prompt for one question verbatim. `python -m pytest tests/test_golden.py --update-golden` rewrites both files |
| end to end | `test_e2e.py` (10) | upload → … → grounded answer with provenance and cross-scope analogues (hashing double and real model); the benchmark ordering (`test_hypothesis_apparatus`); the chain-eval layers and their invariants (`test_chain_eval_apparatus`); the geo benchmark's generator and one scored scenario (`test_geo_benchmark_apparatus`); an answer embeds the question, never the corpus; missing document vectors are embedded when the state is loaded, never by a question, and saved; `Engine.search` and `search.search` give the same structured context, and the narration adds the answer and its metrics; LLM failure keeps knowledge; empty graph |
| 3D sphere | `test_sphere.py` (3) | every insight and the themes it activates placed inside the unit ball, classes `up` / `down` / `cov`, hover text, edges only between placed nodes (ACTIVATES, RELATED_TO and SPECIALIZES among them); an answer's highlighted edges between placed nodes resolve among the sphere's edges; the projection is deterministic, `GET /api/sphere` serves it, an empty workspace gives the message |
| self-containment | `test_self_contained.py` (4) | every loaded module and `sys.path` entry lies in the repository or the Python environment (interpreter, standard library, site-packages — also of a base environment the venv is layered on), the vendored engines load from the repository; no source string of the seven packages holds a machine path, and a relative path that climbs a folder appears only in a docstring and resolves inside the repository; every path setting of every section resolves inside the repository; the model loads from `models/` |
| package boundaries | `test_architecture.py` (2) | every import of the seven packages is in the allow-list of [12.2](12_architecture.md#122-dependency-direction): the repository packages and the third-party modules each package may import (relative imports resolved); only the Neo4j mirror imports the driver |
| UI | `test_ui_smoke.py` (2) | on the `scripts/dev.py` composite: static files, API upload → READY → graph → node → a grounded Ukrainian chat answer (the narrator asks the graph service over local HTTP) with its highlight; the map demo (`POST /api/demo/geo`: 884 cells at resolution 5, its insights, `/api/geo`, deletion) → delete → empty → a failed upload deleted by batch id → reset; headless Chromium: dataset card, insights table, Ukrainian chat answer with highlight and evidence cards (panels closed by default, open on click), the welcome card's questions (both demos' together, one dataset's under the dataset filter, none after the deletions), the filter naming a geo dataset's level, the legend column (defaults, counts, hover spotlight) switching sphere traces and graph edges, an isolated evidence path kept through a layer switch, sphere click → drawer, a zone of three map cells selected by real Shift-clicks and asked without text (AREA in its prompt) and cleared, a dragged column border, citation → drawer, theme toggle, delete from the card, no page errors |

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
- The browser test takes the newest Chromium in Playwright's browser cache (`PLAYWRIGHT_BROWSERS_PATH`, else `%LOCALAPPDATA%\ms-playwright` on Windows, `~/Library/Caches/ms-playwright` on macOS, `~/.cache/ms-playwright` on Linux), whatever revision the installed Playwright pins, and skips when there is none.
- The containers are verified by `scripts/compose_check.py`, outside pytest ([10.5](#105-quality-gate-and-container-check)).

## 10.2 The demo datasets

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

**The map demo** (`POST /api/demo/geo`) is a real table shipped with the service: the California housing data (Pace & Barry, 1997; 20,640 block groups with coordinates, nine columns, `data/demo/california_housing.csv`), ingested as `geo=auto@5` with `bins=median_income_median:4`. Resolution 5 was chosen on the point counts: 884 occupied cells, 344 with five or more block groups (resolution 4 gives 230 cells, too few rows for the search; 6 gives 2,530 cells, three quarters of them sparse). The band turns the cells' typical income into a dimension next to ocean proximity and the neighbourhood conditions; without it the search has three dimensions and finds ten insights, with it eighteen, among them house values high where income and density are high and low inland, and three on a `hot spot` scope. `test_ui_smoke.py` runs the demo through the API and pins the resolution, the cell count and the band dimension.

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
| transversal | **0.500** | **0.528** | **0.581** | **0.701** |
| structural | 0.000 | 0.000 | 0.078 | 0.000 |
| text_nn | 0.167 | 0.167 | 0.243 | 0.361 |
| vector_nn | 0.028 | 0.028 | 0.229 | 0.507 |

Reading:
- **Structural traversal** never reaches scope-disjoint analogues within its first ranks: they sit 2–4 lattice hops away, mixed with everything else.
- **Text-NN and raw vector kNN** rank the seed's own refinements first.
- **The anchor layer** puts the analogues at the top (MRR 0.58 vs ≤ 0.24) and reaches most of them by `k = 5` (0.70 vs 0.51).

This is an apparatus, not proof: one synthetic dataset, two mechanisms with analogues. `test_hypothesis_apparatus` guards the ordering as a regression.

**The attractor benchmark** (`python scripts/attractor_benchmark.py`, [10.4](#104-measurement-scripts)). The demo cannot tell
the anchor layer's groupings apart: with four anchors over 28 insights the latent layer is complete, so any grouping of
the same vectors gives the walk its full advantage. The benchmark that can is planted at scale. `generate(seed)` draws a
dataset of 8,000 rows over four dimensions with three cross-scope mechanisms out of eight (a driver shifted up or down in
three scopes that share no condition: discount → margin, delivery → returns, wait → satisfaction, price → churn) and one
distractor anomaly; a workspace ingests five such datasets as successive batches, and a case's analogues may come from
any of them. The same walk is run over the engine's attractors, over a static k-means of the same insight vectors with
as many clusters as the engine formed (the same `RELATED_TO` and co-membership rules), and over random memberships of
the same sizes (the null); the vector, text and weight-only rankers are the non-graph baselines. Comparisons are paired,
with a cluster bootstrap over the source datasets and Holm over the four.

Pre-registered, test seeds read once (the record is in the lac repository, [`docs/latent-attractors/experiments.md`](https://github.com/woldemarg/proj_ontology/blob/main/docs/latent-attractors/experiments.md), D1–D2):

| Attractor settings | cases | walk / attractors | walk / static k-means | random null | vector | weight only |
|---|---|---|---|---|---|---|
| the previous band (assignment 0.75–0.80, merge 0.85, cap 40) | 456 | 0.323 / 0.642 | **0.354 / 0.671** | 0.080 / 0.246 | 0.147 / 0.207 | 0.081 / 0.282 |
| **the current defaults** (assignment 0.88–0.92, merge 0.95, cap 80) | 456 | **0.360 / 0.693** | 0.368 / 0.681 | 0.081 / 0.226 | 0.147 / 0.207 | 0.081 / 0.282 |

Values are recall@5 / MRR, re-measured on 2026-10-12 after the kernel changes (mass in `w`, no confidence factor, rank
correlations; the pre-registered read of 2026-10-11 under the earlier kernel gave the same picture: −0.011 against
k-means, +0.066 over the previous band on 455 cases). Under the previous band the lifecycle's groups were larger and
mixed mechanisms, and the walk over k-means won by −0.031 [−0.059, −0.005]. The precision band fixes that: the walk
over the tuned attractors is level with the walk over k-means at the same count (−0.009 [−0.032, +0.013]), +0.037
[+0.005, +0.077] over the previous band on the same cases, +0.21 over vector neighbours; random groups collapse it to
0.08, so the grouping carries the value. The lifecycle chooses its group count
itself, incrementally; k-means was handed it. These are planted data under one embedder: the result speaks to the
generator's mechanisms, not to every insight corpus. Translated to the lac repository's text graphs, the same band
connected fewer evidence pairs (its construction record, C5), so it is this system's default and not that project's.

**Evidence chains (query-free).** `python scripts/chain_eval.py` measures the graph before any question: for each planted
mechanism with two or more patterns, the share of its pattern pairs connected within h semantic hops, per layer. A path
leaves a pattern through its anchors (ACTIVATES, 0 hops), moves between anchors along RELATED_TO / CO_OCCURS (1 hop each),
arrives at a pattern and may continue only along a lattice edge (1 hop); a pattern never bridges two anchors for free.
The null is the share of non-set pattern pairs within h, the ball the share of the patterns within h of a random one.
The same measures, with the same hop rule, run in the lac project's graph-construction record on text corpora and on this
demo exported there, so a construction change is read on both kinds of corpus. On the demo (3 sets of 5, 8 and 2 patterns,
16 scope-disjoint pairs, 4 anchors): the structural layer connects 59 % of the set pairs within two hops and 19 % of the
disjoint ones; the latent layer connects every set pair and every disjoint pair, but also 85 % of the non-set pairs (lift
1.18, two-hop ball 93 %): with four anchors the layer is complete, and the benchmark's ordering comes from the walk's
ranking, not from selective connectivity. `test_chain_eval_apparatus` guards the layer invariants.

**The geo benchmark** (`python scripts/geo_benchmark.py`, [10.4](#104-measurement-scripts)). Point datasets with a planted
effect, ingested with the geo option and scored against the truth they were generated from, so that a spatial change
brings a number, not an impression. Every scenario raises the events' `load` by 2 sd on its planted part. A focus is a
ring-3 disk of 37 cells at resolution 8: an insight covers at least `MIN_SUPPORT_ROWS` (30) cells, so a smaller focus can
never be one, and three foci fit in one cell of resolution 5. Each focus also raises its own signature metric, so each is
its own insight and not one pattern of busy cells. A part is found when the patterns with an upward shift of a planted
metric (target or secondary) that draw at least half their events from it hold at least half of its events. The walk
from a found focus counts an unfound one as a miss.

| Scenario | Planted | Baseline (hashing double; the bundled model where it differs) |
|---|---|---|
| S-regional | three foci in one resolution-5 cell, centres 10–11 cells apart, the nearest cells of two foci 4 apart (no CO_LOCATED between them) | 3/3 foci found (coverage ≥ 0.97); the walk from each reaches the other two, recall@5 1.0, mean rank 2.0; the resolution-5 region holds all three (IN_REGION 0.90–0.97) and the SCALE CHAIN lists it with their keys, but no `same_region` link joins them: each focus gives its three partner slots to the broad patterns it overlaps (`COLOCATED_TOP`); for the benchmark question the first of the ZONES is the planted resolution-5 region (both encoders); with a focus's cells as the asked zone the benchmark question seeds that focus's best pattern and the next focus's signature question still seeds a pattern raising it (`place` 3/3/3, both encoders; `SEED_PLACE_WEIGHT` 0.35, [7.2](07_question_answering.md#72-seeds)) |
| S-control | three foci in three resolution-5 cells | 3/3 found; recall@5 1.0, mean rank 2.5 (1.83); `place` 3/3/3 |
| S-local | one focus, uploaded at 8 and 7 | found (coverage 0.97); level 7 holds no insight (`no_viable_insights`), so the focus is `not_found_coarser` |
| S-levels | the eastern half of a 2.4° × 3.4° area, one upload at 7 and 5 with its table level (point columns left out) | found on every level: 20, 17 and 2 patterns holding 100 %, 99.8 % and 97.5 % of the zone's events; 1 of 3 cross-level pairs of best patterns shares an attractor; the best pattern of 7 is `regional` (its level-5 counterpart corresponds and is folded into it), that of 5 `insufficient` |
| S-parts | the northern half of the same area as three files split by longitude (39 resolution-5 cells hold events of two parts) | found in every part (14–17 patterns, ≥ 99.8 %); the parts' best patterns share an attractor; 208 CO_LOCATED between datasets (185 `adjacent`, 23 `same_region`); the items they lead to share a condition with the seeds, so none is marked as reached by location only; above part A's best pattern the resolution-3 region holds part B too, and all three parts meet in the resolution-1 region |
| S-parts-v2 | S-parts, then the middle part exported again (another seed, the same area) with `supersedes` = its first upload | 4/4 parts found (the old middle part before it went); the same pattern ids as a workspace that got the new part at once, and both anchors the new part activates existed before it (`supersede` `yes 2/2`, both encoders) |
| S-two-sources | source a: two foci in one resolution-5 cell and a third in the neighbouring cell farthest from them; source b, over the same area but for that cell, repeats focus 0 the same way and focus 1 with `load` lowered (both sources register their metrics as the same indicators) | credibility of a's best patterns `6,6,6` alone, `1,5,6` with b, and of b's `1,5` (both encoders): focus 0 confirmed, focus 1 contradicted at its place, focus 2 at 6 as b has no events there (phase A would give it 1 through focus 0 of b) |
| S-poison | source a has one focus; source b repeats it and adds a second, planted in b only, where a has its ordinary events | a's focus `6` alone, `1` with b; b's `1,2` (both encoders): the planted spot is never 1, as no pattern of a lies there; a broad pattern of a's ordinary cells holding the spot shifts `load` the other way but is no contradiction, its place far wider than the spot |
| S-size | three datasets of about 9,500 cells at 8, ten foci each | 12 of 30 foci found; 90 patterns, 348 CO_LOCATED (230 between datasets), 215 SIBLING; 1.9–2.2 s per batch, of which 0.5–1.0 s compiles the graph; 364 (372) nodes visited per seed walk; 12,363 (12,561) prompt characters, 10 % for the SCALE CHAIN and the ZONES |

Measured on 2026-10-15, seed 1, k 5 (`place`, S-parts-v2, S-two-sources, S-poison and the S-size times and prompt on 2026-10-16; the other columns unchanged). Three readings bound what the spatial issues can show:
- **The horizontal walk is saturated.** The foci of S-regional and of S-control are reached alike, through the anchors:
  one anchor they all activate in S-regional, anchors joined by RELATED_TO in S-control. The pattern of busy cells links
  them structurally too (two specialise it, the third is CO_LOCATED with it). A link by shared region cannot raise recall
  here; what separates the two scenarios is the rank of the same-region foci against S-control's, still 2.0 against 2.5
  (1.83): the foci now share their resolution-5 region, but the partner cap leaves no `same_region` link between them
  (the S-regional row). Keeping that link (a cap pool per relation, measured for #58) does not move the rank either: the
  location hop scores as the anchor path does (0.85 · 0.95 · insight weight ≈ 0.63), and scoring it higher would rank
  "the same area" above "the same phenomenon" (system-prompt rule 6).
- **A coarser level finds only broad effects.** At resolution 5 the three foci of S-regional are a handful of cells, far
  below 30, so no coarser insight can confirm them; only a zone of ≥ 30 coarse cells (S-levels) is found at every level.
- **A geo file's coordinates are no metric.** Mined as a plain table, the district condition restates `lon`, which
  becomes the target of 47 of 50 patterns, the planted `load` a secondary shift; the table level of a geo upload leaves
  the point columns out, and all 41 target `load`.

S-size finds fewer than half of its foci because the batch keeps `VALIDATION_BUDGET` (50) candidates, not because the
foci are faint: with 400 it finds 30 of 30. The cut is not even: the foci of one signature (`wait`) go in every file,
while other foci keep up to four patterns each.
`test_geo_benchmark_apparatus` guards the generator (same seed, same file; three foci in one region) and scores S-regional.

**A second source: golden facts.** The synthetic scenarios are the gauge of the trust rules; a real pair of sources checks them next to it. One real event export was split by the organisation that reported each event into two declared sources over one territory (the smaller one's resolution-5 cells lie 84–99 % inside the larger one's), with one schema, so every metric of one is the same indicator in the other; both were ingested at resolution 5 with the bundled model (10 and 15 insights). Seven facts were checked on the raw events of both sources, independently of SIG: five that both sources show the same way (expected credibility 1), one whose cells the second source covers only to 10 % (expected 6) and two areas where the two report the opposite way (expected 5; neither source mines them at resolution 5). The data are not public: the facts, their conditions, regions and raw shares, and the script that rebuilds the two sources stay outside the repository, in `data/second_source/` (git-ignored), and nothing here names a place, a unit or a time.

Measured on 2026-10-16, against the phase-A rules of credibility (anchors, [#7](https://github.com/woldemarg/proj_sig/issues/7)):
- **A shared anchor finds two of the five confirmations.** Two others are one phenomenon learned by two anchors, one per source (each source's patterns join their own), so no anchor is shared; a third pattern reaches its partner only through an anchor whose description its own shifts contradict.
- **The phenomenon-block cosine does not measure contradiction.** Across the two sources it spans −0.99 to 0.98 (median −0.35, a quarter of the pairs at most −0.67): a cutoff of −0.5 marks 71 of the 150 cross-source pairs as contradictions, nearly all of them different subgroups of different metrics, including patterns that also share an anchor with the other source.
- **Decision.** The real facts are not enough to accept M4 on them alone (the two real contradictions are not mined at the common level), so the synthetic scenarios stay the gauge; the facts are a second check for #7 and #15: the three that both rules can see (two confirmations, the uncovered place) and, through place, the other confirmations. Before #7 is built, its contradiction rule needs a measure of the same effect — the same metric (indicator) shifting the opposite way in an overlapping scope or place — rather than the block cosine alone.
- **Phase A as built** ([6.2](06_graph_and_storage.md#62-the-graph-schema): the same metric the other way in the same subgroup; the indicator registry set alike for both sources, as their schema is one). Of the seven pattern checks of the facts, four match the expected level (the three confirmations through a shared anchor and the uncovered place at 6); the other three are the confirmations split between two anchors, at 2 instead of 1. No pattern is marked 5 (10 at 1, 7 at 2, 8 at 6), where the block-cosine rule would have marked most.
- **Phase B** (place, [#15](https://github.com/woldemarg/proj_sig/issues/15)). The same seven checks give the same levels, four of seven: the split confirmations stay at 2, as the rule still needs a shared anchor. Across all patterns phase B gives 9 at 1, 14 at 2 and 2 at 6 where phase A gives 10, 7 and 8: one confirmation from another place does not count, and six patterns with the other source's events over them are 2; the place the second source covers to 10 % stays 6, and no pattern is marked 5. Comparing over the smaller place alone would add 5 for two of the confirmations, each contradicted by a wider pattern holding it.

## 10.4 Measurement scripts

The measurement scripts build throwaway workspaces under `.scratch/` (never `workspace/`), switch Neo4j off and save their results to `.scratch/results/<name>_<label>.json`. The benchmark, grounding and live-answer numbers of [7.7](07_question_answering.md#77-measured-behaviour) and [10.3](#103-hypothesis-benchmark) come from them. `scripts/scratch.py` holds the shared helpers (`scratch_dir`, `write_demo_csv`, `build_engine`, `ingest_ready`, `demo_engine`, `save_result`) and loads `.env` like the services do.

| Script | Measures |
|---|---|
| `experiment.py [--k K] [--label NAME]` | the hypothesis benchmark of [10.3](#103-hypothesis-benchmark) |
| `attractor_benchmark.py [--seeds 201-230] [--per-workspace 5] [--k K] [--label NAME] [--override NAME=VALUE ...]` | the attractor benchmark of [10.3](#103-hypothesis-benchmark): the walk over the engine's attractors against the same walk over static k-means and over random memberships, and the non-graph rankers, on planted datasets; paired comparisons with a cluster bootstrap over datasets and Holm; `--override` sets attractor settings by name for an A/B |
| `geo_benchmark.py [--scenarios S-regional,...] [--seed N] [--k K] [--encoder hashing\|model\|both] [--label NAME]` | the geo benchmark of [10.3](#103-hypothesis-benchmark): seven planted point scenarios ingested with the geo option, per encoder (the model run skipped when it is missing): parts found, the walk between foci (recall@k, reached, rank), attractors shared across levels and files, CO_LOCATED between datasets, batch and compile seconds, edges by type, nodes visited per walk, prompt length, a focus's cells as the asked zone (`place`), a part replaced by `supersedes` (`supersede`) |
| `calibration.py` | the calibration sequence of [5.10](05_latent_anchors.md#510-calibration-per-embedder) under the current attractor settings: the demo, a same-domain batch, `housing.csv`; per batch the orphan rate, the anchors minted and the warnings |
| `export_chain_demo.py <out.json>` | the demo graph (patterns with their committed vectors and scope literals, the planted sets, the snapshot's edges) as one JSON for an external chain evaluation (the lac repository's `eval_multihop` reads it as `sig_demo`) |
| `chain_eval.py [--label NAME]` | the query-free evidence-chain connectivity of [10.3](#103-hypothesis-benchmark): per layer, the share of each planted set's pattern pairs within h semantic hops, the null, the lift and the ball |
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
- **the services:** the console serves the UI with its vendored libraries, the graph service answers through it and embeds on the expected device (`embedding_device` `cpu`, or `cuda` with `--gpu`), the narrator reaches the graph service and the broker's model, Neo4j answers;
- **an empty graph** answers `answer_mode: empty`;
- **ingest:** the demo CSV uploaded through nginx (multipart) to READY, where a SKIPPED upload fails; `GET /api/health` stays under 2 s during the ingest;
- **the mirror:** Neo4j holds the snapshot's nodes label by label and all its relationships;
- **an answer:** the evidence narrated (`use_llm: false`) with evidence items, cited keys and no unknown key;
- **the sphere:** the JSON holds every insight and theme;
- **reset:** `POST /api/reset` empties the graph and the mirror, and the next question answers `empty`.

Options:
- `--tests` builds the `test` target of `insight_graph_service/Dockerfile` and runs `scripts/check.py` inside it, with the model mounted read-only, no GPU (the model tests run on the CPU) and no Chromium, so the browser test skips.
- `--fresh` rebuilds without the build cache and pulls the pinned base images (the clean-room run).
- `--live-llm` asks one question through the broker to the real upstream — a paid call, so it is opt-in.
- `--gpu` adds `compose.gpu.yaml`: the CUDA image is built, the graph service gets the GPU and must embed on CUDA.

## 10.6 Known approximations and limitations

**Statistics**
* The EDA searches only 2- and 3-conjunctions of equality selectors — no single selectors, no numeric intervals; numeric dimensions need explicit bands.
* Significance is an asymptotic median test with Bonferroni over distinct cohorts × metrics: conservative (overlapping cohorts count as independent tests), not a permutation test, and computed on cohorts the screen selected by the same shift — so it gates and is recorded, but the weight does not use it. Pre-validation near-duplicate pruning ranks by the pass-1 `temp_index`, not by the final weight, and keeps at most `VALIDATION_BUDGET` cohorts (the rest are scanned only as their aliases).
* `SD` sums the three largest shifts; `temp_index` adds clipped z-scores of incommensurable quantities (the EDA's design, monotone in each); the EMM shrinkage `sqrt((n − n_min)/(N − n_min))` is a reliability heuristic, not a standard error.
* Weight exponents, block weights, `EMM_COMPONENT_WEIGHT` and the seed-score weights are design choices validated on the synthetic data and the benchmark, not fitted.

**Representation and ontology**
* The phenomenon is linear in label embeddings: with correlated labels (Qwen3: discount · margin = 0.71), a one-metric question and a two-metric insight partly cancel ([7.2](07_question_answering.md#72-seeds)), and two one-off phenomena can share an anchor. A question naming two or more metrics gives each the same `±2` component, while an insight's phenomenon block carries its real robust-z ratios (the running example: discount +2.21, margin −1.10), so the semantic term of such a question measures "both move that way", not the insight's proportions; a question cannot know the ratio, and the lexical terms (target, direction) carry those seeds when a column name matches.
* OMP dictionary learning on tens of rows is far from its intended regime; the signed repair is what makes it usable. The adaptive assignment threshold becomes adaptive only with 10 or more anchors.
* Mutual kNN can leave an anchor without links ([5.7](05_latent_anchors.md#57-links-between-anchors)); links between anchors of different datasets are bridges by design, so a question can draw evidence from another dataset, labelled with its dataset ([5.10](05_latent_anchors.md#510-calibration-per-embedder)). `engine_weight` is not comparable across activation sources.

**Operation**
* Rollback restores the journal, the state and the snapshot; the dataset-folder files of a failed batch remain until the next ingest of that dataset overwrites them.
* The highlighted walk (`traversed`, `edges`) covers up to 15 retrieved patterns; the `evidence` group and the prompt hold the first 10. The LLM runs remotely in the configured deployment, so the prompt — subgroup statistics, not rows — leaves the machine; use a local server for fully local inference.
* One writer process per workspace: while the service runs, another writer process on the same workspace is refused (`WorkspaceBusy`); upload through the service. One graph service replica with one worker ([12.6](12_architecture.md#126-storage-files-no-sql-database-no-blob-store)).
* A workspace of another canonical or representation version is not migrated (proof-of-concept scope): it starts degraded, and `POST /api/reset` starts it over ([6.5](06_graph_and_storage.md#65-versions-degraded-start-and-reset)).
* SIG owns its seven node labels in `NEO4J_DATABASE`: every publish deletes nodes of those labels that the snapshot does not hold, so the database serves one workspace and no other data under those labels.
* The tests cover the LLM, the broker's upstream and Neo4j with a stub server and a fake driver. The live OpenRouter runs are verified by hand ([7.7](07_question_answering.md#77-measured-behaviour)) and, opt-in, by `scripts/compose_check.py --live-llm`. The Neo4j publisher is verified against a real Neo4j container by `scripts/compose_check.py` ([10.5](#105-quality-gate-and-container-check); [6.6](06_graph_and_storage.md#66-neo4j-mirror)).
