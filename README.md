# SIG — Latent Transversal Insight Representation

A research prototype that turns a tabular file into **statistically validated local insights**, embeds them in a structured vector space, and connects them to **latent statistical concepts** (attractors / latent anchors). The result is persisted as a **dual-layer graph**, explored in a web UI, and queried in natural language. Answers are grounded in graph evidence and verbalised by a local **Gemma 4**.

The hypothesis it makes testable:

> Structurally different subgroups that exhibit related statistical behaviour can be connected through latent attractor concepts. This enables *transversal* retrieval that purely structural graph traversal or naive nearest-neighbour text retrieval does not reliably achieve.

Theory and design: [`docs/init_concepts/latent_insight_graph_architecture.md`](docs/init_concepts/latent_insight_graph_architecture.md) · reconnaissance: [`docs/architecture/current_state.md`](docs/architecture/current_state.md) · specifications: [`docs/sdd/`](docs/sdd/README.md) — the mathematics in [SDD 16](docs/sdd/16_core_mathematics.md), the text contracts in [SDD 17](docs/sdd/17_textual_contracts.md) · rules for contributors and coding agents: [`AGENTS.md`](AGENTS.md).

---

## What is reused, what is new

| Layer | Source | How |
|---|---|---|
| Statistical discovery (profiling, macro screen, search space, robust median shifts, EMM correlation divergence, volume utility, bootstrap, JS confounders) | [`ltir/engines/eda/main_upd.py`](ltir/engines/eda/main_upd.py), copied from `eda/scripts/main_upd.py` | vendored with 3 integration edits and documented numerical repairs; the unused standalone runner and print-only step 5 removed ([`PROVENANCE.md`](ltir/engines/PROVENANCE.md)) |
| Dynamic ontology (ConceptStore, EMA with inertia, adaptive threshold, orphans, OMP K-sweep, soft merge, mutual kNN, journal, metrics) and the prosphera sphere projector | [`ltir/engines/lac/`](ltir/engines/lac), copied from `lac/v2_orchestrator` + `lac/v1_single_pass/visualisation/projector.py` | vendored; running-mean centering removed, signed extraction repair, per-attractor EMA damping hooks, config-driven health warnings (`PROVENANCE.md`); SIG drives lac's batch lifecycle with insight vectors |
| Embedding model | [`models/Qwen3-Embedding-0.6B/`](models) (default; Matryoshka-truncated to 384-d, bf16 on CUDA, pinned revision, fetched by `scripts/download_model.py`) or `models/paraphrase-multilingual-MiniLM-L12-v2/` (copied from lac's cache) | loaded offline from the folder |
| New in `sig/ltir` | adapter (closed intents, duplicate cohorts pruned before validation), insight model, selection & weight, canonicalisation, tripartite encoder, ontology guards, structural lattice, graph, persistence + migration, traversal, evidence, LLM, UI, tests | see [`docs/sdd/01_project_architecture.md`](docs/sdd/01_project_architecture.md) |

```text
file → ingestion → EDA pass 1 (reused) → closed intents, identical / near-duplicate cohorts merged → EDA pass 2 validation
     → selection + insight_weight → canonical scope / target / signed components → tripartite vectors
     → lac ontology (reused, with density damping + trust region): attractors, ACTIVATES, RELATED_TO
     → structural lattice: SPECIALIZES / GENERALIZES / SIBLING / CONTRASTS → journals + state + graph snapshot (+ Neo4j)
question → parse + seeds → Pattern ─ACTIVATES→ Attractor ─RELATED_TO→ Attractor ←ACTIVATES─ Pattern ─lattice→ …
         → structured evidence → Gemma 4 (or evidence-only fallback) → cited answer + highlighted path in the UI
```

---

## Setup

Python **3.12+** (the EDA script uses PEP 701 f-strings).

**Clean environment**
```powershell
cd sig
python -m venv .venv                 # or: conda create -n env_sig python=3.12
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126   # or CPU torch
.venv\Scripts\pip install -r requirements.txt
copy .env.sample .env                 # optional; defaults work
.venv\Scripts\python.exe scripts\download_model.py   # once: Qwen3-Embedding-0.6B into models/ (1.2 GB)
```
**Self-contained:** `sig/` needs nothing from the sibling `eda/` or `lac/` folders. The reused engines are vendored in `ltir/engines/` (origin, hashes and exact differences in [`ltir/engines/PROVENANCE.md`](ltir/engines/PROVENANCE.md)). The embedding model is in `models/`, and demo data is in `data/`. `tests/test_self_contained.py` fails if any code, model or config path resolves into those repos. Only pip packages come from the Python environment.

**This development machine** (reuses the existing `env_ont` without modifying it):
```powershell
cd D:\llm\sig_proj\sig
D:\conda_envs\env_ont\python.exe -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install regex pysubgroup==0.7.7 fastapi uvicorn python-multipart pytest playwright
```

### Gemma 4 (OpenRouter)
The configured deployment calls `google/gemma-4-26b-a4b-it` on OpenRouter with pinned providers, exactly like `spectr/agentic-data-science` (its `GEMMA_*` settings map to SIG's `LLM_*`). Put this in `sig/.env` (gitignored):
```ini
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemma-4-26b-a4b-it
LLM_PROVIDER_ORDER=dekallm/bf16,parasail/bf16,nextbit/bf16
LLM_API_KEY=<your OpenRouter key>
```
Check it with `.venv\Scripts\python.exe -m ltir llm-check`. Any other OpenAI-compatible endpoint also works: Ollama (`http://localhost:11434/v1`, `gemma4`) or LM Studio (`http://localhost:1234/v1`, `google/gemma-4-26b-a4b`).
Without an LLM everything still works: answers fall back to a cited, evidence-only summary, and the LLM health pill turns red.

### Neo4j mirror
In `sig/.env`: `NEO4J_ENABLED=true`, `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE=sigv1` (created automatically on multi-database editions; the DBMS must be running). Every READY batch is MERGE-published, and `python -m ltir neo4j-sync` backfills. For exploration in Neo4j Browser, use [`ltir/cypher/queries/transversal.cypher`](ltir/cypher/queries/transversal.cypher). The local journal/state/snapshot is the source of truth; Neo4j failures only produce a warning.

---

## Run

```powershell
cd D:\llm\sig_proj\sig
.venv\Scripts\python.exe -m ltir.web          # then open http://127.0.0.1:8765  (Ctrl+C to stop)
```
Startup loads the embedding model onto the GPU (~10 s, `EMBEDDING_DEVICE=cuda`; the header shows `cuda:0`; ≈ 1.2 GB resident, ≈ 1.7 GB peak). The LLM is never loaded by SIG: it is reached only through `LLM_BASE_URL`, so a local 26B model must run in its own server (CPU or partial GPU offload on an 8 GB card). The header shows the LLM, Neo4j and embedding status. Settings come from `sig/.env`; `WEB_PORT` changes the port.
1. **Add data**: drop a CSV/TSV/Parquet file on the left rail (or click **Try demo**). Under *Column options*, number-coded columns can be declared categories (`Store, Holiday_Flag`) and numeric columns split into band dimensions (`median_income:4`).
2. Watch the **batch card** move through the lifecycle stages. It shows rows, columns, candidate patterns, validated insights, latent attractors, orphan rate and graph edges.
3. **Explore.** *Graph*: the *Two planes* layout shows themes (latent anchors) on top and insights below, grouped by theme; layer toggles for hierarchy / contrasts / siblings / theme links / memberships / columns. *Insights*: a sortable, filterable table. Click any node or row for the details drawer: a one-line reading, shifts with meters, evidence facts, score breakdown, canonical form, connections and provenance. Number-coded columns (store ids, flags) can be declared under *Column options → Treat as categories*; numeric columns can be split into quantile bands.
4. **Sphere 3D** (toolbar switch). The same insights are shown as embeddings projected on a sphere, using lac's prosphera projector (KernelPCA cosine → sphere), coloured by latent anchor, with RELATED_TO lines. Rotate, zoom, hover, and click legend entries to toggle anchors or layers. After a question, the retrieval path, seed, evidence and cross-scope hits are drawn on the sphere. A standalone copy is written to `workspace/graph/sphere.html`.
5. **Chat** with the data (or click a suggestion). The answer cites `[P#]` evidence; citations open the details drawer. Seeds (gold), visited themes, evidence and **cross-segment** hits (red double ring) are highlighted, with the used paths in amber. Open *Evidence & how it was found* and click a card to isolate its path, e.g. `P → theme A-4 → P`; *What the model saw* shows the exact evidence prompt. The switch below the composer turns the LLM explanation off (evidence-only answers).

CLI equivalents:
```powershell
.venv\Scripts\python.exe -m ltir demo                                   # synthetic dataset → READY
.venv\Scripts\python.exe -m ltir ingest data\housing.csv --bins median_income:4,housing_median_age:4
.venv\Scripts\python.exe -m ltir ingest Walmart.csv --categories Store,Holiday_Flag      # integer-coded dimensions
.venv\Scripts\python.exe -m ltir query "Why is margin lower for phones in the US?"   # --no-llm, --json
.venv\Scripts\python.exe -m ltir status | experiment --k 3 | rebuild-graph | neo4j-sync | llm-check | reset --yes
.venv\Scripts\python.exe -m ltir migrate --yes                          # rebuild an outdated workspace (old copy kept)
```

## Tests and quality gate
```powershell
.venv\Scripts\python.exe scripts\check.py            # the gate: ruff check + ruff format --check + all 69 tests (≈ 85 s)
.venv\Scripts\python.exe scripts\check.py --quick    # lint + the fast tests
.venv\Scripts\python.exe -m pytest                   # tests only (never reads sig/.env)
```
Markers: `model` (needs the local embedding model) and `browser` (Playwright + an installed Chromium); both skip automatically when unavailable. The test map is in [`docs/sdd/15_testing_strategy.md`](docs/sdd/15_testing_strategy.md). Measurement scripts (prompt tokens, embedder comparison, live answer evaluation) run on throwaway workspaces under `.scratch/`; see SDD 15.

---

## Example execution (actual output)

`python -m ltir demo` on `data/demo/retail_synthetic.csv` (5 000 rows, planted phenomena, see [`ltir/synth.py`](ltir/synth.py)):
```text
READY  rows=5000 candidates=84 validated_insights=28 pruned=22 attractors=4 (+4) orphan_rate=0.00 edges=279 duration=12.6s
```
Latent anchors learned, each a recurring phenomenon across scopes:

| Attractor | Patterns | Planted phenomenon |
|---|---|---|
| `delivery days ↑ · return rate ↑` | 10 | delay → returns (APAC∧online, US∧laptops∧retail, …) |
| `discount ↑ · margin ↓` | 9 | discount erosion (US∧phones, EU∧tablets∧retail, APAC∧tablets, …) |
| `margin ↑` | 7 | EU laptops uplift + stronger online specialisation |
| `corr(discount~margin) weakens · margin ↓` | 2 | the two one-off phenomena: correlation break EU∧phones and contrasting subgroup EU∧laptops∧retail (MiniLM keeps them apart; SDD 06) |

`python -m ltir query "Why is margin lower for phones in the US?"` with Gemma 4 via OpenRouter:
```text
Observations:
In the US phone category, lower margins (median 14.75 vs. 19.45 overall) are associated with a strong increase in discounts
(median 19.19 vs. 10.74 overall) [P1]. This pattern of high discounts and lower margins persists across all US phone sales
channels, including online [P2], retail [P3], and partner [P7].

This phenomenon (latent anchor A-1: "discount up and margin down") is a recurring pattern observed in scope-disjoint segments
of the data, such as tablets in the EU [P4], APAC [P5, P8, P9, P10], and general tablet retail [P6].

Interpretation (hypotheses):
* The lower margin for phones in the US is likely driven by aggressive discounting strategies [P1, P2, P3, P7]. ...
```
The citation check passed: 10 evidence items cited, 0 unknown keys (`mode=llm`, `grounded=True`).

Evidence behind that answer (as in the evidence-only fallback):
```text
- category is phones and region is US | discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall); margin: moderate decrease, -1.10 sd (median 14.75 vs 19.45 overall) | n=438 [P1]
- category is tablets, channel is retail, and region is EU | discount: strong increase, +2.14 sd (median 18.91 vs 10.74 overall); margin: moderate decrease, -1.03 sd (median 15.03 vs 19.45 overall) | n=134 [P4] (scope-disjoint from the seed, linked via a latent anchor)
- category is tablets, channel is retail, and region is APAC | discount: strong increase, +2.09 sd (median 18.74 vs 10.74 overall); margin: mild decrease, -1.00 sd (median 15.17 vs 19.45 overall) | n=108 [P5] (scope-disjoint from the seed, linked via a latent anchor)
  [P4] transversal category=tablets AND channel=retail AND region=EU  via P-bc4657a04746 -ACTIVATES(0.98)-> A-1 <-ACTIVATES(0.99)- P-ddfe04dc0882
  [P10] transversal category=tablets AND channel=online AND region=APAC  via P-bc4657a04746 -ACTIVATES(0.98)-> A-1 <-ACTIVATES(0.90)- P-de94f9a092ae -GENERALIZES(1.00)-> P-33b175b166ec
Sources: [P1] P-bc4657a04746 = category=='phones' AND region=='US' (dataset ds-6e53eb7fb0f9, retail_synthetic.csv, batch B…); …
```
The seed `US∧phones` shares no condition with the tablet patterns; the lattice reaches them only via 2–4 sibling hops among many others. The attractor `discount ↑ · margin ↓` connects them in two hops. The prompt is plain ASCII, defines "sd" once and states every shift as a phrase plus a signed number (SDD 17).

**Hypothesis benchmark** (`python -m ltir experiment --k 3`; 12 seed cases; analogue = same planted phenomenon, no shared scope condition):

| method | recall@3 | MRR | recall@5 |
|---|---|---|---|
| transversal (this system) | **0.521** | **0.581** | **0.729** |
| structural-only BFS | 0.000 | 0.079 | 0.000 |
| naive text-NN (canonical documents) | 0.111 | 0.319 | 0.299 |
| insight-vector kNN (no attractor graph) | 0.028 | 0.229 | 0.507 |

With MiniLM the transversal row is 0.333 / 0.567 / 0.729 ([SDD 06](docs/sdd/06_embedding_layer.md) compares the embedders). Labels come from the planted ground truth (scope containment), not from the measured shifts. See SDD 15 for the reading and the caveats.

**Realistic dataset** (`data/housing.csv`, 20 640 rows, one native categorical plus two derived bands): 102 candidates → 50 validated → 40 insights → 11 new attractors (e.g. `longitude ↓ · latitude ↑`, 8 patterns). This batch was ingested *after* the demo, so it exercised lac's streaming path: all 40 were orphans for the unrelated retail attractors (at the calibrated `MIN_ASSIGN_THRESHOLD` 0.75), and the orphan buffer triggered OMP extraction. One RELATED_TO edge links a retail and a housing anchor (0.57); evidence for retail and housing questions stays within its own dataset ([SDD 07](docs/sdd/07_latent_ontology.md) §Calibration).

---

## Known limitations
* Gemma 4 runs remotely on OpenRouter. The evidence prompt (subgroup statistics, not raw rows) leaves the machine; use Ollama/LM Studio for fully local inference.
* Automated tests cover the LLM and Neo4j with a stub server / fake driver; the live OpenRouter and Neo4j runs were verified manually (SDD 09, SDD 12).
* The EDA searches only 2- and 3-conjunctions of equality selectors (no single selectors, no numeric intervals). Numeric dimensions need explicit bands.
* Query parsing is lexical plus embedding similarity; paraphrases outside the graph vocabulary rely on the semantic score.
* The significance estimate is an asymptotic median test with Bonferroni correction over distinct cohorts × metrics; it is conservative, not a permutation test, and ignores subgroup overlap. The EDA's aggregate score sums the top-3 shifts and its EMM reliability shrinkage is a heuristic (documented in SDD 03).
* All vectors share one uncentred frame (lac's running-mean centering is not used). Batches from unrelated domains arrive as orphans and mint their own attractors, provided `MIN_ASSIGN_THRESHOLD` is calibrated for the embedder (0.75 for Qwen3, 0.55 for MiniLM; SDD 07). Under Qwen3 one RELATED_TO edge can still join anchors of unrelated datasets.
* Workspaces built before `ltir-rep-3` / `ltir-canon-3` (or with another embedder) are refused with `representation_mismatch`. `python -m ltir migrate --yes` rebuilds them from their stored sources and keeps the old copy as `<workspace>.bak-<time>`.
* The hypothesis benchmark uses one synthetic dataset with two multi-scope mechanisms; it is an apparatus, not evidence.

## Repository layout
```text
sig/
  ltir/            package (config, ingestion, discovery, models, quality, canonical, encoder, ontology,
                   structural, graph, store, neo4j_sink, query, traversal, evidence, llm, qa, pipeline, migrate,
                   synth, experiment, cli, web/, cypher/)
  tests/           69 contract / integration / E2E / UI tests
  docs/sdd/        17 specifications (kept in sync with the code; 16 = mathematics, 17 = text contracts)
  scripts/         check.py quality gate (AGENTS.md Rule 0; ruff configuration in pyproject.toml) and the
                   measurement scripts (prompt_tokens, compare_embedders, eval_answers, download_model)
  docs/architecture/current_state.md   reconnaissance of the existing repo
  ltir/engines/    vendored EDA + lac engines (PROVENANCE.md)
  models/          bundled embedding models (Qwen3-Embedding-0.6B default, paraphrase-multilingual-MiniLM-L12-v2)
  data/            demo/retail_synthetic.csv (synthetic), housing.csv (realistic)
  workspace/       runtime data (created on first run; safe to delete via `ltir reset --yes`)
```
