# SIG — Statistical Insight Graph

A research prototype that turns a tabular file into **statistically validated local insights**, embeds them in a structured vector space, and connects them to **latent statistical concepts** (attractors / latent anchors). The result is persisted as a **dual-layer graph**, explored in a web console, and queried in natural language. Answers are grounded in graph evidence and verbalised by **Gemma 4** (on OpenRouter, or a local OpenAI-compatible server).

The hypothesis it makes testable:

> Structurally different subgroups that exhibit related statistical behaviour can be connected through latent attractor concepts. This enables *transversal* retrieval that purely structural graph traversal or naive nearest-neighbour text retrieval does not reliably achieve.

**Documentation:** [`docs/`](docs/README.md) has eleven chapters that follow the data from the uploaded table to the cited answer, each with its formulas, text contracts, code and measured behaviour. A twelfth chapter covers the [architecture](docs/12_architecture.md): bounded contexts, services, contracts, containers, and the storage decision. Start with the [reading guide](docs/README.md) and the [overview](docs/01_overview.md). Theory the design started from: [`docs/init_concepts/`](docs/init_concepts/). Rules for contributors and coding agents: [`AGENTS.md`](AGENTS.md).

---

## Architecture

```text
browser ──> sig-web-console (nginx: the UI; /api/chat/ -> narrator, /api/ -> graph service)
              │
              ├─ /api/chat/ ──> evidence-narrator ── HTTP ──> llm-model-broker ── HTTPS ──> upstream model (OpenRouter, a local server)
              │                      │ POST /api/search                 one model, its key, the provider routing
              └─ /api/ ─────> insight-graph (graph service) ── bolt ──> neo4j (graph mirror)
                              subgroup_miner -> attractor_topology -> snapshot -> graph_query_engine
                              workspace volume (journals, state, snapshot); Qwen3 on CPU
```

| Package | Role |
|---|---|
| [`insight_contracts/`](insight_contracts) | the shared kernel (standard library only): the insight, the graph vocabulary, how an insight reads, the evidence payload |
| [`subgroup_miner/`](subgroup_miner/README.md) | a table → validated, weighted insights; usable alone, e.g. as statistical context for an LLM prompt |
| [`attractor_topology/`](attractor_topology/README.md) | insights → tripartite vectors (Qwen3-Embedding-0.6B) and a living set of latent attractors |
| [`graph_query_engine/`](graph_query_engine/README.md) | a question over the dual graph → grounding, seeds, transversal paths, evidence |
| [`insight_graph_service/`](insight_graph_service) | the graph service: `core/` (engine, workspace, snapshot compiler, Neo4j mirror, model provisioning) and `server/` (HTTP API) |
| [`evidence_narrator_service/`](evidence_narrator_service) | the chat: the graph's evidence verbalised by the LLM, every citation checked |
| [`llm_model_broker/`](llm_model_broker) | one OpenAI-compatible endpoint in front of the upstream model |
| [`sig_web_console/`](sig_web_console) | the static UI and its nginx edge |

Each library imports only the shared kernel; `tests/test_architecture.py` enforces every allowed import. Bounded contexts, contracts, start order and why these services: [docs/12_architecture.md](docs/12_architecture.md).

```text
file → ingestion → EDA pass 1 (reused) → closed intents, identical / near-duplicate cohorts merged → EDA pass 2 validation
     → validity rules + insight_weight → admission → canonical scope / target / signed components → tripartite vectors
     → lac ontology (reused, with density damping + trust region): attractors, ACTIVATES, RELATED_TO
     → structural lattice: SPECIALIZES / GENERALIZES / SIBLING / CONTRASTS → journals + state + graph snapshot (+ Neo4j)
question → parse + seeds → Pattern ─ACTIVATES→ Attractor ─RELATED_TO→ Attractor ←ACTIVATES─ Pattern ─lattice→ …
         → evidence payload → Gemma 4 (or evidence-only fallback) → cited answer + highlighted path in the console
```

| Reused | Source | How |
|---|---|---|
| Statistical discovery (profiling, macro screen, search space, robust median shifts, EMM correlation divergence, volume utility, bootstrap, JS confounders) | the eda project: `scripts/main_upd.py` | vendored as [`subgroup_miner/vendor/eda/main_upd.py`](subgroup_miner/vendor/eda/main_upd.py); its differences in [`PROVENANCE.md`](subgroup_miner/vendor/PROVENANCE.md) |
| Dynamic ontology (ConceptStore, EMA with inertia, adaptive threshold, orphans, OMP K-sweep, soft merge, mutual kNN, metrics) | the lac project: `v2_orchestrator/` | vendored as [`attractor_topology/vendor/lac/`](attractor_topology/vendor/lac); its differences in [`PROVENANCE.md`](attractor_topology/vendor/PROVENANCE.md) |
| Embedding model | Qwen3-Embedding-0.6B at a pinned revision (Matryoshka-truncated to 384-d, bf16) | provisioned and verified file by file (`python -m insight_graph_service.core.model_store`) |

---

## Setup and run

Python **3.12+** (the EDA script uses PEP 701 f-strings).

**Docker** (the service suite)
```powershell
copy .env.sample .env                     # then edit: GEMMA_* (the broker's upstream, model and key), tunables
docker compose up -d --build --wait       # then http://127.0.0.1:8765
python scripts\compose_check.py           # verification: its own stack, volumes and ports — runs beside yours
```
The first start fills the `sig-models` volume. It copies a verified `./models/Qwen3-Embedding-0.6B` when there is one, and otherwise downloads the pinned revision. Neo4j Browser runs at http://127.0.0.1:17474; connect to `bolt://127.0.0.1:17687` as `neo4j` with `SIG_NEO4J_PASSWORD` (default `sig-local-password`). The containers compute embeddings on the CPU (≈ 2.5 GB RAM for the graph service). The host setup below can use the GPU.

**Host development**
```powershell
python -m venv .venv                                                                # from the repository root
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126   # or CPU torch
.venv\Scripts\pip install -r requirements-dev.txt                                    # every service's runtime + pytest, ruff, playwright
.venv\Scripts\python.exe -m insight_graph_service.core.model_store                  # once: Qwen3-Embedding-0.6B into models/ (1.19 GB), verified
.venv\Scripts\python.exe -m llm_model_broker                                         # the broker on :8080 (reads GEMMA_* from .env)
.venv\Scripts\python.exe scripts\dev.py                                              # console + graph service + narrator on http://127.0.0.1:8765
```
`scripts/dev.py` routes paths as the console's nginx does, so the UI behaves as in the stack. Without a `.env`, the code defaults apply: the broker on `http://127.0.0.1:8080/v1`, Neo4j off, device `auto`. Copy `.env.sample` to `.env` and edit it.

**Self-contained:** at run time the repository needs nothing outside itself.
- **The reused engines** are vendored in the packages; their origin, hashes and exact differences are in the `PROVENANCE.md` files.
- **The embedding model** lives in `models/`, and the data in `data/`. Both folders are local and git-ignored.
- **The check.** `tests/test_self_contained.py` fails in any of these cases:
  - a loaded module or a `sys.path` entry lies outside the repository and the Python environment;
  - a source file holds a machine path, or a relative path leading out of the repository;
  - a path setting resolves outside the repository.

### Gemma 4 through the LLM model broker
The narrator reaches the LLM only through `LLM_BASE_URL`, by default the broker (`llm_model_broker/`, [12.4](docs/12_architecture.md#124-services-and-contracts)). The broker holds the upstream, the model, its key and the OpenRouter provider pinning. The configured deployment calls `google/gemma-4-26b-a4b-it` on OpenRouter with pinned providers. Put this in `.env` (gitignored; template `.env.sample`):
```ini
GEMMA_BASE_URL=https://openrouter.ai/api/v1
GEMMA_CHAT_ENDPOINT=/chat/completions
GEMMA_MODEL_NAME=google/gemma-4-26b-a4b-it
GEMMA_PROVIDER=dekallm/bf16,parasail/bf16,nextbit/bf16
GEMMA_API_KEY=<your OpenRouter key>
```
Only the broker reads these; the other services never hold the key ([12.4](docs/12_architecture.md#124-services-and-contracts)).

`GEMMA_BASE_URL` can instead point at any other OpenAI-compatible server, for example Ollama (`http://host.docker.internal:11434/v1` from a container, model `gemma4`) or LM Studio.

Without an LLM everything still works: answers fall back to a cited, evidence-only summary, and the LLM health pill turns red.

### Neo4j mirror
**In the stack.** Compose runs its own Neo4j Community container with database `neo4j`.

**On the host,** set these in `.env`: `NEO4J_ENABLED=true`, `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE=sigv1` (created automatically on multi-database editions; the DBMS must be running).

**What keeps it in sync:**
- Every READY batch and dataset deletion makes the mirror equal to the snapshot, and stale nodes and properties are removed.
- A reset clears it.
- A start of the graph service syncs a non-empty, healthy workspace.

SIG owns its six node labels in that database, so use one workspace per database. For exploration in Neo4j Browser, use [`transversal.cypher`](insight_graph_service/core/cypher/queries/transversal.cypher). The local journal, state and snapshot are the source of truth; a Neo4j failure only produces a warning.

### Using the console
1. **Add data.** Drop a CSV, TSV or Parquet file on the left rail, or click **Try demo**.
   - Under *Column options*, number-coded columns can be declared categories (`Store, Holiday_Flag`), and numeric columns split into band dimensions (`median_income:4`).
   - A dataset card's *Delete* button removes that dataset again: its insights, vectors and memberships, and the themes left with no insight and no theme link ([6.8](docs/06_graph_and_storage.md#68-deleting-a-dataset)).
2. **Watch the batch card** move through the lifecycle stages. When READY it shows the insights, the new themes and the rows, the subgroups tested, how many were filtered out, the duration and the dimension tags.
3. **Explore.** *Graph*, *3D Sphere* and *Insights* are views of the same knowledge base under one legend.
   - The legend's link switches show the layers in both views. Hierarchy and theme links are on by default; contrasts, siblings, memberships and columns come on demand.
   - Hovering a legend entry spotlights it.
   - Click any node or row for the details drawer: a one-line reading, shifts, evidence facts, the canonical form, connections and provenance.
4. **3D Sphere.** The same insights appear as embeddings in the unit ball, projected by the graph service (KernelPCA, cosine) and drawn in the browser with the graph's colours, markers and layers. After a question, the retrieval path, seed, evidence and cross-scope hits are drawn on it.
5. **Chat** with the data, or click a suggestion.
   - Answers are in Ukrainian. Every data literal (`margin`, `phones`, `US`, ids) stays exactly as the data holds it.
   - The answer cites its `[P#]` evidence, and a citation opens the details drawer.
   - Seeds, visited themes, evidence and **cross-segment** hits are highlighted with the used paths. Click an evidence card to isolate its path.
   - The *Prompt* tab shows the exact evidence the model saw.

**Without the UI**, use the HTTP API through the console's port:
```powershell
curl -X POST http://127.0.0.1:8765/api/demo                                        # the synthetic dataset -> a batch record
curl -F file=@data\housing.csv -F bins=median_income:4,housing_median_age:4 http://127.0.0.1:8765/api/upload
curl http://127.0.0.1:8765/api/batches                                             # status of every batch
curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"Why is margin lower for phones in the US?\", \"use_llm\": false}" http://127.0.0.1:8765/api/chat/query
curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"Why is margin lower for phones in the US?\"}" http://127.0.0.1:8765/api/search   # the evidence only
curl -X DELETE http://127.0.0.1:8765/api/datasets/<dataset id>
curl -X POST http://127.0.0.1:8765/api/reset                                       # delete the workspace (and clear the mirror)
```

## Tests and quality gate
```powershell
.venv\Scripts\python.exe scripts\check.py            # the gate: ruff check + ruff format --check + all tests (≈ 2 min)
.venv\Scripts\python.exe scripts\check.py --quick    # lint + the fast tests
.venv\Scripts\python.exe -m pytest                   # tests only (never reads .env)
```
**Markers.** `model` needs the local embedding model, and `browser` needs Playwright and an installed Chromium; both skip automatically when unavailable.

**In containers.** `python scripts\compose_check.py --tests` runs the gate inside the graph service's image as well ([10.5](docs/10_verification.md#105-quality-gate-and-container-check)).

**Where to look next.** The test map is in [10.1](docs/10_verification.md#101-test-map). The measurement scripts (live answer evaluation, the multilingual benchmark, the hypothesis experiment) run on throwaway workspaces under `.scratch/` ([10.4](docs/10_verification.md#104-measurement-scripts)).

---

## Example execution (actual output)

`POST /api/demo` on an empty workspace ingests the demo dataset (5 000 rows with planted phenomena, see [`insight_graph_service/core/demo.py`](insight_graph_service/core/demo.py)). An excerpt of the batch record that `GET /api/batches/<id>` then returns:
```text
{"batch_id": "B20261004T005725-2b9928", "dataset_id": "ds-6e53eb7fb0f9", "filename": "retail_synthetic.csv", …, "status": "READY", …,
 "metrics": {"input_rows": 5000, …, "candidate_patterns": 84, "validated_candidates": 50, "validated_insights": 28, …, "pruned_total": 22, …,
             "attractors_total": 4, "attractors_new": 4, "orphan_rate": 0.0, …, "avg_attractor_degree": 1.5, "graph_edges": 279, …,
             "processing_duration_s": 12.613771399999678}, …}
```
Latent anchors learned, each a recurring phenomenon across scopes:

| Attractor | Patterns | Planted phenomenon |
|---|---|---|
| `delivery days ↑ · return rate ↑` | 10 | delay → returns (APAC∧online, US∧laptops∧retail, …) |
| `discount ↑ · margin ↓` | 9 | discount erosion (US∧phones, EU∧tablets∧retail, APAC∧tablets, …) |
| `margin ↑` | 7 | EU laptops uplift + stronger online specialisation |
| `corr(discount~margin) weakens · margin ↓` | 2 | the two one-off phenomena: correlation break EU∧phones and contrasting subgroup EU∧laptops∧retail |

"Why is margin lower for phones in the US?" with `use_llm: false` gives the evidence-only answer (`answer_mode: "fallback"`):
```text
Спостереження:
- category=phones, region=US | discount +2.21 sd (медіана 19.19 проти 10.74); margin -1.10 sd (медіана 14.75 проти 19.45) | n=438 [P1]
- category=phones, channel=online, region=US | discount +2.25 sd (медіана 19.35 проти 10.74); margin -1.10 sd (медіана 14.72 проти 19.45) | n=205 [P2]
…
- category=tablets, channel=retail, region=EU | discount +2.14 sd (медіана 18.91 проти 10.74); margin -1.03 sd (медіана 15.03 проти 19.45) | n=134 [P4] (інший сегмент: без спільної умови із запитом, знайдено через латентну тему)
- category=tablets, channel=retail, region=APAC | discount +2.09 sd (медіана 18.74 проти 10.74); margin -1.00 sd (медіана 15.17 проти 19.45) | n=108 [P5] (інший сегмент: без спільної умови із запитом, знайдено через латентну тему)
…
Інтерпретація (гіпотези): не сформовано (відповідь мовної моделі недоступна).
```
The citation check passed: all 10 evidence items cited, 0 unknown keys (`grounded: true`). With the broker configured, the LLM writes the same evidence as a Ukrainian answer with `[P#]` citations (`answer_mode: "llm"`), checked the same way.

The prompt (the *Prompt* tab) shows how [P4] was reached: `P-bc4657a04746 -ACTIVATES(0.98)-> A-1 <-ACTIVATES(0.99)- P-ddfe04dc0882`.
- **No shared scope.** The seed `US∧phones` shares no condition with the tablet patterns.
- **Lattice vs anchor.** The lattice reaches them only through 2–4 hops among many other patterns. The anchor `discount ↑ · margin ↓` connects them in two hops.
- **The prompt** is plain ASCII. It defines "sd" once and states every shift as a phrase plus a signed number ([7.4](docs/07_question_answering.md#74-the-evidence-object)).

**Hypothesis benchmark** (`python scripts/experiment.py --k 3`). It uses 12 seed cases; an analogue is a pattern with the same planted phenomenon and no shared scope condition.

| method | recall@3 | MRR | recall@5 |
|---|---|---|---|
| transversal (this system) | **0.521** | **0.581** | **0.729** |
| structural-only BFS | 0.000 | 0.079 | 0.000 |
| naive text-NN (canonical documents) | 0.111 | 0.321 | 0.299 |
| insight-vector kNN (no attractor graph) | 0.028 | 0.229 | 0.507 |

Labels come from the planted ground truth (scope containment), not from the measured shifts. See [10.3](docs/10_verification.md#103-hypothesis-benchmark) for the reading and the caveats.

**Realistic dataset** (`data/housing.csv`, 20 640 rows, one native categorical plus two derived bands):

| Stage | Count |
|---|---|
| candidates | 102 |
| validated | 50 |
| insights | 40 |
| new attractors | 11 (e.g. `longitude ↓ · latitude ↑`, 8 patterns) |

- **Ingested after the demo,** this batch exercised lac's streaming path. All 40 insights were orphans for the unrelated retail attractors at the calibrated `MIN_ASSIGN_THRESHOLD` of 0.75, and the orphan buffer triggered OMP extraction.
- **Cross-dataset links.** One RELATED_TO edge links a retail anchor and a housing anchor (0.57).
- **Evidence stayed local.** On six retail and housing questions the evidence stayed within its own dataset ([5.10](docs/05_latent_anchors.md#510-calibration-per-embedder)).

---

## Known limitations
The full list, with the reasons, is in [10.6](docs/10_verification.md#106-known-approximations-and-limitations). The ones that matter first:
* Gemma 4 runs remotely on OpenRouter: the evidence prompt (subgroup statistics, not raw rows) leaves the machine; use Ollama or LM Studio for fully local inference.
* The EDA searches only 2- and 3-conjunctions of equality selectors (no single selectors, no numeric intervals); numeric dimensions need explicit bands.
* Significance is an asymptotic median test with Bonferroni over distinct cohorts × metrics — conservative, not a permutation test. A correlation-change insight has no median test: it is validated by its correlation change alone.
* The embedder is Qwen3-Embedding-0.6B and the cosine thresholds are calibrated for it ([5.10](docs/05_latent_anchors.md#510-calibration-per-embedder)); under Qwen3 a few RELATED_TO edges join anchors of unrelated datasets.
* **Workspaces from another representation are not migrated (proof-of-concept scope).** A workspace written by another canonical or representation version starts degraded (409 `workspace_degraded`, [6.5](docs/06_graph_and_storage.md#65-versions-degraded-start-and-reset)); one with another representation fingerprint ([4.6](docs/04_representation.md#46-representation-identity-and-versions)) fails batches at EMBEDDING and refuses questions (409 `representation_mismatch`). `POST /api/reset` starts either over.
* One graph service replica and one worker per workspace (a writer lock): the file workspace is the store ([12.6](docs/12_architecture.md#126-storage-files-no-sql-database-no-blob-store)).
* Ukrainian questions are grounded onto the data's literals through the embedding model (*телефонів* → `phones`, *США* → `US`). The literals Qwen3 cannot bridge on the demo (*маржа*, *частка повернень*, *роздріб*) are left unmatched rather than guessed ([7.1.1](docs/07_question_answering.md#711-literal-grounding)).
* Deleting a dataset keeps a theme that still links to another theme even when it has no insight left (its centroid can receive future data); such a theme reads `Attractor k` until it has members again.
* The hypothesis benchmark uses one synthetic dataset with two multi-scope mechanisms; it is an apparatus, not evidence.

## Repository layout
```text
sig/
  insight_contracts/          the shared kernel (standard library only)
  subgroup_miner/             table -> insights (vendor/eda, PROVENANCE.md, README.md, requirements.txt)
  attractor_topology/         insights -> vectors -> latent attractors (vendor/lac, PROVENANCE.md, README.md, requirements.txt)
  graph_query_engine/         question -> evidence over the dual graph (README.md, requirements.txt)
  insight_graph_service/      the graph service: core/ (engine, batch, snapshot, workspace, neo4j_mirror, demo, model_store,
                              settings), server/ (app, views); Dockerfile (targets runtime, test), requirements.txt
  evidence_narrator_service/  the chat service (Dockerfile, requirements.txt)
  llm_model_broker/           the LLM model broker (Dockerfile, requirements.txt)
  sig_web_console/            the static UI, nginx.conf, Dockerfile, PROVENANCE.md (vendored browser libraries)
  compose.yaml, constraints.txt, .dockerignore   the service suite in start order; the images' pinned versions
  requirements-dev.txt        host development: every service's runtime + the test tools
  tests/                      contract / integration / E2E / UI tests (doubles.py: the deterministic test embedder)
  docs/                       eleven chapters in pipeline order and the architecture chapter; init_concepts/ (the theory)
  scripts/                    check.py (the gate), compose_check.py (the Docker check), dev.py (host composite),
                              multilingual_benchmark.py, eval_answers.py, experiment.py (measurements), scratch.py (their helpers)
  models/, data/              local, git-ignored: the embedding model; housing.csv
  workspace/                  the knowledge base of host runs (created on first run; POST /api/reset deletes it)
```
