# 12. Architecture — bounded contexts and services

> **In one paragraph.** SIG has four libraries and four services of its own, plus Neo4j.
>
> **The libraries:**
> - **`insight_contracts`**, the shared kernel. It holds the insight, the graph vocabulary, the text an insight reads as, and the evidence payload, all in the standard library only.
> - **`subgroup_miner`** turns a table into validated, weighted insights.
> - **`attractor_topology`** turns insights into vectors and a living set of latent attractors.
> - **`graph_query_engine`** turns a question over a compiled graph into evidence.
>
> Each library imports only the kernel, so each can be lifted into another project.
>
> **The services:**
> - **`insight_graph_service`** composes the three libraries over one workspace. It owns the batch lifecycle, the dual graph and the Neo4j mirror, and serves the evidence for a question.
> - **`evidence_narrator_service`** turns that evidence into a checked answer through the LLM.
> - **`llm_model_broker`** holds the upstream model and its key.
> - **`sig_web_console`** serves the UI and is the single origin for both APIs.
>
> Compose starts them in dependency order, and a test enforces every allowed import.

**Code** the packages below, `compose.yaml`, the `Dockerfile` of each service, `constraints.txt` · **Tests** `tests/test_architecture.py`, `tests/test_*_standalone.py`, `tests/test_narrator.py`, `tests/test_llm_model_broker.py`, `scripts/compose_check.py` · **Previous** [11. Reference](11_reference.md)

---

## 12.1 Bounded contexts

| Package | Context | Owns | Owning chapter |
|---|---|---|---|
| `insight_contracts/` | shared kernel (standard library only) | `insight` (`Condition`, `Shift`, `Insight`, `Rejection`, `pattern_id`, `PhenomenonThresholds`), `graph` (edge types and planes, node ids, `SNAPSHOT_VERSION` and the snapshot schema), `text` (how an insight reads: scope, shift, correlation and validation phrases), `payload` (`EvidencePayload`) | [3](03_insights.md), [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| `subgroup_miner/` | subgroup and phenomenon discovery | `ingestion` (read, validate, bands), `discovery` (profiling, closed intents, pruning, bootstrap validation, insight records), `selection` (validity rules R1–R3, evidence weight), `lattice` (SPECIALIZES / GENERALIZES / SIBLING / CONTRASTS), `describe` (insights as LLM context), `vendor/eda/` | [2](02_discovery.md), [3](03_insights.md) |
| `attractor_topology/` | representation and the living attractor space | `canonical` (the embedding inputs), `encoder` (Qwen3-Embedding-0.6B, tripartite vectors), `ontology` (OMP extraction, EMA centroids, mutual-kNN links, ACTIVATES records; its state in the folder it is given), `models` (`CanonicalInsight`, `EmbeddingSpec`, versions), `vendor/lac/` | [4](04_representation.md), [5](05_latent_anchors.md) |
| `graph_query_engine/` | grounding, seeds, transversal search, evidence | `graph` (`DualGraph`, the read model of a snapshot; `LatentFrame`), `question` (parse, literal catalog, grounding), `seeds`, `traversal`, `evidence` (the prompt and the deterministic summary), `search` (one question over a `CommittedState`; `SearchResult.payload()`), `ports` (the embedder shape it needs) | [7](07_question_answering.md) |
| `insight_graph_service/` | the aggregate root: graph lifecycle and persistence | `core/` has no web framework: `settings` (one environment for every package), `engine` (batches, admission, deletion, reset, the committed read state, `evidence`), `batch` (admission R4/R7, journal records, metrics), `snapshot` (the dual-graph compiler), `workspace` + `chunk_journal` + `fileio` (the file workspace), `neo4j_mirror` + `cypher/`, `demo`, `model_store`. `server/` holds `app` (the HTTP API) and `views` (graph, node and sphere JSON) | [6](06_graph_and_storage.md), [9](09_operations.md) |
| `evidence_narrator_service/` | grounded verbalisation | `narration` (the system prompt, the LLM-or-summary answer, the citation check, the footer), `llm_client` (OpenAI-compatible), `settings`, `app` | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) |
| `llm_model_broker/` | the upstream model | one OpenAI-compatible endpoint; the provider, the model, its key and the provider routing | [12.4](#124-services-and-contracts) |
| `sig_web_console/` | presentation and ingress | the static UI (graph, sphere, table, chat), vendored Cytoscape.js and plotly.js (`PROVENANCE.md`), `nginx.conf` | [8](08_interface.md) |

**Thresholds shared by three packages.** The miner, the topology and the query engine must agree on what makes a measurement part of a phenomenon. `PhenomenonThresholds` (`MIN_EMM_SCORE`, `WEIGHT_EMM_REF`, `MIN_COMPONENT_Z`) carries those values, together with the predicates that use them. The service's `load_settings` builds one instance and hands it to every package config. `Settings` refuses three that differ.

## 12.2 Dependency direction

```text
insight_contracts                       <- every package below (standard library only)
subgroup_miner, attractor_topology, graph_query_engine  -> insight_contracts only
insight_graph_service.core              -> the three libraries; no web or HTTP framework; only neo4j_mirror imports the driver
insight_graph_service.server            -> core (+ attractor_topology, graph_query_engine for types)
evidence_narrator_service               -> insight_contracts; reaches the graph service and the broker over HTTP
llm_model_broker                        -> nothing from the repository
```

`tests/test_architecture.py` holds this table with each package's allowed third-party modules. It resolves relative imports and fails on any import outside it.

## 12.3 Using a library on its own

| Use | Library | Example |
|---|---|---|
| statistical context for an LLM prompt from a table | `subgroup_miner` | `subgroup_miner/README.md`: `load_dataset` → `run_discovery` → `build_insights` → `select_insights` → `describe` |
| vectors and latent anchors for insights from any source | `attractor_topology` | `attractor_topology/README.md`: `canonicalize` → `InsightEncoder.encode` → `LatentOntology.ingest` |
| evidence for a question over a compiled graph | `graph_query_engine` | `graph_query_engine/README.md`: `search(question, CommittedState(...), encoder, QueryConfig())` → `payload()` |

The libraries are folders, not pip packages. Another project copies a folder, or puts the repository root on its `PYTHONPATH`, together with `insight_contracts/`, and installs the folder's `requirements.txt`. The service's `requirements.txt` includes the three library files, so each dependency is listed once.

## 12.4 Services and contracts

| Service | Routes |
|---|---|
| graph service (`python -m insight_graph_service.server`; port 8000 in the stack) | `GET /api/health`, `GET /api/batches[/{id}]`, `POST /api/upload`, `POST /api/demo`, `GET /api/graph`, `GET /api/nodes/{id}`, `GET /api/sphere` (projected points and edges, JSON), `DELETE /api/datasets/{id}`, `POST /api/reset`, `POST /api/search` (the evidence for one question) |
| narrator (`python -m evidence_narrator_service`) | `POST /api/chat/query` (`{question, use_llm}` → the answer), `GET /api/chat/health` (always 200: the LLM and the graph service) |
| broker (`python -m llm_model_broker`) | `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` |

**`EvidencePayload`** (`insight_contracts/payload.py`) is the whole contract between the graph service and the narrator:
- `question`.
- `graph_empty`: `true` while the graph holds no insight (an ordinary state).
- `evidence_prompt`: the LLM-ready evidence.
- `evidence_summary`: the deterministic, cited observation lines, which the narrator frames as the answer when no LLM answers.
- `citations`: the citation manifest, each `P#` → `{pattern_id, expression, dataset_id, filename, batch_id}`.
- `view`: what the console draws: `evidence` (the cards, with `parsed`: targets, direction, conditions, grounding), `traversal` (with the baselines) and `highlight` (node and edge ids per group: `seeds`, `traversed`, `anchors`, `evidence`, `edges`, `transversal_only`).
- `metrics`: `retrieval_s`, `seed_count`, `traversal_depth`, `visited_states`, `retrieved_evidence`, `anchors_visited`.

The narrator reads the prompt, the summary, the citation manifest and the metrics. Its answer, `QAResult`, carries `view` untouched for the console:
- `question`, `answer`, `answer_mode` (`llm`, `fallback` or `empty`).
- `llm`: `{model, ok, latency_s, error[, usage]}`.
- `citations`: `{cited [{key, pattern_id}], unknown, uncited, grounded}`.
- `prompt`: the evidence prompt the model is shown.
- `view`: the payload's `view` as received.
- `metrics`: the payload's metrics plus `total_s`, `llm_latency_s` and `prompt_chars`.
- `provenance_footer`.

A missing field fails `EvidencePayload.from_dict`, and the narrator answers 502: a broken contract is an error, not a silent default.

**Status semantics:**
- **An empty graph is an ordinary state.** `/api/search` answers 200 with `graph_empty: true`, and the narrator answers 200 with `answer_mode: "empty"`.
- **A workspace that cannot answer** gets 409 with `{detail, code}`; `code` is `representation_mismatch` or `workspace_degraded`. The narrator passes it through as sent.
- **Every refusal of the graph service** ([9.4](09_operations.md#94-error-codes)) has the body `{detail, code}`: 413 for `file_too_large`, 404 for `unknown_dataset`, 409 for every other code (`busy`, `workspace_degraded`, `representation_mismatch`, `rollback_pending`, …). An unexpected error answers 500 with `code: internal_error`.
- **An unreachable graph service**, any other status from it, or a payload that breaks the contract gives a 502 from the narrator.

**The broker** replaces the request's `model` with the configured one. Every other field (`temperature`, `max_tokens`, `top_p`, `stop`, `response_format`, …) is forwarded as sent. `stream: true` and a malformed body are refused with a 400, not FastAPI's 422. The broker adds the bearer key and, for OpenRouter, the pinned provider order without fallbacks. Error statuses:

| Situation | Status |
|---|---|
| Upstream client error | kept as sent, with `Retry-After` (a 429 stays a 429) |
| Upstream server error | 502 |
| Timeout | 504 |
| Missing settings | 503 |

Error bodies have the OpenAI shape `{"error": {"message", "type"}}`.

| Broker setting | Default | Meaning |
|---|---|---|
| `GEMMA_BASE_URL` | — (required) | upstream base URL, e.g. `https://openrouter.ai/api/v1`, or a local Ollama / LM Studio |
| `GEMMA_MODEL_NAME` | — (required) | upstream model id |
| `GEMMA_CHAT_ENDPOINT` | `/chat/completions` | path appended to the base URL |
| `GEMMA_API_KEY` | "" | bearer token for the upstream |
| `GEMMA_PROVIDER` | "" | OpenRouter provider slugs, tried in order, no fallbacks |
| `GEMMA_CONNECT_TIMEOUT_SEC`, `GEMMA_READ_TIMEOUT_SEC` | 5, 120 | upstream timeouts |
| `BROKER_HOST`, `BROKER_PORT` | `127.0.0.1`, `8080` | where `python -m llm_model_broker` listens (`0.0.0.0` in the image) |

The narrator knows only `LLM_BASE_URL`, the broker, and sends no model name or key. Changing the model or the provider therefore changes only the broker's settings. Only the broker reads the `GEMMA_*` settings, and the other services never hold the key: compose passes the broker the `GEMMA_*` values explicitly and keeps them from the other containers ([12.5](#125-containers-and-start-order)), and the other host entry points skip `GEMMA_*` lines when they read `.env`.

## 12.5 Containers and start order

| # | Service | Image | Waits for | Health | Networks | State |
|---|---|---|---|---|---|---|
| 1 | `model-init` | the graph service image | — | runs once, exits 0 | edge | volume `sig-models` (read-write); `./models` read-only as a seed |
| 1 | `neo4j` | `neo4j:5.26.31-community` | — | `cypher-shell RETURN 1` | internal, edge (ports 17474 / 17687) | volume `neo4j-data` |
| 1 | `llm-model-broker` | `llm_model_broker/Dockerfile` | — | `GET /health` | internal, edge (egress to the provider) | none |
| 2 | `insight-graph` | `insight_graph_service/Dockerfile` (CPU torch) | `model-init` completed, `neo4j` healthy | `GET /api/health` (it listens once the model is checked and warm) | internal | volume `sig-workspace` at `/data`; `sig-models` read-only |
| 3 | `evidence-narrator` | `evidence_narrator_service/Dockerfile` | `insight-graph` and the broker healthy | `GET /api/chat/health` | internal | none |
| 4 | `sig-web-console` | `sig_web_console/Dockerfile` (`nginx-unprivileged`) | both services healthy | `GET /` | internal, edge (port 8765) | none |

```text
docker compose up -d --build --wait               # then http://127.0.0.1:8765
python scripts/compose_check.py                   # verification: its own project, ports and volumes, beside your stack
python scripts/compose_check.py --fresh --tests   # clean room: no build cache, pulled base images, the test suite inside the image
```

**Networks.** `internal` has no egress. `edge` publishes ports and has egress: `model-init`, `neo4j`, the broker and the console sit on it; the graph service and the narrator are on `internal` only, with no egress.

**Hardening.** Every image except Neo4j runs as an unprivileged user with a read-only root filesystem, a tmpfs at `/tmp`, `no-new-privileges` and no capabilities.

**Settings.** The graph service reads `.env` (`env_file`). Its container values are overridden in `environment`: the workspace and model paths, `EMBEDDING_DEVICE=cpu`, the Neo4j address and password (`SIG_NEO4J_PASSWORD`), and blanks for the broker's `GEMMA_*` values and for `LLM_API_KEY`: the graph service reads no LLM key, so a key an `.env` file may still hold never reaches it. The narrator gets no `env_file`, only an allow-list (`INSIGHT_GRAPH_URL`, `INSIGHT_GRAPH_TIMEOUT_S`, `LLM_BASE_URL`, `LLM_TIMEOUT_S`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`), so no key from `.env` reaches it either.

**The model.** `model-init` checks the pinned Qwen3 checkpoint file by file against `core/model_store.py`'s manifest (sizes and SHA-256 values, checked against the Hugging Face Hub):
- A copy in the volume that verifies is kept.
- Otherwise a verified seed from `./models` is copied.
- Otherwise the pinned revision is downloaded.
- Anything that does not verify stops the start.

Before torch loads, the graph service checks presence and sizes again and refuses to start with a message naming the step to run.

**Robustness.**
- **Neo4j at start.** The graph service syncs Neo4j once at start, and never from an empty or degraded workspace.
- **Stopping.** It gets 60 s to stop, so a running batch can finish; a batch cut short is rolled back at the next start ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)).
- **Workers.** It runs one uvicorn worker, because the workspace has one writer.
- **Health.** `/api/health` runs on the event loop, so it answers while a batch runs.

**Memory and versions.** The graph service needs about 2.5 GB of RAM on CPU. `constraints.txt` holds the exact package versions the images were verified with.

**Why these services.** Each one has its own runtime, state and reason to change:
- **The graph service** holds the embedding model, the workspace and the in-memory graph. Retrieval needs all three, so search lives here, beside the topology it walks.
- **The narrator** needs only HTTP and the answer rules. It changes with the prompt and the citation policy, and it scales or fails without touching the graph.
- **The broker** holds the provider key and changes with the model or the provider.
- **The console** is static files and an edge proxy.
- **Neo4j** is a database with its own storage.

## 12.6 Storage: files, no SQL database, no blob store

The graph service is the only writer of its workspace volume. The workspace already gives what a database would:
- atomic file replacement;
- transactions with checkpoints and rollback;
- recovery at start;
- a writer lock ([6.3–6.4](06_graph_and_storage.md#63-the-workspace-on-disk)).

Neo4j mirrors the graph for people and other applications. The narrator is stateless; its query log is one JSON line per answer on its log. Neither SQLite nor Postgres has a consumer. The limitation is explicit: one graph service replica and one worker.

| When this becomes true | Add |
|---|---|
| chat history must persist (one narrator instance) | SQLite in the narrator |
| a second graph replica, or several users' histories | Postgres for the registry and journal, object storage (S3 / MinIO) for uploads and vectors |

## 12.7 Observability

**Logging.** Logging is the standard `logging` module. Every module that logs uses `logging.getLogger(__name__)`, except the broker, which logs as `llm_model_broker`, and the narrator's query log (below); each service's entry point configures it.

**The domain's own records:**
- **The batch record:** stage times, metrics, warnings and errors ([9.1](09_operations.md#91-the-batch-lifecycle)).
- **The ontology's metrics CSV** ([5](05_latent_anchors.md)).
- **The narrator's query log:** logger `evidence_narrator_service.queries`, one JSON line with the time, the question, the mode, the metrics, the cited evidence (the manifest's pattern ids) and the citation check.
- **Per-answer metrics:** the `metrics` of every answer.

**The broker** logs one line per completion: model, messages, prompt and completion tokens, and latency. It never logs the prompt itself.

**Liveness.** There is no tracing or metrics backend; the Docker health checks cover liveness.
