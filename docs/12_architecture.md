# 12. Architecture — layers, services and reuse

> **In one paragraph.** SIG is one Python package, `ltir`, cut into layers by responsibility, and two more processes beside it. The *analytical core* — `analysis` (a table becomes validated insights, vectors, latent anchors and the dual graph), `retrieval` (a question becomes seeds, transversal paths and an evidence object) and `storage` (the file workspace and the Neo4j mirror) — imports no presentation, chat or LLM code. The `Engine` orchestrates the core over one workspace; `answering` turns a search into a chat answer through `llm_client`; `web` serves the API, the chat endpoint and the UI. The LLM itself sits behind a separate gateway service, and Neo4j runs as its own container. Every cross-layer import is in one table, and a test enforces it.

**Code** `ltir/` (every package below), `llm_gateway/`, `Dockerfile`, `llm_gateway/Dockerfile`, `compose.yaml`, `constraints.txt` · **Tests** `tests/test_architecture.py`, `tests/test_llm_gateway.py`, `scripts/compose_check.py` · **Previous** [11. Reference](11_reference.md)

---

## 12.1 Layers and responsibilities

| Layer | Path | Responsibility | Owning chapter |
|---|---|---|---|
| shared contracts | `ltir/config.py`, `ltir/models.py`, `ltir/fileio.py` | every tunable; typed records (`Insight`, `CanonicalInsight`, `EmbeddingSpec`, `LatentFrame`, graph ids and edge types, version strings, `utc_now`); atomic file replacement | [9.3](09_operations.md#93-configuration), [3](03_insights.md), [6.3](06_graph_and_storage.md#63-the-workspace-on-disk) |
| vendored engines | `ltir/engines/` | the EDA engine and the lac ontology, as copied and repaired (`PROVENANCE.md`) | [2](02_discovery.md), [5](05_latent_anchors.md) |
| analytical core | `ltir/analysis/` | `ingestion` (read, validate, bands), `discovery` (profiling, EDA, insight records), `quality` (selection, evidence weight), `canonical` (text contract), `encoder` (embedders, tripartite vectors), `ontology` (latent anchors; keeps lac's state in the directory it is given), `structural` (lattice edges), `graph` (snapshot construction from data, the read index `DualGraph`) | [2](02_discovery.md)–[6](06_graph_and_storage.md) |
| retrieval | `ltir/retrieval/` | `question` (parse, literal grounding), `seeds` (seed scoring), `traversal` (the transversal walk), `evidence` (the evidence object and its prompt text), `search` (one question end to end over a `CommittedState`, with baselines and highlight groups) | [7](07_question_answering.md) |
| storage | `ltir/storage/` | `workspace` (the layout below the workspace root and every file in it: journals, state, registry, artefacts, uploads, caches; transactions, recovery, the writer lock), `neo4j_mirror` + `cypher/` | [6](06_graph_and_storage.md) |
| application | `ltir/engine.py`, `ltir/migrate.py` | `Engine`: the batch lifecycle, deletion, the committed read state, `search` and `ask`; workspace migration | [9](09_operations.md), [6.5](06_graph_and_storage.md#65-versions-and-migration) |
| chat | `ltir/answering.py`, `ltir/llm_client.py` | the system prompt, the LLM-or-summary answer, the citation check; the OpenAI-compatible client and its `ChatModel` contract | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) |
| presentation | `ltir/web/` (`app.py`, `web/sphere.py`, `static/`), `ltir/cli.py` | the HTTP API with the chat endpoint, the single-page UI, the 3D sphere; the command line | [8](08_interface.md), [9.2](09_operations.md#92-command-line) |
| evaluation | `ltir/evaluation/` | `synthetic` (the demo dataset with planted phenomena), `experiment` (the hypothesis experiment) | [10](10_verification.md) |
| LLM service | `llm_gateway/` | one OpenAI-compatible endpoint in front of the upstream model; owns the provider, the model, its key and the provider routing | [12.4](#124-the-llm-service) |

## 12.2 Dependency direction

```text
cli -> web, migrate, evaluation, engine, storage, llm_client        the entry point: one command per layer
web -> engine, evaluation (the demo data), storage (its errors), analysis (labels)
migrate -> engine, storage            evaluation -> engine, retrieval
engine -> answering, llm_client, retrieval, analysis, storage
answering -> retrieval, llm_client    retrieval -> analysis
analysis -> engines                   storage -> engines, fileio            engines -> fileio
every layer -> config, models;        llm_gateway -> nothing from ltir
```

This is the whole set of cross-layer imports, and each points down the stack; `tests/test_architecture.py` holds the same table, resolves relative imports and `from ltir import x`, and fails on any import outside it. The core (`analysis`, `retrieval`, `storage`, plus `config`, `models`, `fileio`, `engines`) imports none of `fastapi`, `uvicorn`, `starlette`, `plotly` or `httpx`, and only `storage/neo4j_mirror.py` imports the Neo4j driver. Graph construction reads no files: `Engine` gathers the journal records, activations, vectors and batch records (each READY one carries its dataset's profile) and passes them to `graph.build_snapshot`. Retrieval reads no files either: `search.search` works on a `CommittedState` — the `DualGraph`, the `LatentFrame` and the literal catalog of one commit.

## 12.3 Reusing the analytical core

Everything a project needs to turn its tables into a queryable insight graph is `ltir/` without `web/`, `cli.py` and `evaluation/`: the core, `engine.py`, `answering.py` and `llm_client.py` (the engine composes the answer; the client makes a request only when an answer asks the LLM). The runtime dependencies are `requirements.txt` minus FastAPI, uvicorn, python-multipart and plotly (httpx stays, for the client).

The package is not installed with pip: another project puts the repository root on its `PYTHONPATH` (or vendors `ltir/`), keeps the embedding model where `MODEL_DIR` points (default `<repo>/models`, filled by `scripts/download_model.py`), and gives absolute paths in `load_config(...)` overrides — overrides are taken as given, while paths from the environment resolve against the repository root ([9.3](09_operations.md#93-configuration)).

```python
from ltir.config import load_config
from ltir.engine import Engine

engine = Engine(load_config(workspace_dir="/data/kb", embedding_backend="sentence-transformers"))
engine.ingest_file("sales.csv", bins="price:4")               # READY / FAILED / SKIPPED record
found = engine.search("Why is margin lower for phones in the US?")
found.evidence.items          # typed evidence: scope, shifts, statistics, the path that reached each item
found.evidence.to_prompt()    # the same evidence as LLM-ready text (docs/07_question_answering.md §7.4)
found.highlight()             # the node and edge ids it touched
engine.ask("...", use_llm=True)                                # + the LLM answer and the citation check
```

The pieces work without the engine as well: `ingestion.load_dataset` → `discovery.run_discovery` → `discovery.build_insights` → `quality.select_insights` → `canonical.canonicalize` → `InsightEncoder.encode` give scored, encoded insights for one table; `graph.build_snapshot` builds the dual graph from data; `search.search` answers a question over any `CommittedState`.

## 12.4 The LLM service

`llm_gateway/` is a separate deployable with its own image and requirements (FastAPI, uvicorn, httpx). It serves:

* `GET /health` — liveness, always 200, with `status` `ok` or `unconfigured`, the model and the upstream host;
* `GET /v1/models` — the one configured model (none while unconfigured);
* `POST /v1/chat/completions` — OpenAI-compatible: the request's `model` is replaced by the configured model, every other field (`temperature`, `max_tokens`, `top_p`, `stop`, `response_format`, …) is forwarded as sent, `stream: true` and a malformed body are refused (400, not FastAPI's 422). It adds the bearer key and — for OpenRouter — the pinned provider order without fallbacks, and returns the upstream response unchanged. An upstream client error keeps its status and `Retry-After` (a 429 stays a 429), an upstream server error becomes 502, a timeout 504, missing settings 503; error bodies have the OpenAI shape `{"error": {"message", "type"}}`.

| Setting | Default | Meaning |
|---|---|---|
| `GEMMA_BASE_URL` | — (required) | upstream base URL, e.g. `https://openrouter.ai/api/v1`, or a local Ollama / LM Studio |
| `GEMMA_MODEL_NAME` | — (required) | upstream model id |
| `GEMMA_CHAT_ENDPOINT` | `/chat/completions` | path appended to the base URL |
| `GEMMA_API_KEY` | "" | bearer token for the upstream |
| `GEMMA_PROVIDER` | "" | OpenRouter provider slugs, tried in order, no fallbacks |
| `GEMMA_CONNECT_TIMEOUT_SEC`, `GEMMA_READ_TIMEOUT_SEC` | 5, 120 | upstream timeouts |
| `GATEWAY_HOST`, `GATEWAY_PORT` | `127.0.0.1`, `8080` | where `python -m llm_gateway` listens (`0.0.0.0` in the image) |

The settings live in `.env.gemma` in the repository root (template `.env.gemma.sample`): Compose passes the file to the `llm` service, and `python -m llm_gateway` reads it wherever it is started. The backend knows only `LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`, the gateway); with `LLM_MODEL` empty it uses the one model the endpoint lists. Replacing the model, the provider or the serving software therefore changes only the gateway's settings; pointing `LLM_BASE_URL` at another OpenAI-compatible endpoint replaces the gateway itself — any gateway that serves `POST /chat/completions`, with `LLM_MODEL` set to the route it expects when it lists no models. The backend never holds the provider key.

## 12.5 Containers

| Service | Image | Port (host) | State | Health check | Starts after |
|---|---|---|---|---|---|
| `sig` — the backend: API, chat endpoint, UI, analytical core | `Dockerfile` (target `runtime`): `python:3.12.15-slim`, CPU torch, `requirements.txt` pinned by `constraints.txt`, `ltir/` | `127.0.0.1:8765` (`SIG_WEB_PORT`) | volume `sig-workspace` at `/data` (`WORKSPACE_DIR=/data/workspace`); `./models` mounted read-only at `/app/models` | `GET /api/config` (the server listens only after the embedding model is warm) | `neo4j` and `llm` started |
| `llm` — the LLM gateway | `llm_gateway/Dockerfile`: the same base and pins | none: the backend reaches `http://llm:8080` inside the stack | none; settings from `.env.gemma` | `GET /health` | — |
| `neo4j` — the graph mirror | `neo4j:5.26.31-community` | browser `127.0.0.1:17474`, bolt `127.0.0.1:17687` (`SIG_NEO4J_HTTP_PORT`, `SIG_NEO4J_BOLT_PORT`) | volume `neo4j-data` | `cypher-shell RETURN 1` | — |

```text
docker compose up -d --build                      # then http://127.0.0.1:8765
docker compose stop sig && docker compose run --rm sig python -m ltir migrate --yes     # a command against the stack's workspace
python scripts/compose_check.py                   # verification: its own project, ports and volumes, beside your stack
python scripts/compose_check.py --fresh --tests   # clean room: no build cache, pulled base images, the test suite inside the image
```

Before the first start: the embedding model in `./models` (`python scripts/download_model.py`) and `.env.gemma`. `SIG_NEO4J_PASSWORD` (default `sig-local-password`, at least 8 characters) sets the mirror's password when its volume is created; the volume keeps it, so a later change needs `ALTER USER` in Neo4j or a new volume. The backend uses database `neo4j`, the one Community Edition serves. Neither Neo4j nor the gateway gates the backend's start: a batch that commits while Neo4j is still starting records the mirror as failed (a warning; the next commit or `neo4j-sync` catches up), and answers fall back to the evidence-only summary while the LLM is unavailable. The backend runs as an unprivileged user and holds the workspace's writer lock inside the volume (`/data/workspace.writer.lock`), so `migrate` can rename the workspace there; it runs under `init` and gets 60 s to stop, so a running batch can finish — one cut short is rolled back at the next start ([6.4](06_graph_and_storage.md#64-commit-rollback-and-recovery)). The image computes embeddings on the CPU in the checkpoint's dtype; the representation fingerprint does not depend on the device ([4.6](04_representation.md#46-representation-identity-and-versions)). `constraints.txt` holds the exact package versions both images were verified with; the image's `test` target adds the test tools and runs the quality gate (`scripts/check.py`).

**Why three services and not more.** Each container has its own runtime and lifecycle: the backend needs the embedding model, the workspace and the Python data stack; the gateway needs only an HTTP stack and the provider's key, and is replaced when the model or the provider changes; Neo4j is a database with its own storage. The analytical core, retrieval and storage stay modules of the backend: they share the in-memory graph, the vectors and the embedding model, and a process boundary between them would copy that state for no independent scaling or deployment need.

**Chat stays in the backend.** The chat endpoint (`POST /api/query`) is a thin call to `Engine.ask`: retrieval needs the committed graph, the vectors and the embedding model in memory, and the answer needs the evidence it produced. A separate chat service would either load a second copy of the model (≈ 1.2 GB) or forward every question to the backend, and the UI's chat panel is static JavaScript served with the other views. The LLM call — the part with its own runtime, cost and provider — is already behind the gateway.

**The UI stays in the backend.** It is three static files (HTML, CSS, JavaScript) with no build step, served by the same FastAPI app as the API it calls; the 3D sphere page is rendered on request by `web/sphere.py`.

## 12.6 Observability

Logging is the standard `logging` module: the modules that log use `logging.getLogger(__name__)` (`engine`, `storage.workspace`, `web.app`, `config`; the gateway's logger is `llm_gateway`), and the entry points configure it (`python -m ltir`, `python -m ltir.web`, `python -m llm_gateway`). The domain keeps its own records: the batch record (stage times, metrics, warnings, errors; [9.1](09_operations.md#91-the-batch-lifecycle)), the ontology's metrics CSV ([5](05_latent_anchors.md)), the query log `logs/queries.jsonl` ([6.3](06_graph_and_storage.md#63-the-workspace-on-disk)), and the per-answer `metrics` of `QAResult` ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)). The gateway logs one line per completion (model, messages, prompt and completion tokens, latency), never the prompt itself. There is no tracing or metrics backend; the Docker health checks cover liveness.
