# syntax=docker/dockerfile:1
# The graph service (docs/12_architecture.md): the analytical core over one workspace, its HTTP API and the Neo4j
# mirror. Build arg TORCH picks the torch build: cpu (the default, image sig/insight-graph:cpu) or cu130 (compose.gpu.yaml,
# image sig/insight-graph:gpu); Qwen3 embeds on the GPU when the container has one, else on the CPU. Build context: the
# repository root. Target `test` adds the whole test suite (scripts/compose_check.py).

FROM ghcr.io/astral-sh/uv:0.11.17 AS uv

# The exact versions of uv.lock into /venv; uv stays in this stage. The cache mount keeps the downloaded wheels
# (≈ 2.4 GB for cu130) between builds, outside the image.
FROM python:3.12.15-slim AS deps
ARG TORCH=cpu
COPY --from=uv /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=600
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group graph --group ${TORCH}

FROM python:3.12.15-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
# /data holds the workspace and the upload spool (a volume); /app/models the checkpoint (a volume filled by model-init)
RUN useradd --uid 10001 --create-home sig && mkdir -p /data/tmp /app/models && chown -R sig /data /app/models
COPY --from=deps /venv /venv
COPY data/demo ./data/demo
COPY insight_contracts ./insight_contracts
COPY subgroup_miner ./subgroup_miner
COPY attractor_topology ./attractor_topology
COPY graph_query_engine ./graph_query_engine
COPY insight_graph_service ./insight_graph_service
USER sig
# auto: the GPU when the container has one (compose.gpu.yaml), else the CPU (the `test` image below runs without one)
ENV PATH=/venv/bin:$PATH PYTHONPATH=/app WORKSPACE_DIR=/data/workspace MODEL_DIR=/app/models EMBEDDING_DEVICE=auto WEB_HOST=0.0.0.0 WEB_PORT=8000 HOME=/tmp TMPDIR=/data/tmp
EXPOSE 8000
# the server listens only after the model is checked and warm, so any answer means ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --start-interval=1s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
CMD ["python", "-m", "insight_graph_service.server"]

FROM runtime AS test
ARG TORCH=cpu
USER root
COPY --from=uv /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock pytest.ini ./
# adds the test tools to the runtime venv (torch included already); HOME=/tmp here, so the cache dir is named
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group dev --group ${TORCH}
COPY evidence_narrator_service ./evidence_narrator_service
COPY llm_model_broker ./llm_model_broker
COPY sig_web_console ./sig_web_console
COPY tests ./tests
COPY scripts ./scripts
RUN chown -R sig /app
USER sig
ENV WORKSPACE_DIR=/app/workspace
HEALTHCHECK NONE
CMD ["python", "scripts/check.py"]
