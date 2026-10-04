# syntax=docker/dockerfile:1
# The graph service (docs/12_architecture.md): the analytical core over one workspace, its HTTP API and the Neo4j
# mirror; CPU torch. Build context: the repository root. Target `test` adds the whole test suite (scripts/compose_check.py).

FROM python:3.12.15-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
WORKDIR /app
# /data holds the workspace and the upload spool (a volume); /app/models the checkpoint (a volume filled by model-init)
RUN useradd --uid 10001 --create-home sig && mkdir -p /data/tmp /app/models && chown -R sig /data /app/models
COPY constraints.txt ./
COPY subgroup_miner/requirements.txt subgroup_miner/
COPY attractor_topology/requirements.txt attractor_topology/
COPY graph_query_engine/requirements.txt graph_query_engine/
COPY insight_graph_service/requirements.txt insight_graph_service/
# one resolver pass, pinned to the verified image (constraints.txt); torch is the CPU build
RUN pip install -r insight_graph_service/requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY insight_contracts ./insight_contracts
COPY subgroup_miner ./subgroup_miner
COPY attractor_topology ./attractor_topology
COPY graph_query_engine ./graph_query_engine
COPY insight_graph_service ./insight_graph_service
USER sig
ENV PYTHONPATH=/app WORKSPACE_DIR=/data/workspace MODEL_DIR=/app/models EMBEDDING_DEVICE=cpu WEB_HOST=0.0.0.0 WEB_PORT=8000 HOME=/tmp TMPDIR=/data/tmp
EXPOSE 8000
# the server listens only after the model is checked and warm, so any answer means ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
CMD ["python", "-m", "insight_graph_service.server"]

FROM runtime AS test
USER root
COPY requirements-dev.txt ./
COPY evidence_narrator_service/requirements.txt evidence_narrator_service/
COPY llm_model_broker/requirements.txt llm_model_broker/
RUN pip install -r requirements-dev.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY evidence_narrator_service ./evidence_narrator_service
COPY llm_model_broker ./llm_model_broker
COPY sig_web_console ./sig_web_console
COPY tests ./tests
COPY scripts ./scripts
COPY pyproject.toml pytest.ini ./
RUN chown -R sig /app
USER sig
ENV WORKSPACE_DIR=/app/workspace
HEALTHCHECK NONE
CMD ["python", "scripts/check.py"]
