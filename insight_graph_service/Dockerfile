# syntax=docker/dockerfile:1
# The graph service (docs/12_architecture.md): the analytical core over one workspace, its HTTP API and the Neo4j
# mirror. Build arg TORCH picks the torch build: cpu (the default, image sig/insight-graph:cpu) or cu130 (compose.gpu.yaml,
# image sig/insight-graph:gpu); Qwen3 embeds on the GPU when the container has one, else on the CPU. Build context: the
# repository root. Target `test` adds the whole test suite (scripts/compose_check.py).

FROM python:3.12.15-slim AS runtime
ARG TORCH=cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
WORKDIR /app
# /data holds the workspace and the upload spool (a volume); /app/models the checkpoint (a volume filled by model-init)
RUN useradd --uid 10001 --create-home sig && mkdir -p /data/tmp /app/models && chown -R sig /data /app/models
COPY constraints.txt ./
COPY subgroup_miner/requirements.txt subgroup_miner/
COPY attractor_topology/requirements.txt attractor_topology/
COPY graph_query_engine/requirements.txt graph_query_engine/
COPY insight_graph_service/requirements.txt insight_graph_service/
# Pinned to the verified image (constraints.txt), torch at its +TORCH build. The wheels are fetched into a build-cache
# mount and installed from there: pypi.nvidia.com serves the CUDA wheels as no-store, so pip's own cache never keeps
# them, and any changed pin would otherwise download ≈ 2.4 GB again. Both mounts stay outside the image.
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/wheels,sharing=locked \
    echo "torch==$(tr -d '\r' < constraints.txt | sed -n 's/^torch==\([^+]*\).*/\1/p')+${TORCH}" > /tmp/torch.txt \
 && pip download --timeout 600 -d /wheels -r insight_graph_service/requirements.txt -c constraints.txt -c /tmp/torch.txt \
        --extra-index-url https://download.pytorch.org/whl/${TORCH} \
 && pip install --no-index --find-links /wheels -r insight_graph_service/requirements.txt -c constraints.txt -c /tmp/torch.txt
COPY insight_contracts ./insight_contracts
COPY subgroup_miner ./subgroup_miner
COPY attractor_topology ./attractor_topology
COPY graph_query_engine ./graph_query_engine
COPY insight_graph_service ./insight_graph_service
COPY data/demo ./data/demo
USER sig
# auto: the GPU when the container has one (compose.gpu.yaml), else the CPU (the `test` image below runs without one)
ENV PYTHONPATH=/app WORKSPACE_DIR=/data/workspace MODEL_DIR=/app/models EMBEDDING_DEVICE=auto WEB_HOST=0.0.0.0 WEB_PORT=8000 HOME=/tmp TMPDIR=/data/tmp
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
# the runtime packages (torch included) are installed already; this adds the test tools from PyPI
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt -c constraints.txt
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
