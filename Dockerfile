# syntax=docker/dockerfile:1
# SIG backend: the analytical core, the HTTP API with the chat endpoint, and the web UI (docs/12_architecture.md §12.5).
# The embedding model is mounted read-only at /app/models (scripts/download_model.py fetches it);
# the workspace lives on a volume at /data. Target `test` adds the test suite: see scripts/compose_check.py.

FROM python:3.12.15-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
WORKDIR /app
RUN useradd --uid 10001 --create-home sig && mkdir -p /data /app/data /app/models && chown sig /data /app/data
# one resolver pass, pinned to the verified image (constraints.txt); torch is the CPU build
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY ltir ./ltir
USER sig
ENV WORKSPACE_DIR=/data/workspace WEB_HOST=0.0.0.0 WEB_PORT=8765 EMBEDDING_DEVICE=cpu LTIR_NO_DOTENV=1
EXPOSE 8765
# the server listens only after the embedding model is warm, so any answer means ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/config', timeout=4)"]
CMD ["python", "-m", "ltir.web"]

FROM runtime AS test
USER root
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY llm_gateway ./llm_gateway
COPY tests ./tests
COPY scripts ./scripts
COPY pyproject.toml pytest.ini ./
RUN chown -R sig /app
USER sig
ENV WORKSPACE_DIR=/app/workspace
HEALTHCHECK NONE
CMD ["python", "scripts/check.py"]
