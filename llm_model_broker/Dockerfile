# syntax=docker/dockerfile:1
# SIG LLM model broker: one OpenAI-compatible endpoint in front of the configured upstream model (docs/12_architecture.md §12.4).
# Build context: the repository root (compose.yaml). Settings: GEMMA_* (compose reads them from .env).

FROM ghcr.io/astral-sh/uv:0.11.17 AS uv

FROM python:3.12.15-slim AS deps
COPY --from=uv /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group broker

FROM python:3.12.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=deps /venv /venv
COPY llm_model_broker ./llm_model_broker
USER 65532:65532
ENV PATH=/venv/bin:$PATH BROKER_HOST=0.0.0.0 BROKER_PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --start-interval=1s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]
CMD ["python", "-m", "llm_model_broker"]
