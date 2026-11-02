# syntax=docker/dockerfile:1
# The evidence narrator (docs/12_architecture.md): the graph service's evidence, verbalised by the LLM and citation-checked.
# Build context: the repository root. It carries only the shared kernel and itself.

FROM ghcr.io/astral-sh/uv:0.11.17 AS uv

FROM python:3.12.15-slim AS deps
COPY --from=uv /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group narrator

FROM python:3.12.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=deps /venv /venv
COPY insight_contracts ./insight_contracts
COPY evidence_narrator_service ./evidence_narrator_service
USER 65532:65532
ENV PATH=/venv/bin:$PATH PYTHONPATH=/app NARRATOR_HOST=0.0.0.0 NARRATOR_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --start-interval=1s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/chat/health', timeout=4)"]
CMD ["python", "-m", "evidence_narrator_service"]
