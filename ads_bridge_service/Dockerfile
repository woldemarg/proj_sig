# syntax=docker/dockerfile:1
# The ADS bridge (docs/12_architecture.md): a drawn zone's question handed to ADS with the zone's insights.
# Build context: the repository root. It carries only itself.

FROM ghcr.io/astral-sh/uv:0.11.17 AS uv

FROM python:3.12.15-slim AS deps
COPY --from=uv /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group ads

FROM python:3.12.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=deps /venv /venv
COPY ads_bridge_service ./ads_bridge_service
USER 65532:65532
ENV PATH=/venv/bin:$PATH PYTHONPATH=/app ADS_BRIDGE_HOST=0.0.0.0 ADS_BRIDGE_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --start-interval=1s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ads/health', timeout=4)"]
CMD ["python", "-m", "ads_bridge_service"]
