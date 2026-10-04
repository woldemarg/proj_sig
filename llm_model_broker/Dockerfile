# syntax=docker/dockerfile:1
# SIG LLM model broker: one OpenAI-compatible endpoint in front of the configured upstream model (docs/12_architecture.md §12.4).
# Build context: the repository root (compose.yaml). Settings: GEMMA_* (compose reads them from .env).

FROM python:3.12.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
WORKDIR /app
COPY llm_model_broker/requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt
COPY llm_model_broker ./llm_model_broker
USER 65532:65532
ENV BROKER_HOST=0.0.0.0 BROKER_PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]
CMD ["python", "-m", "llm_model_broker"]
