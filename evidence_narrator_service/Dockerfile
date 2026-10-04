# syntax=docker/dockerfile:1
# The evidence narrator (docs/12_architecture.md): the graph service's evidence, verbalised by the LLM and citation-checked.
# Build context: the repository root. It carries only the shared kernel and itself.

FROM python:3.12.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
WORKDIR /app
COPY evidence_narrator_service/requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt
COPY insight_contracts ./insight_contracts
COPY evidence_narrator_service ./evidence_narrator_service
USER 65532:65532
ENV PYTHONPATH=/app NARRATOR_HOST=0.0.0.0 NARRATOR_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/chat/health', timeout=4)"]
CMD ["python", "-m", "evidence_narrator_service"]
