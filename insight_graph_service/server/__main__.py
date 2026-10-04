"""``python -m insight_graph_service.server``: serve the graph service (host runs read the repository's .env)."""

import logging

import uvicorn

from insight_graph_service.core.settings import load_env_file, load_settings
from insight_graph_service.server.app import ready_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
load_env_file()  # a host run reads the repository's .env; a container gets its environment from compose
settings = load_settings()
uvicorn.run(ready_app(settings), host=settings.web_host, port=settings.web_port, log_level="info")
