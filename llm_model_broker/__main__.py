"""``python -m llm_model_broker``: serve the broker on BROKER_HOST:BROKER_PORT (127.0.0.1:8080), reading the repository's ``.env``."""

from __future__ import annotations

import logging
import os

import uvicorn

from llm_model_broker.server import Settings, create_app, load_env_file

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env_file()
    app = create_app(Settings.from_env(os.environ))
    uvicorn.run(app, host=os.environ.get("BROKER_HOST", "127.0.0.1"), port=int(os.environ.get("BROKER_PORT", "8080")))
