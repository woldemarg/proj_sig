"""``python -m evidence_narrator_service``: serve the narrator on NARRATOR_HOST:NARRATOR_PORT (127.0.0.1:8766)."""

from __future__ import annotations

import logging
import os

import uvicorn

from evidence_narrator_service.app import create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("NARRATOR_HOST", "127.0.0.1"), port=int(os.environ.get("NARRATOR_PORT", "8766")))
