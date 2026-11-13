"""``python -m ads_bridge_service``: serve the bridge on ADS_BRIDGE_HOST:ADS_BRIDGE_PORT (127.0.0.1:8767)."""

from __future__ import annotations

import logging
import os

import uvicorn

from ads_bridge_service.app import create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("ADS_BRIDGE_HOST", "127.0.0.1"), port=int(os.environ.get("ADS_BRIDGE_PORT", "8767")))
