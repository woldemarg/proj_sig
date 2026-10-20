"""The whole stack on one local port for host development: the console's static files, the graph service and the
narrator, routed by path as the web console's nginx routes them (``/api/chat/`` to the narrator, every other
``/api/`` path to the graph service, anything else to the static files). Mirror any routing change in
``sig_web_console/nginx.conf``.

    python scripts/dev.py            # http://127.0.0.1:8765 (WEB_HOST / WEB_PORT); the LLM via python -m llm_model_broker

Reads the repository's .env like the services; the GPU is used when EMBEDDING_DEVICE allows it.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_narrator_service.app import create_app as create_narrator  # noqa: E402
from evidence_narrator_service.settings import NarratorSettings  # noqa: E402
from insight_graph_service.server.app import ready_app  # noqa: E402

CONSOLE = Path(__file__).resolve().parents[1] / "sig_web_console" / "static"


def compose(graph_service: Any, narrator: Any) -> Any:
    """One ASGI app: ``/api/chat/...`` to the narrator, other ``/api/...`` (and the lifespan) to the graph service,
    the rest to the console's static files; responses over 1 KB gzipped for a client that accepts it, as nginx does."""
    from starlette.middleware.gzip import GZipMiddleware
    from starlette.staticfiles import StaticFiles

    console = StaticFiles(directory=CONSOLE, html=True)

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        target = narrator if path.startswith("/api/chat/") else graph_service if path.startswith("/api/") or scope["type"] != "http" else console
        await target(scope, receive, send)

    return GZipMiddleware(app, minimum_size=1024, compresslevel=5)  # as nginx: level 9 takes 7x longer on a 94 MB cell file for 11 % less


def chromium() -> str | None:
    """The newest Chromium in Playwright's browser cache, whichever revision this Playwright pins (the browser test and
    the performance benchmark drive it); None without one."""
    cache = {"win32": Path(os.environ.get("LOCALAPPDATA", "")), "darwin": Path.home() / "Library" / "Caches"}.get(
        sys.platform, Path.home() / ".cache"
    )
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or cache / "ms-playwright")
    patterns = ("chromium-*/chrome-win64/chrome.exe", "chromium-*/chrome-linux*/chrome", "chromium-*/chrome-mac*/*.app/Contents/MacOS/*")
    found = [exe for pattern in patterns for exe in root.glob(pattern)]
    return str(max(found, key=lambda exe: int(exe.relative_to(root).parts[0].removeprefix("chromium-")))) if found else None


def main() -> None:
    import uvicorn

    from insight_graph_service.core.settings import load_env_file, load_settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env_file()
    settings = load_settings()
    narrator = create_narrator(NarratorSettings.from_env(insight_graph_url=f"http://127.0.0.1:{settings.web_port}"))
    uvicorn.run(compose(ready_app(settings), narrator), host=settings.web_host, port=settings.web_port, log_level="info")


if __name__ == "__main__":
    main()
