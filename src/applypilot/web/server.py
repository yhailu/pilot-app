"""FastAPI application factory and uvicorn launcher for the ApplyPilot web UI.

Kept deliberately small — all endpoints live in `routes.py`. This module
exists so `applypilot serve` can wire up logging, lifespan hooks, and
static/template directories in one place.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from applypilot import __version__
from applypilot.config import ensure_dirs, load_env
from applypilot.database import init_db

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Bootstrap once on startup; clean up streams on shutdown."""
    load_env()
    ensure_dirs()
    init_db()
    log.info("ApplyPilot web UI %s ready", __version__)
    try:
        yield
    finally:
        # Best-effort cleanup of any in-memory subprocesses started via /retailor
        from applypilot.web.streams import shutdown_all_streams

        shutdown_all_streams()


def create_app() -> FastAPI:
    """Build a fresh FastAPI app instance.

    Used as a factory by `uvicorn.run(..., factory=True)` so reload mode
    can re-import the module without re-using a stale state graph.
    """
    app = FastAPI(
        title="ApplyPilot",
        version=__version__,
        description="Local job-application pipeline UI.",
        lifespan=_lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Routes are wired here so they import cleanly under `factory=True`.
    from applypilot.web.routes import register_routes

    register_routes(app, templates_dir=TEMPLATES_DIR)
    return app


def run_server(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Boot uvicorn against the create_app factory."""
    import uvicorn

    log.info("Starting ApplyPilot web UI on http://%s:%d", host, port)
    uvicorn.run(
        "applypilot.web.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
        access_log=False,
    )
