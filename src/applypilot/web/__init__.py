"""ApplyPilot web UI package.

Provides a FastAPI-based web interface for browsing jobs, re-tailoring
resumes, and watching live AI streams during apply runs. Imported lazily
from `applypilot serve` so the optional FastAPI dependency is only
required when the user opts in via `pip install -e ".[web]"`.
"""

from __future__ import annotations

__all__ = ["create_app", "run_server"]


def create_app(*args, **kwargs):
    from applypilot.web.server import create_app as _factory

    return _factory(*args, **kwargs)


def run_server(*args, **kwargs):
    from applypilot.web.server import run_server as _run

    return _run(*args, **kwargs)
